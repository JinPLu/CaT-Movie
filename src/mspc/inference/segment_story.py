"""消费分镜适配器产出的分段和目标文字，逐段渲染一条故事。

逐镜与联合生成共用此链；分段只由 ``comparison_methods.prompts.adapt_storyboard`` 决定，不在这里重新打包。

参考只来自真实已生成的镜（窗前全部镜），每镜至多一张帧，最多 ``frames_max`` 张。推理有五种历史（:data:`HISTORY_POLICIES`）：

- ``none``：不给参考；
- ``plan_selection``（按分镜选帧，锚视角）：:func:`mspc.data.reference_selection.select_references`——每个在场实体在
  外观不变期间取同框人少、景别清楚、最早的一镜，地点取同地点人少、景别远、最早的一镜；每张取
  :func:`mspc.data.reference_selection.reference_position`（镜内有衣着或状态变化取末帧，否则中帧）；
- ``plan_selection_distinct_camera``：同上，锚视角排序在外观不过期之后先避开与目标镜同机位的镜；
- ``recent_shots``（取最近几镜，论文消融「不看分镜的选帧」）：张数与同一窗按分镜选帧相同，只隔离「选哪几镜」，
  取窗前最近的那几镜，每镜中帧，不看分镜内容；
- ``clip_retrieval``（按画面相似度选帧，论文消融「CLIP 检索」，规则见 :mod:`mspc.inference.clip_frame_selection`）：张数同上，
  每个已生成镜存至多 6 张 CLIP 去重后的帧，按它们与本窗各镜文字的 CLIP 文图相似度取前几镜，每镜取最像的一帧。

几种给参考的模式之后完全相同：参考按镜／帧升序进入文字、图像与 latent 消费者；读者分派按渲染器的
``model.memory_prefix.attention_policy``，就是训练同一个 :func:`mspc.data.reference_routing.reference_readers`，选中的参考
全部交给渲染器（11-14 删掉外人路由，依据见 :mod:`mspc.data.reference_selection`）。

参考行按 ``row_policy`` 写（:func:`reference_row`）：``fixed_role_v1`` 只写固定角色句；写可见身份的两种
（``FRAME_ENTITY_ROW_POLICIES``）与训练同一个编译器（:func:`mspc.data.shot_text.reference_text_with_entity_spans`），身份
实体取参考源镜分镜在场的登记人物（:func:`mspc.data.shot_text.shot_characters`），地点取源镜 ``reference_setting``，物体
取源镜在场物体（只 ``frame_entities_with_shot_objects_v1``）。训练的实体是离线逐帧标签「这一帧里看得到的登记人物」；推理
不读图。出片链里没有读图服务：
源身份路由（``model.source_identity``）要逐帧源点、点控制加特征注入（``planner_features``）要规划器读图特征，这两种
渲染器不能出片（:func:`require_story_renderer`）。

参考只按本段自己选：候选只取本段起点之前已生成的镜，所需实体按本段各镜算。
目标镜的布局句只由计划编译（计划没有布局就没有）。
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from mspc.data.reference_frame_latents import frame_index
from mspc.data.reference_routing import reference_readers, reference_slot_metadata
from mspc.data.reference_selection import reference_position, select_references
from mspc.core.wan22_contract import (FRAME_ENTITY_ROW_POLICIES, ROUTED_MEMORY_ATTENTION_POLICIES,
                                      MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1)
from mspc.data.shot_text import (UMT5_PROMPT_TOKEN_BUDGET, reference_setting, reference_text,
                                reference_text_with_entity_spans, shot_characters, shot_object_phrases,
                                shot_text_with_entity_spans)
from mspc.data.storyboard import validate_storyboard
from mspc.inference.continuity_dev_32 import wan22_shot_plan_from_frame_counts
from mspc.inference import clip_frame_selection
from mspc.inference.wan22 import MemoryPrefixInput, Wan22InferenceRuntime

NO_REFERENCES = "none"
PLAN_SELECTION = "plan_selection"
#: 按分镜选帧＋避开同机位（配训练参考 ``distinct_camera_first``）：锚视角排序在外观不过期之后先比是否与目标镜同机位。
PLAN_SELECTION_DISTINCT_CAMERA = "plan_selection_distinct_camera"
RECENT_SHOTS = "recent_shots"
CLIP_RETRIEVAL = "clip_retrieval"
HISTORY_POLICIES = (NO_REFERENCES, PLAN_SELECTION, PLAN_SELECTION_DISTINCT_CAMERA, RECENT_SHOTS, CLIP_RETRIEVAL)


@dataclass(frozen=True)
class WholeReferenceFrame:
    """一张已生成的参考帧：源镜、镜内帧号、记录里怎么写（按分镜选帧与取最近几镜写帧位置 first/middle/last，CLIP 检索写
    帧号）、单帧潜变量。"""
    shot_index: int
    frame: int
    label: str | int
    latent: Any


def history_position(policy: str, shot: Mapping[str, Any]) -> str:
    """每个已生成镜存哪一帧作候选参考：按分镜选帧存 :func:`reference_position`，取最近几镜存中帧。"""
    if policy in (PLAN_SELECTION, PLAN_SELECTION_DISTINCT_CAMERA):
        return reference_position(shot)
    if policy == RECENT_SHOTS:
        return "middle"
    raise ValueError(f"{policy!r} keeps no reference history")


def select_reference_shots(policy: str, annotation: Mapping[str, Any], *, span: Sequence[int],
                           frames_max: int) -> list[tuple[int, str]]:
    """只看分镜的两种：一窗的参考 (源镜下标, 取哪一帧)，按镜序。候选是窗前全部已生成的镜；张数由按分镜选帧定，取最近几镜
    照这个张数取。CLIP 检索要看已生成的画面，在 :func:`generate_story` 里按同一张数检索。"""
    start = span[0]
    planned = select_references(annotation, span=span, candidates=range(start), frames_max=frames_max,
                                distinct_camera=policy == PLAN_SELECTION_DISTINCT_CAMERA)
    if policy in (PLAN_SELECTION, PLAN_SELECTION_DISTINCT_CAMERA):
        return [(index, history_position(policy, annotation["shots"][index])) for index in planned]
    if policy == RECENT_SHOTS:
        return [(index, history_position(policy, annotation["shots"][index]))
                for index in range(start - len(planned), start)]
    raise ValueError(f"{policy!r} selects no references")


def require_story_renderer(model_cfg: Mapping[str, Any]) -> None:
    """出片链只从分镜写参考身份行、不调读图服务：源身份路由（要每张参考里逐人的源点）与点控制加特征注入（要规划器
    读参考帧出的特征）的渲染器在这里出不了片，开跑前就报错。"""
    if model_cfg.get("source_identity") is not None:
        raise ValueError("source identity routing needs per-person source points read from each reference frame; "
                         "story rendering writes reference identities from the storyboard and reads no frames")
    if (model_cfg.get("planner_tokens") or {}).get("content") == "planner_features":
        raise ValueError("planner-feature tokens are read from the reference frames by the planner reading service, "
                         "which story rendering no longer calls")


def reference_row(annotation: Mapping[str, Any], shot_index: int, row_policy: str | None) -> str:
    """源镜 ``shot_index`` 那张参考的参考行，只由分镜定、不看画面。写可见身份的两种与训练同一个编译器：身份实体是源镜
    在场的登记人物（:func:`mspc.data.shot_text.shot_characters`），地点取源镜地点，源镜在场物体只
    ``frame_entities_with_shot_objects_v1`` 写；其余（``fixed_role_v1``）只写固定角色句。"""
    if row_policy not in FRAME_ENTITY_ROW_POLICIES:
        return reference_text()
    return reference_text_with_entity_spans(
        [{"name": name} for name in shot_characters(annotation, shot_index)], characters=annotation["characters"],
        setting=reference_setting(annotation, shot_index),
        shot_objects=(shot_object_phrases(annotation, shot_index)
                      if row_policy == MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1 else ()))["text"]


def _ordered_references(chosen: Sequence[WholeReferenceFrame], *, start_shot: int,
                        generated_frame_counts: Mapping[int, int], frames_max: int) -> list[WholeReferenceFrame]:
    """在唯一消费边界核对真实已生成来源，按实际镜／帧排序，整条帧记录一起移动。"""
    if len(chosen) > frames_max:
        raise ValueError(f"{len(chosen)} references exceed frames_max {frames_max}")
    moments = []
    for frame in chosen:
        index = frame.shot_index
        if type(index) is not int or not 0 <= index < start_shot or index not in generated_frame_counts:
            raise ValueError(f"reference shot {index} is not generated history before shot {start_shot}")
        number = frame.frame
        if type(number) is not int or not 0 <= number < generated_frame_counts[index]:
            raise ValueError(f"reference frame {index}/{number} is outside generated history")
        moments.append((index, number))
    if len({index for index, _ in moments}) != len(moments):
        raise ValueError("references require at most one reference frame per shot")
    return [frame for _, frame in sorted(zip(moments, chosen), key=lambda entry: entry[0])]


def generate_story(runtime: Wan22InferenceRuntime, *, annotation: Mapping[str, Any], shot_ids: Sequence[str],
                   segments: Sequence[Mapping[str, Any]], output: Path, token_counter: Callable[[str], int],
                   frames_max: int | None, seed: int, segment_seed_stride: int, on_segment: Callable[[dict], None],
                   history_policy: str, token_edits: Mapping[int, Any] | None = None) -> list[dict]:
    """逐段渲染。只看分镜的两种：每个已生成镜存一张帧（:func:`history_position`），各窗按 :func:`select_reference_shots`
    从中取；``history_policy`` 为 ``none`` 时不存不取，``frames_max`` 也为 None。

    参考行由 :func:`reference_row` 按源镜分镜写；参考行写可见身份时，每段记录带 ``reference_entities``（每张参考一项：源镜
    号、所取帧号、写进身份行的登记人物），与 ``references`` 同序。

    渲染器带规划器 token（``model.planner_tokens``，人物编号或人物几何）时每段另给 token：镜首点取计划镜首框中心
    （``reference_pointing.plan_starts``）。``token_edits``（段序号 → :class:`~mspc.inference.planner_token_reading.TokenEdit`
    的元组）给移点、去 token、换内容检验，整片重出。

    CLIP 检索（``clip_retrieval``）：每个已生成镜按 :func:`mspc.inference.clip_frame_selection.shot_keyframes` 存至多 6 张帧
    （图像嵌入与单帧潜变量），各窗按 :func:`~mspc.inference.clip_frame_selection.retrieve` 取与按分镜选帧同样张数；记录的
    ``references`` 写 [源镜号, 帧号]，``library_added`` 写每镜入库的帧号，另带 ``retrieval_scores``（窗前每镜：最像的帧号与
    分）。CLIP 按卡加载一次（:func:`~mspc.inference.clip_frame_selection.embedder_for`）。

    渲染器的 ``model.memory_prefix.attention_policy`` 按镜路由时，最终参考按训练同一个
    :func:`mspc.data.reference_routing.reference_readers` 路由给本段各目标镜，路由随参考交给渲染器并记进
    ``reference_routing``；没有参考的段不带路由。"""
    from mspc.inference.planner_token_reading import apply_token_edits, person_numbers, swap_rows, token_condition
    from mspc.inference.reference_pointing import plan_geometry, plan_starts

    if history_policy not in HISTORY_POLICIES:
        raise ValueError(f"unknown generation history policy: {history_policy!r}")
    require_story_renderer(runtime.model_cfg)
    token_content = (runtime.model_cfg.get("planner_tokens") or {}).get("content")
    spatial_enabled = runtime.model_cfg.get("spatial_layout") is not None and token_content is None
    memory_cfg = runtime.model_cfg.get("memory_prefix")
    row_policy = (memory_cfg or {}).get("row_policy")
    reference_policy = (memory_cfg or {}).get("attention_policy")
    routed_references = reference_policy in ROUTED_MEMORY_ATTENTION_POLICIES
    frame_entities = history_policy != NO_REFERENCES and row_policy in FRAME_ENTITY_ROW_POLICIES
    if (history_policy == NO_REFERENCES) != (frames_max is None):
        raise ValueError("frames_max is given exactly when references are kept (history_policy other than none)")
    if frames_max is not None and (type(frames_max) is not int or not 1 <= frames_max <= 10):
        raise ValueError("frames_max must be an int in 1..10 when history is kept")
    if token_content is None and any((token_edits or {}).values()):
        raise ValueError("target token edits belong to a renderer with planner tokens")
    annotation = validate_storyboard(annotation)
    if len(annotation["shots"]) != len(shot_ids):
        raise ValueError("annotation shots and shot_ids must have the same length")
    retrieval = history_policy == CLIP_RETRIEVAL
    embedder = clip_frame_selection.embedder_for(runtime.device) if retrieval else None
    # CLIP 检索的库：源镜 → {帧号: 参考帧}，与 {帧号: 图像嵌入}。
    library: dict[int, dict[int, WholeReferenceFrame]] = {}
    library_embeddings: dict[int, dict[int, torch.Tensor]] = {}
    history_frames: dict[int, WholeReferenceFrame] = {}
    generated_frame_counts: dict[int, int] = {}
    capture_history = frames_max is not None
    records = []
    for ordinal, segment in enumerate(segments):
        start, end = segment["start_shot"], segment["end_shot"]
        label = f"g{ordinal + 1}"
        ids, counts = list(shot_ids[start:end]), list(segment["frame_counts"])
        chosen: list[WholeReferenceFrame] = []
        retrieval_scores = None
        if retrieval:
            # 张数与同一窗按分镜选帧相同；窗前每镜取它最像本窗文字的一帧，再按分取前 count 个镜。
            count = len(select_references(annotation, span=(start, end), candidates=range(start), frames_max=frames_max,
                                          distinct_camera=False))
            best = (clip_frame_selection.best_frames({index: library_embeddings[index] for index in range(start)},
                                                     embedder.texts(list(segment["prompts"]))) if start else {})
            chosen = [library[index][frame] for index, frame in clip_frame_selection.retrieve(best, count)]
            retrieval_scores = {shot_ids[index]: [frame, round(score, 4)] for index, (frame, score) in sorted(best.items())}
        elif capture_history:
            for index, position in select_reference_shots(history_policy, annotation, span=(start, end),
                                                          frames_max=frames_max):
                if index not in history_frames or history_frames[index].label != position:
                    raise ValueError(f"history shot {index + 1} has no generated {position} frame")
                chosen.append(history_frames[index])
        if capture_history:
            chosen = _ordered_references(chosen, start_shot=start,
                generated_frame_counts=generated_frame_counts, frames_max=frames_max)
        available_count = sum(index < start for index in generated_frame_counts)
        if frames_max is None:
            unfilled_reason = "history_disabled"
        elif len(chosen) == frames_max:
            unfilled_reason = None
        elif not available_count:
            unfilled_reason = "no_generated_history"
        else:
            unfilled_reason = "plan_names_fewer_references"
        reference_supply = {"available_count": available_count, "selected_count": len(chosen),
                            "capacity": frames_max, "unfilled_reason": unfilled_reason}
        identity_rows = None
        if frame_entities:
            # 排序、每镜择一完成后才编身份行；实体只由分镜定（源镜在场的登记人物），不看画面。
            identity_rows = tuple(
                {"shot": frame.shot_index + 1,
                 "frame": frame.frame,
                 "entities": [{"name": name, "identity_status": "registered"}
                              for name in shot_characters(annotation, frame.shot_index)]}
                for frame in chosen)
        routing, routing_record = None, None
        if chosen and routed_references:
            routing, reasons = reference_readers(reference_policy, annotation["shots"],
                                                 reference_shots=[frame.shot_index for frame in chosen],
                                                 target_shots=range(start, end))
            routing_record = {"routing": [list(entry) for entry in routing], "reasons": list(reasons)}
        token_inputs, token_record = {}, None
        if token_content is not None:
            from mspc.data.shot_targets import latent_frames

            edits = tuple((token_edits or {}).get(ordinal, ()))
            starts, applied = apply_token_edits(plan_starts(annotation, (start, end)), edits)
            content = swap_rows(person_numbers(starts), starts, edits)
            geometry_rows = (plan_geometry(annotation, (start, end))
                             if token_content == "person_geometry" else None)
            if geometry_rows is not None and edits:
                # 复用同一个点编辑规则；只改位置字段，保留宽高、关系及全部人物行。
                geometry_starts, _ = apply_token_edits([
                    (row["shot_index"], entity["name"], entity["point_2d"])
                    for row in geometry_rows for entity in row["characters"]], edits)
                edited_points = {(shot, name): point for shot, name, point in geometry_starts}
                for row in geometry_rows:
                    for entity in row["characters"]:
                        entity["point_2d"] = edited_points[(row["shot_index"], entity["name"])]
            geometry_inputs = ({} if geometry_rows is None else
                               {"geometry_rows": geometry_rows,
                                "max_people": runtime.model_cfg["planner_tokens"]["max_people"]})
            condition = token_condition(starts, content, span=(start, end),
                latent_lengths=[latent_frames(n) for n in counts], **geometry_inputs)
            if geometry_rows is not None and condition is not None and any(edit.kind == "swap" for edit in edits):
                # 换的是完整内容（ID和几何行），不是位置；关系列仍索引原Character ID，不换列。
                condition = replace(condition, geometry=swap_rows(condition.geometry, starts, edits))
            token_inputs = {"planner_tokens": condition}
            token_record = {"starts": [list(entry) for entry in starts], "edits": applied}
            if geometry_rows is not None:
                token_record["geometry"] = geometry_rows
        reference_rows = [reference_row(annotation, frame.shot_index, row_policy) for frame in chosen]
        target_rows = list(segment["prompts"])
        memory = None
        slots = reference_slot_metadata(annotation["shots"], target_shots=range(start, end),
            reference_count=len(chosen), reference_entities=identity_rows, routing=routing)
        compiled_targets, final_rows = [], []
        for index, text, references in zip(range(start, end), target_rows, slots, strict=True):
            prepared = shot_text_with_entity_spans(annotation, index)
            if prepared["text"] == text:
                # 实际slot只作校验并进本镜登记，参考身份及外貌写在各自的参考行。
                compiled = shot_text_with_entity_spans(annotation, index, references=references)
                compiled_targets.append({"shot_index": index, **compiled})
                final_rows.append(compiled["text"])
            else:
                if "layouts" in annotation and spatial_enabled:
                    raise ValueError("spatial layout spans cannot be inferred for a prepared prompt differing "
                                     "from its annotation compiler output; preserve its actual text and spans")
                # 既有外部扩写仍保原文；它的任何字符都不由annotation猜测重造。
                final_rows.append(text)
        target_rows = final_rows
        token_counts = [token_counter(row) for row in reference_rows + target_rows]
        if max(token_counts) > UMT5_PROMPT_TOKEN_BUDGET:
            raise ValueError(f"{label}: a text row exceeds {UMT5_PROMPT_TOKEN_BUDGET} umT5 tokens: {token_counts}")
        if chosen:
            memory = MemoryPrefixInput(latents=torch.cat([frame.latent for frame in chosen], dim=2),
                                       shot_texts=tuple(reference_rows), reference_routing=routing)
        # 此段已按共同source窗口和实际媒体派生；原480缓存的308帧上限不是704执行容量。
        plan = wan22_shot_plan_from_frame_counts(counts, max_rgb_frames=segment["num_frames"], label=label)
        path = output / f"{label}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        spatial_inputs = {}
        if "layouts" in annotation and spatial_enabled:
            from mspc.models.spatial_layout_inputs import compile_spatial_layout_inputs
            layouts = compile_spatial_layout_inputs(annotation, tuple(range(start, end)), compiled_rows=compiled_targets)
            if layouts:
                spatial_inputs["target_layouts"] = layouts
        added = []
        # 一个卡进程里几部故事共用一份运行时：出片和读回本段历史帧这一段独占运行时（它把本段的帧存在自身上），
        # 一部在锁外编文字、存段时另一部可以用卡。单部故事时锁不争用，计算与原来相同。
        with runtime.lock:
            started = time.perf_counter()
            readout = runtime.generate_case(prompts=target_rows,
                shot_groups=plan["shot_groups"], latent_shot_groups=plan["latent_shot_groups"],
                shot_frame_counts=counts, output_path=path, seed=seed + ordinal * segment_seed_stride,
                memory=memory, capture_history=capture_history, **spatial_inputs,
                **token_inputs)
            seconds = time.perf_counter() - started
            if capture_history:
                if len(runtime.last_generated_shot_frames) != len(ids):
                    raise ValueError(f"{label}: generated {len(runtime.last_generated_shot_frames)} shots, planned {len(ids)}")
                for offset, candidates in enumerate(runtime.last_generated_shot_frames):
                    shot_index = start + offset
                    generated_frame_counts[shot_index] = counts[offset]
                    if retrieval:
                        video = runtime.last_generated_shot_videos[offset]
                        if len(video) != counts[offset]:
                            raise ValueError(f"{label}: shot {shot_index} has {len(video)} frames, planned {counts[offset]}")
                        # 帧 1…N−1（不取第 0 帧，同 frame_index）：CLIP 同镜去重入库，每张另存单帧潜变量。
                        embeddings = embedder.images(video[1:])
                        kept = clip_frame_selection.shot_keyframes(embeddings)
                        library_embeddings[shot_index] = {frame: embeddings[frame - 1] for frame in kept}
                        library[shot_index] = {frame: WholeReferenceFrame(
                            shot_index, frame, frame, runtime.encode_single_frame(video[frame]).detach().cpu())
                            for frame in kept}
                        added.append([ids[offset], kept])
                        continue
                    position = history_position(history_policy, annotation["shots"][shot_index])
                    if position not in candidates:
                        raise ValueError(f"{label}: shot {shot_index} has no {position} frame")
                    history_frames[shot_index] = WholeReferenceFrame(
                        shot_index, frame_index(position, counts[offset]), position,
                        runtime.encode_single_frame(candidates[position]).detach().cpu())
                    added.append([ids[offset], position])
        references = [[shot_ids[frame.shot_index], frame.label] for frame in chosen]
        record = dict(segment, segment_id=label, shot_ids=ids, frame_counts=counts,
            references=references, history_policy=history_policy,
            reference_supply=reference_supply,
            library_added=added, prompts=reference_rows + target_rows, token_counts=token_counts,
            seconds=seconds, seed=seed + ordinal * segment_seed_stride, output=str(path), readout=readout)
        if token_record is not None:
            record["planner_tokens"] = token_record
        if retrieval_scores is not None:
            record["retrieval_scores"] = retrieval_scores
        if routing_record is not None:
            # 每个目标镜一项，是 ``references`` 里的参考下标；``reasons`` 每张参考一项。
            record["reference_routing"] = routing_record
        if identity_rows is not None:
            # 每张参考一项，与 ``references`` 同序：源镜号（1 起）、所取帧号（镜内）、写进身份行的登记人物。
            record["reference_entities"] = [
                {"shot": row["shot"], "frame": row["frame"], "entities": [{"name": entity["name"]}
                                                                          for entity in row["entities"]]}
                for row in identity_rows]
        records.append(record)
        on_segment(record)
    return records
