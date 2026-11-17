"""规划器（Qwen3.5-9B 底座，transformers）读参考画面的两种用法；故事出片不再调它们（11-14 起参考身份行由分镜写，见
:mod:`mspc.inference.segment_story`），留给分析工具与训练预计算：

- 认参考里的人（``ground_references``）：看每张参考 RGB，答画面里确认可见的登记人物（分析工具量读图与标签、文字冲突）。
- 点控制加特征注入的特征（``plan_features``）：把参考帧、段前最近一镜末帧、不带方位句的逐镜文字与「人名＋镜首点」拼成的
  作答对规划器做一次教师强制前向，取每人每镜的特征。与训练预计算（``tools/run/planner_token_precompute.py``）同一份提示
  与取特征代码（``reference_pointing.plan_token_content``、``planner_token_features.token_features``）。

规划器在自己的环境里起一个本机服务（``tools/run/planner_reading_service.py``，与预计算同一环境），调用方经
:class:`PlannerReadingClient` 调它：图片按段存成未压缩的 ``.npy``，特征经 npz 回传。服务一个地址，后面可有几个
副本进程（``--devices``），同时到的请求由空闲副本各自算。

检验改动（:class:`TokenEdit`）：``move`` 改某人这一镜的镜首点；``swap`` 对调两人这一镜的 token 内容（人物编号或几何行）；
``remove`` 去掉某人这一镜的 token。整片重出，种子与噪声与原片相同。

``read`` 是零样本两步（``select_moments`` 看历史选时刻，再看图指人给镜首点），只给分析工具量规划器布局准度用，不进出片。
``think`` 是思考模式的
作答（官方思考档采样），同样只给分析工具用；其余各处都走非思考、贪心的 ``_generate``。
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

# Qwen 官方「思考预算」的提前收尾句（Qwen 文档 Quickstart「Thinking Budget」）：思考写满预算时接在后面，再让模型作答。
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
              "directly now.\n</think>\n\n")


@dataclass(frozen=True)
class TokenEdit:
    """检验用的改动，落在目标镜 ``shot``（块内下标）的 token 上，参考帧与文字都不动：

    - ``move``：把 ``names[0]`` 在这一镜的镜首点改成 ``point``（移点）；
    - ``swap``：把两人在这一镜的 token 内容对调，位置不变（换 token）；
    - ``remove``：去掉 ``names[0]`` 在这一镜的 token（点记为 null，与训练时按人丢点同一效果）；全片每个条目各一条即
      「token 全去、参考照给」。
    """
    kind: str
    names: tuple[str, ...]
    shot: int
    point: tuple[int, int] | None = None

    def __post_init__(self):
        if type(self.shot) is not int or self.shot < 0:
            raise ValueError("a token edit needs a target shot index")
        if self.kind == "move":
            if len(self.names) != 1 or self.point is None:
                raise ValueError("a move edit needs one name and a point")
        elif self.kind == "swap":
            if len(self.names) != 2 or self.names[0] == self.names[1] or self.point is not None:
                raise ValueError("a swap edit needs exactly two different names and no point")
        elif self.kind == "remove":
            if len(self.names) != 1 or self.point is not None:
                raise ValueError("a remove edit needs one name and no point")
        else:
            raise ValueError(f"unknown token edit {self.kind!r}")

    def to_mapping(self) -> dict:
        return {"kind": self.kind, "names": list(self.names), "shot": self.shot,
                "point": None if self.point is None else list(self.point)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TokenEdit":
        if set(value) != {"kind", "names", "shot", "point"}:
            raise ValueError("a token edit needs exactly kind, names, shot and point")
        return cls(kind=value["kind"], names=tuple(value["names"]), shot=value["shot"],
                   point=None if value["point"] is None else tuple(value["point"]))


class PlannerTokenReader:
    """``feature_pooling``、``feature_layer`` 与训练配置的 ``dataset.planner_tokens`` 相同。"""

    def __init__(self, *, model_path, device, feature_pooling: str, feature_layer: int, max_new_tokens: int,
                 cut_margin: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        # 默认在CPU构造BF16权重，再搬到指定设备；device_map或device上下文均会额外要求Accelerate。
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.feature_key = (feature_pooling, feature_layer)
        self.max_new_tokens = max_new_tokens
        self.cut_margin = cut_margin
        self.generation_record: dict[str, Any] = {}
        self.runtime_info = {
            "backend": "transformers", "model": str(model_path),
            "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
            "model_eos_token_id": self.model.generation_config.eos_token_id,
            "tokenizer_eos_token_id": self.processor.tokenizer.eos_token_id,
            "processor": type(self.processor).__name__,
            "source_grounding_decoder": "json_schema_token_enforcer",
        }

    def _generate(self, content, *, answer_schema: Mapping[str, Any] | None, **media) -> str:
        """非思考、贪心，最多 ``max_new_tokens``。"""
        self.generation_record = {}
        import torch
        from mspc.inference.planner_decode import finish_reason, stop_token_ids
        from mspc.inference.planner_token_features import prompt_text

        prompt = prompt_text(self.processor, content)
        stops = stop_token_ids(self.processor.tokenizer, self.model.generation_config)
        self.generation_record = {"prompt_text": prompt, "enable_thinking": False,
            "generation_kwargs": {"max_new_tokens": self.max_new_tokens, "do_sample": False,
                                  "eos_token_id": stops},
            "answer_schema": answer_schema, "stage": "preprocessing"}
        started = time.perf_counter()
        inputs = self._inputs(prompt, media)
        constrained = {}
        if answer_schema is not None:
            from mspc.inference.planner_decode import schema_prefix_allowed_tokens_fn
            constrained["prefix_allowed_tokens_fn"] = schema_prefix_allowed_tokens_fn(
                self.processor.tokenizer, dict(answer_schema))
            self.generation_record["format_enforcer_version"] = version("lm-format-enforcer")
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.generation_record["generation_kwargs"], **constrained)
        new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
        raw = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        self.generation_record.update(stage="finished", new_tokens=len(new_ids),
            finish_reason=finish_reason(new_ids, stops, self.max_new_tokens),
            last_token_id=new_ids[-1] if new_ids else None,
            elapsed_seconds=time.perf_counter() - started, raw_content=raw)
        return raw

    def _inputs(self, prompt: str, media: Mapping[str, Any]):
        """处理器输入（搬到模型设备），并把媒体的形状记进 ``generation_record``。"""
        self.generation_record.update(
            media_type="video" if "videos" in media else "image" if "images" in media else "text",
            media_count=len(media.get("videos", media.get("images", []))))
        inputs = self.processor(text=[prompt], return_tensors="pt", **media).to(self.model.device)
        self.generation_record.update({key: inputs[key].tolist() for key in ("image_grid_thw", "video_grid_thw")
                                       if key in inputs})
        if "video_metadata" in media:
            self.generation_record["video_metadata"] = [
                {key: getattr(row, key) for key in ("total_num_frames", "fps", "duration", "frames_indices")}
                for row in media["video_metadata"]]
            self.generation_record["do_sample_frames"] = media["do_sample_frames"]
        self.generation_record.update(prompt_tokens=inputs["input_ids"].shape[1], stage="generation")
        return inputs

    def think(self, content, *, sampling: Mapping[str, float], seed: int, thinking_budget: int, answer_tokens: int,
              **media) -> str:
        """思考模式（聊天模板 ``enable_thinking=True``），按 ``sampling``（temperature、top_p、top_k）采样，种子 ``seed``。
        Qwen 官方「思考预算」的两次生成：先生成至多 ``thinking_budget`` 个 token；没以停止符收尾时，还没写出 ``</think>``
        就接上官方的提前收尾句 :data:`EARLY_STOP`，已写出就原样接着，再生成至多 ``answer_tokens`` 个。返回两次合起来的
        新文字（思考在 ``</think>`` 之前，作答在其后）。"""
        self.generation_record = {}
        import torch
        from mspc.inference.planner_decode import finish_reason, stop_token_ids

        if set(sampling) != {"temperature", "top_p", "top_k"}:
            raise ValueError("thinking sampling needs exactly temperature, top_p and top_k")
        tokenizer = self.processor.tokenizer
        prompt = self.processor.apply_chat_template([{"role": "user", "content": list(content)}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=True)
        stops = stop_token_ids(tokenizer, self.model.generation_config)
        kwargs = {"do_sample": True, **sampling, "eos_token_id": stops}
        self.generation_record = {"prompt_text": prompt, "enable_thinking": True, "seed": seed,
            "generation_kwargs": {**kwargs, "thinking_budget": thinking_budget, "answer_tokens": answer_tokens},
            "answer_schema": None, "stage": "preprocessing"}
        started = time.perf_counter()
        inputs = self._inputs(prompt, media)
        prompt_length = inputs["input_ids"].shape[1]
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=thinking_budget, **kwargs)
        first = out[0, prompt_length:].tolist()
        spliced = False
        if first and first[-1] in stops:
            new_ids, reason = first, finish_reason(first, stops, thinking_budget)
        else:
            spliced = tokenizer.convert_tokens_to_ids("</think>") not in first
            if spliced:
                extra = torch.tensor([tokenizer(EARLY_STOP, add_special_tokens=False)["input_ids"]],
                                     dtype=out.dtype, device=out.device)
                out = torch.cat([out, extra], dim=1)
            # 与提示等长的逐 token 字段（如图文 token 类型）按文字 token 补 0；图像特征不动。
            again_inputs = {key: torch.cat([value, value.new_zeros((1, out.shape[1] - prompt_length))], dim=1)
                            if torch.is_tensor(value) and tuple(value.shape) == (1, prompt_length) else value
                            for key, value in inputs.items()}
            again_inputs.update(input_ids=out, attention_mask=torch.ones_like(out))
            with torch.no_grad():
                again = self.model.generate(**again_inputs, max_new_tokens=answer_tokens, **kwargs)
            new_ids = again[0, prompt_length:].tolist()
            reason = finish_reason(again[0, out.shape[1]:].tolist(), stops, answer_tokens)
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        self.generation_record.update(stage="finished", thinking_tokens=len(first), early_stop_spliced=spliced,
            new_tokens=len(new_ids), finish_reason=reason, last_token_id=new_ids[-1] if new_ids else None,
            elapsed_seconds=time.perf_counter() - started, raw_content=raw)
        return raw

    def select_moments(self, annotation: Mapping[str, Any], *, span: Sequence[int], history: Mapping[int, np.ndarray],
                       target_prompts: Sequence[str], needed: Sequence[str], max_moments: int) -> dict:
        """第一步：每个所需角色选一个时刻。返回记录（原答、参考时刻、逐人 picks、丢弃记录）；没有所需角色或没有历史时
        时刻为空，不调模型。作答读不成 JSON 时所有人都没有参考。"""
        from mspc.inference.reference_pointing import (parse_json_answer, step_one_content, step_one_moments,
                                                       video_inputs)

        record: dict[str, Any] = {"needed": list(needed), "moments": [], "step_one_picks": []}
        if not needed or not history:
            return record
        lengths = {index: len(frames) for index, frames in history.items()}
        videos, metadata = video_inputs(self.processor.video_processor, history)
        try:
            record["step_one_raw"] = self._generate(
                step_one_content(annotation, span=span, history_frames=lengths, target_prompts=target_prompts,
                                 needed=needed),
                answer_schema=None, videos=videos, video_metadata=metadata, do_sample_frames=False)
            record["step_one_generation"] = self.generation_record
        except Exception as error:
            record["step_one_generation"] = self.generation_record
            error.record = record
            raise
        try:
            answer = parse_json_answer(record["step_one_raw"])
        except json.JSONDecodeError:
            answer = record["step_one_raw"]
        record["moments"], record["step_one_picks"], record["step_one_dropped"] = step_one_moments(
            answer, needed=needed, history_frames=lengths, cut_margin=self.cut_margin, max_moments=max_moments)
        return record

    def ground_references(self, annotation: Mapping[str, Any], *, references: Sequence[Sequence[int]],
                          images: Sequence[np.ndarray]) -> dict:
        """第二步对最终实际RGB重指：所有四组共用可见IDs，只有源绑定组消费点。"""
        from PIL import Image
        from mspc.inference.reference_pointing import (grounded_reference_entities, parse_json_answer,
                                                       source_grounding_content, source_grounding_schema)
        if len(images) != len(references):
            raise ValueError("one actual RGB is required per final reference moment")
        record = {"references": [list(moment) for moment in references], "grounding_raw": None,
                  "grounding_generation": None}
        self.generation_record = {}
        try:
            empty_answer = {"references": [{"reference": i + 1, "characters": [], "unknown_characters": []}
                                           for i in range(len(references))]}
            grounded_reference_entities(empty_answer, annotation=annotation, references=references)
            if not references or not annotation["characters"]:
                answer = empty_answer
            else:
                pictures = []
                for image in images:
                    pixels = np.asarray(image)
                    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                        raise ValueError("source grounding requires the actual uint8 reference RGB")
                    pictures.append(Image.fromarray(pixels))
                record["grounding_raw"] = self._generate(source_grounding_content(annotation, references),
                    answer_schema=source_grounding_schema(annotation, references), images=pictures)
                record["grounding_generation"] = self.generation_record
                answer = parse_json_answer(record["grounding_raw"])
            record["reference_entities"] = grounded_reference_entities(answer, annotation=annotation,
                                                                        references=references)
        except Exception as error:
            record["grounding_generation"] = self.generation_record
            error.record = record
            raise
        return record

    def plan_features(self, annotation: Mapping[str, Any], *, span: Sequence[int], images: Sequence[np.ndarray],
                      references: Sequence[Sequence[int]], last_shot: int | None, target_prompts: Sequence[str],
                      starts: Sequence[tuple[int, str, Sequence[int] | None]]) -> np.ndarray:
        """点控制加特征注入：教师强制预填，返回与 ``starts`` 同序的特征 [人数, 隐维]（按训练的口径与层）。
        ``images`` 依次是各参考帧与段前最近一镜末帧（[H, W, 3] uint8）。"""
        from PIL import Image
        from mspc.inference.planner_token_features import prompt_text, token_features
        from mspc.inference.reference_pointing import answer_from_starts, plan_token_content

        content = plan_token_content(annotation, span=span, references=references, last_shot=last_shot,
                                     target_prompts=target_prompts)
        pictures = [Image.fromarray(np.asarray(image, dtype=np.uint8)) for image in images]
        entries, features = token_features(self.model, self.processor, prompt=prompt_text(self.processor, content),
                                           images=pictures or None, answer=answer_from_starts(starts),
                                           layers=[self.feature_key[1]])
        where = {(key - 1, name): i for i, (_, key, name) in enumerate(entries)}
        return features[self.feature_key][[where[(index, name)] for index, name, _ in starts]]

    def read(self, annotation: Mapping[str, Any], *, span: Sequence[int], history: Mapping[int, np.ndarray],
             target_prompts: Sequence[str], needed: Sequence[str]) -> dict:
        """零样本第二步（只给布局准度分析用，不进出片）。``history``：段前各已生成镜（块内下标 → [帧数, H, W, 3] uint8）。
        返回记录（两步原答、参考时刻、第二步作答与丢弃记录）；作答读不成时抛错，异常带 ``record``。"""
        from PIL import Image
        from mspc.inference.reference_pointing import (MAX_REFERENCES, parse_json_answer, step_two_content,
                                                       target_people, validate_answer)

        record: dict[str, Any] = {"needed": list(needed)}
        try:
            record.update(self.select_moments(annotation, span=span, history=history, target_prompts=target_prompts,
                                              needed=needed, max_moments=MAX_REFERENCES))
            moments = record["moments"]
            last = max(history) if history else None
            images = [Image.fromarray(np.asarray(history[shot][frame])) for shot, frame in moments]
            if last is not None:
                images.append(Image.fromarray(np.asarray(history[last][-1])))
            content = step_two_content(annotation, span=span, references=moments, last_shot=last,
                                       target_prompts=target_prompts, needed=needed)
            record["step_two_raw"] = self._generate(content, answer_schema=None,
                                                      **({"images": images} if images else {}))
            record["step_two_generation"] = self.generation_record
            record["answer"], record["dropped"] = validate_answer(parse_json_answer(record["step_two_raw"]),
                                                                  references=len(moments), needed=needed,
                                                                  targets=target_people(annotation, span))
        except Exception as error:
            record.update(getattr(error, "record", {}))
            record["generation_failure"] = self.generation_record
            error.record = record  # type: ignore[attr-defined]
            raise
        return record


class PlannerReadingClient:
    """调用方这一侧：与 :class:`PlannerTokenReader` 同样的 ``ground_references``、``plan_features``，经本机服务调规划器。

    ``workdir`` 下每段的图片与特征各存一份。开跑先核服务读的口径与层与调用方给的一致。"""

    def __init__(self, *, service_url: str, workdir, feature_pooling: str, feature_layer: int, cut_margin: int,
                 timeout_seconds: int):
        from pathlib import Path

        self.url, self.timeout = service_url.rstrip("/"), timeout_seconds
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        info = self._call("info", {})
        if [info["feature_pooling"], info["feature_layer"], info["cut_margin"]] != [feature_pooling, feature_layer,
                                                                                   cut_margin]:
            raise ValueError(f"the planner service reads {info}, the renderer expects "
                             f"{feature_pooling}/{feature_layer}/cut margin {cut_margin}")
        self.calls = 0

    def _call(self, route: str, payload: Mapping[str, Any]) -> dict:
        import urllib.request

        request = urllib.request.Request(f"{self.url}/{route}", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def _reading(self, route: str, payload: Mapping[str, Any]) -> dict:
        reply = self._call(route, payload)
        if reply["status"] != "ok":
            error = ValueError(reply["error"])
            error.record = reply["record"]  # type: ignore[attr-defined]
            raise error
        return reply["record"]

    def ground_references(self, annotation, *, references, images) -> dict:
        path = self.workdir / f"source-grounding-{self.calls:05d}.npy"
        self.calls += 1
        np.save(path, np.stack([np.asarray(image) for image in images]) if images
                else np.zeros((0, 480, 832, 3), dtype=np.uint8))
        return self._reading("ground_references", {"annotation": annotation,
            "references": [list(moment) for moment in references], "images_path": str(path)})

    def plan_features(self, annotation, *, span, images, references, last_shot, target_prompts, starts) -> np.ndarray:
        path = self.workdir / f"plan-images-shot{span[0] + 1:03d}.npy"
        np.save(path, np.stack([np.asarray(image, dtype=np.uint8) for image in images]) if images
                else np.zeros((0, 1, 1, 3), dtype=np.uint8))
        features_path = self.workdir / f"features-shot{span[0] + 1:03d}.npz"
        self._reading("plan_features", {"annotation": annotation, "span": list(span), "images_path": str(path),
                                        "references": [list(r) for r in references], "last_shot": last_shot,
                                        "target_prompts": list(target_prompts),
                                        "starts": [[i, n, None if p is None else list(p)] for i, n, p in starts],
                                        "features_path": str(features_path)})
        with np.load(features_path) as data:
            return data["features"]


def apply_token_edits(starts: Sequence[tuple[int, str, Sequence[int] | None]], edits: Sequence[TokenEdit]
                      ) -> tuple[list[tuple[int, str, list[int] | None]], list[dict]]:
    """移点改 ``starts`` 里那人那一镜的点，去 token 把它记为 null；换 token 留给 :func:`swap_rows`（内容出来之后再换）。
    返回新 starts 与记录。"""
    starts = [(i, n, None if p is None else list(p)) for i, n, p in starts]
    applied = []
    for edit in edits:
        record = edit.to_mapping()
        if edit.kind in ("move", "remove"):
            where = next((k for k, (i, n, _) in enumerate(starts) if i == edit.shot and n == edit.names[0]), None)
            if where is None:
                raise ValueError(f"{edit.kind} edit {record} names no start entry in the plan")
            record["moved_from"] = starts[where][2]
            starts[where] = (edit.shot, edit.names[0], None if edit.kind == "remove" else list(edit.point))
        applied.append(record)
    return starts, applied


def swap_rows(content, starts: Sequence[tuple[int, str, Any]], edits: Sequence[TokenEdit]):
    """换 token：对调两人那一镜的 token 内容（规划器特征的行，或人物编号）；两人里有谁在那一镜没有条目就报错。"""
    where = {(i, n): k for k, (i, n, _) in enumerate(starts)}
    out = content.copy() if hasattr(content, "copy") else content.clone()
    for edit in edits:
        if edit.kind != "swap":
            continue
        a, b = ((edit.shot, name) for name in edit.names)
        if a not in where or b not in where:
            raise ValueError(f"swap edit {edit.to_mapping()} names a start entry the plan does not have")
        out[where[a]], out[where[b]] = content[where[b]], content[where[a]]
    return out


def person_numbers(starts: Sequence[tuple[int, str, Any]]):
    """点控制：每个条目的人物编号（名字 Character k 的 k），与 ``starts`` 同序。"""
    import re

    import torch

    numbers = []
    for _, name, _ in starts:
        match = re.fullmatch(r"Character (\d+)", name)
        if match is None:
            raise ValueError(f"point control numbers registered characters named 'Character k', got {name!r}")
        numbers.append(int(match.group(1)))
    return torch.tensor(numbers, dtype=torch.long)


def token_condition(starts: Sequence[tuple[int, str, Sequence[int] | None]], content, *, span: Sequence[int],
                    latent_lengths: Sequence[int], geometry_rows: Sequence[Mapping[str, Any]] | None = None,
                    max_people: int | None = None):
    """镜首点与 token 内容（与 ``starts`` 同序：规划器特征 [人数, 隐维] 或人物编号 [人数]）→ PlannerTokenCondition；
    没有一个人有点时 None。完整空间接口显式给 geometry_rows 和 max_people，共用训练的几何打包器；放置不变。"""
    from itertools import accumulate

    import torch
    from mspc.models.planner_tokens import PlannerTokenCondition, geometry_token_features, token_placement

    if (geometry_rows is None) != (max_people is None):
        raise ValueError("geometry rows and their explicit max_people must be supplied together")
    geometry = None if geometry_rows is None else geometry_token_features(starts, geometry_rows, max_people)

    placed = token_placement(starts=list(starts), span_start=span[0], latent_starts=[0, *accumulate(latent_lengths)],
                             shot_lookup={(i, n): k for k, (i, n, _) in enumerate(starts)})
    if placed is None:
        return None
    features = content if isinstance(content, torch.Tensor) else torch.from_numpy(np.asarray(content, dtype=np.float32))
    return PlannerTokenCondition(features=features, shot_tokens=tuple(tuple(r) for r in placed["shot_tokens"]),
                                 geometry=geometry)
