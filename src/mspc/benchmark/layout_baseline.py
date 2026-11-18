"""布局基线五条件的推理编排。

出片只调用 ``mspc.inference.render_story``。评分清单交给
``mspc.evaluation.benchmark.load_manifest``。配对区间交给
``mspc.evaluation.paired_bootstrap.cluster_bootstrap``，按源片重采样，不另写一套。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mspc.config.shared_root import expand_shared_root_token
from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES
from mspc.data.shot_text import without_target_text
from mspc.data.storyboard import validate_storyboard
from mspc.evaluation.benchmark import load_manifest, write_json
from mspc.evaluation.paired_bootstrap import cluster_bootstrap, paired
from mspc.inference.render_story import RuntimeCache, render_story
from mspc.inference.segment_story import (CLIP_RETRIEVAL, HISTORY_POLICIES, NO_REFERENCES, PLAN_SELECTION,
                                          RECENT_SHOTS, reference_row, select_reference_shots)
from mspc.models.spatial_layout import (
    SPATIAL_LAYOUT_MODE,
    SpatialLayoutAnchor,
    SpatialLayoutCondition,
    SpatialLayoutEntity,
    SpatialLayoutShot,
    build_spatial_layout_biases,
)

# film-003759-b0001 第一镜 Character 1 的真实框：832×480 上 xywh [124, 0, 481, 480]。
# 只用来量「有框时 B 是否非零」，不是开发集计划里的框。
PROBE_BOX = (124 / 832, 0.0, 605 / 832, 1.0)
PROBE_SOURCE = "tests/fixtures/block_layouts/film-003759-b0001.json layouts are mask tracks; shot s01 Character 1 bbox_xywh [124, 0, 481, 480] on 832x480"

ROOT_FIELDS = ("run_dir", "split", "input_tier", "requests", "storyboards", "conditions")
CONDITION_FIELDS = (
    "name", "renderer_config", "checkpoint", "history_policy", "plan_dir",
    "seed", "segment_seed_stride", "segment_max_shots", "segment_max_frames",
    "output_dir",
)
EXPERT_LORA_CHECKPOINT_KEYS = ("high_noise_lora", "low_noise_lora")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_run(path: Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(ROOT_FIELDS):
        raise ValueError(f"run manifest must contain exactly {ROOT_FIELDS}")
    if raw["split"] not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    if raw["input_tier"] != "shots":
        raise ValueError("layout baseline inference reads the shot tier")
    conditions = raw["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty list")
    names = []
    for condition in conditions:
        # 带规划器 token 的条件另带 TOKEN_FIELDS；其余条件不带。
        tokens = isinstance(condition, dict) and "token_edits_dir" in condition
        expected = set(CONDITION_FIELDS) | (set(TOKEN_FIELDS) if tokens else set())
        if not isinstance(condition, dict) or set(condition) != expected:
            raise ValueError(f"each condition must contain exactly {sorted(expected)}")
        if condition["history_policy"] not in HISTORY_POLICIES:
            raise ValueError(f"unknown history policy: {condition['history_policy']!r}")
        if type(condition["seed"]) is not int or condition["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(condition["segment_seed_stride"]) is not int or condition["segment_seed_stride"] < 0:
            raise ValueError("segment_seed_stride must be a non-negative integer")
        # 渲染器训练时的窗口镜数：null 不限（窗口只限 228 帧，最终联合配方），1 为逐镜，旧的至多 4 镜渲染器给 4。
        shots = condition["segment_max_shots"]
        if shots is not None and (type(shots) is not int or shots < 1):
            raise ValueError("segment_max_shots must be null (frame-limited windows) or a positive integer")
        # 整窗帧数预算：最终联合配方训练窗口 228；121 是 Wan2.2-TI2V-5B 的预训练长度。单镜超过时独占一窗。
        frames = condition["segment_max_frames"]
        if type(frames) is not int or not 1 <= frames <= SEGMENT_MAX_FRAMES:
            raise ValueError(f"segment_max_frames must be an integer in 1..{SEGMENT_MAX_FRAMES}")
        condition["checkpoint"] = map_checkpoint_paths(condition["checkpoint"], lambda value:
            expand_shared_root_token(value, f"{path}:{condition['name']}.checkpoint"))
        names.append(condition["name"])
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    seeds = {condition["seed"] for condition in conditions}
    plans = {condition["plan_dir"] for condition in conditions}
    strides = {condition["segment_seed_stride"] for condition in conditions}
    if len(seeds) != 1 or len(plans) != 1 or len(strides) != 1:
        raise ValueError("the five conditions share one plan directory, one seed and one segment stride")
    return raw


def _stories(run: Mapping[str, Any], root: Path) -> list[dict]:
    catalog = json.loads(resolve_path(run["requests"], root).read_text(encoding="utf-8"))
    selected = [story for story in catalog["stories"] if story["split"] == run["split"]]
    if not selected:
        raise ValueError(f"no stories in split {run['split']}")
    return selected


def execution_storyboard(board: Mapping[str, Any]) -> dict:
    """给定分镜表或规划记录 → 执行计划。人物框（``layouts``）照留：编译器从框推出布局句，与规划器写的布局字段同一写法；
    旧合同记录里的参考镜号去掉——它总由在场名单决定（``shot_reference_labels``），选帧照旧。"""
    compiled = json.loads(json.dumps(board))
    compiled["shots"] = [{key: value for key, value in shot.items() if key != "references"} for shot in compiled["shots"]]
    return validate_storyboard(compiled)


def with_frame_counts(board: Mapping[str, Any], pacing: Mapping[str, Any]) -> dict:
    """给定分镜表只换逐镜 ``frame_count``：取另一份计划同一故事的逐镜帧数，镜数与总帧数都必须相同。"""
    given = [shot["frame_count"] for shot in board["shots"]]
    taken = [shot["frame_count"] for shot in pacing["shots"]]
    if len(given) != len(taken):
        raise ValueError(f"pacing plan has {len(taken)} shots, the storyboard has {len(given)}")
    if sum(given) != sum(taken):
        raise ValueError(f"pacing plan totals {sum(taken)} frames, the storyboard totals {sum(given)}")
    paced = json.loads(json.dumps(board))
    for shot, frames in zip(paced["shots"], taken):
        shot["frame_count"] = frames
    return paced


def load_picture_types(path: Path) -> dict[str, list[str]]:
    """``configs/benchmark/benchmark_picture_types_v4.json`` 形式的逐镜画面类型：故事 → 逐镜列表。"""
    stories = json.loads(Path(path).read_text(encoding="utf-8"))["stories"]
    return {story_id: list(entry["picture_types"]) for story_id, entry in stories.items()}


def with_picture_types(board: Mapping[str, Any], picture_types: Sequence[str]) -> dict:
    """请求写明画风时，计划逐镜写上画面类型（渲染器文字随之有画风句）；镜数必须与计划相同。"""
    if len(picture_types) != len(board["shots"]):
        raise ValueError(f"{len(picture_types)} picture types for a plan with {len(board['shots'])} shots")
    return {**board, "shots": [{**shot, "picture_type": kind} for shot, kind in zip(board["shots"], picture_types)]}


def write_given_plans(manifest: Path, *, story_ids: Sequence[str] | None = None,
                      frame_counts_from: Path | None = None, picture_types_from: Path | None = None) -> dict:
    """给定分镜表写成计划目录。``frame_counts_from`` 给时长分配对照：其余字段照给定分镜表，逐镜帧数取自
    该目录里同一故事的计划（见 ``with_frame_counts``）。``picture_types_from`` 给请求写明画风的输入：
    逐镜填上画面类型（见 ``with_picture_types``）。"""
    root = repo_root()
    run = load_run(manifest)
    plan_dir = resolve_path(run["conditions"][0]["plan_dir"], root)
    if plan_dir.exists():
        raise FileExistsError(f"plan directory already exists: {plan_dir}")
    boards = json.loads(resolve_path(run["storyboards"], root).read_text(encoding="utf-8"))
    by_id = {story["story_id"]: story for story in boards["stories"]}
    selected = _stories(run, root)
    if story_ids is not None:
        unknown = set(story_ids) - {story["story_id"] for story in selected}
        if unknown or len(story_ids) != len(set(story_ids)):
            raise ValueError(f"story ids are not distinct members of split {run['split']}: {sorted(unknown)}")
        wanted = set(story_ids)
        selected = [story for story in selected if story["story_id"] in wanted]
    picture_types = None if picture_types_from is None else load_picture_types(picture_types_from)
    plan_dir.mkdir(parents=True)
    layout_present = []
    for story in selected:
        source = by_id[story["story_id"]]["storyboard"]
        if frame_counts_from is not None:
            pacing = json.loads((frame_counts_from / f"{story['story_id']}.json").read_text(encoding="utf-8"))
            source = with_frame_counts(source, pacing)
        if picture_types is not None:
            source = with_picture_types(source, picture_types[story["story_id"]])
        layout_present.append("layouts" in source)
        write_json(plan_dir / f"{story['story_id']}.json", execution_storyboard(source))
    if any(layout_present) != all(layout_present):
        raise ValueError("layouts must be present on every source storyboard or on none")
    provenance = {
        "source_requests": run["requests"],
        "source_storyboards": run["storyboards"],
        "layouts": "present in the source storyboards" if all(layout_present) else "omitted; the source storyboards have no layouts field",
        "frame_counts": ("source storyboards" if frame_counts_from is None
                         else f"taken shot by shot from {frame_counts_from}; shot count and total frames unchanged"),
        "picture_types": ("none; the requests state no visual style" if picture_types_from is None
                          else f"stated in the request, written on every shot from {picture_types_from}"),
        "story_ids": [story["story_id"] for story in selected],
    }
    write_json(plan_dir / "provenance.json", provenance)
    return provenance


def write_planner_plans(manifest: Path, records: Path, *, without_layouts: bool = False,
                        without_changes: bool = False, picture_types_from: Path | None = None) -> dict:
    """把 generation.plan 写下的计划记录转成五条件共用的计划目录。

    只有 status 为 completed 的记录写成分镜；其余故事记进 provenance 的 planning_failed，
    出片时在分母里落 planning_failed，不补给定计划。

    ``without_layouts`` 给"给与不给规划器布局"一对：同一批记录，每镜的构图词、人物关系与镜末侧去掉（旧记录的人物框
    一并去掉），渲染器文字里没有这组布局句；语义分镜不变。

    ``without_changes`` 保留旧的changes删除输入：每镜 ``changes`` 置空，渲染器文字没有
    "By the end of the shot" 一句。新编译器还用changes.before恢复开态、判断衣着是否稳定，
    因此这个历史输入操作不再等同于只删末态句；本轮原生比较不使用它。

    ``picture_types_from`` 给请求写明画风的输入：规划器不写画风，请求已给的画风按"给定不改"原样逐镜写进计划
    （见 ``with_picture_types``）。
    """
    root = repo_root()
    run = load_run(manifest)
    plan_dir = resolve_path(run["conditions"][0]["plan_dir"], root)
    if plan_dir.exists():
        raise FileExistsError(f"plan directory already exists: {plan_dir}")
    records = Path(records).resolve()
    selected = _stories(run, root)
    picture_types = None if picture_types_from is None else load_picture_types(picture_types_from)
    plan_dir.mkdir(parents=True)
    written, failed = [], {}
    for story in selected:
        source = records / f"{story['story_id']}.json"
        if not source.is_file():
            raise FileNotFoundError(f"planning has not run for {story['story_id']}: {source}")
        record = json.loads(source.read_text(encoding="utf-8"))
        if record["status"] != "completed":
            failed[story["story_id"]] = {"status": record["status"], "error": record.get("error"), "record": str(source)}
            continue
        source_board = record["storyboard"]
        if without_layouts:
            source_board = without_target_text(source_board, frozenset({"layouts"}))
        if without_changes:
            source_board = {**source_board, "shots": [{**shot, "changes": []} for shot in source_board["shots"]]}
        if picture_types is not None:
            source_board = with_picture_types(source_board, picture_types[story["story_id"]])
        write_json(plan_dir / f"{story['story_id']}.json", execution_storyboard(source_board))
        written.append(story["story_id"])
    provenance = {
        "source_records": str(records),
        "layouts": ("composition, relations and ending sides (and person boxes in older records) removed from every plan"
                    if without_layouts else "as written by the planner"),
        "changes": "emptied on every shot" if without_changes else "as written by the planner",
        "picture_types": ("none; the requests state no visual style" if picture_types_from is None
                          else f"stated in the request, written on every shot from {picture_types_from}"),
        "story_ids": written,
        "planning_failed": failed,
    }
    write_json(plan_dir / "provenance.json", provenance)
    return provenance


#: 带规划器 token 的条件：``token_edits_dir`` 是移点、换 token、去 token 检验的逐故事改动（目录下 <story_id>.json），
#: 普通出片为 null。
TOKEN_FIELDS = ("token_edits_dir",)


def reference_plan(annotation: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], *,
                   policy: str, frames_max: int | None) -> list[list[dict]]:
    """预检时各窗选中的参考（:func:`mspc.inference.segment_story.select_reference_shots`，与出片同一函数）。
    这里只记选帧；谁读哪张在出片时由 :func:`mspc.data.reference_routing.reference_readers` 定。CLIP 检索要看已生成的画面，
    预检选不出，各窗记空（实际参考与分数在出片的 ``segments.json``）。"""
    if policy == NO_REFERENCES or policy == CLIP_RETRIEVAL:
        if (policy == NO_REFERENCES) != (frames_max is None):
            raise ValueError("frames_max is given exactly when references are kept (history_policy other than none)")
        return [[] for _ in segments]
    if policy not in (PLAN_SELECTION, RECENT_SHOTS):
        raise ValueError(f"unknown history policy: {policy!r}")
    if type(frames_max) is not int:
        raise ValueError("history policies require frames_max from the renderer config")
    return [[{"shot": index + 1, "position": position}
             for index, position in select_reference_shots(policy, annotation, span=(segment["start_shot"],
                                                                                    segment["end_shot"]),
                                                           frames_max=frames_max)]
            for segment in segments]


def layout_bias_peak(*, strength: float, box: tuple[float, float, float, float] | None) -> float:
    """有框且 strength>0 时峰值应大于 0；strength=0 或没有框时峰值为 0。"""
    entities = () if box is None else (SpatialLayoutEntity((1,), (SpatialLayoutAnchor(0.25, box),)),)
    condition = SpatialLayoutCondition(
        shots=(SpatialLayoutShot(num_frames=9, entities=entities, text_token_mask=(True, True, True, False, False)),),
        strength=strength, layer_indices=(0,), timestep_range=(0.0, 1000.0), mode=SPATIAL_LAYOUT_MODE,
        region_dilation=None, onset_latent_frames=None)
    bias, = build_spatial_layout_biases(
        condition, latent_shot_groups=((0, 3),), latent_frames=3, patch_height=15, patch_width=26,
        prefix_frames=0, text_length=5, timestep=torch.tensor([500.0]), num_layers=30, device="cpu")
    return float(bias.abs().max())


def box_pathway() -> dict:
    structural = math.log(2)
    return {
        "source": PROBE_SOURCE,
        "box_xyxy": list(PROBE_BOX),
        "text_strength": 0.0,
        "text_max_abs": layout_bias_peak(strength=0.0, box=PROBE_BOX),
        "structural_strength": structural,
        "structural_max_abs": layout_bias_peak(strength=structural, box=PROBE_BOX),
        "structural_without_box_max_abs": layout_bias_peak(strength=structural, box=None),
    }


def film_paired_interval(left: Mapping[str, Mapping[str, float | None]],
                         right: Mapping[str, Mapping[str, float | None]],
                         films: Mapping[str, str], metric: str) -> dict:
    """右减左的故事配对差。点估计故事等权；区间按源片重采样，不跨种子平均。

    ``cluster_bootstrap`` 的 ``stories`` 在这里是源片数，``shots`` 是配对数。
    """
    stories, left_values, right_values = paired(left, right, metric)
    if len(stories) < 2:
        raise ValueError("film-grouped interval needs at least two paired stories")
    grouped: dict[str, list[float]] = {}
    for story, before, after in zip(stories, left_values, right_values, strict=True):
        if story not in films:
            raise ValueError(f"missing source film for {story}")
        grouped.setdefault(films[story], []).append(float(after - before))
    result = cluster_bootstrap(grouped)
    result["metric"] = metric
    result["difference"] = "right - left"
    result["story_ids"] = stories
    return result


def checkpoint_paths(checkpoint: str | Mapping[str, str] | None) -> tuple[str, ...]:
    """条件里的权重：原生为空、全量为一个文件、双专家为显式高／低噪 LoRA 对。"""
    if checkpoint is None:
        return ()
    if isinstance(checkpoint, Mapping):
        if set(checkpoint) != set(EXPERT_LORA_CHECKPOINT_KEYS):
            raise ValueError(f"expert checkpoint requires exactly {EXPERT_LORA_CHECKPOINT_KEYS}")
        paths = tuple(checkpoint[key] for key in EXPERT_LORA_CHECKPOINT_KEYS)
    else:
        paths = (checkpoint,)
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in paths):
        raise ValueError("checkpoint paths must be nonempty trimmed strings")
    return paths


def map_checkpoint_paths(checkpoint: str | Mapping[str, str] | None, transform):
    paths = checkpoint_paths(checkpoint)
    if isinstance(checkpoint, Mapping):
        return {key: transform(value) for key, value in zip(EXPERT_LORA_CHECKPOINT_KEYS, paths, strict=True)}
    return None if checkpoint is None else transform(paths[0])


def checkpoint_identity(inference: Mapping[str, Any]) -> str | dict[str, str] | None:
    """从渲染器已声明的权重来源取出条件须独立绑定的导出路径，不以单份全量权重代替专家对。"""
    checkpoint = inference["dit_checkpoint"]
    if inference["weight_source"] == "pretrained_model_paths":
        if checkpoint is not None:
            raise ValueError("native initialization requires dit_checkpoint=null")
        return None
    if inference["weight_source"] == "expert_lora_checkpoints":
        if checkpoint is not None:
            raise ValueError("expert LoRA initialization requires dit_checkpoint=null")
        pair = {key: inference["experts"][key] for key in EXPERT_LORA_CHECKPOINT_KEYS}
        checkpoint_paths(pair)
        return pair
    if inference["weight_source"] != "full_dit_checkpoint":
        raise ValueError("requires an explicit native, full DiT or expert LoRA initialization")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("full DiT initialization requires an explicit checkpoint")
    return checkpoint


def _checkpoint(config_path: Path) -> str | dict[str, str] | None:
    from mspc.config import load_experiment_config
    config = load_experiment_config(config_path)
    return map_checkpoint_paths(checkpoint_identity(config.resolved["inference"]), lambda value: str(Path(value).resolve()))


def scoring_rows(run: Mapping[str, Any], root: Path) -> list[dict]:
    """整次实验的评分清单：全部条件 × 全部故事。与哪个进程出了哪几部无关。"""
    rows = []
    for condition in run["conditions"]:
        output_dir = resolve_path(condition["output_dir"], root)
        for story in _stories(run, root):
            request = story["inputs"]["shots"]
            story_output = output_dir / story["story_id"]
            rows.append({
                "method": condition["name"],
                "story_id": story["story_id"],
                "input_tier": "shots",
                "seed": condition["seed"],
                "story_request": request["text"],
                "video_path": str(story_output / "video.mp4"),
                "generation_record_path": str(story_output / "generation.json"),
                "expected_shot_count": request["expected_shot_count"],
                "expected_cut_times_seconds": request["expected_cut_times_seconds"],
            })
    return rows


def write_scoring_manifest(manifest: Path) -> Path:
    """评分清单只由一个进程写一次；分片出片的进程不写。"""
    root = repo_root()
    run = load_run(manifest)
    scoring = {
        "evaluation_scope": "renderer",
        "comparisons": [{
            "input_tier": "shots",
            "seed": run["conditions"][0]["seed"],
            "methods": [condition["name"] for condition in run["conditions"]],
            "story_ids": [story["story_id"] for story in _stories(run, root)],
        }],
        "requests": scoring_rows(run, root),
    }
    manifest_path = resolve_path(run["run_dir"], root) / "inputs" / "requests.json"
    write_json(manifest_path, scoring)
    load_manifest(manifest_path)
    return manifest_path


def prepare(manifest: Path, *, story_ids: Sequence[str] | None, prepare_only: bool, runtimes,
            condition_names: Sequence[str] | None = None, label: str | None = None) -> dict:
    """按所给条件和故事出片（或只预检）。只出一部分时要给 label，证据写到 run_dir/evidence/<label>.json。
    ``runtimes`` 是 ``render_story.RuntimeCache``：同一进程里先后出的故事共用加载好的渲染运行时。"""
    from mspc.inference import render_story as render_entry

    root = repo_root()
    run = load_run(manifest)
    partial = story_ids is not None or condition_names is not None
    if partial and not label:
        raise ValueError("a partial run needs a label for its evidence file")
    conditions = run["conditions"]
    if condition_names is not None:
        unknown = set(condition_names) - {condition["name"] for condition in conditions}
        if unknown or len(condition_names) != len(set(condition_names)):
            raise ValueError(f"condition names are not distinct members of the run: {sorted(unknown)}")
        conditions = [condition for condition in conditions if condition["name"] in set(condition_names)]
    selected = _stories(run, root)
    if story_ids is not None:
        unknown = [story for story in story_ids if story not in {item["story_id"] for item in selected}]
        if unknown or len(story_ids) != len(set(story_ids)):
            raise ValueError(f"story ids are outside split {run['split']}: {unknown}")
        wanted = set(story_ids)
        selected = [story for story in selected if story["story_id"] in wanted]
    plan_dir = resolve_path(run["conditions"][0]["plan_dir"], root)
    provenance_path = plan_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None
    run_dir = resolve_path(run["run_dir"], root)
    reports = []
    for condition in conditions:
        config_path = resolve_path(condition["renderer_config"], root)
        checkpoint = _checkpoint(config_path)
        declared_checkpoint = map_checkpoint_paths(condition["checkpoint"], lambda value: str(Path(value).resolve()))
        if checkpoint != declared_checkpoint:
            raise ValueError(f"{condition['name']} checkpoint does not match {config_path}")
        if resolve_path(condition["plan_dir"], root) != plan_dir:
            raise ValueError("condition plan_dir drifted from the shared directory")
        output_dir = (run_dir / "preflight" / condition["name"] if prepare_only
                      else resolve_path(condition["output_dir"], root))
        output_dir.mkdir(parents=True, exist_ok=True)
        from mspc.config import load_experiment_config
        config = load_experiment_config(config_path)
        tokenizer = render_entry._load_tokenizer(config.to_dict()["model"])
        condition_report = {
            "name": condition["name"],
            "history_policy": condition["history_policy"],
            "segment_max_shots": condition["segment_max_shots"],
            "segment_max_frames": condition["segment_max_frames"],
            "spatial_layout": config.to_dict()["model"]["spatial_layout"],
            "checkpoint": checkpoint,
            "weight_source": config.to_dict()["inference"]["weight_source"],
            "stories": [],
        }
        failed_plans = {} if provenance is None else provenance.get("planning_failed", {})
        for story in selected:
            board_path = plan_dir / f"{story['story_id']}.json"
            story_output = output_dir / story["story_id"]
            if story["story_id"] in failed_plans:
                story_output.mkdir(parents=True, exist_ok=False)
                write_json(story_output / "generation.json", {"story_id": story["story_id"], "status": "planning_failed",
                                                          "plan": failed_plans[story["story_id"]]})
                condition_report["stories"].append({"story_id": story["story_id"], "status": "planning_failed",
                                                    "layouts_present": None, "segments": []})
                continue
            record = render_story(
                config, storyboard=board_path, output=story_output, story_id=story["story_id"],
                seed=condition["seed"], segment_seed_stride=condition["segment_seed_stride"],
                segment_max_shots=condition["segment_max_shots"], segment_max_frames=condition["segment_max_frames"],
                history_policy=condition["history_policy"],
                runtimes=runtimes,
                token_edits=(None if condition.get("token_edits_dir") is None else
                             resolve_path(condition["token_edits_dir"], root) / f"{story['story_id']}.json"),
                prepare_only=prepare_only)
            inputs = json.loads((story_output / "inputs.json").read_text(encoding="utf-8"))
            annotation = inputs["annotation"]
            memory = inputs["resolved_config"]["model"]["memory_prefix"]
            frames_max = None if condition["history_policy"] == "none" else memory["frames_max"]
            planned = reference_plan(annotation, inputs["segments"], policy=condition["history_policy"], frames_max=frames_max)
            segments = []
            for segment, references in zip(inputs["segments"], planned, strict=True):
                # 出片时的参考行同样只由分镜写，预检就是实际的行。
                reference_rows = [reference_row(annotation, reference["shot"] - 1, memory["row_policy"])
                                  for reference in references]
                segments.append({
                    "start_shot": segment["start_shot"],
                    "end_shot": segment["end_shot"],
                    "frame_counts": segment["frame_counts"],
                    "prompts": segment["prompts"],
                    "token_counts": segment["token_counts"],
                    "references": references,
                    "reference_token_counts": [render_entry._token_count(tokenizer, row) for row in reference_rows],
                })
            condition_report["stories"].append({
                "story_id": story["story_id"],
                "status": record["status"],
                "layouts_present": "layouts" in annotation,
                "segments": segments,
            })
        reports.append(condition_report)
    manifest_path = None if partial else write_scoring_manifest(manifest)
    evidence = {
        "prepare_only": prepare_only,
        "split": run["split"],
        "seed": run["conditions"][0]["seed"],
        "segment_seed_stride": run["conditions"][0]["segment_seed_stride"],
        "experiment_story_ids": [story["story_id"] for story in _stories(run, root)],
        "prepared_story_ids": [story["story_id"] for story in selected],
        "prepared_conditions": [condition["name"] for condition in conditions],
        "films": {story["story_id"]: story["film_key"] for story in selected},
        "plan_provenance": provenance,
        "scoring_manifest": None if manifest_path is None else str(manifest_path),
        "conditions": reports,
        "box_pathway": box_pathway(),
        "history_note": {
            PLAN_SELECTION: "窗前全部已生成镜为候选：窗内每个在场登记实体取它最近一次出现的镜，再加最近一次同地点的镜，"
                            "去重、至多 frames_max 张；镜内有变化取末帧、否则中帧；谁读哪张与训练同一个 reference_readers。"
                            "参考身份行写源镜分镜在场的登记人物，不读图。",
            RECENT_SHOTS: "张数与同一窗按分镜选帧相同，取窗前最近的那几镜中帧，不看分镜内容；之后的路由与身份行相同。",
            CLIP_RETRIEVAL: "张数与同一窗按分镜选帧相同；每个已生成镜存至多 6 张 CLIP 去重的帧，按与本窗各镜文字的 CLIP "
                            "文图相似度取前几镜、每镜最像的一帧，不看分镜内容；之后的路由与身份行相同。预检选不出，记空。",
        },
    }
    write_json(run_dir / "evidence" / f"{label}.json" if partial else run_dir / "prepare-evidence.json", evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    writing = commands.add_parser("write-given-plans")
    writing.add_argument("--conditions", type=Path, required=True)
    writing.add_argument("--story-ids", nargs="+")
    writing.add_argument("--frame-counts-from", type=Path)
    writing.add_argument("--picture-types-from", type=Path)
    planned = commands.add_parser("write-planner-plans")
    planned.add_argument("--conditions", type=Path, required=True)
    planned.add_argument("--records", type=Path, required=True)
    planned.add_argument("--without-layouts", action="store_true")
    planned.add_argument("--without-changes", action="store_true")
    planned.add_argument("--picture-types-from", type=Path)
    scoring = commands.add_parser("write-scoring-manifest")
    scoring.add_argument("--conditions", type=Path, required=True)
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--conditions", type=Path, required=True)
    preparing.add_argument("--story-ids", nargs="+")
    preparing.add_argument("--prepare-only", action="store_true")
    preparing.add_argument("--condition-names", nargs="+")
    preparing.add_argument("--label")
    args = parser.parse_args(argv)
    if args.command == "write-given-plans":
        frame_counts_from = args.frame_counts_from.resolve() if args.frame_counts_from else None
        picture_types_from = args.picture_types_from.resolve() if args.picture_types_from else None
        print(json.dumps(write_given_plans(args.conditions, story_ids=args.story_ids, frame_counts_from=frame_counts_from,
                                           picture_types_from=picture_types_from), ensure_ascii=False))
        return 0
    if args.command == "write-planner-plans":
        print(json.dumps(write_planner_plans(args.conditions, args.records, without_layouts=args.without_layouts,
                                             without_changes=args.without_changes,
                                             picture_types_from=(args.picture_types_from.resolve()
                                                                 if args.picture_types_from else None)),
                         ensure_ascii=False))
        return 0
    if args.command == "write-scoring-manifest":
        print(json.dumps({"scoring_manifest": str(write_scoring_manifest(args.conditions))}))
        return 0
    evidence = prepare(args.conditions, story_ids=args.story_ids, prepare_only=args.prepare_only,
                       runtimes=RuntimeCache(), condition_names=args.condition_names, label=args.label)
    summary = {
        "prepared_story_ids": evidence["prepared_story_ids"],
        "scoring_manifest": evidence["scoring_manifest"],
        "box_pathway": evidence["box_pathway"],
        "conditions": [{
            "name": condition["name"],
            "spatial_layout": condition["spatial_layout"],
            "stories": [{
                "story_id": story["story_id"],
                "segments": len(story["segments"]),
                "frames": [segment["frame_counts"] for segment in story["segments"]],
                "token_counts": [segment["token_counts"] for segment in story["segments"]],
                "references": [segment["references"] for segment in story["segments"]],
                "reference_token_counts": [segment["reference_token_counts"] for segment in story["segments"]],
                "layouts_present": story["layouts_present"],
            } for story in condition["stories"]],
        } for condition in evidence["conditions"]],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
