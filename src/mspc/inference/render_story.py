"""Render one multi-shot story video from a storyboard JSON via the Wan2.2 CaT-Movie pipeline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from mspc.config import ExperimentConfig, load_experiment_config
from mspc.comparison_methods.prompts import adapt_storyboard
from mspc.core.video import concat_shot_videos
from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES, STORYBOARD_FPS
from mspc.data.frame_planner import resample_frame_count
from mspc.data.scene_training_manifest import media_from_task
from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET, without_target_text
from mspc.data.storyboard import validate_storyboard
from mspc.inference.segment_story import HISTORY_POLICIES, generate_story, reference_row, require_story_renderer
from mspc.inference.wan22 import Wan22InferenceRuntime
from mspc.models.memory_prefix_options import MemoryPrefixModelOptions
from mspc.models.spatial_layout_inputs import (
    SpatialLayoutOptions, compile_spatial_layout_inputs, encode_spatial_layout_inputs,
)
from mspc.models.wan22_loader import require_model_paths, require_path


def read_storyboard(path: Path) -> dict:
    """只接受执行态Storyboard根对象，不猜规划答复或benchmark包装格式。"""
    return validate_storyboard(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_tokenizer(model: dict) -> Any:
    root = require_path(model["diffsynth_root"], "model.diffsynth_root", directory=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("diffsynth.models.wan_video_text_encoder")
    if not Path(module.__file__).resolve().is_relative_to(root / "diffsynth"):
        raise RuntimeError("the story tokenizer must come from model.diffsynth_root")
    return module.HuggingfaceTokenizer(name=model["paths"]["tokenizer"],
        seq_len=UMT5_PROMPT_TOKEN_BUDGET, clean="whitespace", local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int:
    # 同一wrapper的真实清理和tokenizer；关闭截断来检查原512-token消费者的容量。
    return len(tokenizer.tokenizer([tokenizer._clean(text)], padding=False, truncation=False,
                                  add_special_tokens=True)["input_ids"][0])


@dataclass(frozen=True)
class PreparedStory:
    annotation: dict
    segments: list[dict]
    runtime_kwargs: dict
    token_counter: Callable[[str], int]
    frames_max: int | None
    ffmpeg: str | None


def prepare_story(config: ExperimentConfig, *, storyboard: Path, story_id: str, seed: int,
                  segment_seed_stride: int, segment_max_shots: int | None, segment_max_frames: int,
                  history_policy: str) -> PreparedStory:
    """只做CPU预检；不构造renderer。参考行只由分镜写（:func:`mspc.inference.segment_story.reference_row`），预检就核每镜
    参考行的 token 数。

    窗口先由共同计划的16fps逐镜帧数切：镜头连续、整窗 ≤``segment_max_frames``（1..``SEGMENT_MAX_FRAMES``，单镜超过时独占一窗），
    从第一镜贪心装满（:func:`mspc.comparison_methods.prompts.pack_segments`；计划不带窗口字段，以后按内容分窗只换那一个函数）。
    新原生配方给128（约8秒源窗）；按task媒体派生执行帧数后不重装窗，两底座的镜头span保持一致。
    ``segment_max_shots`` 是渲染器训练时的窗口镜数：``None`` 不限镜数（最终联合配方，训练窗口只限帧数）；1 为逐镜，
    之前各镜都成为干净历史；更早只训过至多 4 镜窗口的渲染器给 4。参考按本段自己选（见 :mod:`mspc.inference.segment_story`）。"""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if type(seed) is not int or type(segment_seed_stride) is not int or segment_seed_stride < 0:
        raise ValueError("seed must be an integer and segment_seed_stride a nonnegative integer")
    if not isinstance(story_id, str) or not story_id.strip() or story_id != story_id.strip():
        raise ValueError("story_id must be an explicit nonempty trimmed string")
    if history_policy not in HISTORY_POLICIES:
        raise ValueError(f"unknown history policy: {history_policy!r}")
    if segment_max_shots is not None and (type(segment_max_shots) is not int or segment_max_shots < 1):
        raise ValueError("segment_max_shots must be None (frame-limited windows) or a positive integer")
    if type(segment_max_frames) is not int or not 1 <= segment_max_frames <= SEGMENT_MAX_FRAMES:
        raise ValueError(f"segment_max_frames must be an integer in 1..{SEGMENT_MAX_FRAMES}")
    data = config.to_dict()
    model, inference = data["model"], data["inference"]
    if model["type"] != "wan22_diffsynth" or not inference["enabled"]:
        raise ValueError("story rendering requires wan22_diffsynth and inference.enabled=true")
    if inference["weight_source"] not in ("pretrained_model_paths", "full_dit_checkpoint", "expert_lora_checkpoints"):
        raise ValueError("story rendering requires explicitly selected pretrained paths, a full DiT checkpoint "
                         "or the two A14B experts' LoRA exports")
    model["paths"] = require_model_paths(model["paths"])
    checkpoint = inference["dit_checkpoint"]
    if inference["weight_source"] == "full_dit_checkpoint":
        checkpoint = str(require_path(checkpoint, "inference.dit_checkpoint"))
    elif checkpoint is not None:
        raise ValueError(f"{inference['weight_source']} requires dit_checkpoint=null")
    if inference["experts"] is not None:
        experts = inference["experts"]
        experts["low_noise_dit"] = [str(require_path(path, "inference.experts.low_noise_dit"))
                                    for path in experts["low_noise_dit"]]
        for name in ("high_noise_lora", "low_noise_lora"):
            experts[name] = str(require_path(experts[name], f"inference.experts.{name}"))
    media = media_from_task(data["task"])
    if (inference["fps"] != media["target_fps"] or inference["target_height"] != media["height"]
            or inference["target_width"] != media["width"]):
        raise ValueError("story inference media must match the explicit task media")
    if inference["target_height"] % 32 or inference["target_width"] % 32:
        raise ValueError("Wan output height and width must be multiples of the 32-pixel output grid")
    require_story_renderer(model)
    memory = model["memory_prefix"]
    frames_max = None
    if history_policy != "none":
        if memory is None:
            raise ValueError("history conditioning requires model.memory_prefix")
        frames_max = MemoryPrefixModelOptions.from_mapping(memory).frames_max
    annotation = read_storyboard(storyboard)
    # 与训练同一规则：``model.omit_target_text`` 里的部分不编进目标文字（计划带了也省）。
    omitted = frozenset(model["omit_target_text"])
    if "layouts" in omitted and model["spatial_layout"] is not None:
        raise ValueError("a renderer trained without layout sentences takes no layout biases")
    text_annotation = without_target_text(annotation, omitted)
    options = None if model["spatial_layout"] is None else SpatialLayoutOptions.from_mapping(model["spatial_layout"])
    # 计划缺 layouts 就是没有框：不补空列表，也不编区域偏置。有该字段才编译。
    if options is None or "layouts" not in annotation:
        layouts = None
    else:
        layouts = compile_spatial_layout_inputs(annotation, tuple(range(len(annotation["shots"]))))
    tokenizer = _load_tokenizer(model)
    count_tokens = lambda text: _token_count(tokenizer, text)
    segments = adapt_storyboard(text_annotation, family="wan_segment", limits={
        "max_shots": segment_max_shots, "max_frames": segment_max_frames,
        "max_tokens": UMT5_PROMPT_TOKEN_BUDGET}, count_tokens=count_tokens)
    # 先在共同source坐标装窗，再派生本底座的真实执行帧数；不能只换输出fps，也不能因24fps取整改两组source spans。
    for segment in segments:
        source_counts = segment["frame_counts"]
        counts = [resample_frame_count(count, source_fps=STORYBOARD_FPS,
                    target_fps=media["target_fps"], temporal_stride=media["vae_temporal_stride"])
                  for count in source_counts]
        segment.update(source_frame_counts=source_counts, source_fps=STORYBOARD_FPS,
                       output_fps=media["target_fps"], frame_counts=counts, num_frames=sum(counts))
    if history_policy != "none":
        counts = [count_tokens(reference_row(annotation, index, memory["row_policy"]))
                  for index in range(len(annotation["shots"]))]
        if max(counts) > UMT5_PROMPT_TOKEN_BUDGET:
            raise ValueError(f"a reference row exceeds the umT5 token budget: {counts}")
    if layouts is not None:
        for segment in segments:
            encode_spatial_layout_inputs(tokenizer, layouts[segment["start_shot"]:segment["end_shot"]],
                prompts=segment["prompts"], frame_counts=segment["frame_counts"], options=options)
    ffmpeg = shutil.which("ffmpeg") if len(segments) > 1 else None
    if len(segments) > 1 and ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to concatenate the story's generated segments")
    inference["model_paths"] = model["paths"]
    inference["seed"] = seed
    return PreparedStory(annotation, segments,
        {"model_cfg": model, "inference_cfg": inference, "pipeline_cfg": data["pipeline_inputs"], "dit_ckpt": checkpoint},
        count_tokens, frames_max, ffmpeg)


class RuntimeCache:
    """一个进程里连续或同时出片的故事共用的渲染运行时：构造参数相同就复用，不同就等正在用的故事都出完、
    释放旧的再建（同时只留一份 5B）。几部故事同时用一份时，出片那一段由 ``runtime.lock`` 轮流独占。

    运行时本来就跨段反复调用 ``generate_case``（每段种子显式给），跨故事复用与跨段相同。"""

    def __init__(self) -> None:
        self.key: str | None = None
        self.runtime: Any = None
        self.users = 0
        self.wanted: str | None = None  # 有故事在等换成这份参数时，旧参数的新故事也先等，免得它一直等不到
        self.changed = threading.Condition()

    @contextmanager
    def use(self, runtime_kwargs: Mapping[str, Any]):
        # 故事种子写在 inference_cfg 里只作记录：运行时不读它，每段种子由 generate_case 显式给。不剔掉它，
        # 同卡相邻条目换个种子就会重载整个模型（11-05 实测换后仍每部加载约 35 秒）。
        inference = {key: value for key, value in runtime_kwargs["inference_cfg"].items() if key != "seed"}
        key = json.dumps({**runtime_kwargs, "inference_cfg": inference}, sort_keys=True, default=str)
        with self.changed:
            while (key != self.key and self.users) or (key == self.key and self.wanted not in (None, key)):
                if key != self.key:
                    self.wanted = key
                self.changed.wait()
            if key != self.key:
                self.key, self.runtime = None, None
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.runtime = Wan22InferenceRuntime(**runtime_kwargs)
                self.key, self.wanted = key, None
                self.changed.notify_all()
            self.users += 1
            runtime = self.runtime
        try:
            yield runtime
        finally:
            with self.changed:
                self.users -= 1
                self.changed.notify_all()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def render_story(config: ExperimentConfig, *, storyboard: Path, output: Path, story_id: str,
                 seed: int, segment_seed_stride: int, segment_max_shots: int | None, segment_max_frames: int,
                 history_policy: str,
                 runtimes: RuntimeCache,
                 token_edits: Path | None = None,
                 prepare_only: bool = False) -> dict:
    """新目录内执行一条故事；失败保留已交片段与错误，不续跑或静默重抽。

    ``token_edits`` 是移点、换 token、去 token 检验的改动（JSON：段序号 → TokenEdit 字段的列表），只随带 token 的渲染器给，
    整片重出。
    ``runtimes`` 是本进程的 :class:`RuntimeCache`：同进程连续出几部故事时只加载一次模型。"""
    if token_edits is not None and config.to_dict()["model"].get("planner_tokens") is None:
        raise ValueError("token_edits belong to a renderer with planner tokens")
    edits = None
    if token_edits is not None:
        from mspc.inference.planner_token_reading import TokenEdit
        edits = {int(k): tuple(TokenEdit.from_mapping(e) for e in v)
                 for k, v in json.loads(Path(token_edits).read_text(encoding="utf-8")).items()}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, records = time.perf_counter(), []
    record = {"story_id": story_id, "status": "preflight_started", "stage": "preflight",
              "storyboard": str(Path(storyboard).resolve()), "config": str(config.source_path),
              "output": str(output), "seed": seed, "segment_seed_stride": segment_seed_stride,
              "segment_max_shots": segment_max_shots, "segment_max_frames": segment_max_frames,
              "history_policy": history_policy,
              "prepare_only": prepare_only, "completed_segments": 0,
              "token_edits": None if token_edits is None else json.loads(Path(token_edits).read_text(encoding="utf-8"))}
    _write_json(output / "generation.json", record)
    try:
        prepared = prepare_story(config, storyboard=storyboard, story_id=story_id, seed=seed,
            segment_seed_stride=segment_seed_stride, segment_max_shots=segment_max_shots,
            segment_max_frames=segment_max_frames, history_policy=history_policy)
        record.update(weight_source=prepared.runtime_kwargs["inference_cfg"]["weight_source"],
                      dit_checkpoint=prepared.runtime_kwargs["dit_ckpt"],
                      experts=prepared.runtime_kwargs["inference_cfg"]["experts"],
                      model_paths=prepared.runtime_kwargs["model_cfg"]["paths"])
        _write_json(output / "inputs.json", {**record, "resolved_config": config.to_dict(),
            "runtime": prepared.runtime_kwargs, "annotation": prepared.annotation,
            "segments": prepared.segments,
            "segment_seeds": [seed + index * segment_seed_stride for index in range(len(prepared.segments))],
            "ffmpeg": prepared.ffmpeg})
        record.update(status="prepared", stage="preflight_completed", planned_segments=len(prepared.segments))
        _write_json(output / "generation.json", record)
        if prepare_only:
            return record
        record.update(status="running", stage="runtime_loading")
        _write_json(output / "generation.json", record)
        with runtimes.use(prepared.runtime_kwargs) as runtime:
            record["stage"] = "generation"
            _write_json(output / "generation.json", record)

            def on_segment(segment: dict) -> None:
                records.append(segment)
                _write_json(output / "segments.json", records)
                record["completed_segments"] = len(records)
                _write_json(output / "generation.json", record)

            generate_story(runtime, annotation=prepared.annotation,
                shot_ids=[f"shot-{shot['shot']}" for shot in prepared.annotation["shots"]],
                segments=prepared.segments, output=output / "segments", token_counter=prepared.token_counter,
                frames_max=prepared.frames_max, seed=seed, segment_seed_stride=segment_seed_stride,
                on_segment=on_segment, history_policy=history_policy, token_edits=edits)
            record["stage"] = "concatenation"
            _write_json(output / "generation.json", record)
            video = concat_shot_videos([segment["output"] for segment in records],
                                      output_path=output / "video.mp4", ffmpeg=prepared.ffmpeg)
        record.update(status="completed", stage="completed", video=str(video),
                      elapsed_seconds=time.perf_counter() - started)
        _write_json(output / "generation.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      completed_segments=len(records), elapsed_seconds=time.perf_counter() - started)
        _write_json(output / "generation.json", record)
        raise


def window_shots(text: str) -> int | None:
    """命令行的窗口镜数：``none`` 不限（窗口只限源帧预算），否则正整数。"""
    if text == "none":
        return None
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError("expected none or a positive integer")
    return int(text)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "storyboard"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="新输出目录，保存输入、逐段记录和video.mp4")
    result.add_argument("--story-id", required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--segment-seed-stride", type=int, required=True)
    result.add_argument("--segment-max-shots", type=window_shots, required=True,
                        help="渲染器训练时的窗口镜数：none 不限（窗口只限帧数）；1 为每镜单独一段；旧的至多 4 镜渲染器给 4")
    result.add_argument("--segment-max-frames", type=int, required=True,
                        help=f"共同16fps计划的源窗预算 1..{SEGMENT_MAX_FRAMES}；新原生配方给128，超帽完整单镜独占")
    result.add_argument("--history-policy", choices=HISTORY_POLICIES, required=True,
                        help="none 不给参考；plan_selection 按分镜选帧；recent_shots 取最近几镜；clip_retrieval 按画面相似度选帧（CLIP 文图检索）；后两种张数与按分镜选帧相同")
    result.add_argument("--token-edits", type=Path, help="带 token 的渲染器：移点、换 token 检验的改动 JSON")
    result.add_argument("--prepare-only", action="store_true", help="仅CPU预检并保存输入，不构造模型或生成")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    record = render_story(load_experiment_config(args.config), storyboard=args.storyboard,
        output=args.output, story_id=args.story_id, seed=args.seed, segment_seed_stride=args.segment_seed_stride,
        segment_max_shots=args.segment_max_shots, segment_max_frames=args.segment_max_frames,
        history_policy=args.history_policy,
        runtimes=RuntimeCache(), token_edits=args.token_edits, prepare_only=args.prepare_only)
    print(json.dumps(record, ensure_ascii=False))
    return 0
