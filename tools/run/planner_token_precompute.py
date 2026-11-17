"""点控制加特征注入的规划器特征预计算：每个训练样本，Qwen3.5-9B 底座（不挂 LoRA）对「这一段渲染器拿到的参考帧＋段前最近
一镜末帧＋不带方位句的逐镜文字＋人名与镜首点拼成的作答」做一次教师强制前向，取作答里每人每镜条目在配置那一层、那种口径的
平均隐状态，写 ``<features_root>/<样本号:06d>.npz``：

- ``shot_<口径>_layer<L>`` [镜首条目数, 4096] float16，配 ``shot_index``（块内镜下标，从 0 起）、``shot_name``。

与推理同一条路：取特征是 ``PlannerTokenReader.plan_features``（``planner_reading_service`` 出片时调的也是它），提示是
``reference_pointing.plan_token_content``，作答是 ``answer_from_starts``。训练这一侧的输入：

- 参考帧：用训练配置建 ``BlockDataset``，取 ``draw(i).references``，与渲染器训练时拿到的同一份现抽（整组丢弃时规划器也
  看不到参考）；段前末帧是目标段前一镜的最后一帧。图片按训练取帧读真实源片（与渲染器读参考帧同一裁切），每段一组亮度、
  噪声、JPEG 增强（``planner_token_features.augment``）。生成画面与真片的差别没有增强，验收时看。
- 逐镜文字：``BlockDataset.text`` 给渲染器的目标镜文字（带 token 的配置不带方位句）。
- 镜首点：标签（标注掩码外接框中心），不抖动；抖动与丢点只作用在渲染器的 token 放置上。

先写临时文件再改名，进程中途被杀不会留下被当成已完成的半截文件。按 --shard/--shards 分片，已有输出的跳过（续跑）。
源片解码与增强在 --decode-workers 个后台线程里提前做（ffmpeg 子进程与 PIL 不占 GIL），与 GPU 前向重叠。

用法：python planner_token_precompute.py --config C --split train|val --model Q --ffmpeg F --shard i --shards n \\
       --decode-workers 2 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/wan22_runtime"), str(ROOT)]


def wanted_frames(member, draw) -> list[tuple[int, int]]:
    """规划器要看的帧：渲染器这一段的各参考，再加段前一镜的末帧；(镜下标, 训练帧号)。"""
    start = draw.span[0]
    last = [(start - 1, member["shots"][start - 1]["frames"] - 1)] if start else []
    return [tuple(r) for r in draw.references] + last


def decode_frames(member, wanted, *, ffmpeg, error_path) -> list[np.ndarray]:
    """按训练取帧读真实源片（整帧按 crop_box 裁、再居中缩放到 832×480），与训练读参考帧同一裁切；用 ffmpeg 顺序读，
    规划器环境没有 PyAV。ffmpeg 报错写进 ``error_path``，成功后删掉。"""
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
    from PIL import Image
    from mspc.data.frames import center_crop, iter_scene_frames
    from mspc.data.shot_targets import planned_row

    scene = {**member["scene"], "shots": member["shots"]}
    grids = {i: planned_row(scene, i, media=ANNOTATION_MEDIA)["source_frame_indices_by_shot"][0] for i in {shot for shot, _ in wanted}}
    numbers = [grids[shot][frame] for shot, frame in wanted]
    decoded = {}
    for number, array in iter_scene_frames(member["scene"], sorted(set(numbers)), ffmpeg=ffmpeg,
                                           error_path=error_path, video_decode_num_threads=4):
        decoded[number] = np.asarray(center_crop(Image.fromarray(array).crop(member["scene"]["crop_box"])))
    error_path.unlink()
    return [decoded[n] for n in numbers]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--config", "--model", "--ffmpeg"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    for flag in ("--shard", "--shards", "--decode-workers"):
        parser.add_argument(flag, type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from mspc.config import load_experiment_config
    from mspc.inference.planner_token_features import augment
    from mspc.inference.planner_token_reading import PlannerTokenReader
    from mspc.training.block_dataset import BlockDataset, build_validation_dataset
    from mspc.training.training_args import build_common_training_args

    training_args = build_common_training_args(load_experiment_config(args.config))
    if args.split == "train":
        with open(training_args.dataset_cfg["sampling_manifest_path"]) as stream:
            budget = sum(1 for _ in stream)
        samples = BlockDataset(**{**training_args.dataset_cfg, "sample_budget": budget})
    else:
        samples = build_validation_dataset(training_args)
    tokens = samples.planner_tokens
    if tokens is None or tokens.cfg["content"] != "planner_features":
        raise ValueError(f"{args.config} does not train on planner features")
    features_root = tokens.features_root
    suffix = f"{tokens.cfg['feature_pooling']}_layer{tokens.cfg['feature_layer']}"
    indices = list(range(len(samples)))
    if args.limit is not None:
        indices = indices[:args.limit]
    indices = indices[args.shard::args.shards]
    features_root.mkdir(parents=True, exist_ok=True)
    reader = PlannerTokenReader(model_path=args.model, device="cuda", feature_pooling=tokens.cfg["feature_pooling"],
                                feature_layer=tokens.cfg["feature_layer"], max_new_tokens=1, cut_margin=0)
    timing_path = features_root / f"timing.{args.shard:02d}-of-{args.shards:02d}.jsonl"
    errors_root = features_root / "decode-errors"
    errors_root.mkdir(exist_ok=True)

    def load_images(index):
        """源片解码、裁切、增强（后台线程）；返回 (图列表, 秒)。"""
        started = time.monotonic()
        draw = samples.draw(index)
        member = samples.members[draw.row.case_id]
        wanted = wanted_frames(member, draw)
        if not wanted:
            return [], time.monotonic() - started
        # 各线程各写一份 ffmpeg 报错文件，成功后删掉，只留出错样本的。
        decoded = decode_frames(member, wanted, ffmpeg=args.ffmpeg, error_path=errors_root / f"{index:06d}.stderr")
        frames = augment(np.stack(decoded), seed=samples.training_seed * 1_000_003 + index)
        return list(frames), time.monotonic() - started

    todo = [i for i in indices if not (features_root / f"{i:06d}.npz").exists()]
    with ThreadPoolExecutor(args.decode_workers) as pool:
        queue, upcoming = deque(), iter(todo)

        def submit():
            index = next(upcoming, None)
            if index is not None:
                queue.append((index, pool.submit(load_images, index)))

        for _ in range(args.decode_workers + 1):
            submit()
        while queue:
            started = time.monotonic()
            index, pending = queue.popleft()
            images, decode_seconds = pending.result()
            submit()
            waited = time.monotonic()
            draw = samples.draw(index)
            starts = tokens.starts(index, draw.span)
            count = len(draw.references)
            if starts:
                value = reader.plan_features(
                    samples.annotations[draw.row.case_id], span=draw.span, images=images, references=draw.references,
                    last_shot=draw.span[0] - 1 if draw.span[0] else None,
                    target_prompts=list(samples.text(index).prompts[count:]), starts=starts)
            else:
                value = np.zeros((0, 4096), dtype=np.float32)
            computed = time.monotonic()
            arrays = {f"shot_{suffix}": np.asarray(value, dtype=np.float16),
                      "shot_index": np.array([i for i, _, _ in starts], dtype=np.int64),
                      "shot_name": np.array([n for _, n, _ in starts], dtype=str)}
            path = features_root / f"{index:06d}.npz"
            # 临时文件带进程号：分片偶尔同时算到同一条，共用一个临时名会让一方改名时找不到文件。
            partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
            with partial.open("wb") as stream:
                np.savez(stream, **arrays)
            os.replace(partial, path)
            # decode 是后台线程里的解码与增强；wait 是主线程等它的时间（重叠充分时接近 0）；forward 含处理器与前向。
            timing = {"sample_index": index, "decode_seconds": decode_seconds, "wait_seconds": waited - started,
                      "forward_seconds": computed - waited, "total_seconds": time.monotonic() - started,
                      "images": len(images), "references": count, "entries": len(starts)}
            with timing_path.open("a") as stream:
                stream.write(json.dumps(timing) + "\n")
            print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
