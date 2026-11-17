"""场景级清单 → 训练窗口与教师标注故事：规格、排除与切窗（纯函数，CPU）。

镜头合格、渲染帧数、片名与裁切框一律复用测试集线
``mspc.benchmark.long_story_holdout`` 的同名规则（:func:`clean_runs`、:func:`render_frame_count`、
:func:`film_key`、:func:`crop_box`），训练窗与留出故事窗按同一把尺子定义「一镜」「一段」，这里不另写一份。

两种产物：

- **训练窗**（R3 用）：每个合格场景的每段连续合格镜（run）从头往后**不重叠**地切 1–5 镜窗口；余量≥3时K由
  ``random.Random(seed ^ 场景序号)`` 在 3..5 里抽；装不下单段上限（≤ :data:`MAX_SEGMENT_FRAMES` 帧）时换更小的
  K，3 镜也装不下就跳过这一镜往后。末尾或原短段1／2镜各成窗，不设窗长下限。不重叠是为了每个真镜在训练池里只出现一次。
- **教师故事**（P1 标注用）：≥6 镜场景里 6–12 镜的连续合格窗，规则同留出集的 ``story_windows``，每个场景
  至多取一条。

排除：运行 config 显式指定的冻结排除清单（按片排除，不只按场景）、视频带硬字幕标记的场景、
CinePile 无片名片段不排除（它不在留出集）。
"""

from __future__ import annotations

import random
import math

from mspc.data.scene_shot_manifest import MIN_SHOT_SECONDS
from typing import Any, Iterable, Mapping, Sequence

from mspc.benchmark.long_story_holdout import (
    MAX_SEGMENT_FRAMES,
    MAX_SEGMENT_SHOTS,
    SUBTITLE_FLAG,
    clean_runs,
    crop_box,
    film_key,
    render_frame_count,
    shot_seconds,
    story_windows,
)

TRAIN_MAX_SHOTS = MAX_SEGMENT_SHOTS
SHORT_WINDOW_VERSION = "scene_1to5_v3_short"
TEACHER_MIN_SHOTS, TEACHER_MAX_SHOTS = 6, 12
WINDOW_SCHEMA = "scene_window_v1"


def _frames(row: Mapping[str, Any], shots: Sequence[Mapping[str, Any]]) -> list[int]:
    return [render_frame_count(shot_seconds(row, shot)) for shot in shots]


def window_record(row: Mapping[str, Any], shots: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, Any]:
    """一条窗口：源视频、裁切框、逐镜源帧区间与渲染帧数（4n+1，渲染帧率为时空契约 ``target_fps``）。"""

    frames = _frames(row, shots)
    first = shots[0]["source_shot_index"]
    return {
        "schema_version": WINDOW_SCHEMA,
        "window_id": f"{row['scene_id']}:{kind}:{first:04d}-{first + len(shots) - 1:04d}",
        "kind": kind,
        "scene_id": row["scene_id"],
        "film_key": None if row["source"] == "cinedance" else film_key(row),
        "source": row["source"],
        "source_video_path": row["source_video_path"],
        "fps": row["fps"],
        "container_preroll_frames": row["container_preroll_frames"],
        "crop_box": crop_box(row),
        "shot_count": len(shots),
        "render_frames": sum(frames),
        "shots": [
            {
                "source_shot_index": shot["source_shot_index"],
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "seconds": round(shot_seconds(row, shot), 3),
                "render_frames": count,
            }
            for shot, count in zip(shots, frames)
        ],
    }


def training_runs(row: Mapping[str, Any]) -> tuple[list[list[dict]], list[dict]]:
    """训练镜头同一把尺子；像素 flags 由来源测量层提供，不在此补测。"""
    rejected, shots = [], []
    for shot in row["shots"]:
        reason = (
            "resolution_below_minimum" if row["active_box"][2] - row["active_box"][0] < 832
                or row["active_box"][3] - row["active_box"][1] < 480 else
            "fps_below_minimum" if row["fps"] < 16 else
            "shorter_than_min_seconds" if shot_seconds(row, shot) < MIN_SHOT_SECONDS else
            "shot_over_frame_cap" if render_frame_count(shot_seconds(row, shot)) > 228 else
            "source_flags" if shot["flags"] else None
        )
        if reason:
            rejected.append({"source_shot_index": shot["source_shot_index"], "reason": reason})
        else:
            shots.append(shot)
    # clean_runs owns index continuity; source intervals must also touch.
    runs = []
    for run in clean_runs({**row, "shots": shots}):
        current = []
        for shot in run:
            if current and current[-1]["end_frame"] != shot["start_frame"]:
                runs.append(current)
                current = []
            current.append(shot)
        if current:
            runs.append(current)
    return runs, rejected


def training_windows(row: Mapping[str, Any], *, seed: int, scene_index: int,
                     stride_shots: int | None = None, audit: dict | None = None) -> list[dict[str, Any]]:
    """3–5镜铺法与随机流不变；余下1／2镜成窗，不设窗长下限、不消耗随机数。"""
    if stride_shots is not None and (type(stride_shots) is not int or stride_shots < 1):
        raise ValueError("stride_shots must be a positive integer or None (window length)")
    rng = random.Random(seed ^ scene_index)
    windows, dropped, short_rejected = [], [], []
    runs, rejected = training_runs(row)
    for run in runs:
        start = 0
        while len(run) - start >= 3:
            wanted = rng.randint(3, TRAIN_MAX_SHOTS)
            taken = None
            for count in range(min(wanted, len(run) - start), 2, -1):
                if sum(_frames(row, run[start:start + count])) <= MAX_SEGMENT_FRAMES:
                    taken = count
                    break
            if taken is None:
                dropped.append({"source_shot_index": run[start]["source_shot_index"],
                                "reason": "window_over_frame_cap"})
                start += 1
                continue
            windows.append(window_record(row, run[start:start + taken], kind="train"))
            start += taken if stride_shots is None else stride_shots
        remainder = run[start:]
        if remainder:
            total = sum(_frames(row, remainder))
            reason = "short_window_over_frame_cap" if total > MAX_SEGMENT_FRAMES else None
            if reason:
                dropped.extend({"source_shot_index": s["source_shot_index"], "reason": reason} for s in remainder)
                short_rejected.append({"source_shot_indices": [s["source_shot_index"] for s in remainder],
                                       "shot_count": len(remainder), "render_frames": total, "reason": reason})
            else:
                window = window_record(row, remainder, kind="train")
                window.update(window_id=f"{SHORT_WINDOW_VERSION}:{window['window_id']}",
                              cut_version=SHORT_WINDOW_VERSION)
                windows.append(window)
    if audit is not None:
        used = {s["source_shot_index"] for w in windows for s in w["shots"]}
        audit.update(eligible_shots=sum(map(len, runs)), covered_shots=len(used),
                     dropped=rejected + [d for d in dropped if d["source_shot_index"] not in used],
                     rejected_short_windows=short_rejected)
    return windows


def teacher_story(row: Mapping[str, Any], *, seed: int, scene_index: int) -> dict[str, Any] | None:
    """一个场景至多一条 6–12 镜教师故事：K 在场景能给的范围里随机，起点在合格窗里随机。"""

    rng = random.Random(seed ^ scene_index)
    counts = list(range(TEACHER_MIN_SHOTS, TEACHER_MAX_SHOTS + 1))
    rng.shuffle(counts)
    for count in counts:
        options = story_windows(row, count)
        if options:
            return window_record(row, options[rng.randrange(len(options))], kind="teacher")
    return None


def sample_frame_positions(shot: Mapping[str, Any], fractions: Sequence[float]) -> list[int]:
    """一镜源帧区间 ``[start, end)`` 里按比例取的帧号（清单帧号，即 ffmpeg 按 edit list 解码的帧序号）。"""

    start, end = int(shot["start_frame"]), int(shot["end_frame"])
    if end <= start:
        raise ValueError(f"empty shot [{start}, {end})")
    return [start + int(fraction * (end - start - 1)) for fraction in fractions]


def build(
    rows: Iterable[Mapping[str, Any]], *, held_out_films: set[str], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """全部场景 → (训练窗, 教师故事, 计数)。场景序号按读入顺序。"""

    train: list[dict[str, Any]] = []
    teacher: list[dict[str, Any]] = []
    counts = {"scenes": 0, "held_out_scenes": 0, "subtitle_scenes": 0}
    for index, row in enumerate(rows):
        counts["scenes"] += 1
        if SUBTITLE_FLAG in row["video_flags"]:
            counts["subtitle_scenes"] += 1
            continue
        if film_key(row) in held_out_films:
            counts["held_out_scenes"] += 1
            continue
        train.extend(training_windows(row, seed=seed, scene_index=index))
        story = teacher_story(row, seed=seed, scene_index=index)
        if story is not None:
            teacher.append(story)
    return train, teacher, counts


def balanced_block_sizes(length: int, max_shots: int) -> list[int]:
    """按镜均分：前面的余数块各多一镜，块数恰为 ceil(length / max_shots)。"""
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if type(max_shots) is not int or max_shots < 1:
        raise ValueError("max_shots must be a positive integer")
    count = (length + max_shots - 1) // max_shots
    size, extra = divmod(length, count)
    return [size + (index < extra) for index in range(count)]


def story_blocks(scenes: Sequence[Mapping[str, Any]], *, sid: str, max_shots: int,
                 version: str, min_window_seconds: float, thresholds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    """判过且带质量 flags／分数的源场景 → 块。用修界前区间连段，复用已存修界；不依赖窗。"""
    from mspc.data.shot_gate import RULE_VERSION, revised_scene_window

    balanced_block_sizes(1, max_shots)
    if not isinstance(version, str) or not version:
        raise ValueError("blocks.version must be a nonempty string")
    if isinstance(min_window_seconds, bool) or not math.isfinite(min_window_seconds) or min_window_seconds <= 0:
        raise ValueError("min_window_seconds must be positive and finite")
    runs, rejected = [], []
    for row in scenes:
        scene_runs, dropped = training_runs(row)
        by_index = {shot["source_shot_index"]: shot for shot in row["shots"]}
        for item in dropped:
            shot = by_index[item["source_shot_index"]]
            if shot["eligible"]:
                rejected.append({"sid": sid, "scene_id": row["scene_id"], **item,
                                 "reasons": shot["flags"] if item["reason"] == "source_flags" else [item["reason"]]})
        runs.extend((row, run) for run in scene_runs)
    runs.sort(key=lambda pair: (pair[1][0]["start_frame"], pair[0]["scene_id"]))
    blocks = []
    for row, run in runs:
        offset = 0
        for size in balanced_block_sizes(len(run), max_shots):
            selected = run[offset:offset + size]
            offset += size
            scene = window_record(row, selected, kind="block")
            case_id = f"{version}:{row['scene_id']}:{selected[0]['source_shot_index']:04d}-{selected[-1]['source_shot_index']:04d}"
            scene.update(window_id=case_id, cut_version=version, width=row["width"], height=row["height"],
                         video_frame_count=row["video_frame_count"], active_box=row["active_box"],
                         frame_basis="ffmpeg_sequential_zero_based")
            guard = thresholds["boundary_guard_frames"]
            gates = [{**shot["gate"], "video": row["source_video_path"],
                      "segment": [shot["start_frame"], shot["end_frame"]],
                      "rule_version": RULE_VERSION, "coordinate": "ffmpeg_sequential",
                      "boundary_guard_frames": guard,
                      "boundary_revised_segment": [shot["gate"]["revised_segment"][0] - guard,
                                                   shot["gate"]["revised_segment"][1] + guard]} for shot in selected]
            scene = revised_scene_window(scene, {"window_id": case_id, "family": "scene_window_v1", "status": "kept",
                "coordinate": "ffmpeg_sequential", "rule_version": RULE_VERSION,
                "boundary_guard_frames": guard, "thresholds": dict(thresholds), "shots": gates})
            if sum(shot["seconds"] for shot in scene["shots"]) < min_window_seconds:
                rejected.extend({"sid": sid, "scene_id": row["scene_id"], "source_shot_index": s["source_shot_index"],
                                 "reason": "block_below_min_seconds", "reasons": ["block_below_min_seconds"]} for s in selected)
                continue
            for shot, original in zip(scene["shots"], selected, strict=True):
                shot.update(sid=sid, scene_id=row["scene_id"], score_key=original["score_key"])
                for name in ("mean_luma", "min_luma", "max_luma", "dover_technical", "laion_aesthetic", "scene", "characters"):
                    if name in original:
                        shot[name] = original[name]
            scene["quality"] = {f"min_{name}": min(s[name] for s in scene["shots"])
                                for name in ("dover_technical", "laion_aesthetic")}
            blocks.append({"key": f"{sid}-b{len(blocks):04d}", "case_id": case_id,
                           "family": row["family"], "source": row["source"], "scene": scene,
                           "proxy": None, "same_as_window": None})
    return blocks, {"runs": len(runs), "rejected": rejected}
