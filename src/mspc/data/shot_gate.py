"""整源片逐帧打分与查表镜头关；窗族拥有帧号，镜头关不换算。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from mspc.data.shot_boundary_revision import partition_boundary_cuts
from mspc.evaluation.renderer_qualification_public_metrics import transition_run_peaks, cut_boundaries_from_peaks

RULE_VERSION = "shot_gate_v2_full_guarded_single"

COORDINATES = {"scene_window_v1": "ffmpeg_sequential", "ci_vid": "decord", "cine250k": "decord"}


def read_windows(path: Path, *, family: str) -> list[dict]:
    """原生场景 JSONL 或池 CSV → 明确窗族与半开区间；不加预卷。"""
    if family not in ("scene_window_v1", "pool205k"):
        raise ValueError("input family must be scene_window_v1 or pool205k")
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()] if family == "scene_window_v1" else list(csv.DictReader(handle))
    result = []
    for row in rows:
        if family == "scene_window_v1":
            if row["schema_version"] != family:
                raise ValueError("expected scene_window_v1")
            shots = [{"video": row["source_video_path"], "segment": [s["start_frame"], s["end_frame"]]} for s in row["shots"]]
            wid = row["window_id"]
        else:
            if row["dataset"] not in ("ci_vid", "cine250k"):
                raise ValueError(f"unsupported pool dataset {row['dataset']}")
            shots = [{"video": p, "segment": s} for p, s in zip(json.loads(row["source_video_paths"]), json.loads(row["source_shot_segments"]), strict=True)]
            wid = row["case_id"]
        window = {"window_id": wid, "family": family if family == "scene_window_v1" else row["dataset"], "shots": shots}
        validate_window(window)
        result.append(window)
    return result


def validate_window(window):
    COORDINATES[window["family"]]
    if not window["shots"]:
        raise ValueError("window has no shots")
    for shot in window["shots"]:
        start, end = shot["segment"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"invalid segment {shot['segment']}")
        if not isinstance(shot["video"], str) or not shot["video"]:
            raise ValueError("video path required")


def load_scores(path):
    with np.load(path, allow_pickle=False) as data:
        return {**json.loads(str(data["metadata"])), "single": data["single"], "content": data["content"]}


def gate_shot(shot, scores, *, coordinate, boundary_threshold, single_threshold, content_threshold, edge_seconds, boundary_guard_frames):
    if type(boundary_guard_frames) is not int or boundary_guard_frames < 0:
        raise ValueError("boundary_guard_frames must be a nonnegative integer")
    for value in (boundary_threshold, single_threshold, content_threshold, edge_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("thresholds and edge_seconds must be positive finite numbers")
    if scores["coordinate"] != coordinate or scores["video"] != shot["video"]:
        raise ValueError("score coordinate/video differs from window family/source")
    start, end = shot["segment"]
    count = scores["frame_count"]
    if not 0 <= start < end <= count or len(scores["single"]) != count or len(scores["content"]) != count:
        raise ValueError("segment or score length outside source frame count")
    # 原聚峰函数用 >；nextafter 实现用户指定的 >=，其余峰值与 p+1 规则不变。
    peaks = transition_run_peaks(scores["single"][start:end], threshold=math.nextafter(boundary_threshold, -math.inf))
    cuts = [start + p for p in cut_boundaries_from_peaks(peaks, frame_count=end - start)] if end - start > 1 else []
    revised, _ = partition_boundary_cuts(start=start, end=end, fps=scores["fps"], cuts=cuts, edge_seconds=edge_seconds)
    guarded = [revised[0] + boundary_guard_frames, revised[1] - boundary_guard_frames]
    edge = edge_seconds * scores["fps"]
    reasons = []
    for signal, threshold in (("single", single_threshold), ("content", content_threshold)):
        for frame in range(start, end):
            interior = start + edge < frame < end - edge
            full_guarded_single = signal == "single" and guarded[0] <= frame < guarded[1]
            if (interior or full_guarded_single) and scores[signal][frame] >= threshold:
                reasons.append({"signal": signal, "frame": frame, "score": float(scores[signal][frame])})
    if guarded[1] - guarded[0] < 0.5 * scores["fps"]:
        reasons.append({"signal": "duration", "reason": "guarded_shot_shorter_than_0.5_seconds",
                        "frames": guarded[1] - guarded[0], "fps": scores["fps"], "minimum_seconds": 0.5})
    return {**shot, "rule_version": RULE_VERSION, "coordinate": coordinate, "boundary_revised_segment": revised,
            "boundary_guard_frames": boundary_guard_frames, "revised_segment": guarded, "boundary_cuts": cuts,
            "status": "rejected" if reasons else "kept", "reasons": reasons}


def gate_window(window, tables, **thresholds):
    validate_window(window)
    coordinate = COORDINATES[window["family"]]
    shots = [gate_shot(shot, tables[shot["video"]], coordinate=coordinate, **thresholds) for shot in window["shots"]]
    return {**window, "rule_version": RULE_VERSION, "boundary_guard_frames": thresholds["boundary_guard_frames"], "coordinate": coordinate, "status": "rejected" if any(s["reasons"] for s in shots) else "kept", "shots": shots,
            "reasons": [{"shot": i, **r} for i, s in enumerate(shots, 1) for r in s["reasons"]], "thresholds": thresholds}


def revised_scene_window(window: dict, gate: dict) -> dict:
    """把已保留的镜头关结果写回场景窗；最终区间已包含保护帧，不再裁一次。"""
    from mspc.benchmark.long_story_holdout import render_frame_count

    if window["schema_version"] != "scene_window_v1" or gate["family"] != "scene_window_v1":
        raise ValueError("scene_window_v1 required")
    if gate["window_id"] != window["window_id"] or gate["status"] != "kept":
        raise ValueError("matching kept gate required")
    if gate["coordinate"] != "ffmpeg_sequential":
        raise ValueError("FFmpeg sequential coordinates required")
    for key in ("width", "height", "video_frame_count"):
        window[key]
    shots = []
    for original, result in zip(window["shots"], gate["shots"], strict=True):
        if result["video"] != window["source_video_path"] or result["segment"] != [original["start_frame"], original["end_frame"]]:
            raise ValueError("gate must start from the original source interval")
        start, end = result["revised_segment"]
        if not 0 <= start < end <= window["video_frame_count"]:
            raise ValueError("revised interval outside source")
        seconds = (end - start) / window["fps"]
        shots.append({**original, "start_frame": start, "end_frame": end,
                      "seconds": seconds, "render_frames": render_frame_count(seconds)})
    return {**window, "shots": shots, "render_frames": sum(s["render_frames"] for s in shots),
            "boundary_revision": {"version": gate["rule_version"], "coordinate": gate["coordinate"],
                                  "boundary_guard_frames": gate["boundary_guard_frames"],
                                  "thresholds": gate["thresholds"], "shots": gate["shots"]}}
