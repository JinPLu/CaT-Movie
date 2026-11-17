"""R6 场景族的清单、FFmpeg 顺序读帧及裁切来源；文字由后续成员编译。"""
from __future__ import annotations

import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from mspc.data.scene_training_manifest import MEDIA_COLUMNS, require_training_media, crop_geometry

SCENE_COLUMNS = (
    "mapping_schema", "source_fps", "source_num_frames", "source_width", "source_height",
    "source_shot_segments", "source_frame_indices_by_shot", "active_box", "train_crop_box",
    "film_key", "source", "scene_id", "container_preroll_frames",
)


def prepare_scene_row(row: dict[str, Any], *, video_path: str, shot_groups_column: str, media: Mapping) -> dict[str, Any]:
    from mspc.data.frame_planner import parse_positive_float, parse_positive_int
    from mspc.data.multishot import parse_shot_groups, parse_source_shot_segments, parse_source_frame_indices_by_shot, row_string

    media = require_training_media(media)
    # The task always declares the complete target media. Derived manifests may
    # repeat it in five scalar columns; frozen source manifests retain their
    # original height/width cells. Validate every declared cell, never invent a
    # default media contract or regenerate their frame grid/crop.
    columns = {**MEDIA_COLUMNS, "frozen_height": "height", "frozen_width": "width"}
    for key, column in columns.items():
        if column not in row:
            continue
        media_key = key.removeprefix("frozen_")
        value = parse_positive_float(row[column], column) if media_key == "target_fps" else parse_positive_int(row[column], column)
        if value != media[media_key]:
            raise ValueError(f"scene manifest spacetime cells ({column}) differ from configured task media")
    meta = {key: parse_positive_int(row[key], key) for key in
            ("source_num_frames", "source_width", "source_height")}
    meta["source_fps"] = parse_positive_float(row["source_fps"], "source_fps")
    boxes = {}
    for key in ("active_box", "train_crop_box"):
        box = json.loads(row_string(row, key, key))
        if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError(f"{key} must be four integer pixel edges")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= meta["source_width"] and 0 <= y0 < y1 <= meta["source_height"]):
            raise ValueError(f"{key} outside source image")
        boxes[key] = box
    ax0, ay0, ax1, ay1 = boxes["active_box"]
    x0, y0, x1, y1 = boxes["train_crop_box"]
    if not (ax0 <= x0 < x1 <= ax1 and ay0 <= y0 < y1 <= ay1):
        raise ValueError("train_crop_box outside active_box")
    if (x1 - x0) % 2 or (y1 - y0) % 2:
        raise ValueError("train_crop_box must have even width and height")
    # 生产端裁框按偶数像素取整，允许一个偶数像素量级的比例舍入。
    if abs((x1 - x0) - (y1 - y0) * media["width"] / media["height"]) > 2:
        raise ValueError("train_crop_box aspect ratio differs from configured task media")
    groups = parse_shot_groups(row[shot_groups_column], video_path)
    if sum(end - start for start, end in groups) > media["max_rgb_frames"]:
        raise ValueError("scene shot groups exceed configured task max_rgb_frames")
    row["_source_shot_segments"] = parse_source_shot_segments(row["source_shot_segments"], video_path)
    row["_source_frame_indices_by_shot"] = parse_source_frame_indices_by_shot(row["source_frame_indices_by_shot"], video_path)
    row["_shot_groups"] = groups
    row["_scene_metadata"] = meta
    row["_scene_boxes"] = boxes
    for key in ("source", "scene_id"):
        row_string(row, key, key)
    row["_prompt_source_path"] = None
    row["_prompts"] = ()
    return row


@lru_cache(maxsize=1)
def ffmpeg_runtime() -> tuple[str, str]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("scene_window_v1 requires ffmpeg on PATH")
    version = subprocess.run([executable, "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return executable, version


def read_scene_video(row: dict[str, Any], frame_indices: list[int], *, output_mode: str, video_decode_num_threads: int, media: Mapping):
    import numpy as np
    import torch
    from PIL import Image
    from mspc.data.frames import center_crop, iter_scene_frames

    media = require_training_media(media)
    height, width = media["height"], media["width"]
    if output_mode not in {"uint8", "normalized_float"}:
        raise ValueError(f"unsupported scene video output_mode={output_mode!r}")
    if not frame_indices or frame_indices != sorted(set(frame_indices)):
        raise ValueError("scene frame indices must be non-empty, ordered and unique")
    meta = row["_scene_metadata"]
    scene = {"source_video_path": row["video"], "video_frame_count": meta["source_num_frames"],
             "width": meta["source_width"], "height": meta["source_height"]}
    frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="mspc-scene-") as tmp:
        for offset, (index, frame) in enumerate(iter_scene_frames(
            scene, frame_indices, ffmpeg=ffmpeg_runtime()[0], error_path=Path(tmp) / "decode.stderr",
            video_decode_num_threads=video_decode_num_threads
        )):
            if index != frame_indices[offset]:
                raise ValueError("FFmpeg frame order differs from scene mapping")
            frames[offset] = np.asarray(center_crop(Image.fromarray(frame).crop(row["_scene_boxes"]["train_crop_box"]), width=width, height=height))
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
    return tensor if output_mode == "uint8" else tensor / 255 * 2 - 1


def describe_scene_media(row: dict[str, Any], frame_plan: dict[str, Any], *, media: Mapping) -> dict[str, Any]:
    media = require_training_media(media)
    meta = row["_scene_metadata"]
    path = Path(row["video"])
    stat = path.stat()
    geometry = crop_geometry({"width": meta["source_width"], "height": meta["source_height"],
                              "active_box": row["_scene_boxes"]["active_box"]},
                             crop_box=row["_scene_boxes"]["train_crop_box"], media=media)
    # CSV 空字符串经 pandas 成为 NaN；只有合同允许为空的两个追溯列转回 null。
    trace = {key: (None if row[key] != row[key] else row[key]) for key in
             ("source", "scene_id", "film_key", "container_preroll_frames")}
    return {"schema": row["mapping_schema"],
        "files": [{"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, **meta}],
        "source_shot_segments": row["_source_shot_segments"], "sampling": frame_plan,
        "scene": trace, "preprocessing": {"decoder": "ffmpeg-sequential",
            "decoder_version": ffmpeg_runtime()[1], "frame_basis": "ffmpeg_sequential_zero_based",
            "preroll_added": 0, "torchvision_version": importlib.metadata.version("torchvision"),
            "pillow_version": importlib.metadata.version("Pillow"), "resample": "PIL.Image.resize_RGB_default_BICUBIC",
            "spatial": [geometry], "pixel_conversion": "uint8_RGB_to_float_minus1_plus1"}}
