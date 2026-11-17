"""场景窗训练清单的生产端；帧号、中心裁均复用现有唯一规则。"""
from __future__ import annotations

from itertools import accumulate
from math import gcd
from collections.abc import Mapping

from mspc.benchmark.long_story_holdout import crop_box
from mspc.data.frame_planner import (
    FpsAwareTargetFpsPlanner, parse_positive_float, parse_positive_int,
    validate_source_shot_segments,
)


# 旧标注代理片的明确合同；旧标注／审阅调用者须显式传入，不是生产参数缺失时的回退。
ANNOTATION_MEDIA = {"height": 480, "width": 832, "target_fps": 16,
                    "vae_temporal_stride": 4, "max_rgb_frames": 308}
MEDIA_FIELDS = ("height", "width", "target_fps", "vae_temporal_stride", "max_rgb_frames")
MEDIA_COLUMNS = {"height": "target_height", "width": "target_width", "target_fps": "target_fps",
                 "vae_temporal_stride": "vae_temporal_stride", "max_rgb_frames": "max_rgb_frames"}


def require_training_media(media: Mapping) -> dict:
    """显式的目标像素和采样合同，来自现有task；不存在默认分辨率或帧率。"""
    if not isinstance(media, Mapping) or set(media) != set(MEDIA_FIELDS):
        raise ValueError(f"training media requires exactly {MEDIA_FIELDS}")
    result = dict(media)
    for key in ("height", "width", "vae_temporal_stride", "max_rgb_frames"):
        result[key] = parse_positive_int(result[key], key)
    result["target_fps"] = parse_positive_float(result["target_fps"], "target_fps")
    if result["vae_temporal_stride"] != 4:
        raise ValueError("scene training media requires the independent-shot temporal stride 4")
    return result


def media_from_task(task: Mapping) -> dict:
    return require_training_media({key: task[key] for key in MEDIA_FIELDS})


def crop_inside(box: list[int], *, width: int, height: int) -> list[int]:
    """本轮已核验的偶数中心内裁公式；benchmark与生产复用同一owner。"""
    x0, y0, x1, y1 = box
    divisor = gcd(width, height)
    a, b = width // divisor, height // divisor
    unit = min((x1 - x0) // a, (y1 - y0) // b)
    unit -= unit % 2
    if unit <= 0:
        raise ValueError("source crop cannot contain a positive even target-aspect rectangle")
    w, h = a * unit, b * unit
    left, top = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
    return [left, top, left + w, top + h]


def training_crop_box(scene: Mapping, *, media: Mapping) -> list[int]:
    media = require_training_media(media)
    annotation_crop = list(scene["crop_box"])
    if len(annotation_crop) != 4 or any(type(v) is not int for v in annotation_crop):
        raise ValueError("annotation crop must contain four integer pixel edges")
    # 同标注比例保留原crop，包含原有偶数取整；仅提高像素密度不会重裁原480画面。
    if media["width"] * ANNOTATION_MEDIA["height"] == media["height"] * ANNOTATION_MEDIA["width"]:
        return annotation_crop
    return crop_inside(annotation_crop, width=media["width"], height=media["height"])


def crop_geometry(scene: Mapping, *, crop_box: list[int], media: Mapping) -> dict:
    """显式源crop的PIL取整／中心裁缩仿射，输出size按width,height。"""
    media = require_training_media(media)
    x0, y0, x1, y1 = crop_box
    width, height = x1 - x0, y1 - y0
    tw, th = media["width"], media["height"]
    scale = max(tw / width, th / height)
    rw, rh = max(tw, round(width * scale)), max(th, round(height * scale))
    left, top = (rw - tw) // 2, (rh - th) // 2
    sx, sy = rw / width, rh / height
    return {"source_size": [scene["width"], scene["height"]],
            "active_box": list(scene["active_box"]), "train_crop_box": [x0, y0, x1, y1],
            "resize_size": [rw, rh], "crop_box_resized": [left, top, left + tw, top + th],
            "output_size": [tw, th], "coordinate_system": "pixel_edges",
            "source_to_target_matrix": [[sx, 0, -sx * x0 - left],
                                        [0, sy, -sy * y0 - top], [0, 0, 1]]}


def training_geometry(scene: Mapping, *, media: Mapping) -> dict:
    """源像素边缘→本训练画幅，matrix3×3，output_size=[width,height]。"""
    return crop_geometry(scene, crop_box=training_crop_box(scene, media=media), media=media)


def media_columns(media: Mapping) -> dict:
    return {column: require_training_media(media)[key] for key, column in MEDIA_COLUMNS.items()}


def manifest_fields(window: dict, *, media: Mapping) -> dict:
    """已过镜头关的真实窗 → 已定字段。schema 名称由模型线显式提供。"""
    media = require_training_media(media)
    revision = window["boundary_revision"]
    if revision["coordinate"] != "ffmpeg_sequential" or revision["boundary_guard_frames"] != 2:
        raise ValueError("expected FFmpeg sequential coordinates and N=2 guard")
    segments = [[s["start_frame"], s["end_frame"]] for s in window["shots"]]
    validate_source_shot_segments(segments, window["window_id"])
    if segments != [s["revised_segment"] for s in revision["shots"]]:
        raise ValueError("window shots differ from final guarded gate segments")
    fps = parse_positive_float(window["fps"], "source_fps")
    count = parse_positive_int(window["video_frame_count"], "source_num_frames")
    width = parse_positive_int(window["width"], "source_width")
    height = parse_positive_int(window["height"], "source_height")
    if any(end > count for _, end in segments):
        raise ValueError("source segment exceeds source_num_frames")
    planner = FpsAwareTargetFpsPlanner(max_rgb_frames=media["max_rgb_frames"], target_fps=media["target_fps"],
                                        vae_temporal_stride=media["vae_temporal_stride"])
    indices = [planner.clean_indices_for_shot(start, end, fps)[0] for start, end in segments]
    ends = [0, *accumulate(map(len, indices))]
    return {
        "case_id": window["window_id"], "video": window["source_video_path"],
        "source_fps": fps, "source_num_frames": count,
        "source_width": width, "source_height": height,
        "source_shot_segments": segments, "source_frame_indices_by_shot": indices,
        "shot_groups": [[a, b] for a, b in zip(ends, ends[1:])],
        "active_box": window["active_box"], "train_crop_box": training_crop_box({**window, "crop_box": crop_box(window)}, media=media),
        **media_columns(media),
        "film_key": window["film_key"], "source": window["source"],
        "scene_id": window["scene_id"], "container_preroll_frames": window["container_preroll_frames"],
    }


def manifest_row(window: dict, *, mapping_schema: str, media: Mapping) -> dict:
    if not isinstance(mapping_schema, str) or not mapping_schema.strip():
        raise ValueError("mapping_schema must be explicitly named by the model-line contract")
    return {**manifest_fields(window, media=media), "mapping_schema": mapping_schema}


def frozen_scene_frame_plan(row: dict, *, target_fps: float, max_rgb_frames: int,
                            vae_temporal_stride: int) -> dict:
    """显式源帧清单是事实；只核几何/边界与长度，绝不重新抽帧。"""
    from mspc.data.frame_planner import (
        FRAME_POLICY_FPS_AWARE_TARGET_FPS, VAE_ALIGNMENT_POLICY_CLEAN,
        align_rgb_length_to_vae, rgb_to_latent_shot_groups, round_frame_count,
    )

    segments = row["_source_shot_segments"]
    groups = row["_shot_groups"]
    indices = row["_source_frame_indices_by_shot"]
    meta = row["_scene_metadata"]
    validate_source_shot_segments(segments, row["video"])
    if not len(segments) == len(groups) == len(indices):
        raise ValueError("frozen scene source/group counts differ")
    lengths, durations, targets = [], [], []
    for (start, end), (a, b), selected in zip(segments, groups, indices, strict=True):
        if not 0 <= start < end <= meta["source_num_frames"]:
            raise ValueError("frozen scene segment outside native source")
        length = b - a
        if (length < 5 or (length - 1) % vae_temporal_stride
                or len(selected) != length or selected != sorted(set(selected))
                or selected[0] != start or selected[-1] != end - 1
                or any(not start <= frame < end for frame in selected)):
            raise ValueError("frozen scene indices differ from the clean distinct source-grid contract")
        duration = (end - start) / meta["source_fps"]
        target = min(end - start, round_frame_count(duration * target_fps))
        if align_rgb_length_to_vae(target, vae_temporal_stride) != length:
            raise ValueError("frozen scene length differs from declared fps/VAE alignment")
        lengths.append(length); durations.append(duration); targets.append(target)
    if sum(lengths) > max_rgb_frames:
        raise ValueError("frozen scene frame count exceeds explicit max_rgb_frames")
    latent_groups = rgb_to_latent_shot_groups(groups, vae_temporal_stride)
    total_duration = sum(durations)
    return {
        "frame_policy": FRAME_POLICY_FPS_AWARE_TARGET_FPS,
        "mapping_kind": "scene_ffmpeg_sequential_explicit_v1",
        "raw_shot_groups": [list(group) for group in segments],
        "source_shot_segments": [list(group) for group in segments],
        "source_video_paths": [row["video"]] * len(groups),
        "source_fps_by_shot": [meta["source_fps"]] * len(groups),
        "source_num_frames_by_shot": [meta["source_num_frames"]] * len(groups),
        "source_fps": meta["source_fps"], "source_num_frames": meta["source_num_frames"],
        "target_fps": target_fps, "max_rgb_frames": max_rgb_frames,
        "vae_temporal_stride": vae_temporal_stride, "shot_durations_sec": durations,
        "duration_weights": [duration / total_duration for duration in durations],
        "target_rgb_lengths": targets, "aligned_rgb_lengths": lengths,
        "vae_padding_frames": [0] * len(groups),
        "vae_truncated_frames": [target - length for target, length in zip(targets, lengths)],
        "vae_alignment_policy": VAE_ALIGNMENT_POLICY_CLEAN,
        "latent_lengths": [b - a for a, b in latent_groups],
        "frame_indices_by_shot": [list(selected) for selected in indices],
        "frame_indices": [frame for selected in indices for frame in selected],
        "train_shot_groups": [list(group) for group in groups],
        "shot_groups": [list(group) for group in groups],
        "latent_shot_groups": latent_groups, "prompt_index_by_shot": list(range(len(groups))),
        "source_rgb_frames": sum(end - start for start, end in segments),
        "train_rgb_frames": sum(lengths), "train_latent_frames": latent_groups[-1][-1],
        "source_duration_sec": total_duration, "target_duration_sec": sum(targets) / target_fps,
        "effective_train_fps": sum(lengths) / total_duration, "rgb_was_scaled": False,
    }


def source_slice_row(metadata: dict, *, annotation_window_id: str) -> dict:
    if not isinstance(annotation_window_id, str) or not annotation_window_id:
        raise ValueError("annotation_window_id is required")
    return {"metadata": metadata, "window_id": annotation_window_id,
            "case_id": metadata["case_id"],
            "frame_counts": [end-start for start, end in metadata["shot_groups"]]}


def audit_row(row: dict) -> dict:
    """逐项记录，不修帧、不修框、不丢行；比例与坐标偶数的歧义分列。"""
    indices, segments = row["source_frame_indices_by_shot"], row["source_shot_segments"]
    counts = list(map(len, indices))
    x0, y0, x1, y1 = row["train_crop_box"]
    a, b, c, d = row["active_box"]
    checks = {
        "mapping_schema_named": isinstance(row.get("mapping_schema"), str) and bool(row["mapping_schema"]),
        "shot_4n_plus_1": all(n >= 5 and n % 4 == 1 for n in counts),
        "window_le_308": sum(counts) <= 308,
        "shot_le_228": all(n <= 228 for n in counts),
        "endpoints_and_unique": all(p[0] == start and p[-1] == end-1 and len(set(p)) == len(p)
                                    and all(start <= f < end for f in p)
                                    for p, (start, end) in zip(indices, segments, strict=True)),
        "source_bounds": all(0 <= start < end <= row["source_num_frames"] for start, end in segments),
        "crop_inside_active": 0 <= a <= x0 < x1 <= c <= row["source_width"]
                              and 0 <= b <= y0 < y1 <= d <= row["source_height"],
        "crop_exact_832_480": (x1-x0)*480 == (y1-y0)*832,
        "crop_even_size": (x1-x0) % 2 == 0 and (y1-y0) % 2 == 0,
        "crop_even_coordinates": all(v % 2 == 0 for v in (x0, y0, x1, y1)),
    }
    return {"case_id": row["case_id"], "checks": checks, "failures": [k for k,v in checks.items() if not v],
            "frame_counts": counts, "crop_ratio_relative_error": (x1-x0)/(y1-y0)/(832/480)-1}


def decoder_scene(row: dict) -> dict:
    """只转字段名；解码仍由 iter_scene_frames 实现。"""
    return {"source_video_path": row["video"], "width": row["source_width"],
            "height": row["source_height"], "video_frame_count": row["source_num_frames"]}
