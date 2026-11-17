"""MSM global-caption frame sampling copied from MultiShotMaster.

Pinned script: KlingAIResearch/MultiShotMaster@1df812dda262639e4a3ca6e4a1da9000b0a8e124
``multi_shot_caption_annotation.py``. Decoder here is decord (repo standard)
instead of cv2.VideoCapture; resize is the same stretch to 672×384 JPEG.
Any size change is recorded. There is no silent skip or alternate size.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Sequence

from mspc.data.multishot import (
    parse_multisource_source_shot_segments,
    parse_source_frame_indices_by_shot,
    parse_source_video_paths,
)


MSM_FRAME_SPEC_VERSION = "mspc_msm_global_frame_spec_v1"
MSM_SOURCE_COMMIT = "1df812dda262639e4a3ca6e4a1da9000b0a8e124"
MAX_TOTAL_FRAMES = 20
SAMPLE_INTERVAL = 15
MIN_FRAMES_PER_SHOT = 5
MIN_ALLOCATED_PER_SHOT = 2
RESIZE_WIDTH = 672
RESIZE_HEIGHT = 384
JPEG_QUALITY = 95
DECODER = "decord.VideoReader"
RESAMPLE = "PIL.Image.Resampling.BILINEAR"
MSM_OFFICIAL_RESIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)


FrameReader = Callable[[Any, Sequence[int]], list[Any]]


def msm_global_frame_spec() -> dict[str, Any]:
    return {
        "spec_version": MSM_FRAME_SPEC_VERSION,
        "source_commit": MSM_SOURCE_COMMIT,
        "max_total_frames": MAX_TOTAL_FRAMES,
        "sample_interval": SAMPLE_INTERVAL,
        "min_frames_per_shot": MIN_FRAMES_PER_SHOT,
        "min_allocated_per_shot": MIN_ALLOCATED_PER_SHOT,
        "resize_width": RESIZE_WIDTH,
        "resize_height": RESIZE_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "decoder": DECODER,
        "resample": RESAMPLE,
        "aspect_ratio": "stretch_to_672x384",
        "note": "MSM official uses cv2.resize to 672x384; this path records the stretch.",
    }


def _shot_range(group: Sequence[int], *, index: int) -> tuple[int, int]:
    if not isinstance(group, Sequence) or isinstance(group, (str, bytes)) or len(group) != 2:
        raise ValueError(f"shot_groups[{index}] must be a [start, end) pair")
    start, end = group
    if type(start) is not int or type(end) is not int:
        raise ValueError(f"shot_groups[{index}] values must be ints")
    if start < 0 or end <= start:
        raise ValueError(f"shot_groups[{index}] must be a non-empty half-open range")
    return start, end


def parse_msm_shot_groups(value: Any) -> list[tuple[int, int]]:
    parsed = value
    if isinstance(value, str):
        import ast

        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("shot_groups must be a non-empty list")
    groups = [_shot_range(item, index=index) for index, item in enumerate(parsed)]
    seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(groups):
        if pair in seen:
            raise ValueError(f"shot_groups[{index}] is duplicated")
        seen.add(pair)
    return groups


def plan_shot_sample_indices(start: int, end: int) -> list[int]:
    """First-stage MSM per-shot sample: every 15 frames, pad to at least 5."""

    start, end = _shot_range((start, end), index=0)
    length = end - start
    if length < MIN_FRAMES_PER_SHOT:
        return list(range(start, end))
    indices = list(range(start, end, SAMPLE_INTERVAL))
    if len(indices) < MIN_FRAMES_PER_SHOT:
        sampled = set(indices)
        remaining = [position for position in range(start, end) if position not in sampled]
        needed = MIN_FRAMES_PER_SHOT - len(indices)
        if remaining:
            step = len(remaining) // needed
            if step < 1:
                step = 1
            for offset in range(0, len(remaining), step):
                if len(indices) >= MIN_FRAMES_PER_SHOT:
                    break
                indices.append(remaining[offset])
    return indices


def allocate_global_frame_counts(
    shot_lengths: Sequence[int],
    available_counts: Sequence[int],
    *,
    max_total: int = MAX_TOTAL_FRAMES,
) -> list[int]:
    if len(shot_lengths) != len(available_counts) or not shot_lengths:
        raise ValueError("shot_lengths and available_counts must be non-empty and aligned")
    if type(max_total) is not int or max_total < MIN_ALLOCATED_PER_SHOT:
        raise ValueError("max_total must be an int >= 2")
    total_length = sum(shot_lengths)
    if total_length <= 0:
        raise ValueError("shot_lengths must sum to a positive int")
    remaining = max_total
    counts: list[int] = []
    for length, available in zip(shot_lengths, available_counts):
        if type(length) is not int or type(available) is not int or length <= 0 or available <= 0:
            raise ValueError("each shot must have positive length and at least one sampled frame")
        allocated = max(
            MIN_ALLOCATED_PER_SHOT,
            min(available, round(length / total_length * max_total)),
        )
        remaining -= allocated
        counts.append(allocated)
    guard = 0
    while remaining < 0:
        max_idx = counts.index(max(counts))
        if counts[max_idx] <= MIN_ALLOCATED_PER_SHOT:
            raise ValueError("MSM global allocation cannot reduce any shot below 2 frames")
        counts[max_idx] -= 1
        remaining += 1
        guard += 1
        if guard > max_total * len(counts):
            raise ValueError("MSM global allocation failed to resolve a negative remainder")
    return counts


def subsample_indices(indices: Sequence[int], count: int) -> list[int]:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive int")
    if not indices:
        raise ValueError("indices must be non-empty")
    if len(indices) <= count:
        return list(indices)
    step = len(indices) / count
    picked = [int(index * step) for index in range(count)]
    return [indices[position] for position in picked]


def plan_msm_global_indices(shot_groups: Any) -> dict[str, Any]:
    groups = parse_msm_shot_groups(shot_groups)
    per_shot = [plan_shot_sample_indices(start, end) for start, end in groups]
    lengths = [end - start for start, end in groups]
    counts = allocate_global_frame_counts(lengths, [len(item) for item in per_shot])
    selected: list[int] = []
    selected_by_shot: list[list[int]] = []
    for indices, count in zip(per_shot, counts):
        chosen = subsample_indices(indices, count)
        selected_by_shot.append(chosen)
        selected.extend(chosen)
    if not selected or len(selected) > MAX_TOTAL_FRAMES:
        raise ValueError(
            f"MSM global sample must contain 1..{MAX_TOTAL_FRAMES} frames, got {len(selected)}"
        )
    return {
        "spec": msm_global_frame_spec(),
        "shot_groups": [list(pair) for pair in groups],
        "per_shot_sample_indices": per_shot,
        "allocated_counts": counts,
        "global_frame_indices": selected,
        "global_frame_indices_by_shot": selected_by_shot,
        "n_frames": len(selected),
    }


def _as_rgb(frame: Any):
    from PIL import Image
    import numpy as np

    if hasattr(frame, "convert"):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("MSM frame must be an RGB image")
    return Image.fromarray(array.astype("uint8"), mode="RGB")


def encode_msm_global_jpeg(frame: Any) -> tuple[bytes, dict[str, Any]]:
    from PIL import Image

    image = _as_rgb(frame)
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("MSM frame has non-positive size")
    resized = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    payload = buffer.getvalue()
    if not payload:
        raise ValueError("MSM JPEG encoding produced an empty payload")
    meta = {
        "source_width": source_width,
        "source_height": source_height,
        "output_width": RESIZE_WIDTH,
        "output_height": RESIZE_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "resample": RESAMPLE,
        "stretched": (source_width, source_height) != (RESIZE_WIDTH, RESIZE_HEIGHT),
    }
    return payload, meta


def annotation_data_url(jpeg_bytes: bytes) -> str:
    if not jpeg_bytes:
        raise ValueError("JPEG payload must be non-empty")
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def default_msm_frame_reader(path: Any, indices: Sequence[int]) -> list[Any]:
    from mspc.data.frames import default_frame_reader

    return default_frame_reader(path, indices)


def _absent_media_field(value: Any) -> bool:
    return value in (None, "", [], ())


def map_concat_indices_to_sources(
    concat_indices: Sequence[int],
    *,
    shot_groups: Sequence[tuple[int, int]],
    source_video_paths: Sequence[str],
    source_shot_segments: Sequence[Sequence[int]],
    source_frame_indices_by_shot: Sequence[Sequence[int]] | None = None,
) -> list[tuple[str, int]]:
    """Map concatenated-timeline frame indices onto per-clip files.

    CI-VID windows are distinct mp4s; ``video`` is the first clip, not a concat.
    Encoded ``shot_groups`` are often shorter than native ``source_shot_segments``
    because of fps resampling. In that case map through
    ``source_frame_indices_by_shot`` instead of a linear offset.
    """
    if len(source_video_paths) != len(shot_groups):
        raise ValueError(
            "source_video_paths length must equal shot_groups "
            f"({len(source_video_paths)} != {len(shot_groups)})"
        )
    if len(source_shot_segments) != len(shot_groups):
        raise ValueError(
            "source_shot_segments length must equal shot_groups "
            f"({len(source_shot_segments)} != {len(shot_groups)})"
        )
    resampled = source_frame_indices_by_shot not in (None, [], ())
    if resampled and len(source_frame_indices_by_shot) != len(shot_groups):
        raise ValueError(
            "source_frame_indices_by_shot length must equal shot_groups "
            f"({len(source_frame_indices_by_shot)} != {len(shot_groups)})"
        )
    native_ranges: list[tuple[int, int]] = []
    for shot_index, (start, end) in enumerate(shot_groups):
        source_start, source_end = _shot_range(source_shot_segments[shot_index], index=shot_index)
        native_ranges.append((source_start, source_end))
        concat_length = end - start
        source_length = source_end - source_start
        if resampled:
            indices = source_frame_indices_by_shot[shot_index]
            if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
                raise ValueError(f"source_frame_indices_by_shot[{shot_index}] must be a list")
            if len(indices) != concat_length:
                raise ValueError(
                    f"source_frame_indices_by_shot[{shot_index}] length {len(indices)} != "
                    f"shot_groups[{shot_index}] length {concat_length}"
                )
            continue
        if concat_length != source_length:
            raise ValueError(
                f"shot_groups[{shot_index}] length {concat_length} != "
                f"source_shot_segments[{shot_index}] length {source_length}; "
                "CI-VID resampling requires source_frame_indices_by_shot; "
                "refusing to invent a concatenated video"
            )
    mapped: list[tuple[str, int]] = []
    for concat_index in concat_indices:
        if type(concat_index) is not int:
            raise ValueError("concat frame indices must be ints")
        hit: tuple[str, int] | None = None
        for shot_index, (start, end) in enumerate(shot_groups):
            if not (start <= concat_index < end):
                continue
            source_start, source_end = native_ranges[shot_index]
            local_encoded = concat_index - start
            if resampled:
                local = source_frame_indices_by_shot[shot_index][local_encoded]
                if type(local) is not int:
                    raise ValueError(
                        f"source_frame_indices_by_shot[{shot_index}][{local_encoded}] must be an int"
                    )
            else:
                local = source_start + local_encoded
            if not (source_start <= local < source_end):
                raise ValueError(
                    f"concat index {concat_index} mapped to native {local} outside "
                    f"source_shot_segments[{shot_index}] {[source_start, source_end]}"
                )
            hit = (source_video_paths[shot_index], local)
            break
        if hit is None:
            raise ValueError(f"concat index {concat_index} is not inside shot_groups")
        mapped.append(hit)
    return mapped


def decode_mapped_frames(
    mapped: Sequence[tuple[str, int]],
    *,
    frame_reader: FrameReader,
) -> list[Any]:
    from collections import defaultdict
    from pathlib import Path

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for order, (path, local_index) in enumerate(mapped):
        grouped[path].append((order, local_index))
    frames: list[Any | None] = [None] * len(mapped)
    for path, items in grouped.items():
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"MSM global vision source video is missing: {resolved}")
        locals_indices = [local_index for _, local_index in items]
        decoded = frame_reader(resolved, locals_indices)
        if len(decoded) != len(items):
            raise ValueError(
                f"{resolved}: decoded {len(decoded)} frames, expected {len(items)}"
            )
        for (order, _), frame in zip(items, decoded):
            frames[order] = frame
    if any(frame is None for frame in frames):
        raise ValueError("MSM multisource decode left an empty frame slot")
    return [frame for frame in frames if frame is not None]


def _extract_concat_indices(
    concat_indices: Sequence[int],
    *,
    video_path: Any,
    shot_groups: Sequence[tuple[int, int]],
    frame_reader: FrameReader,
    source_video_paths: Any = None,
    source_shot_segments: Any = None,
    source_frame_indices_by_shot: Any = None,
) -> dict[str, Any]:
    from pathlib import Path

    if not concat_indices:
        raise ValueError("MSM extract requires a non-empty concat index list")
    for index, value in enumerate(concat_indices):
        if type(value) is not int:
            raise ValueError(f"concat_indices[{index}] must be an int")
    groups = [tuple(pair) for pair in shot_groups]
    label = Path(video_path).expanduser() if video_path not in (None, "") else None
    mapped: list[tuple[str, int]] | None = None
    if not _absent_media_field(source_video_paths) or not _absent_media_field(
        source_shot_segments
    ):
        if _absent_media_field(source_video_paths) or _absent_media_field(
            source_shot_segments
        ):
            raise ValueError(
                "CI-VID multisource extract requires both source_video_paths and "
                "source_shot_segments; refusing to read only the first clip"
            )
        data_path = str(label) if label is not None else "multisource"
        paths = parse_source_video_paths(source_video_paths, data_path)
        segments = parse_multisource_source_shot_segments(source_shot_segments, data_path)
        indices_by_shot = None
        if not _absent_media_field(source_frame_indices_by_shot):
            indices_by_shot = parse_source_frame_indices_by_shot(
                source_frame_indices_by_shot, data_path
            )
        mapped = map_concat_indices_to_sources(
            concat_indices,
            shot_groups=groups,
            source_video_paths=paths,
            source_shot_segments=segments,
            source_frame_indices_by_shot=indices_by_shot,
        )
        frames = decode_mapped_frames(mapped, frame_reader=frame_reader)
        media_layout = "multisource"
        recorded_video = str(label) if label is not None else paths[0]
        source_paths_out = paths
        source_segments_out = segments
        indices_by_shot_out = indices_by_shot
    else:
        if label is None:
            raise ValueError("video_path is required for single-file MSM extract")
        if not label.is_file():
            raise FileNotFoundError(f"MSM global vision video is missing: {label}")
        frames = frame_reader(label, concat_indices)
        media_layout = "single_file"
        recorded_video = str(label)
        source_paths_out = None
        source_segments_out = None
        indices_by_shot_out = None
    if len(frames) != len(concat_indices):
        raise ValueError(
            f"{recorded_video}: decoded {len(frames)} frames, expected {len(concat_indices)}"
        )
    data_urls: list[str] = []
    resize_log: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        jpeg, meta = encode_msm_global_jpeg(frame)
        data_urls.append(annotation_data_url(jpeg))
        resize_log.append({"frame_index": concat_indices[index], **meta})
    payload = {
        "video_path": recorded_video,
        "media_layout": media_layout,
        "image_data_urls": data_urls,
        "resize_log": resize_log,
        "n_frames": len(concat_indices),
        "concat_frame_indices": list(concat_indices),
    }
    if source_paths_out is not None:
        payload["source_video_paths"] = source_paths_out
        payload["source_shot_segments"] = source_segments_out
        payload["source_frame_indices"] = [local for _, local in mapped or ()]
        if indices_by_shot_out is not None:
            payload["source_frame_indices_by_shot"] = indices_by_shot_out
    return payload


def extract_msm_shot_frames(
    *,
    video_path: Any,
    shot_groups: Any,
    shot_index: int,
    frame_reader: FrameReader = default_msm_frame_reader,
    source_video_paths: Any = None,
    source_shot_segments: Any = None,
    source_frame_indices_by_shot: Any = None,
) -> dict[str, Any]:
    groups = parse_msm_shot_groups(shot_groups)
    if type(shot_index) is not int or not 0 <= shot_index < len(groups):
        raise ValueError(f"shot_index must be in 0..{len(groups) - 1}")
    start, end = groups[shot_index]
    indices = plan_shot_sample_indices(start, end)
    if not indices:
        raise ValueError(f"shot_groups[{shot_index}] produced no sample indices")
    extracted = _extract_concat_indices(
        indices,
        video_path=video_path,
        shot_groups=groups,
        frame_reader=frame_reader,
        source_video_paths=source_video_paths,
        source_shot_segments=source_shot_segments,
        source_frame_indices_by_shot=source_frame_indices_by_shot,
    )
    return {
        **extracted,
        "shot_index": shot_index,
        "shot_groups": [list(pair) for pair in groups],
        "shot_frame_indices": list(indices),
        "spec": msm_global_frame_spec(),
    }


def extract_msm_global_frames(
    *,
    video_path: Any,
    shot_groups: Any,
    frame_reader: FrameReader = default_msm_frame_reader,
    source_video_paths: Any = None,
    source_shot_segments: Any = None,
    source_frame_indices_by_shot: Any = None,
) -> dict[str, Any]:
    plan = plan_msm_global_indices(shot_groups)
    groups = [tuple(pair) for pair in plan["shot_groups"]]
    extracted = _extract_concat_indices(
        plan["global_frame_indices"],
        video_path=video_path,
        shot_groups=groups,
        frame_reader=frame_reader,
        source_video_paths=source_video_paths,
        source_shot_segments=source_shot_segments,
        source_frame_indices_by_shot=source_frame_indices_by_shot,
    )
    return {
        **plan,
        **extracted,
        "global_frame_indices": plan["global_frame_indices"],
    }
