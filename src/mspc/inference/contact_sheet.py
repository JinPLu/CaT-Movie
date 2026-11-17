"""Full-temporal contact sheets for generated-video human review."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence


def shot_group_sample_indices(shot_groups: Sequence[Sequence[int]]) -> list[int]:
    """Three endpoint frames per shot span -- ``{start, mid, end-1}`` -- unioned and sorted.

    Lifted out of ``mspc.training.wan22_smoke_gate`` (was ``_contact_sheet_frame_indices``,
    private) so a second caller (``tools/run/build_render_comparison_sheets.py``, which also
    needs the *per-shot* three-frame sample used inside one shot span) does not duplicate this
    formula. Passing a single-shot ``[[start, end]]`` list returns that one shot's own three
    indices, which is how the comparison-sheet tool reuses it per column instead of only for
    the whole-video union this module's original caller wanted.
    """

    indices: set[int] = set()
    for start, end in shot_groups:
        indices.update((start, start + (end - start - 1) // 2, end - 1))
    return sorted(indices)


def full_temporal_frame_indices(
    decoded_frames: int,
    *,
    native_fps: float,
    sample_fps: float = 2.0,
) -> list[int]:
    if type(decoded_frames) is not int or decoded_frames <= 0:
        raise ValueError("decoded_frames must be a positive integer")
    if not isinstance(native_fps, (int, float)) or native_fps <= 0:
        raise ValueError("native_fps must be positive")
    if not isinstance(sample_fps, (int, float)) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    duration = decoded_frames / float(native_fps)
    count = int(math.floor(duration * float(sample_fps))) + 1
    indices = [
        min(decoded_frames - 1, int(round(index * native_fps / sample_fps)))
        for index in range(count)
    ]
    if indices[-1] != decoded_frames - 1:
        indices.append(decoded_frames - 1)
    unique = sorted(set(indices))
    if unique[0] != 0 or unique[-1] != decoded_frames - 1:
        raise RuntimeError("full-temporal sampling must include the first and last frames")
    return unique


def write_labeled_contact_sheet(
    frames: dict[int, Any],
    output_path: Path,
    *,
    columns: int = 8,
) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("cannot render a contact sheet without decoded frames")
    if type(columns) is not int or columns <= 0:
        raise ValueError("columns must be a positive integer")
    images: list[tuple[int, Any]] = []
    for frame_index, frame in sorted(frames.items()):
        image = Image.fromarray(frame).convert("RGB")
        target_width = min(image.width, 240)
        target_height = max(1, round(image.height * target_width / image.width))
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        images.append((frame_index, image))
    cell_width = max(image.width for _, image in images)
    cell_height = max(image.height for _, image in images)
    label_height = 22
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for position, (frame_index, image) in enumerate(images):
        column = position % columns
        row = position // columns
        left = column * cell_width
        top = row * (cell_height + label_height)
        draw.text((left + 4, top + 3), f"f{frame_index}", fill="black")
        sheet.paste(image, (left, top + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_full_temporal_contact_sheet(
    video_path: Path,
    output_path: Path,
    *,
    sample_fps: float = 2.0,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
) -> dict[str, Any]:
    import imageio
    import numpy as np

    reader: Any | None = None
    sampled: dict[int, Any] = {}
    decoded_frames = 0
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        metadata = reader.get_meta_data()
        observed_fps = float(metadata["fps"])
        if expected_fps is not None and abs(observed_fps - expected_fps) > 0.05:
            raise ValueError(
                f"generated video fps mismatch: expected {expected_fps}, got {observed_fps}"
            )
        for frame_index, frame in enumerate(reader):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    f"decoded frame {frame_index} must have RGB shape, got {array.shape}"
                )
            decoded_frames += 1
            sampled[frame_index] = array.copy()
        if expected_frames is not None and decoded_frames != expected_frames:
            raise ValueError(
                f"generated video frame count mismatch: expected {expected_frames}, got {decoded_frames}"
            )
        keep = set(
            full_temporal_frame_indices(
                decoded_frames,
                native_fps=observed_fps,
                sample_fps=sample_fps,
            )
        )
        sampled_frames = {index: sampled[index] for index in keep}
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"could not decode video for contact sheet: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
    write_labeled_contact_sheet(sampled_frames, output_path)
    return {
        "path": str(output_path),
        "decoded_frames": decoded_frames,
        "fps": observed_fps,
        "sample_fps": sample_fps,
        "sampled_frame_indices": sorted(sampled_frames),
    }
