"""Shot-level media access for the quality scorer backends.

Every backend scores a *shot*, never a window and never a concatenation. The
mapping kinds reach a shot differently and the difference is not cosmetic:

* ``ci_vid`` shots are ``[start, end)`` inside the shot's own file.
* ``cine250k`` shots are the inclusive ``[start_frame, end_frame]`` span of the
  window's single source film.
* ``scene_window_v1`` is half-open, in FFmpeg sequential coordinates; its
  SceneFrameReader is selected by the required input coordinate declaration.

Decoding is lazy and per-index: a Cine film is thousands of frames long and
only the sampled indices are ever read.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence


HALF_OPEN = "half_open"
INCLUSIVE = "inclusive"
# Fetch a contiguous span when it is at most this times longer than the unique
# set. DOVER's interval=2 recipe is just inside this bound, so Cine films do
# one sequential scan instead of one seek per sampled frame.
DENSE_SPAN_FACTOR = 2


class ShotDecodeError(RuntimeError):
    """A shot could not be decoded. Recorded as ``invalid``, never as a zero score."""


@dataclass(frozen=True)
class ShotLocator:
    case_id: str
    source: str
    mapping_kind: str
    shot_index: int
    source_path: str
    frame_start: int
    frame_end: int
    range_semantics: str

    @property
    def frame_count(self) -> int:
        if self.range_semantics == HALF_OPEN:
            return self.frame_end - self.frame_start
        if self.range_semantics == INCLUSIVE:
            return self.frame_end - self.frame_start + 1
        raise ValueError(f"unknown range semantics {self.range_semantics!r}")

    def absolute_index(self, local_index: int) -> int:
        count = self.frame_count
        if not 0 <= local_index < count:
            raise ValueError(
                f"local frame {local_index} is outside the shot's {count} frames"
            )
        return self.frame_start + local_index


def require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def require_positive_int_env(name: str, env: Any = None) -> int:
    raw = (os.environ if env is None else env).get(name)
    if raw is None or raw == "":
        raise ValueError(f"{name} is required")
    if type(raw) is not str or not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {raw!r}")
    return int(raw)


def fetch_absolute_indices(absolute: Sequence[int]) -> list[int]:
    """Unique frames actually handed to the decoder, in decode order.

    Repeats (loop-padding, two named views sharing a shot) are collapsed.
    A short dense span is filled so the decoder can scan forward once;
    a sparse span stays a unique sorted list.
    """

    if len(absolute) == 0:
        raise ValueError("no absolute frames to fetch")
    unique = sorted(set(int(index) for index in absolute))
    lo, hi = unique[0], unique[-1]
    span = hi - lo + 1
    if span <= len(unique) * DENSE_SPAN_FACTOR + 1:
        return list(range(lo, hi + 1))
    return unique


def decode_local_frames(
    reader: Any,
    locator: ShotLocator,
    local_indices: Sequence[int],
) -> Any:
    """Return an ``uint8`` array of shape ``[len(local_indices), H, W, 3]`` (RGB).

    Requested indices are returned in the caller's order, including repeats.
    The decoder is only asked for each unique (or dense-span) index once.
    """

    if len(local_indices) == 0:
        raise ShotDecodeError(f"{locator.case_id}#{locator.shot_index}: no frames requested")
    try:
        total = len(reader)
    except Exception as exc:
        raise ShotDecodeError(f"{locator.source_path}: frame count unavailable: {exc}") from exc
    absolute = []
    for index in local_indices:
        frame = locator.absolute_index(int(index))
        if frame >= total:
            raise ShotDecodeError(
                f"{locator.case_id}#{locator.shot_index}: frame {frame} is past the "
                f"{total}-frame file {locator.source_path}"
            )
        absolute.append(frame)
    fetch = fetch_absolute_indices(absolute)
    if fetch[-1] >= total:
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: frame {fetch[-1]} is past the "
            f"{total}-frame file {locator.source_path}"
        )
    try:
        batch = reader.get_batch(fetch)
    except Exception as exc:
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: frame reader failed on frames "
            f"{fetch[0]}..{fetch[-1]} of {locator.source_path}: {exc}"
        ) from exc
    decoded = _as_numpy(batch)
    if len(decoded) != len(fetch):
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: decord returned {len(decoded)} "
            f"frames for {len(fetch)} requested indices"
        )
    by_abs = {index: decoded[position] for position, index in enumerate(fetch)}
    missing = [index for index in absolute if index not in by_abs]
    if missing:
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: decode plan missed frames {missing[:8]}"
        )
    import numpy as np  # noqa: PLC0415 - array stack only, after the decoder returned

    return np.stack([by_abs[index] for index in absolute], axis=0)


def _as_numpy(batch: Any) -> Any:
    if hasattr(batch, "asnumpy"):
        return batch.asnumpy()
    if hasattr(batch, "numpy"):
        return batch.numpy()
    return batch


def clamp_shot_to_file(locator: ShotLocator, reader: Any) -> ShotLocator:
    """Fail loudly when a declared shot range does not fit its file."""

    total = len(reader)
    if locator.frame_start >= total:
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: shot starts at frame "
            f"{locator.frame_start} but {locator.source_path} has {total} frames"
        )
    last = locator.frame_start + locator.frame_count - 1
    if last >= total:
        raise ShotDecodeError(
            f"{locator.case_id}#{locator.shot_index}: shot ends at frame {last} but "
            f"{locator.source_path} has {total} frames"
        )
    return locator


class SceneFrameReader:
    """FFmpeg 顺序坐标的媒体适配器；复用场景审计读帧，不加预卷。"""

    def __init__(self, scene: dict[str, Any], *, frame_coordinate: str,
                 ffmpeg: str, error_path: Path) -> None:
        if frame_coordinate != "ffmpeg_sequential_zero_based":
            raise ValueError("scene windows require ffmpeg_sequential_zero_based")
        for key in ("width", "height", "video_frame_count"):
            require_positive_int(scene[key], key)
        if float(scene["fps"]) <= 0:
            raise ValueError("scene fps must be positive")
        self.scene = scene
        self.ffmpeg = ffmpeg
        self.error_path = error_path

    def __len__(self) -> int:
        return self.scene["video_frame_count"]

    def get_avg_fps(self) -> float:
        return float(self.scene["fps"])

    def get_batch(self, indices: Sequence[int]) -> Any:
        import numpy as np
        from mspc.data.frames import iter_scene_frames

        wanted = sorted(set(indices))
        try:
            frames = dict(iter_scene_frames(self.scene, wanted, ffmpeg=self.ffmpeg,
                                           error_path=self.error_path, video_decode_num_threads=1))
        except (ValueError, RuntimeError) as exc:
            raise ShotDecodeError(f"{type(exc).__name__}: {exc}".strip()) from exc
        return np.stack([frames[i] for i in indices])
