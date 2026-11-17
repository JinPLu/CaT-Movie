"""Scene -> shot manifests for the round-1 scene-level training pool.

Two sources, two ways to get shots and scenes:

- CinePile-T2V ``raw_videos_batched``: MovieClips clips, each one titled as one
  scene of a film.  Shots come from TransNetV2 (the evaluation stack's engine and
  its peak -> boundary convention); one clip is one scene group.
- MovieChat-1K + MovieChat-SSeg: human shot and scene annotations.  The SSeg
  frame indices are checked against TransNetV2 cuts on the same decoded frames
  before a video is admitted.

Detection (decode, TransNetV2, per-frame measurements) is separate from building
(rules, filtering, grouping, statistics), so the flag thresholds can be changed
without decoding the videos again.

Frame convention: every frame index is a decoder frame index of the source file
(``-fps_mode passthrough``); shot and scene spans are half-open ``[start, end)``
like ``source_shot_segments`` in the multishot CSVs.
"""

from __future__ import annotations

import base64
import json
import math
import random
import re
import subprocess
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mspc.evaluation.renderer_qualification_public_metrics import (
    cut_boundaries_from_peaks,
    transition_run_peaks,
)


SCHEMA_VERSION = "scene_shot_manifest_v1"
DETECTION_SCHEMA_VERSION = "scene_shot_detection_v1"
SOURCE_CINEPILE = "cinepile_t2v"
SOURCE_MOVIECHAT = "moviechat_1k"
SOURCES = (SOURCE_CINEPILE, SOURCE_MOVIECHAT)

# Frames are decoded once at the measurement grid (ffmpeg bicubic); TransNetV2
# gets the exact 4x4 block mean of that grid, i.e. its native 48x27 input.
MEASURE_WIDTH, MEASURE_HEIGHT = 192, 108
TRANSNET_WIDTH, TRANSNET_HEIGHT = 48, 27
TRANSNET_THRESHOLD = 0.5
DECODE_CHUNK_FRAMES = 256

MIN_SHOT_SECONDS = 1.0

# Per-frame measurements on the luma of the measurement grid, stored as uint8.
# name -> (scale applied before rounding, what it measures)
MEASUREMENT_CHANNELS: dict[str, tuple[float, str]] = {
    "black_fraction": (250.0, "share of pixels with luma < BLACK_LUMA"),
    "dark_fraction": (250.0, "share of pixels with luma < DARK_LUMA"),
    "bright_fraction": (1000.0, "share of pixels with luma >= BRIGHT_LUMA, saturates at 0.255"),
    "abs_diff": (10.0, "mean |luma_t - luma_t-1|, saturates at 25.5; frame 0 is saturated"),
    "left_abs_diff": (10.0, "abs_diff over the left quarter of the frame (where the end-card poster sits)"),
}
BLACK_LUMA = 16
DARK_LUMA = 32
BRIGHT_LUMA = 170
LETTERBOX_LUMA = 16

# Frame-level criteria (on de-quantized measurements) and the share of a shot's
# frames that must meet them for the shot to carry the flag.  Flags mark, they
# never delete.
# "frozen" is a repeated picture (abs_diff rounds to 0.0), not a calm shot:
# locked-off dialogue shots measure 0.1-0.3 at this grid.  "text_on_black" is a
# title or credit card: a black frame with a few bright marks.
FRAME_FLAG_RULES: dict[str, dict[str, float]] = {
    "black": {"black_fraction_min": 0.99, "shot_share_min": 0.9},
    "frozen": {"abs_diff_max": 0.05, "shot_share_min": 0.9},
    "text_on_black": {
        "black_fraction_min": 0.9,
        "bright_fraction_min": 0.002,
        "bright_fraction_max": 0.1,
        "shot_share_min": 0.5,
    },
}
# Older MovieClips uploads end with a burned-in card: a still poster in the left
# quarter and playing thumbnails on the right, usually entered by a dissolve that
# TransNetV2 does not cut.  The card is the trailing run of frames whose left
# quarter is still and not black; any shot holding at least half a second of that
# run carries the flag, including a film shot the dissolve merged into the card.
END_CARD_FLAG = "end_card"
END_CARD_RULE = {
    "left_abs_diff_max": 0.3,
    "dark_fraction_max": 0.9,
    "min_seconds": 1.5,
    "shot_overlap_seconds_min": 0.5,
}

# Burned-in subtitles: a few frames sampled across the video at a size where
# subtitle strokes survive; white pixels outlined by dark ones are counted in a
# bottom band and in a top control band.  A video whose bottom band carries such
# marks in enough samples (and clearly more than the top) gets a video flag.
TEXT_PROBE = {
    "samples": 12,
    "width": 640,
    "height": 360,
    "bottom_band": (0.78, 0.97),
    "top_band": (0.03, 0.22),
    "columns": (0.1, 0.9),
    "white_min": 215,
    "outline_max": 70,
    "outline_reach": 2,
}
SUBTITLE_RULE = {"band_score_min": 0.004, "bottom_over_top_min": 4.0, "samples_min": 4}
SUBTITLE_FLAG = "burned_in_subtitles"

# SSeg shot boundaries vs TransNetV2 boundaries on the same decoded frames.  A
# video is aligned when enough annotated boundaries sit on detected cuts and no
# shift of the annotation explains clearly more of them.
ALIGNMENT_RULE = {
    "tolerance_frames": 2,
    "max_offset_frames": 12,
    "min_recall": 0.6,
    "max_recall_gain_from_shift": 0.05,
}


# ---------------------------------------------------------------------------
# Source enumeration and metadata


_UNNAMED = re.compile(r"^tmp[A-Za-z0-9_]{8}$")
_YEAR = re.compile(r"\((\d{4})(?!\d)")
_CLIP_INDEX = re.compile(r"(?:\(|_)(\d+)_(\d+)\)|_#(\d+)_")
_TITLE_NUMBER = re.compile(r"_\(\d{4}|_\(\d+_\d+\)|_\d+_\d+\)|_#\d+(?:_|$)")


def _title_text(slug: str) -> str:
    # The uploader's file-name sanitizer turned ":" into "__" and spaces into "_".
    return slug.replace("__", ": ").replace("_", " ").strip()


def parse_cinepile_clip_name(file_name: str) -> dict[str, Any]:
    """Best-effort film title, year and clip index from a MovieClips upload name.

    The raw zips mix two naming styles (``... Scene (k/N) | Movieclips`` and the
    older ``Title (k/N) Movie CLIP - ... HD``) plus hand-edited variants and
    ``Scene title - Film (k/N) Movie CLIP (year) HD`` and unnamed files.
    Film-level holdout and training exclusion consume these fields.
    """

    stem = Path(file_name).stem
    if _UNNAMED.match(stem):
        return {"name_style": "unnamed_tmp"}
    if "_|_Movieclips" in stem:
        style = "movieclips_scene"
    elif "Movie_CLIP" in stem:
        style = "movie_clip_hd"
    else:
        style = "other"
    year = _YEAR.search(stem)
    index = _CLIP_INDEX.search(stem)
    # Movie CLIP terminates the title region even when no (k/N) is present.
    # Otherwise its trailing scene title can masquerade as a scene-first film.
    title_region = stem.split("_Movie_CLIP", 1)[0] if style == "movie_clip_hd" else stem
    numbered = _TITLE_NUMBER.search(title_region)
    prefix = title_region[:numbered.start()] if numbered else title_region
    if style == "movie_clip_hd":
        # A scene separator after Movie CLIP identifies film-first uploads;
        # any earlier dash then belongs to the film (Mockingjay - Part 1).
        scene_after_clip = stem.split("_Movie_CLIP", 1)[1].startswith("_-_")
        title = prefix.rsplit("_-_", 1)[1] if "_-_" in prefix and not scene_after_clip else prefix
    elif style == "movieclips_scene" and year:
        # A dash before the year belongs to the film (Mission: Impossible -
        # Rogue Nation); the scene separator follows the year.
        title = stem[:year.start()].rstrip("_")
        if "_Scene_-_" in title:
            title = title.rsplit("_Scene_-_", 1)[1]
    else:
        title = prefix.split("_-_", 1)[0]
    return {
        "name_style": style,
        "film_title": _title_text(title),
        "year": int(year.group(1)) if year else None,
        "clip_index": int(index.group(1) or index.group(3)) if index else None,
        "clip_total": int(index.group(2)) if index and index.group(2) else None,
    }


def enumerate_cinepile_videos(video_root: Path) -> list[dict[str, str]]:
    """Every extracted clip under ``<root>/<batch>/<file>.mp4``, sorted."""

    root = Path(video_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CinePile video root does not exist: {root}")
    videos = []
    for path in sorted(root.glob("*/*.mp4")):
        videos.append({
            "source": SOURCE_CINEPILE,
            "video_id": f"{path.parent.name}/{path.stem}",
            "video_path": str(path),
        })
    if not videos:
        raise ValueError(f"no mp4 clips under {root}/<batch>/")
    return videos


def _read_int_pairs(path: Path) -> list[tuple[int, int]]:
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected two integers, got {line!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise ValueError(f"{path} is empty")
    return pairs


def parse_sseg_video(shot_text_path: Path, scene_text_path: Path) -> dict[str, Any]:
    """SSeg files use inclusive indices; return half-open shots and scenes.

    ``shots[i] = [start, end)`` in frames; ``scenes[j] = [first_shot, last_shot + 1)``.
    The release invariants (contiguous from 0, scenes partition the shots) are
    checked here because a violation means the two files do not describe the
    same segmentation.
    """

    shots_inclusive = _read_int_pairs(Path(shot_text_path))
    scenes_inclusive = _read_int_pairs(Path(scene_text_path))
    shots: list[list[int]] = []
    expected = 0
    for start, end in shots_inclusive:
        if start != expected or end < start:
            raise ValueError(f"{shot_text_path}: shots are not contiguous from 0 at {start} {end}")
        shots.append([start, end + 1])
        expected = end + 1
    scenes: list[list[int]] = []
    expected = 0
    for first, last in scenes_inclusive:
        if first != expected or last < first:
            raise ValueError(f"{scene_text_path}: scenes are not contiguous from shot 0 at {first} {last}")
        scenes.append([first, last + 1])
        expected = last + 1
    if expected != len(shots):
        raise ValueError(
            f"{scene_text_path}: scenes end at shot {expected - 1}, shot list has {len(shots)} shots"
        )
    return {"shots": shots, "scenes": scenes}


def load_moviechat_sseg_index(sseg_root: Path, moviechat_root: Path) -> list[dict[str, Any]]:
    """Map every SSeg id to its local MovieChat video and annotation files.

    The SSeg manifest keys each canonical id to the upstream repo path
    (``raw_videos/<id>.mp4`` in the train repo, ``videos/<id>.mp4`` in the test
    repo); they were downloaded to ``<moviechat_root>/{train,test}``.
    """

    sseg_root = Path(sseg_root)
    moviechat_root = Path(moviechat_root)
    manifest = json.loads((sseg_root / "data/metadata/video_manifest.json").read_text(encoding="utf-8"))
    genres: dict[str, str] = {}
    for line in (sseg_root / "data/metadata/video_type.csv").read_text(encoding="utf-8").splitlines()[1:]:
        video_id, genre = line.split(",")
        genres[video_id] = genre
    videos = []
    for entry in manifest:
        split = entry["split"]
        if split not in ("train", "test"):
            raise ValueError(f"unknown MovieChat split {split!r} for {entry['id']}")
        video_id = entry["id"]
        videos.append({
            "source": SOURCE_MOVIECHAT,
            "video_id": video_id,
            "video_path": str(moviechat_root / split / entry["repo_path"]),
            "split": split,
            "genre": genres[video_id],
            "shot_file": str(sseg_root / "data/shot" / f"{video_id}.txt"),
            "scene_file": str(sseg_root / "data/scene" / f"{video_id}.txt"),
        })
    if len(videos) != len(genres):
        raise ValueError(f"SSeg manifest has {len(videos)} videos, video_type.csv has {len(genres)}")
    return sorted(videos, key=lambda item: item["video_id"])


# ---------------------------------------------------------------------------
# Detection: probe, decode, measure, TransNetV2


def probe_video(ffprobe: str, video_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name",
            "-of", "json", video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"{video_path}: expected one video stream, ffprobe returned {len(streams)}")
    stream = streams[0]
    fps = Fraction(stream["avg_frame_rate"]) if stream.get("avg_frame_rate") not in (None, "0/0") else None
    if fps is None or fps <= 0:
        raise ValueError(f"{video_path}: no usable avg_frame_rate ({stream.get('avg_frame_rate')!r})")
    nb_frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "fps_fraction": f"{fps.numerator}/{fps.denominator}",
        "r_frame_rate": stream.get("r_frame_rate"),
        "codec": stream.get("codec_name"),
        "container_frame_count": int(nb_frames) if nb_frames not in (None, "N/A") else None,
    }


def _quantize(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(values * scale), 0, 255).astype(np.uint8)


def encode_channel(values: np.ndarray) -> str:
    if values.dtype != np.uint8 or values.ndim != 1:
        raise ValueError("measurement channels are 1-D uint8")
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_channel(text: str, name: str) -> np.ndarray:
    scale = MEASUREMENT_CHANNELS[name][0]
    return np.frombuffer(base64.b64decode(text), dtype=np.uint8).astype(np.float32) / scale


def measure_frames(rgb: np.ndarray, previous_luma: np.ndarray | None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per-frame measurements and TransNetV2 input for one decoded chunk.

    ``rgb`` is ``(n, MEASURE_HEIGHT, MEASURE_WIDTH, 3)`` uint8.  Returns the
    unquantized channels, the 48x27 frames (exact 4x4 block mean, the same as
    OpenCV INTER_AREA at this integer factor) and the chunk's luma.
    """

    if rgb.ndim != 4 or rgb.shape[1:] != (MEASURE_HEIGHT, MEASURE_WIDTH, 3):
        raise ValueError(f"expected (n, {MEASURE_HEIGHT}, {MEASURE_WIDTH}, 3), got {rgb.shape}")
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    previous = luma[:1] if previous_luma is None else previous_luma[None]
    change = np.abs(luma - np.concatenate([previous, luma[:-1]]))
    diffs = change.mean(axis=(1, 2))
    left_diffs = change[:, :, : MEASURE_WIDTH // 4].mean(axis=(1, 2))
    if previous_luma is None:
        diffs[0] = left_diffs[0] = np.inf
    channels = {
        "black_fraction": (luma < BLACK_LUMA).mean(axis=(1, 2)),
        "dark_fraction": (luma < DARK_LUMA).mean(axis=(1, 2)),
        "bright_fraction": (luma >= BRIGHT_LUMA).mean(axis=(1, 2)),
        "abs_diff": diffs,
        "left_abs_diff": left_diffs,
    }
    fy, fx = MEASURE_HEIGHT // TRANSNET_HEIGHT, MEASURE_WIDTH // TRANSNET_WIDTH
    small = rgb.reshape(len(rgb), TRANSNET_HEIGHT, fy, TRANSNET_WIDTH, fx, 3).mean(axis=(2, 4))
    return channels, np.rint(small).astype(np.uint8), luma


def decode_and_measure(video_path: str, *, ffmpeg: str, threads: int) -> dict[str, Any]:
    """Decode every frame once at the measurement grid and measure it."""

    frame_bytes = MEASURE_WIDTH * MEASURE_HEIGHT * 3
    command = [
        ffmpeg, "-v", "error", "-nostdin", "-threads", str(threads), "-i", video_path,
        "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{MEASURE_WIDTH}x{MEASURE_HEIGHT}", "pipe:1",
    ]
    parts: dict[str, list[np.ndarray]] = {name: [] for name in MEASUREMENT_CHANNELS}
    small_parts: list[np.ndarray] = []
    row_sum = np.zeros(MEASURE_HEIGHT, dtype=np.float64)
    col_sum = np.zeros(MEASURE_WIDTH, dtype=np.float64)
    previous = None
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
        assert process.stdout is not None
        while True:
            buffer = process.stdout.read(frame_bytes * DECODE_CHUNK_FRAMES)
            if not buffer:
                break
            if len(buffer) % frame_bytes:
                process.kill()
                raise RuntimeError(f"{video_path}: truncated raw frame from ffmpeg")
            rgb = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, MEASURE_HEIGHT, MEASURE_WIDTH, 3)
            channels, small, luma = measure_frames(rgb, previous)
            for name, values in channels.items():
                parts[name].append(values)
            small_parts.append(small)
            row_sum += luma.sum(axis=(0, 2))
            col_sum += luma.sum(axis=(0, 1))
            previous = luma[-1]
        return_code = process.wait()
        stderr_file.seek(0)
        stderr_lines = stderr_file.read().decode("utf-8", "replace").splitlines()
    if return_code != 0:
        tail = " | ".join(stderr_lines[-3:])
        raise RuntimeError(f"{video_path}: ffmpeg exited {return_code}: {tail}")
    if not small_parts:
        raise RuntimeError(f"{video_path}: ffmpeg decoded no frames")
    frame_count = sum(len(part) for part in small_parts)
    measurements = {
        name: encode_channel(_quantize(np.concatenate(values), MEASUREMENT_CHANNELS[name][0]))
        for name, values in parts.items()
    }
    measurements["row_luma"] = [int(v) for v in np.rint(row_sum / (frame_count * MEASURE_WIDTH))]
    measurements["col_luma"] = [int(v) for v in np.rint(col_sum / (frame_count * MEASURE_HEIGHT))]
    return {
        "frames": np.concatenate(small_parts),
        "measurements": measurements,
        "decode_error_lines": len(stderr_lines),
        "decode_error_tail": stderr_lines[-2:],
    }


def text_band_score(gray: np.ndarray, band: tuple[float, float]) -> float:
    """Share of band pixels that are white and have a dark pixel within reach."""

    height, width = gray.shape
    top, bottom = int(band[0] * height), int(band[1] * height)
    left, right = (int(edge * width) for edge in TEXT_PROBE["columns"])
    region = gray[top:bottom, left:right]
    reach = TEXT_PROBE["outline_reach"]
    dark = np.pad(region <= TEXT_PROBE["outline_max"], reach)
    near_dark = np.zeros(region.shape, dtype=bool)
    for dy in range(2 * reach + 1):
        for dx in range(2 * reach + 1):
            near_dark |= dark[dy:dy + region.shape[0], dx:dx + region.shape[1]]
    return float(((region >= TEXT_PROBE["white_min"]) & near_dark).mean())


def probe_text_bands(ffmpeg: str, video_path: str, duration_s: float) -> list[list[float]]:
    """``[bottom, top]`` band scores on evenly spaced sampled frames."""

    width, height, samples = TEXT_PROBE["width"], TEXT_PROBE["height"], TEXT_PROBE["samples"]
    scores = []
    for index in range(samples):
        seconds = (index + 0.5) * duration_s / samples
        completed = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{seconds:.3f}", "-i", video_path, "-map", "0:v:0",
             "-frames:v", "1", "-s", f"{width}x{height}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
            check=True,
            capture_output=True,
        )
        if len(completed.stdout) != width * height:
            raise RuntimeError(f"{video_path}: no frame at {seconds:.3f}s for the text probe")
        gray = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width)
        scores.append([
            round(text_band_score(gray, TEXT_PROBE["bottom_band"]), 5),
            round(text_band_score(gray, TEXT_PROBE["top_band"]), 5),
        ])
    return scores


def subtitle_hits(samples: Sequence[Sequence[float]]) -> int:
    return sum(
        1
        for bottom, top in samples
        if bottom >= SUBTITLE_RULE["band_score_min"] and bottom >= SUBTITLE_RULE["bottom_over_top_min"] * top
    )


def transnet_cuts(probabilities: Sequence[float], frame_count: int) -> tuple[list[int], list[float]]:
    """Cut boundaries (first frame of the incoming shot) and each cut's peak probability."""

    if len(probabilities) != frame_count:
        raise ValueError(
            f"TransNetV2 returned {len(probabilities)} probabilities for {frame_count} frames"
        )
    peaks = transition_run_peaks(probabilities, threshold=TRANSNET_THRESHOLD)
    boundaries = cut_boundaries_from_peaks(peaks, frame_count=frame_count)
    return list(boundaries), [round(float(probabilities[b - 1]), 4) for b in boundaries]


# TransNetV2 at TRANSNET_THRESHOLD already defined the frozen shot spans, so a
# second pass at that threshold cannot see the hard cuts it merged into one
# shot. Intra-shot audit uses the PySceneDetect ContentDetector score instead:
# the mean of the three channel-wise mean absolute differences on OpenCV HSV
# (H in 0..180, S and V in 0..255). One place owns that decision.
CONTENT_DIFF_WIDTH = 96
CONTENT_DIFF_HEIGHT = 54
# PySceneDetect ContentDetector's default, kept after the 2026-10-08 sweep on
# the 29 test-114 shots whose visual review described an intra-shot hard cut.
# A lower floor also fires on pans whose score never leaves the motion band.
INTRA_SHOT_CONTENT_THRESHOLD = 27.0
INTRA_SHOT_CONTENT_MIN_GAP_SECONDS = 0.4


def bgr_to_opencv_hsv(bgr: np.ndarray) -> np.ndarray:
    """OpenCV ``COLOR_BGR2HSV`` for uint8 BGR images, HxWx3 or NxHxWx3."""

    if bgr.dtype != np.uint8 or bgr.shape[-1] != 3 or bgr.ndim not in (3, 4):
        raise ValueError(f"expected uint8 HxWx3 or NxHxWx3 BGR, got {getattr(bgr, 'shape', None)} {getattr(bgr, 'dtype', None)}")
    image = bgr.astype(np.float32)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.maximum(np.maximum(red, green), blue)
    minc = np.minimum(np.minimum(red, green), blue)
    diff = maxc - minc
    sat = np.zeros_like(maxc)
    np.divide(diff * 255.0, maxc, out=sat, where=maxc > 0)
    hue = np.zeros_like(maxc)
    nonzero = diff > 0
    red_max = nonzero & (red >= green) & (red >= blue)
    green_max = nonzero & (green > red) & (green >= blue)
    blue_max = nonzero & (blue > red) & (blue > green)
    hue[red_max] = ((green[red_max] - blue[red_max]) / diff[red_max]) % 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / diff[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / diff[blue_max]) + 4.0
    hsv = np.stack([hue * 30.0, sat, maxc], axis=-1)
    return np.clip(np.rint(hsv), 0, 255).astype(np.uint8)


def content_diff_scores(bgr_frames: np.ndarray) -> np.ndarray:
    """Content score introducing each frame after the first.

    ``bgr_frames`` is uint8 BGR ``(n, h, w, 3)``. ``scores[i]`` compares frame
    ``i`` with frame ``i + 1`` and equals ``(mean|ΔH| + mean|ΔS| + mean|ΔV|) / 3``.
    """

    if bgr_frames.dtype != np.uint8 or bgr_frames.ndim != 4 or bgr_frames.shape[-1] != 3:
        raise ValueError("content-diff frames must be uint8 NxHxWx3 BGR")
    if len(bgr_frames) < 2:
        return np.zeros((0,), dtype=np.float64)
    hsv = bgr_to_opencv_hsv(bgr_frames).astype(np.int16)
    delta = np.abs(hsv[1:] - hsv[:-1])
    return delta.mean(axis=(1, 2)).mean(axis=1)


def content_diff_cuts(
    scores: Sequence[float], *, threshold: float, min_gap_frames: int
) -> list[int]:
    """Incoming-frame indices of hard cuts inside one already-cut shot.

    A cut is the strongest frame of a contiguous run whose scores are all at
    least ``threshold`` (the same run convention as :func:`transition_run_peaks`).
    Frame ``i + 1`` is that incoming frame when ``scores[i]`` is the run's peak.
    Runs whose peaks are closer than ``min_gap_frames`` keep the higher peak,
    and the earlier frame on a tie. The caller chooses ``threshold``; this
    function does not fill one in.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("content-diff threshold must be a positive finite number")
    if isinstance(min_gap_frames, bool) or not isinstance(min_gap_frames, int) or min_gap_frames < 1:
        raise ValueError("min_gap_frames must be a positive int")
    values = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in values):
        raise ValueError("content-diff scores must be finite")
    peaks: list[tuple[int, float]] = []
    index = 0
    while index < len(values):
        if values[index] < threshold:
            index += 1
            continue
        start = index
        while index + 1 < len(values) and values[index + 1] >= threshold:
            index += 1
        peak = max(range(start, index + 1), key=lambda item: (values[item], -item))
        peaks.append((peak + 1, values[peak]))
        index += 1
    kept: list[tuple[int, float]] = []
    for frame, score in peaks:
        if kept and frame - kept[-1][0] < min_gap_frames:
            if score > kept[-1][1]:
                kept[-1] = (frame, score)
            continue
        kept.append((frame, score))
    return [frame for frame, _score in kept]


def detect_video(video: Mapping[str, Any], *, engine: Any, ffmpeg: str, ffprobe: str, threads: int) -> dict[str, Any]:
    """One detection row: probe, decode+measure, TransNetV2 cuts."""

    probe = probe_video(ffprobe, video["video_path"])
    decoded = decode_and_measure(video["video_path"], ffmpeg=ffmpeg, threads=threads)
    frames = decoded.pop("frames")
    boundaries, peak_probabilities = transnet_cuts(engine.predict(frames), len(frames))
    text_bands = probe_text_bands(ffmpeg, video["video_path"], len(frames) / probe["fps"])
    return {
        "schema_version": DETECTION_SCHEMA_VERSION,
        "status": "ok",
        "source": video["source"],
        "video_id": video["video_id"],
        "video_path": video["video_path"],
        **probe,
        "decoded_frame_count": int(len(frames)),
        "transnet_cut_boundaries": boundaries,
        "transnet_cut_peak_probabilities": peak_probabilities,
        "transnet_threshold": TRANSNET_THRESHOLD,
        "text_band_samples": text_bands,
        **decoded,
    }


# ---------------------------------------------------------------------------
# Rules: flags, letterbox, alignment


def frame_flag_masks(measurements: Mapping[str, Any]) -> dict[str, np.ndarray]:
    black = decode_channel(measurements["black_fraction"], "black_fraction")
    bright = decode_channel(measurements["bright_fraction"], "bright_fraction")
    diff = decode_channel(measurements["abs_diff"], "abs_diff")
    black_rule = FRAME_FLAG_RULES["black"]
    frozen_rule = FRAME_FLAG_RULES["frozen"]
    text_rule = FRAME_FLAG_RULES["text_on_black"]
    return {
        "black": black >= black_rule["black_fraction_min"],
        "frozen": diff <= frozen_rule["abs_diff_max"],
        "text_on_black": (
            (black >= text_rule["black_fraction_min"])
            & (bright >= text_rule["bright_fraction_min"])
            & (bright <= text_rule["bright_fraction_max"])
        ),
    }


def shot_flags(masks: Mapping[str, np.ndarray], start: int, end: int) -> list[str]:
    flags = []
    for name, rule in FRAME_FLAG_RULES.items():
        # The first frame of a shot follows a cut, so its difference is not a
        # property of the shot.
        window = masks[name][start + 1:end] if name == "frozen" and end - start > 1 else masks[name][start:end]
        if len(window) and float(window.mean()) >= rule["shot_share_min"]:
            flags.append(name)
    return flags


def end_card_start(measurements: Mapping[str, Any], fps: float) -> int | None:
    """First frame of the trailing end-card run, or None when there is none."""

    left = decode_channel(measurements["left_abs_diff"], "left_abs_diff")
    dark = decode_channel(measurements["dark_fraction"], "dark_fraction")
    card_like = (left <= END_CARD_RULE["left_abs_diff_max"]) & (dark <= END_CARD_RULE["dark_fraction_max"])
    run = 0
    while run < len(card_like) and card_like[len(card_like) - 1 - run]:
        run += 1
    if run / fps < END_CARD_RULE["min_seconds"]:
        return None
    return len(card_like) - run


def active_box(row_luma: Sequence[int], col_luma: Sequence[int], width: int, height: int) -> list[int]:
    """Picture area inside letterbox / pillarbox bars, in source pixels ``[x0, y0, x1, y1)``."""

    def span(profile: Sequence[int]) -> tuple[int, int]:
        bright = [index for index, value in enumerate(profile) if value >= LETTERBOX_LUMA]
        if not bright:
            return 0, len(profile)
        return bright[0], bright[-1] + 1

    top, bottom = span(row_luma)
    left, right = span(col_luma)
    return [
        int(round(left * width / len(col_luma))),
        int(round(top * height / len(row_luma))),
        int(round(right * width / len(col_luma))),
        int(round(bottom * height / len(row_luma))),
    ]


def boundary_alignment(
    annotated: Sequence[int], detected: Sequence[int], *, tolerance: int, max_offset: int
) -> dict[str, Any]:
    """How well annotated shot boundaries sit on detected cuts, and at which shift.

    ``recall_at_zero`` is the share of annotated boundaries with a detected cut
    within ``tolerance`` frames; ``best_offset`` is the shift of the annotation
    (in frames) that maximizes that share.
    """

    detected_sorted = np.asarray(sorted(detected), dtype=np.int64)
    annotated_array = np.asarray(annotated, dtype=np.int64)
    if len(annotated_array) == 0 or len(detected_sorted) == 0:
        return {"annotated": len(annotated_array), "detected": len(detected_sorted),
                "recall_at_zero": 0.0, "best_offset": 0, "recall_at_best": 0.0}

    def recall(shift: int) -> float:
        shifted = annotated_array + shift
        index = np.searchsorted(detected_sorted, shifted)
        left = np.abs(shifted - detected_sorted[np.clip(index - 1, 0, len(detected_sorted) - 1)])
        right = np.abs(shifted - detected_sorted[np.clip(index, 0, len(detected_sorted) - 1)])
        return float((np.minimum(left, right) <= tolerance).mean())

    scores = {shift: recall(shift) for shift in range(-max_offset, max_offset + 1)}
    best_offset = max(scores, key=lambda shift: (scores[shift], -abs(shift)))
    return {
        "annotated": int(len(annotated_array)),
        "detected": int(len(detected_sorted)),
        "recall_at_zero": round(scores[0], 4),
        "best_offset": int(best_offset),
        "recall_at_best": round(scores[best_offset], 4),
    }


# ---------------------------------------------------------------------------
# Building scene rows


def _shot_entries(
    segments: Sequence[Sequence[int]], *, fps: float, masks: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    shots = []
    for index, (start, end) in enumerate(segments):
        count = int(end - start)
        shots.append({
            "source_shot_index": index,
            "start_frame": int(start),
            "end_frame": int(end),
            "frame_count": count,
            "duration_s": round(count / fps, 3),
            "flags": shot_flags(masks, int(start), int(end)),
        })
    return shots


def _video_fields(detection: Mapping[str, Any]) -> dict[str, Any]:
    container = detection["container_frame_count"]
    hits = subtitle_hits(detection["text_band_samples"])
    return {
        "source": detection["source"],
        "video_id": detection["video_id"],
        "source_video_path": detection["video_path"],
        "fps": detection["fps"],
        "fps_fraction": detection["fps_fraction"],
        "width": detection["width"],
        "height": detection["height"],
        "video_frame_count": detection["decoded_frame_count"],
        # Container-minus-sequential-decoded count, retained as a diagnostic.
        # It is NOT a universal decord index offset: the initial GOP of
        # MovieChat earth11-7 fails that mapping (FX3 scene 127). Scene frame
        # indices are consumed with FFmpeg sequential decoding, as produced.
        "container_preroll_frames": None if container is None else container - detection["decoded_frame_count"],
        "active_box": active_box(
            detection["measurements"]["row_luma"],
            detection["measurements"]["col_luma"],
            detection["width"],
            detection["height"],
        ),
        "subtitle_sample_hits": hits,
        "video_flags": [SUBTITLE_FLAG] if hits >= SUBTITLE_RULE["samples_min"] else [],
    }


def segments_from_boundaries(boundaries: Sequence[int], frame_count: int) -> list[list[int]]:
    edges = [0, *boundaries, frame_count]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("cut boundaries must be strictly increasing inside the video")
    return [[a, b] for a, b in zip(edges, edges[1:])]


def cinepile_scene_rows(detection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One MovieClips clip is one scene; its shots are the TransNetV2 shots.

    The filename is the only per-clip metadata in the raw zips and it names a
    single scene, so the clip is not split further here; within-clip location
    changes are left to the shot-boundary labels (``scene_change``).
    """

    frame_count = detection["decoded_frame_count"]
    masks = frame_flag_masks(detection["measurements"])
    segments = segments_from_boundaries(detection["transnet_cut_boundaries"], frame_count)
    shots = _shot_entries(segments, fps=detection["fps"], masks=masks)
    card_start = end_card_start(detection["measurements"], detection["fps"])
    if card_start is not None:
        for shot in shots:
            inside = max(0, shot["end_frame"] - max(shot["start_frame"], card_start))
            if inside / detection["fps"] >= END_CARD_RULE["shot_overlap_seconds_min"]:
                shot["flags"].append(END_CARD_FLAG)
    kept = [shot for shot in shots if shot["duration_s"] >= MIN_SHOT_SECONDS]
    if not kept:
        return []
    return [{
        "schema_version": SCHEMA_VERSION,
        "scene_id": f"{SOURCE_CINEPILE}:{detection['video_id']}:scene000",
        **_video_fields(detection),
        "scene_index": 0,
        "scene_frame_span": [0, frame_count],
        "shot_boundary_source": "transnetv2",
        "scene_boundary_source": "movieclips_clip",
        "end_card_start_frame": card_start,
        "shot_count": len(kept),
        "dropped_short_shots": len(shots) - len(kept),
        "shots": kept,
        "metadata": parse_cinepile_clip_name(Path(detection["video_path"]).name),
    }]


def moviechat_alignment(detection: Mapping[str, Any], sseg: Mapping[str, Any]) -> dict[str, Any]:
    annotated_end = sseg["shots"][-1][1]
    annotated_boundaries = [start for start, _ in sseg["shots"][1:]]
    report = boundary_alignment(
        annotated_boundaries,
        detection["transnet_cut_boundaries"],
        tolerance=ALIGNMENT_RULE["tolerance_frames"],
        max_offset=ALIGNMENT_RULE["max_offset_frames"],
    )
    report["annotated_frame_end"] = annotated_end
    report["decoded_frame_count"] = detection["decoded_frame_count"]
    if annotated_end > detection["decoded_frame_count"]:
        report["status"] = "beyond_video"
    elif report["recall_at_best"] - report["recall_at_zero"] > ALIGNMENT_RULE["max_recall_gain_from_shift"]:
        report["status"] = "shifted"
    elif report["recall_at_zero"] < ALIGNMENT_RULE["min_recall"]:
        # No shift helps, but too few annotated cuts are visible to TransNetV2
        # (mostly dissolves), so the alignment cannot be confirmed.
        report["status"] = "low_agreement"
    else:
        report["status"] = "aligned"
    return report


def moviechat_scene_rows(
    detection: Mapping[str, Any], video: Mapping[str, Any], sseg: Mapping[str, Any], alignment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """One row per SSeg scene of an aligned video; shots are the SSeg shots."""

    if alignment["status"] != "aligned":
        return []
    masks = frame_flag_masks(detection["measurements"])
    all_shots = _shot_entries(sseg["shots"], fps=detection["fps"], masks=masks)
    rows = []
    for scene_index, (first, last) in enumerate(sseg["scenes"]):
        scene_shots = all_shots[first:last]
        kept = [shot for shot in scene_shots if shot["duration_s"] >= MIN_SHOT_SECONDS]
        if not kept:
            continue
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "scene_id": f"{SOURCE_MOVIECHAT}:{detection['video_id']}:scene{scene_index:03d}",
            **_video_fields(detection),
            "scene_index": scene_index,
            "scene_frame_span": [scene_shots[0]["start_frame"], scene_shots[-1]["end_frame"]],
            "shot_boundary_source": "moviechat_sseg",
            "scene_boundary_source": "moviechat_sseg",
            "shot_count": len(kept),
            "dropped_short_shots": len(scene_shots) - len(kept),
            "shots": kept,
            "metadata": {"genre": video["genre"], "split": video["split"]},
            "alignment": {key: alignment[key] for key in ("recall_at_zero", "best_offset", "annotated", "detected")},
        })
    return rows


# ---------------------------------------------------------------------------
# Statistics and review sampling


SHOT_COUNT_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 10**9))
DURATION_BUCKETS_S = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, math.inf))
FLAGS = (*FRAME_FLAG_RULES, END_CARD_FLAG)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        **{f"p{q}": round(float(np.percentile(array, q)), 3) for q in (10, 25, 50, 75, 90)},
        "max": round(float(array.max()), 3),
    }


def _bucket_label(low: float, high: float) -> str:
    if low == high:
        return f"{low:g}"
    if math.isinf(high) or high >= 10**9:
        return f"{low:g}+"
    return f"{low:g}-{high:g}"


def scene_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and distributions for one set of scene rows."""

    shot_counts = [row["shot_count"] for row in rows]
    clean_counts = [
        0 if row["video_flags"] else sum(1 for shot in row["shots"] if not shot["flags"]) for row in rows
    ]
    durations = [shot["duration_s"] for row in rows for shot in row["shots"]]
    flag_counts = Counter(flag for row in rows for shot in row["shots"] for flag in shot["flags"])
    scenes = len(rows)

    def share(count: int) -> float:
        return round(count / scenes, 4) if scenes else 0.0

    shot_hist = {}
    for low, high in SHOT_COUNT_BUCKETS:
        count = sum(1 for value in shot_counts if low <= value <= high)
        shot_hist[_bucket_label(low, high)] = {"scenes": count, "share": share(count)}
    duration_hist = {}
    for low, high in DURATION_BUCKETS_S:
        count = sum(1 for value in durations if low <= value < high)
        duration_hist[_bucket_label(low, high)] = {
            "shots": count, "share": round(count / len(durations), 4) if durations else 0.0
        }
    at_least = {k: sum(1 for value in shot_counts if value >= k) for k in (2, 3, 5)}
    clean_at_least = {k: sum(1 for value in clean_counts if value >= k) for k in (2, 3, 5)}
    return {
        "scenes": scenes,
        "videos": len({row["video_id"] for row in rows}),
        "shots": len(durations),
        "dropped_short_shots": sum(row["dropped_short_shots"] for row in rows),
        "scenes_with_at_least": {str(k): {"scenes": v, "share": share(v)} for k, v in at_least.items()},
        "scenes_with_at_least_unflagged": {str(k): {"scenes": v, "share": share(v)} for k, v in clean_at_least.items()},
        "shots_per_scene": _percentiles(shot_counts),
        "shots_per_scene_histogram": shot_hist,
        "shot_duration_s": _percentiles(durations),
        "shot_duration_histogram": duration_hist,
        "flagged_shots": {flag: flag_counts.get(flag, 0) for flag in FLAGS},
        "scenes_in_videos_flagged": {
            SUBTITLE_FLAG: sum(1 for row in rows if SUBTITLE_FLAG in row["video_flags"])
        },
        "total_shot_hours": round(sum(durations) / 3600, 2),
    }


def review_scene_sample(
    rows: Sequence[Mapping[str, Any]], *, per_source: int, min_shots: int, seed: int
) -> list[Mapping[str, Any]]:
    """Seeded sample of scenes per source for the contact-sheet review."""

    chosen: list[Mapping[str, Any]] = []
    for source in SOURCES:
        pool = sorted(
            (row for row in rows if row["source"] == source and row["shot_count"] >= min_shots),
            key=lambda row: row["scene_id"],
        )
        chosen.extend(random.Random(f"{seed}:{source}").sample(pool, min(per_source, len(pool))))
    return chosen


def review_frame_indices(shot: Mapping[str, Any]) -> tuple[int, int, int]:
    """First, middle and last frame of a shot (half-open span)."""

    start, end = shot["start_frame"], shot["end_frame"]
    return start, (start + end - 1) // 2, end - 1
