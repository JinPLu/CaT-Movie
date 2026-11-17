"""Shared multi-shot frame planning utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


FRAME_POLICY_FPS_AWARE_TARGET_FPS = "fps_aware_target_fps"
VAE_ALIGNMENT_POLICY_LEGACY_FLOOR = "legacy_floor_truncate_v1"
VAE_ALIGNMENT_POLICY_TAIL_PAD = "tail_duplicate_padding_v1"
# 源镜头区间已由 shot_boundary_revision 按检测切点修过首尾，规划器在整段内唯一重采样。
VAE_ALIGNMENT_POLICY_CLEAN = "revised_boundary_unique_resample_v2"


@dataclass(frozen=True)
class ShotPlan:
    raw_shot_groups: list[list[int]]
    source_fps: float
    source_num_frames: int
    target_fps: float
    max_rgb_frames: int
    vae_temporal_stride: int
    shot_durations_sec: list[float]
    duration_weights: list[float]
    target_rgb_lengths: list[int]
    aligned_rgb_lengths: list[int]
    vae_padding_frames: list[int]
    vae_truncated_frames: list[int]
    vae_alignment_policy: str
    latent_lengths: list[int]
    frame_indices: list[int]
    train_shot_groups: list[list[int]]
    latent_shot_groups: list[list[int]]
    prompt_index_by_shot: list[int]
    source_rgb_frames: int
    train_rgb_frames: int
    train_latent_frames: int
    source_duration_sec: float
    target_duration_sec: float
    effective_train_fps: float
    rgb_was_scaled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_policy": FRAME_POLICY_FPS_AWARE_TARGET_FPS,
            "raw_shot_groups": self.raw_shot_groups,
            "source_fps": self.source_fps,
            "source_num_frames": self.source_num_frames,
            "target_fps": self.target_fps,
            "max_rgb_frames": self.max_rgb_frames,
            "vae_temporal_stride": self.vae_temporal_stride,
            "shot_durations_sec": self.shot_durations_sec,
            "duration_weights": self.duration_weights,
            "target_rgb_lengths": self.target_rgb_lengths,
            "aligned_rgb_lengths": self.aligned_rgb_lengths,
            "vae_padding_frames": self.vae_padding_frames,
            "vae_truncated_frames": self.vae_truncated_frames,
            "vae_alignment_policy": self.vae_alignment_policy,
            "latent_lengths": self.latent_lengths,
            "frame_indices": self.frame_indices,
            "train_shot_groups": self.train_shot_groups,
            "shot_groups": self.train_shot_groups,
            "latent_shot_groups": self.latent_shot_groups,
            "prompt_index_by_shot": self.prompt_index_by_shot,
            "source_rgb_frames": self.source_rgb_frames,
            "train_rgb_frames": self.train_rgb_frames,
            "train_latent_frames": self.train_latent_frames,
            "source_duration_sec": self.source_duration_sec,
            "target_duration_sec": self.target_duration_sec,
            "effective_train_fps": self.effective_train_fps,
            "rgb_was_scaled": self.rgb_was_scaled,
        }


def parse_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be a positive number")
    return parsed


def parse_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive int")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive int")
    return parsed


def validate_shot_groups(shot_groups: list[list[int]], data_path: str) -> None:
    if not isinstance(shot_groups, list) or not shot_groups:
        raise ValueError(f"{data_path}: shot_groups must be a non-empty list")
    expected_start = 0
    for shot_index, shot_group in enumerate(shot_groups):
        if not isinstance(shot_group, (list, tuple)) or len(shot_group) != 2:
            raise ValueError(f"{data_path}: invalid shot group at index {shot_index}: {shot_group}")
        shot_start, shot_end = shot_group
        if type(shot_start) is not int or type(shot_end) is not int:
            raise ValueError(f"{data_path}: shot group values must be ints at index {shot_index}")
        if shot_start != expected_start:
            raise ValueError(
                f"{data_path}: non-contiguous shot group at index {shot_index}: "
                f"expected start {expected_start}, got {shot_start}"
            )
        if shot_end <= shot_start:
            raise ValueError(f"{data_path}: empty shot group at index {shot_index}: {shot_group}")
        expected_start = shot_end


def validate_source_shot_segments(segments: list[list[int]], data_path: str) -> None:
    """Validate ordered half-open source segments while permitting transition gaps."""

    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{data_path}: source_shot_segments must be a non-empty list")
    previous_end = 0
    for shot_index, segment in enumerate(segments):
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError(
                f"{data_path}: invalid source shot segment at index {shot_index}: {segment}"
            )
        start, end = segment
        if type(start) is not int or type(end) is not int:
            raise ValueError(
                f"{data_path}: source shot segment values must be ints at index {shot_index}"
            )
        if start < 0 or end <= start:
            raise ValueError(
                f"{data_path}: invalid source shot segment at index {shot_index}: {segment}"
            )
        if shot_index and start < previous_end:
            raise ValueError(
                f"{data_path}: overlapping source shot segment at index {shot_index}: {segment}"
            )
        previous_end = end


def validate_multisource_shot_segments(segments: list[list[int]], data_path: str) -> None:
    """Validate one half-open source segment per encoded shot.

    Unlike ``validate_source_shot_segments``, entries can overlap because each
    one may belong to a different source video.  Their correspondence to media
    is carried by the same-index ``source_video_paths`` field.
    """

    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{data_path}: source_shot_segments must be a non-empty list")
    for shot_index, segment in enumerate(segments):
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError(
                f"{data_path}: invalid source shot segment at index {shot_index}: {segment}"
            )
        start, end = segment
        if type(start) is not int or type(end) is not int:
            raise ValueError(
                f"{data_path}: source shot segment values must be ints at index {shot_index}"
            )
        if start < 0 or end <= start:
            raise ValueError(
                f"{data_path}: invalid source shot segment at index {shot_index}: {segment}"
            )


def round_frame_count(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("frame count source value must be positive")
    return max(1, int(math.floor(value + 0.5)))


def resample_frame_count(frame_count: int, *, source_fps: float, target_fps: float,
                         temporal_stride: int) -> int:
    """显式时间坐标之间换算镜长，取最近的 ``1 mod stride``，保留官方 Wan 舍入。"""
    frames = parse_positive_int(frame_count, "frame_count")
    source = parse_positive_float(source_fps, "source_fps")
    target = parse_positive_float(target_fps, "target_fps")
    stride = parse_positive_int(temporal_stride, "temporal_stride")
    return max(1, stride * round((frames * target / source - 1) / stride) + 1)


def align_rgb_length_to_vae(rgb_frames: int, stride: int) -> int:
    """Return the legacy floor-aligned RGB length.

    This remains solely for manifests produced before the no-story-compression
    contract.  New explicit mappings must use ``pad_rgb_length_to_vae``.
    """

    frames = parse_positive_int(rgb_frames, "rgb_frames")
    temporal_stride = parse_positive_int(stride, "vae_temporal_stride")
    return 1 + ((frames - 1) // temporal_stride) * temporal_stride


def pad_rgb_length_to_vae(rgb_frames: int, stride: int) -> int:
    """Expand to ``1 mod stride`` without removing a selected temporal sample."""

    frames = parse_positive_int(rgb_frames, "rgb_frames")
    temporal_stride = parse_positive_int(stride, "vae_temporal_stride")
    return frames + (-(frames - 1) % temporal_stride)


def tail_pad_frame_indices(
    frame_indices: list[int],
    *,
    aligned_length: int,
    label: str,
) -> list[int]:
    """Pad one encoded shot by duplicating only its final selected source frame."""

    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError(f"{label}: frame_indices must be a non-empty list")
    if any(type(index) is not int or index < 0 for index in frame_indices):
        raise ValueError(f"{label}: frame_indices must be non-negative ints")
    target_length = parse_positive_int(aligned_length, "aligned_length")
    if target_length < len(frame_indices):
        raise ValueError(
            f"{label}: aligned_length={target_length} would remove selected frame indices"
        )
    return [*frame_indices, *([frame_indices[-1]] * (target_length - len(frame_indices)))]


def latent_length_for_aligned_rgb(rgb_frames: int, stride: int) -> int:
    frames = parse_positive_int(rgb_frames, "rgb_frames")
    temporal_stride = parse_positive_int(stride, "vae_temporal_stride")
    if (frames - 1) % temporal_stride != 0:
        raise ValueError(f"rgb_frames={frames} is not 1 mod vae_temporal_stride={temporal_stride}")
    return 1 + (frames - 1) // temporal_stride


def rgb_to_latent_shot_groups(shot_groups: list[list[int]], stride: int) -> list[list[int]]:
    latent_groups: list[list[int]] = []
    latent_start = 0
    for shot_start, shot_end in shot_groups:
        latent_len = latent_length_for_aligned_rgb(shot_end - shot_start, stride)
        latent_end = latent_start + latent_len
        latent_groups.append([latent_start, latent_end])
        latent_start = latent_end
    return latent_groups


class FpsAwareTargetFpsPlanner:
    def __init__(
        self,
        *,
        max_rgb_frames: int,
        target_fps: float,
        vae_temporal_stride: int,
    ):
        self.max_rgb_frames = parse_positive_int(max_rgb_frames, "max_rgb_frames")
        self.target_fps = parse_positive_float(target_fps, "target_fps")
        self.vae_temporal_stride = parse_positive_int(vae_temporal_stride, "vae_temporal_stride")

    def build_frame_plan(
        self,
        shot_groups: list[list[int]],
        data_path: str,
        *,
        source_fps: float,
        source_num_frames: int,
    ) -> dict[str, Any]:
        validate_shot_groups(shot_groups, data_path)
        fps = parse_positive_float(source_fps, f"{data_path}: source_fps")
        num_frames = parse_positive_int(source_num_frames, f"{data_path}: source_num_frames")
        source_rgb_frames = shot_groups[-1][-1]
        if source_rgb_frames > num_frames:
            raise ValueError(
                f"{data_path}: shot_groups end at frame {source_rgb_frames}, "
                f"but source_num_frames={num_frames}"
            )

        shot_durations_sec = [(end - start) / fps for start, end in shot_groups]
        source_duration_sec = sum(shot_durations_sec)
        target_lengths = [round_frame_count(duration * self.target_fps) for duration in shot_durations_sec]
        target_lengths, rgb_was_scaled = self._fit_target_lengths(target_lengths, data_path)
        aligned_lengths = [align_rgb_length_to_vae(length, self.vae_temporal_stride) for length in target_lengths]

        frame_indices: list[int] = []
        train_shot_groups: list[list[int]] = []
        cursor = 0
        for (shot_start, shot_end), rgb_len in zip(shot_groups, aligned_lengths):
            frame_indices.extend(self.sample_indices_for_shot(shot_start, shot_end, rgb_len))
            next_cursor = cursor + rgb_len
            train_shot_groups.append([cursor, next_cursor])
            cursor = next_cursor

        latent_shot_groups = rgb_to_latent_shot_groups(train_shot_groups, self.vae_temporal_stride)
        train_latent_frames = latent_shot_groups[-1][-1]
        if cursor > self.max_rgb_frames:
            raise ValueError(f"{data_path}: train_rgb_frames={cursor} exceeds max_rgb_frames={self.max_rgb_frames}")
        if len(frame_indices) != cursor:
            raise ValueError(f"{data_path}: frame index count does not match train_rgb_frames")

        plan = ShotPlan(
            raw_shot_groups=[list(group) for group in shot_groups],
            source_fps=fps,
            source_num_frames=num_frames,
            target_fps=self.target_fps,
            max_rgb_frames=self.max_rgb_frames,
            vae_temporal_stride=self.vae_temporal_stride,
            shot_durations_sec=shot_durations_sec,
            duration_weights=[duration / source_duration_sec for duration in shot_durations_sec],
            target_rgb_lengths=target_lengths,
            aligned_rgb_lengths=aligned_lengths,
            vae_padding_frames=[0 for _ in target_lengths],
            vae_truncated_frames=[target - aligned for target, aligned in zip(target_lengths, aligned_lengths)],
            vae_alignment_policy=VAE_ALIGNMENT_POLICY_LEGACY_FLOOR,
            latent_lengths=[end - start for start, end in latent_shot_groups],
            frame_indices=frame_indices,
            train_shot_groups=train_shot_groups,
            latent_shot_groups=latent_shot_groups,
            prompt_index_by_shot=list(range(len(shot_groups))),
            source_rgb_frames=source_rgb_frames,
            train_rgb_frames=cursor,
            train_latent_frames=train_latent_frames,
            source_duration_sec=source_duration_sec,
            target_duration_sec=sum(target_lengths) / self.target_fps,
            effective_train_fps=cursor / source_duration_sec,
            rgb_was_scaled=rgb_was_scaled,
        )
        return plan.to_dict()

    def build_explicit_frame_plan(
        self,
        *,
        source_shot_segments: list[list[int]],
        train_shot_groups: list[list[int]],
        frame_indices: list[int],
        data_path: str,
        source_fps: float,
        source_num_frames: int,
        vae_alignment_policy: str = VAE_ALIGNMENT_POLICY_LEGACY_FLOOR,
    ) -> dict[str, Any]:
        """Validate a precomputed source-time mapping from source segments.

        ``legacy_floor_truncate_v1`` is kept only to read historical explicit
        manifests.  New data must use ``tail_duplicate_padding_v1`` so that
        VAE compatibility cannot remove temporal content.
        """

        validate_source_shot_segments(source_shot_segments, data_path)
        validate_shot_groups(train_shot_groups, data_path)
        if len(source_shot_segments) != len(train_shot_groups):
            raise ValueError(f"{data_path}: source and train shot counts do not match")
        fps = parse_positive_float(source_fps, f"{data_path}: source_fps")
        num_frames = parse_positive_int(source_num_frames, f"{data_path}: source_num_frames")
        if source_shot_segments[-1][1] > num_frames:
            raise ValueError(
                f"{data_path}: source segments end at frame {source_shot_segments[-1][1]}, "
                f"but source_num_frames={num_frames}"
            )
        if not isinstance(frame_indices, list) or any(type(value) is not int for value in frame_indices):
            raise ValueError(f"{data_path}: source_frame_indices must be a list of ints")

        if vae_alignment_policy not in {
            VAE_ALIGNMENT_POLICY_LEGACY_FLOOR,
            VAE_ALIGNMENT_POLICY_TAIL_PAD,
        }:
            raise ValueError(f"{data_path}: unsupported vae_alignment_policy={vae_alignment_policy!r}")

        durations = [(end - start) / fps for start, end in source_shot_segments]
        target_lengths = [round_frame_count(duration * self.target_fps) for duration in durations]
        if vae_alignment_policy == VAE_ALIGNMENT_POLICY_TAIL_PAD:
            aligned_lengths = [
                pad_rgb_length_to_vae(length, self.vae_temporal_stride)
                for length in target_lengths
            ]
        else:
            aligned_lengths = [
                align_rgb_length_to_vae(length, self.vae_temporal_stride)
                for length in target_lengths
            ]
        actual_lengths = [end - start for start, end in train_shot_groups]
        if actual_lengths != aligned_lengths:
            raise ValueError(
                f"{data_path}: train shot lengths {actual_lengths} do not match deterministic "
                f"{self.target_fps:g}-fps/VAE lengths {aligned_lengths}"
            )
        if sum(actual_lengths) > self.max_rgb_frames:
            raise ValueError(
                f"{data_path}: train_rgb_frames={sum(actual_lengths)} exceeds "
                f"max_rgb_frames={self.max_rgb_frames}"
            )
        expected_indices: list[int] = []
        for (start, end), target_length, aligned_length in zip(
            source_shot_segments, target_lengths, aligned_lengths
        ):
            if vae_alignment_policy == VAE_ALIGNMENT_POLICY_TAIL_PAD:
                selected = self.sample_indices_for_shot(start, end, target_length)
                selected = tail_pad_frame_indices(
                    selected,
                    aligned_length=aligned_length,
                    label=data_path,
                )
            else:
                selected = self.sample_indices_for_shot(start, end, aligned_length)
            expected_indices.extend(selected)
        if frame_indices != expected_indices:
            raise ValueError(f"{data_path}: source_frame_indices do not match deterministic mapping")

        latent_groups = rgb_to_latent_shot_groups(train_shot_groups, self.vae_temporal_stride)
        source_duration = sum(durations)
        plan = ShotPlan(
            raw_shot_groups=[list(group) for group in source_shot_segments],
            source_fps=fps,
            source_num_frames=num_frames,
            target_fps=self.target_fps,
            max_rgb_frames=self.max_rgb_frames,
            vae_temporal_stride=self.vae_temporal_stride,
            shot_durations_sec=durations,
            duration_weights=[duration / source_duration for duration in durations],
            target_rgb_lengths=target_lengths,
            aligned_rgb_lengths=aligned_lengths,
            vae_padding_frames=[aligned - target for target, aligned in zip(target_lengths, aligned_lengths)],
            vae_truncated_frames=[max(0, target - aligned) for target, aligned in zip(target_lengths, aligned_lengths)],
            vae_alignment_policy=vae_alignment_policy,
            latent_lengths=[end - start for start, end in latent_groups],
            frame_indices=list(frame_indices),
            train_shot_groups=[list(group) for group in train_shot_groups],
            latent_shot_groups=latent_groups,
            prompt_index_by_shot=list(range(len(train_shot_groups))),
            source_rgb_frames=sum(end - start for start, end in source_shot_segments),
            train_rgb_frames=sum(actual_lengths),
            train_latent_frames=latent_groups[-1][-1],
            source_duration_sec=source_duration,
            target_duration_sec=sum(target_lengths) / self.target_fps,
            effective_train_fps=sum(actual_lengths) / source_duration,
            rgb_was_scaled=False,
        )
        return plan.to_dict()

    def build_multisource_explicit_frame_plan(
        self,
        *,
        source_video_paths: list[str],
        source_shot_segments: list[list[int]],
        train_shot_groups: list[list[int]],
        frame_indices_by_shot: list[list[int]],
        data_path: str,
        source_fps_by_shot: list[float],
        source_num_frames_by_shot: list[int],
        vae_alignment_policy: str = VAE_ALIGNMENT_POLICY_CLEAN,
    ) -> dict[str, Any]:
        """Validate a no-reencode mapping whose encoded shots span many files.

        Each output shot has exactly one source path, half-open source segment,
        and ordered list of selected source frame indices.  This supports
        datasets such as CI-VID where a semantic sequence is stored as several
        independent clip files, while retaining the same 15fps and per-shot
        unique-sample VAE contract.  ``source_shot_segments`` are the
        boundary-revised spans written by the producer
        (:mod:`mspc.data.shot_boundary_revision`).
        """

        validate_multisource_shot_segments(source_shot_segments, data_path)
        validate_shot_groups(train_shot_groups, data_path)
        shot_count = len(train_shot_groups)
        grouped_values = {
            "source_video_paths": source_video_paths,
            "source_shot_segments": source_shot_segments,
            "frame_indices_by_shot": frame_indices_by_shot,
            "source_fps_by_shot": source_fps_by_shot,
            "source_num_frames_by_shot": source_num_frames_by_shot,
        }
        for label, values in grouped_values.items():
            if not isinstance(values, list) or len(values) != shot_count:
                raise ValueError(
                    f"{data_path}: {label} must contain one entry per train shot ({shot_count})"
                )
        if vae_alignment_policy != VAE_ALIGNMENT_POLICY_CLEAN:
            raise ValueError(
                f"{data_path}: multisource mappings require "
                f"{VAE_ALIGNMENT_POLICY_CLEAN}"
            )

        paths: list[str] = []
        fps_values: list[float] = []
        source_frame_counts: list[int] = []
        target_lengths: list[int] = []
        aligned_lengths: list[int] = []
        expected_indices_by_shot: list[list[int]] = []
        durations: list[float] = []
        for shot_index, (
            path,
            segment,
            actual_indices,
            source_fps,
            source_num_frames,
        ) in enumerate(
            zip(
                source_video_paths,
                source_shot_segments,
                frame_indices_by_shot,
                source_fps_by_shot,
                source_num_frames_by_shot,
                strict=True,
            )
        ):
            if not isinstance(path, str) or not path.strip() or path != path.strip():
                raise ValueError(f"{data_path}: invalid source video path at shot {shot_index}")
            fps = parse_positive_float(source_fps, f"{data_path}: source_fps_by_shot[{shot_index}]")
            num_frames = parse_positive_int(
                source_num_frames,
                f"{data_path}: source_num_frames_by_shot[{shot_index}]",
            )
            start, end = source_shot_segments[shot_index]
            if end > num_frames:
                raise ValueError(
                    f"{data_path}: source segment {segment} exceeds source_num_frames={num_frames} "
                    f"at shot {shot_index}"
                )
            if not isinstance(actual_indices, list) or any(
                type(index) is not int for index in actual_indices
            ):
                raise ValueError(
                    f"{data_path}: frame_indices_by_shot[{shot_index}] must be a list of ints"
                )
            expected, target_length = self.clean_indices_for_shot(start, end, fps)
            duration = (end - start) / fps
            aligned_length = len(expected)
            if actual_indices != expected:
                raise ValueError(
                    f"{data_path}: frame_indices_by_shot[{shot_index}] do not match deterministic mapping"
                )
            paths.append(path)
            fps_values.append(fps)
            source_frame_counts.append(num_frames)
            durations.append(duration)
            target_lengths.append(target_length)
            aligned_lengths.append(aligned_length)
            expected_indices_by_shot.append(expected)

        actual_lengths = [end - start for start, end in train_shot_groups]
        if actual_lengths != aligned_lengths:
            raise ValueError(
                f"{data_path}: train shot lengths {actual_lengths} do not match deterministic "
                f"{self.target_fps:g}-fps/VAE lengths {aligned_lengths}"
            )
        train_rgb_frames = sum(actual_lengths)
        if train_rgb_frames > self.max_rgb_frames:
            raise ValueError(
                f"{data_path}: train_rgb_frames={train_rgb_frames} exceeds "
                f"max_rgb_frames={self.max_rgb_frames}"
            )
        latent_shot_groups = rgb_to_latent_shot_groups(
            train_shot_groups,
            self.vae_temporal_stride,
        )
        source_duration_sec = sum(durations)
        if source_duration_sec <= 0:
            raise AssertionError(f"{data_path}: source duration must be positive")
        unique_fps = {round(value, 9) for value in fps_values}
        return {
            "frame_policy": FRAME_POLICY_FPS_AWARE_TARGET_FPS,
            "mapping_kind": "multisource_explicit_clean_v2",
            "raw_shot_groups": [list(group) for group in source_shot_segments],
            "source_shot_segments": [list(group) for group in source_shot_segments],
            "source_video_paths": paths,
            "source_fps_by_shot": fps_values,
            "source_num_frames_by_shot": source_frame_counts,
            "source_fps": fps_values[0] if len(unique_fps) == 1 else None,
            "source_num_frames": None,
            "target_fps": self.target_fps,
            "max_rgb_frames": self.max_rgb_frames,
            "vae_temporal_stride": self.vae_temporal_stride,
            "shot_durations_sec": durations,
            "duration_weights": [duration / source_duration_sec for duration in durations],
            "target_rgb_lengths": target_lengths,
            "aligned_rgb_lengths": aligned_lengths,
            "vae_padding_frames": [0 for _ in target_lengths],
            "vae_truncated_frames": [target - aligned for target, aligned in zip(target_lengths, aligned_lengths)],
            "vae_alignment_policy": vae_alignment_policy,
            "latent_lengths": [end - start for start, end in latent_shot_groups],
            "frame_indices_by_shot": expected_indices_by_shot,
            "frame_indices": [
                index for shot_indices in expected_indices_by_shot for index in shot_indices
            ],
            "train_shot_groups": [list(group) for group in train_shot_groups],
            "shot_groups": [list(group) for group in train_shot_groups],
            "latent_shot_groups": latent_shot_groups,
            "prompt_index_by_shot": list(range(shot_count)),
            "source_rgb_frames": sum(end - start for start, end in source_shot_segments),
            "train_rgb_frames": train_rgb_frames,
            "train_latent_frames": latent_shot_groups[-1][-1],
            "source_duration_sec": source_duration_sec,
            "target_duration_sec": sum(target_lengths) / self.target_fps,
            "effective_train_fps": train_rgb_frames / source_duration_sec,
            "rgb_was_scaled": False,
        }

    def _fit_target_lengths(self, lengths: list[int], data_path: str) -> tuple[list[int], bool]:
        if len(lengths) > self.max_rgb_frames:
            raise ValueError(
                f"{data_path}: shot_count={len(lengths)} cannot fit max_rgb_frames={self.max_rgb_frames}"
            )
        if sum(lengths) <= self.max_rgb_frames:
            return lengths, False

        low, high = 0.0, 1.0
        best: list[int] | None = None
        for _ in range(64):
            scale = (low + high) / 2
            candidate = [max(1, round_frame_count(length * scale)) for length in lengths]
            if sum(candidate) <= self.max_rgb_frames:
                best = candidate
                low = scale
            else:
                high = scale
        if best is None:
            raise ValueError(f"{data_path}: max_rgb_frames={self.max_rgb_frames} cannot fit target shot groups")
        return best, True

    def clean_indices_for_shot(self, start: int, end: int, source_fps: float) -> tuple[list[int], int]:
        """在已修订的源镜头区间 ``[start, end)`` 内采样到4n+1；不复制补帧。

        镜界混叠帧由生产端的 shot_boundary_revision 按检测切点裁掉，这里不再
        固定丢镜尾一帧。首末帧显式入选，使Screen和历史参考绑定可追溯源索引。
        """
        if end - start < 2:
            raise ValueError("shot too short after boundary revision")
        fps = parse_positive_float(source_fps, "source_fps")
        available = end - start
        target = min(available, round_frame_count(available / fps * self.target_fps))
        aligned = align_rgb_length_to_vae(target, self.vae_temporal_stride)
        if aligned < 2:
            raise ValueError("shot cannot retain distinct first/last frames at VAE stride")
        indices = [start + round(i * (available - 1) / (aligned - 1)) for i in range(aligned)]
        if len(set(indices)) != aligned:
            raise ValueError("clean frame mapping repeated a source frame")
        return indices, target

    def sample_indices_for_shot(self, start: int, end: int, target_frame_count: int) -> list[int]:
        frames = parse_positive_int(target_frame_count, "target_frame_count")
        span = end - start
        indices: list[int] = []
        for offset in range(frames):
            raw_position = start + (offset + 0.5) * span / frames
            index = min(end - 1, max(start, int(math.floor(raw_position))))
            indices.append(index)
        return indices
