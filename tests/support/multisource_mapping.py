"""多源镜头映射夹具生成器：几份测试拿它造真实形状的 15 fps 帧映射。

三个函数从已删的 ``mspc.data.multisource_temporal_contract`` 原样搬来（只留测试用到的路径）：
``build_unscaled_multisource_mapping`` 逐镜修订边界再按干净的 15 fps／VAE 对齐造映射，
``select_one_contiguous_multisource_window`` 在一段序列里挑至多一个窗口。
"""

from __future__ import annotations

from typing import Any, Mapping

from mspc.data.frame_planner import (
    FpsAwareTargetFpsPlanner,
    VAE_ALIGNMENT_POLICY_CLEAN,
)
from mspc.data.shot_boundary_revision import (
    BOUNDARY_DETECTOR,
    SHOT_BOUNDARY_REVISION_VERSION,
    ShotBoundaryRejected,
    revise_shot_boundary,
)
from mspc.data.temporal_contract_base import (
    MAX_TRAIN_SHOTS,
    MIN_TRAIN_RGB_FRAMES,
    select_one_contiguous_window,
)


def _train_groups(lengths: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    cursor = 0
    for length in lengths:
        groups.append([cursor, cursor + length])
        cursor += length
    return groups


def build_unscaled_multisource_mapping(
    *,
    source_video_paths: list[str],
    source_shot_segments: list[list[int]],
    source_fps_by_shot: list[float],
    source_num_frames_by_shot: list[int],
    shot_cut_evidence: list[Mapping[str, Any]],
    planner: FpsAwareTargetFpsPlanner,
    label: str,
) -> dict[str, Any] | None:
    """Revise each source shot's boundary, then build the clean 15fps/VAE map.

    The revision (:mod:`mspc.data.shot_boundary_revision`) is the only place
    that decides which source frames belong to a shot.  A cut inside a shot or
    a shot too short to plan after trimming raises :class:`ShotBoundaryRejected`;
    ``None`` still means the RGB ceiling is exceeded.
    """

    expected_count = len(source_shot_segments)
    values = {
        "source_video_paths": source_video_paths,
        "source_fps_by_shot": source_fps_by_shot,
        "source_num_frames_by_shot": source_num_frames_by_shot,
        "shot_cut_evidence": shot_cut_evidence,
    }
    for name, value in values.items():
        if not isinstance(value, list) or len(value) != expected_count:
            raise ValueError(f"{label}: {name} must contain one entry per source shot")

    revisions: list[dict[str, Any]] = []
    for shot_index, (path, segment, fps, evidence) in enumerate(
        zip(source_video_paths, source_shot_segments, source_fps_by_shot, shot_cut_evidence, strict=True)
    ):
        if evidence["source_video_path"] != path or evidence["source_shot_segment"] != list(segment):
            raise ValueError(f"{label}: shot {shot_index} cut evidence was measured on a different span")
        if evidence["source_fps"] != fps:
            raise ValueError(f"{label}: shot {shot_index} cut evidence fps differs from source metadata")
        try:
            revisions.append(revise_shot_boundary(evidence))
        except ShotBoundaryRejected as rejection:
            raise ShotBoundaryRejected(rejection.reason, f"{label}: shot {shot_index}: {rejection}") from None
    revised_segments = [revision["revised_shot_segment"] for revision in revisions]

    frame_indices_by_shot: list[list[int]] = []
    for shot_index, ((start, end), fps) in enumerate(zip(revised_segments, source_fps_by_shot, strict=True)):
        try:
            indices, _target = planner.clean_indices_for_shot(start, end, fps)
        except ValueError as error:
            raise ShotBoundaryRejected(
                "shot_too_short", f"{label}: shot {shot_index} revised to [{start}, {end}): {error}"
            ) from None
        frame_indices_by_shot.append(indices)
    aligned_lengths = [len(indices) for indices in frame_indices_by_shot]
    if sum(aligned_lengths) > planner.max_rgb_frames:
        return None
    plan = planner.build_multisource_explicit_frame_plan(
        source_video_paths=source_video_paths,
        source_shot_segments=revised_segments,
        train_shot_groups=_train_groups(aligned_lengths),
        frame_indices_by_shot=frame_indices_by_shot,
        data_path=label,
        source_fps_by_shot=source_fps_by_shot,
        source_num_frames_by_shot=source_num_frames_by_shot,
        vae_alignment_policy=VAE_ALIGNMENT_POLICY_CLEAN,
    )
    plan["shot_boundary_revision"] = {
        "version": SHOT_BOUNDARY_REVISION_VERSION,
        "boundary_detector": BOUNDARY_DETECTOR,
        "unrevised_source_shot_segments": [list(segment) for segment in source_shot_segments],
        **{
            key: [revision[key] for revision in revisions]
            for key in ("trimmed_head_frames", "trimmed_tail_frames", "transnet_cuts")
        },
    }
    return plan


def select_one_contiguous_multisource_window(
    *,
    source_video_paths: list[str],
    source_shot_segments: list[list[int]],
    source_fps_by_shot: list[float],
    source_num_frames_by_shot: list[int],
    shot_cut_evidence: list[Mapping[str, Any]],
    planner: FpsAwareTargetFpsPlanner,
    min_shots: int,
    label: str,
    max_shots: int = MAX_TRAIN_SHOTS,
    min_rgb_frames: int = MIN_TRAIN_RGB_FRAMES,
) -> list[tuple[list[int], dict[str, Any]]]:
    """CI-VID/Cine adapter: at most one non-overlapping window per sequence.

    Returns a 0-or-1 list so the candidate writer can iterate. Selection is
    ``select_one_contiguous_window`` (MSM one-row-one-sample), not a sliding
    maximal-overlap scan.  A window containing a boundary-rejected shot is not
    a candidate, and growing it cannot remove that shot, so it stops growth
    the same way the RGB ceiling does.
    """

    def build_plan(start: int, end: int) -> dict[str, Any] | None:
        try:
            return build_unscaled_multisource_mapping(
                source_video_paths=source_video_paths[start:end],
                source_shot_segments=source_shot_segments[start:end],
                source_fps_by_shot=source_fps_by_shot[start:end],
                source_num_frames_by_shot=source_num_frames_by_shot[start:end],
                shot_cut_evidence=shot_cut_evidence[start:end],
                planner=planner,
                label=f"{label}:shots[{start}:{end}]",
            )
        except ShotBoundaryRejected:
            return None

    selected = select_one_contiguous_window(
        len(source_shot_segments),
        min_shots=min_shots,
        max_shots=max_shots,
        min_rgb_frames=min_rgb_frames,
        build_plan=build_plan,
        rgb_frame_count=lambda plan: int(plan["train_rgb_frames"]),
    )
    if selected is None:
        return []
    indices, plan = selected
    return [(indices, dict(plan))]
