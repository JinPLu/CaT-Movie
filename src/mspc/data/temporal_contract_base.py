"""Shared temporal-contract base for multisource source pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


MULTISOURCE_TEMPORAL_ARTIFACT = "multisource_temporal_contract_manifest"
MIN_TRAIN_RGB_FRAMES = 77
MAX_TRAIN_SHOTS = 5


def select_one_contiguous_window(
    shot_count: int,
    *,
    min_shots: int,
    max_shots: int,
    min_rgb_frames: int,
    build_plan: Callable[[int, int], Mapping[str, Any] | None],
    rgb_frame_count: Callable[[Mapping[str, Any]], int],
) -> tuple[list[int], Mapping[str, Any]] | None:
    """Pick one non-overlapping window per sequence (MSM/CI-VID official).

    Preference is more shots, then more aligned RGB frames, then earlier start.
    ``build_plan(start, end)`` must return ``None`` when the half-open shot
    range exceeds the RGB ceiling. Extending a range never shrinks frame count,
    so a ``None`` plan stops further growth from that start.
    """

    if type(shot_count) is not int or shot_count < 0:
        raise ValueError("shot_count must be a non-negative int")
    if type(min_shots) is not int or min_shots < 1:
        raise ValueError("min_shots must be an int >= 1")
    if type(max_shots) is not int or max_shots < min_shots:
        raise ValueError("max_shots must be an int >= min_shots")
    if type(min_rgb_frames) is not int or min_rgb_frames < 1:
        raise ValueError("min_rgb_frames must be a positive int")

    best: tuple[tuple[int, int, int], list[int], Mapping[str, Any]] | None = None
    max_length = min(shot_count, max_shots)
    for start in range(shot_count):
        for length in range(min_shots, max_length + 1):
            end = start + length
            if end > shot_count:
                break
            plan = build_plan(start, end)
            if plan is None:
                break
            n_frames = rgb_frame_count(plan)
            if type(n_frames) is not int or n_frames <= 0:
                raise ValueError("rgb_frame_count must return a positive int")
            if n_frames < min_rgb_frames:
                continue
            score = (length, n_frames, -start)
            if best is None or score > best[0]:
                best = (score, list(range(start, end)), plan)
    if best is None:
        return None
    return best[1], best[2]


@runtime_checkable
class TemporalContract(Protocol):
    """Publish a source-level temporal contract manifest."""

    source_id: str
    artifact_type: str

    def build_manifest(self, *, output_dir: Path, overwrite: bool, **kwargs: Any) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class TemporalContractManifest:
    artifact_type: str
    source_id: str
    manifest_path: Path
    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, path: Path, value: Mapping[str, Any]) -> "TemporalContractManifest":
        artifact_type = value.get("artifact_type")
        source_id = value.get("source_id") or value.get("source")
        if artifact_type != MULTISOURCE_TEMPORAL_ARTIFACT:
            raise ValueError(f"{path}: artifact_type must be {MULTISOURCE_TEMPORAL_ARTIFACT!r}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{path}: source_id must be a non-empty string")
        return cls(
            artifact_type=str(artifact_type),
            source_id=source_id,
            manifest_path=path,
            values=value,
        )
