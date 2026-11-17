"""Coarse media-ready filter for CI-VID, Shot2Story, and Cine partial.

Mechanical contract before bulk Qwen Flash annotation: light decode + 2–5 shots
+ attribute-complete mapping columns.  Missing media is dropped (not hunted).
Caption keep (312,311) is an input ceiling, not train N.  Semantic lane
admission still requires downstream annotation and human contact-sheet review.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mspc.data import hard_filter_keep_windows as keep_windows
from mspc.data.multishot import (
    CAPTION_PATH_COLUMN,
    HISTORICAL_CAPTION_PATH_COLUMN,
    MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
    is_multisource_mapping_schema,
    msm_caption_payload,
    parse_multisource_source_shot_segments,
    parse_shot_groups,
    parse_source_frame_indices,
    parse_source_frame_indices_by_shot,
    parse_source_video_paths,
)
from mspc.data.temporal_contract_base import MAX_TRAIN_SHOTS, MIN_TRAIN_RGB_FRAMES
from mspc.data.strict_json import load_strict_json_file


SCHEMA_VERSION = "multishot_media_ready_v1"
LANE_GENERIC_MULTISHOT = "generic_multishot"
LANE_EVENTFUL_TRANSITION = "eventful_transition"
LANE_SUBJECT_CONTEXT = "subject_context"
MEDIA_READY_LANES = frozenset(
    {
        LANE_GENERIC_MULTISHOT,
        LANE_EVENTFUL_TRANSITION,
        LANE_SUBJECT_CONTEXT,
    }
)
SUPPORTED_DATASETS = frozenset({"shot2story", "ci_vid", "cine250k"})
MIN_SHOT_COUNT = 2
MAX_SHOT_COUNT = MAX_TRAIN_SHOTS
MIN_SHORT_SIDE = 480
HQ_DATASETS = frozenset({"ci_vid", "cine250k"})
TRAINING_ADMISSION_MECHANICAL_HOLDOUT = "mechanical_holdout"
CAPTION_PROVENANCE_SELECTED_SHOTS = "selected_shot_captions_rejoined_v1"
# Vision-lite MSM two-stage (global frames + per-shot text rewrite). Mechanical
# join above is fallback/对照, not the new bulk default.
CAPTION_PROVENANCE_MSM_VISION_LITE = "msm_vision_lite_two_stage_v1"
# MSM local/per-shot vision upgrade after coreference fail. Smoke only; not bulk default.
CAPTION_PROVENANCE_MSM_LOCAL_VISION = "msm_vision_local_stage_v1"


@dataclass(frozen=True)
class MediaReadyProbeSpec:
    """Mechanical decode probe contract shared by coarse-filter owners."""

    decoder: str
    coverage: str
    frames: str
    semantic_filtering: bool
    target_fps: float
    max_rgb_frames: int
    vae_temporal_stride: int


@dataclass(frozen=True)
class MediaReadyConfig:
    schema_version: str
    target_fps: float
    max_rgb_frames: int
    vae_temporal_stride: int
    min_shots: int
    max_shots: int
    datasets: frozenset[str]
    lanes: frozenset[str]
    min_rgb_frames: int | None
    min_short_side: int | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MediaReadyConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("media-ready config must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported media-ready schema_version={schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        target_fps = payload.get("target_fps")
        max_rgb_frames = payload.get("max_rgb_frames")
        vae_temporal_stride = payload.get("vae_temporal_stride")
        min_shots = payload.get("min_shots", MIN_SHOT_COUNT)
        max_shots = payload.get("max_shots", MAX_SHOT_COUNT)
        min_rgb_frames = payload.get("min_rgb_frames")
        min_short_side = payload.get("min_short_side")
        if type(target_fps) not in (int, float) or target_fps <= 0:
            raise ValueError("target_fps must be a positive number")
        for name, value in (
            ("max_rgb_frames", max_rgb_frames),
            ("vae_temporal_stride", vae_temporal_stride),
            ("min_shots", min_shots),
            ("max_shots", max_shots),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if min_shots < MIN_SHOT_COUNT or max_shots > MAX_SHOT_COUNT or min_shots > max_shots:
            raise ValueError(
                f"shot bounds must satisfy {MIN_SHOT_COUNT} <= min_shots <= max_shots <= {MAX_SHOT_COUNT}"
            )
        if min_rgb_frames is not None and (type(min_rgb_frames) is not int or min_rgb_frames < 1):
            raise ValueError("min_rgb_frames must be a positive int when set")
        if min_short_side is not None and (type(min_short_side) is not int or min_short_side < 1):
            raise ValueError("min_short_side must be a positive int when set")
        datasets = payload.get("datasets")
        lanes = payload.get("lanes")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("datasets must be a non-empty list")
        if not isinstance(lanes, list) or not lanes:
            raise ValueError("lanes must be a non-empty list")
        normalized_datasets = frozenset(_require_dataset(item) for item in datasets)
        normalized_lanes = frozenset(_require_lane(item) for item in lanes)
        return cls(
            schema_version=schema_version,
            target_fps=float(target_fps),
            max_rgb_frames=max_rgb_frames,
            vae_temporal_stride=vae_temporal_stride,
            min_shots=min_shots,
            max_shots=max_shots,
            datasets=normalized_datasets,
            lanes=normalized_lanes,
            min_rgb_frames=min_rgb_frames,
            min_short_side=min_short_side,
        )

    @classmethod
    def load(cls, path: str | Path) -> MediaReadyConfig:
        payload = load_strict_json_file(str(path), label=f"media-ready config {path}")
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: media-ready config must be an object")
        return cls.from_mapping(payload)


def _require_dataset(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("dataset entries must be non-empty trimmed strings")
    if value not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset={value!r}")
    return value


def _require_lane(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("lane entries must be non-empty trimmed strings")
    if value not in MEDIA_READY_LANES:
        raise ValueError(f"unsupported lane={value!r}")
    return value


def build_media_ready_probe_spec(config: MediaReadyConfig) -> MediaReadyProbeSpec:
    return MediaReadyProbeSpec(
        decoder="decord.VideoReader",
        coverage="every_manifest_row",
        frames="first_middle_last_per_shot_plus_training_plan_quartiles",
        semantic_filtering=False,
        target_fps=config.target_fps,
        max_rgb_frames=config.max_rgb_frames,
        vae_temporal_stride=config.vae_temporal_stride,
    )


def classify_media_ready_lane(
    *,
    dataset: str,
    event_state_candidate: bool = False,
    generic_multishot_candidate: bool = True,
) -> str:
    """Return the primary coarse lane before Qwen Flash subject/context writing."""

    dataset = _require_dataset(dataset)
    if event_state_candidate and dataset in {"ci_vid", "shot2story"}:
        return LANE_EVENTFUL_TRANSITION
    if generic_multishot_candidate:
        return LANE_GENERIC_MULTISHOT
    return LANE_SUBJECT_CONTEXT



class MediaReadyAttributeError(ValueError):
    """一行媒体属性不合规，``reason_code`` 说明是哪一条不合规。

    此前调用方是按 ``"shot_count=" in str(error)`` / ``"below min_rgb_frames" in ...``
    来把它归进 ``rejected.jsonl`` 的四个 reason 之一的。判据落在异常消息的字面上：
    改一句话，分母解释就静默从 ``shot_count_out_of_bounds`` 滑到
    ``attribute_incomplete``——accepted 集合不变，但账本错了，而账本正是用来解释分母的。
    """

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def validate_media_ready_row_attrs(
    row: Mapping[str, Any],
    *,
    config: MediaReadyConfig,
    dataset: str,
    record_id: str,
) -> dict[str, Any]:
    """Mechanically validate shot count and mapping columns for one CSV row."""

    dataset = _require_dataset(dataset)
    if dataset not in config.datasets:
        raise ValueError(f"{record_id}: dataset {dataset!r} is outside media-ready config")
    mapping_schema = row.get("mapping_schema") or row.get("schema") or ""
    if isinstance(mapping_schema, str) and mapping_schema and not (
        mapping_schema == MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA
        or is_multisource_mapping_schema(mapping_schema)
    ):
        raise ValueError(f"{record_id}: unsupported mapping_schema={mapping_schema!r}")

    shot_groups = parse_shot_groups(row.get("shot_groups"), record_id)
    shot_count = len(shot_groups)
    if not config.min_shots <= shot_count <= config.max_shots:
        raise MediaReadyAttributeError(
            f"{record_id}: shot_count={shot_count} outside "
            f"[{config.min_shots}, {config.max_shots}]",
            reason_code="shot_count_out_of_bounds",
        )
    n_frames = int(shot_groups[-1][1])
    if config.min_rgb_frames is not None and n_frames < config.min_rgb_frames:
        raise MediaReadyAttributeError(
            f"{record_id}: aligned_rgb_frames={n_frames} below min_rgb_frames={config.min_rgb_frames}",
            reason_code="frame_count_below_min",
        )
    if n_frames > config.max_rgb_frames:
        raise MediaReadyAttributeError(
            f"{record_id}: aligned_rgb_frames={n_frames} exceeds max_rgb_frames={config.max_rgb_frames}",
            reason_code="frame_count_above_max",
        )

    if mapping_schema == MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA:
        paths = parse_source_video_paths(row.get("source_video_paths"), record_id)
        segments = parse_multisource_source_shot_segments(
            row.get("source_shot_segments"),
            record_id,
        )
        indices_by_shot = parse_source_frame_indices_by_shot(
            row.get("source_frame_indices_by_shot"),
            record_id,
        )
        if not (len(paths) == len(segments) == len(indices_by_shot) == shot_count):
            raise ValueError(f"{record_id}: multisource shot columns do not align")
        missing_paths = [
            path
            for path in paths
            if not Path(path).is_file() or Path(path).stat().st_size <= 0
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"{record_id}: missing or empty multisource clip(s): {missing_paths}"
            )
        mapping_kind = "multisource"
    else:
        video = row.get("video") or row.get("multishot_video")
        if not isinstance(video, str) or not video.strip() or video != video.strip():
            raise ValueError(f"{record_id}: missing single-file video column")
        path = Path(video)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{record_id}: missing or empty video: {path}")
        mapping_kind = "single_file"

    return {
        "record_id": record_id,
        "dataset": dataset,
        "mapping_kind": mapping_kind,
        "shot_count": shot_count,
        "lane": classify_media_ready_lane(dataset=dataset),
    }


ACCEPTED_CSV_COLUMNS = (
    "case_id",
    "dataset",
    "lane",
    "mapping_kind",
    "mapping_schema",
    "shot_count",
    "video",
    "shot_groups",
    "source_frame_indices",
    "source_video_paths",
    "source_shot_segments",
    "source_frame_indices_by_shot",
    "state_candidate",
    "generic_multishot_candidate",
    "transition_candidate",
    "event_status",
    CAPTION_PATH_COLUMN,
    "caption_provenance",
    "training_allowed",
    "training_admission",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_jsonl_objects(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, row


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_keep_rows_full(keep_dir: Path) -> list[dict[str, Any]]:
    """Load caption-keep rows with lane flags.  Count is contract input, not train N."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in keep_windows.KEEP_DATASETS:
        path = _require_file(keep_dir / f"{dataset}.jsonl", f"keep {dataset}")
        for line_number, row in _iter_jsonl_objects(path):
            record_id = keep_windows._require_string(
                row.get("record_id"), f"{path}:{line_number}.record_id"
            )
            row_dataset = keep_windows._require_string(
                row.get("dataset"), f"{record_id}.dataset"
            )
            if row_dataset != dataset:
                raise ValueError(f"{record_id}: dataset {row_dataset!r} != {dataset!r}")
            if record_id in seen:
                raise ValueError(f"duplicate keep record_id={record_id}")
            seen.add(record_id)
            shot_count = row.get("shot_count")
            if type(shot_count) is not int or shot_count <= 0:
                raise ValueError(f"{record_id}: shot_count must be a positive int")
            rows.append(
                {
                    "record_id": record_id,
                    "dataset": dataset,
                    "shot_count": shot_count,
                    "state_candidate": bool(row.get("state_candidate")),
                    "generic_multishot_candidate": bool(
                        row.get("generic_multishot_candidate")
                    ),
                    "transition_candidate": bool(row.get("transition_candidate")),
                    "event_status": row.get("event_status"),
                    "shot_captions": row.get("shot_captions"),
                    "whole_caption": row.get("whole_caption"),
                }
            )
    return rows


def _index_cine_ledgers_soft(
    pool_index: Mapping[str, Mapping[str, Any]],
    needed_pool_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Join cine keep IDs to ledger csv_rows; missing IDs become drop reasons."""

    needed_by_manifest: dict[str, dict[str, str]] = {}
    missing_reasons: dict[str, str] = {}
    for pool_id in needed_pool_ids:
        pool = pool_index.get(pool_id)
        if pool is None:
            missing_reasons[pool_id] = "missing_join_key"
            continue
        manifest = pool["temporal_contract_manifest"]
        needed_by_manifest.setdefault(manifest, {})[pool["source_case_id"]] = pool_id

    found: dict[str, dict[str, Any]] = {}
    for manifest, cases in needed_by_manifest.items():
        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            for case_id, pool_id in cases.items():
                missing_reasons[pool_id] = "cine_missing_temporal_manifest"
            continue
        ledger_path = manifest_path.parent / "mapping-ledger.jsonl"
        if not ledger_path.is_file():
            for case_id, pool_id in cases.items():
                missing_reasons[pool_id] = "cine_missing_mapping_ledger"
            continue
        remaining = set(cases)
        for _line_number, row in _iter_jsonl_objects(ledger_path):
            if not remaining:
                break
            if row.get("status") != "full_sequence_candidate":
                continue
            csv_row = row.get("csv_row")
            if not isinstance(csv_row, dict):
                continue
            case_id = csv_row.get("case_id")
            if case_id not in remaining:
                continue
            pool_id = cases[case_id]
            fingerprint = csv_row.get("mapping_fingerprint")
            expected = pool_index[pool_id]["mapping_fingerprint"]
            if fingerprint != expected:
                missing_reasons[pool_id] = "cine_mapping_fingerprint_mismatch"
                remaining.remove(case_id)
                continue
            found[pool_id] = {
                "multishot_video": csv_row.get("multishot_video"),
                "shot_groups": csv_row.get("shot_groups"),
                "source_frame_indices": csv_row.get("source_frame_indices"),
                "source_frame_indices_by_shot": csv_row.get(
                    "source_frame_indices_by_shot"
                ),
                "source_video_paths": csv_row.get("source_video_paths"),
                "source_shot_segments": csv_row.get("source_shot_segments"),
                HISTORICAL_CAPTION_PATH_COLUMN: csv_row.get(HISTORICAL_CAPTION_PATH_COLUMN),
            }
            remaining.remove(case_id)
        for case_id in remaining:
            pool_id = cases[case_id]
            missing_reasons.setdefault(pool_id, "cine_missing_ledger_csv_row")
    return found, missing_reasons


def _probe_frame_indices(frame_indices: Sequence[int], source_num_frames: int) -> list[int]:
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")
    if source_num_frames <= 0:
        raise ValueError("source_num_frames must be positive")
    count = len(frame_indices)
    selected = {
        int(frame_indices[0]),
        int(frame_indices[count // 2]),
        int(frame_indices[-1]),
    }
    invalid = sorted(index for index in selected if index < 0 or index >= source_num_frames)
    if invalid:
        raise ValueError(
            f"probe indices exceed decodable range 0..{source_num_frames - 1}: {invalid[:8]}"
        )
    return sorted(selected)


def _decode_probe(path: str, frame_indices: list[int]) -> None:
    import decord

    frames = (
        decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        .get_batch(frame_indices)
        .asnumpy()
    )
    if frames.shape[0] != len(frame_indices):
        raise RuntimeError(f"decoded {frames.shape[0]} frames, expected {len(frame_indices)}")


def _light_decode_one_path(
    path: str,
    frame_indices: Sequence[int],
    *,
    min_short_side: int | None,
    record_id: str,
) -> None:
    import decord

    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    if min_short_side is not None:
        frame = reader[0]
        short_side = min(int(frame.shape[0]), int(frame.shape[1]))
        if short_side < min_short_side:
            raise ValueError(f"short_side_below_min:{short_side}")
    probe = _probe_frame_indices(frame_indices, len(reader))
    frames = reader.get_batch(probe).asnumpy()
    if frames.shape[0] != len(probe):
        raise RuntimeError(
            f"{record_id}: decoded {frames.shape[0]} frames, expected {len(probe)}"
        )


def light_decode_compiled_row(
    compiled: Mapping[str, str],
    *,
    min_short_side: int | None = None,
) -> None:
    """Light decord probe: first/middle/last planned frames per shot file."""

    if min_short_side is not None and (type(min_short_side) is not int or min_short_side < 1):
        raise ValueError("min_short_side must be a positive int when set")
    record_id = compiled["case_id"]
    mapping_schema = compiled.get("mapping_schema") or ""
    if mapping_schema == MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA or is_multisource_mapping_schema(
        mapping_schema
    ):
        paths = parse_source_video_paths(compiled["source_video_paths"], record_id)
        indices_by_shot = parse_source_frame_indices_by_shot(
            compiled["source_frame_indices_by_shot"],
            record_id,
        )
        if len(paths) != len(indices_by_shot):
            raise ValueError(f"{record_id}: multisource path/index length mismatch")
        for path, shot_indices in zip(paths, indices_by_shot, strict=True):
            _light_decode_one_path(
                path,
                shot_indices,
                min_short_side=min_short_side,
                record_id=record_id,
            )
        return

    video = compiled["video"]
    frames = parse_source_frame_indices(compiled["source_frame_indices"], record_id)
    _light_decode_one_path(
        video,
        frames,
        min_short_side=min_short_side,
        record_id=record_id,
    )


def probe_decode_media_ready_row(
    payload: tuple[Mapping[str, str], int | None],
) -> tuple[bool, str]:
    """Decode half of the coarse-wash probe, as a picklable module-level worker.

    Takes only primitives so it can run under ProcessPoolExecutor: decord decode is
    CPU-bound Python-level work, so threads are capped by the GIL.  Returns
    ``(ok, reject_reason)``; a decode failure is a typed reject, while anything
    outside the decode call propagates so a bug crashes the wash.
    """

    compiled, min_short_side = payload
    try:
        if min_short_side is None:
            light_decode_compiled_row(compiled)
        else:
            light_decode_compiled_row(compiled, min_short_side=min_short_side)
    except ValueError as error:
        if str(error).startswith("short_side_below_min"):
            return False, "short_side_below_min"
        return False, f"decode:{type(error).__name__}"
    except Exception as error:  # noqa: BLE001 — decode backends raise varied types
        return False, f"decode:{type(error).__name__}"
    return True, ""


PROBE_STALL_SECONDS = 180


def _probe_chunksize(n_payloads: int, workers: int) -> int:
    """Batch enough rows per IPC hop to amortize dispatch without starving the tail."""

    return max(1, min(8, n_payloads // (workers * 16) or 1))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _accepted_row_from_parts(
    *,
    compiled: Mapping[str, str],
    keep: Mapping[str, Any],
    lane: str,
    mapping_kind: str,
    shot_count: int,
    caption_path: str,
    caption_provenance: str,
) -> dict[str, str]:
    if not isinstance(caption_path, str) or caption_path != caption_path.strip():
        raise ValueError(f"{compiled['case_id']}: caption_path must be a trimmed string")
    if caption_path == "" and caption_provenance not in {
        "keep_shot_captions_absent",
        CAPTION_PROVENANCE_SELECTED_SHOTS,
    }:
        raise ValueError(
            f"{compiled['case_id']}: empty caption_path is only allowed before sidecar "
            "write (selected_shot_captions_rejoined_v1) or when keep captions are absent"
        )
    if not isinstance(caption_provenance, str) or not caption_provenance.strip():
        raise ValueError(f"{compiled['case_id']}: caption_provenance must be a non-empty string")
    return {
        "case_id": compiled["case_id"],
        "dataset": compiled["dataset"],
        "lane": lane,
        "mapping_kind": mapping_kind,
        "mapping_schema": compiled.get("mapping_schema") or "",
        "shot_count": str(shot_count),
        "video": compiled["video"],
        "shot_groups": compiled["shot_groups"],
        "source_frame_indices": compiled.get("source_frame_indices") or "",
        "source_video_paths": compiled.get("source_video_paths") or "",
        "source_shot_segments": compiled.get("source_shot_segments") or "",
        "source_frame_indices_by_shot": compiled.get("source_frame_indices_by_shot") or "",
        "state_candidate": _bool_text(bool(keep["state_candidate"])),
        "generic_multishot_candidate": _bool_text(bool(keep["generic_multishot_candidate"])),
        "transition_candidate": _bool_text(bool(keep["transition_candidate"])),
        "event_status": "" if keep.get("event_status") is None else str(keep["event_status"]),
        CAPTION_PATH_COLUMN: caption_path,
        "caption_provenance": caption_provenance,
        "training_allowed": "false",
        "training_admission": TRAINING_ADMISSION_MECHANICAL_HOLDOUT,
    }


@dataclass(frozen=True)
class MediaReadyWashPaths:
    keep_dir: Path
    shot2story_window_candidates: Path
    ci_vid_window_candidates: Path
    cine_pool_dirs: tuple[Path, ...]
    output_dir: Path
    ci_vid_mapping_ledger: Path | None = None
    ci_vid_full_sequence_csv: Path | None = None


def load_media_ready_wash_paths(payload: Mapping[str, Any]) -> MediaReadyWashPaths:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("wash config requires object field inputs")
    keep_dir = inputs.get("keep_dir")
    s2s = inputs.get("shot2story_window_candidates")
    ci = inputs.get("ci_vid_window_candidates")
    cine_dirs = inputs.get("cine_pool_dirs")
    output_dir = payload.get("output_dir")
    if not isinstance(keep_dir, str) or not keep_dir.strip():
        raise ValueError("inputs.keep_dir must be a non-empty string")
    if not isinstance(s2s, str) or not s2s.strip():
        raise ValueError("inputs.shot2story_window_candidates must be a non-empty string")
    if not isinstance(ci, str) or not ci.strip():
        raise ValueError("inputs.ci_vid_window_candidates must be a non-empty string")
    if not isinstance(cine_dirs, list) or not cine_dirs:
        raise ValueError("inputs.cine_pool_dirs must be a non-empty list")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string")
    ledger = inputs.get("ci_vid_mapping_ledger")
    full_csv = inputs.get("ci_vid_full_sequence_csv")
    if ledger is not None and (not isinstance(ledger, str) or not ledger.strip()):
        raise ValueError("inputs.ci_vid_mapping_ledger must be a non-empty string when set")
    if full_csv is not None and (not isinstance(full_csv, str) or not full_csv.strip()):
        raise ValueError("inputs.ci_vid_full_sequence_csv must be a non-empty string when set")
    normalized_cine: list[Path] = []
    for item in cine_dirs:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError("cine_pool_dirs entries must be non-empty trimmed strings")
        normalized_cine.append(Path(item))
    return MediaReadyWashPaths(
        keep_dir=Path(keep_dir),
        shot2story_window_candidates=Path(s2s),
        ci_vid_window_candidates=Path(ci),
        cine_pool_dirs=tuple(normalized_cine),
        output_dir=Path(output_dir),
        ci_vid_mapping_ledger=None if ledger is None else Path(ledger),
        ci_vid_full_sequence_csv=None if full_csv is None else Path(full_csv),
    )


def _publish_dir(staged: Path, output_dir: Path, *, overwrite: bool) -> None:
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, output_dir)


def _shot_texts_from_keep(keep: Mapping[str, Any], *, shot_count: int) -> list[str] | None:
    raw = keep.get("shot_captions")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != shot_count:
        raise ValueError(
            f"{keep.get('record_id')}: shot_captions length must equal shot_count={shot_count}"
        )
    texts: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(
                f"{keep.get('record_id')}: shot_captions[{index}] must be a non-empty trimmed string"
            )
        texts.append(item)
    return texts


def write_msm_caption_sidecar(
    *,
    directory: Path,
    case_id: str,
    shot_captions: list[str],
) -> Path:
    payload = msm_caption_payload(shot_captions)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case_id.replace('/', '_')}.json"
    if path.exists():
        raise FileExistsError(f"caption sidecar already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _video_short_side(path: str) -> int:
    import decord

    frame = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)[0]
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{path}: decoded frame has non-positive size {width}x{height}")
    return min(height, width)


def compiled_short_side(compiled: Mapping[str, str]) -> int:
    record_id = compiled["case_id"]
    mapping_schema = compiled.get("mapping_schema") or ""
    if mapping_schema == MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA or is_multisource_mapping_schema(
        mapping_schema
    ):
        paths = parse_source_video_paths(compiled["source_video_paths"], record_id)
        sides = [_video_short_side(path) for path in paths]
        return min(sides)
    return _video_short_side(compiled["video"])


def run_media_ready_coarse_wash(
    *,
    config: MediaReadyConfig,
    paths: MediaReadyWashPaths,
    workers: int,
    overwrite: bool,
    expected_keep_rows: int | None = keep_windows.HARD_FILTER_KEEP_N,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Join caption keep → windows, drop missing media, light-decode, emit lane lists."""

    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive int")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("progress_every must be a positive int")

    print(
        json.dumps({"phase": "load_keep", "keep_dir": str(paths.keep_dir)}, ensure_ascii=False),
        flush=True,
    )
    keep_rows = _load_keep_rows_full(paths.keep_dir)
    if expected_keep_rows is not None and len(keep_rows) != expected_keep_rows:
        raise ValueError(
            f"keep dir has {len(keep_rows)} rows, expected {expected_keep_rows} "
            "(caption keep ceiling; not train N)"
        )
    print(
        json.dumps({"phase": "join_start", "keep_n": len(keep_rows)}, ensure_ascii=False),
        flush=True,
    )

    s2s_index = (
        keep_windows._index_window_candidates(paths.shot2story_window_candidates)
        if "shot2story" in config.datasets
        else {}
    )
    if paths.ci_vid_window_candidates.is_file() and paths.ci_vid_window_candidates.suffix == ".jsonl":
        ci_index = keep_windows._index_window_candidates(paths.ci_vid_window_candidates)
    else:
        ci_index = {}
        if "ci_vid" in config.datasets and paths.ci_vid_mapping_ledger is None:
            raise FileNotFoundError(
                "ci_vid window-candidates is missing and inputs.ci_vid_mapping_ledger is not set"
            )
    ci_full_index = (
        keep_windows._index_ci_vid_full_sequence_ledger(paths.ci_vid_mapping_ledger)
        if paths.ci_vid_mapping_ledger is not None
        else {}
    )
    cine_pools = keep_windows._index_cine_pools(paths.cine_pool_dirs)
    cine_needed = {row["record_id"] for row in keep_rows if row["dataset"] == "cine250k"}
    cine_ledgers, cine_missing = _index_cine_ledgers_soft(cine_pools, cine_needed)
    for pool_id in sorted(cine_needed - set(cine_pools) - set(cine_missing)):
        cine_missing[pool_id] = "missing_join_key"
    print(
        json.dumps(
            {
                "phase": "join_indexes_ready",
                "ci_full_index_n": len(ci_full_index),
                "cine_pool_n": len(cine_pools),
                "cine_ledger_n": len(cine_ledgers),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    rejected_by_reason: Counter[str] = Counter()
    rejected_records: list[dict[str, str]] = []
    candidates: list[tuple[dict[str, Any], dict[str, str], str, str, int]] = []

    def _reject(keep: Mapping[str, Any], reason: str) -> None:
        rejected_by_reason[reason] += 1
        rejected_records.append(
            {
                "record_id": str(keep["record_id"]),
                "dataset": str(keep["dataset"]),
                "reason": reason,
            }
        )

    for keep in keep_rows:
        record_id = keep["record_id"]
        dataset = keep["dataset"]
        if dataset not in config.datasets:
            _reject(keep, "dataset_not_whitelisted")
            continue
        if not config.min_shots <= keep["shot_count"] <= config.max_shots:
            _reject(keep, "shot_count_out_of_bounds")
            continue
        if dataset == "shot2story":
            compiled = keep_windows._compile_shot2story(record_id, s2s_index.get(record_id))
        elif dataset == "ci_vid":
            window = ci_index.get(record_id)
            if window is None and record_id in ci_full_index:
                window = keep_windows._window_from_full_sequence_csv_row(ci_full_index[record_id])
            compiled = keep_windows._compile_ci_vid(record_id, window)
        elif dataset == "cine250k":
            if record_id in cine_missing and record_id not in cine_ledgers:
                _reject(keep, cine_missing[record_id])
                continue
            compiled = keep_windows._compile_cine(record_id, cine_ledgers.get(record_id))
        else:
            raise ValueError(f"{record_id}: unsupported dataset {dataset!r}")
        if isinstance(compiled, str):
            _reject(keep, compiled)
            continue

        try:
            attr = validate_media_ready_row_attrs(
                compiled,
                config=config,
                dataset=dataset,
                record_id=record_id,
            )
        except FileNotFoundError:
            _reject(keep, "missing_or_empty_media")
            continue
        except MediaReadyAttributeError as error:
            _reject(keep, error.reason_code)
            continue
        except ValueError:
            # 其余属性问题没有细分 code，仍按整体不完整记账。
            _reject(keep, "attribute_incomplete")
            continue

        lane = classify_media_ready_lane(
            dataset=dataset,
            event_state_candidate=bool(keep["state_candidate"]),
            generic_multishot_candidate=bool(keep["generic_multishot_candidate"]),
        )
        if lane not in config.lanes:
            raise ValueError(f"{record_id}: lane {lane!r} outside wash config")
        candidates.append(
            (keep, compiled, lane, str(attr["mapping_kind"]), int(attr["shot_count"]))
        )

    def _finish_probe(
        item: tuple[dict[str, Any], dict[str, str], str, str, int],
        decode_result: tuple[bool, str],
    ) -> tuple[bool, str, dict[str, str] | None]:
        keep, compiled, lane, mapping_kind, shot_count = item
        ok, reason = decode_result
        if not ok:
            return False, reason, None
        shot_texts = _shot_texts_from_keep(keep, shot_count=shot_count)
        provenance = (
            CAPTION_PROVENANCE_SELECTED_SHOTS
            if shot_texts is not None
            else "keep_shot_captions_absent"
        )
        return True, "", _accepted_row_from_parts(
            compiled=compiled,
            keep=keep,
            lane=lane,
            mapping_kind=mapping_kind,
            shot_count=shot_count,
            caption_path="",
            caption_provenance=provenance,
        )

    def _probe_one(
        item: tuple[dict[str, Any], dict[str, str], str, str, int],
    ) -> tuple[bool, str, dict[str, str] | None]:
        return _finish_probe(
            item, probe_decode_media_ready_row((item[1], config.min_short_side))
        )

    accepted: list[tuple[dict[str, str], list[str] | None]] = []
    completed = 0
    print(
        json.dumps(
            {
                "phase": "probe_start",
                "candidates": len(candidates),
                "workers": workers,
                "rejected_before_probe": len(rejected_records),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    def _consume_probe(
        item: tuple[dict[str, Any], dict[str, str], str, str, int],
        result: tuple[bool, str, dict[str, str] | None],
        *,
        emit_progress: bool = True,
    ) -> None:
        nonlocal completed
        ok, reason, row = result
        completed += 1
        keep = item[0]
        if ok:
            assert row is not None
            shot_texts = _shot_texts_from_keep(keep, shot_count=item[4])
            accepted.append((row, shot_texts))
        else:
            _reject(keep, reason)
        if emit_progress and (completed == 1 or completed % progress_every == 0):
            print(
                json.dumps(
                    {
                        "progress": completed,
                        "candidates": len(candidates),
                        "accepted_so_far": len(accepted),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    probe_timed_out = False
    if workers == 1:
        for item in candidates:
            _consume_probe(item, _probe_one(item))
    else:
        ordered_results: list[tuple[bool, str, dict[str, str] | None] | None] = [
            None
        ] * len(candidates)
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            future_map = {
                executor.submit(
                    probe_decode_media_ready_row,
                    (item[1], config.min_short_side),
                ): index
                for index, item in enumerate(candidates)
            }
            pending = set(future_map)
            filled = 0
            while pending:
                finished, pending = wait(
                    pending,
                    timeout=PROBE_STALL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                if not finished:
                    probe_timed_out = True
                    print(
                        json.dumps(
                            {
                                "phase": "probe_stall_timeout",
                                "hung_n": len(pending),
                                "filled": filled,
                                "stall_seconds": PROBE_STALL_SECONDS,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    for fut in pending:
                        ordered_results[future_map[fut]] = (
                            False,
                            "decode:TimeoutError",
                            None,
                        )
                    pending.clear()
                    break
                for fut in finished:
                    index = future_map[fut]
                    try:
                        decode_result = fut.result()
                    except Exception as error:  # noqa: BLE001
                        decode_result = (False, f"decode:{type(error).__name__}")
                    ordered_results[index] = _finish_probe(
                        candidates[index], decode_result
                    )
                    filled += 1
                    if filled == 1 or filled % progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "progress": filled,
                                    "candidates": len(candidates),
                                    "accepted_so_far": sum(
                                        1
                                        for row in ordered_results
                                        if row is not None and row[0]
                                    ),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        finally:
            if probe_timed_out:
                executor.shutdown(wait=False, cancel_futures=True)
                for proc in getattr(executor, "_processes", {}).values():
                    proc.terminate()
            else:
                executor.shutdown(wait=True)
        for item, result in zip(candidates, ordered_results, strict=True):
            if result is None:
                raise RuntimeError("probe result missing after worker pool exit")
            _consume_probe(item, result, emit_progress=False)

    output_dir = paths.output_dir.expanduser()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(dir=str(output_dir.parent), prefix=f".{output_dir.name}.stage.")
    )
    published = False
    try:
        lanes_dir = staged / "lanes"
        lists_dir = staged / "training_lists"
        lanes_dir.mkdir(parents=True, exist_ok=True)
        lists_dir.mkdir(parents=True, exist_ok=True)

        accepted_rows: list[dict[str, str]] = []
        captions_dir = staged / "captions"
        n_captions = 0
        for row, shot_texts in accepted:
            if shot_texts is not None:
                sidecar = write_msm_caption_sidecar(
                    directory=captions_dir,
                    case_id=row["case_id"],
                    shot_captions=shot_texts,
                )
                row[CAPTION_PATH_COLUMN] = str(output_dir / "captions" / sidecar.name)
                row["caption_provenance"] = CAPTION_PROVENANCE_SELECTED_SHOTS
                n_captions += 1
            accepted_rows.append(row)

        accepted_path = staged / "accepted.csv"
        with accepted_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ACCEPTED_CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(accepted_rows)

        by_lane: dict[str, list[dict[str, str]]] = {lane: [] for lane in sorted(config.lanes)}
        by_dataset: Counter[str] = Counter()
        by_mapping: Counter[str] = Counter()
        for row in accepted_rows:
            by_lane[row["lane"]].append(row)
            by_dataset[row["dataset"]] += 1
            by_mapping[row["mapping_kind"]] += 1

        lane_counts = {lane: len(rows) for lane, rows in by_lane.items()}
        for lane, rows in by_lane.items():
            lane_csv = lanes_dir / f"{lane}.csv"
            with lane_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ACCEPTED_CSV_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            list_path = lists_dir / f"{lane}_case_ids.txt"
            list_path.write_text(
                "".join(f"{row['case_id']}\n" for row in rows),
                encoding="utf-8",
            )

        subject_queue = staged / "lanes" / "subject_context_annotation_queue.jsonl"
        with subject_queue.open("w", encoding="utf-8") as handle:
            for row in accepted_rows:
                handle.write(
                    json.dumps(
                        {
                            "case_id": row["case_id"],
                            "dataset": row["dataset"],
                            "primary_lane": row["lane"],
                            "annotation_status": "not_started",
                            "caption_path": row[CAPTION_PATH_COLUMN],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        rejected_path = staged / "rejected.jsonl"
        with rejected_path.open("w", encoding="utf-8") as handle:
            for record in rejected_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        all_ids = lists_dir / "decode_valid_case_ids.txt"
        all_ids.write_text(
            "".join(f"{row['case_id']}\n" for row in accepted_rows),
            encoding="utf-8",
        )

        summary = {
            "artifact_type": "multishot_media_ready_coarse_wash_v1",
            "status": "complete",
            "generated_at": _now_iso(),
            "paper_cell": "main_table_prerequisite_data_infra",
            "schema_version": config.schema_version,
            "caption_keep_n": len(keep_rows),
            "caption_keep_is_train_n": False,
            "msm_caption_sidecar_n": n_captions,
            "candidates_after_join_attrs_shots": len(candidates),
            "decode_valid_n": len(accepted_rows),
            "rejected_n": len(keep_rows) - len(accepted_rows),
            "lane_counts": lane_counts,
            "subject_context_annotation_queue_n": len(accepted_rows),
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_mapping_kind": dict(sorted(by_mapping.items())),
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "training_allowed": False,
            "training_admission": TRAINING_ADMISSION_MECHANICAL_HOLDOUT,
            "annotation_status": "not_started",
            "probe_contract": build_media_ready_probe_spec(config).__dict__,
            "inputs": {
                "keep_dir": str(paths.keep_dir),
                "shot2story_window_candidates": str(paths.shot2story_window_candidates),
                "ci_vid_window_candidates": str(paths.ci_vid_window_candidates),
                "ci_vid_mapping_ledger": (
                    None
                    if paths.ci_vid_mapping_ledger is None
                    else str(paths.ci_vid_mapping_ledger)
                ),
                "cine_pool_dirs": [str(path) for path in paths.cine_pool_dirs],
            },
            "outputs": {
                "accepted_csv": str(output_dir / "accepted.csv"),
                "rejected_jsonl": str(output_dir / "rejected.jsonl"),
                "captions_dir": str(output_dir / "captions"),
                "lanes_dir": str(output_dir / "lanes"),
                "training_lists_dir": str(output_dir / "training_lists"),
                "summary_json": str(output_dir / "summary.json"),
            },
        }
        (staged / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_dir(staged, output_dir, overwrite=overwrite)
        published = True
        return summary
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)


def _shot_texts_from_caption_json(path: str, *, shot_count: int, label: str) -> list[str]:
    payload = load_strict_json_file(path, label=f"source caption {label}")
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: caption JSON must be an object")
    texts: list[str] = []
    for index in range(shot_count):
        key = f"shot{index}"
        value = payload.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label}: {key} must be a non-empty trimmed string")
        texts.append(value)
    return texts


def load_hq_pool_paths(payload: Mapping[str, Any]) -> MediaReadyWashPaths:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("HQ pool config requires object field inputs")
    full_csv = inputs.get("ci_vid_full_sequence_csv")
    ledger = inputs.get("ci_vid_mapping_ledger")
    cine_dirs = inputs.get("cine_pool_dirs")
    output_dir = payload.get("output_dir")
    if not isinstance(full_csv, str) or not full_csv.strip():
        raise ValueError("inputs.ci_vid_full_sequence_csv must be a non-empty string")
    if not isinstance(ledger, str) or not ledger.strip():
        raise ValueError("inputs.ci_vid_mapping_ledger must be a non-empty string")
    if not isinstance(cine_dirs, list) or not cine_dirs:
        raise ValueError("inputs.cine_pool_dirs must be a non-empty list")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string")
    window_candidates = inputs.get("ci_vid_window_candidates")
    if window_candidates is not None and (
        not isinstance(window_candidates, str) or not window_candidates.strip()
    ):
        raise ValueError("inputs.ci_vid_window_candidates must be a non-empty string when set")
    s2s = inputs.get("shot2story_window_candidates")
    if s2s is not None and (not isinstance(s2s, str) or not s2s.strip()):
        raise ValueError("inputs.shot2story_window_candidates must be a non-empty string when set")
    normalized_cine = []
    for item in cine_dirs:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError("cine_pool_dirs entries must be non-empty trimmed strings")
        normalized_cine.append(Path(item))
    keep_dir = inputs.get("keep_dir")
    if not isinstance(keep_dir, str) or not keep_dir.strip():
        raise ValueError("inputs.keep_dir must be a non-empty string")
    return MediaReadyWashPaths(
        keep_dir=Path(keep_dir),
        shot2story_window_candidates=Path(s2s) if s2s else Path(full_csv),
        ci_vid_window_candidates=Path(window_candidates) if window_candidates else Path(full_csv),
        cine_pool_dirs=tuple(normalized_cine),
        output_dir=Path(output_dir),
        ci_vid_mapping_ledger=Path(ledger),
        ci_vid_full_sequence_csv=Path(full_csv),
    )


def assert_hq_pool_config(config: MediaReadyConfig) -> None:
    if "shot2story" in config.datasets:
        raise ValueError("HQ pool datasets must not include shot2story")
    extra = config.datasets - HQ_DATASETS
    if extra:
        raise ValueError(f"HQ pool datasets outside whitelist: {sorted(extra)}")
    if config.min_rgb_frames != MIN_TRAIN_RGB_FRAMES:
        raise ValueError(f"HQ pool min_rgb_frames must be {MIN_TRAIN_RGB_FRAMES}")
    if config.min_short_side != MIN_SHORT_SIDE:
        raise ValueError(f"HQ pool min_short_side must be {MIN_SHORT_SIDE}")


def _hq_keep_row(
    *,
    record_id: str,
    dataset: str,
    shot_count: int,
    shot_captions: list[str],
    transition_candidate: bool,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "dataset": dataset,
        "shot_count": shot_count,
        "state_candidate": False,
        "generic_multishot_candidate": True,
        "transition_candidate": transition_candidate,
        "event_status": None,
        "shot_captions": shot_captions,
    }


def _cine_caption_path(pool: Mapping[str, Any], csv_row: Mapping[str, Any], pool_id: str) -> str:
    caption_path = pool.get("source_caption_path")
    if not isinstance(caption_path, str) or not caption_path.strip():
        caption_path = csv_row.get(HISTORICAL_CAPTION_PATH_COLUMN)
    if not isinstance(caption_path, str) or not caption_path.strip() or caption_path != caption_path.strip():
        raise ValueError(f"{pool_id}: cine caption path missing from pool and ledger")
    return caption_path


def materialize_hq_keep_from_full_sequences(
    *,
    ci_vid_full_sequence_csv: Path,
    cine_pool_dirs: Sequence[Path],
    output_keep_dir: Path,
) -> dict[str, Any]:
    """Write keep JSONL for HQ: CI-VID full-sequence + Cine pools, no shot2story rows."""

    if output_keep_dir.exists():
        raise FileExistsError(f"HQ keep dir already exists: {output_keep_dir}")
    output_keep_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            dir=str(output_keep_dir.parent), prefix=f".{output_keep_dir.name}.stage."
        )
    )
    csv_path = _require_file(ci_vid_full_sequence_csv, "ci-vid full-sequence CSV")
    n_ci_vid = 0
    n_cine = 0
    published = False
    try:
        (staged / "shot2story.jsonl").write_text("", encoding="utf-8")
        print(json.dumps({"phase": "keep_ci_vid_start"}, ensure_ascii=False), flush=True)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            with (staged / "ci_vid.jsonl").open("w", encoding="utf-8") as dest:
                for row in csv.DictReader(handle):
                    case_id = keep_windows._require_string(
                        row.get("case_id"), "full-sequence case_id"
                    )
                    groups = parse_shot_groups(row["shot_groups"], case_id)
                    caption_path = row.get(HISTORICAL_CAPTION_PATH_COLUMN)
                    if not isinstance(caption_path, str) or not caption_path.strip():
                        raise ValueError(f"{case_id}: historical caption path column is missing")
                    shot_captions = _shot_texts_from_caption_json(
                        caption_path, shot_count=len(groups), label=case_id
                    )
                    dest.write(
                        json.dumps(
                            _hq_keep_row(
                                record_id=case_id,
                                dataset="ci_vid",
                                shot_count=len(groups),
                                shot_captions=shot_captions,
                                transition_candidate=False,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_ci_vid += 1
                    if n_ci_vid == 1 or n_ci_vid % 20000 == 0:
                        print(
                            json.dumps(
                                {"phase": "keep_ci_vid", "n": n_ci_vid},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        print(
            json.dumps({"phase": "keep_cine_start", "n_ci_vid": n_ci_vid}, ensure_ascii=False),
            flush=True,
        )
        cine_pools = keep_windows._index_cine_pools(cine_pool_dirs)
        cine_ledgers, cine_missing = _index_cine_ledgers_soft(cine_pools, set(cine_pools))
        with (staged / "cine250k.jsonl").open("w", encoding="utf-8") as dest:
            for pool_id, pool in cine_pools.items():
                csv_row = cine_ledgers.get(pool_id)
                if csv_row is None:
                    continue
                groups = parse_shot_groups(csv_row.get("shot_groups"), pool_id)
                caption_path = _cine_caption_path(pool, csv_row, pool_id)
                shot_captions = _shot_texts_from_caption_json(
                    caption_path, shot_count=len(groups), label=pool_id
                )
                dest.write(
                    json.dumps(
                        _hq_keep_row(
                            record_id=pool_id,
                            dataset="cine250k",
                            shot_count=len(groups),
                            shot_captions=shot_captions,
                            transition_candidate=True,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_cine += 1
                if n_cine == 1 or n_cine % 20000 == 0:
                    print(
                        json.dumps({"phase": "keep_cine", "n": n_cine}, ensure_ascii=False),
                        flush=True,
                    )
        _publish_dir(staged, output_keep_dir, overwrite=False)
        published = True
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    return {
        "keep_dir": str(output_keep_dir),
        "n_ci_vid": n_ci_vid,
        "n_cine250k": n_cine,
        "n_cine_missing_ledger": len(cine_missing),
        "training_admission": TRAINING_ADMISSION_MECHANICAL_HOLDOUT,
    }
