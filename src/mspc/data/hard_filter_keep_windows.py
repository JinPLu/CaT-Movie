"""Join hard-filter keep IDs to materialize window rows.

Single-file sources emit ``video`` + flat ``source_frame_indices``.  CI-VID
windows whose shots bind to distinct clip files emit the multisource mapping
columns instead; no concatenated proxy video is invented.  Fail loud if a keep
``record_id`` cannot produce the required media columns.  Does not set
``training_allowed``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mspc.data.multishot import (
    MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
    parse_shot_groups,
    parse_source_frame_indices,
    parse_source_frame_indices_by_shot,
    parse_multisource_source_shot_segments,
    parse_source_video_paths,
)

HARD_FILTER_KEEP_N = 312_311
KEEP_DATASETS = ("shot2story", "ci_vid", "cine250k")
CSV_COLUMNS = (
    "case_id",
    "dataset",
    "mapping_schema",
    "video",
    "shot_groups",
    "source_frame_indices",
    "source_video_paths",
    "source_shot_segments",
    "source_frame_indices_by_shot",
)
_EXAMPLE_LIMIT = 8


class KeepWindowCompileError(ValueError):
    """Raised when any keep row cannot be compiled into a single-video CSV row."""

    def __init__(
        self,
        message: str,
        *,
        missing_by_reason: Mapping[str, int],
        examples: Sequence[Mapping[str, str]],
        joined_ok: int,
    ) -> None:
        super().__init__(message)
        self.missing_by_reason = dict(missing_by_reason)
        self.examples = list(examples)
        self.joined_ok = joined_ok


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _iter_jsonl_objects(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, row


def _load_keep_rows(keep_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in KEEP_DATASETS:
        path = _require_file(keep_dir / f"{dataset}.jsonl", f"keep {dataset}")
        for line_number, row in _iter_jsonl_objects(path):
            record_id = _require_string(row.get("record_id"), f"{path}:{line_number}.record_id")
            row_dataset = _require_string(row.get("dataset"), f"{record_id}.dataset")
            if row_dataset != dataset:
                raise ValueError(f"{record_id}: dataset {row_dataset!r} != {dataset!r}")
            if record_id in seen:
                raise ValueError(f"duplicate keep record_id={record_id}")
            seen.add(record_id)
            rows.append({"record_id": record_id, "dataset": dataset})
    return rows


def _index_window_candidates(path: Path) -> dict[str, dict[str, Any]]:
    path = _require_file(path, "window-candidates")
    index: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl_objects(path):
        window_id = _require_string(row.get("window_id"), f"{path}:{line_number}.window_id")
        if window_id in index:
            raise ValueError(f"{path}: duplicate window_id={window_id}")
        plan = row.get("plan")
        if not isinstance(plan, dict):
            plan = {}
        index[window_id] = {
            "video_path": row.get("video_path"),
            "shot_groups": plan.get("shot_groups"),
            "frame_indices": plan.get("frame_indices"),
            "source_video_paths": plan.get("source_video_paths"),
            "source_shot_segments": plan.get("source_shot_segments"),
            "frame_indices_by_shot": plan.get("frame_indices_by_shot"),
        }
    return index


def _index_cine_pools(pool_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    if not pool_dirs:
        raise ValueError("cine pool dirs are required")
    index: dict[str, dict[str, Any]] = {}
    for pool_dir in pool_dirs:
        path = _require_file(pool_dir / "ordinary-multishot.jsonl", f"cine pool {pool_dir}")
        for line_number, row in _iter_jsonl_objects(path):
            pool_id = _require_string(row.get("pool_id"), f"{path}:{line_number}.pool_id")
            if pool_id in index:
                raise ValueError(f"duplicate cine pool_id={pool_id}")
            entry = {
                "source_case_id": _require_string(
                    row.get("source_case_id"), f"{pool_id}.source_case_id"
                ),
                "mapping_fingerprint": _require_string(
                    row.get("mapping_fingerprint"), f"{pool_id}.mapping_fingerprint"
                ),
                "temporal_contract_manifest": _require_string(
                    row.get("temporal_contract_manifest"),
                    f"{pool_id}.temporal_contract_manifest",
                ),
            }
            caption_path = row.get("source_caption_path")
            if isinstance(caption_path, str) and caption_path.strip() and caption_path == caption_path.strip():
                entry["source_caption_path"] = caption_path
            index[pool_id] = entry
    return index


def _index_cine_ledgers(
    pool_index: Mapping[str, Mapping[str, Any]],
    needed_pool_ids: set[str],
) -> dict[str, dict[str, Any]]:
    needed_by_manifest: dict[str, dict[str, str]] = {}
    for pool_id in needed_pool_ids:
        pool = pool_index[pool_id]
        manifest = pool["temporal_contract_manifest"]
        needed_by_manifest.setdefault(manifest, {})[pool["source_case_id"]] = pool_id

    found: dict[str, dict[str, Any]] = {}
    for manifest, cases in needed_by_manifest.items():
        manifest_path = _require_file(Path(manifest), "cine temporal contract manifest")
        ledger_path = _require_file(
            manifest_path.parent / "mapping-ledger.jsonl",
            f"cine mapping-ledger for {manifest_path.parent}",
        )
        remaining = set(cases)
        for line_number, row in _iter_jsonl_objects(ledger_path):
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
                raise ValueError(
                    f"{pool_id}: mapping_fingerprint mismatch "
                    f"{fingerprint!r} != {expected!r}"
                )
            found[pool_id] = {
                "multishot_video": csv_row.get("multishot_video"),
                "shot_groups": csv_row.get("shot_groups"),
                "source_frame_indices": csv_row.get("source_frame_indices"),
                "source_frame_indices_by_shot": csv_row.get(
                    "source_frame_indices_by_shot"
                ),
                "source_video_paths": csv_row.get("source_video_paths"),
                "source_shot_segments": csv_row.get("source_shot_segments"),
            }
            remaining.remove(case_id)
        if remaining:
            raise KeepWindowCompileError(
                "cine mapping-ledger missing full_sequence_candidate csv_row for "
                f"{len(remaining)} keep pool_id(s); published full-sequence-candidates.csv "
                f"is not a substitute when absent. ledger={ledger_path}",
                missing_by_reason={"cine_missing_ledger_csv_row": len(remaining)},
                examples=tuple(
                    {
                        "record_id": cases[case_id],
                        "dataset": "cine250k",
                        "reason": "cine_missing_ledger_csv_row",
                    }
                    for case_id in list(remaining)[:_EXAMPLE_LIMIT]
                ),
                joined_ok=0,
            )
    return found


def _single_video_from_paths(
    paths_value: Any,
    *,
    record_id: str,
    preferred: str | None = None,
) -> str | None:
    try:
        paths = parse_source_video_paths(paths_value, record_id)
    except ValueError:
        return None
    unique = {path for path in paths}
    if len(unique) != 1:
        return None
    video = next(iter(unique))
    if preferred is not None and preferred != video:
        return None
    return video


def _empty_multisource_columns() -> dict[str, str]:
    return {
        "source_video_paths": "",
        "source_shot_segments": "",
        "source_frame_indices_by_shot": "",
    }


def _finalize_row(
    *,
    record_id: str,
    dataset: str,
    video: Any,
    shot_groups: Any,
    source_frame_indices: Any,
) -> dict[str, str] | str:
    if not isinstance(video, str) or not video.strip() or video != video.strip():
        return "missing_video"
    try:
        groups = parse_shot_groups(shot_groups, record_id)
        frames = parse_source_frame_indices(source_frame_indices, record_id)
    except ValueError:
        if shot_groups in (None, "", []):
            return "missing_shot_groups"
        return "missing_source_frame_indices"
    if groups[-1][1] != len(frames):
        return "shot_groups_frame_mismatch"
    return {
        "case_id": record_id,
        "dataset": dataset,
        "mapping_schema": "",
        "video": video,
        "shot_groups": _dumps(groups),
        "source_frame_indices": _dumps(frames),
        **_empty_multisource_columns(),
    }


def _finalize_multisource_row(
    *,
    record_id: str,
    dataset: str,
    source_video_paths: Any,
    shot_groups: Any,
    source_shot_segments: Any,
    source_frame_indices_by_shot: Any,
) -> dict[str, str] | str:
    try:
        paths = parse_source_video_paths(source_video_paths, record_id)
        groups = parse_shot_groups(shot_groups, record_id)
        segments = parse_multisource_source_shot_segments(source_shot_segments, record_id)
        indices_by_shot = parse_source_frame_indices_by_shot(
            source_frame_indices_by_shot,
            record_id,
        )
    except ValueError:
        return "missing_multisource_mapping"
    if not (len(groups) == len(paths) == len(segments) == len(indices_by_shot)):
        return "multisource_shot_count_mismatch"
    if len({path for path in paths}) <= 1:
        return "ci_vid_single_video_use_flat_mapping"
    return {
        "case_id": record_id,
        "dataset": dataset,
        "mapping_schema": MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
        "video": paths[0],
        "shot_groups": _dumps(groups),
        "source_frame_indices": "",
        "source_video_paths": _dumps(paths),
        "source_shot_segments": _dumps(segments),
        "source_frame_indices_by_shot": _dumps(indices_by_shot),
    }


def _compile_shot2story(record_id: str, window: Mapping[str, Any] | None) -> dict[str, str] | str:
    if window is None:
        return "missing_join_key"
    return _finalize_row(
        record_id=record_id,
        dataset="shot2story",
        video=window.get("video_path"),
        shot_groups=window.get("shot_groups"),
        source_frame_indices=window.get("frame_indices"),
    )


def _index_ci_vid_full_sequence_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Index CI-VID ``full_sequence_candidate`` csv_rows by ``case_id``.

    Same join path as ``_index_cine_ledgers``: read ledger ``csv_row``, do not
    require a published full-sequence CSV.
    """

    path = _require_file(path, "ci-vid mapping-ledger")
    index: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl_objects(path):
        if row.get("status") != "full_sequence_candidate":
            continue
        csv_row = row.get("csv_row")
        if not isinstance(csv_row, dict):
            raise ValueError(f"{path}:{line_number}: full_sequence_candidate missing csv_row object")
        case_id = _require_string(csv_row.get("case_id"), f"{path}:{line_number}.csv_row.case_id")
        if case_id in index:
            raise ValueError(f"{path}: duplicate full-sequence case_id={case_id}")
        index[case_id] = csv_row
    return index


def _parse_csv_cell(value: Any, label: str) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON string or list")
    return json.loads(value)


def _window_from_full_sequence_csv_row(csv_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "video_path": csv_row.get("multishot_video"),
        "shot_groups": _parse_csv_cell(csv_row.get("shot_groups"), "shot_groups"),
        "frame_indices": _parse_csv_cell(
            csv_row.get("source_frame_indices"), "source_frame_indices"
        ),
        "source_video_paths": _parse_csv_cell(
            csv_row.get("source_video_paths"), "source_video_paths"
        ),
        "source_shot_segments": _parse_csv_cell(
            csv_row.get("source_shot_segments"), "source_shot_segments"
        ),
        "frame_indices_by_shot": _parse_csv_cell(
            csv_row.get("source_frame_indices_by_shot"), "source_frame_indices_by_shot"
        ),
    }


def _compile_ci_vid(record_id: str, window: Mapping[str, Any] | None) -> dict[str, str] | str:
    if window is None:
        return "missing_join_key"
    paths_value = window.get("source_video_paths")
    try:
        paths = parse_source_video_paths(paths_value, record_id)
    except ValueError:
        paths = []
    if len({path for path in paths}) > 1:
        return _finalize_multisource_row(
            record_id=record_id,
            dataset="ci_vid",
            source_video_paths=paths_value,
            shot_groups=window.get("shot_groups"),
            source_shot_segments=window.get("source_shot_segments"),
            source_frame_indices_by_shot=window.get("frame_indices_by_shot"),
        )
    video = window.get("video_path")
    if not (isinstance(video, str) and video.strip() and video == video.strip()):
        video = _single_video_from_paths(paths_value, record_id=record_id)
        if video is None:
            return "missing_video"
    frames = window.get("frame_indices")
    if frames in (None, "", []):
        by_shot = window.get("frame_indices_by_shot")
        if by_shot not in (None, "", []):
            try:
                frames = [
                    index
                    for shot in parse_source_frame_indices_by_shot(by_shot, record_id)
                    for index in shot
                ]
            except ValueError:
                frames = None
    return _finalize_row(
        record_id=record_id,
        dataset="ci_vid",
        video=video,
        shot_groups=window.get("shot_groups"),
        source_frame_indices=frames,
    )


def _finalize_cine_v4_row(
    *,
    record_id: str,
    video: str,
    shot_groups: Any,
    source_video_paths: Any,
    source_shot_segments: Any,
    source_frame_indices_by_shot: Any,
) -> dict[str, str] | str:
    """Emit Cine as v4 even when every shot repeats the same file.

    The renderer CSV schema is mixed: CI-VID true multisource plus Cine
    single-file rows that still need ``mapping_schema`` and the three v4
    columns so pandas does not turn empty cells into NaN.
    """

    try:
        paths = parse_source_video_paths(source_video_paths, record_id)
        groups = parse_shot_groups(shot_groups, record_id)
        segments = parse_multisource_source_shot_segments(source_shot_segments, record_id)
        indices_by_shot = parse_source_frame_indices_by_shot(
            source_frame_indices_by_shot,
            record_id,
        )
    except ValueError:
        return "missing_cine_v4_mapping"
    if not (len(groups) == len(paths) == len(segments) == len(indices_by_shot)):
        return "cine_v4_shot_count_mismatch"
    if len(set(paths)) != 1 or paths[0] != video:
        return "cine_multi_video"
    frames = [index for shot in indices_by_shot for index in shot]
    if groups[-1][1] != len(frames):
        return "shot_groups_frame_mismatch"
    return {
        "case_id": record_id,
        "dataset": "cine250k",
        "mapping_schema": MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
        "video": video,
        "shot_groups": _dumps(groups),
        "source_frame_indices": _dumps(frames),
        "source_video_paths": _dumps(paths),
        "source_shot_segments": _dumps(segments),
        "source_frame_indices_by_shot": _dumps(indices_by_shot),
    }


def _compile_cine(record_id: str, csv_row: Mapping[str, Any] | None) -> dict[str, str] | str:
    if csv_row is None:
        return "missing_join_key"
    video = csv_row.get("multishot_video")
    if not (isinstance(video, str) and video.strip() and video == video.strip()):
        return "missing_video"
    if not _single_video_from_paths(
        csv_row.get("source_video_paths"),
        record_id=record_id,
        preferred=video,
    ):
        return "cine_multi_video"
    return _finalize_cine_v4_row(
        record_id=record_id,
        video=video,
        shot_groups=csv_row.get("shot_groups"),
        source_video_paths=csv_row.get("source_video_paths"),
        source_shot_segments=csv_row.get("source_shot_segments"),
        source_frame_indices_by_shot=csv_row.get("source_frame_indices_by_shot"),
    )


def _raise_missing(
    missing: Sequence[Mapping[str, str]],
    *,
    joined_ok: int,
    expected_rows: int,
) -> None:
    counts = Counter(item["reason"] for item in missing)
    examples = list(missing[:_EXAMPLE_LIMIT])
    raise KeepWindowCompileError(
        "hard-filter keep → window CSV refused: "
        f"{len(missing)} / {expected_rows} rows missing a required media column or join key. "
        f"joined_ok={joined_ok} missing_by_reason={dict(counts)}. "
        "Do not invent concatenated videos or use media_valid_4096_v1. "
        "No CSV written. "
        + (
            "ci_vid_single_video_use_flat_mapping means a CI-VID window still "
            "binds every shot to the same clip and must use the flat mapping."
            if counts.get("ci_vid_single_video_use_flat_mapping")
            else ""
        ),
        missing_by_reason=counts,
        examples=examples,
        joined_ok=joined_ok,
    )


def compile_hard_filter_keep_windows(
    *,
    keep_dir: Path,
    shot2story_window_candidates: Path,
    ci_vid_window_candidates: Path,
    cine_pool_dirs: Sequence[Path],
    output_csv: Path,
    expected_rows: int = HARD_FILTER_KEEP_N,
    ci_vid_mapping_ledger: Path | None = None,
) -> dict[str, Any]:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    output_csv = output_csv.expanduser()
    if output_csv.exists():
        raise FileExistsError(f"output CSV already exists: {output_csv}")

    keep_rows = _load_keep_rows(keep_dir)
    if len(keep_rows) != expected_rows:
        raise ValueError(
            f"keep dir has {len(keep_rows)} rows, expected {expected_rows}"
        )

    s2s_index = _index_window_candidates(shot2story_window_candidates)
    ci_index = _index_window_candidates(ci_vid_window_candidates)
    ci_full_index = (
        _index_ci_vid_full_sequence_ledger(ci_vid_mapping_ledger)
        if ci_vid_mapping_ledger is not None
        else {}
    )
    cine_pools = _index_cine_pools(cine_pool_dirs)
    cine_needed = {row["record_id"] for row in keep_rows if row["dataset"] == "cine250k"}
    missing_pool = sorted(cine_needed - set(cine_pools))
    if missing_pool:
        _raise_missing(
            [
                {
                    "record_id": record_id,
                    "dataset": "cine250k",
                    "reason": "missing_join_key",
                }
                for record_id in missing_pool
            ],
            joined_ok=0,
            expected_rows=expected_rows,
        )
    cine_ledgers = _index_cine_ledgers(cine_pools, cine_needed)

    compiled: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    n_ci_full = 0
    for row in keep_rows:
        record_id = row["record_id"]
        dataset = row["dataset"]
        if dataset == "shot2story":
            result = _compile_shot2story(record_id, s2s_index.get(record_id))
        elif dataset == "ci_vid":
            window = ci_index.get(record_id)
            if window is None and record_id in ci_full_index:
                window = _window_from_full_sequence_csv_row(ci_full_index[record_id])
                n_ci_full += 1
            result = _compile_ci_vid(record_id, window)
        elif dataset == "cine250k":
            result = _compile_cine(record_id, cine_ledgers.get(record_id))
        else:
            raise ValueError(f"{record_id}: unsupported dataset {dataset!r}")
        if isinstance(result, str):
            missing.append(
                {"record_id": record_id, "dataset": dataset, "reason": result}
            )
        else:
            compiled.append(result)

    if missing:
        _raise_missing(missing, joined_ok=len(compiled), expected_rows=expected_rows)
    if len(compiled) != expected_rows:
        raise ValueError(
            f"compiled {len(compiled)} rows, expected {expected_rows}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_csv.parent,
        prefix=f".{output_csv.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(compiled)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, output_csv)
    except BaseException:
        handle.close()
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise

    return {
        "output_csv": str(output_csv),
        "n_rows": len(compiled),
        "expected_rows": expected_rows,
        "by_dataset": dict(Counter(row["dataset"] for row in compiled)),
        "ci_vid_full_sequence_joined": n_ci_full,
        "training_allowed": False,
        "training_admission": "mechanical_holdout",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile hard-filter keep record_id values into a materialize CSV. "
            "CI-VID multi-clip windows use multisource columns. Fails loud if "
            "any required media column is missing."
        )
    )
    parser.add_argument("--keep-dir", type=Path, required=True)
    parser.add_argument("--shot2story-window-candidates", type=Path, required=True)
    parser.add_argument("--ci-vid-window-candidates", type=Path, required=True)
    parser.add_argument("--ci-vid-mapping-ledger", type=Path, default=None)
    parser.add_argument(
        "--cine-pool-dir",
        type=Path,
        action="append",
        required=True,
        dest="cine_pool_dirs",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=HARD_FILTER_KEEP_N)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compile_hard_filter_keep_windows(
        keep_dir=args.keep_dir,
        shot2story_window_candidates=args.shot2story_window_candidates,
        ci_vid_window_candidates=args.ci_vid_window_candidates,
        ci_vid_mapping_ledger=args.ci_vid_mapping_ledger,
        cine_pool_dirs=args.cine_pool_dirs,
        output_csv=args.output_csv,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
