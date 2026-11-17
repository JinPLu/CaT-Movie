from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mspc.data.hard_filter_keep_windows import (
    HARD_FILTER_KEEP_N,
    KeepWindowCompileError,
    compile_hard_filter_keep_windows,
)


def _write_jsonl(path: Path, rows: list[dict[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _keep_dir(tmp_path: Path, rows_by_dataset: dict[str, list[dict[str, object]]]) -> Path:
    keep_dir = tmp_path / "keep"
    for name in ("shot2story", "ci_vid", "cine250k"):
        _write_jsonl(keep_dir / f"{name}.jsonl", rows_by_dataset.get(name, []))
    return keep_dir


def _s2s_window(*, window_id: str, video: str) -> dict[str, object]:
    return {
        "window_id": window_id,
        "video_path": video,
        "plan": {
            "shot_groups": [[0, 2], [2, 4]],
            "frame_indices": [10, 11, 20, 21],
        },
    }


def _ci_window(*, window_id: str, videos: list[str]) -> dict[str, object]:
    return {
        "window_id": window_id,
        "plan": {
            "source_video_paths": videos,
            "source_shot_segments": [[0, 100], [0, 120]],
            "shot_groups": [[0, 2], [2, 4]],
            "frame_indices": [1, 2, 3, 4],
            "frame_indices_by_shot": [[1, 2], [3, 4]],
        },
    }


def _cine_pool(
    *,
    pool_id: str,
    case_id: str,
    fingerprint: str,
    manifest: Path,
) -> dict[str, object]:
    return {
        "pool_id": pool_id,
        "source_case_id": case_id,
        "mapping_fingerprint": fingerprint,
        "temporal_contract_manifest": str(manifest),
    }


def _cine_ledger_row(
    *,
    case_id: str,
    fingerprint: str,
    video: str,
    videos: list[str] | None = None,
) -> dict[str, object]:
    return {
        "status": "full_sequence_candidate",
        "csv_row": {
            "case_id": case_id,
            "mapping_fingerprint": fingerprint,
            "multishot_video": video,
            "source_video_paths": json.dumps(videos or [video, video]),
            "shot_groups": "[[0,2],[2,4]]",
            "source_shot_segments": "[[10,12],[20,22]]",
            "source_frame_indices_by_shot": "[[10,11],[20,21]]",
        },
    }


def _compile(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    output = tmp_path / "keep-windows.csv"
    return compile_hard_filter_keep_windows(
        keep_dir=kwargs["keep_dir"],  # type: ignore[arg-type]
        shot2story_window_candidates=kwargs["s2s"],  # type: ignore[arg-type]
        ci_vid_window_candidates=kwargs["ci"],  # type: ignore[arg-type]
        cine_pool_dirs=kwargs["cine_pools"],  # type: ignore[arg-type]
        output_csv=output,
        expected_rows=kwargs["expected"],  # type: ignore[arg-type]
    )


def test_compiles_single_video_rows_and_flattens_cine(tmp_path: Path) -> None:
    manifest = tmp_path / "cine" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [
                {"record_id": "s2s__shots_0_1", "dataset": "shot2story"},
            ],
            "ci_vid": [
                {"record_id": "ci_vid:train:1__shots_0_1", "dataset": "ci_vid"},
            ],
            "cine250k": [
                {"record_id": "full:cine250k:cine250k:1", "dataset": "cine250k"},
            ],
        },
    )
    s2s = tmp_path / "s2s-windows.jsonl"
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(s2s, [_s2s_window(window_id="s2s__shots_0_1", video="/v/s2s.mp4")])
    _write_jsonl(
        ci,
        [_ci_window(window_id="ci_vid:train:1__shots_0_1", videos=["/v/same.mp4", "/v/same.mp4"])],
    )
    _write_jsonl(
        pool_dir / "ordinary-multishot.jsonl",
        [
            _cine_pool(
                pool_id="full:cine250k:cine250k:1",
                case_id="cine250k:1",
                fingerprint="abc",
                manifest=manifest,
            )
        ],
    )
    _write_jsonl(
        manifest.parent / "mapping-ledger.jsonl",
        [
            _cine_ledger_row(
                case_id="cine250k:1",
                fingerprint="abc",
                video="/v/cine.mp4",
            )
        ],
    )

    result = _compile(
        tmp_path,
        keep_dir=keep_dir,
        s2s=s2s,
        ci=ci,
        cine_pools=[pool_dir],
        expected=3,
    )
    output = Path(result["output_csv"])
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert result["n_rows"] == 3
    assert result["training_allowed"] is False
    assert [row["case_id"] for row in rows] == [
        "s2s__shots_0_1",
        "ci_vid:train:1__shots_0_1",
        "full:cine250k:cine250k:1",
    ]
    assert rows[0]["video"] == "/v/s2s.mp4"
    assert rows[0]["mapping_schema"] == ""
    assert rows[1]["video"] == "/v/same.mp4"
    assert rows[1]["mapping_schema"] == ""
    assert rows[2]["video"] == "/v/cine.mp4"
    assert rows[2]["mapping_schema"] == "multishot_csv_multisource_mapping_v6_boundary_revised"
    assert json.loads(rows[2]["source_frame_indices"]) == [10, 11, 20, 21]
    assert json.loads(rows[2]["shot_groups"]) == [[0, 2], [2, 4]]
    assert json.loads(rows[2]["source_video_paths"]) == ["/v/cine.mp4", "/v/cine.mp4"]
    assert json.loads(rows[2]["source_shot_segments"]) == [[10, 12], [20, 22]]
    assert json.loads(rows[2]["source_frame_indices_by_shot"]) == [[10, 11], [20, 21]]


def test_ci_multi_video_emits_multisource_columns(tmp_path: Path) -> None:
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [],
            "ci_vid": [
                {"record_id": "ci_vid:train:1__shots_0_1", "dataset": "ci_vid"},
            ],
            "cine250k": [],
        },
    )
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(
        ci,
        [
            _ci_window(
                window_id="ci_vid:train:1__shots_0_1",
                videos=["/v/a.mp4", "/v/b.mp4"],
            )
        ],
    )
    _write_jsonl(pool_dir / "ordinary-multishot.jsonl", [])
    s2s = tmp_path / "s2s-windows.jsonl"
    _write_jsonl(s2s, [])
    result = _compile(
        tmp_path,
        keep_dir=keep_dir,
        s2s=s2s,
        ci=ci,
        cine_pools=[pool_dir],
        expected=1,
    )
    rows = list(csv.DictReader(Path(result["output_csv"]).open(encoding="utf-8")))
    assert rows[0]["mapping_schema"] == "multishot_csv_multisource_mapping_v6_boundary_revised"
    assert rows[0]["video"] == "/v/a.mp4"
    assert rows[0]["source_frame_indices"] == ""
    assert json.loads(rows[0]["source_video_paths"]) == ["/v/a.mp4", "/v/b.mp4"]
    assert json.loads(rows[0]["source_frame_indices_by_shot"]) == [[1, 2], [3, 4]]


def test_ci_multi_video_missing_segments_fails_loud(tmp_path: Path) -> None:
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [],
            "ci_vid": [
                {"record_id": "ci_vid:train:1__shots_0_1", "dataset": "ci_vid"},
            ],
            "cine250k": [],
        },
    )
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(
        ci,
        [
            {
                "window_id": "ci_vid:train:1__shots_0_1",
                "plan": {
                    "source_video_paths": ["/v/a.mp4", "/v/b.mp4"],
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices_by_shot": [[1, 2], [3, 4]],
                },
            }
        ],
    )
    _write_jsonl(pool_dir / "ordinary-multishot.jsonl", [])
    s2s = tmp_path / "s2s-windows.jsonl"
    _write_jsonl(s2s, [])
    output = tmp_path / "keep-windows.csv"
    with pytest.raises(KeepWindowCompileError, match="missing_multisource_mapping"):
        compile_hard_filter_keep_windows(
            keep_dir=keep_dir,
            shot2story_window_candidates=s2s,
            ci_vid_window_candidates=ci,
            cine_pool_dirs=[pool_dir],
            output_csv=output,
            expected_rows=1,
        )
    assert not output.exists()


def test_missing_join_key_fails_loud(tmp_path: Path) -> None:
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [
                {"record_id": "s2s__missing", "dataset": "shot2story"},
            ],
            "ci_vid": [],
            "cine250k": [],
        },
    )
    s2s = tmp_path / "s2s-windows.jsonl"
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(s2s, [])
    _write_jsonl(ci, [])
    _write_jsonl(pool_dir / "ordinary-multishot.jsonl", [])
    output = tmp_path / "keep-windows.csv"
    with pytest.raises(KeepWindowCompileError, match="missing_join_key"):
        compile_hard_filter_keep_windows(
            keep_dir=keep_dir,
            shot2story_window_candidates=s2s,
            ci_vid_window_candidates=ci,
            cine_pool_dirs=[pool_dir],
            output_csv=output,
            expected_rows=1,
        )
    assert not output.exists()


def test_cine_distinct_videos_fail_loud(tmp_path: Path) -> None:
    manifest = tmp_path / "cine" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [],
            "ci_vid": [],
            "cine250k": [
                {"record_id": "full:cine250k:cine250k:1", "dataset": "cine250k"},
            ],
        },
    )
    s2s = tmp_path / "s2s-windows.jsonl"
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(s2s, [])
    _write_jsonl(ci, [])
    _write_jsonl(
        pool_dir / "ordinary-multishot.jsonl",
        [
            _cine_pool(
                pool_id="full:cine250k:cine250k:1",
                case_id="cine250k:1",
                fingerprint="abc",
                manifest=manifest,
            )
        ],
    )
    _write_jsonl(
        manifest.parent / "mapping-ledger.jsonl",
        [
            _cine_ledger_row(
                case_id="cine250k:1",
                fingerprint="abc",
                video="/v/cine.mp4",
                videos=["/v/cine.mp4", "/v/other.mp4"],
            )
        ],
    )
    output = tmp_path / "keep-windows.csv"
    with pytest.raises(KeepWindowCompileError, match="cine_multi_video"):
        compile_hard_filter_keep_windows(
            keep_dir=keep_dir,
            shot2story_window_candidates=s2s,
            ci_vid_window_candidates=ci,
            cine_pool_dirs=[pool_dir],
            output_csv=output,
            expected_rows=1,
        )
    assert not output.exists()


def test_ci_full_sequence_ledger_joins_without_window_id(tmp_path: Path) -> None:
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [],
            "ci_vid": [{"record_id": "ci_vid:train:000000001", "dataset": "ci_vid"}],
            "cine250k": [],
        },
    )
    ledger = tmp_path / "mapping-ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {
                "status": "full_sequence_candidate",
                "csv_row": {
                    "case_id": "ci_vid:train:000000001",
                    "multishot_video": "/v/a.mp4",
                    "source_video_paths": json.dumps(["/v/a.mp4", "/v/b.mp4"]),
                    "shot_groups": json.dumps([[0, 2], [2, 4]]),
                    "source_shot_segments": json.dumps([[0, 100], [0, 120]]),
                    "source_frame_indices_by_shot": json.dumps([[1, 2], [3, 4]]),
                },
            }
        ],
    )
    ci = tmp_path / "ci-windows.jsonl"
    s2s = tmp_path / "s2s-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(ci, [])
    _write_jsonl(s2s, [])
    _write_jsonl(pool_dir / "ordinary-multishot.jsonl", [])
    result = compile_hard_filter_keep_windows(
        keep_dir=keep_dir,
        shot2story_window_candidates=s2s,
        ci_vid_window_candidates=ci,
        ci_vid_mapping_ledger=ledger,
        cine_pool_dirs=[pool_dir],
        output_csv=tmp_path / "keep-windows.csv",
        expected_rows=1,
    )
    rows = list(csv.DictReader(Path(result["output_csv"]).open(encoding="utf-8")))
    assert result["ci_vid_full_sequence_joined"] == 1
    assert rows[0]["case_id"] == "ci_vid:train:000000001"
    assert rows[0]["mapping_schema"] == "multishot_csv_multisource_mapping_v6_boundary_revised"
    assert json.loads(rows[0]["source_video_paths"]) == ["/v/a.mp4", "/v/b.mp4"]


def test_keep_count_mismatch_fails_before_write(tmp_path: Path) -> None:
    keep_dir = _keep_dir(
        tmp_path,
        {
            "shot2story": [],
            "ci_vid": [],
            "cine250k": [],
        },
    )
    s2s = tmp_path / "s2s-windows.jsonl"
    ci = tmp_path / "ci-windows.jsonl"
    pool_dir = tmp_path / "pool"
    _write_jsonl(s2s, [])
    _write_jsonl(ci, [])
    _write_jsonl(pool_dir / "ordinary-multishot.jsonl", [])
    output = tmp_path / "keep-windows.csv"
    with pytest.raises(ValueError, match="keep dir has 0 rows"):
        compile_hard_filter_keep_windows(
            keep_dir=keep_dir,
            shot2story_window_candidates=s2s,
            ci_vid_window_candidates=ci,
            cine_pool_dirs=[pool_dir],
            output_csv=output,
            expected_rows=HARD_FILTER_KEEP_N,
        )
    assert not output.exists()
