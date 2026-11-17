from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mspc.data.multishot_media_ready import (
    LANE_EVENTFUL_TRANSITION,
    LANE_GENERIC_MULTISHOT,
    MediaReadyConfig,
    _probe_chunksize,
    classify_media_ready_lane,
    load_media_ready_wash_paths,
    probe_decode_media_ready_row,
    run_media_ready_coarse_wash,
    validate_media_ready_row_attrs,
)

_WASH_CONTRACT = {
    "schema_version": "multishot_media_ready_v1",
    "target_fps": 15,
    "max_rgb_frames": 308,
    "vae_temporal_stride": 4,
    "min_shots": 2,
    "max_shots": 5,
    "datasets": ["shot2story", "ci_vid", "cine250k"],
    "lanes": ["generic_multishot", "eventful_transition", "subject_context"],
}


def test_classify_lane_prefers_eventful_for_ci_vid() -> None:
    assert (
        classify_media_ready_lane(dataset="ci_vid", event_state_candidate=True)
        == LANE_EVENTFUL_TRANSITION
    )
    assert (
        classify_media_ready_lane(dataset="cine250k", event_state_candidate=True)
        == LANE_GENERIC_MULTISHOT
    )


def test_validate_multisource_row_attrs_checks_shot_bounds(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    config = MediaReadyConfig.from_mapping(
        {
            "schema_version": "multishot_media_ready_v1",
            "target_fps": 15,
            "max_rgb_frames": 308,
            "vae_temporal_stride": 4,
            "min_shots": 2,
            "max_shots": 5,
            "datasets": ["ci_vid"],
            "lanes": ["generic_multishot", "eventful_transition", "subject_context"],
        }
    )
    row = {
        "mapping_schema": "multishot_csv_multisource_mapping_v6_boundary_revised",
        "shot_groups": json.dumps([[0, 2], [2, 4]]),
        "source_video_paths": json.dumps([str(first), str(second)]),
        "source_shot_segments": json.dumps([[0, 100], [0, 120]]),
        "source_frame_indices_by_shot": json.dumps([[0, 1], [0, 1]]),
    }

    summary = validate_media_ready_row_attrs(
        row,
        config=config,
        dataset="ci_vid",
        record_id="ci_vid:test:1",
    )

    assert summary["mapping_kind"] == "multisource"
    assert summary["shot_count"] == 2
    assert summary["lane"] == LANE_GENERIC_MULTISHOT


def test_validate_multisource_row_missing_clip_fails_loud(tmp_path: Path) -> None:
    config = MediaReadyConfig.from_mapping(
        {
            "schema_version": "multishot_media_ready_v1",
            "target_fps": 15,
            "max_rgb_frames": 308,
            "vae_temporal_stride": 4,
            "min_shots": 2,
            "max_shots": 5,
            "datasets": ["ci_vid"],
            "lanes": ["generic_multishot"],
        }
    )
    row = {
        "mapping_schema": "multishot_csv_multisource_mapping_v6_boundary_revised",
        "shot_groups": json.dumps([[0, 2], [2, 4]]),
        "source_video_paths": json.dumps(["/missing/a.mp4", "/missing/b.mp4"]),
        "source_shot_segments": json.dumps([[0, 100], [0, 120]]),
        "source_frame_indices_by_shot": json.dumps([[0, 1], [0, 1]]),
    }

    with pytest.raises(FileNotFoundError, match="missing or empty multisource"):
        validate_media_ready_row_attrs(
            row,
            config=config,
            dataset="ci_vid",
            record_id="ci_vid:test:2",
        )


def _write_jsonl(path: Path, rows: list[dict[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tiny_mp4(path: Path) -> None:
    """Write a minimal decodable mp4 via imageio/ffmpeg if available, else skip."""
    pytest.importorskip("decord")
    try:
        import imageio_ffmpeg
        import subprocess
    except ImportError:
        pytest.skip("imageio_ffmpeg required to synthesize probe mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            "8",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_coarse_wash_drops_missing_and_emits_lanes(tmp_path: Path) -> None:
    pytest.importorskip("decord")
    video = tmp_path / "ok.mp4"
    _tiny_mp4(video)

    keep_dir = tmp_path / "keep"
    _write_jsonl(
        keep_dir / "shot2story.jsonl",
        [
            {
                "record_id": "s2s_ok",
                "dataset": "shot2story",
                "shot_count": 2,
                "state_candidate": True,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "persistent_change",
            },
            {
                "record_id": "s2s_missing",
                "dataset": "shot2story",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "stable",
            },
        ],
    )
    _write_jsonl(keep_dir / "ci_vid.jsonl", [])
    _write_jsonl(keep_dir / "cine250k.jsonl", [])
    s2s = tmp_path / "s2s.jsonl"
    _write_jsonl(
        s2s,
        [
            {
                "window_id": "s2s_ok",
                "video_path": str(video),
                "plan": {
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices": [0, 1, 2, 3],
                },
            },
            {
                "window_id": "s2s_missing",
                "video_path": str(tmp_path / "missing.mp4"),
                "plan": {
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices": [0, 1, 2, 3],
                },
            },
        ],
    )
    ci = tmp_path / "ci.jsonl"
    _write_jsonl(ci, [])
    pool = tmp_path / "pool"
    pool.mkdir()
    _write_jsonl(pool / "ordinary-multishot.jsonl", [])

    config = MediaReadyConfig.from_mapping(
        {
            "schema_version": "multishot_media_ready_v1",
            "target_fps": 15,
            "max_rgb_frames": 308,
            "vae_temporal_stride": 4,
            "min_shots": 2,
            "max_shots": 5,
            "datasets": ["shot2story", "ci_vid", "cine250k"],
            "lanes": ["generic_multishot", "eventful_transition", "subject_context"],
        }
    )
    paths = load_media_ready_wash_paths(
        {
            "inputs": {
                "keep_dir": str(keep_dir),
                "shot2story_window_candidates": str(s2s),
                "ci_vid_window_candidates": str(ci),
                "cine_pool_dirs": [str(pool)],
            },
            "output_dir": str(tmp_path / "out"),
        }
    )
    summary = run_media_ready_coarse_wash(
        config=config,
        paths=paths,
        workers=1,
        overwrite=False,
        expected_keep_rows=2,
    )

    assert summary["decode_valid_n"] == 1
    assert summary["lane_counts"][LANE_EVENTFUL_TRANSITION] == 1
    assert summary["caption_keep_is_train_n"] is False
    accepted = list(csv.DictReader((tmp_path / "out" / "accepted.csv").open(encoding="utf-8")))
    assert [row["case_id"] for row in accepted] == ["s2s_ok"]
    assert accepted[0]["training_allowed"] == "false"
    assert accepted[0]["training_admission"] == "mechanical_holdout"
    assert "annotation_model" not in accepted[0]
    assert (tmp_path / "out" / "training_lists" / "eventful_transition_case_ids.txt").read_text(
        encoding="utf-8"
    ) == "s2s_ok\n"


def test_coarse_wash_ci_vid_multisource_and_cine_culls_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_a = tmp_path / "ci_a.mp4"
    clip_b = tmp_path / "ci_b.mp4"
    cine_ok = tmp_path / "cine_ok.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    cine_ok.write_bytes(b"c")
    monkeypatch.setattr(
        "mspc.data.multishot_media_ready.light_decode_compiled_row",
        lambda compiled: None,
    )

    keep_dir = tmp_path / "keep"
    _write_jsonl(keep_dir / "shot2story.jsonl", [])
    _write_jsonl(
        keep_dir / "ci_vid.jsonl",
        [
            {
                "record_id": "ci_ok",
                "dataset": "ci_vid",
                "shot_count": 2,
                "state_candidate": True,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "persistent_change",
            },
            {
                "record_id": "ci_missing_clip",
                "dataset": "ci_vid",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "stable",
            },
        ],
    )
    _write_jsonl(
        keep_dir / "cine250k.jsonl",
        [
            {
                "record_id": "cine_ok",
                "dataset": "cine250k",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": True,
                "event_status": None,
            },
            {
                "record_id": "cine_absent_pool",
                "dataset": "cine250k",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": True,
                "event_status": None,
            },
            {
                "record_id": "cine_missing_video",
                "dataset": "cine250k",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": True,
                "event_status": None,
            },
        ],
    )

    _write_jsonl(tmp_path / "s2s.jsonl", [])
    _write_jsonl(
        tmp_path / "ci.jsonl",
        [
            {
                "window_id": "ci_ok",
                "plan": {
                    "source_video_paths": [str(clip_a), str(clip_b)],
                    "source_shot_segments": [[0, 100], [0, 120]],
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices_by_shot": [[0, 1], [0, 1]],
                },
            },
            {
                "window_id": "ci_missing_clip",
                "plan": {
                    "source_video_paths": [str(clip_a), str(tmp_path / "gone.mp4")],
                    "source_shot_segments": [[0, 100], [0, 120]],
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices_by_shot": [[0, 1], [0, 1]],
                },
            },
        ],
    )

    manifest = tmp_path / "cine_contract" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    pool = tmp_path / "pool"
    _write_jsonl(
        pool / "ordinary-multishot.jsonl",
        [
            {
                "pool_id": "cine_ok",
                "source_case_id": "case_ok",
                "mapping_fingerprint": "fp_ok",
                "temporal_contract_manifest": str(manifest),
            },
            {
                "pool_id": "cine_missing_video",
                "source_case_id": "case_missing",
                "mapping_fingerprint": "fp_missing",
                "temporal_contract_manifest": str(manifest),
            },
        ],
    )
    _write_jsonl(
        manifest.parent / "mapping-ledger.jsonl",
        [
            {
                "status": "full_sequence_candidate",
                "csv_row": {
                    "case_id": "case_ok",
                    "mapping_fingerprint": "fp_ok",
                    "multishot_video": str(cine_ok),
                    "source_video_paths": json.dumps([str(cine_ok), str(cine_ok)]),
                    "shot_groups": "[[0,2],[2,4]]",
                    "source_shot_segments": "[[0,2],[0,2]]",
                    "source_frame_indices_by_shot": "[[0,1],[2,3]]",
                },
            },
            {
                "status": "full_sequence_candidate",
                "csv_row": {
                    "case_id": "case_missing",
                    "mapping_fingerprint": "fp_missing",
                    "multishot_video": str(tmp_path / "cine_gone.mp4"),
                    "source_video_paths": json.dumps(
                        [str(tmp_path / "cine_gone.mp4"), str(tmp_path / "cine_gone.mp4")]
                    ),
                    "shot_groups": "[[0,2],[2,4]]",
                    "source_shot_segments": "[[0,2],[0,2]]",
                    "source_frame_indices_by_shot": "[[0,1],[2,3]]",
                },
            },
        ],
    )

    config = MediaReadyConfig.from_mapping(_WASH_CONTRACT)
    paths = load_media_ready_wash_paths(
        {
            "inputs": {
                "keep_dir": str(keep_dir),
                "shot2story_window_candidates": str(tmp_path / "s2s.jsonl"),
                "ci_vid_window_candidates": str(tmp_path / "ci.jsonl"),
                "cine_pool_dirs": [str(pool)],
            },
            "output_dir": str(tmp_path / "out"),
        }
    )
    summary = run_media_ready_coarse_wash(
        config=config,
        paths=paths,
        workers=1,
        overwrite=False,
        expected_keep_rows=5,
    )

    assert summary["decode_valid_n"] == 2
    assert summary["by_dataset"] == {"ci_vid": 1, "cine250k": 1}
    assert summary["by_mapping_kind"] == {"multisource": 2}
    assert summary["lane_counts"][LANE_EVENTFUL_TRANSITION] == 1
    assert summary["lane_counts"][LANE_GENERIC_MULTISHOT] == 1
    assert summary["rejected_by_reason"] == {
        "missing_join_key": 1,
        "missing_or_empty_media": 2,
    }
    accepted = list(csv.DictReader((tmp_path / "out" / "accepted.csv").open(encoding="utf-8")))
    assert {row["case_id"] for row in accepted} == {"ci_ok", "cine_ok"}
    by_id = {row["case_id"]: row for row in accepted}
    assert by_id["ci_ok"]["mapping_kind"] == "multisource"
    assert by_id["ci_ok"]["lane"] == LANE_EVENTFUL_TRANSITION
    assert by_id["cine_ok"]["mapping_kind"] == "multisource"
    assert by_id["cine_ok"]["lane"] == LANE_GENERIC_MULTISHOT
    assert by_id["ci_ok"]["training_allowed"] == "false"
    assert by_id["ci_ok"]["training_admission"] == "mechanical_holdout"
    assert "annotation_model" not in by_id["ci_ok"]
    rejected = [
        json.loads(line)
        for line in (tmp_path / "out" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["record_id"]: row["reason"] for row in rejected} == {
        "ci_missing_clip": "missing_or_empty_media",
        "cine_absent_pool": "missing_join_key",
        "cine_missing_video": "missing_or_empty_media",
    }


def test_probe_worker_is_module_level_and_picklable() -> None:
    """ProcessPoolExecutor can only dispatch the probe if it pickles by reference."""

    import pickle

    assert pickle.loads(pickle.dumps(probe_decode_media_ready_row)) is (
        probe_decode_media_ready_row
    )


def test_probe_worker_types_decode_failure_as_reject(tmp_path: Path) -> None:
    compiled = {
        "case_id": "probe:absent",
        "dataset": "shot2story",
        "mapping_schema": "",
        "video": str(tmp_path / "absent.mp4"),
        "source_frame_indices": json.dumps([0, 1, 2, 3]),
    }

    ok, reason = probe_decode_media_ready_row((compiled, None))

    assert ok is False
    assert reason.startswith("decode:")


def test_probe_chunksize_stays_bounded() -> None:
    assert _probe_chunksize(0, 88) == 1
    assert _probe_chunksize(500, 88) == 1
    assert _probe_chunksize(206_399, 88) == 8
    assert _probe_chunksize(206_399, 1) == 8


def _equivalence_fixture(tmp_path: Path, tag: str) -> tuple[MediaReadyConfig, object]:
    """Half decodable-path rows, half absent rows, so both reject stages are used."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    keep_dir = tmp_path / "keep"
    windows: list[dict[object, object]] = []
    keep_rows: list[dict[object, object]] = []
    for index in range(12):
        record_id = f"s2s_{index:02d}"
        present = index % 2 == 0
        video = tmp_path / f"clip_{index:02d}.mp4"
        if present:
            video.write_bytes(b"not really an mp4")
        keep_rows.append(
            {
                "record_id": record_id,
                "dataset": "shot2story",
                "shot_count": 2,
                "state_candidate": index % 3 == 0,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "stable",
            }
        )
        windows.append(
            {
                "window_id": record_id,
                "video_path": str(video),
                "plan": {
                    "shot_groups": [[0, 2], [2, 4]],
                    "frame_indices": [0, 1, 2, 3],
                },
            }
        )
    _write_jsonl(keep_dir / "shot2story.jsonl", keep_rows)
    _write_jsonl(keep_dir / "ci_vid.jsonl", [])
    _write_jsonl(keep_dir / "cine250k.jsonl", [])
    s2s = tmp_path / "s2s.jsonl"
    _write_jsonl(s2s, windows)
    ci = tmp_path / "ci.jsonl"
    _write_jsonl(ci, [])
    pool = tmp_path / "pool"
    _write_jsonl(pool / "ordinary-multishot.jsonl", [])

    config = MediaReadyConfig.from_mapping(_WASH_CONTRACT)
    paths = load_media_ready_wash_paths(
        {
            "inputs": {
                "keep_dir": str(keep_dir),
                "shot2story_window_candidates": str(s2s),
                "ci_vid_window_candidates": str(ci),
                "cine_pool_dirs": [str(pool)],
            },
            "output_dir": str(tmp_path / f"out_{tag}"),
        }
    )
    return config, paths


_VOLATILE_SUMMARY_FIELDS = ("generated_at", "outputs")


def test_process_pool_probe_matches_serial_probe(tmp_path: Path) -> None:
    """workers>1 now forks processes; accept count and reject histogram must not move."""

    serial_config, serial_paths = _equivalence_fixture(tmp_path / "serial", "serial")
    pooled_config, pooled_paths = _equivalence_fixture(tmp_path / "pooled", "pooled")

    serial = run_media_ready_coarse_wash(
        config=serial_config,
        paths=serial_paths,
        workers=1,
        overwrite=False,
        expected_keep_rows=12,
    )
    pooled = run_media_ready_coarse_wash(
        config=pooled_config,
        paths=pooled_paths,
        workers=4,
        overwrite=False,
        expected_keep_rows=12,
    )

    for field in _VOLATILE_SUMMARY_FIELDS:
        serial.pop(field)
        pooled.pop(field)
    serial.pop("inputs")
    pooled.pop("inputs")

    assert pooled == serial
    assert serial["rejected_n"] == 12 - serial["decode_valid_n"]
    assert serial["rejected_by_reason"]["missing_or_empty_media"] == 6

    serial_rejects = [
        json.loads(line)
        for line in (tmp_path / "serial" / "out_serial" / "rejected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    pooled_rejects = [
        json.loads(line)
        for line in (tmp_path / "pooled" / "out_pooled" / "rejected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["reason"] for row in pooled_rejects] == [
        row["reason"] for row in serial_rejects
    ]
    assert [row["record_id"] for row in pooled_rejects] == [
        row["record_id"] for row in serial_rejects
    ]


def test_process_pool_probe_propagates_non_decode_bugs(tmp_path: Path) -> None:
    """A malformed keep row is a bug, not a reject, even on the pooled path."""

    pytest.importorskip("decord")
    video = tmp_path / "ok.mp4"
    _tiny_mp4(video)

    keep_dir = tmp_path / "keep"
    _write_jsonl(
        keep_dir / "shot2story.jsonl",
        [
            {
                "record_id": "s2s_bad_captions",
                "dataset": "shot2story",
                "shot_count": 2,
                "state_candidate": False,
                "generic_multishot_candidate": True,
                "transition_candidate": False,
                "event_status": "stable",
                "shot_captions": ["only one caption for two shots"],
            }
        ],
    )
    _write_jsonl(keep_dir / "ci_vid.jsonl", [])
    _write_jsonl(keep_dir / "cine250k.jsonl", [])
    s2s = tmp_path / "s2s.jsonl"
    _write_jsonl(
        s2s,
        [
            {
                "window_id": "s2s_bad_captions",
                "video_path": str(video),
                "plan": {"shot_groups": [[0, 2], [2, 4]], "frame_indices": [0, 1, 2, 3]},
            }
        ],
    )
    ci = tmp_path / "ci.jsonl"
    _write_jsonl(ci, [])
    pool = tmp_path / "pool"
    _write_jsonl(pool / "ordinary-multishot.jsonl", [])

    config = MediaReadyConfig.from_mapping(_WASH_CONTRACT)
    paths = load_media_ready_wash_paths(
        {
            "inputs": {
                "keep_dir": str(keep_dir),
                "shot2story_window_candidates": str(s2s),
                "ci_vid_window_candidates": str(ci),
                "cine_pool_dirs": [str(pool)],
            },
            "output_dir": str(tmp_path / "out"),
        }
    )

    with pytest.raises(ValueError, match="shot_captions length"):
        run_media_ready_coarse_wash(
            config=config,
            paths=paths,
            workers=4,
            overwrite=False,
            expected_keep_rows=1,
        )


def test_rejection_reasons_come_from_a_code_not_from_the_message_text() -> None:
    """归类走 ``reason_code``，不走异常消息的字面。

    此前是 ``"shot_count=" in str(error)`` / ``"below min_rgb_frames" in ...``。
    改一句话，分母解释就静默从 ``shot_count_out_of_bounds`` 滑到
    ``attribute_incomplete``——accepted 集合不变，但 ``rejected.jsonl`` 错了，
    而它正是用来解释分母的账本。

    四个 reason 的**字面取值不能变**：分母账本已经按这些名字发布过。
    """

    from mspc.data.multishot_media_ready import MediaReadyAttributeError

    error = MediaReadyAttributeError("x: shot_count=9 outside [2, 5]", reason_code="shot_count_out_of_bounds")
    assert isinstance(error, ValueError)
    assert error.reason_code == "shot_count_out_of_bounds"

    # 消息改写不影响归类——这正是修复要达到的效果
    reworded = MediaReadyAttributeError("完全不同的一句话", reason_code="shot_count_out_of_bounds")
    assert reworded.reason_code == error.reason_code
