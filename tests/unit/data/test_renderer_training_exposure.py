import json

import pytest

from mspc.data.renderer_training_members import summarize_exposure
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA


@pytest.fixture(autouse=True)
def isolated_frame_plans(monkeypatch):
    """这里只测重复曝光计数，源时间到目标格由真实媒体计划测试覆盖。"""
    monkeypatch.setattr('mspc.data.renderer_training_members.member_target_frame_counts',
                        lambda member, *, media: tuple(shot['frames'] for shot in member['shots']))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exposure_counts_repeated_and_overlapping_segments(tmp_path):
    members = tmp_path / "members.jsonl"
    sampling = tmp_path / "sampling.jsonl"
    write_rows(members, [{"case_id": "block", "shots": [
        {"frames": 16, "seconds": 1.1, "usable": True},
        {"frames": 32, "seconds": 2.2, "usable": True},
    ]}])
    write_rows(sampling, [
        {"sample_index": 0, "case_id": "block", "span": [0, 2]},
        {"sample_index": 1, "case_id": "block", "span": [0, 2]},
        {"sample_index": 2, "case_id": "block", "span": [1, 2]},
    ])

    result = summarize_exposure(members=members, sampling=sampling, media=ANNOTATION_MEDIA)

    assert result["sampled_segments"] == 3
    assert result["unique_case_span_segments"] == 2
    assert result["repeated_segment_draws"] == 1
    assert result["target_shot_presentations"] == 5
    assert result["unique_target_shots"] == 2
    assert result["target_frame_presentations"] == 128
    assert result["target_grid_hours"] == pytest.approx(8 / 3600)
    assert result["source_shot_hours_with_repetition"] == pytest.approx(8.8 / 3600)


def test_exposure_rejects_missing_sampled_member(tmp_path):
    members = tmp_path / "members.jsonl"
    sampling = tmp_path / "sampling.jsonl"
    write_rows(members, [])
    write_rows(sampling, [{"sample_index": 0, "case_id": "missing", "span": [0, 1]}])

    with pytest.raises(ValueError, match="sampled members missing"):
        summarize_exposure(members=members, sampling=sampling, media=ANNOTATION_MEDIA)
