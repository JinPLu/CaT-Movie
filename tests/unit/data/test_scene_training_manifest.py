"""R9先跑200真实窗之后固定的边界行，不以合成影片替代真实坐标。"""
import json
from pathlib import Path

import pytest

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data.scene_training_manifest import (
    audit_row, manifest_fields, manifest_row, source_slice_row,
)

REAL = json.loads((Path(__file__).parent / "fixtures/scene_manifest_real.json").read_text())["rows"]


@pytest.mark.parametrize("case", REAL, ids=lambda c: str(c["slice_index"]))
def test_real_source_mapping_and_crop(case):
    row = manifest_fields(case["window"], media=ANNOTATION_MEDIA)
    assert {key: row[key] for key in case["expected_metadata"]} == case["expected_metadata"]
    assert {key: row[key] for key in ("target_height", "target_width", "target_fps", "vae_temporal_stride", "max_rgb_frames")} == {
        "target_height": 480, "target_width": 832, "target_fps": 16, "vae_temporal_stride": 4, "max_rgb_frames": 308}
    assert [len(p) for p in row["source_frame_indices_by_shot"]] == [s["render_frames"] for s in case["window"]["shots"]]


def test_preroll_is_diagnostic_and_endpoints_stay_guarded():
    case = next(c for c in REAL if c["slice_index"] == 62)
    assert case["window"]["container_preroll_frames"] == 42
    row = manifest_fields(case["window"], media=ANNOTATION_MEDIA)
    for shot, grid in zip(case["window"]["boundary_revision"]["shots"], row["source_frame_indices_by_shot"], strict=True):
        assert [grid[0], grid[-1]] == [shot["revised_segment"][0], shot["revised_segment"][1]-1]


def test_real_crop_rounding_discrepancy_is_reported():
    row = manifest_fields(next(c for c in REAL if c["slice_index"] == 4)["window"], media=ANNOTATION_MEDIA)
    assert row["train_crop_box"] == [265, 0, 1655, 802]
    checks = audit_row(row)["checks"]
    assert checks["crop_inside_active"] and checks["crop_even_size"]
    assert not checks["crop_exact_832_480"]
    assert not checks["crop_even_coordinates"]


def test_short_shot_and_near_capacity_are_preserved():
    short = manifest_fields(next(c for c in REAL if c["slice_index"] == 12)["window"], media=ANNOTATION_MEDIA)
    near = manifest_fields(next(c for c in REAL if c["slice_index"] == 109)["window"], media=ANNOTATION_MEDIA)
    assert min(map(len, short["source_frame_indices_by_shot"])) == 9
    assert sum(map(len, near["source_frame_indices_by_shot"])) == 305


def test_source_slice_preserves_contract_metadata_and_annotation_id():
    row = manifest_row(REAL[0]["window"], mapping_schema="explicit-test-contract", media=ANNOTATION_MEDIA)
    result = source_slice_row(row, annotation_window_id="trial-0000")
    assert result["metadata"] == row
    assert result["window_id"] == "trial-0000"
    assert result["case_id"] == REAL[0]["window"]["window_id"]
    assert result["frame_counts"] == [25, 29, 89, 29]


@pytest.mark.parametrize("schema", [None, "", " "])
def test_missing_contract_schema_raises(schema):
    with pytest.raises(ValueError, match="mapping_schema"):
        manifest_row(REAL[0]["window"], mapping_schema=schema, media=ANNOTATION_MEDIA)
