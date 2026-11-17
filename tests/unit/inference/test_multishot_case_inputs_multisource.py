from __future__ import annotations

import json
from pathlib import Path

import pytest

from mspc.data.frame_planner import FpsAwareTargetFpsPlanner
from mspc.data.multishot import (
    MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
    PROMPT_FORMAT_CAPTION_JSON_V1,
)
from mspc.data.shot_boundary_revision import SHOT_BOUNDARY_REVISION_VERSION
from tests.support.multisource_mapping import build_unscaled_multisource_mapping
from mspc.inference.common import (
    build_multishot_case_inputs,
    generated_eval_manifest_row,
)


def _multisource_case(tmp_path: Path) -> tuple[dict[str, str], FpsAwareTargetFpsPlanner]:
    source_video_paths = [tmp_path / "shot0.mp4", tmp_path / "shot1.mp4"]
    for path in source_video_paths:
        path.write_bytes(b"fixture")
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps(
            {
                "global_caption": "A two-shot story.",
                "shot0": "The first action.",
                "shot1": "The second action.",
            }
        ),
        encoding="utf-8",
    )
    planner = FpsAwareTargetFpsPlanner(
        max_rgb_frames=308,
        target_fps=15,
        vae_temporal_stride=4,
    )
    mapping = build_unscaled_multisource_mapping(
        source_video_paths=[str(path) for path in source_video_paths],
        source_shot_segments=[[0, 40], [0, 40]],
        source_fps_by_shot=[15.0, 15.0],
        source_num_frames_by_shot=[40, 40],
        planner=planner,
        shot_cut_evidence=[
            {"version": SHOT_BOUNDARY_REVISION_VERSION, "source_video_path": str(path),
             "source_shot_segment": [0, 40], "source_fps": 15.0, "content_diff_cuts": [], "transnet_cuts": []}
            for path in source_video_paths
        ],
        label="ci_vid:k2",
    )
    assert mapping is not None
    row = {
        "multishot_video": str(source_video_paths[0]),
        "caption_path": str(caption_path),
        "shot_groups": json.dumps(mapping["shot_groups"]),
        "source_shot_segments": json.dumps(mapping["source_shot_segments"]),
        "source_video_paths": json.dumps(mapping["source_video_paths"]),
        "source_frame_indices_by_shot": json.dumps(mapping["frame_indices_by_shot"]),
    }
    return row, planner


def _build_case_inputs(
    row: dict[str, str],
    planner: FpsAwareTargetFpsPlanner,
) -> dict[str, object]:
    return build_multishot_case_inputs(
        row,
        planner,
        video_column="multishot_video",
        caption_column="caption_path",
        shot_groups_column="shot_groups",
        schema=MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
        row_label="ci_vid:k2",
        prompt_format=PROMPT_FORMAT_CAPTION_JSON_V1,
        read_video_metadata_fn=lambda path: {
            "source_fps": 15.0,
            "source_num_frames": 40,
        },
    )


def test_multisource_case_inputs_k2_happy_path(tmp_path: Path) -> None:
    row, planner = _multisource_case(tmp_path)

    case_inputs = _build_case_inputs(row, planner)

    assert len(case_inputs["latent_shot_groups"]) == 2
    assert len(case_inputs["source_video_paths"]) == 2
    assert len(case_inputs["frame_indices_by_shot"]) == 2
    manifest_row = generated_eval_manifest_row(
        case_id="ci_vid:k2",
        output_path=tmp_path / "generated.mp4",
        generation_paradigm="wan_test",
        case_inputs=case_inputs,
    )
    assert manifest_row["shot_count"] == len(manifest_row["source_video_paths"]) == 2
    assert manifest_row["frame_indices_by_shot"] == case_inputs["frame_indices_by_shot"]


def test_multisource_case_inputs_missing_clip_fails_loud(tmp_path: Path) -> None:
    row, planner = _multisource_case(tmp_path)
    source_video_paths = json.loads(row["source_video_paths"])
    Path(source_video_paths[1]).unlink()

    with pytest.raises(FileNotFoundError):
        _build_case_inputs(row, planner)


@pytest.mark.parametrize("invalid_value", ["[]", '[""]'])
def test_multisource_case_inputs_rejects_missing_source_paths(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    row, planner = _multisource_case(tmp_path)
    row["source_video_paths"] = invalid_value

    with pytest.raises(ValueError, match="source_video_paths"):
        _build_case_inputs(row, planner)


def test_multisource_case_inputs_empty_clip_fails_loud(tmp_path: Path) -> None:
    row, planner = _multisource_case(tmp_path)
    source_video_paths = json.loads(row["source_video_paths"])
    Path(source_video_paths[1]).write_bytes(b"")

    with pytest.raises(ValueError, match="empty source video"):
        _build_case_inputs(row, planner)


def test_multisource_case_inputs_rejects_shot2story_prompt_source(
    tmp_path: Path,
) -> None:
    row, planner = _multisource_case(tmp_path)

    with pytest.raises(ValueError, match="shot2story_prompt_source is not supported"):
        build_multishot_case_inputs(
            row,
            planner,
            video_column="multishot_video",
            caption_column=None,
            shot_groups_column="shot_groups",
            schema=MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
            row_label="ci_vid:k2",
            prompt_format=PROMPT_FORMAT_CAPTION_JSON_V1,
            shot2story_prompt_source={"kind": "shot2story_raw"},
            read_video_metadata_fn=lambda path: {
                "source_fps": 15.0,
                "source_num_frames": 40,
            },
        )
