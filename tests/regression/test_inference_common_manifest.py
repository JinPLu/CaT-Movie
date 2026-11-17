from __future__ import annotations

import sys
import types
import pytest
from mspc.config import parse_experiment_mapping
from tests.unit.config.test_experiment_config import experiment_mapping

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_data_stub = types.ModuleType("torch.utils.data")
    torch_data_stub.Dataset = object
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_data_stub

from mspc.inference.common import (
    build_inference_dataset_config,
    build_inference_frame_planner,
    generated_eval_manifest_row,
    infer_case_id,
)


def test_generated_eval_manifest_row_is_scorer_ready():
    row = generated_eval_manifest_row(
        case_id="case_001",
        output_path="/tmp/case_001.mp4",
        generation_paradigm="wan_test",
        case_inputs={
            "story_prompt": "A person prepares tea.",
            "shot_prompts": ["The kettle is on the table.", "The cup is filled."],
            "target_fps": 16.0,
            "max_rgb_frames": 128,
            "vae_temporal_stride": 4,
            "prompt_format": "caption_json_v1",
            "train_rgb_frames": 42,
            "train_latent_frames": 12,
            "shot_groups": [[0, 13], [13, 42]],
            "latent_shot_groups": [[0, 4], [4, 12]],
        },
        source_row={
            "story_id": "story_001",
            "entity_descriptions": '["person: blue coat"]',
            "conditioning_policy": "native_multishot",
            "source_dataset": "unit_test",
        },
    )

    assert row["case_id"] == "case_001"
    assert row["video_path"] == "/tmp/case_001.mp4"
    assert row["shot_paths"] == ["/tmp/case_001.mp4"]
    assert row["shot_count"] == 2
    assert row["generation_paradigm"] == "wan_test"
    assert row["story_id"] == "story_001"
    assert row["story_prompt"] == "A person prepares tea."
    assert row["shot_prompts"] == ["The kettle is on the table.", "The cup is filled."]
    assert row["entity_descriptions"] == ["person: blue coat"]
    assert row["conditioning_policy"] == "native_multishot"
    assert row["source_dataset"] == "unit_test"
    assert row["target_fps"] == 16.0
    assert row["max_rgb_frames"] == 128
    assert row["vae_temporal_stride"] == 4
    assert row["train_rgb_frames"] == 42
    assert row["train_latent_frames"] == 12
    assert row["shot_groups"] == [[0, 13], [13, 42]]
    assert row["latent_shot_groups"] == [[0, 4], [4, 12]]


def test_inference_dataset_config_allows_optional_case_id_column(tmp_path):
    mapping = experiment_mapping()
    mapping["dataset"]["case_id_column"] = None
    mapping["dataset"].update(
        video_column="video",
        caption_column="caption",
        shot_groups_column="shots",
    )
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "config.yaml")

    dataset_cfg = build_inference_dataset_config(config, require_case_id_column=False)

    assert dataset_cfg["video_column"] == "video"
    assert "case_id_column" not in dataset_cfg
    assert infer_case_id({}, row_index=7, dataset_cfg=dataset_cfg, source_label="cases.csv") == "7"


def test_inference_dataset_config_preserves_prompt_format(tmp_path):
    mapping = experiment_mapping()
    mapping["dataset"]["prompt_format"] = "caption_json_v1"
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "config.yaml")

    dataset_cfg = build_inference_dataset_config(config, require_case_id_column=False)

    assert dataset_cfg["prompt_format"] == "caption_json_v1"


def test_inference_dataset_config_requires_case_id_when_requested(tmp_path):
    mapping = experiment_mapping()
    mapping["dataset"]["case_id_column"] = "case_id"
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "config.yaml")

    dataset_cfg = build_inference_dataset_config(config, require_case_id_column=True)

    assert dataset_cfg["case_id_column"] == "case_id"
    assert infer_case_id({"case_id": "case_001"}, row_index=0, dataset_cfg=dataset_cfg, source_label="cases.csv") == "case_001"
    with pytest.raises(ValueError, match="empty case id"):
        infer_case_id({"case_id": ""}, row_index=0, dataset_cfg=dataset_cfg, source_label="cases.csv")


def test_inference_frame_planner_rejects_fps_mismatch():
    dataset_cfg = {
        "max_rgb_frames": 128,
        "target_fps": 16,
        "vae_temporal_stride": 4,
    }

    with pytest.raises(ValueError, match="inference.fps must equal"):
        build_inference_frame_planner(dataset_cfg, inference_fps=24)
