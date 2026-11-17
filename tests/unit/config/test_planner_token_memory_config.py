"""源实体与目标点接口均可消费同一显式参考前缀；不新增训练配方。"""
import json

import pytest

from mspc.config.experiment import ExperimentConfigError, parse_experiment_mapping
from tests.support.paths import REPO_ROOT

PATH = REPO_ROOT / "tests/fixtures/block_renderer_config.json"


def _mapping(*, source_points, target_points):
    raw = json.loads(PATH.read_text())
    raw["model"]["memory_prefix"]["row_policy"] = "frame_entities_v1"
    raw["dataset"]["reference_entities"] = {"labels_path": "/fixture/reference-entities.jsonl"}
    if source_points:
        raw["model"]["source_identity"] = {
            "loss_weight": 0.1, "alignment_layer": 14, "max_alignment_cells_per_reference": 32}
    if target_points:
        raw["model"]["planner_tokens"] = {"content": "person_index", "max_people": 32, "learning_rate": 1e-4}
        raw["dataset"]["planner_tokens"] = {
            "content": "person_index", "labels_path": "/fixture/target-points.jsonl",
            "point_jitter": 0.02, "point_dropout_p": 0.1, "token_dropout_p": 0.2}
    return raw


@pytest.mark.parametrize("source_points", [False, True])
@pytest.mark.parametrize("target_points", [False, True])
def test_source_target_interfaces_support_the_same_memory_prefix(source_points, target_points):
    parsed = parse_experiment_mapping(_mapping(source_points=source_points, target_points=target_points), source_path=PATH)
    model = parsed.resolved["model"]
    assert model["memory_prefix"]["frames_max"] == 10
    assert model["memory_prefix"]["attention_policy"] == "bidirectional_v1"
    assert (model["source_identity"] is not None) == source_points
    assert (model["planner_tokens"] is not None) == target_points


def test_target_points_still_require_full_video_attention():
    raw = _mapping(source_points=False, target_points=True)
    raw["model"]["self_attention_policy"] = "first_frame_interface"
    with pytest.raises(ExperimentConfigError, match="planner tokens go with self_attention_policy full"):
        parse_experiment_mapping(raw, source_path=PATH)


def _geometry_mapping():
    raw = _mapping(source_points=True, target_points=True)
    raw["model"]["planner_tokens"].update(content="person_geometry", max_people=64)
    raw["dataset"]["planner_tokens"].update(content="person_geometry", max_people=64,
                                            geometry_labels_path="/fixture/train-geometry.jsonl")
    raw["training"]["validation"] = {
        "members_path": "/fixture/validation.jsonl", "sampling_manifest_path": "/fixture/val-sampling.jsonl",
        "sample_budget": 8, "every_steps": 20, "draws": 2,
        "planner_tokens": {"labels_path": "/fixture/val-points.jsonl",
                           "geometry_labels_path": "/fixture/val-geometry.jsonl"},
        "reference_entities": {"labels_path": "/fixture/val-reference-entities.jsonl"}}
    return raw


def test_joint_geometry_config_declares_matching_capacity_and_separate_validation_labels():
    from mspc.training.training_args import build_common_training_args
    parsed = parse_experiment_mapping(_geometry_mapping(), source_path=PATH)
    args = build_common_training_args(parsed)
    assert parsed.resolved["model"]["planner_tokens"] == {
        "content": "person_geometry", "max_people": 64, "learning_rate": 1e-4}
    assert args.dataset_cfg["planner_tokens"]["geometry_labels_path"] == "/fixture/train-geometry.jsonl"
    assert args.validation["planner_tokens"]["geometry_labels_path"] == "/fixture/val-geometry.jsonl"


@pytest.mark.parametrize("change, message", [
    ("missing_labels", "geometry_labels_path"), ("different_capacity", "max_people differ"),
    ("missing_validation_geometry", "geometry_labels_path"),
    ("reused_validation_geometry", "cannot reuse"),
])
def test_geometry_configuration_cannot_guess_capacity_or_validation_inputs(change, message):
    raw = _geometry_mapping()
    if change == "missing_labels":
        del raw["dataset"]["planner_tokens"]["geometry_labels_path"]
    elif change == "different_capacity":
        raw["dataset"]["planner_tokens"]["max_people"] = 8
    elif change == "missing_validation_geometry":
        del raw["training"]["validation"]["planner_tokens"]["geometry_labels_path"]
    else:
        raw["training"]["validation"]["planner_tokens"]["geometry_labels_path"] = "/fixture/train-geometry.jsonl"
    with pytest.raises(ExperimentConfigError, match=message):
        parse_experiment_mapping(raw, source_path=PATH)
