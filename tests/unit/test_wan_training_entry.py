from __future__ import annotations

from mspc.config import parse_experiment_mapping
from mspc.core.component_registry import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
)
from mspc.training.training_args import build_common_training_args
from tests.unit.config.test_experiment_config import experiment_mapping


def config(tmp_path):
    return parse_experiment_mapping(
        experiment_mapping(),
        source_path=tmp_path / "wan22_diffsynth.yaml",
    )


def test_components_are_selected_independently(tmp_path):
    data = config(tmp_path).resolved
    assert MODEL_REGISTRY.modules[data["model"]["type"]].endswith(".wan22")
    assert DATASET_REGISTRY.modules[data["dataset"]["type"]].endswith(".multishot")
    assert TRAINER_REGISTRY.modules[data["trainer"]].endswith(".loop")


def test_common_training_args_are_component_neutral(tmp_path):
    args = build_common_training_args(config(tmp_path))
    assert args.training_seed == 42
    assert args.per_device_batch_size == 1
    assert args.dataloader_split_batches is False
    assert args.dataloader_drop_last is True
    assert args.output_root == str(tmp_path / "outputs")
    assert args.run_root is None
    assert not hasattr(args, "resume_mode")
