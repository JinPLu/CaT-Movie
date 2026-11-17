from __future__ import annotations

import json
from pathlib import Path

import pytest

from mspc.config.experiment import (
    ExperimentConfigError,
    canonical_json,
    load_experiment_config,
    parse_experiment_mapping,
)


DIFFSYNTH_ROOT = "/srv/storyboard/third_party/wan22_runtime"


def experiment_mapping() -> dict:
    model = {
        "type": "wan22_diffsynth",
        "diffsynth_root": DIFFSYNTH_ROOT,
        "temporal_position_mode": "global_time_native_rope_v1",
        "self_attention_policy": "full",
        "paths": {
            "dit": ["/srv/models/wan22/dit-00001.safetensors"],
            "t5": "/srv/models/wan22/t5.pth",
            "vae": "/srv/models/wan22/vae.pth",
            "tokenizer": "/srv/models/wan22/tokenizer",
        },
    }
    training = {
        "shot_forward_mode": "joint",
        "seed": 42,
        "task": "sft",
        "mixed_precision": "no",
        "scheduler_train_timesteps": 1000,
        "flow_shift": 5.0,
        "num_epochs": 1,
        "save_steps": 500,
        "max_train_steps": None,
        "gradient_accumulation_steps": 1,
        "use_gradient_checkpointing": True,
        "activation_checkpointing": "recompute_all",
        "find_unused_parameters": False,
        "dit_parameter_freeze": {"recipe": "full_dit_v1"},
    }
    inference = {
        "enabled": True,
        "device": "cuda",
        "target_height": 480,
        "target_width": 832,
        "fps": 16,
        "seed": 42,
        "negative_prompt": "negative",
        "num_inference_steps": 50,
        "sigma_shift": 5.0,
        "cfg_scale": 5.0,
        "video_quality": 5,
        "sampler": "flow_unipc_official",
        "test_csv_path": "data/eval.csv",
        "output_dir": "outputs/inference",
        "weight_source": "full_dit_checkpoint",
        "dit_checkpoint": "checkpoints/step-500.safetensors",
    }
    pipeline_inputs = {
        "tiled": False,
        "tile_size": [30, 52],
        "tile_stride": [15, 26],
    }
    return {
        "schema_version": 1,
        "kind": "train",
        "trainer": "sft",
        "identity": {
            "model_key": "model",
            "dataset_key": "data",
            "run_name": "run-name",
            "experiment_key": "model__data",
        },
        "task": {
            "paradigm": "text_to_multishot_video",
            "height": 480,
            "width": 832,
            "frame_policy": "fps_aware_target_fps",
            "max_rgb_frames": 308,
            "target_fps": 16,
            "vae_temporal_stride": 4,
        },
        "runtime": {
            "python": "/env/bin/python",
            "distributed_timeout_seconds": 3600,
            "deepspeed": {"zero_optimization": {"stage": 2, "contiguous_gradients": False,
                                                              "reduce_bucket_size": 64_000_000,
                                                              "allgather_bucket_size": 64_000_000,
                                                              "overlap_comm": False}, "gradient_clipping": 1.0,
                          "wall_clock_breakdown": True,
                          "fp16": {"enabled": False}, "bf16": {"enabled": False},
                          "torch_autocast": {"enabled": True, "dtype": "bfloat16",
                                             "lower_precision_safe_modules": []}},
            "hf_cache": "/cache",
            "offline_hf": True,
        },
        "environment": {
            "tokenizers_parallelism": "false",
            "python_unbuffered": "1",
            "wandb_mode": "disabled",
        },
        "nccl": {
            "debug": "WARN",
            "async_error_handling": "1",
            "device_max_connections": "32",
            "blocking_wait": "0",
            "buffsize": "8388608",
            "nthreads": "512",
        },
        "launch": {
            "train_entrypoint": "train.py",
            "infer_entrypoint": "infer.py",
            "train_pythonpath": ["src", DIFFSYNTH_ROOT],
            "infer_pythonpath": ["src", DIFFSYNTH_ROOT],
            "cuda_visible_devices": None,
        },
        "model": model,
        "dataset": {
            "type": "multishot_csv",
            "schema": "multishot_csv_v1",
            "prompt_format": "caption_json_v1",
            "metadata_path": "data/train.csv",
            "video_column": "multishot_video",
            "caption_column": "gemini_caption",
            "shot_groups_column": "shot_groups",
            "case_id_column": None,
            "case_id_prefix": "shot2story",
            "repeat": 1,
        },
        "training": training,
        "pipeline_inputs": pipeline_inputs,
        "optimization": {
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
            "lr_scheduler_factor": 1.0,
            "lr_scheduler_total_iters": 1,
        },
        "dataloader": {
            "per_device_batch_size": 1,
            "split_batches": False,
            "drop_last": True,
            "dataset_num_workers": 4,
            "video_decode_num_threads": 1,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
        },
        "timing": {"log_steps": 50, "log_every": 0, "sync_cuda": False},
        "output": {"root": "outputs"},
        "inference": inference,
        "checkpoint": None,
    }


def test_parse_active_component_contract(tmp_path: Path) -> None:
    parsed = parse_experiment_mapping(
        experiment_mapping(),
        source_path=tmp_path / "experiment.yaml",
    )
    data = parsed.to_dict()
    assert data["model"]["type"] == "wan22_diffsynth"
    assert data["dataset"]["type"] == "multishot_csv"
    assert data["trainer"] == "sft"
    assert data["output"] == {"root": str(tmp_path / "outputs")}
    assert not hasattr(parsed, "resolved_config_hash")


def test_scheduler_factor_cannot_silently_scale_the_configured_lr(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"]["lr_scheduler_factor"] = 0.5
    with pytest.raises(ExperimentConfigError, match="exactly 1.0"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "scaled-lr.json")


def test_checkpoint_is_the_only_optional_training_input(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["checkpoint"] = "checkpoints/step-500.safetensors"
    parsed = parse_experiment_mapping(
        mapping, source_path=tmp_path / "experiment.yaml"
    )
    assert parsed.to_dict()["checkpoint"] == str(
        tmp_path / "checkpoints/step-500.safetensors"
    )

    mapping["resume"] = {"mode": "never", "checkpoint": None}
    with pytest.raises(ExperimentConfigError, match="unknown fields"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize(
    "field",
    ["gradient_accumulation_steps", "use_gradient_checkpointing", "activation_checkpointing"],
)
def test_training_requires_explicit_checkpointing_fields(
    tmp_path: Path,
    field: str,
) -> None:
    mapping = experiment_mapping()
    mapping["training"].pop(field)
    with pytest.raises(ExperimentConfigError, match="missing required fields"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_python_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "experiment.py"
    path.write_text(
        "EXPERIMENT_CONFIG = " + repr(experiment_mapping()) + "\n"
        "identty = {'run_name': 'typo'}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentConfigError, match="unknown top-level keys: \\['identty'\\]"
    ):
        load_experiment_config(path)


def test_training_allows_explicitly_disabling_gradient_checkpointing(
    tmp_path: Path,
) -> None:
    mapping = experiment_mapping()
    mapping["training"]["use_gradient_checkpointing"] = False

    resolved = parse_experiment_mapping(
        mapping, source_path=tmp_path / "experiment.yaml"
    ).resolved

    assert resolved["training"]["use_gradient_checkpointing"] is False


@pytest.mark.parametrize("policy", ["recompute_all", "save_self_attention"])
def test_training_preserves_the_explicit_activation_checkpointing_policy(tmp_path: Path, policy: str) -> None:
    mapping = experiment_mapping()
    mapping["training"]["activation_checkpointing"] = policy
    parsed = parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")
    assert parsed.resolved["training"]["activation_checkpointing"] == policy


@pytest.mark.parametrize("policy", [None, False, 0, "save_attention", ""])
def test_training_rejects_unknown_or_untyped_activation_checkpointing_policy(tmp_path: Path, policy) -> None:
    mapping = experiment_mapping()
    mapping["training"]["activation_checkpointing"] = policy
    with pytest.raises(ExperimentConfigError, match="training.activation_checkpointing"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_saving_self_attention_requires_gradient_checkpointing(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["training"].update(activation_checkpointing="save_self_attention", use_gradient_checkpointing=False)
    with pytest.raises(ExperimentConfigError, match="save_self_attention.*requires.*use_gradient_checkpointing"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize("layers", [None, 1, 20, 30])
def test_self_attention_cache_layer_count_is_explicit(tmp_path: Path, layers) -> None:
    mapping = experiment_mapping()
    mapping["training"].update(activation_checkpointing="save_self_attention",
                               self_attention_cache_layers=layers)
    resolved = parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml").resolved
    assert resolved["training"]["self_attention_cache_layers"] == layers


@pytest.mark.parametrize("layers", [0, -1, 31, True, 20.0, "20", float("inf"), float("nan")])
def test_self_attention_cache_layer_count_rejects_invalid_values(tmp_path: Path, layers) -> None:
    mapping = experiment_mapping()
    mapping["training"].update(activation_checkpointing="save_self_attention",
                               self_attention_cache_layers=layers)
    with pytest.raises(ExperimentConfigError, match="training.self_attention_cache_layers"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_partial_self_attention_cache_requires_save_policy(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["training"]["self_attention_cache_layers"] = 20
    with pytest.raises(ExperimentConfigError, match="self_attention_cache_layers requires"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize("value", [0, 2])
def test_dataloader_rejects_batching_multiple_complete_samples_per_device(
    tmp_path: Path,
    value: int,
) -> None:
    mapping = experiment_mapping()
    mapping["dataloader"]["per_device_batch_size"] = value
    with pytest.raises(
        ExperimentConfigError,
        match="dataloader.per_device_batch_size must be (?:>= 1|1 because)",
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_batches", True, "split_batches must be false"),
        ("drop_last", False, "drop_last must be true"),
    ],
)
def test_dataloader_requires_complete_world_size_batches(
    tmp_path: Path,
    field: str,
    value: bool,
    message: str,
) -> None:
    mapping = experiment_mapping()
    mapping["dataloader"][field] = value
    with pytest.raises(ExperimentConfigError, match=message):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_decode_threads_require_a_positive_integer(tmp_path: Path, value) -> None:
    mapping = experiment_mapping()
    mapping["dataloader"]["video_decode_num_threads"] = value
    with pytest.raises(ExperimentConfigError, match="video_decode_num_threads"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_decode_threads_are_explicit(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    del mapping["dataloader"]["video_decode_num_threads"]
    with pytest.raises(ExperimentConfigError, match="missing required fields.*video_decode_num_threads"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_training_accepts_gradient_accumulation(
    tmp_path: Path,
) -> None:
    mapping = experiment_mapping()
    mapping["training"]["gradient_accumulation_steps"] = 2
    resolved = parse_experiment_mapping(
        mapping, source_path=tmp_path / "experiment.yaml"
    ).resolved

    assert resolved["training"]["gradient_accumulation_steps"] == 2


def test_training_config_does_not_validate_dataset_against_static_world_size(
    tmp_path: Path,
) -> None:
    mapping = experiment_mapping()
    resolved = parse_experiment_mapping(
        mapping, source_path=tmp_path / "experiment.yaml"
    ).resolved

    assert "rows" not in resolved["dataset"]


@pytest.mark.parametrize(
    "field",
    ["gpus_per_node", "nnodes", "node_rank", "master_addr", "master_port"],
)
def test_training_launch_rejects_dynamic_topology_fields(
    tmp_path: Path,
    field: str,
) -> None:
    mapping = experiment_mapping()
    mapping["launch"][field] = 1

    with pytest.raises(ExperimentConfigError, match="unknown fields"):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "unknown fields"),
        (lambda value: value.pop("trainer"), "missing required fields"),
        (lambda value: value.update(trainer="other"), "trainer must be one of"),
        (lambda value: value["model"].update(type="other"), "model.type must be one of"),
        (lambda value: value["dataset"].update(type="other"), "dataset.type must be one of"),
        (lambda value: value["training"].update(seed=-1), "training.seed must be >= 0"),
    ],
)
def test_strict_schema_rejects_invalid_component_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    mapping = experiment_mapping()
    mutation(mapping)
    with pytest.raises(ExperimentConfigError, match=message):
        parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")


def test_canonical_serialization_and_freeze(tmp_path: Path) -> None:
    parsed = parse_experiment_mapping(
        experiment_mapping(), source_path=tmp_path / "experiment.json"
    )
    assert canonical_json(parsed.to_dict()) == canonical_json(
        dict(reversed(list(parsed.to_dict().items())))
    )
    with pytest.raises(TypeError):
        parsed.resolved["runtime"]["python"] = "/other/python"


def test_loaders_share_the_same_schema(tmp_path: Path) -> None:
    json_path = tmp_path / "experiment.json"
    json_path.write_text(json.dumps(experiment_mapping()), encoding="utf-8")
    assert load_experiment_config(json_path).source_path == json_path

    python_path = tmp_path / "experiment.py"
    python_path.write_text(
        "EXPERIMENT_CONFIG = " + repr(experiment_mapping()), encoding="utf-8"
    )
    assert load_experiment_config(python_path).resolved["identity"]["run_name"] == "run-name"


def test_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="duplicate config key"):
        load_experiment_config(path)


def test_constant_with_warmup_declares_only_the_keys_it_reads(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"] = {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 300,
    }

    parsed = parse_experiment_mapping(mapping, source_path=tmp_path / "warmup.json")

    optimization = parsed.to_dict()["optimization"]
    assert optimization["warmup_steps"] == 300
    assert "lr_scheduler_factor" not in optimization
    assert "lr_scheduler_total_iters" not in optimization


def test_constant_with_warmup_requires_warmup_steps(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"] = {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
    }
    with pytest.raises(
        ExperimentConfigError,
        match=r"optimization is missing required fields: \['warmup_steps'\]",
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "warmup.json")


@pytest.mark.parametrize(
    "field, value",
    [("lr_scheduler_factor", 1.0), ("lr_scheduler_total_iters", 1)],
)
def test_constant_with_warmup_rejects_constant_only_fields(
    tmp_path: Path,
    field: str,
    value: float | int,
) -> None:
    mapping = experiment_mapping()
    mapping["optimization"] = {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 300,
        field: value,
    }
    with pytest.raises(
        ExperimentConfigError,
        match=rf"optimization contains unknown fields: \['{field}'\]",
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "warmup.json")


def test_constant_scheduler_rejects_warmup_steps(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"]["warmup_steps"] = 300
    with pytest.raises(
        ExperimentConfigError,
        match=r"optimization contains unknown fields: \['warmup_steps'\]",
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "constant.json")


def test_warmup_steps_must_be_a_non_negative_integer(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"] = {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": -1,
    }
    with pytest.raises(
        ExperimentConfigError,
        match="optimization.warmup_steps must be >= 0",
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "warmup.json")


def test_unknown_scheduler_type_names_the_supported_set(tmp_path: Path) -> None:
    mapping = experiment_mapping()
    mapping["optimization"]["lr_scheduler_type"] = "cosine"
    with pytest.raises(
        ExperimentConfigError,
        match=(
            r"must be one of \['constant', 'constant_with_warmup'\]"
        ),
    ):
        parse_experiment_mapping(mapping, source_path=tmp_path / "cosine.json")
