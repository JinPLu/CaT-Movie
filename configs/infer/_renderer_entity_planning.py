"""Shared inference helper inheriting media, sampling, and model settings from a training config."""
from copy import deepcopy
from pathlib import Path
import runpy


def consumer_config(training_file: str, run_id: str | None, step: int | None) -> dict:
    """Use run_id=None and step=None for native per-shot baseline; otherwise specify checkpoint run_id and step."""
    repo = Path(__file__).resolve().parents[2]
    config = deepcopy(runpy.run_path(str(repo / "configs/train" / training_file))["EXPERIMENT_CONFIG"])
    arm = config["identity"]["run_name"]
    # Text context cache is used only during training; inference encodes prompts dynamically.
    config["model"].pop("text_context_cache", None)
    if (run_id is None) != (step is None):
        raise ValueError("native initialization requires explicit run_id=None and step=None together")
    if run_id is None:
        checkpoint = None
        output = str(Path(config["output"]["root"]).parent)
        config["model"].update(self_attention_policy="full", temporal_position_mode="global_time_native_rope_v1",
                               memory_prefix=None)
    else:
        # Checkpoints may come from periodic saves or full-state stops; existence is checked during prepare.
        if type(step) is not int or not 0 < step <= config["training"]["max_train_steps"]:
            raise ValueError(f"{training_file} checkpoint step must be a positive integer up to "
                             f"{config['training']['max_train_steps']}; got step {step!r}")
        if not isinstance(run_id, str) or not run_id.startswith(arm + "__"):
            raise ValueError("final checkpoint run_id must belong to this consumer")
        checkpoint = f"{config['output']['root']}/{run_id}/checkpoints/step-{step}.safetensors"
    root = "${MSPC_SHARED_ROOT}"
    if run_id is not None:
        output = f"{root}/Storyboard/outputs/experiments/renderer-entity-planning-dev30-20261102"
    unused_csv = f"{root}/Storyboard/outputs/experiments/layout-baseline-dev30/schema-required-inference.csv"
    # render_story loads an explicit storyboard JSON directly via the multishot_csv inference schema.
    media = config["task"]
    config["dataset"] = {"type": "multishot_csv", "schema": "multishot_csv_explicit_mapping_v3_tail_padding",
        "prompt_format": "caption_json_v1", "metadata_path": unused_csv, "video_column": "video",
        "caption_column": "caption_path", "shot_groups_column": "shot_groups", "case_id_column": "case_id",
        "case_id_prefix": "unused", "repeat": 1}
    config["runtime"]["deepspeed"] = None
    config["training"].pop("validation")
    config["training"].pop("step_unit")
    if "state_checkpoint" in config["training"]:
        del config["training"]["state_checkpoint"]
    if "workload_cost" in config["training"]:
        config["training"]["workload_cost"] = None
    config["training"]["loss_tracking"] = None
    config["dataloader"]["dataset_num_workers"] = 0
    config["timing"] = {"log_steps": 1, "log_every": 0, "sync_cuda": False}
    config["output"] = {"root": f"{output}/inference/raw/{arm}"}
    config["inference"] = {**config["inference"], "enabled": True, "device": "cuda", "target_height": media["height"],
        "target_width": media["width"], "fps": media["target_fps"],
        "test_csv_path": unused_csv,
        "output_dir": f"{output}/inference/raw/{arm}",
        "weight_source": "pretrained_model_paths" if run_id is None else "full_dit_checkpoint",
        "dit_checkpoint": checkpoint}
    config["checkpoint"] = None
    config["precompute"] = None
    return config
