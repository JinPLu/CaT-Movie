"""Shared supervision and sampling contract for native-backbone joint training (see 480p and 720p entry configs).
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_joint_wan_layout.py")))["EXPERIMENT_CONFIG"])
_data = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_inputs = f"{_data}/two-backbone-training-preparation-20261108/inputs"
_sampling = f"{_inputs}/shot-count-weighted-windows"
_members = f"{_inputs}/entity-numbered-actions"
_config["identity"] = {**_config["identity"],
    "run_name": "joint_native_training", "model_key": "wan22_5b_joint_native_training",
    "experiment_key": "native_backbone_joint_training",
    "dataset_key": "v18_shot_count_weighted_complete_source_windows"}
_config["dataset"]["members_path"] = f"{_members}/train.jsonl"
_config["dataset"]["sampling_manifest_path"] = f"{_sampling}/train-sampling.jsonl"
_config["dataset"]["reference_policy"] = "random_frame"
_config["dataset"]["target_mask_weights"] = None
_config["dataset"]["reference_entities"] = None
_config["dataset"]["reference_row_dropout_p"] = 0.0
_config["dataset"]["picture_type_dropout_p"] = 0.5
_config["dataset"]["box_dropout_p"] = 0.5
# Sample random in-block reference frames with 0.2 full-prefix dropout while keeping visual pixels.
_config["model"]["memory_prefix"]["dropout_p"] = 0.2
# Continuous flow sigma calibrated for Wan timestep threshold (t <= 400).
_config["model"]["memory_prefix"]["noise"] = {"train_sigma_max": 0.4, "inference_sigma": 0.0}
_config["model"]["omit_target_text"] = []
_config["training"].update(max_train_steps=12000, save_steps=3000, state_checkpoint={"keep": 1},
    activation_checkpointing="recompute_all", self_attention_cache_layers=None, workload_cost=None)
_config["training"]["validation"].update(
    every_steps=500,
    intermediate_panel={"sample_budget": 32, "draws": 1},
    members_path=f"{_members}/val.jsonl",
    sampling_manifest_path=f"{_sampling}/val-sampling.jsonl",
    reference_entities=None)
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
                             "renderer-native-backbones-20261108/base/training"}
EXPERIMENT_CONFIG = _config
