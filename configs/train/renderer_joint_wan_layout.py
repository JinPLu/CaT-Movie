"""Joint Wan2.2-TI2V-5B recipe with first-frame sparse attention, routed in-grid references, full layout prompts, and duration-bucketed windows.
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(
    str(Path(__file__).with_name("renderer_joint_routed_references_full_pool.py")))["EXPERIMENT_CONFIG"])
_name = "joint_wan_layout"
_root = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_windows = f"{_root}/frame-window-training-inputs-20261106/joint"
_output = f"{_root}/renderer-final-recipe-20261106/{_name}"
_config["identity"] = {**_config["identity"], "run_name": _name, "model_key": f"wan22_5b_{_name}",
                       "experiment_key": "joint_generation_final_recipe"}
_config["model"]["omit_target_text"] = []
_config["dataset"]["picture_type_dropout_p"] = None
_config["dataset"]["sampling_manifest_path"] = f"{_windows}/train-sampling.jsonl"
_config["dataset"]["reference_entities"] = {"labels_path": f"{_windows}/train-reference-entities.jsonl"}
_config["training"]["max_train_steps"] = 12000
_config["training"]["save_steps"] = 3000
_config["output"] = {"root": f"{_output}/training"}
EXPERIMENT_CONFIG = _config
