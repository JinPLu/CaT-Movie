"""Ablation (without per-reference entity identity lines): replaces registered entity descriptions per reference with a fixed role label.

Overrides ``model.memory_prefix.row_policy`` from ``frame_entities_v1`` to ``fixed_role_v1`` and disables ``reference_entities``.
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_inputs = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
           "renderer-wan22-final-20261113/inputs")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_fixed_role_rows",
                       "model_key": "wan22_ti2v_5b_final_704p_fixed_role_rows"}
_config["model"]["memory_prefix"]["row_policy"] = "fixed_role_v1"
_config["dataset"]["reference_entities"] = None
_config["training"]["validation"]["reference_entities"] = None
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission-fixed-role-rows.json"
_config["training"]["validation"]["text_admission_path"] = f"{_inputs}/text/val-text-admission-fixed-role-rows.json"
_config["model"]["text_context_cache"] = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
                                          "text-cache-setting/fixed-role-rows/context-cache/manifest.json")
EXPERIMENT_CONFIG = _config
