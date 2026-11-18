"""Ablation (without visual reference prefix): removes memory reference frames while preserving the same backbone, data, and windowing.

Sets ``model.memory_prefix = None`` alongside ``reference_entities = None`` and ``reference_crops = None``.
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_experiment = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
               "renderer-wan22-final-20261113")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_without_references",
                       "model_key": "wan22_ti2v_5b_final_704p_without_references"}
_config["model"]["memory_prefix"] = None
_config["dataset"]["reference_entities"] = None
_config["dataset"]["reference_crops"] = None
_config["dataset"]["reference_row_dropout_p"] = None
_config["training"]["validation"]["reference_entities"] = None
_config["dataset"]["text_admission_path"] = f"{_experiment}/inputs/text/train-text-admission-without-references.json"
_config["training"]["validation"]["text_admission_path"] = f"{_experiment}/inputs/text/val-text-admission-without-references.json"
_config["model"]["text_context_cache"] = f"{_experiment}/text-cache-setting/without-references/context-cache/manifest.json"
# Checkpoint output root for the no-reference ablation run.
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/ckp/renderer-wan22-final-20261113/training"}
EXPERIMENT_CONFIG = _config
