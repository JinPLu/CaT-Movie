"""Ablation (without entity-based reference routing): lets every target shot attend to all reference frames in the window.

Overrides ``model.memory_prefix.attention_policy`` from ``routed_self_only_v1`` to ``window_shared_self_only_v1``.
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_inputs = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
           "renderer-wan22-final-20261113/inputs")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_window_shared_references",
                       "model_key": "wan22_ti2v_5b_final_704p_window_shared_references"}
_config["model"]["memory_prefix"]["attention_policy"] = "window_shared_self_only_v1"
# Uses the same {69 latent frames, 4,209 frame pairs} workload cap as the base config to keep filtered samples identical.
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission-window-shared-references.json"
_config["training"]["validation"]["text_admission_path"] = (
    f"{_inputs}/text/val-text-admission-window-shared-references.json")
_config["model"]["text_context_cache"] = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
                                          "text-cache-setting/window-shared-references/context-cache/manifest.json")
EXPERIMENT_CONFIG = _config
