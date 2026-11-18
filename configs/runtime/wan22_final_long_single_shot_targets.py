"""Manifest config for precomputing Wan2.2 704p/24fps VAE target latents on long single shots (>=5s, 2,053 shots).

Used exclusively with ``tools/data/build_shot_targets.py manifest`` (same VAE checkpoint and BF16 untiled encoding contract).
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).parents[1] / "train/renderer_wan22_final.py"))["EXPERIMENT_CONFIG"])
_config["dataset"]["target_roots"] = {"shot": (
    "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
    "target-cache/long-single-shots/renderer-720p24-targets-20261108")}
EXPERIMENT_CONFIG = _config
