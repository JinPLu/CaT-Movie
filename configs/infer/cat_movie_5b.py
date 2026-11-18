"""Clean public inference configuration for the released CaT-Movie 5B renderer (`step-3000.safetensors`).

Uses clean local directories under ``${MSPC_SHARED_ROOT}/checkpoints/`` rather than internal
training-run folders:
- ``${MSPC_SHARED_ROOT}/checkpoints/Wan2.2-TI2V-5B/`` for the base Wan2.2-TI2V-5B files
- ``${MSPC_SHARED_ROOT}/checkpoints/CaT-Movie/step-3000.safetensors`` for the CaT-Movie weights

All model architecture, attention routing, reference memory, and sampling hyperparameters
are inherited identically from ``renderer_wan22_final_step3000.py``.
"""

from copy import deepcopy
from pathlib import Path
import runpy
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BASE = deepcopy(
    runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final_step3000.py")))["EXPERIMENT_CONFIG"]
)
_WAN22_DIR = "${MSPC_SHARED_ROOT}/checkpoints/Wan2.2-TI2V-5B"
_CKPT_PATH = "${MSPC_SHARED_ROOT}/checkpoints/CaT-Movie/step-3000.safetensors"

_BASE["runtime"]["python"] = sys.executable
_BASE["runtime"]["hf_cache"] = "${MSPC_SHARED_ROOT}/checkpoints"
_BASE["model"]["paths"] = {
    "dit": [
        f"{_WAN22_DIR}/diffusion_pytorch_model-00001-of-00003.safetensors",
        f"{_WAN22_DIR}/diffusion_pytorch_model-00002-of-00003.safetensors",
        f"{_WAN22_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors",
    ],
    "t5": f"{_WAN22_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
    "vae": f"{_WAN22_DIR}/Wan2.2_VAE.pth",
    "tokenizer": f"{_WAN22_DIR}/google/umt5-xxl",
}
_BASE["inference"]["dit_checkpoint"] = _CKPT_PATH
_BASE["inference"]["output_dir"] = "${MSPC_SHARED_ROOT}/outputs/cat_movie_5b"
_BASE["output"]["root"] = "${MSPC_SHARED_ROOT}/outputs/cat_movie_5b"

EXPERIMENT_CONFIG = _BASE
