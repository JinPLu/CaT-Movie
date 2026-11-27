"""Public inference configuration for the CaT-Movie 5B renderer v1.1 (`v1.1/step-3000.safetensors`).

Identical to ``cat_movie_5b.py`` (v1.0, the paper checkpoint) except for the renderer weights, read from
``${MSPC_SHARED_ROOT}/checkpoints/CaT-Movie/v1.1/step-3000.safetensors``. v1.1 changes only training data and
training-time reference selection (see the HF model card), so architecture, attention routing, reference memory
and sampling hyperparameters are inherited unchanged.
"""

from copy import deepcopy
from pathlib import Path
import runpy

_BASE = deepcopy(runpy.run_path(str(Path(__file__).with_name("cat_movie_5b.py")))["EXPERIMENT_CONFIG"])
_BASE["inference"]["dit_checkpoint"] = "${MSPC_SHARED_ROOT}/checkpoints/CaT-Movie/v1.1/step-3000.safetensors"
_BASE["inference"]["output_dir"] = "${MSPC_SHARED_ROOT}/outputs/cat_movie_5b_v1_1"
_BASE["output"]["root"] = "${MSPC_SHARED_ROOT}/outputs/cat_movie_5b_v1_1"

EXPERIMENT_CONFIG = _BASE
