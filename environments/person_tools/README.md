# Person tools environment (RTMW, InsightFace, OSNet-AIN)

Whole-body keypoints (for facing direction), face embeddings and person re-identification embeddings for the
multi-shot spatial-fact extraction (`tools/run/spatial_facts.py people`) and the block spatial layer
(`tools/data/spatial_layer.py pose`). InsightFace embeddings are an internal intermediate; no identity labels derived
from them are released.

- `uv.lock` in this directory pins Python 3.12, torch 2.10.0 and torchvision 0.25.0 (cu128), `rtmlib==0.0.16`,
  `insightface==2.1`, and `torchreid` from `KaiyangZhou/deep-person-reid@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`
  (package 1.4.0).
- `onnxruntime-gpu==1.26.0` replaces the CPU `onnxruntime` that insightface and rtmlib declare (the two cannot be
  installed together), through `override-dependencies`. 1.26.0 is the last CUDA 12 build. CUDA and cuDNN come from the
  `nvidia-*-cu12` wheels torch installs: import torch before calling `onnxruntime.preload_dlls()`.
- torchreid's `setup.py` imports the package while building, so the lock carries its static metadata
  (`[[tool.uv.dependency-metadata]]`), it builds without isolation, and installation takes two steps (below).
- OpenCV is pinned to 4.x (`opencv-python<5`), and only `opencv-python` is kept: rtmlib also declares
  `opencv-contrib-python`, which installs over the same `cv2` module.
- The RTMW ONNX model takes NCHW input `[batch, 3, 384, 288]`; the tool passes `model_input_size=(288, 384)` and checks
  the model's actual input size when loading.

## Install

```bash
export UV_PROJECT_ENVIRONMENT=$MSPC_SHARED_ROOT/env/person_tools_cu128
uv sync --project environments/person_tools --locked --python 3.12 --no-install-package torchreid
uv sync --project environments/person_tools --locked --python 3.12
```

## Weights

`tools/run/spatial_facts.py people` takes the weights as arguments: `--rtmw` (RTMW-x 384×288 `end2end.onnx`),
`--insightface-root` (a directory holding the InsightFace `buffalo_l` pack) and `--osnet` (`osnet_ain_x1_0` weights for
torchreid).
