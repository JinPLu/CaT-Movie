# MoGe-2 monocular geometry environment

Depth order for the cross-shot layout tools (`tools/run/cross_shot_layout.py depth`): the median MoGe-2 depth inside each
SAM 3.1 instance mask orders the subjects of a shot from front to back. The geometry annotation tools reuse the same
environment and weights.

- `uv.lock` in this directory pins Python 3.12, torch 2.10.0 and torchvision 0.25.0 (cu128), and `moge` from
  `microsoft/MoGe@925b8ed835a7a9cdb7578ba15c658a0afc969030`, the last MoGe-2 commit before MoGe-3 (package 2.0.0, MIT).
- Weights: Hugging Face `Ruicheng/moge-2-vitl-normal` at revision `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`
  (`model.pt`, MIT), expected at `$MSPC_SHARED_ROOT/OoVMetric/models_ckpt/moge-2-vitl-normal/model.pt`.
- MoGe-2 was chosen partly for its license: Depth Anything V2 Base and Large are CC BY-NC 4.0, and Depth Pro uses a
  research license.

## Install

```bash
UV_PROJECT_ENVIRONMENT=$MSPC_SHARED_ROOT/env/moge2_cu128 uv sync --project environments/moge2 --locked --python 3.12
```

## Use

`LD_LIBRARY_PATH=` keeps a system CUDA library path from shadowing the CUDA libraries that come with torch.

```bash
LD_LIBRARY_PATH= $MSPC_SHARED_ROOT/env/moge2_cu128/bin/python tools/run/cross_shot_layout.py depth --plan … --instances … \
  --model $MSPC_SHARED_ROOT/OoVMetric/models_ckpt/moge-2-vitl-normal/model.pt \
  --model-revision cb0e8bbd6b1e243589717c78e750b1ba4c093acf --device cuda --out …
```
