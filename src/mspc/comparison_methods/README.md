# Comparison methods

This package runs the baselines of the CaT-MBench comparison from the same storyboard per story. Each method keeps
its official code, weights and sampling settings; only the input is adapted. The shared plan is converted into the
method's native text, entity naming, shot boundaries and input conditions, and a long story is split into the method's
native windows with its own history mechanism. Once a deployment is confirmed faithful to the official implementation,
its remaining failures count as the method's result; no method is changed to rescue a bad video.

## Methods

| Method | Upstream (pinned) | Config |
|---|---|---|
| Wan2.2-TI2V-5B, shot by shot | DiffSynth runtime in `third_party/wan22_runtime`; official entry `Wan-Video/Wan2.2@1ea34ff` | `configs/wan.json`, `configs/wan_official.json` |
| MultiShotMaster 1.3B | `1df812dda262639e4a3ca6e4a1da9000b0a8e124` | `configs/multishotmaster.json` |
| HoloCine full 14B | `yihao-meng/HoloCine@0a7630ba448e13ddc7b872638b578e5618208c45` | `configs/holocine.json` |
| StoryMem MI2V 14B | `Kevin-thu/StoryMem@052c68d2627a22a95f79dbb4d3376cc30c96f1a3` | `configs/storymem.json` |
| StoryDiffusion + Wan I2V | `8de45e424887766fdd84dc917436ff8605f00149` | `configs/storydiffusion.json` |
| IC-LoRA + Wan I2V | workflow and config `966e00a826da91512fc473705bd09183f3946b21` | `configs/ic_lora.json` |

The two keyframe methods animate their keyframes with the StoryMem code base and Wan2.2 I2V-A14B. Weight revisions:
SDXL `462165984030d82259a11f4367a4eed129e94a7b`, FLUX.1-dev `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`, IC-LoRA
`16dae427a8509229309b85bc5345dfeffee5fc2e`. Server paths of every asset are in `deployment.json`.

## Comparison conditions

- Every method receives the same plan, the same request and seed 42.
- Native windows: Wan one shot at a time; MultiShotMaster up to 5 shots or 308 frames; HoloCine up to 241 frames;
  StoryMem one shot at a time, up to 225 frames.
- MultiShotMaster and HoloCine get stable entity ids, a global description and per-shot actions; StoryMem refers to
  characters by appearance. Keyframe methods take the opening facts of each shot, and the I2V stage takes the full
  action and closing state. StoryDiffusion's auxiliary identity images come only from the given appearance and first
  outfit; no source footage is used.
- Sampling: Wan, MultiShotMaster and HoloCine 50 steps, CFG 5, shift 5; StoryMem 40 UniPC steps, CFG 3.5, shift 12
  for T2V and 4 for M2V; the shared I2V stage 40 steps, CFG 3.5, shift 5.
- Videos are 832×480 at 16 fps, HoloCine included (`sampling.output_fps`; its official default is 15 fps), matching the paper's HoloCine videos.
- StoryMem's memory is the story's own generated history: the official `extract_keyframes.py` keeps frames that pass
  the HPSv3 quality gate (threshold 3.0) and a CLIP diversity check. When no frame passes, the memory stays empty, as
  in the official pipeline; no fallback is added.

## Files

| File | Role |
|---|---|
| `prompts.py`, `request_storyboard.py` | Convert the shared storyboard into each method's native text, shot boundaries, frame counts and windows |
| `renderer.py` | `RendererSession` owns the model lifetime of a frozen method; native Wan calls share `mspc/inference/segment_story.py` |
| `dispatch.py`, `queue.py`, `handoff.py`, `configs/full_generation.json` | One resident model process per worker consumes a fixed story queue under a time budget |
| `storydiffusion.py`, `ic_lora.py`, `keyframe_video.py`, `keyframe_story.py` | Keyframe methods and the shared Wan I2V stage |
| `st_bench.py` | Request conversion for the per-shot benchmark format |
| `deployment.json`, `inspect_deployment.py` | Pinned upstream revisions and asset paths; a read-only check of sources, weights and package metadata |
| `audit_storymem_memory.py` | Replays StoryMem's official keyframe selection on a generated shot |
| `review.py`, `review.html` | Side-by-side viewer for generated videos |
| `environments/ic_lora/` | Locked SentencePiece dependency for IC-LoRA |
| `storyengine.py` | StoryEngine wiring, not part of the comparison |

## Commands

Run from the repository root with `MSPC_SHARED_ROOT` set.

Read-only deployment check (loads no model and uses no GPU):

```bash
PYTHONPATH=src python -m mspc.comparison_methods.inspect_deployment --output deployment-readback.json
```

Generation runs through `mspc.benchmark.generation prepare|plan|render` with `configs/video_models.json`. The keyframe
methods run through `python -m mspc.comparison_methods.storydiffusion --config …` or `python -m
mspc.comparison_methods.ic_lora --config …`; `keyframe_story.py` runs preparation, keyframes, per-shot video and
concatenation story by story.

Side-by-side review:

```bash
PYTHONPATH=src python -m mspc.comparison_methods.review --config src/mspc/comparison_methods/configs/review.json --port 8879
```
