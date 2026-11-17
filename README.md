# CaT-Movie

**Balancing Global Planning and Local Rendering for Long Multi-Shot Video Generation**

[Project page](https://jinplu.github.io/CaT-Movie-page/) ·
[Renderer weights](https://huggingface.co/JinPLU/CaT-Movie) ·
[Training data](https://huggingface.co/datasets/JinPLU/CaT-Movie-Data) ·
[CaT-MBench](https://huggingface.co/datasets/JinPLU/CaT-MBench) ·
[CaT-MBench videos](https://huggingface.co/datasets/JinPLU/CaT-MBench-Videos) ·
Paper: coming soon

A diffusion pass renders a limited duration, so a long movie has to be generated window by window. CaT-Movie
keeps **C**oherence **a**cross **T**ime with one structured storyboard written before the first frame:

- **Global story planner.** A public language model (Qwen3.8-27B, no fine-tuning) expands a synopsis, a paragraph
  or a shot list into an entity registry and a plan per shot. Deterministic code carries attire and held items
  across shots, compiles a self-contained prompt per shot, and indexes anchor views of recurring entities in the
  frames generated so far.
- **Local window renderer.** Wan2.2-TI2V-5B fine-tuned without new parameters. Shot-wise cross-attention lets each
  shot read only its own prompt; reference-routed sparse self-attention lets it read its own tokens, the first
  latent frame of each co-generated shot, and only the recalled frames that share its entities.
- **CaT-MBench.** 122 stories from held-out films with a binary question bank. Consistency counts only across cuts
  that open the planned new camera setup, so repeated framing cannot pass as consistency.

This repository holds the code for all three, plus the data pipeline that built the 350K-shot training set.

## Results on CaT-MBench (92 test stories)

| Method | Cut F1 | Execution | Character | Object | Scene | Establish | Persist | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.2-TI2V-5B, per shot | 0.999 | 0.996 | 0.227 | 0.275 | 0.310 | 0.387 | 0.701 | 0.497 |
| StoryMem | 0.928 | 0.719 | 0.631 | 0.453 | 0.687 | 0.484 | 0.545 | 0.527 |
| MultiShotMaster | 0.994 | 0.990 | 0.538 | 0.254 | 0.582 | 0.371 | 0.587 | 0.527 |
| HoloCine | 0.748 | 0.967 | 0.517 | 0.512 | 0.586 | 0.694 | 0.691 | 0.571 |
| **CaT-Movie** | 0.995 | 0.974 | **0.651** | 0.497 | 0.630 | 0.468 | **0.785** | **0.597** |

Avg. averages transition control, inter-shot consistency, state continuity and text alignment. Baselines render
the reference storyboard; CaT-Movie renders its own planner's storyboard written from the shot-list request.
All scores and every judge answer behind them are in [CaT-MBench](https://huggingface.co/datasets/JinPLU/CaT-MBench).

## Layout

- `src/mspc/` — the Python package (data, annotation, training, inference, evaluation, comparison methods)
- `run/` — thin launch scripts (`run/train.sh CONFIG`, `run/precompute.sh CONFIG`, `run/vllm_serve.py`)
- `tools/` — data-pipeline, planning and analysis scripts
- `configs/` — explicit experiment configs
- `third_party/` — upstream pins, patches, licenses and the modified DiffSynth runtime
- `tests/` — pytest suite

## Setup

```bash
uv sync --locked
uv run --locked pytest -q
export MSPC_SHARED_ROOT=/path/to/workspace   # models, data, outputs
```

Configs refer to data, model and output locations as `${MSPC_SHARED_ROOT}/...`. `run/train.sh` and
`run/precompute.sh` run the interpreter named in the config's `runtime.python`; the separate environments for the
planner server and the data tools are locked under `environments/`.

Weights go where the configs expect them:

```bash
# base model (T5 encoder, VAE, tokenizer and the original DiT shards)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "$MSPC_SHARED_ROOT/OoVMetric/models_ckpt/wan22_5b_native"
# CaT-Movie renderer, training step 3,000 (the checkpoint behind the results above)
hf download JinPLU/CaT-Movie step-3000.safetensors --local-dir \
  "$MSPC_SHARED_ROOT/Storyboard/outputs/experiments/renderer-wan22-final-20261113/training/wan22_final_704p__20261112T231417_893962Z/checkpoints"
# planner
hf download Qwen/Qwen3.8-27B --local-dir "$MSPC_SHARED_ROOT/OoVMetric/models_ckpt/Qwen3.8-27B"
```

## Generate the test stories with CaT-Movie

```bash
# 1. serve the planner
python run/vllm_serve.py --config configs/serving/vllm_planner_qwen38_27b.yaml --provenance-dir outputs/planner-serving
# 2. write storyboards from the shot-list requests
python tools/run/planner_form_plans.py plan --requests configs/benchmark/story_requests_v4.json --split test \
  --prompt current --sampling thinking-medium --base-url http://127.0.0.1:8188/v1 --model Qwen3.8-27B \
  --provenance outputs/planner-records/provenance.json --seeds 42 --tiers shots \
  --output outputs/planner-records --workers 4
# 3. maintain and compile the storyboards, recall frames, and render every story window by window
python -m mspc.benchmark.layout_baseline write-planner-plans --conditions configs/infer/planner_renderer_test92.json \
  --records outputs/planner-records
python -m mspc.benchmark.layout_baseline prepare --conditions configs/infer/planner_renderer_test92.json
```

`configs/infer/planner_renderer_test92.json` renders with a 7 s window cap (`segment_max_frames` 112 at 16 fps)
and storyboard-indexed recall (`history_policy` `plan_selection`); it reads
`configs/infer/renderer_wan22_final_step3000.py`. To render your own storyboard, see
`python run/render_story.py --help`.

## Entry points

| Part | Entry point |
|---|---|
| Shot screening and blocks | `tools/data/preprocess_sources.py`, `score_shots.py`, `quality_lines.py`, `build_block_members.py` |
| Block annotation and spatial fields | `tools/data/annotate_windows.py`, `validate_annotations.py`, `spatial_layer.py` |
| Training windows and target latents | `tools/data/select_training_windows.py`, `build_shot_targets.py`, `run/precompute.sh` |
| Renderer training | `run/train.sh configs/train/renderer_wan22_final.py` (3,000 steps on 8 A100 GPUs) |
| Ablations: without identity lines, routing, references | `configs/train/renderer_wan22_final_fixed_role_rows.py`, `_window_shared.py`, `_without_references.py` |
| Planner | `tools/run/planner_form_plans.py plan`, served by `run/vllm_serve.py` |
| Story rendering | `python -m mspc.benchmark.layout_baseline write-planner-plans`, then `prepare` |
| Benchmark stories | `configs/benchmark/story_requests_v4.json`, `benchmark_storyboards_v4.json` |
| Questions, judges and scores | `src/mspc/evaluation/question_writer.py`, `gemini_checklist_batch.py`, `consistency_pair_batch.py`, `tools/analysis/planned_cut_accuracy.py`, `python -m mspc.evaluation.benchmark score\|summarize` |
| Paired intervals | `src/mspc/evaluation/paired_bootstrap.py` |
| Baselines | `src/mspc/comparison_methods/` |

## Recompute the CaT-MBench scores

```bash
hf download JinPLU/CaT-MBench --repo-type dataset --local-dir cat-mbench
cd cat-mbench && python -m mspc.evaluation.checklist_scoring readout-v17 \
  --bank questions.v17.json --cut-items cut_items.v17.json \
  --cut-root judge/cut --pair-root judge/pairs --video-root judge/video \
  --rows wan22-final-test92-planner-27b-seed42 test92-holocine test92-storymem test92-multishotmaster \
         test92-storydiffusion test92-ic-lora test92-official-wan test92-wan21-1p3b-native \
  --answer-scope existing --output readout.json
```

This reproduces `readouts/readout_v17.json` value for value. Judging a new method calls Gemini through the judge
scripts listed above and needs your own API key.

## Data

[CaT-Movie Data](https://huggingface.co/datasets/JinPLU/CaT-Movie-Data) releases 75,461 annotated blocks
(358,302 shots, 332.8 hours) and the renderer's 96,000 training samples. Source videos are not redistributed:
blocks name their upstream clip and the frame span of every shot, so the clips can be rebuilt from CinePile-T2V,
CineDance and MovieChat-1K under their own terms.

## License

- Code: Apache-2.0 (`LICENSE`). Third-party components keep their own licenses; see `THIRD_PARTY_NOTICES.md`.
- Renderer weights, training data, CaT-MBench and its videos: CC BY-NC-SA 4.0, for non-commercial research.
  The base model Wan2.2-TI2V-5B is Apache-2.0; baseline videos also carry their generators' licenses.

## Citation

```bibtex
@misc{catmovie2026,
  title  = {CaT-Movie: Balancing Global Planning and Local Rendering for Long Multi-Shot Video Generation},
  author = {To be announced},
  year   = {2026}
}
```
