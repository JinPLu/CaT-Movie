<div align="center">

# CaT-Movie: Balancing Global Planning and Local Rendering for Long Multi-Shot Video Generation

<p>
  <a href="https://jinplu.github.io/CaT-Movie/"><img src="https://img.shields.io/badge/Project%20Page-Live-blue?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/JinPLU/cat-movie-6afcfb908ad5cc8c7f1f9206"><img src="https://img.shields.io/badge/HF%20Collection-CaT--Movie-yellow?style=flat-square&logo=huggingface" alt="Hugging Face Collection"></a>
  <a href="https://huggingface.co/JinPLU/CaT-Movie"><img src="https://img.shields.io/badge/Weights-CaT--Movie%205B-orange?style=flat-square&logo=huggingface" alt="Weights"></a>
  <a href="https://huggingface.co/JinPLU/CaT-Movie-Data"><img src="https://img.shields.io/badge/Dataset-CaT--Movie--Data-green?style=flat-square&logo=huggingface" alt="Dataset"></a>
  <a href="https://huggingface.co/JinPLU/CaT-MBench"><img src="https://img.shields.io/badge/Benchmark-CaT--MBench-red?style=flat-square&logo=huggingface" alt="Benchmark"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square" alt="License"></a>
  <!-- paper-link -->
</p>

<p>
  <strong>Jinpeng Lu</strong><sup>*</sup> ·
  <strong>Haoyuan Shi</strong><sup>*</sup> ·
  <strong>Yuanshen Guan</strong> ·
  <strong>Jiyuan Xia</strong> ·
  <strong>Bin Wu</strong> ·
  <strong>Yinda Chen</strong> ·
  <strong>Zhiwei Xiong</strong>
</p>
<p>
  University of Science and Technology of China &nbsp;·&nbsp; <sup>*</sup>Equal contribution
</p>

<p>
  <img src="assets/overview.png" width="98%" alt="CaT-Movie framework overview: Planner, Prompt Compiler & Frame Recall, and Reference-Routed Sparse Renderer">
</p>

</div>

---

## Overview

Existing multi-shot video generation approaches sit on an uncomfortable trade-off between **transition control** and **inter-shot consistency**:

- **Per-shot generation** executes cuts faithfully but drifts in identity, attire, objects, and scene state across shots.
- **Joint multi-shot synthesis** models short windows together, often softening or dropping planned cuts and scaling poorly to long stories.
- **Naive visual memory** over-attends to retrieved frames, encouraging background copying over planned compositional changes.

**CaT-Movie** decouples global narrative planning from local multi-shot rendering around **three design principles**:

1. **Code-Maintained Storyboard Planning.** A structured planner (**Qwen3.8-27B** by default) plans shots with explicit entity registries, attire/object persistence, shot-level opening/action/end states, and layout cues—eliminating long-horizon narrative drift before any pixel is synthesized.
2. **Localized Prompt Compilation & Frame Recall.** Each shot receives its own localized text prompt and selectively recalled keyframe references anchored with negative temporal RoPE indices.
3. **Reference-Routed Sparse Renderer (704p, 24 fps).** Built on **Wan2.2-TI2V-5B**, the renderer pairs first-frame cross-shot sparse attention (`holocine_first_frame`) with entity-routed, self-only reference attention (`routed_self_only_v1`) and shot-wise cross-attention, keeping transitions crisp while carrying identities across long shot gaps.

<p align="center">
  <img src="assets/pareto_consistency_control.png" width="68%" alt="Pareto frontier of Transition Control vs. Inter-Shot Consistency on CaT-MBench">
</p>

---

## Release Resources

All weights, annotations, benchmarks, and generated comparison videos are organized in the **[CaT-Movie Hugging Face Collection](https://huggingface.co/collections/JinPLU/cat-movie-6afcfb908ad5cc8c7f1f9206)**:

| Resource | Tier | Description | Link |
|---|---|---|---|
| **CaT-Movie Codebase** | **Core** | Training, inference, data construction, planner & CaT-MBench evaluation (`mspc` package) | [GitHub](https://github.com/JinPLu/CaT-Movie) (Apache-2.0) |
| **CaT-Movie-5B Weights** | **Core** | Finetuned Wan2.2-TI2V-5B multi-shot renderer (`step-3000.safetensors`, 9.4 GiB) | [`JinPLU/CaT-Movie`](https://huggingface.co/JinPLU/CaT-Movie) |
| **CaT-Movie-Data** | **Core** | 75,461 annotated multi-shot blocks (`358,302` shots, `332.8 h`) + `96,000` training samples (`blocks.jsonl`, `block_annotations.jsonl`, `training_inputs/`) | [`JinPLU/CaT-Movie-Data`](https://huggingface.co/JinPLU/CaT-Movie-Data) |
| **CaT-MBench** | **Core** | 122 stories (`92` test / `30` dev), cut-gated multi-shot VQA (`1,012` test / `349` dev scored questions), judge verdicts & human labels | [`JinPLU/CaT-MBench`](https://huggingface.co/JinPLU/CaT-MBench) |
| **CaT-MBench-Videos** | **Supporting** | All `736` generated 92-story videos across `8` evaluated methods (`13.7 GiB`) | [`JinPLU/CaT-MBench-Videos`](https://huggingface.co/JinPLU/CaT-MBench-Videos) |
| **CaT-MBench-Author-Labels** | **Archive** | Earlier single-author VLM-judge calibration frames & labels *(superseded by `CaT-MBench/human/`)* | [`JinPLU/CaT-MBench-Author-Labels`](https://huggingface.co/JinPLU/CaT-MBench-Author-Labels) |

> **Source video & cache redistribution policy:** Source MP4 clips from CinePile-T2V, MovieChat-1K, and CineDance are **not** redistributed; `CaT-Movie-Data` provides exact upstream clip identifiers, frame boundaries, quality scores, and offline reconstruction tools.

---

## Main Results on CaT-MBench (92 Test Stories)

Full evaluation on `CaT-MBench` (`1,012` test questions evaluated whole-video at `2 fps` with Gemini 2.5 Flash under the cut-execution gate). Best score per metric is **bold**; second best is *italic*.

| Method | Backbone | Cut F1 ↑ | Cut Exec. ↑ | Character ↑ | Object ↑ | Scene ↑ | Establish ↑ | Persist ↑ | Text Align. ↑ | Avg. ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.1, per shot | Wan2.1-T2V-1.3B | 0.998 | 0.994 | 0.219 | 0.169 | 0.310 | 0.274 | 0.694 | 0.181 | 0.474 |
| Wan2.2, per shot | Wan2.2-TI2V-5B | **0.999** | **0.996** | 0.227 | 0.275 | 0.310 | 0.387 | *0.701* | 0.174 | 0.497 |
| StoryMem | Wan2.2-I2V-A14B | 0.928 | 0.719 | *0.631* | 0.453 | **0.687** | 0.484 | 0.545 | 0.179 | 0.527 |
| StoryDiffusion | Wan2.2-I2V-A14B | 0.905 | 0.825 | 0.300 | 0.265 | 0.245 | *0.597* | 0.535 | 0.171 | 0.468 |
| IC-LoRA | Wan2.2-I2V-A14B | 0.976 | 0.961 | 0.168 | 0.133 | 0.406 | 0.452 | 0.663 | 0.176 | 0.484 |
| MultiShotMaster | Wan2.1-T2V-1.3B | 0.994 | 0.990 | 0.538 | 0.254 | 0.582 | 0.371 | 0.587 | 0.178 | 0.527 |
| HoloCine | Wan2.2-T2V-A14B | 0.748 | 0.967 | 0.517 | **0.512** | 0.586 | **0.694** | 0.691 | **0.195** | *0.571* |
| **CaT-Movie (Ours)** | Wan2.2-TI2V-5B | 0.995 | 0.974 | **0.651** | *0.497* | *0.630* | 0.468 | **0.785** | *0.186* | **0.597** |

---

## Installation

### 1. Clone and install core package

```bash
git clone https://github.com/JinPLu/CaT-Movie.git
cd CaT-Movie
uv sync
source .venv/bin/activate
```

> **Note on package name:** The Python package and CLI use `src/mspc/` (`mspc` — Multi-Shot Planning & Consistency, the internal working title of the codebase) so existing configs, imports, and checkpoints load cleanly without breaking compatibility.

### 2. Optional extras and auxiliary environments

Additional isolated environments are provided under [`environments/`](environments/) and [`src/mspc/comparison_methods/environments/`](src/mspc/comparison_methods/environments/):

| Extra / Env | Purpose |
|---|---|
| `uv sync --extra api` | Planner API calls and `CaT-MBench` VLM evaluation |
| `uv sync --extra gpu-wan` | GPU training and Wan2.2 inference |
| [`environments/planner_inference/`](environments/planner_inference/) | Serving `Qwen3.8-27B` / `Qwen3.5-9B` planners via vLLM |
| [`environments/person_tools/`](environments/person_tools/) & [`environments/moge2/`](environments/moge2/) | Training-set preprocessing (SAM / MoGe2) |
| [`src/mspc/comparison_methods/`](src/mspc/comparison_methods/README.md) | Baseline runners & scoring environments |

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream Wan2.2 / DiffSynth-Studio / MultiShotMaster provenance and offline verification.

---

## Quickstart: Single-Story Inference

> **Important:** `step-3000.safetensors` is trained with CaT-Movie's first-frame sparse window attention, routed negative-RoPE visual references, and shot-wise cross-attention (`configs/infer/cat_movie_5b.py`). Run inference through `run/render_story.py` rather than vanilla Wan2.2 `generate.py`.

### 1. Download base Wan2.2-TI2V-5B and CaT-Movie weights

```bash
export MSPC_SHARED_ROOT=/path/to/cat_movie_workspace
export MSPC_CODE_ROOT="$(pwd)"

hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$MSPC_SHARED_ROOT/checkpoints/Wan2.2-TI2V-5B"

hf download JinPLU/CaT-Movie step-3000.safetensors \
  --local-dir "$MSPC_SHARED_ROOT/checkpoints/CaT-Movie"
```

### 2. Render a story (704p, 24 fps)

You can render the bundled example storyboard (`examples/sample_storyboard.json`), or replace `--storyboard` with any story JSON generated by the planner:

```bash
PYTHONPATH=.:src python run/render_story.py \
  --config configs/infer/cat_movie_5b.py \
  --storyboard examples/sample_storyboard.json \
  --story-id cinedance-008935-b0001-shot01 \
  --output outputs/demo/cinedance-008935-b0001-shot01 \
  --seed 42 \
  --segment-seed-stride 1009 \
  --segment-max-shots none \
  --segment-max-frames 112 \
  --history-policy plan_selection
```

Pass `--prepare-only` first to run CPU preflight checks and inspect the compiled per-shot prompts and reference routing before loading weights onto GPU.

---

## Reproducing Paper Results

### 1. Plan & Render the 92 `CaT-MBench` Test Stories

Using `configs/infer/planner_renderer_test92.json` (which runs exact paper checkpoint/path bindings) and a running planner endpoint (`configs/serving/vllm_planner_qwen38_27b.yaml`):

```bash
# Write planner storyboards from story requests
PYTHONPATH=src python -m mspc.benchmark.layout_baseline write-planner-plans \
  --conditions configs/infer/planner_renderer_test92.json \
  --records path/to/planner_records.jsonl

# Render all 92 test stories across segments (704p, max 112 frames/window)
PYTHONPATH=src python -m mspc.benchmark.layout_baseline prepare \
  --conditions configs/infer/planner_renderer_test92.json
```

### 2. Recompute or Run `CaT-MBench` Evaluation

Download [`JinPLU/CaT-MBench`](https://huggingface.co/JinPLU/CaT-MBench) (and optionally [`JinPLU/CaT-MBench-Videos`](https://huggingface.co/JinPLU/CaT-MBench-Videos)) and run the unified benchmark evaluator:

```bash
# Summarize shipped verdicts and compute Table 1 + paired bootstrap statistics
PYTHONPATH=src python -m mspc.evaluation.benchmark summarize \
  --manifest path/to/eval_manifest.json \
  --output outputs/eval/cat_mbench_summary

# Run metrics and VLM judge on your own generated videos
PYTHONPATH=src python -m mspc.evaluation.benchmark score \
  --manifest path/to/eval_manifest.json \
  --output outputs/eval/custom_run \
  --components all
```

### 3. Training & Method Ablations

`CaT-Movie` trains from `Wan2.2-TI2V-5B` under `configs/train/renderer_wan22_final.py`, alongside three single-variable ablations:

- `configs/train/renderer_wan22_final.py` — Full CaT-Movie model (`Step 3000` released)
- `configs/train/renderer_wan22_final_fixed_role_rows.py` — Without per-reference entity identity descriptions
- `configs/train/renderer_wan22_final_window_shared.py` — Without entity-based reference routing (window-shared references)
- `configs/train/renderer_wan22_final_without_references.py` — Without reference prefix frames

Launch distributed training with `run/train.sh` after preparing target/text caches (`tools/data/build_shot_targets.py`, `configs/runtime/wan22_final_long_single_shot_targets.py`).

### 4. Reconstruct Training Data from Upstream Datasets

Download [`JinPLU/CaT-Movie-Data`](https://huggingface.co/JinPLU/CaT-Movie-Data) (`blocks.jsonl`, `block_annotations.jsonl`, `training_inputs/`) and obtain upstream video clips from **CinePile-T2V**, **MovieChat-1K**, and **CineDance** under their respective licenses, then run `tools/data/preprocess_sources.py` and `tools/data/prepare_renderer_training.py`. See [`JinPLU/CaT-Movie-Data`](https://huggingface.co/JinPLU/CaT-Movie-Data) for full instructions.

---

## Testing

```bash
MSPC_SHARED_ROOT=/tmp/mspc_shared_root \
MSPC_CODE_ROOT="$(pwd)" \
PYTHONPATH=src \
pytest -q
```

---

## License & Acknowledgements

- **Code:** Released under the [Apache-2.0 License](LICENSE). Third-party components retain their original licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Weights, datasets, and benchmarks (`CaT-Movie` checkpoint, `CaT-Movie-Data`, `CaT-MBench`, `CaT-MBench-Videos`, `CaT-MBench-Author-Labels`):** Released under **CC BY-NC-SA 4.0** for non-commercial research (required by upstream CineDance ShareAlike terms on training annotations). The base model `Wan2.2-TI2V-5B` is Apache-2.0.
- We build upon [Wan2.2](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Qwen](https://github.com/QwenLM), [MultiShotMaster](https://github.com/KlingAIResearch/MultiShotMaster), [CinePile](https://huggingface.co/datasets/tomg-group-umd/cinepile), [MovieChat-1K](https://huggingface.co/datasets/Enxin/MovieChat-1K), and [CineDance](https://huggingface.co/datasets/CineDance/CineDance_01).

---

## Citation

```bibtex
@misc{catmovie2026,
  title         = {CaT-Movie: Balancing Global Planning and Local Rendering for Long Multi-Shot Video Generation},
  author        = {Jinpeng Lu and Haoyuan Shi and Yuanshen Guan and Jiyuan Xia and Bin Wu and Yinda Chen and Zhiwei Xiong},
  year          = {2026},
  url           = {https://jinplu.github.io/CaT-Movie/}
}
```
