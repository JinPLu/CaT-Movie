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
  <img src="assets/overview.png" width="98%" alt="Overview of CaT-Movie: Global Storyboard Planner, Prompt Compiler & Anchor Recall, and Local Window Renderer with Reference-Routed Sparse Self-Attention and Shot-Wise Cross-Attention">
</p>

</div>

---

## Method Overview: Balancing Global Planning and Local Rendering

Long multi-shot video generators face an inherent trade-off between **transition control** and **inter-shot consistency**:

- **Per-shot generation** executes planned cuts cleanly (`Cut F1 > 0.99`) yet forgets character identity, objects, and scene state once shots cut (`Character ~ 0.22–0.30`).
- **Joint multi-shot generation** coordinates shots within a short temporal window yet loses identity across window boundaries (`617 / 675` entity returns in our test storyboards cross a window boundary under a `7 s` cap) or sacrifices cut timing precision.
- **Similarity- or recency-based visual memory** bundles identity together with camera framing and outdated states, encouraging copied camera setups across planned cuts or error chaining across windows.

**CaT-Movie** resolves this trade-off around one shared **structured storyboard** connecting three components:

1. **Global Storyboard Planner (`Sec. 3.2`, `tools/run/planner_form_plans.py`):** A training-free language model (**Qwen3.8-27B** with constrained JSON decoding) expands multi-granularity requests (`synopsis`, `paragraph`, or numbered `shots`) into an explicit entity registry (`Character N`, `Object N`, `Location N`) and shot-by-shot plans (location, camera setup, optional opening composition, present entities with posture/state/held items/attire, and within-shot changes). Deterministic code—the **Prompt Compiler**—propagates attire changes and held-item outcomes forward, packs consecutive shots under the window cap (`8 s` / `128` frames in training; `7 s` / `112` frames at test time), and compiles a self-contained prompt $c_k$ per shot.
2. **Storyboard-Indexed Anchor Recall (`Sec. 3.3`, `--history-policy plan_selection`):** Rather than retrieving recent frames or matching CLIP embeddings, each returning entity deterministically queries the storyboard for its clearest prior **anchor view** generated *after* its latest planned attire or state change—ranking candidates lexicographically by pre-change indicator $u_j$, fewer co-occurring characters $n_j$, legible close-up shot size $d_j$, and earliest shot index $j$—alongside the widest few-character view of the shot's location (up to `10` recalled source shots per window).
3. **Local Window Renderer (`Sec. 3.3`, `Wan2.2-TI2V-5B`, `704p`, `24 fps`):** Fine-tuned across all `5.0B` transformer blocks (`3,000` updates on `8× A100`, rectified flow objective, no added parameters):
   - **Shot-wise cross-attention:** each target shot $Z_k$ attends exclusively to its compiled prompt $c_k$.
   - **Reference-routed sparse self-attention (`holocine_first_frame` + `routed_self_only_v1`):** each shot $Z_k$ attends to itself, the first latent frames $\{Z_j^{\mathrm{f}}\}_{j \in \mathcal{W}, j \neq k}$ of co-generated shots in its window, and its entity-routed clean recalled references $R_k$ placed at negative temporal RoPE offsets ($\Delta t = -5$ ending at $p_k - 5$). Recalled frames attend only to themselves and cross-attend to per-reference **identity lines** (`frame_entities_v1`, $c^{\mathrm{ref}}_r$) naming their visible entities and source location in the same identifiers as $c_k$. Training uses light reference noise ($\sigma \sim 0.2\,\mathcal{U}[0,1)$), whole-group dropout (`0.2`), and random entity-preserving zoom crops (`65–100%` area on `~47%` of references) over cross-setup scene shots so the renderer transfers identity without copying camera framing.

<p align="center">
  <img src="assets/pareto_consistency_control.png" width="72%" alt="Inter-shot consistency vs. transition control on CaT-MBench 92 test stories, breaking the baseline Pareto frontier">
</p>

---

## Release Resources

All code, weights, datasets, caches, source clips, benchmark protocols, and generated comparison videos are available and linked in the **[CaT-Movie Hugging Face Collection](https://huggingface.co/collections/JinPLU/cat-movie-6afcfb908ad5cc8c7f1f9206)**:

| Resource | Tier | Description | Link |
|---|---|---|---|
| **CaT-Movie Codebase** | **Core** | Storyboard planner, training, inference, data curation & `CaT-MBench` (`mspc` package) | [GitHub](https://github.com/JinPLu/CaT-Movie) (Apache-2.0) |
| **CaT-Movie-5B Weights** | **Core** | Finetuned `Wan2.2-TI2V-5B` multi-shot renderer (`step-3000.safetensors`, `9.44 GiB`) | [`JinPLU/CaT-Movie`](https://huggingface.co/JinPLU/CaT-Movie) |
| **CaT-Movie-Data** | **Core** | `75,461` scene blocks (`358,302` shots), `Qwen3.7-Flash` annotations, spatial fields, `96,000` training windows & `SAM 3.1` loss weights | [`JinPLU/CaT-Movie-Data`](https://huggingface.co/datasets/JinPLU/CaT-Movie-Data) |
| **CaT-Movie-T5-Cache** | **Core** | Precomputed `umT5-XXL` BF16 text context caches (`332,516` unique prompts covering Wan2.2 final, 3 ablations & Wan2.1 480p) | [`JinPLU/CaT-Movie-T5-Cache`](https://huggingface.co/datasets/JinPLU/CaT-Movie-T5-Cache) |
| **CaT-Movie-VAE-Caches** | **Core** | Precomputed target VAE latents (`Wan2.2 704p` `104,534` shots / `538.4 GB` + `Wan2.1 480p` `103,365` shots / `197.9 GB`) | [`JinPLU/CaT-Movie-VAE-Caches`](https://huggingface.co/datasets/JinPLU/CaT-Movie-VAE-Caches) |
| **CaT-Movie-Source-Videos** | **Supporting** | Complete `18,173` block-yielding source MP4 clips across CineDance, CinePile-T2V & MovieChat-1K | [`JinPLU/CaT-Movie-Source-Videos`](https://huggingface.co/datasets/JinPLU/CaT-Movie-Source-Videos) |
| **CaT-MBench** | **Core** | `122` held-out stories (`30` dev + `92` test), `GPT-6.1-sol` + rule-derived VQA bank (`1,012` test / `349` dev), `Gemini 3.8 Flash` verdicts & `557` human labels | [`JinPLU/CaT-MBench`](https://huggingface.co/datasets/JinPLU/CaT-MBench) |
| **CaT-MBench-Videos** | **Supporting** | All `736` generated 92-story test videos across all `8` evaluated methods (`13.65 GiB`) | [`JinPLU/CaT-MBench-Videos`](https://huggingface.co/datasets/JinPLU/CaT-MBench-Videos) |
| **CaT-MBench-Author-Labels** | **Archive** | Early single-author VLM calibration archive *(superseded by `CaT-MBench/human/`)* | [`JinPLU/CaT-MBench-Author-Labels`](https://huggingface.co/datasets/JinPLU/CaT-MBench-Author-Labels) |

> **Full research open-source release:** All checkpoints, multi-shot annotations, entity grounding metadata, precomputed VAE / `umT5-XXL` caches, `18,173` source clips, and benchmark assets are publicly released strictly for **non-commercial academic research** (`CC BY-NC-SA 4.0` where upstream ShareAlike applies; codebase under `Apache-2.0`).

---

## Dataset & Benchmark (`CaT-Movie-Data` & `CaT-MBench`)

One film pool and one unified storyboard schema serve both model training (**Film Storyboard Dataset**) and evaluation (**`CaT-MBench`**):

<p align="center">
  <img src="assets/data_and_benchmark.png" width="98%" alt="Overview of the Film Storyboard Dataset and CaT-MBench: (a) dataset curation and Qwen3.7-Flash annotation, (b) entity grounding via Qwen3.8-27B, SAM 3.1 and VGGT-Omega, center composition chart, (c) benchmark stories and GPT-6.1-sol question bank, and (d) Gemini 3.8 Flash cut-gated judging">
</p>

### 1. Film Storyboard Dataset (`JinPLU/CaT-Movie-Data`)

- **Curation & Shot-First Screening:** Screening `19,535` source videos (`CinePile-T2V`, `CineDance`, `MovieChat-1K`, with `18,173` clips yielding blocks) via TransNetV2 and interior-cut/luminance checks yields **`75,461` scene blocks** (**`358,302` shots**, **`332.8 hours`**). Individual failing shots (`LAION aesthetic < 4.5`, `DOVER < -0.149`, malformed annotations, or prompts `> 512` umT5 tokens) become breakpoints rather than discarding whole windows, retaining **`+28.1%` more usable shots** (`290,267` vs. `226,648` window-first) across **`67,489` usable blocks** split by film into **`66,573` training** and **`916` held-out** blocks, sampled into **`96,000`** stratified training windows (`15% / 25% / 30% / 30%` for `1 / 2 / 3 / 4+` shots).
- **One-Pass Schema & Three Granularities:** **Qwen3.7-Flash** annotates each block once with persistent entity IDs, per-shot camera framing, opening/end states, within-shot changes (`22.6%` of shots), a continuous scene `paragraph`, and a one-sentence `synopsis`.
- **Entity Grounding (`tools/data/spatial_layer.py`):** **Qwen3.8-27B** localizes earliest clear bounding boxes -> **SAM 3.1** propagates tracked instance masks -> **VGGT-Ω** estimates relative per-instance depth, yielding `left_of`, `in_front_of`, opening screen composition, and horizontal screen-side transitions.

### 2. Cut-Gated Story Benchmark (`JinPLU/CaT-MBench`)

[**`JinPLU/CaT-MBench`**](https://huggingface.co/datasets/JinPLU/CaT-MBench) curates **`122` stories** (`30` dev + `92` test across `67` disjoint held-out films; `6–12` shots, `≤ 60 s` each) containing a within-shot state change, a character returning across non-adjacent shots, and a dynamic action shot, paired with three request tiers (`synopsis`, `paragraph`, `shots`).

<p align="center">
  <img src="assets/cut_gate_teaser.png" width="84%" alt="Cross-window drift vs. copied framing across planned cuts, and cut-gated evaluation in CaT-MBench">
</p>

#### Why Cut-Gated Evaluation?

Naive cross-shot similarity metrics (such as ViCLIP or DINOv2) reward models that ignore planned transitions and repeat identical camera setups or backgrounds across cuts. **`CaT-MBench`** removes this shortcut across [`src/mspc/evaluation/`](src/mspc/evaluation/):

1. **Transition Control (`cut_execution` + TransNetV2):**
   - **`Cut F1` (`740` planned test cuts):** One-to-one matching between planned cuts and TransNetV2 detections within **`0.125 s`** (`2` frames @ `16 fps`).
   - **`Execution` (`674` planned setup-changing cuts):** Planned cuts that change location, shot size, camera angle, or cast pass only if TransNetV2 detects a boundary within **`±0.5 s`** **and** the later shot opens a **new camera setup** (pairs with `96×54` mid-frame grayscale `corr ≥ 0.9` fail directly; remaining pairs are judged by **Gemini 3.8 Flash** from 4 native-resolution endpoint frames).
2. **Gated VLM Checks (`Gemini 3.8 Flash`, `1,012` Test / `349` Dev Questions from `GPT-6.1-sol` + Deterministic Compiler):**
   - **Inter-shot Consistency (`Character`: `662`, `Object`: `65`, `Scene`: `194`):** Checked per compared shot pair from **6 native-resolution frames** (`first`, `middle`, `last` frame per shot). Any check across planned-different shots behind a failed cut gate is **scored `0`** (`StoryMem` loses `15.7%` of checks behind failed gates vs. `0.7%` for `CaT-Movie`).
   - **State Continuity (`Establish`: `44`, `Persist`: `47`):** Checked over the **complete video at `2 fps`** with per-shot timestamps (`Establish` verifies the requested change occurs in its shot alongside the causative action; `Persist` verifies the new state holds when the entity returns, gated on opening a new setup across different shots).
   - **Text Alignment:** Per-shot **ViCLIP** cosine similarity between `8` rendered frames and each shot's request line.
3. **Human Validation & Calibration (`557` Reviewer-Adjudicated Blind Items):** Across 7 baselines, blind labels from two annotators (`94` shared items, pre-review $\kappa = 0.64$) were resolved by a reviewer into `557` ground-truth items (`human/`). **Gemini 3.8 Flash** reaches **$\kappa = 0.70$** (`0.704`, 95% CI `[0.64, 0.77]`, balanced accuracy **`0.851`** against pooled annotator blind $\kappa = 0.853$), whereas ViCLIP achieves only AUC **`0.49–0.70`** (Kendall $\tau = 0.21$ with judge character scores).

### Main Comparison on `CaT-MBench` (`92` Test Stories)

Following **Table 1** of the paper (seed `42` for all methods): baselines convert the annotated **reference storyboard** into each method's native prompt format, whereas **CaT-Movie** renders its **Qwen3.8-27B** planner's storyboard generated solely from the shot-list request (`configs/benchmark/story_requests_v4.json`; baseline adapters live in [`src/mspc/comparison_methods/`](src/mspc/comparison_methods/README.md)). Best per column is **bold**; second best is *italic*.

| Regime | Method | Video Base | Cut F1 ↑ | Execution ↑ | Character ↑ | Object ↑ | Scene ↑ | Establish ↑ | Persist ↑ | Text Alignment ↑ | Avg. ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Per-shot* | Wan2.1, per shot | Wan2.1-T2V-1.3B | *0.998* | *0.994* | 0.219 | 0.169 | 0.310 | 0.274 | 0.694 | 0.181 | 0.474 |
| *Per-shot* | Wan2.2, per shot | Wan2.2-TI2V-5B | **0.999** | **0.996** | 0.227 | 0.275 | 0.310 | 0.387 | *0.701* | 0.174 | 0.497 |
| *Per-shot* | StoryMem | Wan2.2-I2V-A14B | 0.928 | 0.719 | *0.631* | 0.453 | **0.687** | 0.484 | 0.545 | 0.179 | 0.527 |
| *Per-shot* | StoryDiffusion | Wan2.2-I2V-A14B | 0.905 | 0.825 | 0.300 | 0.265 | 0.245 | *0.597* | 0.535 | 0.171 | 0.468 |
| *Per-shot* | IC-LoRA | Wan2.2-I2V-A14B | 0.976 | 0.961 | 0.168 | 0.133 | 0.406 | 0.452 | 0.663 | 0.176 | 0.484 |
| *Joint* | MultiShotMaster | Wan2.1-T2V-1.3B | 0.994 | 0.990 | 0.538 | 0.254 | 0.582 | 0.371 | 0.587 | 0.178 | 0.527 |
| *Joint* | HoloCine | Wan2.2-T2V-A14B | 0.748 | 0.967 | 0.517 | **0.512** | 0.586 | **0.694** | 0.691 | **0.195** | *0.571* |
| *Joint* | **CaT-Movie (Ours)** | Wan2.2-TI2V-5B | 0.995 | 0.974 | **0.651** | *0.497* | *0.630* | 0.468 | **0.785** | *0.186* | **0.597** |

*(Overall Inter-shot Consistency mean across Character, Object, Scene: **CaT-Movie** reaches **`0.593`** vs. **`0.590`** for StoryMem and **`0.538`** for HoloCine while executing **`0.974`** of planned setup changes.)*

### Running `CaT-MBench` Evaluation

```bash
# 1. Download CaT-MBench metadata, judge verdicts & (optionally) all 736 videos
hf download JinPLU/CaT-MBench --repo-type dataset --local-dir data/cat_mbench
hf download JinPLU/CaT-MBench-Videos --repo-type dataset --local-dir data/cat_mbench_videos

# 2. Recompute exact Table 1 scores & film-clustered paired bootstrap CIs (CPU only)
PYTHONPATH=src python -m mspc.evaluation.checklist_scoring readout-v17 \
  --bank data/cat_mbench/questions.v17.json \
  --cut-items data/cat_mbench/cut_items.v17.json \
  --cut-root data/cat_mbench/judge/cut \
  --pair-root data/cat_mbench/judge/pairs \
  --video-root data/cat_mbench/judge/video \
  --rows wan22-final-test92-planner-27b-seed42 test92-holocine test92-multishotmaster \
         test92-ic-lora test92-storydiffusion test92-storymem \
         test92-official-wan test92-wan21-1p3b-native \
  --answer-scope existing \
  --output outputs/eval/readout_v17_recomputed.json

PYTHONPATH=src python -m mspc.evaluation.checklist_scoring paired \
  --readout outputs/eval/readout_v17_recomputed.json \
  --requests configs/benchmark/story_requests_v4.json \
  --pair test92-holocine wan22-final-test92-planner-27b-seed42 \
  --output outputs/eval/paired_vs_holocine.json
```

To score **your own model's** generated videos under the identical two-stage judge protocol (`gemini-3.8-flash`):
1. Run `mspc.evaluation.checklist_scoring detect-cuts` to extract TransNetV2 shot boundaries.
2. Run `mspc.evaluation.cut_execution` (`prepare` -> `submit` -> `collect`) for cut-gate verdicts.
3. Run `mspc.evaluation.consistency_pair_batch` (`character` / `object` / `scene` across 6 native-resolution frames) and `mspc.evaluation.gemini_checklist_batch` (`event` / `state` across complete `2 fps` video), then aggregate with `checklist_scoring readout-v17 --answer-scope complete`.
4. To reproduce the `8` baseline comparison rows in [`src/mspc/comparison_methods/`](src/mspc/comparison_methods/README.md), verify deployment assets first with `PYTHONPATH=src python -m mspc.comparison_methods.inspect_deployment`.

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
| `uv sync --extra api` | Storyboard planner, `GPT-6.1-sol` question writer & `Gemini 3.8 Flash` evaluation |
| `uv sync --extra gpu-wan` | GPU training and `Wan2.2-TI2V-5B` multi-shot rendering |
| [`environments/planner_inference/`](environments/planner_inference/) | Serving `Qwen3.8-27B` / `Qwen3.5-9B` planners via vLLM (`run/vllm_serve.py`) |
| [`environments/person_tools/`](environments/person_tools/) & [`environments/moge2/`](environments/moge2/) | Entity grounding environments (**SAM 3.1** masks + **VGGT-Ω** relative depth in `spatial_layer.py`; MoGe-2 layout helper) |
| [`src/mspc/comparison_methods/`](src/mspc/comparison_methods/README.md) | Baseline runners, environments & scoring pins |

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream `Wan2.2`, `DiffSynth-Studio`, and `MultiShotMaster` provenance and offline verification.

---

## Quickstart: Single-Story Inference

> **Important:** `step-3000.safetensors` is trained with CaT-Movie's first-frame sparse window attention (`holocine_first_frame`), entity-routed self-only references (`routed_self_only_v1`) with negative temporal RoPE offsets ($\Delta t = -5$), and shot-wise cross-attention (`configs/infer/cat_movie_5b.py`). Run inference through `run/render_story.py` rather than vanilla Wan2.2 `generate.py`.

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

Render the bundled example storyboard (`examples/sample_storyboard.json`), or replace `--storyboard` with any story JSON generated by the planner:

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

Pass `--prepare-only` first to run CPU preflight checks and inspect compiled per-shot prompts $c_k$, identity lines $c^{\mathrm{ref}}_r$, and storyboard anchor recall before loading weights onto GPU.

---

## Reproducing Paper Results & Codebase Map

| Pipeline Stage | Entry Point(s) |
|---|---|
| **Shot screening & scene blocks** | [`tools/data/preprocess_sources.py`](tools/data/preprocess_sources.py), [`score_shots.py`](tools/data/score_shots.py), [`quality_lines.py`](tools/data/quality_lines.py), [`build_block_members.py`](tools/data/build_block_members.py) |
| **Storyboard annotation (`Qwen3.7-Flash`) & spatial grounding** | [`tools/data/annotate_windows.py`](tools/data/annotate_windows.py), [`validate_annotations.py`](tools/data/validate_annotations.py), [`spatial_layer.py`](tools/data/spatial_layer.py) |
| **Training windows & VAE/T5 precomputation** | [`tools/data/select_training_windows.py`](tools/data/select_training_windows.py), [`build_shot_targets.py`](tools/data/build_shot_targets.py), [`run/precompute.sh`](run/precompute.sh) |
| **Renderer training & Table 2 ablations** | [`run/train.sh`](run/train.sh) with [`configs/train/renderer_wan22_final.py`](configs/train/renderer_wan22_final.py) |
| **Storyboard planner (`Qwen3.8-27B`)** | [`tools/run/planner_form_plans.py`](tools/run/planner_form_plans.py) (`plan --prompt current --sampling thinking-medium`), served via [`run/vllm_serve.py`](run/vllm_serve.py) (`configs/serving/vllm_planner_qwen38_27b.yaml`) |
| **Table 1 multi-story planning & rendering** | `mspc.benchmark.layout_baseline write-planner-plans` & `prepare` (`configs/infer/planner_renderer_test92.json` -> `configs/infer/renderer_wan22_final_step3000.py`) |
| **Planner text analyses (`DeepSeek-V4.1-Flash`)** | [`src/mspc/evaluation/planner_state_checks.py`](src/mspc/evaluation/planner_state_checks.py), [`shot_text_information.py`](src/mspc/evaluation/shot_text_information.py) |
| **`CaT-MBench` VQA, judges & bootstrap CIs** | [`question_writer.py`](src/mspc/evaluation/question_writer.py), [`cut_execution.py`](src/mspc/evaluation/cut_execution.py), [`consistency_pair_batch.py`](src/mspc/evaluation/consistency_pair_batch.py), [`gemini_checklist_batch.py`](src/mspc/evaluation/gemini_checklist_batch.py), [`checklist_scoring.py`](src/mspc/evaluation/checklist_scoring.py), [`paired_bootstrap.py`](src/mspc/evaluation/paired_bootstrap.py), [`planned_cut_accuracy.py`](tools/analysis/planned_cut_accuracy.py), [`benchmark.py`](src/mspc/evaluation/benchmark.py) |

### 1. Plan & Render the 92 `CaT-MBench` Test Stories

Using `configs/infer/planner_renderer_test92.json` and a running planner endpoint (`configs/serving/vllm_planner_qwen38_27b.yaml`):

```bash
# Write planner storyboards from story requests
PYTHONPATH=src python -m mspc.benchmark.layout_baseline write-planner-plans \
  --conditions configs/infer/planner_renderer_test92.json \
  --records path/to/planner_records.jsonl

# Render all 92 test stories across windows (704p, 112-frame / 7 s window cap)
PYTHONPATH=src python -m mspc.benchmark.layout_baseline prepare \
  --conditions configs/infer/planner_renderer_test92.json
```

### 2. Training & Paper Table 2 Ablations

`CaT-Movie` trains from `Wan2.2-TI2V-5B` under `configs/train/renderer_wan22_final.py`, alongside the three single-variable ablations in **Table 2** (`30` development stories) and joint baseline configs:

- `configs/train/renderer_wan22_final.py` — **CaT-Movie** full model (`Step 3000` released)
- `configs/train/renderer_wan22_final_fixed_role_rows.py` — Paper Table 2: **Trained without identity lines** (`row_policy = fixed_role_v1`)
- `configs/train/renderer_wan22_final_window_shared.py` — Paper Table 2: **Trained without routing** (`attention_policy = window_shared_self_only_v1`)
- `configs/train/renderer_wan22_final_without_references.py` — Paper Table 2: **Trained without references** (`memory_prefix = None`)
- `configs/train/renderer_joint_wan22_720p.py` — Joint `704p` baseline configuration

Launch distributed training with `run/train.sh` after restoring the precomputed caches below (or regenerating them via `tools/data/build_shot_targets.py` and `configs/runtime/wan22_final_long_single_shot_targets.py`).

### 3. Complete Training Data, Caches & Source Videos

All training annotations, precomputed caches, and source video clips are publicly hosted on Hugging Face:

```bash
export MSPC_SHARED_ROOT=/path/to/cat_movie_workspace

# Annotations, member manifests, sampling tables, reference crops/entities, SAM 3.1 weights & text admission
hf download JinPLU/CaT-Movie-Data --repo-type dataset --local-dir "$MSPC_SHARED_ROOT/releases/CaT-Movie-Data"

# Precomputed umT5-XXL BF16 text context caches (Wan2.2 final, 3 ablations & Wan2.1 480p)
hf download JinPLU/CaT-Movie-T5-Cache --repo-type dataset --local-dir "$MSPC_SHARED_ROOT/releases/CaT-Movie-T5-Cache"

# Precomputed Wan2.2 704p & Wan2.1 480p VAE target latent caches
hf download JinPLU/CaT-Movie-VAE-Caches --repo-type dataset --local-dir "$MSPC_SHARED_ROOT/releases/CaT-Movie-VAE-Caches"

# Complete 18,173 block-yielding source MP4 clips (CineDance, CinePile-T2V & MovieChat-1K)
hf download JinPLU/CaT-Movie-Source-Videos --repo-type dataset --local-dir "$MSPC_SHARED_ROOT/releases/CaT-Movie-Source-Videos"
```

Each cache/video repository includes a zero-dependency restoration script (`restore_training_inputs.py`, `restore_t5_cache.py`, `restore_vae_caches.py`, `restore_source_videos.py`) to unpack tar shards and expand `${MSPC_SHARED_ROOT}` paths into the exact training layout. See [`JinPLU/CaT-Movie-Data`](https://huggingface.co/datasets/JinPLU/CaT-Movie-Data) for schema details.

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
- **Weights, datasets, caches, source clips & benchmarks (`CaT-Movie` checkpoint, `CaT-Movie-Data`, `CaT-Movie-T5-Cache`, `CaT-Movie-VAE-Caches`, `CaT-Movie-Source-Videos`, `CaT-MBench`, `CaT-MBench-Videos`, `CaT-MBench-Author-Labels`):** Released under **CC BY-NC-SA 4.0** strictly for non-commercial academic research. The base model `Wan2.2-TI2V-5B` is Apache-2.0.
- We build upon [Wan2.2](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Qwen](https://github.com/QwenLM), [MultiShotMaster](https://github.com/KlingAIResearch/MultiShotMaster), [CinePile](https://huggingface.co/datasets/CinematicT2vData/raw_videos_batched), [MovieChat-1K](https://huggingface.co/datasets/Enxin/MovieChat-1K-test) ([train](https://huggingface.co/datasets/Enxin/MovieChat-1K_train)), and [CineDance](https://huggingface.co/datasets/CineDance/CineDance_01).

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
