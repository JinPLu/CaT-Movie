"""Final CaT-Movie renderer training configuration on Wan2.2-TI2V-5B (1280x704, 24 fps, 12k schedule; step 3,000 released).

Uses first-frame sparse cross-shot attention (``holocine_first_frame``), entity-routed self-only visual references
(``routed_self_only_v1``), per-reference registered identity lines (``frame_entities_v1``), cropped reference augmentations,
and stratified 1/2/3/4+ shot windows. Single-variable ablations override this base config:
- ``renderer_wan22_final_fixed_role_rows.py`` (no per-reference entity identity lines)
- ``renderer_wan22_final_window_shared.py`` (no entity-based reference routing)
- ``renderer_wan22_final_without_references.py`` (no visual reference prefix)
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_joint_wan22_720p.py")))["EXPERIMENT_CONFIG"])
_experiment = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
               "renderer-wan22-final-20261113")
_inputs = f"{_experiment}/inputs"
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p",
                       "model_key": "wan22_ti2v_5b_final_704p", "experiment_key": "wan22_final"}
_config["dataset"]["reference_policy"] = "random_frame"
# Reference noise ceiling set to sigma_max=0.2 (higher noise degrades identity similarity below same-story distractors).
_config["model"]["memory_prefix"]["noise"] = {"train_sigma_max": 0.2, "inference_sigma": 0.0}
_config["dataset"]["sampling_manifest_path"] = f"{_inputs}/sampling/train-sampling.jsonl"
_config["dataset"]["reference_entities"] = {"labels_path": f"{_inputs}/labels/train-reference-entities.jsonl"}
_config["dataset"]["reference_crops"] = {"path": f"{_inputs}/crops/train-reference-crops.jsonl"}
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission.json"
# Per-arm umT5-XXL text context cache covering the full 12k schedule, reusing unchanged prompt embeddings.
_config["model"]["text_context_cache"] = f"{_experiment}/text-cache-setting/final-balanced/context-cache/manifest.json"
# Verified 8-GPU workload envelope (69 total latent frames, 4,209 sparse attention frame pairs under first-frame attention).
_config["dataset"]["workload_cap"] = {"latent_frames": 69, "attention_frame_pairs": 4209}
# Cost-bucketed batching calibrated from measured step forward latency using latent frames + attention pairs (reordering only).
_config["training"]["workload_cost"] = {"linear_seconds_per_latent": 0.066824,
                                        "attention_seconds_per_frame_pair": 0.001297}
# Save checkpoints every 1,000 updates (step 3,000 is the released inference checkpoint and ablation comparison point).
_config["training"]["save_steps"] = 1000
# Merged target VAE latent cache across 104,023 train + 508 validation shots (including 2,053 backfilled long single shots).
_targets = f"{_experiment}/target-cache"
_config["dataset"]["target_index_path"] = f"{_targets}/consumer-index-balanced.jsonl"
_config["dataset"]["cache_provenance_path"] = f"{_targets}/consumer-source-collection-balanced.json"
_config["dataset"]["target_roots"] = {
    "existing_shot": _config["dataset"]["target_roots"]["existing_shot"],
    "weighted_window_shot": _config["dataset"]["target_roots"]["weighted_window_shot"],
    "long_single_shot": f"{_targets}/long-single-shots/renderer-720p24-targets-20261108"}
_config["training"]["validation"]["reference_entities"] = {"labels_path": f"{_inputs}/labels/val-reference-entities.jsonl"}
_config["training"]["validation"]["text_admission_path"] = f"{_inputs}/text/val-text-admission.json"
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/training"}
EXPERIMENT_CONFIG = _config
