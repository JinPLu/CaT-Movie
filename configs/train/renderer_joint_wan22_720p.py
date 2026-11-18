"""Wan2.2-TI2V-5B native multi-shot joint training config (1280x704, 24 fps, 12k updates).

Consumes verified 704p VAE target latents and umT5-XXL context caches.
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_joint_native_training.py")))["EXPERIMENT_CONFIG"])
_source_root = Path(__file__).resolve().parents[2]
_pythonpath = [str(_source_root / "src"), str(_source_root),
               str(_source_root / "third_party/wan22_runtime")]
# Execute from this checkout while resolving datasets and model weights under ${MSPC_SHARED_ROOT}.
_config["model"]["diffsynth_root"] = str(_source_root / "third_party/wan22_runtime")
_config["launch"].update(
    train_entrypoint=str(_source_root / "src/mspc/training/runner.py"),
    infer_entrypoint=str(_source_root / "src/mspc/inference/wan22.py"),
    train_pythonpath=list(_pythonpath), infer_pythonpath=list(_pythonpath))
_data = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_inputs = f"{_data}/two-backbone-training-preparation-20261108/inputs/wan22-720p/shot-count-weighted-windows"
_cache = f"{_data}/renderer-720p24-target-cache-20261108"
_full_targets = f"{_data}/two-backbone-training-preparation-20261108/full-run-target-cache/wan22-704p"
_text_cache = f"{_data}/renderer-native-backbones-20261108/review/training-text-cache"
_config["identity"] = {**_config["identity"],
    "run_name": "joint_wan22_720p", "model_key": "wan22_ti2v_5b_joint_720p"}
_config["task"] = {"paradigm": "text_to_multishot_video", "height": 704, "width": 1280,
    "frame_policy": "fps_aware_target_fps", "max_rgb_frames": 360,
    "target_fps": 24, "vae_temporal_stride": 4}
_config["model"]["text_context_cache"] = f"{_text_cache}/full-run-context-cache/manifest.json"
_config["training"]["save_steps"] = 1000
_config["dataset"]["segment_capacity"] = {"max_frames": 360}
_config["dataset"]["cache_provenance_path"] = f"{_full_targets}/cache-source-collection.json"
_config["dataset"]["target_index_path"] = f"{_full_targets}/target-index.jsonl"
_config["dataset"]["target_roots"] = {
    "existing_shot": f"{_cache}/views/full-eight-second-targets-verified",
    "weighted_window_shot": f"{_full_targets}/full-weighted-targets",
}
_config["dataset"]["reference_entities"] = {"labels_path": f"{_inputs}/train-reference-entities.jsonl"}
_config["dataset"]["text_admission_path"] = f"{_inputs}/train-text-admission.json"
_config["training"]["validation"]["reference_entities"] = {"labels_path": f"{_inputs}/val-reference-entities.jsonl"}
_config["training"]["validation"]["text_admission_path"] = f"{_inputs}/val-text-admission.json"
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
                             "renderer-native-backbones-20261108/wan22-720p/training"}
EXPERIMENT_CONFIG = _config
