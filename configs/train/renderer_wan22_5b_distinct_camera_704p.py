"""Wan2.2-TI2V-5B（704p）全量 DiT 训练，使用异机位共享实体参考。

从原生 Wan2.2 权重起训，沿用定稿的混合参数存储、按镜路由和首潜帧互读。
复用 5B LoRA 已发布的 96,000 窗和三源目标潜变量；单镜／两镜／三镜／四镜及以上
的抽样权重为 25／35／25／15，单镜层一半来自不短于 5 秒的长单镜。
8 卡 × 单卡 1 窗 × 累积 2，ZeRO-2；保留 6,000 步余弦调度和缓存阶段，
正式运行通过现有训练控制命令在第 3,000 步停下。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])

_data = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_lora_experiment = f"{_data}/renderer-wan22-5b-lora-distinct-camera-20261120"
_inputs = f"{_lora_experiment}/inputs"
_targets = f"{_lora_experiment}/target-cache"

_config["identity"].update(
    run_name="wan22_5b_fulldit_704p_distinct_camera",
    model_key="wan22_ti2v_5b_fulldit_704p_distinct_camera",
    experiment_key="wan22_5b_fulldit_distinct_camera",
)
_config["runtime"]["python"] = (
    "${MSPC_SHARED_ROOT}/env/wan22_fun_sm80_cu128_deepspeed0187/bin/python"
)
_config["runtime"]["hf_cache"] = "${MSPC_SHARED_ROOT}/.cache/hf"
_config["model"]["text_context_cache"] = f"{_lora_experiment}/text-cache/context-cache/manifest.json"

_config["dataset"].update(
    reference_policy="distinct_camera_shared_only",
    sampling_manifest_path=f"{_inputs}/sampling/train-sampling.jsonl",
    reference_entities={"labels_path": f"{_inputs}/labels/train-reference-entities.jsonl"},
    reference_crops={"path": f"{_inputs}/crops/train-reference-crops.jsonl"},
    text_admission_path=f"{_inputs}/text/train-text-admission.json",
    target_index_path=f"{_targets}/consumer-index-202k.jsonl",
    cache_provenance_path=f"{_targets}/consumer-source-collection-202k.json",
)
_config["training"]["validation"].update(
    reference_entities={"labels_path": f"{_inputs}/labels/val-reference-entities.jsonl"},
    text_admission_path=f"{_inputs}/text/val-text-admission.json",
)

# 延续定稿的 full_dit_v1、ZeRO-2、存盘间隔和状态保留数，使用已验证的 30 层注意力缓存。
_config["training"].update(
    gradient_accumulation_steps=2,
    max_train_steps=6000,
    activation_checkpointing="save_self_attention",
    self_attention_cache_layers=30,
)
_config["optimization"] = {
    "learning_rate": 1.4e-5,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine_with_warmup",
    "warmup_steps": 200,
    "min_lr_ratio": 0.1,
}
_config["output"] = {
    "root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
            "renderer-wan22-5b-fulldit-distinct-camera-20261121/training",
}

EXPERIMENT_CONFIG = _config
