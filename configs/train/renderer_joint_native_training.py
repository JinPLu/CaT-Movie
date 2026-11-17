"""两个原生基座新训练共用的监督与控制合同；实际入口为对应480p/720p配置。

同一96k按1/2/3/4及以上镜权重15/25/30/30抽层的约8秒完整source窗口，超预算长镜独占，训练沿同块随机镜帧参考，推理使用历史中帧，统一G_i＋本镜开态/动作/末态，
均匀MSE。12k总调度固定，外部cache-stage在3k保存完整state后停，继续时不更换数学配置。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_joint_wan_layout.py")))["EXPERIMENT_CONFIG"])
_data = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_inputs = f"{_data}/two-backbone-training-preparation-20261108/inputs"
_sampling = f"{_inputs}/shot-count-weighted-windows"
_members = f"{_inputs}/entity-numbered-actions"
_config["identity"] = {**_config["identity"],
    "run_name": "joint_native_training", "model_key": "wan22_5b_joint_native_training",
    "experiment_key": "native_backbone_joint_training",
    "dataset_key": "v18_shot_count_weighted_complete_source_windows"}
_config["dataset"]["members_path"] = f"{_members}/train.jsonl"
_config["dataset"]["sampling_manifest_path"] = f"{_sampling}/train-sampling.jsonl"
_config["dataset"]["reference_policy"] = "random_frame"
_config["dataset"]["target_mask_weights"] = None
_config["dataset"]["reference_entities"] = None
_config["dataset"]["reference_row_dropout_p"] = 0.0
_config["dataset"]["picture_type_dropout_p"] = 0.5
_config["dataset"]["box_dropout_p"] = 0.5
# 保留已训练的随机同块参考供给与整组丢弃；身份未知仍保留像素。
_config["model"]["memory_prefix"]["dropout_p"] = 0.2
# 连续flow sigma，按Wan实际model t<=400适配；不是pre-shift schedule比例。
_config["model"]["memory_prefix"]["noise"] = {"train_sigma_max": 0.4, "inference_sigma": 0.0}
_config["model"]["omit_target_text"] = []
_config["training"].update(max_train_steps=12000, save_steps=3000, state_checkpoint={"keep": 1},
    activation_checkpointing="recompute_all", self_attention_cache_layers=None, workload_cost=None)
_config["training"]["validation"].update(
    every_steps=500,
    intermediate_panel={"sample_budget": 32, "draws": 1},
    members_path=f"{_members}/val.jsonl",
    sampling_manifest_path=f"{_sampling}/val-sampling.jsonl",
    reference_entities=None)
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
                             "renderer-native-backbones-20261108/base/training"}
EXPERIMENT_CONFIG = _config
