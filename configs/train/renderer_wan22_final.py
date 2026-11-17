"""Wan2.2-TI2V-5B 定稿训练：从原生权重起训，表 1 我们这一行；三组方法消融各从它只改一个配置值。

与现役 Wan2.2 联合组（``renderer_joint_wan22_720p.py``）同底座、
同监督、同读法（首潜帧互读、参考按镜路由且只读自己、逐帧可见人物身份行）与 12k 调度，差别全在训练数据：
- 参考照旧同块窗前窗后随机抽（``reference_policy="random_frame"``）：张数 1…10 均匀、共有实体优先、随机帧；
  推理端的选帧与外人路由都在这个分布之内，在同一份权重上免训练比较；
- 每张参考都尝试放大裁剪（实际约 47%），裁剪框包住参考帧里全部登记人物（离线表 ``reference_crops``，只训练；
  与读者范围无关，四组共用一份）；
- 新抽样表：四层 15／25／30／30 不变，单镜层一半只抽不短于 5 秒、有动作或镜内变化的长单镜；
- 训练文字写全在场登记物体（文字子任务）；
- 按负载剔除超上限样本（``workload_cap``，按已测八卡包络）。
三组消融：``renderer_wan22_final_fixed_role_rows.py``（参考不带身份行）、
``renderer_wan22_final_window_shared.py``（不路由）、``renderer_wan22_final_without_references.py``（不给参考训练）。
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
# 参考加噪上限 0.4→0.2（再低就认不出人）：σ=0.2 时加噪参考解码后与干净
# 解码的 DINOv2 余弦中位 0.645，0.3 时 0.262，已不如同故事另一镜（0.475）像它自己。
_config["model"]["memory_prefix"]["noise"] = {"train_sigma_max": 0.2, "inference_sigma": 0.0}
_config["dataset"]["sampling_manifest_path"] = f"{_inputs}/sampling/train-sampling.jsonl"
_config["dataset"]["reference_entities"] = {"labels_path": f"{_inputs}/labels/train-reference-entities.jsonl"}
_config["dataset"]["reference_crops"] = {"path": f"{_inputs}/crops/train-reference-crops.jsonl"}
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission.json"
# 文字缓存按完整 12k 阶段、每组一份（清单合同含本组文字准入）；共同的串复用旧标准缓存，新串只编一次。
# 11-16 组批改为先剔除再分组、按实测耗时标定成本后，12k 阶段的次序变了，epoch 尾被 drop_last 丢掉的 5 条样本也随之不同：
# 文字缓存按新阶段重发（旧缓存原样复用、只补编新串），目标潜变量消费索引在旧索引上只扩不换（多 1 镜，三个来源里本来就有）。
_config["model"]["text_context_cache"] = f"{_experiment}/text-cache-setting/final-balanced/context-cache/manifest.json"
# 已测八卡包络（69 总潜帧、4,209 注意力帧对，首潜帧互读口径）；
# 参考不带身份行一组抽签与负载相同，沿用；不路由、窗内全互读两组注意力帧对变大，各自另测后再填。
_config["dataset"]["workload_cap"] = {"latent_frames": 69, "attention_frame_pairs": 4209}
# 组批成本按实测耗时标定：定稿续训第 3,001 步起各卡逐样本前向秒数拟合「模型输入潜帧＋自注意力帧对」，
# R² 0.933（只用潜帧 0.806）。原先只按潜帧分组，长单镜的注意力被低估，各卡前向差约 1 秒；离线重放模拟降到约 0.1 秒。
# 只改一个 epoch 内的次序。
_config["training"]["workload_cost"] = {"linear_seconds_per_latent": 0.066824,
                                        "attention_seconds_per_frame_pair": 0.001297}
# 每 1,000 步存一次推理权重与完整状态（完整状态只留最新一份）：11-17 两组停在第 6,661／6,688 步时最新的推理权重还是
# 第 6,000 步，间隔太大。续训迁移放行存盘间隔改为 1,000。消融读数仍取第 3,000 步那份导出。
_config["training"]["save_steps"] = 1000
# 目标潜变量：旧消费索引只选了旧阶段要的 103,365 镜，新抽样表的长单镜有 5,344 个训练目标镜不在里面。其中 3,291 镜
# 8 秒全量镜来源里本来就有（只是没被选），2,053 镜两个旧来源都没有，按现有 704p 缓存同一编码合同补编成第三个来源
# （清单配置 ``configs/runtime/wan22_final_long_single_shot_targets.py``）。消费索引按本阶段训练 104,023 + 验证 508 个
# 目标镜从三个来源重新合并，来源集合随之重发；旧消费索引与旧集合原样不动。
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
