"""联合组全量：联合段稀疏注意力＋按镜路由的参考（``renderer_joint_sparse_routed_references.py``）换成全量成员与 96,000 行
1–4 镜冻结表，从原生 Wan2.2-TI2V-5B 起训 12,000 更新。

与联合组 3,000 步那份只差输入、步数与存盘：

- 训练成员换成全量 61,804 块（``full-pool-training-inputs-20261028/renderer/inputs/train.jsonl``），取样表换成同目录 96,000 行
  1–4 镜表（镜数权重 15/25/30/30、seed 42；前 24,000 行与 3,000 步那次用的表逐字节相同），参考身份标签按这张表重产。
- SAM 前景权重换成全量那份（同为 ``area_coverage``，覆盖全量 264,257 个可用镜）。
- 验证成员换成全量 877 块，验证面板仍是同一份 256×2（``val-sampling.jsonl`` 与修订监督那份逐字节相同）。
- 8 卡 × micro1 × 累积 1（global batch 8），12,000 更新正好走完 96,000 行；每 2,400 步存盘（2,400、4,800、7,200、9,600、12,000）。

注意力、文字口径（省布局句、画风句随机保留一半）、参考（1–10 张随机帧、整组丢弃 0.2）、学习率 1e−5 与 300 步预热后恒定、
数值栈与 ZeRO-2 设置都不变。数据顺序由 ``training.seed`` 对整张 96,000 行表一次打乱，所以第 2,400 步的权重见过的是全表里
随机的 19,200 行，不是表的前 19,200 行。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(
    str(Path(__file__).with_name("renderer_joint_sparse_routed_references.py")))["EXPERIMENT_CONFIG"])
_name = "joint_routed_references_full_pool"
_root = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_pool = f"{_root}/full-pool-training-inputs-20261028/renderer"
_inputs = f"{_root}/full-pool-renderer-inputs-20261105"
_output = f"{_root}/renderer-full-pool-20261105/{_name}"
_config["identity"] = {**_config["identity"], "run_name": _name, "model_key": f"wan22_5b_{_name}",
                       "dataset_key": "block_annotation_v18_quality_full_pool",
                       "experiment_key": "full_pool_joint_generation_with_routed_references"}
_config["dataset"]["members_path"] = f"{_pool}/inputs/train.jsonl"
_config["dataset"]["sampling_manifest_path"] = f"{_pool}/inputs/train-sampling.jsonl"
_config["dataset"]["reference_entities"] = {"labels_path": f"{_inputs}/joint/train-reference-entities.jsonl"}
_config["dataset"]["target_mask_weights"] = {"index_path": f"{_pool}/sam-weights/index.jsonl",
                                             "root": f"{_pool}/sam-weights", "weighting": "area_coverage"}
_config["training"]["validation"].update(members_path=f"{_pool}/inputs/val.jsonl",
                                         sampling_manifest_path=f"{_pool}/inputs/val-sampling.jsonl")
_config["training"]["max_train_steps"] = 12000
_config["training"]["save_steps"] = 2400
_config["output"] = {"root": f"{_output}/training"}
EXPERIMENT_CONFIG = _config
