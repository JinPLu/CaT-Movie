"""方法消融「不给参考训练」：定稿去掉整个参考前缀，从原始 Wan2.2-TI2V-5B 起训，其余逐项相同。

等于在我们同一份数据、同一底座、同一窗口与训练量上训一个 HoloCine 式联合基线（窗内各镜经首潜帧互读、逐镜文字），
量出参考机制的全部贡献；推理时也不给参考。用它替换窗内全互读，理由：定稿权重上免训练的「不给参考帧」行里
模型八成时间带参考学，不带参考的表现会偏低、高估参考的贡献。
配置值：``model.memory_prefix`` → None；参考标签、裁剪表、参考行丢弃一并为 None（与记忆前缀同时声明）。
目标文字与定稿逐字相同，文字缓存从定稿缓存复用。负载上限仍按 69 潜帧：去掉参考后只剩目标潜帧（≤60），
原先因「目标＋参考 > 69」剔除的 6 条样本这里会保留（95,998 对定稿 95,992）。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_experiment = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
               "renderer-wan22-final-20261113")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_without_references",
                       "model_key": "wan22_ti2v_5b_final_704p_without_references"}
_config["model"]["memory_prefix"] = None
_config["dataset"]["reference_entities"] = None
_config["dataset"]["reference_crops"] = None
_config["dataset"]["reference_row_dropout_p"] = None
_config["training"]["validation"]["reference_entities"] = None
_config["dataset"]["text_admission_path"] = f"{_experiment}/inputs/text/train-text-admission-without-references.json"
_config["training"]["validation"]["text_admission_path"] = f"{_experiment}/inputs/text/val-text-admission-without-references.json"
_config["model"]["text_context_cache"] = f"{_experiment}/text-cache-setting/without-references/context-cache/manifest.json"
# 这组在 collaborator HOST 上训：存盘写 ${MSPC_SHARED_ROOT}/ckp（collaborator 可写、主服务器只读可见，出片不用再拷权重）；
# collaborator 本机盘只剩约 93 GB。
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/ckp/renderer-wan22-final-20261113/training"}
EXPERIMENT_CONFIG = _config
