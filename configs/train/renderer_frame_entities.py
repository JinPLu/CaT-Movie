"""整帧参考对照：可见ID文字与完整目标语义/布局，原生Wan初始化。

共同参考合同、数值栈与曝光服务于同一论文实验；不将修复本身计为方法贡献。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_text_layout_from_base_collaborator.py")))["EXPERIMENT_CONFIG"])
_name = "whole_frame_visible_identity"
_root = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_inputs = f"{_root}/entity-planning-training-inputs-20261101/inputs"
_output = f"{_root}/renderer-entity-planning-20261101/{_name}"
_config["identity"] = {**_config["identity"], "run_name": _name, "model_key": f"wan22_5b_{_name}", "experiment_key": "source_identity_target_layout"}
_config["model"]["memory_prefix"]["row_policy"] = "frame_entities_v1"
_config["model"]["source_identity"] = None
# 五组以完整目标布局文字为共同强基线；旧0.2整组删布局没有误差标定依据。
_config["dataset"]["box_dropout_p"] = 0.0
_config["dataset"]["reference_entities"] = {"labels_path": f"{_inputs}/train-reference-entities.jsonl"}
_config["training"]["validation"]["reference_entities"] = {"labels_path": f"{_inputs}/val-reference-entities.jsonl"}
_config["output"] = {"root": f"{_output}/training"}
_config["checkpoint"] = None
EXPERIMENT_CONFIG = _config
