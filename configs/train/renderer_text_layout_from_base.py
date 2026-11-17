"""今日共同基线：原生5B起训，段外参考1–10张、逐镜一帧、关联软优先、时间升序。

负时间槽−5M…−5、不移空间、参考与目标全双向SA；原有逐镜语义、目标布局文字、
SAM目标前景loss及全局batch8保留。明确picture_type由共用文字编译器写入。
没有结构布局偏置；不因strength=0仍强制走SDPA数学核。
这是一组完整修复配方，不承担各项修改的独立因果归因。计划和验收见renderer-retraining.md。
"""
import copy
import runpy
from pathlib import Path

_base = runpy.run_path(str(Path(__file__).with_name("renderer_text_layout.py")))
_config = copy.deepcopy(_base["EXPERIMENT_CONFIG"])
_name = "chronological_reference_from_base"
_shared = "${MSPC_SHARED_ROOT}"
_run_root = f"{_shared}/Storyboard/outputs/experiments/renderer-planning-memory-20261101"
_config["identity"] = {**_config["identity"], "model_key": f"wan22_5b_{_name}", "run_name": _name,
                       "experiment_key": "planning_memory_ablation"}
_config["training"]["loss_tracking"] = {**_config["training"]["loss_tracking"], "experiment_name": _name}
_config["output"] = {"root": f"{_run_root}/{_name}/training"}
_config["model"]["spatial_layout"] = None
_config["checkpoint"] = None
EXPERIMENT_CONFIG = _config
