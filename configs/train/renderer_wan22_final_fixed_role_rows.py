"""方法消融「参考不带身份行」：定稿只改参考行写法——每张参考只写固定的「另一镜的参考帧」一句，不列逐帧可见人物。

回答引言第二条要求「文字怎样指向召回的帧，免得绑错人」：预期人物、物体降，照抄或串人增；若不降，推理可以去掉读图
服务。配置值 ``model.memory_prefix.row_policy``：``frame_entities_v1`` → ``fixed_role_v1``。这组不需要逐帧身份标签，
训练与验证的 ``reference_entities`` 都是 None（行写法与标签必须同时声明）；抽签、参考帧、路由、裁剪表都与定稿相同。
参考行文字变了，所以文字准入与文字缓存按本组另出。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_inputs = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
           "renderer-wan22-final-20261113/inputs")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_fixed_role_rows",
                       "model_key": "wan22_ti2v_5b_final_704p_fixed_role_rows"}
_config["model"]["memory_prefix"]["row_policy"] = "fixed_role_v1"
_config["dataset"]["reference_entities"] = None
_config["training"]["validation"]["reference_entities"] = None
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission-fixed-role-rows.json"
_config["training"]["validation"]["text_admission_path"] = f"{_inputs}/text/val-text-admission-fixed-role-rows.json"
_config["model"]["text_context_cache"] = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
                                          "text-cache-setting/fixed-role-rows/context-cache/manifest.json")
EXPERIMENT_CONFIG = _config
