"""方法消融「不路由」：定稿只改读者范围——每个目标镜读窗内全部参考（参考仍只读自己），不再按共有实体路由。

回答引言第三条要求「限制谁读参考」：读得太宽会串内容、照抄（定稿训练表上单人镜读到含外人参考约 54%，
整窗共读约 92%）。配置值 ``model.memory_prefix.attention_policy``：``routed_self_only_v1`` →
``window_shared_self_only_v1``；抽签、参考帧、身份标签、裁剪表（包住全部人物，与读者范围无关）都与定稿相同。
读者范围改变了每个目标镜文字里引用哪几张参考，所以文字准入与文字缓存按本组另出。出片时推理端也整窗共读、
外人路由一并关掉（论文消融页第六节）。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_wan22_final.py")))["EXPERIMENT_CONFIG"])
_inputs = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
           "renderer-wan22-final-20261113/inputs")
_config["identity"] = {**_config["identity"], "run_name": "wan22_final_704p_window_shared_references",
                       "model_key": "wan22_ti2v_5b_final_704p_window_shared_references"}
_config["model"]["memory_prefix"]["attention_policy"] = "window_shared_self_only_v1"
# 负载上限沿用定稿 {69 潜帧, 4,209 帧对}：剔除只由潜帧（目标＋参考张数）决定，四组剔除同一批样本；本组帧对在 69 潜帧内
# 最大仍是单镜 l(l+1+R)+R，不超过 4,209（统筹会话 11-13 推导，起训前以本组 sample_workloads 实算核对），容量按本组最坏样本另测。
_config["dataset"]["text_admission_path"] = f"{_inputs}/text/train-text-admission-window-shared-references.json"
_config["training"]["validation"]["text_admission_path"] = (
    f"{_inputs}/text/val-text-admission-window-shared-references.json")
_config["model"]["text_context_cache"] = ("${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
                                          "text-cache-setting/window-shared-references/context-cache/manifest.json")
EXPERIMENT_CONFIG = _config
