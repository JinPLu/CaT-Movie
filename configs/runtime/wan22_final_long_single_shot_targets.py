"""Wan2.2 定稿训练补编长单镜目标潜变量的清单配置：只给 ``tools/data/build_shot_targets.py manifest`` 用，不训练。

定稿新抽样表的单镜层一半只抽不短于 5 秒、有动作的长单镜；其中 2,053 个训练目标镜在现有 704p/24fps 目标缓存的两个来源
（8 秒全量镜、按窗加权镜）里都没有。清单工具只认一个 ``shot`` 根、且根目录名就是缓存版本，所以这里只把定稿配置的
目标根换成补编根，成员、抽样表、媒体、验证集都不动，清单与运行阶段逐键对得上。
编码合同与现有缓存完全相同：同一 VAE 文件与版本、同一份冻结生产代码（实现版本 1ad6307de）、BF16 不切块、
24fps 帧规划、逐镜 reset、潜变量 48×44×80；生产配置沿用该缓存冻结的 ``precompute.json``，只换清单与来源合同路径。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).parents[1] / "train/renderer_wan22_final.py"))["EXPERIMENT_CONFIG"])
_config["dataset"]["target_roots"] = {"shot": (
    "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/renderer-wan22-final-20261113/"
    "target-cache/long-single-shots/renderer-720p24-targets-20261108")}
EXPERIMENT_CONFIG = _config
