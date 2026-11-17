"""主线简单版（``renderer_text_layout_from_base.py``）在主服务器 collaborator A100 上跑的那一份：配方不变，只换这台主机的运行条件。

缓存来源合同与目标潜变量同属全量训练目标库、cu128 环境、曲线由本机转发。
collaborator 磁盘只余约 180 GB，每个检查点约 10 GB，只存第 1500、3000 步。
"""
import copy
import runpy
from pathlib import Path

_base = runpy.run_path(str(Path(__file__).with_name("renderer_text_layout_from_base.py")))
_config = copy.deepcopy(_base["EXPERIMENT_CONFIG"])
_shared = "${MSPC_SHARED_ROOT}"
_full = f"{_shared}/Storyboard/outputs/experiments/full-pool-training-inputs-20261028/renderer"
_config["dataset"]["cache_provenance_path"] = f"{_full}/inputs/cache-source-collection.json"
_config["dataset"]["target_index_path"] = f"{_full}/inputs/target-index.jsonl"
_config["dataset"]["target_roots"] = {"first_batch_window": f"{_full}/targets/first_batch_window",
                                      "shot": f"{_full}/targets/shot"}
_config["runtime"]["python"] = f"{_shared}/env/wan22_fun_sm80_cu128/bin/python"
_config["training"]["loss_tracking"] = None
_config["training"]["save_steps"] = 1500
EXPERIMENT_CONFIG = _config
