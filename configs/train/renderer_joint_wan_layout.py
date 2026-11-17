"""最终联合配方：HoloCine 首潜帧稀疏窗内互读、按镜路由的参考（与目标同一空间网格）、按 Wan 写法带布局、画风每镜都带、
窗口只限总帧数；全量成员，从原生 Wan2.2-TI2V-5B 起训 12,000 步。

与联合组全量重训（``renderer_joint_routed_references_full_pool.py``）只差三项：

- 目标文字带布局：不再省 ``layouts``，按 Wan 写法编成构图词与人物之间的位置，照标注全写、不丢。
- 画风句每镜都带：去掉按样本丢一半。
- 窗口表：块内连续可用镜、总帧数 ≤228、镜数不封顶，按时长档取样；
  ``tools/data/prepare_renderer_training.py draw`` 冻出（三档 ≤80／≤160／≤228 帧等权，衣着冲突定向 1,782 条为上限）。

参考仍在目标网格内（偏移 0,0）：第 7,200 步读数显示原配方的照抄随训练量降到第二版的水平（"更早一镜在参考里"判同
机位 11.4% → 5.0%），所以参考改回网格内。移出网格 (15,26) 改作消融
（``renderer_joint_offset_references_wan_layout.py``）。

步数按读数前写定的规则定为 12,000（第 7,200 步对第 3,000 步人物 +5.3*、切镜执行 +6.2*）。每 3,000 步存盘；
三组消融各训 3,000 步，读同一张表的前 24,000 行，与本组第 3,000 步存盘比（两边同分布、同预算，但样本不逐行相同）。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(
    str(Path(__file__).with_name("renderer_joint_routed_references_full_pool.py")))["EXPERIMENT_CONFIG"])
_name = "joint_wan_layout"
_root = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_windows = f"{_root}/frame-window-training-inputs-20261106/joint"
_output = f"{_root}/renderer-final-recipe-20261106/{_name}"
_config["identity"] = {**_config["identity"], "run_name": _name, "model_key": f"wan22_5b_{_name}",
                       "experiment_key": "joint_generation_final_recipe"}
_config["model"]["omit_target_text"] = []
_config["dataset"]["picture_type_dropout_p"] = None
_config["dataset"]["sampling_manifest_path"] = f"{_windows}/train-sampling.jsonl"
_config["dataset"]["reference_entities"] = {"labels_path": f"{_windows}/train-reference-entities.jsonl"}
_config["training"]["max_train_steps"] = 12000
_config["training"]["save_steps"] = 3000
_config["output"] = {"root": f"{_output}/training"}
EXPERIMENT_CONFIG = _config
