"""Wan2.2-TI2V-5B（704p）全量 DiT：按在场实体补齐参考、去掉无关镜广播、同机位参考打乱构图。

以 ``renderer_wan22_5b_distinct_camera_704p.py``（11-22 纯异机位）为底，改两处：
1. 参考选法换成 ``distinct_camera_character_coverage``：异机位优先，按目标镜的每个在场实体补齐（缺的先补异机位、再补同机位），
   只补一张同地点场景镜，不补与本窗无关的镜（11-24 单变量对照里无关镜全窗广播占 21.86%，人物落后集中在推理时收到广播参考的镜头）；
2. 与读者同机位的参考不再裁剪，改为整帧缩到 0.6–0.8 倍贴到画布一侧（``reference_crops.same_camera_shift``），大特写也离开
   原位置（11-24 单变量对照跨窗照抄 16 → 36，全部落在路由到与前一镜同机位参考的切点）；其余参考照旧按离线裁剪表。
其余逐项不变：原生权重起训、同一张 96,000 窗抽样表、训练种子 42、参考加噪上限 0.2、整组丢弃 0.2、6,000 步余弦（峰值 1.4e-5、
warmup 200）在第 3,000 步停；16 卡（2 × 8 A100）× 单卡 1 窗 × 累积 1。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name(
    "renderer_wan22_5b_distinct_camera_704p.py")))["EXPERIMENT_CONFIG"])

_data = "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments"
_experiment = f"{_data}/renderer-wan22-5b-fulldit-character-coverage-20261125"
_inputs = f"{_experiment}/inputs"

_config["identity"].update(
    run_name="wan22_5b_fulldit_704p_character_coverage",
    model_key="wan22_ti2v_5b_fulldit_704p_character_coverage",
    experiment_key="wan22_5b_fulldit_character_coverage",
)
_config["model"]["text_context_cache"] = f"{_experiment}/text-cache/context-cache/manifest.json"
_config["dataset"].update(
    reference_policy="distinct_camera_character_coverage",
    reference_entities={"labels_path": f"{_inputs}/labels/train-reference-entities.jsonl"},
    reference_crops={"path": f"{_inputs}/crops/train-reference-crops.jsonl",
                     "same_camera_shift": {"scale": [0.6, 0.8], "edge": 0.1}},
    text_admission_path=f"{_inputs}/text/train-text-admission.json",
)
_config["training"]["validation"].update(
    reference_entities={"labels_path": f"{_inputs}/labels/val-reference-entities.jsonl"},
    text_admission_path=f"{_inputs}/text/val-text-admission.json",
)
_config["training"].update(gradient_accumulation_steps=1)
_config["output"] = {"root": "${MSPC_SHARED_ROOT}/Storyboard/outputs/experiments/"
                             "renderer-wan22-5b-fulldit-character-coverage-20261125/training"}

EXPERIMENT_CONFIG = _config
