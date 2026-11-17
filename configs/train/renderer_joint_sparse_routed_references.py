"""联合段稀疏注意力＋按镜路由的参考：整帧可见身份配方（1–4 镜联合段冻结表及其参考身份标签）换成 HoloCine 首帧稀疏
自注意力带参考，从原生 Wan2.2-TI2V-5B 起训。

与 ``renderer_frame_entities.py`` 只差：

- DiT 自注意力 ``holocine_first_frame``：每镜读本镜全部潜帧与各镜首潜帧（本镜首潜帧重复一份）；时间位置改为整段连续
  原生时间 ``global_time_native_rope_v1``（不加镜相位）。
- 参考注意力 ``routed_self_only_v1``：参考按 ``mspc.data.reference_routing`` 路由给目标镜（与参考源镜共享在场实体的镜，
  一镜都不共享时给全部镜；不按场所），每镜只读路由给它的参考，参考键在时间上移到本镜起点之前（镜 i 起点 s_i、第 n
  张参考在 s_i − 5(M−n)），少一些注意力稀释。镜读到全部参考时到参考的距离与
  逐镜生成相同，只读到一部分时参考仍在全局槽位、中间有空槽。参考只读自己这一帧，不读目标与其他参考。
- 目标文字省掉 ``layouts``（方位布局句），画风句按样本随机保留一半（``picture_type_dropout_p=0.5``），与单镜目标第二版
  同一文字口径；布局句既省，``box_dropout_p`` 与布局覆盖率下限一并去掉。
- 8 卡 × micro1 × 累积 1（global batch 仍为 8），只存第 3000 步；ZeRO-2 通信设置与单镜目标第二版相同。

其余（480p、16 fps；参考 1–10 张、每镜随机一帧、共享实体的镜先取、整组丢弃 0.2；SAM 前景权重、3000 更新、
学习率与数值栈、验证面板）不变。整组丢弃参考（M=0）的样本没有路由，走 holocine 目标自己的注意力。
"""
import copy
import runpy
from pathlib import Path

_config = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name("renderer_frame_entities.py")))["EXPERIMENT_CONFIG"])
_name = "joint_sparse_routed_references"
_output = f"${{MSPC_SHARED_ROOT}}/Storyboard/outputs/experiments/renderer-joint-routed-20261104/{_name}"
_config["identity"] = {**_config["identity"], "run_name": _name, "model_key": f"wan22_5b_{_name}",
                       "experiment_key": "joint_generation_with_routed_references"}
_config["model"]["self_attention_policy"] = "holocine_first_frame"
_config["model"]["temporal_position_mode"] = "global_time_native_rope_v1"
_config["model"]["memory_prefix"]["attention_policy"] = "routed_self_only_v1"
_config["model"]["omit_target_text"] = ["layouts"]
_config["dataset"]["picture_type_dropout_p"] = 0.5
_config["dataset"]["box_dropout_p"] = None
_config["dataset"]["minimum_layout_shot_fraction"] = None
_config["training"]["gradient_accumulation_steps"] = 1
_config["training"]["save_steps"] = 3000
_config["runtime"]["deepspeed"]["zero_optimization"].update(
    contiguous_gradients=True, overlap_comm=True, reduce_bucket_size=500_000_000, allgather_bucket_size=500_000_000)
_config["output"] = {"root": f"{_output}/training"}
EXPERIMENT_CONFIG = _config
