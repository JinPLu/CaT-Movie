from .flow_match import FlowMatchScheduler, HiDreamO1FlashScheduler
from .loss import *

# 这里只导本项目真正消费的两样：调度器与 ``FlowMatchSFTLoss``（后者由
# ``mspc.models.wan22_loader`` 与 ``mspc.training.backends.wan22`` 引用）。
#
# 上游自带的训练脚手架与 DMD2、CLI parsers（``training_module`` / ``logger`` /
# ``runner`` / ``dmd2`` / ``parsers``）已随 2026-08-26 的去 vendor 化一并删除：本项目用
# ``mspc.training`` 自己的 trainer，那五个模块只互相引用，簇外零导入、导出符号零消费。
# 它们此前在这里被急切导入，把 peft、accelerate、swanlab 变成 import
# ``diffsynth.pipelines.wan_video`` 的硬前置，于是树里的行为在单测环境一条都测不到——
# story RoPE 走错好几天没人发现，就是这么来的。要恢复某个模块，从上游取那一份，
# 并且只在真正用到的地方按模块导，不要再放回这里。
