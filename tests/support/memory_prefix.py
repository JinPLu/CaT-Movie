"""训练与推理测试共用的参考负时间、原空间网格合同。"""
from __future__ import annotations

from mspc.models.memory_prefix_options import MemoryAttentionWindow, ReferenceNoiseOptions, ReferenceOffsetOptions

INFERENCE_OFFSET = (-5, 0, 0)
OFFSET_CONFIG = {"train": {"dt": [-5, -5], "dh": [0, 0], "dw": [0, 0]}, "inference": [-5, 0, 0]}
NOISE_CONFIG = {"train_sigma_max": 0.0, "inference_sigma": 0.0}


def noise_options() -> ReferenceNoiseOptions:
    return ReferenceNoiseOptions(0.0, 0.0)


def offset_options() -> ReferenceOffsetOptions:
    return ReferenceOffsetOptions.from_mapping(OFFSET_CONFIG, "test.reference_offset")

#: 现行全部配置的参考注意力窗口：全部 30 层、扩散时间 0–1000。
FULL_WINDOW_CONFIG = {"layer_indices": list(range(30)), "timestep_range": [0, 1000]}


def full_window() -> MemoryAttentionWindow:
    return MemoryAttentionWindow.from_mapping(FULL_WINDOW_CONFIG, "test.attention_window")
