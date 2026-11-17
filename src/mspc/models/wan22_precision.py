"""Wan 的参数存储与局部计算精度分开定义。

普通矩阵在 BF16 autocast 下计算，因此保存为 BF16；时间 MLP、输出头、
归一化参数、调制参数、首帧组附加 token（``mspc.models.first_frame_tokens``）和 LoRA 低秩参数（``mspc.models.lora``）保留 FP32。优化器的 FP32 master 是另一层状态。
这是一份待真实多卡验收的训练存储方案，不是官方公开的完整训练配方。
"""
from __future__ import annotations

import torch
from torch import nn


def protected_parameter_names(dit: nn.Module) -> frozenset[str]:
    from mspc.models.lora import LORA_PARAMETER_SUFFIXES
    protected = {name for name, _ in dit.named_parameters()
                 if name.startswith(("time_embedding.", "time_projection.", "head.", "first_frame_tokens."))
                 or name.endswith((".modulation", *LORA_PARAMETER_SUFFIXES))}
    for name, module in dit.named_modules():
        # Wan's custom RMSNorm is not an nn.RMSNorm subclass. These are its
        # architecture-defined Q/K normalization module names.
        if (isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or
                name.rsplit(".", 1)[-1] in {"norm_q", "norm_k", "norm_q_img", "norm_k_img"}):
            protected.update(f"{name}.{parameter}" if name else parameter
                             for parameter, _ in module.named_parameters(recurse=False))
    return frozenset(protected)


def apply_wan22_parameter_storage(dit: nn.Module) -> None:
    """初始化时一次性分配 dtype；必须先直接加载 FP32 源，再降普通矩阵精度。"""
    protected = protected_parameter_names(dit)
    with torch.no_grad():
        for name, parameter in dit.named_parameters():
            if parameter.dtype != torch.float32 or parameter.grad is not None:
                raise ValueError("Wan mixed storage requires untouched FP32 source parameters before the optimizer")
            parameter.data = parameter.data.to(
                dtype=torch.float32 if name in protected else torch.bfloat16)


def verify_wan22_parameter_storage(dit: nn.Module) -> None:
    protected = protected_parameter_names(dit)
    for name, parameter in dit.named_parameters():
        expected = torch.float32 if name in protected else torch.bfloat16
        if parameter.dtype != expected:
            raise ValueError(f"Wan parameter {name} requires storage {expected}, got {parameter.dtype}")
