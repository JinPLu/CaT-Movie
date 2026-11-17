"""DiT 低秩适配（LoRA）：冻结原权重，在指定线性层旁挂可训练的低秩增量。

``W x + (alpha / rank) · B (A x)``；A 按 Kaiming 均匀初始化、B 为零，所以挂上之后、训练之前前向与原模型逐位相同。
低秩参数常驻 FP32（优化器主副本），前向在外层 BF16 autocast 下与原矩阵同精度计算。
挂在哪些层由 ``targets`` 给出，按块内相对名匹配（``self_attn.q``、``cross_attn.o``、``ffn.0`` …），
每个块都必须挂到全部目标，缺一个就报错。推理用同一挂法加载低秩权重、不并回原矩阵：BF16 autocast 下并回会改舍入，
与训练时的前向不再逐位相同。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

LORA_PARAMETER_SUFFIXES = (".lora_down.weight", ".lora_up.weight")


class LoRALinear(nn.Module):
    """包住一个冻结的 ``nn.Linear``；只有 ``lora_down``／``lora_up`` 两个参数可训练。"""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA wraps nn.Linear, got {type(base).__name__}")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        device = base.weight.device
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def inject_lora(dit: nn.Module, *, rank: int, alpha: float, targets: Iterable[str]) -> tuple[str, ...]:
    """冻结整个 DiT，再在每个块的全部目标线性层上挂 LoRA；返回可训练参数名。"""
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("LoRA targets must be a non-empty list without repeats")
    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("LoRA expects a DiT with a non-empty blocks ModuleList")
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    for index, block in enumerate(blocks):
        for target in targets:
            *parents, leaf = target.split(".")
            owner = block
            for part in parents:
                owner = getattr(owner, part) if not part.isdigit() else owner[int(part)]
            base = owner[int(leaf)] if leaf.isdigit() else getattr(owner, leaf, None)
            if not isinstance(base, nn.Linear):
                raise ValueError(f"blocks.{index}.{target} is not an nn.Linear: {type(base).__name__}")
            wrapped = LoRALinear(base, rank=rank, alpha=alpha)
            if leaf.isdigit():
                owner[int(leaf)] = wrapped
            else:
                setattr(owner, leaf, wrapped)
    names = tuple(name for name, parameter in dit.named_parameters() if parameter.requires_grad)
    expected = len(blocks) * len(targets) * 2
    if len(names) != expected or not all(name.endswith(LORA_PARAMETER_SUFFIXES) for name in names):
        raise RuntimeError(f"LoRA injection left {len(names)} trainable parameters, expected {expected} LoRA tensors")
    return names



def load_lora_state_dict(dit: nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    """把一份训练导出的 LoRA 权重（键与挂好的低秩参数逐个相同，缺一个多一个都拒）读进已挂 LoRA 的 DiT。"""
    expected = {name for name, _ in dit.named_parameters() if name.endswith(LORA_PARAMETER_SUFFIXES)}
    if not expected:
        raise ValueError("load a LoRA export only into a DiT that already carries the same LoRA")
    if set(state_dict) != expected:
        raise ValueError("LoRA export keys differ from the mounted LoRA; "
                         f"missing={sorted(expected - set(state_dict))[:3]} extra={sorted(set(state_dict) - expected)[:3]}")
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if unexpected or set(missing) & expected:
        raise ValueError("LoRA export did not load onto every low-rank parameter")
    return len(expected)
