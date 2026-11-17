"""DiT 哪些参数训练（全量或 LoRA）与完整 checkpoint 的严格加载。"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from mspc.config.experiment import (
    WAN22_DIT_PARAMETER_FREEZE_FULL_DIT_V1,
    WAN22_DIT_PARAMETER_FREEZE_LORA_V1,
    parse_wan22_dit_parameter_freeze,
)

FULL_DIT_V1 = WAN22_DIT_PARAMETER_FREEZE_FULL_DIT_V1
LORA_V1 = WAN22_DIT_PARAMETER_FREEZE_LORA_V1


def require_dit_parameter_freeze(value: object, *, location: str = "training.dit_parameter_freeze") -> dict[str, Any]:
    """训练端拿到的是解析后冻结的配置（只读映射、元组），还原成原始 JSON 形状再走配置的同一校验。"""
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: plain(entry) for key, entry in item.items()}
        if isinstance(item, tuple):
            return [plain(entry) for entry in item]
        return item
    return parse_wan22_dit_parameter_freeze(plain(value), location)


def apply_named_dit_freeze(dit: nn.Module, spec: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """全量：全部 DiT 参数可训练。LoRA：冻结原参数，只训挂上去的低秩增量。"""
    spec = require_dit_parameter_freeze(spec)
    if spec["recipe"] == LORA_V1:
        from mspc.models.lora import inject_lora
        trainable = inject_lora(dit, rank=spec["rank"], alpha=float(spec["alpha"]), targets=spec["targets"])
        frozen = tuple(name for name, parameter in dit.named_parameters() if not parameter.requires_grad)
        return {"frozen": frozen, "trainable": trainable}
    names = []
    for name, parameter in dit.named_parameters():
        parameter.requires_grad_(True)
        names.append(name)
    if not names:
        raise RuntimeError("full_dit_v1 matched no DiT parameters")
    return {"frozen": (), "trainable": tuple(names)}


def require_dit_trainable(dit: nn.Module, spec: Mapping[str, Any]) -> None:
    """冻结编码器后 DiT 保持训练态；全量时每个参数可训练，LoRA 时恰好只有低秩参数可训练。"""
    from mspc.models.lora import LORA_PARAMETER_SUFFIXES
    parameters = tuple(dit.named_parameters())
    if not dit.training or not parameters:
        raise RuntimeError("DiT training requires a DiT in training mode")
    lora = require_dit_parameter_freeze(spec)["recipe"] == LORA_V1
    for name, parameter in parameters:
        if parameter.requires_grad != (name.endswith(LORA_PARAMETER_SUFFIXES) if lora else True):
            raise RuntimeError(f"DiT parameter {name} trainability does not match {spec['recipe']}")


def load_dit_state_dict(dit: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> str:
    """完整权重只做严格加载；旧结构的产物用其原始执行 SHA 读取。

    唯一例外：DiT 已挂新通道（门控位置 token 通道、规划器 token 连接器）、而权重是没有这条通道的暖启点时，
    权重必须恰好缺这条通道的全部键（其余逐键严格），新通道保留初值。门控位置 token 门控为 0，行为与暖启点相同；
    规划器 token 连接器末层为 0，但 token 进块后仍是同一个非零 key/value，行为与暖启点不完全相同。
    """
    from mspc.models.planner_tokens import PLANNER_CONNECTOR
    from mspc.models.region_attention import region_parameter_names
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("DiT checkpoint state dict must be a non-empty mapping")
    channel = region_parameter_names(dit) | {name for name in dit.state_dict()
                                             if name.startswith(f"{PLANNER_CONNECTOR}.")}
    if channel and not channel & set(state_dict):
        expected = set(dit.state_dict()) - channel
        if set(state_dict) != expected:
            raise ValueError("warm-start checkpoint must hold exactly the DiT keys outside the new channel; "
                             f"missing={sorted(expected - set(state_dict))[:5]} "
                             f"extra={sorted(set(state_dict) - expected)[:5]}")
        missing, unexpected = dit.load_state_dict(dict(state_dict), strict=False)
        if set(missing) != channel or unexpected:
            raise ValueError("warm-start load must leave exactly the new channel uninitialized from file")
        return "warm_start_plus_new_channel"
    dit.load_state_dict(dict(state_dict), strict=True)
    return "full"
