"""门控位置 token：另开一条「视频 token 读实体位置 token」的读取通道，不改原文字交叉注意力。

每个有框实体一个位置 token＝小网络（该实体描述词的文字特征平均 ⊕ 首锚框的傅里叶编码），另配一个可学习的空 token。
视频 token 只能读到区域覆盖到自己的实体 token（按覆盖率加对数权重），空 token 总可读；输出乘以区域权重的最大值
与 tanh(门控)。门控初值 0：开训时 DiT 的输出与未加通道时逐位相同，训练学「听多少」。
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from mspc.models.spatial_layout import RegionTokenShot, fourier_box_features

REGION_TOKEN_ENCODER = "region_token_encoder"
REGION_ATTENTION = "region_attn"


class RegionTokenEncoder(nn.Module):
    def __init__(self, dim: int, fourier_frequencies: int):
        super().__init__()
        if type(fourier_frequencies) is not int or fourier_frequencies < 1:
            raise ValueError("fourier_frequencies must be a positive int")
        self.fourier_frequencies = fourier_frequencies
        self.mlp = nn.Sequential(nn.Linear(dim + 8 * fourier_frequencies, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null_token = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, context: torch.Tensor, row: RegionTokenShot) -> torch.Tensor:
        """context：该镜已过 text_embedding 的文字行 [L, dim]；返回 [实体数+1, dim]，最后一行是空 token。"""
        if context.ndim != 2:
            raise ValueError("region tokens read one embedded text row [tokens, dim]")
        tokens = []
        for indices, box in zip(row.entity_token_indices, row.boxes, strict=True):
            text = context[list(indices)].mean(dim=0)
            position = fourier_box_features(box, self.fourier_frequencies, device=context.device).to(context.dtype)
            tokens.append(self.mlp(torch.cat([text, position])))
        tokens.append(self.null_token.to(context.dtype))
        return torch.stack(tokens)


class GatedRegionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, rows: Sequence[RegionTokenShot],
                region_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        """x：[1, 视频 token, dim]；返回同形状的增量，区域外、参考前缀与非开头帧为 0。"""
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("region attention requires one video batch")
        if len(rows) != len(region_tokens):
            raise ValueError("one region token set is required per target shot")
        delta = torch.zeros_like(x)
        heads, width = self.num_heads, x.shape[-1] // self.num_heads
        for row, tokens in zip(rows, region_tokens, strict=True):
            entities = row.region_weights.shape[1]
            if tokens.shape[0] != entities + 1:
                raise ValueError("region tokens must be entities plus one null token")
            if entities == 0:
                continue
            weights = row.region_weights.to(device=x.device, dtype=torch.float32)
            scale = weights.max(dim=1).values
            active = scale > 0
            if not bool(active.any()):
                continue
            query = self.norm(x[0, row.token_start:row.token_end][active])
            q = self.norm_q(self.q(query)).view(-1, heads, width).transpose(0, 1)
            k = self.norm_k(self.k(tokens)).view(-1, heads, width).transpose(0, 1)
            v = self.v(tokens).view(-1, heads, width).transpose(0, 1)
            # 实体列加 log(覆盖率)，覆盖率 0 即读不到；空 token 列加 0，总可读。
            logit_bias = torch.cat([weights[active].log(), weights.new_zeros(int(active.sum()), 1)], dim=1)
            logits = (q.float() @ k.float().transpose(1, 2)) / math.sqrt(width) + logit_bias[None]
            read = (logits.softmax(dim=-1).to(v.dtype) @ v).transpose(0, 1).reshape(-1, heads * width)
            out = self.o(read) * (torch.tanh(self.gate) * scale[active][:, None]).to(read.dtype)
            region = delta[0, row.token_start:row.token_end]
            region[active] = out.to(region.dtype)
        return delta


def attach_region_attention(dit: nn.Module, *, layer_indices: Sequence[int], fourier_frequencies: int) -> None:
    """在给定层挂门控通道、在模型上挂位置 token 网络；新参数随 DiT 的设备与精度，均可训练。"""
    if hasattr(dit, REGION_TOKEN_ENCODER):
        raise ValueError("region attention is already attached")
    blocks = dit.blocks
    if not layer_indices or any(type(i) is not int or not 0 <= i < len(blocks) for i in layer_indices) \
            or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("region attention layer_indices must be distinct DiT block indices")
    reference = next(dit.parameters())
    dim, heads = blocks[0].dim, blocks[0].num_heads
    encoder = RegionTokenEncoder(dim, fourier_frequencies).to(device=reference.device, dtype=reference.dtype)
    dit.add_module(REGION_TOKEN_ENCODER, encoder)
    for index in layer_indices:
        module = GatedRegionAttention(dim, heads).to(device=reference.device, dtype=reference.dtype)
        blocks[index].add_module(REGION_ATTENTION, module)


def region_parameter_names(dit: nn.Module) -> set[str]:
    """新增参数的完整名字，用于单独学习率与从无此通道的暖启权重加载。"""
    return {name for name, _ in dit.named_parameters()
            if name.startswith(REGION_TOKEN_ENCODER + ".") or f".{REGION_ATTENTION}." in name}


def attach_layout_modules(dit: nn.Module, options) -> None:
    """按布局配置挂新增模块：只有门控模式需要；其他模式与无布局时 DiT 结构不变。"""
    from mspc.models.spatial_layout import GATED_REGION_MODE
    if options is not None and options.mode == GATED_REGION_MODE:
        attach_region_attention(dit, layer_indices=options.layer_indices,
                                fourier_frequencies=options.region_module["fourier_frequencies"])


def region_channel_anchor(dit: nn.Module) -> torch.Tensor:
    """全部新通道参数之和乘 0：加到视频 token 上数值不变，但让每一步每张卡都对新通道产生梯度。"""
    parameters = [p for name, p in dit.named_parameters()
                  if name.startswith(REGION_TOKEN_ENCODER + ".") or f".{REGION_ATTENTION}." in name]
    if not parameters:
        raise ValueError("the gated region mode requires the attached region channel")
    return sum(p.float().sum() for p in parameters) * 0.0
