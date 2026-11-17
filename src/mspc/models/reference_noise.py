"""参考条件的连续flow加噪；独立样本随机域，不消费目标或全局RNG。"""
from __future__ import annotations

import math
import torch

from mspc.core.draws import unit_draw

# 与target time(seed)／target epsilon(seed^0x5DEECE66D)分开。
REFERENCE_EPSILON_SEED_XOR = 0x5245464E4F495345


def _sigma(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reference sigma must be finite in [0, 1]")
    return float(value)


def _seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("reference noise seed must be an explicit integer in [0, 2**63)")
    return value


def reference_model_timestep(sigma: float) -> float:
    """Wan after-shift模型时间；不是raw schedule index，不再施加flow shift。"""
    return 1000.0 * _sigma(sigma)


def sample_reference_sigma(*, sigma_max: float, seed: int) -> float:
    maximum = _sigma(sigma_max)
    _seed(seed)
    return 0.0 if maximum == 0 else maximum * unit_draw("reference_noise_sigma", seed)


def noise_reference_latents(clean: torch.Tensor, *, sigma: float, seed: int) -> torch.Tensor:
    """同一样本全部参考共享σ，ε逐元素独立；0直接返回旧tensor，保持clean N0。

    不用scheduler.add_noise：离散表nearest lookup不能表示任意连续σ，连t=0
    也会落在非零的最小训练σ。CPU FP32 epsilon使参考噪声不依赖GPU随机实现。
    """
    level = _sigma(sigma)
    _seed(seed)
    if not isinstance(clean, torch.Tensor) or clean.ndim != 5 or not clean.is_floating_point():
        raise ValueError("reference latents must be a floating [B,C,M,H,W] tensor")
    if level == 0 or clean.shape[2] == 0:
        return clean
    generator = torch.Generator(device="cpu").manual_seed(seed ^ REFERENCE_EPSILON_SEED_XOR)
    epsilon = torch.randn(clean.shape, dtype=torch.float32, device="cpu", generator=generator).to(clean.device)
    return ((1.0 - level) * clean.float() + level * epsilon).to(dtype=clean.dtype)
