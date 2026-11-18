"""Wan2.2-A14B 的两个去噪专家：高噪专家管 ``t ≥ boundary·1000``，低噪专家管其余时间步。

官方推理（``wan/text2video.py``）在 ``t >= boundary * num_train_timesteps`` 时用高噪模型，否则用低噪模型；
T2V-A14B 的 ``boundary`` 是 0.875。训练时每个 run 只训一个专家，时间步只从它自己那一段里抽
（DiffSynth-Studio 官方 A14B 脚本同样分专家各训各段）。训练与推理读同一个 ``boundary``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

NOISE_LEVELS = ("high", "low")


def denoising_expert(timestep: float, boundary: float) -> str:
    """原生 Wan 的边界属于高噪专家；路由与逐专家 CFG 共用这一判据。"""
    return "high_noise" if timestep >= boundary * 1000 else "low_noise"


def denoising_guidance_scale(cfg_scale: float | Mapping[str, float], timestep: float,
                             boundary: float | None) -> float:
    """标量 CFG 用于整条轨迹；双专家 CFG 按当前实际时间步取值。"""
    if isinstance(cfg_scale, Mapping):
        if boundary is None:
            raise ValueError("per-expert CFG requires an explicit expert boundary")
        return float(cfg_scale[denoising_expert(timestep, boundary)])
    return float(cfg_scale)


@dataclass(frozen=True)
class NoiseExpertOptions:
    noise_level: str
    boundary: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NoiseExpertOptions":
        if not isinstance(value, Mapping) or set(value) != {"noise_level", "boundary"}:
            raise ValueError("model.expert needs exactly noise_level and boundary")
        level, boundary = value["noise_level"], value["boundary"]
        if level not in NOISE_LEVELS:
            raise ValueError(f"model.expert.noise_level must be one of {NOISE_LEVELS}")
        if type(boundary) is not float or not 0.0 < boundary < 1.0:
            raise ValueError("model.expert.boundary must be a float in (0, 1)")
        return cls(noise_level=level, boundary=boundary)


def expert_timestep_index_range(timesteps: torch.Tensor, options: NoiseExpertOptions,
                                num_train_timesteps: int) -> tuple[int, int]:
    """训练调度器时间步表（降序）里属于该专家的下标区间 ``[lo, hi)``。"""
    values = timesteps.detach().float().cpu()
    if values.ndim != 1 or len(values) < 2 or not bool((values[1:] <= values[:-1]).all()):
        raise ValueError("expert ranges need the descending 1-D training timestep table")
    high = int((values >= options.boundary * num_train_timesteps).sum())
    if not 0 < high < len(values):
        raise ValueError(f"boundary {options.boundary} leaves one expert without training timesteps")
    return (0, high) if options.noise_level == "high" else (high, len(values))


def index_range_as_boundaries(index_range: tuple[int, int], count: int) -> tuple[float, float]:
    """转成损失读取的比例（``int(比例 × count)`` 必须还原出同一下标）。"""
    lo, hi = index_range
    fractions = []
    for index in (lo, hi):
        fraction = index / count
        if int(fraction * count) != index:
            fraction = float(torch.nextafter(torch.tensor(fraction, dtype=torch.float64),
                                             torch.tensor(2.0, dtype=torch.float64)))
        if int(fraction * count) != index:
            raise ValueError(f"timestep index {index} cannot be expressed as a fraction of {count}")
        fractions.append(fraction)
    return fractions[0], fractions[1]
