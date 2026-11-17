"""原文字 CrossAttention 的稀疏二维布局条件与 patch 偏置。

这里只离散化可信的可见框，不插值运动轨迹。``timestep_range`` 是原
Wan scheduler 的 t（0..1000），不是 sigma、视频时间或参考帧的零噪声时间。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch


TEXT_ATTENTION_BACKENDS = ("native", "sdpa")
SPATIAL_LAYOUT_MODE = "text_cross_attention_bias_v1"
# 只用首锚、外扩区域、镜头开头若干潜帧：区域外压低实体词（不加分），或另开门控位置 token 通道。
SUPPRESSION_MODE = "text_cross_attention_suppression_v1"
GATED_REGION_MODE = "gated_region_tokens_v1"
SPATIAL_LAYOUT_MODES = (SPATIAL_LAYOUT_MODE, SUPPRESSION_MODE, GATED_REGION_MODE)
REGION_MODES = (SUPPRESSION_MODE, GATED_REGION_MODE)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class SpatialLayoutAnchor:
    time: float
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not 0 <= _finite_number(self.time, "anchor time") <= 1:
            raise ValueError("anchor time must be in [0, 1]")
        if not isinstance(self.box, tuple) or len(self.box) != 4:
            raise ValueError("anchor box must be a normalized xyxy tuple")
        x0, y0, x1, y1 = (_finite_number(value, "box coordinate") for value in self.box)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("anchor box must have positive area inside [0, 1]")


@dataclass(frozen=True)
class SpatialLayoutEntity:
    token_indices: tuple[int, ...]
    anchors: tuple[SpatialLayoutAnchor, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_indices, tuple)
            or not self.token_indices
            or any(type(index) is not int or index < 0 for index in self.token_indices)
            or len(set(self.token_indices)) != len(self.token_indices)
        ):
            raise ValueError("entity token_indices must be distinct nonnegative integers")
        if (
            not isinstance(self.anchors, tuple)
            or not 1 <= len(self.anchors) <= 2
            or any(not isinstance(anchor, SpatialLayoutAnchor) for anchor in self.anchors)
        ):
            raise ValueError("an entity requires one or two visible layout anchors")
        if len(self.anchors) == 2 and self.anchors[0].time >= self.anchors[1].time:
            raise ValueError("entity anchor times must be strictly increasing")


@dataclass(frozen=True)
class SpatialLayoutShot:
    num_frames: int
    entities: tuple[SpatialLayoutEntity, ...]
    text_token_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if type(self.num_frames) is not int or self.num_frames < 1 or (self.num_frames - 1) % 4:
            raise ValueError("layout shot num_frames must be 4n+1")
        if not isinstance(self.entities, tuple) or any(
            not isinstance(entity, SpatialLayoutEntity) for entity in self.entities
        ):
            raise ValueError("layout entities must be a tuple of SpatialLayoutEntity")
        if (
            not isinstance(self.text_token_mask, tuple)
            or not self.text_token_mask
            or any(type(valid) is not bool for valid in self.text_token_mask)
        ):
            raise ValueError("text_token_mask must mark ordinary tokens in the final padded text row")
        assigned: set[int] = set()
        for entity in self.entities:
            for index in entity.token_indices:
                if index >= len(self.text_token_mask) or not self.text_token_mask[index]:
                    raise ValueError("entity tokens must exclude padding and special text keys")
                if index in assigned:
                    raise ValueError("different entities cannot share the same text key")
                assigned.add(index)


@dataclass(frozen=True)
class SpatialLayoutCondition:
    """``strength``：加分模式是框内加分值；压低模式是区域外减分值；门控模式不用（须为 0）。

    区域模式另需 ``region_dilation``（框四边各外扩的画面比例）与 ``onset_latent_frames``（首锚起作用的潜帧数，
    权重逐帧线性降到 0）；加分模式两者都须为 None。
    """
    shots: tuple[SpatialLayoutShot, ...]
    strength: float
    layer_indices: tuple[int, ...]
    timestep_range: tuple[float, float]
    mode: Literal["text_cross_attention_bias_v1", "text_cross_attention_suppression_v1", "gated_region_tokens_v1"]
    region_dilation: float | None
    onset_latent_frames: int | None

    def __post_init__(self) -> None:
        if self.mode not in SPATIAL_LAYOUT_MODES:
            raise ValueError(f"unsupported spatial layout mode: {self.mode!r}")
        if self.mode in REGION_MODES:
            if not 0 <= _finite_number(self.region_dilation, "region_dilation") < 1:
                raise ValueError("region_dilation must be in [0, 1)")
            if type(self.onset_latent_frames) is not int or self.onset_latent_frames < 1:
                raise ValueError("onset_latent_frames must be a positive int")
        elif self.region_dilation is not None or self.onset_latent_frames is not None:
            raise ValueError("the box-bias mode takes no region_dilation or onset_latent_frames")
        if self.mode == GATED_REGION_MODE and self.strength != 0:
            raise ValueError("the gated region mode reads no text bias; strength must be 0")
        if not isinstance(self.shots, tuple) or not self.shots or any(
            not isinstance(shot, SpatialLayoutShot) for shot in self.shots
        ):
            raise ValueError("spatial layout requires one typed shot per target shot")
        if not 0 <= _finite_number(self.strength, "layout strength") <= torch.finfo(torch.float32).max:
            raise ValueError("layout strength must be nonnegative and finite in float32")
        if (
            not isinstance(self.layer_indices, tuple)
            or not self.layer_indices
            or any(type(index) is not int or index < 0 for index in self.layer_indices)
            or len(set(self.layer_indices)) != len(self.layer_indices)
        ):
            raise ValueError("layout layer_indices must be explicit distinct nonnegative integers")
        if not isinstance(self.timestep_range, tuple) or len(self.timestep_range) != 2:
            raise ValueError("timestep_range must be an explicit scheduler-t (minimum, maximum) tuple")
        lower, upper = (_finite_number(value, "scheduler timestep") for value in self.timestep_range)
        if not 0 <= lower <= upper <= 1000:
            raise ValueError("timestep_range must lie inside scheduler t in [0, 1000]")


def require_text_attention_backend(backend: str) -> str:
    if backend not in TEXT_ATTENTION_BACKENDS:
        raise ValueError(f"text_attention_backend must be one of {TEXT_ATTENTION_BACKENDS}; got {backend!r}")
    return backend


def patch_area_coverage(
    box: tuple[float, float, float, float],
    patch_height: int,
    patch_width: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """框与每个 patch 相交的面积除以 patch 面积，输出 [height, width]。"""
    if type(patch_height) is not int or type(patch_width) is not int or min(patch_height, patch_width) <= 0:
        raise ValueError("patch dimensions must be positive integers")
    x0, y0, x1, y1 = box
    rows = torch.arange(patch_height, device=device, dtype=torch.float32)
    columns = torch.arange(patch_width, device=device, dtype=torch.float32)
    height = (torch.minimum((rows + 1) / patch_height, rows.new_tensor(y1))
              - torch.maximum(rows / patch_height, rows.new_tensor(y0))).clamp_min(0) * patch_height
    width = (torch.minimum((columns + 1) / patch_width, columns.new_tensor(x1))
             - torch.maximum(columns / patch_width, columns.new_tensor(x0))).clamp_min(0) * patch_width
    return (height[:, None] * width[None, :]).clamp_(0, 1)


def build_spatial_layout_biases(
    condition: SpatialLayoutCondition,
    *,
    latent_shot_groups: Sequence[Sequence[int]],
    latent_frames: int,
    patch_height: int,
    patch_width: int,
    prefix_frames: int,
    text_length: int,
    timestep: torch.Tensor,
    num_layers: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    """逐镜 [queries, text keys] 浮点 B；所有层共享，不扩展到 attention heads。

    groups 包括真实参考前缀；condition.shots 只包括目标镜。每镜按自身
    4n+1 RGB 长度离散化：u=time*(N-1)/4，锚只支持相邻 latent 时点。
    """
    if not isinstance(condition, SpatialLayoutCondition):
        raise ValueError("spatial_layout must be a SpatialLayoutCondition")
    if type(latent_frames) is not int or type(prefix_frames) is not int or not 0 <= prefix_frames < latent_frames:
        raise ValueError("layout prefix_frames must be inside the latent frame axis")
    if type(text_length) is not int or text_length < 1:
        raise ValueError("layout text length must be positive")
    if type(num_layers) is not int or num_layers < 1 or max(condition.layer_indices) >= num_layers:
        raise ValueError("layout layer_indices exceed the actual DiT layers")
    if not isinstance(timestep, torch.Tensor) or timestep.numel() != 1:
        raise ValueError("layout requires one original scheduler timestep")
    scheduler_t = float(timestep.detach().item())
    if not math.isfinite(scheduler_t) or not 0 <= scheduler_t <= 1000:
        raise ValueError("layout timestep must be original scheduler t in [0, 1000]")
    strength = condition.strength if condition.timestep_range[0] <= scheduler_t <= condition.timestep_range[1] else 0.0
    groups: list[tuple[int, int]] = []
    previous_end = 0
    for group in latent_shot_groups:
        if len(group) != 2 or any(type(value) is not int for value in group):
            raise ValueError("layout latent groups must be integer [start, end] pairs")
        start, end = group
        if start != previous_end or end <= start:
            raise ValueError("layout latent groups must partition the latent timeline")
        groups.append((start, end))
        previous_end = end
    if previous_end != latent_frames:
        raise ValueError("layout latent groups must cover the actual latent timeline")
    if len(condition.shots) != sum(end > prefix_frames for _, end in groups):
        raise ValueError("layout shots must match target groups after the actual prefix")

    stride = patch_height * patch_width
    # Validate even an all-zero condition's geometry; wrong shapes never turn into a fallback.
    patch_area_coverage((0.0, 0.0, 1.0, 1.0), patch_height, patch_width, device=device)
    biases: list[torch.Tensor] = []
    target_index = 0
    for start, end in groups:
        bias = torch.zeros(((end - start) * stride, text_length), dtype=torch.float32, device=device)
        if end > prefix_frames:
            shot = condition.shots[target_index]
            target_index += 1
            if (shot.num_frames - 1) // 4 + 1 != end - start:
                raise ValueError("layout RGB frames do not match independently encoded shot latents")
            if len(shot.text_token_mask) != text_length:
                raise ValueError("layout token mask must match the final padded text row length")
            if condition.mode == GATED_REGION_MODE:
                raise ValueError("the gated region mode builds region token inputs, not text biases")
            if condition.mode == SUPPRESSION_MODE:
                for entity in shot.entities:
                    onset, region = onset_region_support(condition, shot, entity, end - start,
                                                         patch_height, patch_width, device=device)
                    # 区域内为 0、区域外按开头权重压低；非实体词列保持 0。
                    outside = onset[:, None, None] * (1 - region)[None]
                    bias[:, entity.token_indices] = -strength * outside.reshape(-1, 1)
            else:
                latent_times = torch.arange(end - start, device=device, dtype=torch.float32)
                for entity in shot.entities:
                    support = torch.zeros((end - start, patch_height, patch_width), device=device, dtype=torch.float32)
                    for anchor in entity.anchors:
                        u = anchor.time * (shot.num_frames - 1) / 4
                        temporal = (1 - (latent_times - u).abs()).clamp_min(0)
                        coverage = patch_area_coverage(anchor.box, patch_height, patch_width, device=device)
                        support = torch.maximum(support, temporal[:, None, None] * coverage[None])
                    bias[:, entity.token_indices] = support.reshape(-1, 1) * strength
            if start < prefix_frames:
                bias[:(prefix_frames - start) * stride].zero_()
        biases.append(bias)
    return tuple(biases)


def dilate_box(box: tuple[float, float, float, float], dilation: float) -> tuple[float, float, float, float]:
    """四边各外扩画面的 ``dilation`` 比例，截到 [0, 1]。"""
    x0, y0, x1, y1 = box
    return (max(0.0, x0 - dilation), max(0.0, y0 - dilation), min(1.0, x1 + dilation), min(1.0, y1 + dilation))


def onset_region_support(condition: SpatialLayoutCondition, shot: SpatialLayoutShot, entity: SpatialLayoutEntity,
                         latent_count: int, patch_height: int, patch_width: int, *,
                         device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """区域模式：首锚所在潜帧起 ``onset_latent_frames`` 帧的权重 [latent_count]（1 起逐帧线性降），与外扩首锚框的
    patch 覆盖率 [H, W]。只用首锚：空间只表达镜头开始时的位置与关系，不钉镜尾。"""
    if condition.mode not in REGION_MODES:
        raise ValueError("onset region support is defined only for region modes")
    first = min(entity.anchors, key=lambda anchor: anchor.time)
    start = int(math.floor(first.time * (shot.num_frames - 1) / 4))
    count = condition.onset_latent_frames
    onset = torch.zeros(latent_count, device=device, dtype=torch.float32)
    for k in range(count):
        if start + k < latent_count:
            onset[start + k] = (count - k) / count
    region = patch_area_coverage(dilate_box(first.box, condition.region_dilation), patch_height, patch_width,
                                 device=device)
    return onset, region


def fourier_box_features(box: tuple[float, float, float, float], frequencies: int, *,
                         device: torch.device | str) -> torch.Tensor:
    """框坐标的傅里叶编码：每个坐标 sin/cos(2^k·π·c)，k=0..frequencies-1，共 4·2·frequencies 维。"""
    coords = torch.tensor(box, device=device, dtype=torch.float32)
    scales = (2.0 ** torch.arange(frequencies, device=device, dtype=torch.float32)) * math.pi
    angles = coords[:, None] * scales[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1).reshape(-1)


@dataclass(frozen=True)
class RegionTokenShot:
    """一个目标镜的门控通道输入：视频 token 区间、各实体的文字 token、首锚框与 [Lq, E] 区域权重。"""
    token_start: int
    token_end: int
    entity_token_indices: tuple[tuple[int, ...], ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    region_weights: torch.Tensor


def build_region_token_inputs(
    condition: SpatialLayoutCondition,
    *,
    latent_shot_groups: Sequence[Sequence[int]],
    latent_frames: int,
    patch_height: int,
    patch_width: int,
    prefix_frames: int,
    timestep: torch.Tensor,
    device: torch.device | str,
) -> tuple[RegionTokenShot, ...] | None:
    """门控模式的逐镜输入；扩散时间不在窗口内时返回 None（该步不读位置 token）。参考前缀不参与。"""
    if not isinstance(condition, SpatialLayoutCondition) or condition.mode != GATED_REGION_MODE:
        raise ValueError("region token inputs require a gated region condition")
    scheduler_t = float(timestep.detach().item())
    if not math.isfinite(scheduler_t) or not 0 <= scheduler_t <= 1000:
        raise ValueError("layout timestep must be original scheduler t in [0, 1000]")
    if not condition.timestep_range[0] <= scheduler_t <= condition.timestep_range[1]:
        return None
    stride = patch_height * patch_width
    targets = [(start, end) for start, end in latent_shot_groups if end > prefix_frames]
    if len(targets) != len(condition.shots) or sum(end - start for start, end in latent_shot_groups) != latent_frames:
        raise ValueError("layout shots must match target groups after the actual prefix")
    rows = []
    for (start, end), shot in zip(targets, condition.shots):
        if start < prefix_frames:
            raise ValueError("a target shot cannot share latent frames with the reference prefix")
        if (shot.num_frames - 1) // 4 + 1 != end - start:
            raise ValueError("layout RGB frames do not match independently encoded shot latents")
        weights, tokens, boxes = [], [], []
        for entity in shot.entities:
            onset, region = onset_region_support(condition, shot, entity, end - start, patch_height, patch_width,
                                                 device=device)
            weights.append((onset[:, None, None] * region[None]).reshape(-1))
            tokens.append(entity.token_indices)
            boxes.append(min(entity.anchors, key=lambda anchor: anchor.time).box)
        region_weights = (torch.stack(weights, dim=1) if weights
                          else torch.zeros(((end - start) * stride, 0), device=device, dtype=torch.float32))
        rows.append(RegionTokenShot(start * stride, end * stride, tuple(tokens), tuple(boxes), region_weights))
    return tuple(rows)
