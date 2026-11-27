"""参考合同：至多十帧、有序负槽、原空间；全双向或按镜路由（``MEMORY_ATTENTION_POLICIES``）；固定角色原型或逐帧可见ID。"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping
from mspc.core.wan22_contract import (
    MEMORY_ATTENTION_POLICIES,
    MEMORY_ROW_POLICIES,
    ROUTED_MEMORY_ATTENTION_POLICIES,
)

MEMORY_PREFIX_OPTION_KEYS = (
    "frames_max", "dropout_p", "reference_offset", "attention_policy", "row_policy", "attention_window", "placement", "noise")
#: 参考放置方式 → 至多几张参考。
REFERENCE_PLACEMENTS = {"ordered_slots": 10}
REFERENCE_OFFSET_AXES = ("dt", "dh", "dw")
#: 支持的 DiT 层数：Wan2.1-1.3B 与 Wan2.2-TI2V-5B 30 层，Wan2.2-A14B 每个专家 40 层。
#: 只允许完整窗口（恰好覆盖 0…n−1），前向只核 ``full``、不按层号取舍，所以层号只是声明。
DIT_LAYER_COUNTS = (30, 40)


def _int_triple(value: Any, location: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(type(v) is not int for v in value):
        raise ValueError(f"{location} must be three ints")
    return tuple(value)


@dataclass(frozen=True)
class ReferenceOffsetOptions:
    train: Mapping[str, tuple[int, int]]
    inference: tuple[int, int, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str):
        if not isinstance(value, Mapping) or set(value) != {"train", "inference"}:
            raise ValueError(f"{location} must contain exactly ['train', 'inference']")
        train = value["train"]
        if not isinstance(train, Mapping) or set(train) != set(REFERENCE_OFFSET_AXES):
            raise ValueError(f"{location}.train must contain exactly {list(REFERENCE_OFFSET_AXES)}")
        ranges = []
        for axis in REFERENCE_OFFSET_AXES:  # 与 config 同形（asdict 往返不变）
            bounds = train[axis]
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(type(v) is not int for v in bounds) or bounds[0] > bounds[1]):
                raise ValueError(f"{location}.train.{axis} must be [low, high] ints with low <= high")
            ranges.append(tuple(bounds))
        if ranges[0][1] >= 0 or ranges[1][0] < 0 or ranges[2][0] < 0:
            raise ValueError(f"{location}.train needs dt < 0 and dh, dw >= 0")
        inference = _int_triple(value["inference"], f"{location}.inference")
        if any(not low <= v <= high for v, (low, high) in zip(inference, ranges)):
            raise ValueError(f"{location}.inference must lie inside the training ranges")
        # 偏移是固定值：时间每张间隔 5 个潜帧；空间为 0（参考与目标同网格）或移出网格（11-05 起最终配方取一个网格的
        # 高和宽，见 joint-renderer-design.md）。是否真的清出目标网格由运行时按实际网格大小核（memory_prefix_position_freqs）。
        if any(low != high for low, high in ranges) or ranges[0] != (-5, -5) or inference != tuple(low for low, _ in ranges):
            raise ValueError(f"{location} requires fixed offsets with dt=-5")
        return cls(dict(zip(REFERENCE_OFFSET_AXES, ranges)), inference)

    def draw(self, units: tuple[float, float, float]) -> tuple[int, int, int]:
        """三个 [0,1) 均匀数 → 每轴闭区间上的均匀整数。"""
        if len(units) != 3 or any(not 0 <= u < 1 for u in units):
            raise ValueError("offset draws need three numbers in [0, 1)")
        return tuple(low + int(u * (high - low + 1))
                     for u, (low, high) in zip(units, (self.train[axis] for axis in REFERENCE_OFFSET_AXES)))


@dataclass(frozen=True)
class MemoryAttentionWindow:
    """参考在全部 DiT 层与扩散时间均参与全双向注意力，参考行仅固定角色句。"""

    layer_indices: tuple[int, ...]
    timestep_range: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str):
        if not isinstance(value, Mapping) or set(value) != {"layer_indices", "timestep_range"}:
            raise ValueError(f"{location} must contain exactly ['layer_indices', 'timestep_range']")
        layers, interval = value["layer_indices"], value["timestep_range"]
        if (not isinstance(layers, (list, tuple)) or not layers
                or any(type(i) is not int or not 0 <= i < max(DIT_LAYER_COUNTS) for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError(f"{location}.layer_indices must be unique DiT layer indices in [0,{max(DIT_LAYER_COUNTS)})")
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in interval)
                or not 0 <= interval[0] <= interval[1] <= 1000):
            raise ValueError(f"{location}.timestep_range must be [low, high] scheduler t in [0,1000]")
        result = cls(tuple(layers), tuple(map(float, interval)))
        if not result.full_layers:
            raise ValueError(f"{location} requires all {DIT_LAYER_COUNTS} layers")
        return result

    @property
    def full_layers(self) -> bool:
        return (len(self.layer_indices) in DIT_LAYER_COUNTS
                and set(self.layer_indices) == set(range(len(self.layer_indices))))

    @property
    def full(self) -> bool:
        return self.full_layers and self.timestep_range == (0.0, 1000.0)

    def contains_timestep(self, timestep: float) -> bool:
        return self.timestep_range[0] <= timestep <= self.timestep_range[1]

@dataclass(frozen=True)
class ReferenceNoiseOptions:
    """连续flow σ；Wan实际model t=1000σ，已是shift后的单位。"""

    train_sigma_max: float
    inference_sigma: float

    def __post_init__(self):
        for name in ("train_sigma_max", "inference_sigma"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"memory_prefix.noise.{name} must be a finite flow sigma in [0, 1]")
            object.__setattr__(self, name, float(value))
        if self.inference_sigma > self.train_sigma_max:
            raise ValueError("memory_prefix.noise.inference_sigma must lie inside [0, train_sigma_max]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str):
        if not isinstance(value, Mapping) or set(value) != {"train_sigma_max", "inference_sigma"}:
            raise ValueError(f"{location} must contain exactly ['train_sigma_max', 'inference_sigma']")
        return cls(**value)


@dataclass(frozen=True)
class MemoryPrefixModelOptions:
    frames_max: int
    dropout_p: float
    reference_offset: ReferenceOffsetOptions
    attention_policy: str
    row_policy: str
    attention_window: MemoryAttentionWindow
    placement: str
    noise: ReferenceNoiseOptions

    def __post_init__(self):
        if self.placement not in REFERENCE_PLACEMENTS:
            raise ValueError(f"memory_prefix.placement must be one of {sorted(REFERENCE_PLACEMENTS)}")
        most = REFERENCE_PLACEMENTS[self.placement]
        if type(self.frames_max) is not int or not 1 <= self.frames_max <= most:
            raise ValueError(f"memory_prefix.frames_max must be an int in [1, {most}] for {self.placement}")
        if type(self.dropout_p) not in (int, float) or not math.isfinite(self.dropout_p) or not 0 <= self.dropout_p < 1:
            raise ValueError("memory_prefix.dropout_p must be a number in [0, 1)")
        object.__setattr__(self, "dropout_p", float(self.dropout_p))
        if not isinstance(self.reference_offset, ReferenceOffsetOptions):
            raise ValueError("memory_prefix.reference_offset must be ReferenceOffsetOptions")
        for name, allowed in (("attention_policy", MEMORY_ATTENTION_POLICIES), ("row_policy", MEMORY_ROW_POLICIES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"memory_prefix.{name} must be one of {list(allowed)}")
        if not isinstance(self.attention_window, MemoryAttentionWindow):
            raise ValueError("memory_prefix.attention_window must be MemoryAttentionWindow")
        if self.attention_policy in ROUTED_MEMORY_ATTENTION_POLICIES:
            if not self.attention_window.full_layers:
                raise ValueError("memory_prefix.attention_window requires all layers")
        elif not self.attention_window.full:
            raise ValueError("memory_prefix.attention_window requires the full window")
        if not isinstance(self.noise, ReferenceNoiseOptions):
            raise ValueError("memory_prefix.noise must be ReferenceNoiseOptions")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str = "model.memory_prefix"):
        if not isinstance(value, Mapping) or set(value) != set(MEMORY_PREFIX_OPTION_KEYS):
            raise ValueError(f"{location} must contain exactly {list(MEMORY_PREFIX_OPTION_KEYS)}")
        fields = dict(value)
        fields["reference_offset"] = ReferenceOffsetOptions.from_mapping(
            value["reference_offset"], f"{location}.reference_offset")
        fields["attention_window"] = MemoryAttentionWindow.from_mapping(
            value["attention_window"], f"{location}.attention_window")
        fields["noise"] = ReferenceNoiseOptions.from_mapping(value["noise"], f"{location}.noise")
        return cls(**fields)
