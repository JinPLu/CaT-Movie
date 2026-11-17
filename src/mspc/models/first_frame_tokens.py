"""首帧组附加 token（DiT 属性 ``first_frame_tokens``）：首帧通信下每个目标镜在视频 token 之后追加一组可学 token，
与本镜第一个潜帧合成**首帧组**，承接首帧之外的跨镜信息；不解码、不计损失。只配
``self_attention_policy="first_frame_communication"``。两种做法，二选一：

- ``model.first_frame_twin``（首帧孪生帧）：每镜一整帧，``tokens_per_frame×dim`` 的可学表，每格一行、各镜共用；
  每格 RoPE 与本镜首潜帧同一格相同（时间＝镜起点）。表的行数就是训练分辨率下一帧的 token 数，分辨率不符即报错。
- ``model.shot_registers``（镜 register）：每镜 ``count`` 个，``count×dim`` 的可学表、各镜共用（ViT register 做法：
  DINOv2 官方实现是 ``nn.Parameter(torch.zeros(1, N, D))`` 的 N 行独立向量）；``count`` 个都取本镜首潜帧中心格那一行
  RoPE，彼此只靠各自的可学值区分。

动机：首帧通信下只有各镜首潜帧互读，但多镜 T2V 的首帧还要算损失，因此加类似 ViT register token /
MemLearner 的附加 token 承接首帧额外的信息。11-11 复查发现第一版孪生帧只有一个 1×dim 向量铺满整帧，
少于每帧 token 数 × dim 的预期；register 一类做法在视频生成里是否有效，在 Wan2.1 上用消融验证。

可见关系（序列是 [参考帧][目标各镜][镜0 附加 token]…[镜S-1 附加 token]）：参考帧只读自己；镜 i 的首帧组（第一个潜帧＋
本镜附加 token）读本镜全部潜帧＋本镜附加 token＋各镜首帧组池（本镜首帧组按 HoloCine 在池里重复一份）＋路由给镜 i 的
参考；镜 i 其余潜帧读本镜全部潜帧＋本镜附加 token。两种做法都读本镜那一行文字、与目标 token 同一调度器时间步调制、
FP32 存储、初值 N(0, 0.02²)（在独立 RNG 里抽，不改训练噪声与数据抽签），出 head 前去掉。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

FIRST_FRAME_TOKENS = "first_frame_tokens"
#: 可学表的初值标准差（MAE mask token、Perceiver latent 的常用值）。
FIRST_FRAME_TOKENS_INIT_STD = 0.02


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{location} must be a positive int")
    return value


def _learning_rate(value: Any, location: str) -> float:
    if type(value) is not float or not value > 0:
        raise ValueError(f"{location} must be a positive float")
    return value


@dataclass(frozen=True)
class FirstFrameTwinOptions:
    learning_rate: float
    tokens_per_frame: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any],
                     location: str = "model.first_frame_twin") -> "FirstFrameTwinOptions":
        if not isinstance(value, Mapping) or set(value) != {"learning_rate", "tokens_per_frame"}:
            raise ValueError(f"{location} requires exactly ['learning_rate', 'tokens_per_frame']")
        return cls(learning_rate=_learning_rate(value["learning_rate"], f"{location}.learning_rate"),
                   tokens_per_frame=_positive_int(value["tokens_per_frame"], f"{location}.tokens_per_frame"))


@dataclass(frozen=True)
class ShotRegisterOptions:
    count: int
    learning_rate: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any],
                     location: str = "model.shot_registers") -> "ShotRegisterOptions":
        if not isinstance(value, Mapping) or set(value) != {"count", "learning_rate"}:
            raise ValueError(f"{location} requires exactly ['count', 'learning_rate']")
        return cls(count=_positive_int(value["count"], f"{location}.count"),
                   learning_rate=_learning_rate(value["learning_rate"], f"{location}.learning_rate"))


class _TokenTable(nn.Module):
    def __init__(self, rows: int, dim: int):
        super().__init__()
        # 新参数的随机初值不能改变对照组的训练 RNG 序列（目标噪声、参考加噪、数据抽签）。
        with torch.random.fork_rng(devices=[]):
            self.table = nn.Parameter(torch.randn(rows, dim) * FIRST_FRAME_TOKENS_INIT_STD)

    def states(self, shots: int, dtype: torch.dtype) -> torch.Tensor:
        """[1, shots·rows, dim]：镜 0 的整张表、镜 1 的整张表……"""
        if type(shots) is not int or shots < 1:
            raise ValueError("first-frame tokens need at least one target shot")
        return self.table.to(dtype).unsqueeze(0).repeat(1, shots, 1)


class FirstFrameTwin(_TokenTable):
    """每镜一整帧，每格一行可学值。"""

    def per_shot(self, height: int, width: int) -> int:
        if height * width != self.table.shape[0]:
            raise ValueError(f"the first-frame twin table has {self.table.shape[0]} cells; this video frame has "
                             f"{height}x{width}={height * width}")
        return self.table.shape[0]

    def freqs(self, block_freqs: torch.Tensor, *, starts: Sequence[int], height: int, width: int) -> torch.Tensor:
        """逐镜取本镜第一个潜帧那 ``height·width`` 行 RoPE，逐格相同。"""
        cells = self.per_shot(height, width)
        return _rows(block_freqs, [start * cells + cell for start in starts for cell in range(cells)])


class ShotRegisters(_TokenTable):
    """每镜 ``count`` 个 register，各自一行可学值、同一个位置。"""

    def per_shot(self, height: int, width: int) -> int:
        return self.table.shape[0]

    def freqs(self, block_freqs: torch.Tensor, *, starts: Sequence[int], height: int, width: int) -> torch.Tensor:
        """每镜 ``count`` 行：都取本镜第一个潜帧中心格 token 那一行 RoPE。"""
        center = (height // 2) * width + width // 2
        return _rows(block_freqs, [start * height * width + center for start in starts
                                   for _ in range(self.table.shape[0])])


def _rows(block_freqs: torch.Tensor, rows: list[int]) -> torch.Tensor:
    if not rows or max(rows) >= block_freqs.shape[0]:
        raise ValueError("first-frame token positions fall outside the video RoPE rows")
    return block_freqs[torch.tensor(rows, device=block_freqs.device)]


def first_frame_token_options(model_cfg: Mapping[str, Any]) -> FirstFrameTwinOptions | ShotRegisterOptions | None:
    """模型配置里的首帧组附加 token；两种都配就报错。"""
    twin, registers = model_cfg.get("first_frame_twin"), model_cfg.get("shot_registers")
    if twin is not None and registers is not None:
        raise ValueError("model.first_frame_twin and model.shot_registers are separate experiments")
    if twin is not None:
        return FirstFrameTwinOptions.from_mapping(twin)
    return None if registers is None else ShotRegisterOptions.from_mapping(registers)


def attach_first_frame_tokens(dit: nn.Module, options: FirstFrameTwinOptions | ShotRegisterOptions | None) -> None:
    """配置了才挂；没配置时 DiT 结构不变。FP32，建在 DiT 所在设备上，进块前转成视频 token 的 dtype。"""
    if options is None:
        return
    device = next(dit.parameters()).device
    module = (FirstFrameTwin(options.tokens_per_frame, dit.dim) if isinstance(options, FirstFrameTwinOptions)
              else ShotRegisters(options.count, dit.dim))
    setattr(dit, FIRST_FRAME_TOKENS, module.to(device=device))


def target_shot_starts(groups: Sequence[Sequence[int]], prefix_frames: int) -> tuple[int, ...]:
    """目标各镜第一个潜帧在整段（含参考前缀）里的下标。"""
    if len(groups) <= prefix_frames or any(tuple(groups[j]) != (j, j + 1) for j in range(prefix_frames)):
        raise ValueError("first-frame tokens need one latent group per reference, then the target shots")
    return tuple(int(start) for start, _ in groups[prefix_frames:])
