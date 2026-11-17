"""每潜帧一行的时间调制按帧广播，与每个 token 一行逐位相同（含排在视频 token 之后、各自一行的附加 token）。"""
import sys
from pathlib import Path

import pytest
import torch

_RUNTIME = Path(__file__).resolve().parents[3] / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from diffsynth.models.wan_video_dit import GateModule, modulate  # noqa: E402


def _per_token(rows: torch.Tensor, frames: int, per_frame: int) -> torch.Tensor:
    return torch.cat([rows[:, :frames].repeat_interleave(per_frame, dim=1), rows[:, frames:]], dim=1)


@pytest.mark.parametrize("extra", [0, 3])
def test_frame_rows_broadcast_is_bitwise_the_per_token_result(extra):
    generator = torch.Generator().manual_seed(0)
    frames, per_frame, dim = 5, 7, 16
    video = frames * per_frame
    x = torch.randn(1, video + extra, dim, generator=generator)
    residual = torch.randn(1, video + extra, dim, generator=generator).bfloat16()
    shift, scale, gate = (torch.randn(1, frames + extra, dim, generator=generator) for _ in range(3))

    assert torch.equal(modulate(x, shift, scale, video),
                       modulate(x, _per_token(shift, frames, per_frame), _per_token(scale, frames, per_frame)))
    assert torch.equal(GateModule()(x, gate, residual, video),
                       GateModule()(x, _per_token(gate, frames, per_frame), residual))


def test_rows_that_do_not_tile_the_video_tokens_raise():
    with pytest.raises(ValueError, match="do not tile"):
        modulate(torch.zeros(1, 10, 4), torch.zeros(1, 3, 4), torch.zeros(1, 3, 4), 10)
