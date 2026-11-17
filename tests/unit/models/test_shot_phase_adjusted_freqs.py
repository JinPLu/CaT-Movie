"""固定镜相位与直接坐标公式对拍；参考前缀不改变目标的位置。"""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d, shot_phase_adjusted_freqs
from diffsynth.pipelines.wan_video import memory_prefix_position_freqs, shot_phase_index_sequence
from types import SimpleNamespace


def _tables():
    return SimpleNamespace(freqs=precompute_freqs_cis_3d(128))


def _expected_time(positions):
    # Wan head_dim 128 allocates 44 real temporal channels (22 complex).
    inverse = 1 / (10000 ** (torch.arange(0, 44, 2).double() / 44))
    angles = positions.double()[:, None] * inverse[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def test_fixed_phase_matches_direct_time_coordinates():
    groups = [(0, 2), (2, 5), (5, 7)]
    base = memory_prefix_position_freqs(_tables(), 7, 2, 3, 0, None)
    actual = shot_phase_adjusted_freqs(base, groups, 7, 6, shot_phase_index=(0, 1, 2))
    times = torch.tensor([0, 1, 2.5, 3.5, 4.5, 6, 7]).repeat_interleave(6)
    torch.testing.assert_close(actual[:, 0, :22], _expected_time(times), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[:, :, 22:], base[:, :, 22:], rtol=0, atol=0)


def test_references_keep_own_positions_and_targets_keep_unprefixed_phase():
    dit = _tables()
    groups = [(0, 1), (1, 2), (2, 4), (4, 7)]
    shifted = memory_prefix_position_freqs(dit, 7, 2, 3, 2, (-5, 0, 0))
    indices = shot_phase_index_sequence(groups, 2, (-5, 0, 0))
    actual = shot_phase_adjusted_freqs(shifted, groups, 7, 6, shot_phase_index=indices)
    base = memory_prefix_position_freqs(dit, 5, 2, 3, 0, None)
    target = shot_phase_adjusted_freqs(base, [(0, 2), (2, 5)], 5, 6, shot_phase_index=(0, 1))
    torch.testing.assert_close(actual[:12], shifted[:12], rtol=0, atol=0)
    torch.testing.assert_close(actual[12:], target, rtol=0, atol=0)


def test_fixed_phase_rejects_misaligned_token_layout():
    freqs = memory_prefix_position_freqs(_tables(), 4, 2, 3, 0, None)
    with pytest.raises(ValueError, match="token count"):
        shot_phase_adjusted_freqs(freqs[:-1], [(0, 2), (2, 4)], 4, 6, shot_phase_index=(0, 1))
    with pytest.raises(ValueError, match="one index"):
        shot_phase_adjusted_freqs(freqs, [(0, 2), (2, 4)], 4, 6, shot_phase_index=(0,))
