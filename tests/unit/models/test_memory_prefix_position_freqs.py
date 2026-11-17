"""源时间升序的负槽、原空间网格、目标MSM相位与文字路由合同。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d, shot_phase_adjusted_freqs
from diffsynth.pipelines.wan_video import (
    MEMORY_REFERENCE_TEMPORAL_MODE as RUNTIME_MODE, memory_prefix_position_freqs,
    shot_phase_index_sequence, latent_shot_groups_to_token_ranges,
)
from mspc.core.wan22_contract import MEMORY_REFERENCE_TEMPORAL_MODE

MODE = "global_time_msm_fixed_shot_phase_v1"
POLICY = (-5, 0, 0)  # 固定负时间间隔，原空间网格
H, W = 15, 26


def _dit():
    return SimpleNamespace(freqs=precompute_freqs_cis_3d(128))


def _native(dit, f):
    return torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, H, W, -1),
        dit.freqs[1][:H].view(1, H, 1, -1).expand(f, H, W, -1),
        dit.freqs[2][:W].view(1, 1, W, -1).expand(f, H, W, -1),
    ], dim=-1).reshape(f * H * W, 1, -1)


def test_contract_and_runtime_name_one_reference_temporal_mode():
    assert MEMORY_REFERENCE_TEMPORAL_MODE == RUNTIME_MODE == MODE


def test_no_reference_coordinates_phases_and_text_routes_preserve_original_bytes():
    dit = _dit()
    groups = ((0, 2), (2, 5), (5, 7))
    table = memory_prefix_position_freqs(dit, 7, H, W, 0, None)
    assert torch.equal(table.view(torch.uint8), _native(dit, 7).view(torch.uint8))
    assert shot_phase_index_sequence(groups, 0, None) == tuple(range(len(groups)))
    assert latent_shot_groups_to_token_ranges(groups, 7, H, W) == tuple(
        (a * H * W, b * H * W) for a, b in groups
    )


@pytest.mark.parametrize("references", [1, 2, 4, 10])
def test_references_use_increasing_negative_time_slots_and_native_spatial_grid(references):
    dit = _dit()
    table = memory_prefix_position_freqs(dit, references + 7, H, W, references, POLICY)
    grid = table.reshape(references + 7, H, W, -1)
    temporal_dim, height_dim = dit.freqs[0].shape[-1], dit.freqs[1].shape[-1]
    for n in range(references):
        assert torch.equal(grid[n, 0, 0, :temporal_dim], dit.freqs[0][5 * (references - n)].conj())
        for row in (0, H-1):
            assert torch.equal(grid[n, row, 0, temporal_dim:temporal_dim + height_dim], dit.freqs[1][row])
        for col in (0, W-1):
            assert torch.equal(grid[n, 0, col, temporal_dim + height_dim:], dit.freqs[2][col])
    assert torch.equal(table[references * H * W:], _native(dit, 7))


@pytest.mark.parametrize("references", [1, 2, 4, 10])
def test_reference_prefix_does_not_change_target_shot_phases_or_routes(references):
    dit = _dit()
    target_groups = ((0, 2), (2, 5), (5, 7))
    groups = tuple((n, n+1) for n in range(references)) + tuple((a+references,b+references) for a,b in target_groups)
    phases = shot_phase_index_sequence(groups, references, POLICY)
    assert phases == (0,) * references + (0,1,2)
    table = memory_prefix_position_freqs(dit, references + 7, H, W, references, POLICY)
    moved = shot_phase_adjusted_freqs(table, groups, references+7, H*W, shot_phase_index=phases)
    native = shot_phase_adjusted_freqs(_native(dit,7), target_groups,7,H*W,shot_phase_index=(0,1,2))
    assert torch.equal(moved[references*H*W:].view(torch.uint8), native.view(torch.uint8))
    routes = latent_shot_groups_to_token_ranges(groups,references+7,H,W)
    native_routes = latent_shot_groups_to_token_ranges(target_groups,7,H,W)
    assert tuple((a-references*H*W,b-references*H*W) for a,b in routes[references:]) == native_routes


@pytest.mark.parametrize("offset", [(-3, 0, 0), (-5, 0, 0), (-1, 0, 0)])
def test_reference_order_sets_time_without_spatial_displacement(offset):
    dt, dh, dw = offset
    dit = _dit()
    references = 3
    table = memory_prefix_position_freqs(dit, references + 7, H, W, references, offset)
    grid = table.reshape(references + 7, H, W, -1)
    temporal_dim, height_dim = dit.freqs[0].shape[-1], dit.freqs[1].shape[-1]
    for n in range(references):
        assert torch.equal(grid[n, 0, 0, :temporal_dim], dit.freqs[0][-dt * (references - n)].conj())
        assert torch.equal(grid[n, H - 1, 0, temporal_dim:temporal_dim + height_dim], dit.freqs[1][H - 1 + dh])
        assert torch.equal(grid[n, 0, W - 1, temporal_dim + height_dim:], dit.freqs[2][W - 1 + dw])
    assert torch.equal(table[references * H * W:], _native(dit, 7))


def test_off_grid_references_sit_one_grid_below_and_right_of_the_target():
    # 最终配方：参考整块移出目标网格，偏移一个网格的高和宽；时间槽不变，目标坐标不变。
    dit, references = _dit(), 2
    table = memory_prefix_position_freqs(dit, references + 7, H, W, references, (-5, H, W))
    grid = table.reshape(references + 7, H, W, -1)
    temporal_dim, height_dim = dit.freqs[0].shape[-1], dit.freqs[1].shape[-1]
    for n in range(references):
        assert torch.equal(grid[n, 0, 0, :temporal_dim], dit.freqs[0][5 * (references - n)].conj())
        assert torch.equal(grid[n, 0, 0, temporal_dim:temporal_dim + height_dim], dit.freqs[1][H])
        assert torch.equal(grid[n, H - 1, W - 1, temporal_dim + height_dim:], dit.freqs[2][2 * W - 1])
    assert torch.equal(table[references * H * W:], _native(dit, 7))


@pytest.mark.parametrize("offset", [(-5, H - 1, W), (-5, H, W - 1), (-5, 8, 0)])
def test_a_spatial_offset_must_clear_the_target_grid(offset):
    with pytest.raises(ValueError, match="clear the 15x26 target grid"):
        memory_prefix_position_freqs(_dit(), 3, H, W, 1, offset)


@pytest.mark.parametrize("bad", ["stand_in_v1", (0, 15, 26), (-1, -1, 26), [-1, 15, 26], (-1, 15)])
def test_policy_names_and_malformed_offsets_are_not_accepted(bad):
    with pytest.raises(ValueError, match="reference_offset must be a tuple"):
        memory_prefix_position_freqs(_dit(),3,H,W,1,bad)


def test_rope_table_end_1024_bounds_the_reference_slots():
    dit = _dit()
    assert dit.freqs[1].shape[0] == dit.freqs[2].shape[0] == dit.freqs[0].shape[0] == 1024
    memory_prefix_position_freqs(dit, 12, H, W, 10, POLICY)
    with pytest.raises(ValueError, match="exceed the RoPE table"):
        memory_prefix_position_freqs(dit, 6, H, W, 4, (-256, 0, 0))
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        memory_prefix_position_freqs(dit, 13, H, W, 11, POLICY)


def test_reference_groups_are_single_frame_and_leave_target():
    with pytest.raises(ValueError, match="each reference"):
        shot_phase_index_sequence(((0,2),(2,5)),2,POLICY)
    with pytest.raises(ValueError, match="each reference"):
        shot_phase_index_sequence(((0,1),(1,2)),2,POLICY)


def test_native_ti2v_clean_first_frame_keeps_native_coordinates():
    dit = _dit()
    assert torch.equal(memory_prefix_position_freqs(dit,3,H,W,1,None),_native(dit,3))
