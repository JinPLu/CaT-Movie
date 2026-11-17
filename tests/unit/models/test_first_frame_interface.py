"""镜首帧接口的可见关系（单层注意力上验）：其余帧只读本镜；他镜其余帧不影响本镜任何输出；镜首帧读本镜、他镜首帧与
全部参考；参考只读自己。另在 DiT 上验：整段连续原生时间下能前向，配了镜相位就拒绝。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from mspc.models.first_frame_interface import first_frame_interface_attention  # noqa: E402
from tests.support.memory_prefix import full_window  # noqa: E402
from tests.unit.models.test_memory_prefix_attention import _tiny_dit  # noqa: E402

TPF = 4
# 两张参考（第 0、1 帧），目标两镜：第 2–3 帧、第 4–6 帧。
GROUPS = ((0, 1), (1, 2), (2, 4), (4, 7))
TOTAL = 7 * TPF


def frames(*indices):
    return [i for f in indices for i in range(f * TPF, (f + 1) * TPF)]


def _attn(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _run(v):
    torch.manual_seed(0)
    q, k = torch.randn(1, TOTAL, 8), torch.randn(1, TOTAL, 8)
    return first_frame_interface_attention(_attn, q, k, v, groups=GROUPS, prefix_frames=2, tokens_per_frame=TPF)


def _changed(v, positions):
    out = v.clone()
    out[:, list(positions)] += 1.0
    return out


@pytest.fixture
def base():
    v = torch.randn(1, TOTAL, 8, generator=torch.Generator().manual_seed(1))
    return v, _run(v)


def _same(v, out, changed, read):
    return torch.equal(_run(_changed(v, changed))[:, read], out[:, read])


def test_rest_frames_read_only_their_own_shot(base):
    v, out = base
    rest = frames(3)
    for elsewhere in (frames(0, 1), frames(4, 5, 6)):
        assert _same(v, out, elsewhere, rest)
    assert not _same(v, out, frames(2), rest)


def test_rest_frames_of_another_shot_touch_nothing_in_this_shot(base):
    v, out = base
    assert _same(v, out, frames(5, 6), frames(0, 1, 2, 3))


def test_first_frames_read_every_reference_and_the_other_first_frames(base):
    v, out = base
    for first in (frames(2), frames(4)):
        assert not _same(v, out, frames(0), first)
        assert not _same(v, out, frames(1), first)
    assert not _same(v, out, frames(4), frames(2))   # 他镜首帧
    assert not _same(v, out, frames(2), frames(4))


def test_references_read_only_themselves(base):
    v, out = base
    assert _same(v, out, frames(1), frames(0))
    assert _same(v, out, frames(2, 3, 4, 5, 6), frames(0, 1))


def _forward(dit, *, mode="global_time_native_rope_v1"):
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    torch.manual_seed(1)
    latents = torch.randn(1, 48, 5, 30, 52, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return model_fn_wan_video(
            dit=dit, latents=latents, timestep=torch.tensor([500.0]), context=torch.randn(4, 16, 4096),
            temporal_position_mode=mode, latent_shot_groups=((0, 1), (1, 2), (2, 4), (4, 5)),
            fuse_vae_embedding_in_latents=True, prefix_frames=2, reference_offset=(-5, 0, 0),
            memory_attention_policy="bidirectional_v1", memory_attention_window=full_window(),
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False).noise


def _single_shot(dit):
    """单镜、没有参考：数据不给分组。"""
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    torch.manual_seed(1)
    latents = torch.randn(1, 48, 3, 30, 52, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return model_fn_wan_video(
            dit=dit, latents=latents, timestep=torch.tensor([500.0]), context=torch.randn(1, 16, 4096),
            temporal_position_mode="global_time_native_rope_v1", latent_shot_groups=None,
            fuse_vae_embedding_in_latents=True, prefix_frames=0, reference_offset=None,
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False).noise


def test_a_single_shot_without_references_is_plain_attention():
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "first_frame_interface"
    assert torch.allclose(_single_shot(dit), _single_shot(_tiny_dit()), atol=1e-5)


def test_dit_refuses_the_old_reference_interface_under_the_bidirectional_contract():
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "first_frame_interface"
    with pytest.raises(ValueError, match="does not implement bidirectional reference attention"):
        _forward(dit)
    with pytest.raises(ValueError, match="global_time_native_rope_v1"):
        _forward(dit, mode="global_time_msm_fixed_shot_phase_v1")
