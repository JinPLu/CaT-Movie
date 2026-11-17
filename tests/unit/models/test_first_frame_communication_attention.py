"""首帧通信（``first_frame_communication``）：每镜只有第一个潜帧读外部信息，其余潜帧只读本镜。

可见关系：参考帧只读自己；镜 i 首潜帧读本镜全部＋各镜首潜帧池（本镜首帧在池里重复一份）＋路由给镜 i 的参考
（参考键按镜起点重旋转）；镜 i 其余潜帧只读本镜全部。这里与独立 dense mask 的 SDPA 对拍输出、q/k/v 梯度，并核
「改他镜内容时哪些输出会变」、走 ``model_fn_wan_video`` 的前向反向，以及非法参数组合的拒绝。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from diffsynth.models.wan_video_dit import (  # noqa: E402
    SelfAttention, precompute_freqs_cis_3d, rope_apply,
)
from diffsynth.pipelines.wan_video import memory_prefix_position_freqs  # noqa: E402
from tests.support.memory_prefix import INFERENCE_OFFSET, full_window  # noqa: E402
from tests.unit.models.test_memory_prefix_attention import _tiny_dit  # noqa: E402

HEADS, HEAD_DIM = 2, 12
DIM = HEADS * HEAD_DIM
H, W = 1, 2
TPF = H * W
ATTENTION = SelfAttention(DIM, HEADS).double()


def _split(x):
    return x.reshape(1, x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)


def _masked(q, k, v, mask):
    out = F.scaled_dot_product_attention(_split(q), _split(k), _split(v), attn_mask=mask)
    return out.transpose(1, 2).reshape(1, q.shape[1], DIM)


def _no_reference_dense(q_raw, k_raw, v, groups):
    """无参考时的 dense 对拍：首潜帧读本镜＋各镜首潜帧（本镜首帧重复一次，logit 加 log2），其余潜帧只读本镜。"""
    total = q_raw.shape[1]
    first = [i for a, _ in groups for i in range(a * TPF, (a + 1) * TPF)]
    mask = torch.full((total, total), -torch.inf, dtype=torch.float64)
    for a, b in groups:
        a, b = a * TPF, b * TPF
        mask[a:a + TPF, a:b] = 0
        mask[a:a + TPF, first] = 0
        mask[a:a + TPF, a:a + TPF] = math.log(2)
        if b > a + TPF:
            mask[a + TPF:b, a:b] = 0
    return _masked(q_raw, k_raw, v, mask)


def test_without_references_only_first_latent_frames_touch_other_shots():
    from diffsynth.models.wan_video_dit import SelfAttention

    torch.manual_seed(5)
    attn = SelfAttention(DIM, HEADS).double()
    groups = [(0, 2), (2, 5), (5, 6)]
    q, k, v = [torch.randn(1, 12, DIM, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    actual = attn.routed_shot_attention(q, k, v, groups, TPF, window="first_frame_communication")
    expected = _no_reference_dense(q, k, v, groups)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), (q, k, v), retain_graph=True)
    reference = torch.autograd.grad((expected * probe).sum(), (q, k, v))
    for result, wanted in zip(grads, reference):
        torch.testing.assert_close(result, wanted)


# --- 带参考：照 ``test_routed_reference_attention.py`` 的布局，参考占 2 帧、目标两镜占 3 潜帧 -------------

REFERENCE_FRAMES = 2
GROUPS = ((0, 1), (1, 2), (2, 4), (4, 5))
FRAMES = 5
TOKENS = FRAMES * TPF
TABLE = SimpleNamespace(freqs=precompute_freqs_cis_3d(HEAD_DIM))
FREQS = memory_prefix_position_freqs(TABLE, FRAMES, H, W, REFERENCE_FRAMES, INFERENCE_OFFSET)
SHOTS = ((GROUPS[2][0] * TPF, GROUPS[2][1] * TPF), (GROUPS[3][0] * TPF, GROUPS[3][1] * TPF))
ROUTINGS = (((0, 1), ()), ((), (0,)))


def _frames(*indices):
    return [i for f in indices for i in range(f * TPF, (f + 1) * TPF)]


def _temporal(position):
    table = TABLE.freqs[0]
    return table[position] if position >= 0 else table[-position].conj()


def _references_at(start):
    """把第 n 张参考的时间挪到 start − 5(M−n)，其余 token 不动；与模型里的参考键重旋转等价。"""
    freqs = FREQS.clone()
    width = TABLE.freqs[0].shape[1]
    for n in range(REFERENCE_FRAMES):
        freqs[n * TPF:(n + 1) * TPF, 0, :width] = _temporal(start - 5 * (REFERENCE_FRAMES - n))
    return freqs


def _leaves(seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, TOKENS, DIM, dtype=torch.float64, generator=generator).requires_grad_() for _ in range(3)]


def _dense_with_references(q_raw, k_raw, v, routing):
    """参考只读自己；镜 i 首潜帧读本镜＋各镜首潜帧（本镜首帧重复，logit 加 log2）＋路由参考（按镜起点重旋转）；
    镜 i 其余潜帧只读本镜。"""
    q = rope_apply(q_raw, FREQS, HEADS)
    plain_k = rope_apply(k_raw, FREQS, HEADS)
    pieces = []
    for j in range(REFERENCE_FRAMES):
        mask = torch.full((TPF, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, _frames(j)] = 0
        pieces.append(_masked(q[:, _frames(j)], plain_k, v, mask))
    first = [i for a, _ in SHOTS for i in range(a, a + TPF)]
    for (a, b), references in zip(SHOTS, routing):
        k = rope_apply(k_raw, _references_at(a // TPF - REFERENCE_FRAMES), HEADS)
        mask = torch.full((TPF, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, first] = 0
        mask[:, a:b] = 0
        mask[:, a:a + TPF] = math.log(2)
        mask[:, _frames(*references)] = 0
        pieces.append(_masked(q[:, a:a + TPF], k, v, mask))
        later_mask = torch.full((b - a - TPF, TOKENS), -torch.inf, dtype=torch.float64)
        later_mask[:, a:b] = 0
        pieces.append(_masked(q[:, a + TPF:b], plain_k, v, later_mask))
    return torch.cat(pieces, dim=1)


@pytest.mark.parametrize("routing", ROUTINGS)
def test_with_references_matches_dense_mask_and_gradients(routing):
    q_raw, k_raw, v = leaves = _leaves(7)
    actual = ATTENTION.routed_shot_attention(
        rope_apply(q_raw, FREQS, HEADS), rope_apply(k_raw, FREQS, HEADS), v, GROUPS, TPF,
        window="first_frame_communication", prefix_frames=REFERENCE_FRAMES, routing=routing, freqs=FREQS)
    expected = _dense_with_references(q_raw, k_raw, v, routing)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for result, wanted in zip(grads, reference):
        torch.testing.assert_close(result, wanted)


def _outputs(v, read):
    generator = torch.Generator().manual_seed(3)
    q, k, _ = (torch.randn(1, TOKENS, DIM, dtype=torch.float64, generator=generator) for _ in range(3))
    return ATTENTION.routed_shot_attention(
        q, k, v, GROUPS, TPF, window="first_frame_communication", prefix_frames=REFERENCE_FRAMES,
        routing=ROUTINGS[0], freqs=FREQS)[:, read]


def test_later_latent_frames_do_not_read_other_shots_or_references():
    generator = torch.Generator().manual_seed(11)
    v = torch.randn(1, TOKENS, DIM, dtype=torch.float64, generator=generator)
    moved = v.clone()
    moved[:, _frames(0, 1)] += 3.0  # 改参考与镜 0 之外的参考内容
    moved[:, _frames(4)] += 3.0     # 改镜 1 首潜帧
    # 镜 0 的其余潜帧（第 3 帧）不读参考、不读他镜首帧，因此不变。
    torch.testing.assert_close(_outputs(v, _frames(3)), _outputs(moved, _frames(3)))
    # 镜 0 首潜帧（第 2 帧）读各镜首潜帧，镜 1 首帧变了它就变。
    assert not torch.allclose(_outputs(v, _frames(2)), _outputs(moved, _frames(2)))


def test_forward_branch_uses_the_same_layout():
    module = SelfAttention(DIM, HEADS).double()
    module.self_attention_policy = "first_frame_communication"
    torch.manual_seed(13)
    x = torch.randn(1, TOKENS, DIM, dtype=torch.float64, requires_grad=True)
    actual = module(x, FREQS, latent_shot_groups=GROUPS, tokens_per_frame=TPF,
                    memory_attention_policy="routed_self_only_v1", prefix_frames=REFERENCE_FRAMES,
                    reference_routing=ROUTINGS[0])
    q = rope_apply(module.norm_q(module.q(x)), FREQS, HEADS)
    k = rope_apply(module.norm_k(module.k(x)), FREQS, HEADS)
    v = module.v(x)
    expected = module.o(module.routed_shot_attention(
        q, k, v, GROUPS, TPF, window="first_frame_communication", prefix_frames=REFERENCE_FRAMES,
        routing=ROUTINGS[0], freqs=FREQS))
    torch.testing.assert_close(actual, expected)


def test_illegal_reference_scope_and_memory_policy_are_refused():
    q = k = v = torch.zeros(1, TOKENS, DIM, dtype=torch.float64)
    with pytest.raises(ValueError, match="no separate reference query scope"):
        ATTENTION.routed_shot_attention(
            q, k, v, GROUPS, TPF, window="first_frame_communication", reference_query_scope="all_latents",
            prefix_frames=REFERENCE_FRAMES, routing=ROUTINGS[0], freqs=FREQS)
    module = SelfAttention(DIM, HEADS).double()
    module.self_attention_policy = "first_frame_communication"
    with pytest.raises(ValueError, match="reads references only under one of"):
        module(torch.zeros(1, TOKENS, DIM, dtype=torch.float64), FREQS, latent_shot_groups=GROUPS,
               tokens_per_frame=TPF, memory_attention_policy="bidirectional_v1",
               prefix_frames=REFERENCE_FRAMES, reference_routing=ROUTINGS[0])


# --- 走 model_fn_wan_video 的整条 DiT -----------------------------------------------------------------

DIT_REFERENCES = 2
DIT_GROUPS = ((0, 1), (1, 2), (2, 4), (4, 5))
DIT_ROUTING = ((1,), (0, 1))


def _dit():
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "first_frame_communication"
    return dit


def _model_fn(dit, latents, *, routing=DIT_ROUTING, **extra):
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    return model_fn_wan_video(
        dit=dit, latents=latents, timestep=torch.tensor([500.0]),
        context=torch.randn(len(DIT_GROUPS), 16, 4096, generator=torch.Generator().manual_seed(2)),
        temporal_position_mode="global_time_native_rope_v1", latent_shot_groups=DIT_GROUPS,
        fuse_vae_embedding_in_latents=True, prefix_frames=DIT_REFERENCES, reference_offset=INFERENCE_OFFSET,
        memory_attention_policy="routed_self_only_v1", memory_attention_window=full_window(),
        reference_routing=routing, **extra).noise


def _latents(seed=0):
    return torch.randn(1, 48, 5, 8, 8, generator=torch.Generator().manual_seed(seed))


def test_pipeline_forward_backward_reaches_the_references():
    latents = _latents().requires_grad_()
    result = _model_fn(_dit(), latents, use_gradient_checkpointing=True)
    assert result.shape == latents.shape
    result[:, :, DIT_REFERENCES:].square().mean().backward()
    assert torch.isfinite(latents.grad).all()
    assert latents.grad[:, :, :DIT_REFERENCES].abs().sum() > 0
