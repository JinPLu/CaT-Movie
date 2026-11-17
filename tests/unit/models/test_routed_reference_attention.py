"""HoloCine 稀疏注意力带按镜路由的参考（``routed_self_only_v1``）：与测试里按查询组搭的稠密掩码 SDPA 对拍输出和 q/k/v
梯度；改某帧 V 看哪些输出变；model_fn 的入口校验与 DiT 上的前向反向；DeepSpeed 激活检查点按签名顺序位置传参时每个参数
落在原位。没有参考时的 HoloCine 对拍在 test_holocine_attention.py，走的是同一个 ``routed_shot_attention``。"""
from __future__ import annotations

import inspect
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

from diffsynth.models.wan_video_dit import SelfAttention, precompute_freqs_cis_3d, rope_apply  # noqa: E402
from diffsynth.pipelines.wan_video import memory_prefix_position_freqs  # noqa: E402
from mspc.data.reference_routing import require_reference_routing  # noqa: E402
from tests.support.memory_prefix import INFERENCE_OFFSET, full_window  # noqa: E402
from tests.unit.models.test_memory_prefix_attention import _tiny_dit  # noqa: E402

HEADS, HEAD_DIM = 2, 12
DIM = HEADS * HEAD_DIM
H, W = 1, 2
TPF = H * W
M = 3
# 三张参考（第 0–2 帧），目标三镜：第 3–4、5–7、8 帧（目标潜帧起点 0、2、5）。
GROUPS = ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 9))
FRAMES = 9
TOKENS = FRAMES * TPF
TABLE = SimpleNamespace(freqs=precompute_freqs_cis_3d(HEAD_DIM))
FREQS = memory_prefix_position_freqs(TABLE, FRAMES, H, W, M, INFERENCE_OFFSET)
SHOTS = [(a * TPF, b * TPF) for a, b in GROUPS[M:]]
SHOT_FRAMES = ((3, 4), (5, 6, 7), (8,))
#: 第二组有一镜没有参考。
ROUTINGS = (((0, 2), (1,), (0, 2)), ((0, 1, 2), (), (1,)))
ATTENTION = SelfAttention(DIM, HEADS).double()


def frames(*indices):
    return [i for f in indices for i in range(f * TPF, (f + 1) * TPF)]


def _split(x):
    return x.reshape(1, x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)


def _masked(q, k, v, mask):
    out = F.scaled_dot_product_attention(_split(q), _split(k), _split(v), attn_mask=mask)
    return out.transpose(1, 2).reshape(1, q.shape[1], DIM)


def _leaves(seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, TOKENS, DIM, dtype=torch.float64, generator=generator).requires_grad_() for _ in range(3)]


def _sparse(q, k, v, routing):
    return ATTENTION.routed_shot_attention(q, k, v, GROUPS, TPF, window="first_latents", reference_query_scope="all_latents", prefix_frames=M, routing=routing, freqs=FREQS)


def _temporal(position):
    table = TABLE.freqs[0]
    return table[position] if position >= 0 else table[-position].conj()


def _references_at(start):
    """RoPE 表：第 n 张参考的时间放在 start − 5(M−n)，其余 token 原位、空间不动。"""
    freqs = FREQS.clone()
    width = TABLE.freqs[0].shape[1]
    for n in range(M):
        freqs[n * TPF:(n + 1) * TPF, 0, :width] = _temporal(start - 5 * (M - n))
    return freqs


def _dense(q_raw, k_raw, v, routing):
    """每组查询一张原序列键上的加性掩码；镜 i 的键表直接按 s_i − 5(M−n) 放参考，不复用被测的再旋转。"""
    q = rope_apply(q_raw, FREQS, HEADS)
    firsts = [i for a, _ in SHOTS for i in range(a, a + TPF)]
    pieces = []
    for j in range(M):
        mask = torch.full((TPF, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, frames(j)] = 0
        pieces.append(_masked(q[:, frames(j)], rope_apply(k_raw, FREQS, HEADS), v, mask))
    for (a, b), references in zip(SHOTS, routing):
        k = rope_apply(k_raw, _references_at(a // TPF - M), HEADS)
        mask = torch.full((b - a, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, firsts] = 0
        mask[:, a:b] = 0
        mask[:, a:a + TPF] = math.log(2)  # 本镜首帧重复读取，等价于它的 logit 加 log 2
        mask[:, frames(*references)] = 0
        pieces.append(_masked(q[:, a:b], k, v, mask))
    return torch.cat(pieces, dim=1)


@pytest.mark.parametrize("routing", ROUTINGS)
def test_outputs_and_qkv_gradients_match_a_dense_mask_reference(routing):
    q_raw, k_raw, v = leaves = _leaves(5)
    actual = _sparse(rope_apply(q_raw, FREQS, HEADS), rope_apply(k_raw, FREQS, HEADS), v, routing)
    expected = _dense(q_raw, k_raw, v, routing)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for result, wanted in zip(grads, reference):
        torch.testing.assert_close(result, wanted)


def _dense_full_window(q_raw, k_raw, v, routing):
    """窗内全互读（MultiShotMaster 式）：镜 i 读全部目标 token（不重复首帧）＋路由给它的参考，参考按 s_i − 5(M−n) 放。"""
    q = rope_apply(q_raw, FREQS, HEADS)
    pieces = []
    for j in range(M):
        mask = torch.full((TPF, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, frames(j)] = 0
        pieces.append(_masked(q[:, frames(j)], rope_apply(k_raw, FREQS, HEADS), v, mask))
    for (a, b), references in zip(SHOTS, routing):
        k = rope_apply(k_raw, _references_at(a // TPF - M), HEADS)
        mask = torch.full((b - a, TOKENS), -torch.inf, dtype=torch.float64)
        mask[:, M * TPF:] = 0
        mask[:, frames(*references)] = 0
        pieces.append(_masked(q[:, a:b], k, v, mask))
    return torch.cat(pieces, dim=1)


@pytest.mark.parametrize("routing", ROUTINGS)
def test_full_window_outputs_and_qkv_gradients_match_a_dense_mask_reference(routing):
    q_raw, k_raw, v = leaves = _leaves(7)
    actual = ATTENTION.routed_shot_attention(rope_apply(q_raw, FREQS, HEADS), rope_apply(k_raw, FREQS, HEADS), v,
                                             GROUPS, TPF, window="all_targets", reference_query_scope="all_latents", prefix_frames=M, routing=routing,
                                             freqs=FREQS)
    expected = _dense_full_window(q_raw, k_raw, v, routing)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), leaves)
    reference = torch.autograd.grad((expected * probe).sum(), leaves)
    for result, wanted in zip(grads, reference):
        torch.testing.assert_close(result, wanted)


def test_a_shot_reading_every_reference_sees_them_at_the_per_shot_distance():
    """镜 i 读到全部 M 张参考时，查询（目标时间 s_i+t）对参考 n 的打分只看相对距离 t+5(M−n)，与把这一镜单独放在时间 0、
    参考在 −5(M−n) 的逐镜生成相同。只读到一部分参考时，读到的仍在各自的全局槽位上，不再挤紧。"""
    q_raw, k_raw, _ = [leaf.detach() for leaf in _leaves(9)]
    a, b = SHOTS[1]
    joint_q = rope_apply(q_raw[:, a:b], FREQS[a:b], HEADS)
    joint_k = rope_apply(rope_apply(k_raw[:, :M * TPF], FREQS[:M * TPF], HEADS), FREQS[a:a + 1], HEADS)
    alone = memory_prefix_position_freqs(TABLE, M + (b - a) // TPF, H, W, M, INFERENCE_OFFSET)
    alone_q = rope_apply(q_raw[:, a:b], alone[M * TPF:], HEADS)
    alone_k = rope_apply(k_raw[:, :M * TPF], alone[:M * TPF], HEADS)
    torch.testing.assert_close(_split(joint_q) @ _split(joint_k).transpose(-1, -2),
                               _split(alone_q) @ _split(alone_k).transpose(-1, -2))


def _reads(changed, read, routing=ROUTINGS[0]):
    """改 ``changed`` 这些 token 的 V 后，``read`` 这些输出变没变。"""
    generator = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, TOKENS, DIM, dtype=torch.float64, generator=generator) for _ in range(3))
    moved = v.clone()
    moved[:, changed] += 1.0
    return not torch.equal(_sparse(q, k, v, routing)[:, read], _sparse(q, k, moved, routing)[:, read])


def test_shots_read_their_own_references_and_every_first_frame():
    # ROUTINGS[0]：镜 0 读参考 0、2，镜 1 读参考 1，镜 2 读参考 0、2。
    for shot, references in enumerate(ROUTINGS[0]):
        own = frames(*SHOT_FRAMES[shot])
        for j in range(M):
            assert _reads(frames(j), own) == (j in references)
        for other, (first, *rest) in enumerate(SHOT_FRAMES):
            if other != shot:
                assert _reads(frames(first), own)
                assert not rest or not _reads(frames(*rest), own)


def test_references_read_only_themselves():
    for j in range(M):
        assert not _reads(frames(*[i for i in range(FRAMES) if i != j]), frames(j))
        assert _reads(frames(j), frames(j))


@pytest.mark.parametrize("prefix_frames,routing", [(M, None), (0, ((0,),)), (M, ((0, 1, 2),))])
def test_references_and_routing_come_together_with_one_entry_per_shot(prefix_frames, routing):
    q = k = v = torch.zeros(1, TOKENS, DIM, dtype=torch.float64)
    groups = GROUPS if prefix_frames else [(0, FRAMES)]
    with pytest.raises(ValueError, match="reference_routing for every target shot"):
        ATTENTION.routed_shot_attention(q, k, v, groups, TPF, window="first_latents", reference_query_scope="all_latents", prefix_frames=prefix_frames, routing=routing,
                                        freqs=FREQS)


@pytest.mark.parametrize("routing,message", [
    (((0, 1),), "each of the 2 target shot"),
    (((0,), (0,)), "route every one"),
    (((1, 0), (2,)), "strictly increasing"),
    (((0, 0), (1, 2)), "strictly increasing"),
    (((0, 3), (1, 2)), r"\[0, 3\)"),
    (((True,), (1, 2)), "strictly increasing"),
    ("012", "each of the 2 target shot"),
])
def test_routing_validation_rejects_malformed_routing(routing, message):
    with pytest.raises(ValueError, match=message):
        require_reference_routing(routing, reference_count=3, target_shots=2)


def test_routing_validation_returns_tuples_and_allows_an_empty_shot():
    assert require_reference_routing([[0, 2], [], [1]], reference_count=3, target_shots=3) == ((0, 2), (), (1,))


# --- 走 model_fn_wan_video 的整条 DiT ---------------------------------------------------------------

DIT_REFERENCES = 2
DIT_GROUPS = ((0, 1), (1, 2), (2, 4), (4, 5))
DIT_ROUTING = ((1,), (0, 1))


def _dit(policy="holocine_first_frame", *, layers=None):
    dit = _tiny_dit()
    if layers is not None:
        dit.blocks = torch.nn.ModuleList(list(dit.blocks)[:layers])
    for block in dit.blocks:
        block.self_attn.self_attention_policy = policy
    return dit


def _model_fn(dit, latents, *, memory_policy="routed_self_only_v1", routing=DIT_ROUTING,
              mode="global_time_native_rope_v1", **extra):
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    return model_fn_wan_video(
        dit=dit, latents=latents, timestep=torch.tensor([500.0]),
        context=torch.randn(len(DIT_GROUPS), 16, 4096, generator=torch.Generator().manual_seed(2)),
        temporal_position_mode=mode, latent_shot_groups=DIT_GROUPS, fuse_vae_embedding_in_latents=True,
        prefix_frames=DIT_REFERENCES, reference_offset=INFERENCE_OFFSET, memory_attention_policy=memory_policy,
        memory_attention_window=full_window(), reference_routing=routing, **extra).noise


def _latents(seed=0):
    return torch.randn(1, 48, 5, 8, 8, generator=torch.Generator().manual_seed(seed))


def test_pipeline_forward_backward_reaches_the_references():
    latents = _latents().requires_grad_()
    result = _model_fn(_dit(), latents, use_gradient_checkpointing=True)
    assert result.shape == latents.shape
    result[:, :, DIT_REFERENCES:].square().mean().backward()
    assert torch.isfinite(latents.grad).all()
    assert latents.grad[:, :, :DIT_REFERENCES].abs().sum() > 0


def test_one_dit_block_shot_is_untouched_by_a_reference_not_routed_to_it():
    """一层 DiT：patch、时间调制与 FFN 都按 token，镜 0 的输出只经自注意力读参考；参考 0 只路由给镜 1。"""
    dit = _dit(layers=1)
    latents = _latents(1)
    moved = latents.clone()
    moved[:, :, 0] += 3.0
    with torch.no_grad():
        base, changed = _model_fn(dit, latents), _model_fn(dit, moved)
    assert torch.equal(base[:, :, 2:4], changed[:, :, 2:4])
    assert not torch.allclose(base[:, :, 4], changed[:, :, 4])


def test_full_window_with_routed_references_and_shot_phase_runs_forward_and_backward():
    """MultiShotMaster 式消融：目标全互读＋镜相位，参考的路由与读法与 HoloCine 组相同；没路由给镜 0 的参考 0 不影响镜 0。"""
    latents = _latents().requires_grad_()
    result = _model_fn(_dit("full"), latents, mode="global_time_msm_fixed_shot_phase_v1")
    result[:, :, DIT_REFERENCES:].square().mean().backward()
    assert torch.isfinite(latents.grad).all() and latents.grad[:, :, :DIT_REFERENCES].abs().sum() > 0
    dit, base = _dit("full", layers=1), _latents(1)
    moved = base.clone()
    moved[:, :, 0] += 3.0
    with torch.no_grad():
        before = _model_fn(dit, base, mode="global_time_msm_fixed_shot_phase_v1")
        after = _model_fn(dit, moved, mode="global_time_msm_fixed_shot_phase_v1")
    assert torch.equal(before[:, :, 2:4], after[:, :, 2:4])
    assert not torch.allclose(before[:, :, 4], after[:, :, 4])


@pytest.mark.parametrize("self_policy,memory_policy,routing,mode,message", [
    ("full", "bidirectional_v1", DIT_ROUTING, "global_time_native_rope_v1", "travels with memory_attention_policy"),
    ("holocine_first_frame", "routed_self_only_v1", None, "global_time_native_rope_v1",
     "travels with memory_attention_policy"),
    ("holocine_first_frame", "bidirectional_v1", None, "global_time_native_rope_v1",
     "reads references only under one of"),
    ("holocine_first_frame", "routed_self_only_v1", ((1,), (1,)), "global_time_native_rope_v1", "route every one"),
    ("holocine_first_frame", "routed_self_only_v1", DIT_ROUTING, "global_time_msm_fixed_shot_phase_v1",
     "native temporal positions"),
])
def test_model_fn_refuses_routing_that_does_not_match_the_declared_policies(self_policy, memory_policy, routing,
                                                                           mode, message):
    with pytest.raises(ValueError, match=message), torch.no_grad():
        _model_fn(_dit(self_policy), _latents(), memory_policy=memory_policy, routing=routing, mode=mode)


def test_deepspeed_positional_checkpoint_binds_every_block_argument_in_place(monkeypatch):
    """DeepSpeed 的检查点把 kwargs 的值按出现顺序位置传给 DiTBlock.forward；每个参数都要落在签名里同名的位置。"""
    from diffsynth.core.gradient import gradient_checkpoint
    from diffsynth.models.wan_video_dit import DiTBlock

    signature = inspect.signature(DiTBlock.forward)
    calls = []

    def record(module, args, kwargs):
        bound = signature.bind(module, *args, **kwargs)
        bound.apply_defaults()
        calls.append(bound.arguments)

    dit = _dit()
    for block in dit.blocks:
        block.register_forward_pre_hook(record, with_kwargs=True)
    latents = _latents(2).requires_grad_()
    plain = _model_fn(dit, latents, use_gradient_checkpointing=False)
    fake = SimpleNamespace(checkpointing=SimpleNamespace(is_configured=lambda: True,
                                                         checkpoint=lambda function, *args: function(*args)))
    monkeypatch.setattr(gradient_checkpoint, "_HAS_DEEPSPEED", True)
    monkeypatch.setattr(gradient_checkpoint, "deepspeed", fake, raising=False)
    positional = _model_fn(dit, latents, use_gradient_checkpointing=True)
    torch.testing.assert_close(positional, plain, rtol=0, atol=0)
    layers = len(dit.blocks)
    assert len(calls) == 2 * layers
    for by_name, by_position in zip(calls[:layers], calls[layers:]):
        assert by_name.keys() == by_position.keys()
        for name, value in by_name.items():
            other = by_position[name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other, name
        assert by_position["reference_routing"] == DIT_ROUTING
        assert by_position["planner_layout"] is None and by_position["source_identity_condition"] is None
