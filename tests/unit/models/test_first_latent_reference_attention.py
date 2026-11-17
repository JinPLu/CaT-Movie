"""只在每镜首潜帧直接读参考：独立稠密数学参照、导数局部性与退化情形。

Oracle在原token键空间上构造逐query掩码，不复用按镜拼接代码：重复的本镜首潜帧用
log(2)表示；参考K按每个query所属镜的绝对时间坐标直接旋转，不复用生产的二次RoPE。
导数的零范围只针对单次自注意力，不能外推到多层后的间接信息传播。
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))

from diffsynth.models.wan_video_dit import SelfAttention, rope_apply  # noqa: E402


HEADS, HEAD_DIM = 2, 12
DIM = HEADS * HEAD_DIM
HEIGHT, WIDTH = 2, 3
TOKENS_PER_FRAME = HEIGHT * WIDTH
FIRST_LATENT_POLICY = "routed_first_latent_self_only_v1"


@pytest.fixture(autouse=True)
def cpu_math_attention_backend(monkeypatch):
    from diffsynth.models import wan_video_dit

    for name in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(wan_video_dit, name, False)


def _case(lengths=(2, 3, 1), references=3):
    groups = [(j, j + 1) for j in range(references)]
    end = references
    for length in lengths:
        groups.append((end, end + length))
        end += length
    return tuple(groups), end * TOKENS_PER_FRAME


def _angles(groups, references, *, reference_start=0):
    """直接从(t,h,w)坐标求各旋转对角度，不读生产的时间/参考位置辅助函数。"""
    total_frames = groups[-1][1]
    times = [-5 * (references - j) + reference_start for j in range(references)]
    times += list(range(total_frames - references))
    axis_dims = (HEAD_DIM - 2 * (HEAD_DIM // 3), HEAD_DIM // 3, HEAD_DIM // 3)
    omega = [10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d) for d in axis_dims]
    rows = []
    for time in times:
        for y in range(HEIGHT):
            for x in range(WIDTH):
                rows.append(torch.cat([time * omega[0], y * omega[1], x * omega[2]]))
    return torch.stack(rows)


def _rotate(value, angles):
    """实数二维旋转公式；与生产rope_apply的复数乘法独立。"""
    pairs = value.reshape(value.shape[0], value.shape[1], HEADS, HEAD_DIM // 2, 2)
    cosine, sine = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    first, second = pairs[..., 0], pairs[..., 1]
    return torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1).reshape_as(value)


def _heads(value):
    return value.reshape(value.shape[0], value.shape[1], HEADS, HEAD_DIM).transpose(1, 2)


def _leaves(tokens, seed=7, batch=2):
    rng = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, tokens, DIM, generator=rng, dtype=torch.float64).requires_grad_()
                 for _ in range(3))


def _dense_oracle(q_raw, k_raw, v, *, groups, references, routing, scope):
    """每个query对原序列全部K显式打分，再用语义允许关系与重数作softmax。"""
    n, tokens = TOKENS_PER_FRAME, q_raw.shape[1]
    q = _rotate(q_raw, _angles(groups, references).to(q_raw.device))
    target_groups = groups[references:]
    first_indices = [token for start, _ in target_groups for token in range(start * n, (start + 1) * n)]
    outputs = []
    for query in range(tokens):
        frame = query // n
        mask = torch.full((tokens,), -torch.inf, dtype=torch.float64, device=q_raw.device)
        if frame < references:
            mask[frame * n:(frame + 1) * n] = 0
            angles = _angles(groups, references).to(q_raw.device)
        else:
            shot = next(i for i, (start, stop) in enumerate(target_groups) if start <= frame < stop)
            start, stop = target_groups[shot]
            mask[first_indices] = 0
            mask[start * n:stop * n] = 0
            # 两份完全相同的本镜首帧K/V，在原K空间等价于其logit加log(2)。
            mask[start * n:(start + 1) * n] = math.log(2)
            if references and (scope == "all_latents" or frame == start):
                for reference in routing[shot]:
                    mask[reference * n:(reference + 1) * n] = 0
            angles = _angles(groups, references, reference_start=start - references).to(q_raw.device)
        k = _rotate(k_raw, angles)
        score = (_heads(q[:, query:query + 1]) @ _heads(k).transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        probability = torch.softmax(score + mask, dim=-1)
        output = probability @ _heads(v)
        outputs.append(output.transpose(1, 2).reshape(q_raw.shape[0], 1, DIM))
    return torch.cat(outputs, dim=1)


def _sparse(module, q_raw, k_raw, v, *, groups, references, routing, scope):
    angles = _angles(groups, references).to(q_raw.device)
    freqs = torch.complex(angles.cos(), angles.sin())[:, None]
    return module.routed_shot_attention(
        rope_apply(q_raw, freqs, HEADS), rope_apply(k_raw, freqs, HEADS), v,
        groups, TOKENS_PER_FRAME, window="first_latents", prefix_frames=references,
        routing=routing, freqs=freqs, reference_query_scope=scope,
    )


@pytest.mark.parametrize("routing", [((0, 2), (1,), (0, 2)), ((0, 1, 2), (), (1,))])
@pytest.mark.parametrize("scope", ["all_latents", "first_latent"])
def test_outputs_and_raw_qkv_vjp_match_independent_dense_oracle(routing, scope):
    groups, tokens = _case()
    module = SelfAttention(DIM, HEADS).double()
    leaves = _leaves(tokens)
    args = dict(groups=groups, references=3, routing=routing, scope=scope)
    actual = _sparse(module, *leaves, **args)
    expected = _dense_oracle(*leaves, **args)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    probe = torch.randn(actual.shape, generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    actual_grad = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * probe).sum(), leaves)
    for got, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, wanted, rtol=1e-10, atol=1e-11)


def test_self_attention_forward_and_every_parameter_vjp_match_dense_oracle():
    groups, tokens = _case()
    routing = ((0, 2), (1,), (0, 2))
    torch.manual_seed(23)
    actual_module = SelfAttention(DIM, HEADS).double()
    actual_module.self_attention_policy = "holocine_first_frame"
    expected_module = copy.deepcopy(actual_module)
    x = torch.randn(1, tokens, DIM, dtype=torch.float64).requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    angles = _angles(groups, 3)
    freqs = torch.complex(angles.cos(), angles.sin())[:, None]
    actual = actual_module(
        x, freqs, latent_shot_groups=groups, tokens_per_frame=TOKENS_PER_FRAME,
        memory_attention_policy=FIRST_LATENT_POLICY, prefix_frames=3, reference_routing=routing,
    )
    expected = expected_module.o(_dense_oracle(
        expected_module.norm_q(expected_module.q(expected_x)),
        expected_module.norm_k(expected_module.k(expected_x)),
        expected_module.v(expected_x), groups=groups, references=3,
        routing=routing, scope="first_latent",
    ))
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    probe = torch.randn_like(actual)
    got = torch.autograd.grad((actual * probe).sum(), (x, *actual_module.parameters()))
    wanted = torch.autograd.grad((expected * probe).sum(), (expected_x, *expected_module.parameters()))
    assert tuple(dict(actual_module.named_parameters())) == tuple(dict(expected_module.named_parameters()))
    for name, actual_grad, expected_grad in zip(("input", *dict(actual_module.named_parameters())), got, wanted):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10, msg=name)


def test_reference_derivatives_are_direct_only_for_the_first_latent_queries():
    groups, tokens = _case()
    references, n = 3, TOKENS_PER_FRAME
    routing = ((0, 2), (1,), (0, 2))
    module = SelfAttention(DIM, HEADS).double()
    leaves = _leaves(tokens, seed=29, batch=1)
    output = _sparse(module, *leaves, groups=groups, references=references, routing=routing, scope="first_latent")
    first = groups[references][0] * n
    # 只给本镜首潜帧输出施加cotangent，排除参考自身输出以及别镜的直接路径。
    first_grad = torch.autograd.grad(output[:, first:first + n].square().sum(), leaves, retain_graph=True)
    for gradient in first_grad[1:]:
        assert gradient[:, :n].abs().sum() > 0
        assert torch.count_nonzero(gradient[:, n:2 * n]) == 0  # 参考1只路由给另一镜
        assert gradient[:, 2 * n:3 * n].abs().sum() > 0
    later_grad = torch.autograd.grad(output[:, first + n:first + 2 * n].square().sum(), leaves)
    for gradient in later_grad:
        assert torch.count_nonzero(gradient[:, :references * n]) == 0


def test_reference_outputs_read_only_their_own_reference():
    groups, tokens = _case()
    n = TOKENS_PER_FRAME
    leaves = _leaves(tokens, seed=31, batch=1)
    output = _sparse(SelfAttention(DIM, HEADS).double(), *leaves,
                     groups=groups, references=3, routing=((0, 2), (1,), (0, 2)), scope="first_latent")
    # 参考1输出不得读取参考0/2或任何目标；共享投影参数的梯度不用于判定token可见性。
    gradients = torch.autograd.grad(output[:, n:2 * n].square().sum(), leaves)
    for gradient in gradients:
        assert torch.count_nonzero(gradient[:, :n]) == 0
        assert torch.count_nonzero(gradient[:, 2 * n:]) == 0
        assert gradient[:, n:2 * n].abs().sum() > 0


@pytest.mark.parametrize("lengths", [(1,), (1, 1, 1)])
def test_single_latent_shots_match_the_original_all_queries_policy(lengths):
    groups, tokens = _case(lengths=lengths)
    routing = tuple((0, 1, 2) for _ in lengths)
    q, k, v = _leaves(tokens, seed=37, batch=1)
    module = SelfAttention(DIM, HEADS).double()
    args = dict(groups=groups, references=3, routing=routing)
    first = _sparse(module, q, k, v, scope="first_latent", **args)
    old = _sparse(module, q, k, v, scope="all_latents", **args)
    torch.testing.assert_close(first, old, rtol=0, atol=0)


def test_no_reference_matches_old_scope_and_independent_dense_oracle():
    groups, tokens = _case(references=0)
    leaves = _leaves(tokens, seed=41, batch=1)
    module = SelfAttention(DIM, HEADS).double()
    args = dict(groups=groups, references=0, routing=None)
    first = _sparse(module, *leaves, scope="first_latent", **args)
    old = _sparse(module, *leaves, scope="all_latents", **args)
    expected = _dense_oracle(*leaves, scope="first_latent", **args)
    torch.testing.assert_close(first, old, rtol=0, atol=0)
    torch.testing.assert_close(first, expected, rtol=1e-11, atol=1e-11)


def test_new_policy_does_not_silently_change_full_target_attention():
    groups, tokens = _case()
    module = SelfAttention(DIM, HEADS).double()
    x = torch.randn(1, tokens, DIM, dtype=torch.float64)
    angles = _angles(groups, 3)
    freqs = torch.complex(angles.cos(), angles.sin())[:, None]
    with pytest.raises(ValueError):
        module(x, freqs, latent_shot_groups=groups, tokens_per_frame=TOKENS_PER_FRAME,
               memory_attention_policy=FIRST_LATENT_POLICY, prefix_frames=3,
               reference_routing=((0, 2), (1,), (0, 2)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one leased CUDA GPU")
@pytest.mark.parametrize("scope", ["all_latents", "first_latent"])
def test_cuda_flash_attention_matches_dense_output_and_qkv_vjp(monkeypatch, scope):
    """真实128维head、原生BF16 FA2对独立FP64数学式；跨scope不要求同值。"""
    import functools
    import flash_attn
    from diffsynth.models import wan_video_dit

    assert torch.cuda.device_count() == 1, "test requires exactly one leased CUDA GPU"
    # CPU fixture uses 12-dimensional heads for a small mathematical test. CUDA uses
    # production Wan head_dim=128 and explicit FA2, retaining the independent oracle.
    monkeypatch.setattr(sys.modules[__name__], "HEAD_DIM", 128)
    monkeypatch.setattr(sys.modules[__name__], "DIM", HEADS * 128)
    monkeypatch.setattr(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(flash_attn, "flash_attn_func", functools.partial(flash_attn.flash_attn_func, deterministic=True))
    groups, tokens = _case()
    rng = torch.Generator().manual_seed(71)
    leaves = tuple(torch.randn(1, tokens, DIM, generator=rng).to(
        device="cuda", dtype=torch.bfloat16 if index == 2 else torch.float32).requires_grad_()
        for index in range(3))
    args = dict(groups=groups, references=3, routing=((0, 2), (1,), (0, 2)), scope=scope)
    actual = _sparse(SelfAttention(DIM, HEADS), *leaves, **args)
    expected = _dense_oracle(leaves[0], leaves[1], leaves[2].double(), **args)
    probe = torch.randn(actual.shape, generator=torch.Generator().manual_seed(73)).to("cuda")
    actual_grads = torch.autograd.grad((actual.float() * probe).sum(), leaves, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe.double()).sum(), leaves)
    # Fixed before execution: existing Wan numerical gate, not equality between two
    # deliberately different reference masks. Check every Q/K/V gradient element.
    torch.testing.assert_close(actual.double(), expected, rtol=0.02, atol=0.02)
    for got, wanted in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got.double(), wanted.double(), rtol=0.02, atol=0.02)
