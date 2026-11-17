"""现行自注意力和按镜文字路由与独立dense计算对拍。"""
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import CrossAttention, SelfAttention, rope_apply


@pytest.fixture(autouse=True)
def cpu_math_attention_backend(monkeypatch):
    from diffsynth.models import wan_video_dit

    # The reference is CPU SDPA mathematics, independent of installed CUDA-only backends.
    for flag in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(wan_video_dit, flag, False)


def _heads(value, heads=2):
    return value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2)


def test_native_self_attention_matches_dense_forward_and_gradient():
    torch.manual_seed(3)
    module = SelfAttention(24, 2).double()
    x = torch.randn(1, 15, 24, dtype=torch.float64, requires_grad=True)
    angles = torch.randn(15, 1, 6, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angles), angles)
    actual = module(x, freqs)
    q = rope_apply(module.norm_q(module.q(x)), freqs, 2)
    k = rope_apply(module.norm_k(module.k(x)), freqs, 2)
    v = module.v(x)
    dense = F.scaled_dot_product_attention(_heads(q), _heads(k), _heads(v))
    expected = module.o(dense.transpose(1, 2).reshape_as(x))
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(x)
    actual_grad = torch.autograd.grad((actual * probe).sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_per_shot_text_matches_dense_window_mask_and_gradients():
    torch.manual_seed(7)
    module = CrossAttention(24, 2).double()
    x = torch.randn(1, 15, 24, dtype=torch.float64, requires_grad=True)
    context = torch.randn(3, 5, 24, dtype=torch.float64, requires_grad=True)
    ranges = [(0, 3), (3, 9), (9, 15)]
    actual = module(x, context, ranges).hidden_states
    q = module.norm_q(module.q(x))
    k = module.norm_k(module.k(context)).reshape(1, 15, 24)
    v = module.v(context).reshape(1, 15, 24)
    allowed = torch.zeros(15, 15, dtype=torch.bool)
    for index, (start, end) in enumerate(ranges):
        allowed[start:end, index * 5:(index + 1) * 5] = True
    dense = F.scaled_dot_product_attention(_heads(q), _heads(k), _heads(v), attn_mask=allowed)
    expected = module.o(dense.transpose(1, 2).reshape_as(x))
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(x)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (x, context), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), (x, context))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_cross_attention_reads_only_assigned_text_row():
    torch.manual_seed(9)
    module = CrossAttention(24, 2).double()
    x = torch.randn(1, 9, 24, dtype=torch.float64)
    context = torch.randn(3, 5, 24, dtype=torch.float64)
    ranges = [(0, 3), (3, 6), (6, 9)]
    base = module(x, context, ranges).hidden_states
    changed = context.clone(); changed[1] += 4
    moved = module(x, changed, ranges).hidden_states
    torch.testing.assert_close(base[:, :3], moved[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(base[:, 6:], moved[:, 6:], rtol=0, atol=0)
    assert not torch.allclose(base[:, 3:6], moved[:, 3:6])
