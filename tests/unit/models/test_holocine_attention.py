"""首潜帧稀疏通信与独立 dense mask 的输出、梯度对拍。"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))


def test_sparse_first_frames_match_dense_multiplicity_mask_and_gradients():
    from diffsynth.models.wan_video_dit import SelfAttention
    torch.manual_seed(5)
    attn = SelfAttention(12, 2).double()
    groups, tpf = [(0, 2), (2, 5), (5, 6)], 2
    q, k, v = [torch.randn(1, 12, 12, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    actual = attn.routed_shot_attention(q, k, v, groups, tpf, window="first_latents", reference_query_scope="all_latents")
    # 原视频键空间上的加性掩码：本镜首帧重复读取，等价于其 logit 加 log(2)。
    mask = torch.full((12, 12), -torch.inf, dtype=torch.float64)
    first = [i for a, _ in groups for i in range(a*tpf, (a+1)*tpf)]
    for a, b in groups:
        mask[a*tpf:b*tpf, first] = 0
        mask[a*tpf:b*tpf, a*tpf:b*tpf] = 0
        mask[a*tpf:b*tpf, a*tpf:(a+1)*tpf] = math.log(2)
    split = lambda x: x.reshape(1,12,2,6).transpose(1,2)
    expected = F.scaled_dot_product_attention(split(q),split(k),split(v),attn_mask=mask)
    expected = expected.transpose(1,2).reshape(1,12,12)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual*probe).sum(), (q,k,v), retain_graph=True)
    ref_grads = torch.autograd.grad((expected*probe).sum(), (q,k,v))
    for result, reference in zip(grads, ref_grads):
        torch.testing.assert_close(result, reference)


def test_other_shot_reads_first_frame_but_not_later_frame_directly():
    from diffsynth.models.wan_video_dit import SelfAttention
    attn = SelfAttention(12,2).double()
    q,k,v = [torch.randn(1,8,12,dtype=torch.float64) for _ in range(3)]
    original = attn.routed_shot_attention(q,k,v,[(0,2),(2,4)],2,window="first_latents", reference_query_scope="all_latents")
    changed = v.clone(); changed[:,6:] += 3
    result = attn.routed_shot_attention(q,k,changed,[(0,2),(2,4)],2,window="first_latents", reference_query_scope="all_latents")
    torch.testing.assert_close(original[:,:4],result[:,:4])
    changed[:,4:6] += 3
    result = attn.routed_shot_attention(q,k,changed,[(0,2),(2,4)],2,window="first_latents", reference_query_scope="all_latents")
    assert not torch.allclose(original[:,:4],result[:,:4])


@pytest.mark.parametrize("groups", [None, [(0,2),(2,4)]])
def test_pipeline_forward_backward_for_single_and_multiple_shots(groups):
    from tests.unit.models.test_memory_prefix_attention import _tiny_dit
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "holocine_first_frame"
    latents = torch.randn(1,48,4,8,8,requires_grad=True)
    result = model_fn_wan_video(dit=dit,latents=latents,timestep=torch.tensor([500.]),
        context=torch.randn(1 if groups is None else len(groups),16,4096),
        latent_shot_groups=groups,temporal_position_mode="global_time_native_rope_v1",
        fuse_vae_embedding_in_latents=True,use_gradient_checkpointing=True).noise
    assert result.shape == latents.shape
    result.square().mean().backward()
    assert torch.isfinite(latents.grad).all()


def test_own_shot_window_matches_dense_mask_without_other_shots():
    """镜间读取切除：每镜只读本镜（本镜首帧照旧重复一份），与只留本镜块的加性掩码逐值相同。"""
    from diffsynth.models.wan_video_dit import SelfAttention
    torch.manual_seed(7)
    attn = SelfAttention(12, 2).double()
    groups, tpf = [(0, 2), (2, 5), (5, 6)], 2
    q, k, v = [torch.randn(1, 12, 12, dtype=torch.float64) for _ in range(3)]
    actual = attn.routed_shot_attention(q, k, v, groups, tpf, window="own_first_latent", reference_query_scope="all_latents")
    mask = torch.full((12, 12), -torch.inf, dtype=torch.float64)
    for a, b in groups:
        mask[a*tpf:b*tpf, a*tpf:b*tpf] = 0
        mask[a*tpf:b*tpf, a*tpf:(a+1)*tpf] = math.log(2)
    split = lambda x: x.reshape(1, 12, 2, 6).transpose(1, 2)
    expected = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask).transpose(1, 2).reshape(1, 12, 12)
    torch.testing.assert_close(actual, expected)


def test_own_shot_window_ignores_other_shots_first_frames_and_matches_single_shot_holocine():
    from diffsynth.models.wan_video_dit import SelfAttention
    attn = SelfAttention(12, 2).double()
    q, k, v = [torch.randn(1, 8, 12, dtype=torch.float64) for _ in range(3)]
    groups = [(0, 2), (2, 4)]
    original = attn.routed_shot_attention(q, k, v, groups, 2, window="own_first_latent", reference_query_scope="all_latents")
    changed = v.clone(); changed[:, 4:6] += 3   # 第二镜首潜帧
    result = attn.routed_shot_attention(q, k, changed, groups, 2, window="own_first_latent", reference_query_scope="all_latents")
    torch.testing.assert_close(original[:, :4], result[:, :4])
    single = [(0, 2)]
    torch.testing.assert_close(
        attn.routed_shot_attention(q[:, :4], k[:, :4], v[:, :4], single, 2, window="own_first_latent", reference_query_scope="all_latents"),
        attn.routed_shot_attention(q[:, :4], k[:, :4], v[:, :4], single, 2, window="first_latents", reference_query_scope="all_latents"))
