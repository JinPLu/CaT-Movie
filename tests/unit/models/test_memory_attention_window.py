"""参考只支持全窗口全双向注意力；用独立dense公式检查输出和梯度。"""
from __future__ import annotations

import math

import pytest
import torch

from mspc.models.memory_prefix_options import MemoryAttentionWindow, MemoryPrefixModelOptions
from tests.support.memory_prefix import FULL_WINDOW_CONFIG, INFERENCE_OFFSET, NOISE_CONFIG, OFFSET_CONFIG, full_window
from tests.unit.models.test_memory_prefix_attention import _forward, _latents, _tiny_dit
from tests.unit.models.test_routed_reference_attention import DIT_GROUPS, DIT_REFERENCES, DIT_ROUTING, _dit, _latents as _routed_latents


def _options(**changes):
    raw = {"frames_max": 10, "dropout_p": 0.2, "noise": NOISE_CONFIG, "reference_offset": OFFSET_CONFIG,
           "attention_policy": "bidirectional_v1", "row_policy": "fixed_role_v1",
           "attention_window": FULL_WINDOW_CONFIG, "placement": "ordered_slots"}
    return MemoryPrefixModelOptions.from_mapping({**raw, **changes})


def test_reference_attention_matches_full_dense_attention_and_gradients():
    from diffsynth.models.wan_video_dit import SelfAttention
    module = SelfAttention(6, 2).eval()
    generator = torch.Generator().manual_seed(17)
    actual_inputs = [torch.randn(1, 20, 6, generator=generator).requires_grad_() for _ in range(3)]
    expected_inputs = [x.detach().clone().requires_grad_() for x in actual_inputs]
    actual = module.memory_prefix_attention(*actual_inputs, "bidirectional_v1", 2, 4)
    q, k, v = expected_inputs
    expected = torch.cat([
        torch.softmax(q[..., cut] @ k[..., cut].transpose(-1, -2) / math.sqrt(3), dim=-1) @ v[..., cut]
        for cut in (slice(0, 3), slice(3, 6))
    ], dim=-1)
    torch.testing.assert_close(actual, expected)
    # 同时覆盖参考query与目标query，而不只量单向读取。
    actual.square().mean().backward()
    expected.square().mean().backward()
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_full_window_reaches_every_dit_block():
    dit = _tiny_dit()
    seen = []
    handles = [block.self_attn.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["memory_attention_policy"]), with_kwargs=True)
        for block in dit.blocks]
    try:
        _forward(dit, _latents(7), attention_policy="bidirectional_v1")
    finally:
        for handle in handles:
            handle.remove()
    assert seen == ["bidirectional_v1"] * len(dit.blocks)


def test_model_fn_requires_the_full_window_with_the_policy():
    with pytest.raises(ValueError, match="memory_attention_window declare a memory prefix together"):
        _forward(_tiny_dit(), _latents(8), attention_policy="bidirectional_v1", window=None)
    # 直接构造对象也不能绕过consumer的full-window合同。
    partial = MemoryAttentionWindow((0,), (700.0, 1000.0))
    with pytest.raises(ValueError, match="full window"):
        _forward(_tiny_dit(), _latents(8), attention_policy="bidirectional_v1", window=partial)


def test_window_config_is_required_and_full():
    assert _options().attention_window.full
    with pytest.raises(ValueError, match="attention_window"):
        raw = dict(_options().__dict__)
        raw.pop("attention_window")
        MemoryPrefixModelOptions.from_mapping(raw)
    for window in ({"layer_indices": [30], "timestep_range": [0, 1000]},
                   {"layer_indices": [0], "timestep_range": [700, 600]},
                   {"layer_indices": [0], "timestep_range": [0, 1000]},
                   {"layer_indices": list(range(30)), "timestep_range": [700, 1000]}):
        with pytest.raises(ValueError):
            _options(attention_window=window)


@pytest.mark.parametrize("frames", [1, 4, 10])
def test_ordered_reference_capacity_is_explicit_and_supports_ten(frames):
    assert _options(frames_max=frames).frames_max == frames


@pytest.mark.parametrize("frames", [0, 11])
def test_reference_capacity_outside_supported_range_raises(frames):
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        _options(frames_max=frames)


def test_routed_policy_allows_partial_timestep_range_but_requires_all_layers():
    opts = _options(
        attention_policy="routed_self_only_v1",
        attention_window={"layer_indices": list(range(30)), "timestep_range": [0.0, 874.9]},
    )
    assert opts.attention_window.full_layers
    assert not opts.attention_window.full
    assert opts.attention_window.contains_timestep(874.9)
    assert not opts.attention_window.contains_timestep(875.0)
    with pytest.raises(ValueError, match="all"):
        _options(
            attention_policy="routed_self_only_v1",
            attention_window={"layer_indices": [0, 1], "timestep_range": [0.0, 874.9]},
        )


def test_routed_timestep_gating_matches_m0_outside_window_and_full_inside_window():
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    dit = _dit(policy="full")
    latents = _routed_latents(5)
    target_latents = latents[:, :, DIT_REFERENCES:]
    context = torch.randn(len(DIT_GROUPS), 16, 4096, generator=torch.Generator().manual_seed(9))
    target_context = context[DIT_REFERENCES:]
    target_groups = tuple((s - DIT_REFERENCES, e - DIT_REFERENCES) for s, e in DIT_GROUPS[DIT_REFERENCES:])
    partial_window = MemoryAttentionWindow(tuple(range(30)), (0.0, 874.9))

    with torch.no_grad():
        # Outside window (t=900.0): target output is bit-exact with M=0.
        gated_out = model_fn_wan_video(
            dit=dit,
            latents=latents,
            timestep=torch.tensor([900.0]),
            context=context,
            temporal_position_mode="global_time_native_rope_v1",
            latent_shot_groups=DIT_GROUPS,
            fuse_vae_embedding_in_latents=True,
            prefix_frames=DIT_REFERENCES,
            reference_offset=INFERENCE_OFFSET,
            memory_attention_policy="routed_self_only_v1",
            memory_attention_window=partial_window,
            reference_routing=DIT_ROUTING,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        ).noise[:, :, DIT_REFERENCES:]
        m0_out = model_fn_wan_video(
            dit=dit,
            latents=target_latents,
            timestep=torch.tensor([900.0]),
            context=target_context,
            temporal_position_mode="global_time_native_rope_v1",
            latent_shot_groups=target_groups,
            fuse_vae_embedding_in_latents=False,
            prefix_frames=0,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        ).noise
        assert torch.equal(gated_out, m0_out)

        # Inside window (t=800.0): target output is bit-exact with full_window().
        active_out = model_fn_wan_video(
            dit=dit,
            latents=latents,
            timestep=torch.tensor([800.0]),
            context=context,
            temporal_position_mode="global_time_native_rope_v1",
            latent_shot_groups=DIT_GROUPS,
            fuse_vae_embedding_in_latents=True,
            prefix_frames=DIT_REFERENCES,
            reference_offset=INFERENCE_OFFSET,
            memory_attention_policy="routed_self_only_v1",
            memory_attention_window=partial_window,
            reference_routing=DIT_ROUTING,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        ).noise[:, :, DIT_REFERENCES:]
        full_out = model_fn_wan_video(
            dit=dit,
            latents=latents,
            timestep=torch.tensor([800.0]),
            context=context,
            temporal_position_mode="global_time_native_rope_v1",
            latent_shot_groups=DIT_GROUPS,
            fuse_vae_embedding_in_latents=True,
            prefix_frames=DIT_REFERENCES,
            reference_offset=INFERENCE_OFFSET,
            memory_attention_policy="routed_self_only_v1",
            memory_attention_window=full_window(),
            reference_routing=DIT_ROUTING,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        ).noise[:, :, DIT_REFERENCES:]
        assert torch.equal(active_out, full_out)


def test_training_rejects_partial_timestep_range():
    from mspc.training.backends.wan22 import _memory_prefix_options

    with pytest.raises(ValueError, match="full window"):
        _memory_prefix_options({
            "memory_prefix": {
                "frames_max": 10,
                "dropout_p": 0.2,
                "noise": NOISE_CONFIG,
                "reference_offset": OFFSET_CONFIG,
                "attention_policy": "routed_self_only_v1",
                "row_policy": "fixed_role_v1",
                "attention_window": {"layer_indices": list(range(30)), "timestep_range": [0.0, 874.9]},
                "placement": "ordered_slots",
            }
        })
