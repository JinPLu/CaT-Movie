"""二维布局的数学、原文字 CA、正式 runtime 与 checkpoint 的 CPU 合同。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import math
import sys

import pytest
import torch

from tests.support.memory_prefix import full_window
from mspc.models.spatial_layout import (
    SPATIAL_LAYOUT_MODE,
    SpatialLayoutAnchor,
    SpatialLayoutCondition,
    SpatialLayoutEntity,
    SpatialLayoutShot,
    build_spatial_layout_biases,
    patch_area_coverage,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import CrossAttention, WanModel, text_attention
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video


def _shot(num_frames=9, *, token=1, anchors=None):
    if anchors is None:
        anchors = (SpatialLayoutAnchor(0.25, (0.0, 0.0, 0.5, 1.0)),)
    return SpatialLayoutShot(
        num_frames=num_frames,
        entities=(SpatialLayoutEntity((token,), anchors),),
        text_token_mask=(True, True, True, False, False),
    )


def _condition(shots=None, *, strength=2.0, layers=(0, 1), timestep_range=(0.0, 1000.0)):
    return SpatialLayoutCondition(
        shots=(_shot(),) if shots is None else shots,
        strength=strength, layer_indices=layers,
        timestep_range=timestep_range, mode=SPATIAL_LAYOUT_MODE, region_dilation=None, onset_latent_frames=None,
    )


def _biases(condition, *, groups=((0, 3),), prefix=0, timestep=500.0, h=2, w=2):
    return build_spatial_layout_biases(
        condition, latent_shot_groups=groups, latent_frames=groups[-1][1],
        patch_height=h, patch_width=w, prefix_frames=prefix, text_length=5,
        timestep=torch.tensor([timestep]), num_layers=2, device="cpu",
    )


def test_box_coverage_uses_intersection_area_and_actual_patch_grid():
    coverage = patch_area_coverage((0.25, 0.0, 0.5, 1.0), 2, 2, device="cpu")
    torch.testing.assert_close(coverage, torch.tensor([[0.5, 0.0], [0.5, 0.0]]))
    native = patch_area_coverage((0.0, 0.0, 1.0, 1.0), 15, 26, device="cpu")
    assert native.shape == (15, 26)
    torch.testing.assert_close(native, torch.ones_like(native), atol=3e-6, rtol=0)


def test_fractional_anchor_uses_only_adjacent_latent_times_and_entity_keys():
    bias, = _biases(_condition())
    actual = bias[:, 1].view(3, 2, 2)
    expected = torch.tensor([[[1.0, 0.0], [1.0, 0.0]],
                             [[1.0, 0.0], [1.0, 0.0]],
                             [[0.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(bias[:, [0, 2, 3, 4]]) == 0


def test_two_anchors_take_maximum_and_do_not_fill_a_motion_trajectory():
    full = (0.0, 0.0, 1.0, 1.0)
    overlap = _shot(anchors=(SpatialLayoutAnchor(0.25, full), SpatialLayoutAnchor(0.375, full)))
    bias, = _biases(_condition((overlap,)), h=1, w=1)
    torch.testing.assert_close(bias[:, 1], torch.tensor([1.0, 1.5, 0.0]))
    endpoints = _shot(33, anchors=(SpatialLayoutAnchor(0.0, full), SpatialLayoutAnchor(1.0, full)))
    bias, = _biases(_condition((endpoints,)), groups=((0, 9),), h=1, w=1)
    torch.testing.assert_close(bias[:, 1], torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]))


def test_prefix_queries_and_other_shot_keys_are_zero_with_independent_vae_lengths():
    condition = _condition((_shot(9), _shot(5, token=2)))
    prefix, first, second = _biases(condition, groups=((0, 1), (1, 4), (4, 6)), prefix=1)
    assert torch.count_nonzero(prefix) == 0
    assert first.shape == (12, 5) and second.shape == (8, 5)
    assert torch.count_nonzero(first[:, 2]) == 0
    assert torch.count_nonzero(second[:, 1]) == 0
    assert second[:, 2].sum() > 0
    # Native first-frame conditioning shares a target text row, but its queries stay unbiased.
    first_frame, = _biases(_condition(), prefix=1)
    assert torch.count_nonzero(first_frame[:4]) == 0
    assert first_frame[4:].sum() > 0


def test_scheduler_timestep_window_uses_original_t_units():
    condition = _condition(timestep_range=(400.0, 600.0))
    assert _biases(condition, timestep=500.0)[0].sum() > 0
    assert torch.count_nonzero(_biases(condition, timestep=0.5)[0]) == 0


def test_inconsistent_layout_alignment_is_rejected():
    with pytest.raises(ValueError, match="padding and special"):
        _shot(token=3)
    with pytest.raises(ValueError, match="independently encoded"):
        _biases(_condition(), groups=((0, 4),))
    with pytest.raises(ValueError, match="actual DiT layers"):
        _biases(_condition(layers=(2,)))
    with pytest.raises(ValueError, match="finite in float32"):
        _condition(strength=1e100)


def test_float_bias_is_added_after_qk_scaling_and_keeps_other_keys_finite():
    q = torch.zeros(1, 2, 4)
    k = torch.zeros(1, 3, 4)
    v = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    bias = torch.tensor([[0.0, math.log(2.0), 0.0], [0.0, 0.0, 0.0]])
    actual = text_attention(q, k, v, 2, backend="sdpa", bias=bias)
    torch.testing.assert_close(actual[0, 0], torch.tensor([0.25, 0.5, 0.25, 0.5]))
    torch.testing.assert_close(actual[0, 1], torch.full((4,), 1 / 3))


def test_zero_bias_matches_same_sdpa_kernel_output_and_all_input_parameter_gradients():
    torch.manual_seed(42)
    module = CrossAttention(24, 2).double()
    x = torch.randn(1, 12, 24, dtype=torch.float64, requires_grad=True)
    context = torch.randn(2, 5, 24, dtype=torch.float64, requires_grad=True)
    ranges = ((0, 4), (4, 12))
    baseline = module(x, context, ranges, text_attention_backend="sdpa").hidden_states
    zeros = (torch.zeros(4, 5), torch.zeros(8, 5))
    actual = module(x, context, ranges, zeros, "sdpa").hidden_states
    torch.testing.assert_close(actual, baseline, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(actual)
    inputs = (x, context, *module.parameters())
    expected_grads = torch.autograd.grad((baseline * probe).sum(), inputs, retain_graph=True)
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
    for expected, observed in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)
    # Native versus reference kernels are compared by numerical tolerance, not bits.
    native = module(x, context, ranges).hidden_states
    torch.testing.assert_close(native, baseline, atol=1e-6, rtol=1e-6)


class _FixedTimestepScheduler(FlowMatchScheduler):
    """管线按名字选采样器后保留同类调度器；这里固定一个 t=500 的步，不做真实去噪。"""

    def __init__(self):
        self.timesteps = torch.tensor([500.0])

    def set_timesteps(self, *args, **kwargs):
        return None

    def step(self, prediction, timestep, sample):
        return sample


def _tiny_dit():
    torch.manual_seed(19)
    return WanModel(
        dim=24, in_dim=48, out_dim=48, ffn_dim=32, text_dim=16,
        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        num_layers=2, has_image_input=False, seperated_timestep=True,
        require_clip_embedding=False, require_vae_embedding=False,
    )


def _runtime_inputs(condition):
    torch.manual_seed(17)
    return dict(
        latents=torch.randn(1, 48, 6, 4, 4, requires_grad=True),
        context=torch.randn(3, 5, 16, requires_grad=True),
        timestep=torch.tensor([500.0]),
        latent_shot_groups=((0, 1), (1, 4), (4, 6)),
        fuse_vae_embedding_in_latents=True, prefix_frames=1,
        reference_offset=(-5, 0, 0), memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window(),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        spatial_layout=condition, text_attention_backend="sdpa",
    )


@pytest.mark.parametrize("checkpoint,offload", [(False, False), (True, False), (False, True)])
def test_direct_forward_and_runtime_share_layout_and_checkpoint_gradients(checkpoint, offload):
    condition = _condition((_shot(9), _shot(5, token=2)), layers=(1,))
    expected_model = _tiny_dit()
    actual_model = copy.deepcopy(expected_model)
    expected_inputs = _runtime_inputs(condition)
    actual_inputs = _runtime_inputs(condition)
    actual_latents = actual_inputs.pop("latents")
    expected = model_fn_wan_video(dit=expected_model, **expected_inputs).noise
    actual = actual_model(
        x=actual_latents, **actual_inputs,
        use_gradient_checkpointing=checkpoint, use_gradient_checkpointing_offload=offload,
    ).noise
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual_latents.grad, expected_inputs["latents"].grad)
    torch.testing.assert_close(actual_inputs["context"].grad, expected_inputs["context"].grad)
    for expected_parameter, actual_parameter in zip(expected_model.parameters(), actual_model.parameters(), strict=True):
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_runtime_consumes_only_selected_layers_and_preserves_original_t_before_prefix():
    condition = _condition((_shot(9), _shot(5, token=2)), layers=(1,), timestep_range=(400.0, 600.0))
    model = _tiny_dit()
    observed = []
    handles = [block.cross_attn.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["spatial_layout_biases"]), with_kwargs=True,
    ) for block in model.blocks]
    model_fn_wan_video(dit=model, **_runtime_inputs(condition)).noise
    for handle in handles:
        handle.remove()
    assert observed[0] is None
    assert torch.count_nonzero(observed[1][0]) == 0
    assert observed[1][1].sum() > 0 and observed[1][2].sum() > 0
    with pytest.raises(ValueError, match="requires text_attention_backend"):
        model_fn_wan_video(dit=model, **(_runtime_inputs(condition) | {"text_attention_backend": "native"})).noise


def test_real_pipeline_cfg_calls_share_backend_and_bias_only_positive_branch():
    model = _tiny_dit().eval()
    condition = _condition((_shot(9), _shot(5, token=2)))
    latents = torch.randn(1, 48, 5, 4, 4)
    context = torch.randn(2, 5, 16)
    calls = []
    pipe = SimpleNamespace(
        device="cpu", torch_dtype=torch.float32, dit=model, dit2=None,
        units=[object()], post_units=[], in_iteration_models=["dit"],
        check_resize_height_width=lambda h, w, f: (h, w), load_models_to_device=lambda _: None,
    )
    pipe.scheduler = _FixedTimestepScheduler()
    def unit_runner(unit, pipeline, shared, positive, negative):
        shared.update(latents=latents, context=context)
        return shared, positive, negative
    def model_fn(**kwargs):
        calls.append((kwargs["spatial_layout"], kwargs["text_attention_backend"]))
        return model_fn_wan_video(**kwargs)
    pipe.unit_runner, pipe.model_fn = unit_runner, model_fn
    pipe.vae = SimpleNamespace(decode=lambda value, **kwargs: torch.zeros(1, 3, 4 * (value.shape[2] - 1) + 1, 4, 4))
    WanVideoPipeline.__call__(
        pipe, prompt=["first", "second"], negative_prompt=["", ""],
        height=4, width=4, num_frames=14,
        shot_groups=((0, 9), (9, 14)), latent_shot_groups=((0, 3), (3, 5)),
        spatial_layout=condition, text_attention_backend="sdpa",
        cfg_scale=5.0, num_inference_steps=1, sampler="flow_match_euler",
        output_type="floatpoint", progress_bar_cmd=lambda values: values,
    )
    assert calls == [(condition, "sdpa"), (None, "sdpa")]


def test_real_umt5_plan_compile_encode_to_tiny_runtime_cfg_and_reference_prefix():
    """使用已缓存真实词表，无T5/DiT/VAE权重；context是确定性的token位置测试张量。"""
    from mspc.models.spatial_layout_inputs import (
        SpatialLayoutOptions, compile_spatial_layout_inputs, encode_spatial_layout_inputs,
    )
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

    tokenizer_path = Path.home() / ".cache/huggingface/hub/models--google--umt5-xxl/snapshots/66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
    if not tokenizer_path.is_dir():
        pytest.skip("real umT5 tokenizer is not cached locally; this test never downloads assets")
    tokenizer = HuggingfaceTokenizer(str(tokenizer_path), seq_len=512, clean="whitespace", local_files_only=True)
    fixture_path = Path(__file__).parents[2] / "fixtures/block_annotation_v18/film-002869-b0001.json"
    annotation = json.loads(fixture_path.read_text())["annotation"]
    annotation["layouts"] = [
        {"shot": index + 1, "entities": [{"name": annotation["shots"][index]["present"][0]["name"],
          "anchors": [{"time": 0.25, "visibility": "visible", "box": [0.0, 0.0, 0.5, 1.0]}]}]}
        for index in range(2)
    ]
    compiled = compile_spatial_layout_inputs(annotation, (0, 1))
    target_prompts = [row["text"] for row in compiled]
    options = SpatialLayoutOptions.from_mapping({
        "text_attention_backend": "sdpa", "strength": 0.7,
        "layer_indices": [0, 1], "timestep_range": [400.0, 600.0], "mode": "text_cross_attention_bias_v1",
    })
    condition = encode_spatial_layout_inputs(
        tokenizer, compiled, prompts=target_prompts, frame_counts=(9, 5), options=options,
    )
    prompts = ["An actual reference frame.", *target_prompts]
    negative_prompts = ["", "", ""]
    def token_features(texts):
        ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
        return torch.sin(ids.float()[..., None] / torch.arange(1, 17)) * mask[..., None]
    positive_context, negative_context = token_features(prompts), token_features(negative_prompts)
    model = _tiny_dit().eval()
    torch.manual_seed(21)
    latents = torch.randn(1, 48, 6, 4, 4)
    policies = dict(
        prefix_frames=1, fuse_vae_embedding_in_latents=True, reference_offset=(-5, 0, 0),
        memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window(),
    )
    calls, seen_biases = [], []
    handles = [block.cross_attn.register_forward_pre_hook(
        lambda module, args, kwargs: seen_biases.append(kwargs.get("spatial_layout_biases")), with_kwargs=True,
    ) for block in model.blocks]
    pipe = SimpleNamespace(
        device="cpu", torch_dtype=torch.float32, dit=model, dit2=None,
        units=[object()], post_units=[], in_iteration_models=["dit"],
        check_resize_height_width=lambda h, w, f: (h, w), load_models_to_device=lambda _: None,
    )
    pipe.scheduler = _FixedTimestepScheduler()
    def unit_runner(unit, pipeline, shared, positive, negative):
        shared.update(latents=latents, **policies)
        positive["context"], negative["context"] = positive_context, negative_context
        return shared, positive, negative
    def model_fn(**kwargs):
        result = model_fn_wan_video(**kwargs)
        calls.append((kwargs["spatial_layout"], kwargs["text_attention_backend"], result.noise))
        return result
    pipe.unit_runner, pipe.model_fn = unit_runner, model_fn
    pipe.vae = SimpleNamespace(decode=lambda value, **kwargs: torch.zeros(1, 3, 4 * (value.shape[2] - 1) + 1, 4, 4))
    WanVideoPipeline.__call__(
        pipe, prompt=prompts, negative_prompt=negative_prompts, height=4, width=4, num_frames=15,
        shot_groups=((0, 1), (1, 10), (10, 15)), latent_shot_groups=((0, 1), (1, 4), (4, 6)),
        reference_frames=1, temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        spatial_layout=condition, text_attention_backend="sdpa", cfg_scale=5.0,
        num_inference_steps=1, sampler="flow_match_euler", output_type="floatpoint", progress_bar_cmd=lambda values: values,
    )
    for handle in handles:
        handle.remove()
    assert calls[0][:2] == (condition, "sdpa") and calls[1][:2] == (None, "sdpa")
    assert seen_biases[2:] == [None, None]
    for biases in seen_biases[:2]:
        assert torch.count_nonzero(biases[0]) == 0
        for shot, bias in zip(condition.shots, biases[1:], strict=True):
            allowed = set(shot.entities[0].token_indices)
            assert bias[:, list(allowed)].sum() > 0
            assert torch.count_nonzero(bias[:, [key for key in range(512) if key not in allowed]]) == 0
    # 同一正文字与同内核的零条件比较，验证布局实际改变了tiny DiT的预测。
    baseline = model_fn_wan_video(
        dit=model, latents=latents, context=positive_context, timestep=torch.tensor([500.0]),
        latent_shot_groups=((0, 1), (1, 4), (4, 6)), text_attention_backend="sdpa",
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1", **policies,
    ).noise
    zero_condition = model_fn_wan_video(
        dit=model, latents=latents, context=positive_context, timestep=torch.tensor([500.0]),
        latent_shot_groups=((0, 1), (1, 4), (4, 6)), text_attention_backend="sdpa",
        spatial_layout=replace(condition, strength=0.0),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1", **policies,
    ).noise
    torch.testing.assert_close(zero_condition, baseline, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(calls[0][2], baseline, atol=1e-7, rtol=1e-7)
