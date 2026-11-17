"""空间绑定两版（区域外压低、门控位置 token）与共同布局输入的 CPU 合同。"""

import copy
import math
from pathlib import Path
import sys

import pytest
import torch

from tests.support.memory_prefix import full_window
from mspc.models.region_attention import (
    GatedRegionAttention, attach_layout_modules, attach_region_attention, region_parameter_names,
)
from mspc.models.spatial_layout import (
    GATED_REGION_MODE, SPATIAL_LAYOUT_MODE, SUPPRESSION_MODE, SpatialLayoutAnchor, SpatialLayoutCondition,
    SpatialLayoutEntity, SpatialLayoutShot, build_region_token_inputs, build_spatial_layout_biases, dilate_box,
)
from mspc.models.spatial_layout_inputs import SpatialLayoutOptions
from mspc.training.block_dataset import jitter_layout_entity_boxes, require_box_jitter
from mspc.training.dit_parameter_freeze import load_dit_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video

LEFT = (0.0, 0.0, 0.5, 1.0)


def _shot(num_frames=9, *, token=1, anchors=None):
    anchors = (SpatialLayoutAnchor(0.0, LEFT), SpatialLayoutAnchor(1.0, (0.5, 0.0, 1.0, 1.0))) \
        if anchors is None else anchors
    return SpatialLayoutShot(num_frames=num_frames, entities=(SpatialLayoutEntity((token,), anchors),),
                             text_token_mask=(True, True, True, False, False))


def _condition(mode, shots=None, *, strength=None, layers=(0, 1), window=(0.0, 1000.0), dilation=0.0, onset=2):
    if strength is None:
        strength = 0.0 if mode == GATED_REGION_MODE else math.log(10)
    return SpatialLayoutCondition(
        shots=(_shot(),) if shots is None else shots, strength=strength, layer_indices=layers,
        timestep_range=window, mode=mode, region_dilation=dilation, onset_latent_frames=onset)


def _bias(condition, *, groups=((0, 3),), prefix=0, timestep=950.0):
    return build_spatial_layout_biases(
        condition, latent_shot_groups=groups, latent_frames=groups[-1][1], patch_height=2, patch_width=2,
        prefix_frames=prefix, text_length=5, timestep=torch.tensor([timestep]), num_layers=2, device="cpu")


def test_suppression_is_zero_inside_region_negative_outside_and_only_at_onset_frames():
    bias, = _bias(_condition(SUPPRESSION_MODE))
    grid = bias[:, 1].reshape(3, 2, 2)
    lam = math.log(10)
    # 首锚是左半幅：左列 0、右列 -λ·开头权重；开头 2 潜帧权重 1、1/2，第 3 潜帧为 0（不钉镜尾的右半幅锚）。
    torch.testing.assert_close(grid[0], torch.tensor([[0.0, -lam], [0.0, -lam]]))
    torch.testing.assert_close(grid[1], torch.tensor([[0.0, -lam / 2], [0.0, -lam / 2]]))
    assert torch.count_nonzero(grid[2]) == 0
    assert torch.count_nonzero(bias[:, [0, 2, 3, 4]]) == 0
    assert float(bias.max()) == 0.0


def test_dilation_expands_region_and_window_turns_suppression_off():
    assert dilate_box((0.2, 0.3, 0.4, 0.5), 0.25) == pytest.approx((0.0, 0.05, 0.65, 0.75))
    wide, = _bias(_condition(SUPPRESSION_MODE, dilation=0.5))
    assert torch.count_nonzero(wide) == 0
    off, = _bias(_condition(SUPPRESSION_MODE, window=(900.0, 1000.0)), timestep=500.0)
    assert torch.count_nonzero(off) == 0


def test_region_modes_require_their_fields_and_box_mode_rejects_them():
    with pytest.raises(ValueError, match="region_dilation"):
        _condition(SUPPRESSION_MODE, dilation=None)
    with pytest.raises(ValueError, match="no region_dilation"):
        SpatialLayoutCondition(shots=(_shot(),), strength=1.0, layer_indices=(0,), timestep_range=(0.0, 1000.0),
                               mode=SPATIAL_LAYOUT_MODE, region_dilation=0.1, onset_latent_frames=3)
    with pytest.raises(ValueError, match="strength must be 0"):
        _condition(GATED_REGION_MODE, strength=1.0)
    with pytest.raises(ValueError, match="not text biases"):
        _bias(_condition(GATED_REGION_MODE))


def test_region_inputs_skip_prefix_and_return_none_outside_window():
    condition = _condition(GATED_REGION_MODE, (_shot(9), _shot(5, token=2)), window=(900.0, 1000.0))
    kwargs = dict(latent_shot_groups=((0, 1), (1, 4), (4, 6)), latent_frames=6, patch_height=2, patch_width=2,
                  prefix_frames=1, device="cpu")
    assert build_region_token_inputs(condition, timestep=torch.tensor([500.0]), **kwargs) is None
    rows = build_region_token_inputs(condition, timestep=torch.tensor([950.0]), **kwargs)
    assert [(row.token_start, row.token_end) for row in rows] == [(4, 16), (16, 24)]
    weights = rows[0].region_weights.reshape(3, 2, 2)
    torch.testing.assert_close(weights[0], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert rows[0].boxes == (LEFT,) and rows[1].entity_token_indices == ((2,),)


def _tiny_dit():
    torch.manual_seed(19)
    return WanModel(dim=24, in_dim=48, out_dim=48, ffn_dim=32, text_dim=16, freq_dim=16, eps=1e-6,
                    patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=False,
                    seperated_timestep=True, require_clip_embedding=False, require_vae_embedding=False)


def _inputs(condition, backend):
    torch.manual_seed(17)
    return dict(latents=torch.randn(1, 48, 6, 4, 4), context=torch.randn(3, 5, 16), timestep=torch.tensor([950.0]),
                latent_shot_groups=((0, 1), (1, 4), (4, 6)), fuse_vae_embedding_in_latents=True, prefix_frames=1,
                reference_offset=(-5, 0, 0), memory_attention_policy="bidirectional_v1",
                memory_attention_window=full_window(), temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
                spatial_layout=condition, text_attention_backend=backend)


def test_gated_channel_with_closed_gate_is_bitwise_the_original_dit_and_opens_only_inside_regions():
    condition = _condition(GATED_REGION_MODE, (_shot(9), _shot(5, token=2)), layers=(1,))
    original = _tiny_dit()
    expected = model_fn_wan_video(dit=original, **_inputs(None, "native")).noise
    gated = copy.deepcopy(original)
    attach_region_attention(gated, layer_indices=(1,), fourier_frequencies=2)
    assert torch.equal(model_fn_wan_video(dit=gated, **_inputs(condition, "native")).noise, expected)
    with torch.no_grad():
        gated.blocks[1].region_attn.gate.fill_(1.0)
    opened = model_fn_wan_video(dit=gated, **_inputs(condition, "native")).noise
    assert not torch.equal(opened, expected)
    opened.square().mean().backward()
    assert gated.blocks[1].region_attn.gate.grad is not None
    assert gated.region_token_encoder.mlp[0].weight.grad.abs().sum() > 0


def test_region_attention_writes_nothing_outside_active_region_tokens():
    torch.manual_seed(3)
    module = GatedRegionAttention(8, 2)
    with torch.no_grad():
        module.gate.fill_(1.0)
    condition = _condition(GATED_REGION_MODE)
    rows = build_region_token_inputs(condition, latent_shot_groups=((0, 3),), latent_frames=3, patch_height=2,
                                     patch_width=2, prefix_frames=0, timestep=torch.tensor([950.0]), device="cpu")
    delta = module(torch.randn(1, 12, 8), rows, (torch.randn(2, 8),))
    active = rows[0].region_weights.max(dim=1).values > 0
    assert torch.count_nonzero(delta[0, ~active]) == 0 and torch.count_nonzero(delta[0, active]) > 0


def test_warm_start_loads_base_keys_and_keeps_new_channel_initialized():
    base = _tiny_dit()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    target = _tiny_dit()
    attach_region_attention(target, layer_indices=(0,), fourier_frequencies=2)
    names = region_parameter_names(target)
    assert names and all(n.startswith(("region_token_encoder.", "blocks.0.region_attn.")) for n in names)
    assert load_dit_state_dict(target, state) == "warm_start_plus_new_channel"
    assert float(target.blocks[0].region_attn.gate.detach()) == 0.0
    with pytest.raises(ValueError, match="exactly the DiT keys"):
        load_dit_state_dict(target, {k: v for k, v in state.items() if k != "head.head.weight"})
    full = {k: v.clone() for k, v in target.state_dict().items()}
    assert load_dit_state_dict(target, full) == "full"


def _options(mode, **extra):
    raw = {"text_attention_backend": "native" if mode == GATED_REGION_MODE else "sdpa", "strength": 0,
           "layer_indices": [0, 3], "timestep_range": [900, 1000], "mode": mode, **extra}
    return raw


def test_options_require_exact_keys_per_mode_and_round_trip():
    region = {"region_dilation": 0.1, "onset_latent_frames": 3}
    module = {"region_module": {"fourier_frequencies": 8, "learning_rate": 1e-4}}
    for raw in (_options(SPATIAL_LAYOUT_MODE), _options(SUPPRESSION_MODE, **region),
                _options(GATED_REGION_MODE, **region, **module)):
        options = SpatialLayoutOptions.from_mapping(raw)
        assert SpatialLayoutOptions.from_mapping(options.to_mapping()) == options
    with pytest.raises(ValueError, match="explicit mode"):
        SpatialLayoutOptions.from_mapping({k: v for k, v in _options(SPATIAL_LAYOUT_MODE).items() if k != "mode"})
    with pytest.raises(ValueError, match="requires exactly"):
        SpatialLayoutOptions.from_mapping(_options(SUPPRESSION_MODE))
    with pytest.raises(ValueError, match="text_attention_backend"):
        SpatialLayoutOptions.from_mapping(_options(GATED_REGION_MODE, **region, **module) | {"text_attention_backend": "sdpa"})


def test_attach_layout_modules_only_for_gated_mode():
    model = _tiny_dit()
    attach_layout_modules(model, SpatialLayoutOptions.from_mapping(
        _options(SUPPRESSION_MODE, region_dilation=0.1, onset_latent_frames=3)))
    assert not region_parameter_names(model)
    attach_layout_modules(model, SpatialLayoutOptions.from_mapping(_options(
        GATED_REGION_MODE, region_dilation=0.1, onset_latent_frames=3,
        region_module={"fourier_frequencies": 2, "learning_rate": 1e-4}) | {"layer_indices": [1]}))
    assert hasattr(model.blocks[1], "region_attn") and not hasattr(model.blocks[0], "region_attn")


def test_box_jitter_is_deterministic_bounded_and_shared_by_an_entitys_anchors():
    jitter = require_box_jitter({"center_shift": 0.1, "scale_min": 0.8, "scale_max": 1.25})
    layouts = ({"text": "t", "entity_spans": [], "entities": [{"name": "Character 1", "anchors": [
        {"time": 0.0, "visibility": "visible", "box": [0.2, 0.0, 0.6, 1.0]},
        {"time": 1.0, "visibility": "visible", "box": [0.2, 0.0, 0.6, 1.0]}]}]},)
    first = jitter_layout_entity_boxes(layouts, (0,), jitter=jitter, training_seed=42, sample_index=7)
    again = jitter_layout_entity_boxes(layouts, (0,), jitter=jitter, training_seed=42, sample_index=7)
    other = jitter_layout_entity_boxes(layouts, (0,), jitter=jitter, training_seed=42, sample_index=8)
    assert first == again and first != other
    a, b = (anchor["box"] for anchor in first[0]["entities"][0]["anchors"])
    assert a == b and all(0 <= v <= 1 for v in a) and a[0] < a[2] and a[1] < a[3]
    width = a[2] - a[0]
    assert 0.4 * 0.8 - 1e-9 <= width <= 0.4 * 1.25 + 1e-9
    with pytest.raises(ValueError, match="center_shift"):
        require_box_jitter({"center_shift": 0.6, "scale_min": 0.8, "scale_max": 1.25})


def test_region_channel_receives_gradients_even_outside_the_noise_window():
    """各卡用没用到新通道不一致会让 ZeRO 归约互等；窗口外也要有（为 0 的）梯度，输出不变。"""
    condition = _condition(GATED_REGION_MODE, (_shot(9), _shot(5, token=2)), layers=(1,), window=(900.0, 1000.0))
    original = _tiny_dit()
    gated = copy.deepcopy(original)
    attach_region_attention(gated, layer_indices=(1,), fourier_frequencies=2)
    inputs = _inputs(condition, "native") | {"timestep": torch.tensor([500.0])}
    expected = model_fn_wan_video(dit=original, **(_inputs(None, "native") | {"timestep": torch.tensor([500.0])})).noise
    actual = model_fn_wan_video(dit=gated, **inputs).noise
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    names = region_parameter_names(gated)
    grads = {n: p.grad for n, p in gated.named_parameters() if n in names}
    assert all(g is not None and torch.count_nonzero(g) == 0 for g in grads.values())
