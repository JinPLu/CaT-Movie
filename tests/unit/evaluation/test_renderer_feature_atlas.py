from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import SelfAttention, CrossAttention
from mspc.evaluation.renderer_diagnostics import (
    DiagnosticStep, diagnostic_steps, PositiveTextSchedule, self_attention_key_mask,
    cross_attention_token_mask, activation_patch, pixel_masks_to_token_indices,
    pool_block_features, atlas_scheduler, noisy_target, save_atlas_features,
)


def fixture_modules():
    torch.manual_seed(21)
    dit = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SelfAttention(24, 2).double(),
                                                 cross_attn=CrossAttention(24, 2).double()) for _ in range(2)])
    state = DiagnosticStep()
    state.step, state.branch = 3, "positive"
    x = torch.randn(1, 8, 24, dtype=torch.float64)
    y = torch.randn(2, 6, 24, dtype=torch.float64)
    freqs = torch.polar(torch.ones(8, 1, 6, dtype=torch.float64), torch.randn(8, 1, 6, dtype=torch.float64))
    return dit, state, x, y, freqs


@pytest.mark.parametrize("branch", ["positive", "negative"])
def test_self_key_mask_empty_is_exact_and_masked_values_have_no_effect(branch):
    dit, state, x, _, freqs = fixture_modules()
    state.branch = branch
    attn = dit.blocks[0].self_attn
    expected = attn(x, freqs)
    with self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=[([0, 1], [])]):
        assert torch.equal(expected, attn(x, freqs))
    with self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=[([0, 1], [4, 5])]):
        before = attn(x, freqs)
        changed = x.clone()
        changed[:, 4:6] += 19
        assert torch.equal(before[:, :2], attn(changed, freqs)[:, :2])
        assert not torch.equal(before[:, :2], expected[:, :2])
    assert torch.equal(expected, attn(x, freqs))
    assert "forward" not in attn.__dict__


@pytest.mark.parametrize("kind", ["self", "cross"])
def test_mask_layer_and_step_windows(kind):
    dit, state, x, y, freqs = fixture_modules()
    call = (lambda i: dit.blocks[i].self_attn(x, freqs)) if kind == "self" else (
        lambda i: dit.blocks[i].cross_attn(x, y, [(0, 4), (4, 8)]).hidden_states)
    expected = [call(i) for i in range(2)]
    context = self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=[([0, 1], [4])]) if kind == "self" else (
        cross_attention_token_mask(dit, state, layers=[0], steps=[3], shot_tokens={0: [1]}))
    with context:
        assert not torch.equal(expected[0], call(0))
        assert torch.equal(expected[1], call(1))
        state.step = 4
        assert torch.equal(expected[0], call(0))
    assert torch.equal(expected[0], call(0))


@pytest.mark.parametrize("routed", [True, False])
def test_text_mask_empty_and_blocked_values(routed):
    dit, state, x, y, _ = fixture_modules()
    ranges = [(0, 4), (4, 8)] if routed else None
    y = y if routed else y[:1]
    attn = dit.blocks[0].cross_attn
    expected = attn(x, y, ranges).hidden_states
    with cross_attention_token_mask(dit, state, layers=[0], steps=[3], shot_tokens={0: []}):
        assert torch.equal(expected, attn(x, y, ranges).hidden_states)
    with cross_attention_token_mask(dit, state, layers=[0], steps=[3], shot_tokens={0: [1, 3]}):
        before = attn(x, y, ranges).hidden_states
        changed = y.clone()
        changed[0, [1, 3]] += 23
        assert torch.equal(before, attn(x, changed, ranges).hidden_states)
        assert not torch.equal(expected, before)
        state.branch = "negative"
        assert torch.equal(expected, attn(x, y, ranges).hidden_states)
    assert torch.equal(expected, attn(x, y, ranges).hidden_states)


def test_invalid_masks_and_restore_after_exception():
    dit, state, x, y, freqs = fixture_modules()
    attn = dit.blocks[0].self_attn
    with pytest.raises(ValueError), self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=[]):
        attn(x, freqs, prefix_frames=1)
    assert "forward" not in attn.__dict__
    for groups in [([([0], list(range(8)))]), [([0], [8])], [([0], [1]), ([0], [2])]]:
        with pytest.raises(ValueError), self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=groups):
            attn(x, freqs)
    dit.blocks[1].self_attn.self_attention_policy = "holocine_first_frame"
    with pytest.raises(ValueError), self_attention_key_mask(dit, state, layers=[0], steps=[3], query_key_sets=[]):
        pass
    with pytest.raises(ValueError), cross_attention_token_mask(dit, state, layers=[0], steps=[3], shot_tokens={0: list(range(6))}):
        dit.blocks[0].cross_attn(x, y, [(0, 4), (4, 8)])


def test_activation_patch_is_branch_specific_and_leaves_other_tokens_untouched():
    dit = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Linear(4, 4).double() for _ in range(2)]))
    state, records = DiagnosticStep(), {}
    state.step, state.branch = 3, "positive"
    source, target = torch.randn(1, 6, 4, dtype=torch.float64), torch.randn(1, 6, 4, dtype=torch.float64)
    block = dit.blocks[0]
    expected = block(target)
    for branch, offset in [("positive", 0), ("negative", 10)]:
        state.branch = branch
        with activation_patch(dit, state, layers=[0], steps=[3], tokens=[1, 4], records=records, mode="record"):
            block(source + offset)
    for branch in ["positive", "negative"]:
        state.branch = branch
        with activation_patch(dit, state, layers=[0], steps=[3], tokens=[1, 4], records=records, mode="patch"):
            actual = block(target)
            assert torch.equal(actual[:, [1, 4]], records[(3, 0, branch)][1])
            assert torch.equal(actual[:, [0, 2, 3, 5]], expected[:, [0, 2, 3, 5]])
            state.step = 4
            assert torch.equal(block(target), expected)
            state.step = 3
    assert torch.equal(block(target), expected)


@pytest.mark.parametrize("cfg_scale", [1., 5.])
@pytest.mark.parametrize("switch", [0, 2, 50])
def test_schedule_endpoints_cfg_registration_and_negative_unchanged(cfg_scale, switch):
    scheduler, state, calls = atlas_scheduler(), DiagnosticStep(), []
    a, b, neg = [torch.tensor([v], dtype=torch.float64) for v in [2., 3., -1.]]
    def model_fn(**kw):
        calls.append((state.step, state.branch, kw["context"].clone()))
        return kw["context"] * 2
    pipe = SimpleNamespace(scheduler=scheduler, model_fn=model_fn)
    schedule = PositiveTextSchedule(a, b, switch_step=switch, total_steps=50)
    with diagnostic_steps(pipe, state, cfg_scale=cfg_scale, start_step=0, step_count=50, positive_schedule=schedule):
        for i, t in enumerate(scheduler.timesteps):
            actual = pipe.model_fn(timestep=t.reshape(1), context=a)
            assert torch.equal(actual, 2 * (a if i < switch else b))
            if cfg_scale != 1:
                assert torch.equal(pipe.model_fn(timestep=t.reshape(1), context=neg), 2 * neg)
    assert pipe.model_fn is model_fn
    assert state.step is None
    for i, (step, branch, _) in enumerate(calls):
        assert step == i // (1 if cfg_scale == 1 else 2)
        assert branch == ("positive" if cfg_scale == 1 or i % 2 == 0 else "negative")


def test_counter_errors_restore_original():
    pipe = SimpleNamespace(scheduler=atlas_scheduler(), model_fn=lambda **kw: kw["timestep"])
    original = pipe.model_fn
    state = DiagnosticStep()
    with pytest.raises(RuntimeError), diagnostic_steps(pipe, state, cfg_scale=5, start_step=0, step_count=1):
        pipe.model_fn(timestep=pipe.scheduler.timesteps[:1])
    assert pipe.model_fn is original
    with pytest.raises(ValueError), diagnostic_steps(pipe, state, cfg_scale=5, start_step=0, step_count=1):
        pipe.model_fn(timestep=pipe.scheduler.timesteps[1:2])
    assert pipe.model_fn is original


@pytest.mark.parametrize("step", [0, 2, 25, 49])
def test_flow_formula_matches_scheduler(step):
    z = torch.randn(1, 2, 3, 4, 6, dtype=torch.float64)
    xt, sigma = noisy_target(z, seed=123, step=step)
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(123), dtype=torch.float32).double()
    scheduler = atlas_scheduler()
    t = scheduler.timesteps[step]
    assert torch.equal(xt, scheduler.add_noise(z, eps, t))
    v = scheduler.training_target(z, eps, t)
    torch.testing.assert_close(xt - sigma * v, z, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(xt - sigma * v, scheduler.step(v, t, xt, to_final=True), rtol=0, atol=0)


def test_pixel_masks_sparse_first_frame_and_causal_bins():
    masks = torch.zeros(4, 64, 96, dtype=torch.bool)
    masks[0, 0, 0] = True       # frame 0 -> token 0
    masks[1, 32, 32] = True    # frame 4 -> token 6+4
    masks[2, 63, 95] = True    # frame 1 -> token 6+5
    masks[3, 0, 64] = True     # frame 8 -> token 12+2
    assert pixel_masks_to_token_indices(masks, [0, 4, 1, 8], num_frames=13) == [0, 10, 11, 14]
    assert pixel_masks_to_token_indices(masks[1:2], [4], num_frames=13) == [10]
    assert pixel_masks_to_token_indices(masks[:0], [], num_frames=13) == []
    assert pixel_masks_to_token_indices(torch.zeros_like(masks), [0, 4, 1, 8], num_frames=13) == []
    full = torch.ones(1, 480, 832, dtype=torch.bool)
    assert pixel_masks_to_token_indices(full, [0], num_frames=1) == list(range(15 * 26))
    with pytest.raises(ValueError):
        pixel_masks_to_token_indices(masks, [0, 0, 4, 8], num_frames=13)
    with pytest.raises(ValueError):
        pixel_masks_to_token_indices(masks, [0, 1, 4, 13], num_frames=13)


def test_pooling_and_safetensors_metadata(tmp_path):
    x = torch.arange(48, dtype=torch.float64).reshape(1, 12, 4)
    pooled = pool_block_features(x, (2, 2, 3), {"person": [0, 6]})
    assert torch.equal(pooled["center"], x[:, [4, 10]].float().mean(1))
    assert torch.equal(pooled["spatial_mean"], x.float().mean(1))
    assert torch.equal(pooled["entity.person"], x[:, [0, 6]].float().mean(1))
    path = tmp_path / "one.safetensors"
    save_atlas_features(path, pooled, {"model": "tiny", "layers": [0], "step": 2, "sigma": 0.8, "seed": 3, "token_grid": [2, 2, 3]})
    with safe_open(path, framework="pt") as f:
        assert f.get_tensor("center").dtype == torch.float16
        assert f.metadata()["token_grid"] == "[2, 2, 3]"
    with pytest.raises(FileExistsError):
        save_atlas_features(path, pooled, {})
    with pytest.raises(ValueError):
        pool_block_features(x, (2, 2, 3), {"missing": []})


def test_single_step_real_small_wan_cfg_and_hooks():
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    from mspc.evaluation.renderer_diagnostics import single_step_features
    torch.manual_seed(9)
    dit = WanModel(dim=24, in_dim=4, ffn_dim=48, out_dim=4, text_dim=8, freq_dim=8,
                   eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=2,
                   has_image_input=False, require_vae_embedding=False, require_clip_embedding=False).eval()
    scheduler = atlas_scheduler()
    pipe = SimpleNamespace(dit=dit, scheduler=scheduler, model_fn=model_fn_wan_video,
                           load_models_to_device=lambda models: None)
    # Native Wan 的时间/LayerNorm精度合同为FP32；端到端fixture遵守同一合同。
    xt = torch.randn(1, 4, 3, 4, 6)
    a, b = torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    step, cfg = 4, 5.
    kwargs = dict(dit=dit, latents=xt, timestep=scheduler.timesteps[step:step+1],
                  temporal_position_mode="global_time_native_rope_v1")
    with torch.no_grad():
        pos = pipe.model_fn(**kwargs, context=a).noise
        neg = pipe.model_fn(**kwargs, context=b).noise
    state = DiagnosticStep()
    call = dict(x_t=xt, positive=a, negative=b, cfg_scale=cfg, step=step, state=state,
                entity_tokens={"person": [1, 3]}, temporal_position_mode="global_time_native_rope_v1")
    tensors, metadata = single_step_features(pipe, **call)
    assert torch.equal(tensors["v"], neg + cfg * (pos - neg))
    assert torch.equal(tensors["x0"], xt - scheduler.sigmas[step] * tensors["v"])
    assert len(tensors) == 2 * 2 * 3 + 3
    assert metadata["token_grid"] == [3, 2, 3]
    assert pipe.scheduler is scheduler
    assert pipe.model_fn is model_fn_wan_video
    with self_attention_key_mask(dit, state, layers=[0, 1], steps=[step], query_key_sets=[]), cross_attention_token_mask(
            dit, state, layers=[0, 1], steps=[step], shot_tokens={0: []}):
        empty, _ = single_step_features(pipe, **call)
    assert all(torch.equal(value, empty[key]) for key, value in tensors.items())
    assert all(not block._forward_hooks for block in dit.blocks)


@pytest.mark.parametrize("command", ["extract", "intervene", "schedule"])
def test_manifest_rejects_missing_wrong_type_unknown_and_duplicate(tmp_path, command):
    import json
    from pydantic import ValidationError
    from mspc.evaluation.renderer_diagnostics import load_atlas_manifest
    unit = dict(id="one", latent=dict(path="/explicit/latent.safetensors", key="input_latents", frame_start=0, frame_end=3),
                prompt="a person", seed=42, step=2, entity_tokens={"person": [1, 2]})
    if command == "intervene":
        unit["intervention"] = dict(kind="self_mask", layers=[0], steps=[2], query_key_sets=[[[0], [1]]])
    if command == "schedule":
        unit = dict(id="one", prompts_a=["a person"], prompts_b=["a seated person"], frame_counts=[9], seed=42, switch_step=0)
    path = tmp_path / "manifest.json"
    def load(units):
        path.write_text(json.dumps(dict(model="Wan2.2-TI2V-5B", units=units)))
        return load_atlas_manifest(path, command)
    assert load([unit])[1][0].id == "one"
    for bad in [{k: v for k, v in unit.items() if k != "seed"}, {**unit, "seed": "42"}, {**unit, "extra": 3}]:
        with pytest.raises(ValidationError):
            load([bad])
    with pytest.raises(ValueError):
        load([unit, unit])
    with pytest.raises(ValueError):
        load([{**unit, "id": "../escape"}])
