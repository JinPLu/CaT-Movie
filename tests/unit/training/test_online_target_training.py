"""原视频目标与离线潜变量共用逐镜编码、参考路由和同一损失。"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mspc.models.wan22_loader import WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS
from mspc.training.backends.wan22 import Wan22DiffSynthTrainingModule
from tests.support.memory_prefix import INFERENCE_OFFSET
from tests.unit.training.test_memory_prefix_training import (
    ROUTED_OPTIONS,
    TINY_DIT_KWARGS,
    _end_to_end_module,
    _memory_rows,
)


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    """These tiny loss/gradient comparisons explicitly exercise CPU SDPA math."""
    from diffsynth.models import wan_video_dit

    for flag in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(wan_video_dit, flag, False)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


class _FrozenVideoVAE(nn.Module):
    """保留真实Wan2.1的通道/时空步长，记录独立镜头的编码调用。"""

    z_dim = 16
    upsampling_factor = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.frame_counts = []

    def encode(self, videos, *, device, **kwargs):
        video = torch.stack(videos) if isinstance(videos, list) else videos
        self.frame_counts.append(video.shape[2])
        values = video[:, :, ::4, ::8, ::8].mean(dim=1, keepdim=True)
        return (values * self.scale).repeat(1, self.z_dim, 1, 1, 1).to(device)


def _vae_pipe():
    return SimpleNamespace(
        vae=_FrozenVideoVAE(), device=torch.device("cpu"), torch_dtype=torch.float32,
        height_division_factor=16, width_division_factor=16,
        load_models_to_device=lambda _: None,
    )


PIPELINE = {"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)}


def _training_module(dit):
    module = _end_to_end_module(dit)
    module.memory_prefix_options = ROUTED_OPTIONS
    module.temporal_position_mode = "global_time_native_rope_v1"
    module.timestep_boundaries = (0.0, 1.0)
    module.pipeline_cfg = PIPELINE
    pipe = _vae_pipe()
    object.__setattr__(module, "reference_runtime", SimpleNamespace(
        pipe=pipe, device=pipe.device, dtype=pipe.torch_dtype, pipeline_cfg=PIPELINE))
    return module


def _target_video():
    # 真语料ci_vid:test:000000012的49/29帧分镜：独立编码21潜帧，整段仅20。
    return torch.arange(3 * 78 * 32 * 32).remainder(256).to(torch.uint8).view(3, 78, 32, 32)


def _sample():
    weights = torch.ones(1, 1, 21, 4, 4)
    weights[:, :, :, :2] = 2
    return {
        "target_storage": "video", "video": _target_video(),
        "flow_draw_seed": 20260920, "height": 32, "width": 32, "num_frames": 80,
        "target_shot_ids": [1, 2], "memory_frame_count": 2,
        "reference_images": torch.stack([
            torch.full((3, 32, 32), 31, dtype=torch.uint8),
            torch.full((3, 32, 32), 223, dtype=torch.uint8)]),
        "reference_offset": INFERENCE_OFFSET,
        "shot_groups": ((0, 1), (1, 2), (2, 51), (51, 80)),
        "latent_shot_groups": ((0, 1), (1, 2), (2, 15), (15, 23)),
        "reference_routing": [[1], [0, 1]],
        "prompt": _memory_rows(2) + ["Subject 1 holds the cup.", "Subject 2 takes the cup."],
        "target_loss_weights": weights,
    }


def test_online_target_matches_offline_independent_shots_loss_and_gradients():
    from diffsynth.models.wan_video_dit import WanModel

    torch.manual_seed(7)
    dit = WanModel(**{**TINY_DIT_KWARGS, "in_dim": 16, "out_dim": 16,
                     "seperated_timestep": False, "require_vae_embedding": True,
                     "require_clip_embedding": True})
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "holocine_first_frame"
    online, cached = _training_module(dit), _training_module(copy.deepcopy(dit))
    encoder = object.__new__(Wan22DiffSynthTrainingModule)
    nn.Module.__init__(encoder)
    object.__setattr__(encoder, "_pipe", _vae_pipe())
    encoder.component_mode = WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS
    encoder.pipeline_cfg = PIPELINE
    latents, _, _ = encoder.encode_for_cache({
        "video": _target_video(), "shot_groups": [[0, 49], [49, 78]],
        "latent_shot_groups": [[0, 13], [13, 21]]})
    assert latents.shape == (1, 16, 21, 4, 4)
    assert encoder.pipe.vae.frame_counts == [49, 29]
    cached_sample = _sample()
    del cached_sample["video"]
    cached_sample.update(target_storage="latents", input_latents=latents)
    cached_loss = cached(cached_sample)
    online_loss = online(_sample())
    assert torch.equal(online_loss, cached_loss)
    online_loss.backward()
    cached_loss.backward()
    for (name, actual), (expected_name, expected) in zip(
        online.dit.named_parameters(), cached.dit.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert actual.grad is not None and torch.isfinite(actual.grad).all(), name
        assert torch.equal(actual.grad, expected.grad), name
    assert online.reference_runtime.pipe.vae.frame_counts == [49, 29, 1, 1]
    assert cached.reference_runtime.pipe.vae.frame_counts == [1, 1]
    assert online.reference_runtime.pipe.vae.scale.grad is None
    assert not any("vae" in name for name, _ in online.named_parameters())
    call = online.pipe.model_fn_calls[-1]
    assert call["reference_routing"] == ((1,), (0, 1))
    assert call["prefix_frames"] == 2
    assert call["latent_shot_groups"] == _sample()["latent_shot_groups"]
    assert online.last_timing["target_vae_seconds"] > 0
    assert cached.last_timing["target_vae_seconds"] == 0


def test_online_single_shot_without_references_uses_native_scalar_time():
    from diffsynth.models.wan_video_dit import WanModel

    dit = WanModel(**{**TINY_DIT_KWARGS, "in_dim": 16, "out_dim": 16,
                     "seperated_timestep": False})
    module = _training_module(dit)
    sample = {
        "target_storage": "video", "video": _target_video()[:, :5],
        "flow_draw_seed": 11, "height": 32, "width": 32, "num_frames": 5,
        "memory_frame_count": 0, "target_shot_ids": [1],
        "reference_images": torch.empty(0, 3, 32, 32, dtype=torch.uint8),
        "prompt": ["Subject 1 walks."],
    }
    assert torch.isfinite(module(sample))
    assert module.reference_runtime.pipe.vae.frame_counts == [5]
    call = module.pipe.model_fn_calls[-1]
    assert call["prefix_frames"] == 0
    assert not call["fuse_vae_embedding_in_latents"]
    assert not dit.seperated_timestep


@pytest.mark.parametrize("storage,extra,message", [
    ("video", {"input_latents": torch.zeros(1)}, "must not carry cached"),
    ("latents", {}, "must not carry raw video"),
    ("unspecified", {}, "explicitly be video or latents"),
])
def test_training_rejects_ambiguous_target_storage(storage, extra, message):
    module = _training_module(nn.Linear(1, 1))
    sample = {**_sample(), "target_storage": storage, **extra}
    with pytest.raises(ValueError, match=message):
        module(sample)


def test_online_vae_load_skips_cache_contract_preserves_rng_and_stays_unregistered(monkeypatch):
    from mspc.data import cache_provenance
    from mspc.models import wan22_loader

    module = object.__new__(Wan22DiffSynthTrainingModule)
    nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.model_cfg = {"paths": {"vae": "/explicit/wan21-vae.pth"}}
    module.pipeline_cfg = PIPELINE
    calls = []

    def load(model_cfg, pipeline_cfg, device):
        calls.append((model_cfg, pipeline_cfg, device))
        torch.rand(3)  # 构造VAE不能推进后续flow-matching的RNG。
        return SimpleNamespace(pipe=SimpleNamespace(vae=nn.Linear(1, 1)))

    def forbidden_contract(*args, **kwargs):
        pytest.fail("online targets must not consult a latent-cache contract")

    monkeypatch.setattr(wan22_loader, "load_vae_only_runtime", load)
    monkeypatch.setattr(cache_provenance, "load_cache_contract", forbidden_contract)
    state = torch.get_rng_state().clone()
    module.load_frozen_vae(None)
    assert torch.equal(torch.get_rng_state(), state)
    assert calls == [(module.model_cfg, PIPELINE, module.device)]
    assert module.reference_encoding is None
    assert not module.reference_runtime.pipe.vae.training
    assert all(not parameter.requires_grad for parameter in module.reference_runtime.pipe.vae.parameters())
    assert list(module.named_parameters()) == []


def test_cached_reference_vae_preserves_pixel_conversion_in_encoding_contract(monkeypatch, tmp_path):
    from mspc.data.cache_provenance import load_cache_contract
    from mspc.models import wan22_loader

    module = object.__new__(Wan22DiffSynthTrainingModule)
    nn.Module.__init__(module)
    module.device = torch.device("cpu")
    module.model_cfg = {"paths": {"vae": "/explicit/wan22-vae.pth"}}
    module.pipeline_cfg = PIPELINE
    provenance = tmp_path / "cache-provenance.json"
    provenance.write_text(json.dumps({
        "cache_version": "selected-shot-cache", "models": {
            "vae": {"path": module.model_cfg["paths"]["vae"], "revision": "wan22-vae-revision"}},
        "prompt_template_revision": "none", "implementation_revision": "actual-producer-revision",
    }))
    monkeypatch.setattr(wan22_loader, "load_vae_only_runtime", lambda *_: SimpleNamespace(
        pipe=SimpleNamespace(vae=nn.Linear(1, 1))))

    module.load_frozen_vae(provenance)

    contract = load_cache_contract(provenance, module.model_cfg["paths"], PIPELINE)
    assert module.reference_encoding == ("wan22-vae-revision", {
        key: contract["encoding"][key]
        for key in ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")})
    assert module.reference_encoding[1]["pixel_conversion"] == "uint8_to_bfloat16_mul_2_over_255_add_minus1"
