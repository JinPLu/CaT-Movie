"""实际后端构造入口按已加载组件装配，VAE-only缓存不访问DiT。"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/wan22_runtime"))
from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_ShapeChecker
from diffsynth.models.wan_video_dit import WanModel
from mspc.training.backends import wan22


class _FrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.bfloat16))
        self.frames = []

    def encode(self, videos, **kwargs):
        batch, channels, frames, height, width = videos.shape
        self.frames.append(frames)
        return torch.zeros(batch, 16, (frames - 1) // 4 + 1, height // 8, width // 8,
                           dtype=videos.dtype)


def _args():
    return SimpleNamespace(
        model_cfg={"temporal_position_mode": "global_time_native_rope_v1",
                   "diffsynth_root": "stubbed-at-loader-boundary", "paths": {},
                   "self_attention_policy": "full", "memory_prefix": None,
                   "spatial_layout": None, "planner_tokens": None, "source_identity": None},
        training_cfg={"scheduler_train_timesteps": 1000, "flow_shift": 5.0,
                      "activation_checkpointing": "recompute_all",
                      "dit_parameter_freeze": {"recipe": "full_dit_v1"}},
        pipeline_cfg={"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)},
        checkpoint_path=None,
    )


def test_actual_latents_only_cache_constructor_and_encode_need_only_frozen_vae(monkeypatch):
    from mspc.models import planner_tokens, region_attention

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipe.vae = _FrozenVAE()
    assert pipe.dit is None and pipe.text_encoder is None and pipe.tokenizer is None
    requested = []
    attachments = []

    def load(**kwargs):
        requested.append(kwargs["component_mode"])
        return pipe, WanVideoUnit_ShapeChecker

    monkeypatch.setattr(wan22, "load_pure_t2v_pipeline", load)
    monkeypatch.setattr(region_attention, "attach_layout_modules", lambda dit, options: attachments.append("layout"))
    monkeypatch.setattr(planner_tokens, "attach_planner_connector", lambda dit, options: attachments.append("planner"))
    model = wan22.build_cache_encoder(_args(), torch.device("cpu"), cache_layout="latents_only")
    assert requested == [wan22.WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS]
    assert attachments == []
    assert not hasattr(model, "dit")
    assert list(model.parameters()) == []  # 冻结VAE不注册为训练参数。
    assert not pipe.vae.training and all(not p.requires_grad for p in pipe.vae.parameters())
    # 真语料ci_vid:test:000000012的49/29帧切分，应逐镜编码21潜帧而非整段20。
    latent, context, metadata = model.encode_for_cache({
        "video": torch.zeros(3, 78, 32, 32, dtype=torch.uint8),
        "shot_groups": [[0, 49], [49, 78]], "latent_shot_groups": [[0, 13], [13, 21]]})
    assert pipe.vae.frames == [49, 29]
    assert latent.shape == (1, 16, 21, 4, 4) and latent.dtype == torch.bfloat16
    assert context is None and metadata["latent_shot_groups"] == ((0, 13), (13, 21))


@pytest.mark.parametrize("policy,cache_layers,num_layers,suffix_layers", [
    ("recompute_all", None, 1, None), ("save_self_attention", None, 30, None),
    ("save_self_attention", 20, 30, None), ("save_self_attention", None, 40, None),
    ("save_self_attention", 30, 40, 10),
])
def test_actual_training_constructor_installs_only_the_explicit_checkpoint_policy(
        monkeypatch, policy, cache_layers, num_layers, suffix_layers):
    from mspc.models import planner_tokens, region_attention
    from importlib import import_module

    checkpoint = import_module("diffsynth.core.gradient.gradient_checkpoint")
    selected = []
    def operator(device):
        selected.append(("attention operator", device))
        return torch.ops.aten.mm.default
    monkeypatch.setattr(checkpoint, "_self_attention_forward_operator", operator)

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipe.dit = WanModel(dim=24, in_dim=16, out_dim=16, ffn_dim=32, text_dim=4096,
                        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                        num_layers=num_layers, has_image_input=False)
    pipe.text_encoder = nn.Linear(1, 1)
    calls = []
    actual_layout, actual_planner = region_attention.attach_layout_modules, planner_tokens.attach_planner_connector

    def layout(dit, options):
        calls.append(("layout", dit, options))
        actual_layout(dit, options)

    def planner(dit, options):
        calls.append(("planner", dit, options))
        actual_planner(dit, options)

    monkeypatch.setattr(region_attention, "attach_layout_modules", layout)
    monkeypatch.setattr(planner_tokens, "attach_planner_connector", planner)
    monkeypatch.setattr(wan22, "load_pure_t2v_pipeline", lambda **_: (pipe, WanVideoUnit_ShapeChecker))
    args = _args()
    args.training_cfg["activation_checkpointing"] = policy
    args.training_cfg["self_attention_cache_layers"] = cache_layers
    args.training_cfg["no_recompute_suffix_layers"] = suffix_layers
    model = wan22.build_model(args, torch.device("cpu"))
    assert calls == [("layout", pipe.dit, None), ("planner", pipe.dit, None)]
    assert model.dit is pipe.dit and model.dit.source_identity_options is None
    assert pipe.dit.training and all(p.requires_grad for p in pipe.dit.parameters())
    assert not pipe.text_encoder.training and all(not p.requires_grad for p in pipe.text_encoder.parameters())
    assert set(model.state_dict()) == {"dit." + name for name in pipe.dit.state_dict()}
    assert selected == ([("attention operator", torch.device("cpu"))] if policy == "save_self_attention" else [])
    checkpointed = num_layers - (suffix_layers or 0)
    selected_layers = (checkpointed if cache_layers is None else cache_layers) if policy == "save_self_attention" else 0
    for index, block in enumerate(pipe.dit.blocks):
        assert (block.activation_checkpoint_context_fn is not None) == (checkpointed - selected_layers <= index < checkpointed)
        assert getattr(block, "_skip_activation_checkpointing", False) == (index >= checkpointed)
    assert all("forward" not in block.self_attn.__dict__ for block in pipe.dit.blocks)


def test_installed_training_policy_model_can_run_without_checkpointing_with_identical_math(monkeypatch):
    from importlib import import_module
    checkpoint = import_module("diffsynth.core.gradient.gradient_checkpoint")
    attention = import_module("diffsynth.models.wan_video_dit")
    # This is CPU mathematics even when the actual training environment has
    # flash-attn installed; its CUDA-only kernel cannot consume CPU tensors.
    for flag in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(attention, flag, False)
    monkeypatch.setattr(checkpoint, "_self_attention_forward_operator", lambda device: torch.ops.aten.mm.default)
    torch.manual_seed(25)
    dit = WanModel(dim=24, in_dim=2, out_dim=2, ffn_dim=32, text_dim=4,
                   freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                   num_layers=30, has_image_input=False)
    binding = checkpoint.install_self_attention_checkpointing(dit)
    x, context = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 2, 4)
    timestep = torch.tensor([250.0])
    outputs, gradients = [], []
    for enabled in (False, True):
        dit.zero_grad(set_to_none=True)
        prediction = dit(x, timestep, context, use_gradient_checkpointing=enabled).noise
        prediction.square().sum().backward()
        outputs.append(prediction.detach().clone())
        gradients.append({name: p.grad.clone() for name, p in dit.named_parameters()})
    assert torch.equal(*outputs)
    assert all(torch.equal(gradients[0][name], value) for name, value in gradients[1].items())
    binding.validate()
    binding.restore()
