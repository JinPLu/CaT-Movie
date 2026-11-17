"""干净参考在层内与目标、其他参考全双向交互；逐镜文字CA独立路由。"""
from __future__ import annotations

from tests.support.memory_prefix import INFERENCE_OFFSET, full_window, offset_options
import sys
from pathlib import Path

import pytest
import torch

from mspc.core.wan22_contract import (
    MEMORY_ATTENTION_POLICIES,
    MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

TINY_DIT_KWARGS = dict(
    dim=24,
    num_heads=2,
    num_layers=2,
    in_dim=48,
    out_dim=48,
    ffn_dim=32,
    text_dim=4096,
    freq_dim=16,
    eps=1e-6,
    patch_size=(1, 2, 2),
    has_image_input=False,
    seperated_timestep=True,
    require_clip_embedding=False,
    require_vae_embedding=False,
)
MEMORY_FRAMES = 2
TARGET_FRAMES = 3
H, W = 8, 8


@pytest.fixture(autouse=True)
def cpu_math_attention_backend(monkeypatch):
    from diffsynth.models import wan_video_dit

    # These are CPU algebra/gradient contracts, including in a training
    # environment where the CUDA-only Flash/Sage packages are installed.
    for flag in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(wan_video_dit, flag, False)


def _tiny_dit():
    from diffsynth.models.wan_video_dit import WanModel

    torch.manual_seed(0)
    dit = WanModel(**TINY_DIT_KWARGS).eval()
    return dit


def _groups():
    return tuple((i, i + 1) for i in range(MEMORY_FRAMES)) + (
        (MEMORY_FRAMES, MEMORY_FRAMES + TARGET_FRAMES),
    )


def _forward(dit, latents, *, attention_policy, reference_offset=INFERENCE_OFFSET, window="full", timestep=500.0):
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    groups = _groups()
    torch.manual_seed(1)
    context = torch.randn(len(groups), 16, 4096)
    with torch.no_grad():
        return model_fn_wan_video(
            dit=dit,
            latents=latents,
            timestep=torch.tensor([timestep]),
            context=context,
            temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
            latent_shot_groups=groups,
            fuse_vae_embedding_in_latents=True,
            prefix_frames=MEMORY_FRAMES,
            reference_offset=reference_offset,
            memory_attention_policy=attention_policy,
            memory_attention_window=(None if attention_policy is None else full_window()) if window == "full" else window,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        ).noise


def _latents(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 48, MEMORY_FRAMES + TARGET_FRAMES, H, W, generator=generator)


def _perturb_target(latents: torch.Tensor) -> torch.Tensor:
    changed = latents.clone()
    changed[:, :, MEMORY_FRAMES:] += torch.randn_like(changed[:, :, MEMORY_FRAMES:])
    return changed


def _perturb_memory(latents: torch.Tensor) -> torch.Tensor:
    changed = latents.clone()
    changed[:, :, :MEMORY_FRAMES] += torch.randn_like(changed[:, :, :MEMORY_FRAMES])
    return changed


def test_memory_outputs_do_depend_on_the_target_under_bidirectional() -> None:
    dit = _tiny_dit()
    latents = _latents(2)
    base = _forward(dit, latents, attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1)
    moved = _forward(
        dit, _perturb_target(latents), attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1
    )

    assert not torch.allclose(base[:, :, :MEMORY_FRAMES], moved[:, :, :MEMORY_FRAMES])


def test_target_outputs_read_the_memory_under_bidirectional() -> None:
    dit = _tiny_dit()
    latents = _latents(3)
    base = _forward(dit, latents, attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1)
    moved = _forward(
        dit, _perturb_memory(latents), attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1
    )

    assert not torch.allclose(base[:, :, MEMORY_FRAMES:], moved[:, :, MEMORY_FRAMES:])


@pytest.mark.parametrize(
    ("reference_offset", "attention_policy", "message"),
    [
        (INFERENCE_OFFSET, None, "declare a memory prefix together"),
        (None, MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1, "declare a memory prefix together"),
        (INFERENCE_OFFSET, "memory_only", "memory_attention_policy must be one of"),
    ],
)
def test_model_fn_refuses_an_undeclared_or_unknown_attention_policy(
    reference_offset, attention_policy, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _forward(
            _tiny_dit(),
            _latents(5),
            attention_policy=attention_policy,
            reference_offset=reference_offset,
        )


def test_the_policy_names_are_one_table() -> None:
    from diffsynth.models.wan_video_dit import MEMORY_ATTENTION_POLICIES as VENDORED

    assert tuple(VENDORED) == tuple(MEMORY_ATTENTION_POLICIES)


def test_each_reference_reads_other_reference_images() -> None:
    dit = _tiny_dit()
    latents = _latents(6)
    changed = latents.clone()
    changed[:, :, 1] += 2.0
    base = _forward(dit, latents, attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1)
    moved = _forward(dit, changed, attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1)
    assert not torch.allclose(base[:, :, 0], moved[:, :, 0])
    assert not torch.allclose(base[:, :, 1], moved[:, :, 1])
    assert not torch.allclose(base[:, :, MEMORY_FRAMES:], moved[:, :, MEMORY_FRAMES:])


def test_each_reference_cross_attention_reads_only_its_own_source_row() -> None:
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    dit = _tiny_dit()
    # 一层SA在CA之前，排除下一层SA传播别行文字；本测试只检验CA读取集合。
    dit.blocks = torch.nn.ModuleList([dit.blocks[0]])
    latents = _latents(7)
    context = torch.randn(len(_groups()), 16, 4096)
    altered = context.clone()
    altered[1:] += 2.0
    args = dict(
        dit=dit, latents=latents, timestep=torch.tensor([500.0]),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        latent_shot_groups=_groups(), fuse_vae_embedding_in_latents=True,
        prefix_frames=MEMORY_FRAMES, reference_offset=INFERENCE_OFFSET,
        memory_attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1,
        memory_attention_window=full_window(),
    )
    with torch.no_grad():
        base = model_fn_wan_video(context=context, **args).noise
        moved = model_fn_wan_video(context=altered, **args).noise
    assert torch.equal(base[:, :, 0], moved[:, :, 0])
    assert not torch.allclose(base[:, :, 1:], moved[:, :, 1:])


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("memory_frames", [2, 10])
def test_reference_branch_receives_target_loss_gradients(checkpointing, memory_frames) -> None:
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    from mspc.training.backends.wan22 import Wan22DiffSynthTrainingModule
    from types import SimpleNamespace
    dit = _tiny_dit()
    module = Wan22DiffSynthTrainingModule.__new__(Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(module)
    module.dit = dit
    groups = module.optimizer_param_groups(SimpleNamespace())
    assert any(p is dit.patch_embedding.weight for group in groups for p in group["params"])
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    latents = torch.randn(1, 48, memory_frames + TARGET_FRAMES, H, W,
                          generator=torch.Generator().manual_seed(8)).requires_grad_()
    input_references = latents[:, :, :memory_frames].detach().clone()
    groups = tuple((j, j+1) for j in range(memory_frames)) + ((memory_frames, memory_frames+TARGET_FRAMES),)
    output = model_fn_wan_video(
        dit=dit, latents=latents, timestep=torch.tensor([500.0]),
        context=torch.randn(len(groups), 16, 4096),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        latent_shot_groups=groups, fuse_vae_embedding_in_latents=True,
        prefix_frames=memory_frames, reference_offset=INFERENCE_OFFSET,
        memory_attention_policy=MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1,
        memory_attention_window=full_window(),
        use_gradient_checkpointing=checkpointing,
    ).noise
    assert torch.equal(latents[:, :, :memory_frames].detach(), input_references)
    output[:, :, memory_frames:].square().mean().backward()
    assert latents.grad[:, :, :memory_frames].abs().sum() > 0
    before = dit.patch_embedding.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, dit.patch_embedding.weight)
