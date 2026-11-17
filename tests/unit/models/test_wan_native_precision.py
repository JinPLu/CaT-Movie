"""Wan 的局部 FP32 算术、BF16 kernel 与选择性参数存储边界。"""
import copy
import pytest
import torch
import torch.nn.functional as F

from tests.unit.models.test_memory_prefix_attention import _tiny_dit
from tests.unit.training.test_memory_prefix_training import _FakePipe, _shared_inputs
from diffsynth.models import wan_video_dit
from diffsynth.pipelines.wan_video import TIME_MLP_ROWS, model_fn_wan_video
from mspc.training.backends.wan22 import flow_match_sft_loss
from mspc.models.wan22_precision import apply_wan22_parameter_storage, protected_parameter_names


@pytest.mark.parametrize("latent_dtype", [torch.float32, torch.bfloat16])
def test_native_forward_keeps_source_parameters_and_protected_paths_fp32(latent_dtype):
    dit = _tiny_dit()
    original = {name: value.detach().clone() for name, value in dit.named_parameters()}
    assert any(not torch.equal(value, value.bfloat16().float()) for value in original.values())
    observed = {}
    def record(name):
        def hook(module, args, output):
            observed[name] = (args[0].detach().clone(), output.detach().clone())
        return hook
    modules = {
        "patch": dit.patch_embedding, "time": dit.time_embedding,
        "time_projection": dit.time_projection, "q": dit.blocks[0].self_attn.q,
        "gate": dit.blocks[0].gate, "head": dit.head.head,
    }
    handles = [module.register_forward_hook(record(name)) for name, module in modules.items()]
    timestep = torch.tensor([997.5])
    assert float(timestep.bfloat16()) != float(timestep)
    latents = torch.randn(1, 48, 3, 4, 4).to(latent_dtype).requires_grad_()
    context = torch.randn(1, 7, 4096).bfloat16()
    output = model_fn_wan_video(dit, latents=latents, context=context, timestep=timestep).noise
    for handle in handles:
        handle.remove()
    half = dit.freq_dim // 2
    angles = timestep.double()[:, None] * (10000 ** (-torch.arange(half).double() / half))[None]
    expected_time = torch.cat((angles.cos(), angles.sin()), dim=1).float()
    rows = 3 * -(-TIME_MLP_ROWS // 3)  # 每个潜帧的时间步重复到 TIME_MLP_ROWS 行，不按 token 数
    assert torch.equal(observed["time"][0], expected_time[:, None].expand(1, rows, -1))
    assert observed["time"][1].shape == (1, rows, dit.dim)
    assert observed["time_projection"][1].shape == (1, rows, 6 * dit.dim)
    assert observed["patch"][1].dtype == observed["q"][1].dtype == torch.bfloat16
    for name in ("time", "time_projection", "gate", "head"):
        assert observed[name][1].dtype == torch.float32
    assert observed["head"][0].dtype == output.dtype == torch.float32
    output.square().mean().backward()
    for name, parameter in dit.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter.detach(), original[name])
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
    assert latents.grad is not None and torch.isfinite(latents.grad).all()


def test_time_mlp_runs_on_fixed_rows_per_frame_instead_of_every_token():
    """时间 MLP 不再按 1,170 个 token 算，而是每个潜帧的时间步重复到 TIME_MLP_ROWS 行（三种卡上与按 token 算逐位相同，
    11-17）；tiny参数避免把CPU测试当5B验收。"""
    dit = _tiny_dit()
    inputs = []
    handle = dit.time_embedding.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            model_fn_wan_video(dit, latents=torch.zeros(1, 48, 3, 30, 52),
                               context=torch.zeros(1, 7, 4096), timestep=torch.tensor([500.]),
                               fuse_vae_embedding_in_latents=False, prefix_frames=0).noise
    finally:
        handle.remove()
    assert len(inputs) == 1 and inputs[0].shape == (1, 3 * -(-TIME_MLP_ROWS // 3), dit.freq_dim)
    assert torch.equal(inputs[0], inputs[0][:, :1].expand_as(inputs[0]))


def test_head_matches_official_fp32_equation_inside_bf16_autocast():
    torch.manual_seed(7)
    head = wan_video_dit.Head(24, 48, (1, 2, 2), 1e-6)
    x = torch.randn(1, 5, 24).bfloat16()
    t = torch.randn(1, 5, 24)
    shift, scale = (head.modulation.unsqueeze(0) + t.unsqueeze(2)).chunk(2, dim=2)
    norm = F.layer_norm(x.float(), (24,), eps=1e-6)
    expected = F.linear(norm * (1 + scale.squeeze(2)) + shift.squeeze(2),
                        head.head.weight, head.head.bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head(x, t)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_attention_kernel_uses_value_dtype_and_restores_query_dtype(monkeypatch):
    for flag in ("FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(wan_video_dit, flag, False)
    seen = []
    real = F.scaled_dot_product_attention
    def attention(q, k, v):
        seen.append((q.dtype, k.dtype, v.dtype))
        return real(q, k, v)
    monkeypatch.setattr(wan_video_dit.F, "scaled_dot_product_attention", attention)
    q, k = torch.randn(1, 5, 24), torch.randn(1, 5, 24)
    v = torch.randn(1, 5, 24).bfloat16()
    output = wan_video_dit.flash_attention(q, k, v, 2)
    assert seen == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]
    assert output.dtype == torch.float32


def test_training_timestep_preserves_scheduler_fp32_even_with_bf16_latents(monkeypatch):
    pipe = _FakePipe()
    pipe.torch_dtype = torch.bfloat16
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))
    clean = torch.randn(1, 2, 5, 2, 8).bfloat16()
    flow_match_sft_loss(pipe, **_shared_inputs(clean, prefix_frames=0), reference_noise_sigma_max=0.0)
    actual = pipe.model_fn_calls[0]["timestep"]
    assert actual.dtype == torch.float32
    assert torch.equal(actual, pipe.scheduler.timesteps[2:3])


def test_selective_storage_preserves_protected_values_and_native_amp_forward():
    reference = _tiny_dit()
    candidate = copy.deepcopy(reference)
    protected = protected_parameter_names(candidate)
    original = {name: value.detach().clone() for name, value in candidate.named_parameters()}
    apply_wan22_parameter_storage(candidate)
    for name, parameter in candidate.named_parameters():
        dtype = torch.float32 if name in protected else torch.bfloat16
        assert parameter.dtype == dtype
        assert torch.equal(parameter, original[name].to(dtype))
    latents = torch.randn(1, 48, 3, 4, 4)
    context = torch.randn(1, 7, 4096).bfloat16()
    kwargs = dict(latents=latents, context=context, timestep=torch.tensor([997.5]))
    expected = model_fn_wan_video(reference, **kwargs).noise
    actual = model_fn_wan_video(candidate, **kwargs).noise
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    for parameter in candidate.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.dtype == parameter.dtype
