"""本地 flow-matching SFT loss 与上游 ``FlowMatchSFTLoss`` 逐位相同。

本文件在相同时间步、Gaussian 噪声与输入上比较本地实现和未改动的上游参照：
loss、交给 ``model_fn`` 的加噪潜变量必须逐位相同。抽样改为独立的显式样本随机流。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from unittest.mock import patch

from mspc.training.backends.wan22 import flow_match_sft_loss
from mspc.models.source_identity import WanVideoPrediction


# ---------------------------------------------------------------------------
# 上游 FlowMatchSFTLoss 的参照拷贝
#
# 2026-08-24 只读抄自共享盘物化运行时
# ``${MSPC_SHARED_ROOT}/Storyboard/.runtime/wan22/diffsynth/diffusion/loss.py``。
# 函数体逐字保留（含本项目取不到键的 lora / first_frame_latents / noise_scale 三条
# 分支），只去掉了 ``BasePipeline`` 的类型注解——本机没有 diffsynth 包。
#
# 这是等价单测的参照物：**不要"顺手整理"它**，改一个字这条测试就不再证明
# 本地 loss 与上游一致。
# ---------------------------------------------------------------------------
def upstream_flow_match_sft_loss(pipe, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


DIM = 8
LATENT_SHAPE = (1, 2, 2, 2, DIM)


class _WanLikeScheduler:
    """上游 ``FlowMatchScheduler`` 在 Wan 模板（shift=5）下被 loss 用到的那几个方法。"""

    def __init__(self, steps: int = 8) -> None:
        raw = torch.linspace(1.0, 0.0, steps + 1)[:-1]
        self.sigmas = 5 * raw / (1 + 4 * raw)
        self.timesteps = self.sigmas * 1000
        self.weights = torch.linspace(0.5, 1.5, steps)

    def _index(self, timestep: torch.Tensor) -> int:
        return int(torch.argmin((self.timesteps - timestep.detach().cpu()).abs()))

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[self._index(timestep)]
        return (1 - sigma) * sample + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.weights[self._index(timestep)]


class _TinyDiT(nn.Module):
    """``model_fn_wan_video`` 打了 patch 之后的同形替身：只有视频行的一层 FFN。"""

    def __init__(self) -> None:
        super().__init__()
        self.dim = DIM
        self.ffn = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.video_scale = nn.Parameter(torch.ones(()))

    def forward(self, *, latents: torch.Tensor, **_: object) -> torch.Tensor:
        return self.ffn(latents) * self.video_scale


class _FakePipe:
    text_encoder = None
    vae = None

    def __init__(self, dit: nn.Module) -> None:
        self.dit = dit
        self.in_iteration_models = ("dit",)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.scheduler = _WanLikeScheduler()
        self.model_fn_calls: list[dict[str, object]] = []

    def model_fn(self, dit: nn.Module, **inputs: object):
        self.model_fn_calls.append(inputs)
        return WanVideoPrediction(dit(**inputs), None)


def _legacy_loss_with_matching_draws(reference, pipe, inputs):
    """只匹配参照的抽样和返回类型；保留参照函数的全部 loss 算术。"""
    inputs = dict(inputs)
    seed = inputs.pop("flow_draw_seed")
    model_fn = pipe.model_fn
    noise_generator = torch.Generator().manual_seed(seed ^ 0x5DEECE66D)
    def noise_like(value):
        return torch.randn(value.shape, generator=noise_generator, dtype=torch.float32).to(value)
    with torch.random.fork_rng(devices=[]), patch.object(
        pipe, "model_fn", lambda **kwargs: model_fn(**kwargs).noise
    ), patch.object(torch, "randn_like", noise_like):
        torch.manual_seed(seed)
        return reference(pipe, **inputs)


def _shared_inputs(
    input_latents: torch.Tensor,
    *,
    temporal_position_mode: str = "global_time_msm_fixed_shot_phase_v1",
) -> dict[str, object]:
    return {
        "flow_draw_seed": 20260824,
        "input_latents": input_latents,
        "context": torch.zeros(2, 4, DIM),
        "latent_shot_groups": [[0, 1], [1, 2]],
        "temporal_position_mode": temporal_position_mode,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
        # 非参考帧臂的干净前缀宽度是 0。写出来而不是留给默认值：这个数字由
        # ``_prepare_inputs*`` 显式声明，loss 直接按键取，缺键即抛。
        "prefix_frames": 0,
    }


@pytest.mark.parametrize(
    "temporal_position_mode",
    ["global_time_native_rope_v1", "global_time_msm_fixed_shot_phase_v1"],
)
def test_local_loss_is_bit_exact_with_upstream_flow_match_sft_loss(
    temporal_position_mode: str,
) -> None:
    pipe = _FakePipe(_TinyDiT())
    input_latents = torch.randn(*LATENT_SHAPE)

    torch.manual_seed(20260824)
    local = flow_match_sft_loss(
        pipe,
        reference_noise_sigma_max=0.0,
        **_shared_inputs(input_latents, temporal_position_mode=temporal_position_mode),
    )
    torch.manual_seed(20260824)
    upstream = _legacy_loss_with_matching_draws(upstream_flow_match_sft_loss, pipe,
        _shared_inputs(input_latents, temporal_position_mode=temporal_position_mode))

    assert local.ndim == 0 and torch.isfinite(local) and float(local.detach()) > 0.0
    torch.testing.assert_close(local, upstream, rtol=0.0, atol=0.0)
    local_call, upstream_call = pipe.model_fn_calls
    torch.testing.assert_close(
        local_call["latents"], upstream_call["latents"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        local_call["timestep"], upstream_call["timestep"], rtol=0.0, atol=0.0
    )


def test_local_loss_leaves_the_upstream_kwargs_untouched() -> None:
    pipe = _FakePipe(_TinyDiT())
    inputs = _shared_inputs(torch.randn(*LATENT_SHAPE))

    flow_match_sft_loss(pipe, **inputs, reference_noise_sigma_max=0.0)

    call = pipe.model_fn_calls[-1]
    assert call["latent_shot_groups"] == [[0, 1], [1, 2]]
    assert call["temporal_position_mode"] == "global_time_msm_fixed_shot_phase_v1"


def test_the_drawn_timestep_is_reported_without_touching_the_loss() -> None:
    """逐样本账按时间步分档要用这一步抽到的时间步；报出来不消耗随机数，loss 与不报时逐位相同。"""
    latents = torch.randn(*LATENT_SHAPE)
    torch.manual_seed(5)
    plain = flow_match_sft_loss(_FakePipe(_TinyDiT()), **_shared_inputs(latents.clone()), reference_noise_sigma_max=0.0)
    torch.manual_seed(5)
    pipe, drawn = _FakePipe(_TinyDiT()), {}
    reported = flow_match_sft_loss(pipe, drawn=drawn, **_shared_inputs(latents.clone()), reference_noise_sigma_max=0.0)
    assert torch.equal(plain, reported)
    assert drawn["timestep"] in [float(t) for t in pipe.scheduler.timesteps]
    assert len(drawn["target_frame_mse"]) == LATENT_SHAPE[2]
    assert sum(drawn["target_frame_mse"]) / LATENT_SHAPE[2] == pytest.approx(drawn["unweighted_mse"])
    assert float(reported.detach()) == pytest.approx(drawn["unweighted_mse"] * drawn["loss_weight"])
    assert drawn["target_weighted_mse"] == drawn["unweighted_mse"]
    assert drawn["uniform_loss"] == float(reported.detach())


def test_uniform_loss_diagnostic_uses_the_same_noise_and_timestep_as_weighted_loss() -> None:
    latents = torch.randn(*LATENT_SHAPE)
    weights = torch.ones(1, 1, *LATENT_SHAPE[2:])
    weights[..., :DIM // 2] = 2
    torch.manual_seed(17)
    plain_pipe = _FakePipe(_TinyDiT())
    plain = flow_match_sft_loss(plain_pipe, **_shared_inputs(latents.clone()), target_loss_weights=weights, reference_noise_sigma_max=0.0)
    plain.backward()
    plain_gradients = {name: p.grad.clone() for name, p in plain_pipe.dit.named_parameters()}
    torch.manual_seed(17)
    reported_pipe, drawn = _FakePipe(_TinyDiT()), {}
    reported = flow_match_sft_loss(reported_pipe, drawn=drawn, **_shared_inputs(latents.clone()),
                                   target_loss_weights=weights, reference_noise_sigma_max=0.0)
    reported.backward()
    assert torch.equal(plain, reported)
    assert all(torch.equal(plain_gradients[name], p.grad) for name, p in reported_pipe.dit.named_parameters())
    assert drawn["target_weighted_mse"] != drawn["unweighted_mse"]
    assert drawn["uniform_loss"] == pytest.approx(drawn["unweighted_mse"] * drawn["loss_weight"])
    assert float(reported.detach()) == pytest.approx(drawn["target_weighted_mse"] * drawn["loss_weight"])


def test_explicit_flow_draw_is_independent_of_unrelated_global_rng_consumption() -> None:
    inputs = _shared_inputs(torch.randn(*LATENT_SHAPE))
    pipe = _FakePipe(_TinyDiT())
    state = torch.get_rng_state().clone()
    first = flow_match_sft_loss(pipe, **inputs, reference_noise_sigma_max=0.0)
    assert torch.equal(state, torch.get_rng_state())
    torch.randn(10003)
    second = flow_match_sft_loss(pipe, **inputs, reference_noise_sigma_max=0.0)
    assert torch.equal(first, second)
    for field in ("latents", "timestep"):
        assert torch.equal(pipe.model_fn_calls[0][field], pipe.model_fn_calls[1][field])


def test_source_alignment_has_separate_declared_weight_and_gradient():
    from mspc.models.source_identity import SourceIdentityOptions
    pipe = _FakePipe(_TinyDiT())
    inputs = _shared_inputs(torch.randn(*LATENT_SHAPE))
    flow = flow_match_sft_loss(pipe, **inputs, reference_noise_sigma_max=0.0)
    original = pipe.model_fn
    auxiliary = torch.tensor(2., requires_grad=True)
    pipe.dit.source_identity_options = SourceIdentityOptions(.01, 14, 64)
    pipe.model_fn = lambda **kwargs: WanVideoPrediction(original(**kwargs).noise, auxiliary)
    drawn = {}
    combined = flow_match_sft_loss(pipe, drawn=drawn, **inputs, reference_noise_sigma_max=0.0)
    torch.testing.assert_close(combined, flow + .01 * auxiliary, atol=0, rtol=0)
    combined.backward()
    assert auxiliary.grad.item() == pytest.approx(.01)
    assert drawn["source_alignment_loss"] == 2.
    assert drawn["weighted_source_alignment_loss"] == pytest.approx(.02)
