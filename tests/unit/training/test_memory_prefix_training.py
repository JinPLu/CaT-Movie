"""段渲染器参考：干净输入、目标损失、零参考等价、时间步与逐镜接线。"""

from __future__ import annotations

from tests.support.memory_prefix import INFERENCE_OFFSET, full_window, noise_options, offset_options
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mspc.models.source_identity import WanVideoPrediction
from tests.unit.training.test_flow_match_loss import _legacy_loss_with_matching_draws

from mspc.core.wan22_contract import (
    MEMORY_ROW_POLICY_FIXED_ROLE_V1,
    MEMORY_ROW_ROLE_TEXT,
)
from mspc.models.memory_prefix_options import MemoryPrefixModelOptions
from mspc.models.wan22_loader import WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5
from mspc.training.backends.wan22 import (
    Wan22DiffSynthTrainingModule,
    flow_match_sft_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

DIM = 8
LATENT_FRAMES = 5
LATENT_SHAPE = (1, 2, LATENT_FRAMES, 2, DIM)
SEED = 20260920
OPTIONS = MemoryPrefixModelOptions(
    frames_max=4,
    dropout_p=0.2,
    noise=noise_options(),
    reference_offset=offset_options(),
    attention_policy="bidirectional_v1",
    row_policy=MEMORY_ROW_POLICY_FIXED_ROLE_V1,
    attention_window=full_window(),
    placement="ordered_slots",
)
def _memory_rows(count: int) -> list[str]:
    """每张参考只配同一个固定角色句。"""
    return [MEMORY_ROW_ROLE_TEXT] * count


# ---------------------------------------------------------------------------
# 本改动之前那四行的逐字参照
# ---------------------------------------------------------------------------
def prechange_flow_match_sft_loss(pipe, **inputs):
    """整段加噪、整段算 target、整段 MSE —— ``prefix_frames`` 出现之前的算术。

    上游 ``FlowMatchSFTLoss`` 的逐位参照物（含它自己的 ``first_frame_latents`` 分支）在
    ``tests/unit/training/test_story_token_training.py``，不在这里复制第二份；这条测试要
    钉的只是「M==0 时本地 loss 与改动前逐位相同」。
    """

    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
    )
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(
        inputs["input_latents"], noise, timestep
    )
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    return loss * pipe.scheduler.training_weight(timestep)


class _WanLikeScheduler:
    """上游 ``FlowMatchScheduler`` 在 Wan 模板（shift=5）下被 loss 用到的那几个方法。"""

    def __init__(self, steps: int = 8) -> None:
        raw = torch.linspace(1.0, 0.0, steps + 1)[:-1]
        self.sigmas = 5 * raw / (1 + 4 * raw)
        self.timesteps = self.sigmas * 1000
        self.weights = torch.linspace(0.5, 1.5, steps)

    def _index(self, timestep: torch.Tensor) -> int:
        return int(torch.argmin((self.timesteps - timestep.detach().cpu()).abs()))

    def add_noise(self, sample, noise, timestep):
        sigma = self.sigmas[self._index(timestep)]
        return (1 - sigma) * sample + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        return self.weights[self._index(timestep)]


class _ScaleDiT(nn.Module):
    """预测 = ``latents * scale``：loss 能手算，梯度能回到 ``latents``。"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, *, latents: torch.Tensor, **_: object) -> torch.Tensor:
        return latents * self.scale


class _FakePipe:
    text_encoder = None
    vae = None

    def __init__(self) -> None:
        self.dit = _ScaleDiT()
        self.in_iteration_models = ("dit",)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.scheduler = _WanLikeScheduler()
        self.model_fn_calls: list[dict[str, object]] = []

    def model_fn(self, dit: nn.Module, **inputs: object):
        self.model_fn_calls.append(inputs)
        return WanVideoPrediction(dit(**inputs), None)


def _shared_inputs(input_latents: torch.Tensor, *, prefix_frames: int) -> dict:
    return {
        "flow_draw_seed": SEED,
        "input_latents": input_latents,
        "context": torch.zeros(1, 4, DIM),
        "temporal_position_mode": "global_time_native_rope_v1",
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
        "prefix_frames": prefix_frames,
    }


# ---------------------------------------------------------------------------
# 1. M == 0 与改动前逐位相同
# ---------------------------------------------------------------------------


def test_a_zero_memory_prefix_reproduces_the_pre_change_loss_bit_for_bit() -> None:
    pipe = _FakePipe()
    input_latents = torch.randn(*LATENT_SHAPE)

    torch.manual_seed(SEED)
    local = flow_match_sft_loss(
        pipe, reference_noise_sigma_max=0.0, **_shared_inputs(input_latents, prefix_frames=0)
    )
    torch.manual_seed(SEED)
    prechange = _legacy_loss_with_matching_draws(prechange_flow_match_sft_loss,
        pipe, _shared_inputs(input_latents, prefix_frames=0))

    assert local.ndim == 0 and torch.isfinite(local) and float(local.detach()) > 0.0
    torch.testing.assert_close(local, prechange, rtol=0.0, atol=0.0)
    local_call, prechange_call = pipe.model_fn_calls
    # 在相同抽样上比较目标加噪与 loss 算术，不依赖全局 RNG 消费顺序。
    torch.testing.assert_close(
        local_call["latents"], prechange_call["latents"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        local_call["timestep"], prechange_call["timestep"], rtol=0.0, atol=0.0
    )


# ---------------------------------------------------------------------------
# 2. 记忆帧不加噪、不进 loss
# ---------------------------------------------------------------------------


def test_only_the_target_frames_are_noised_and_scored() -> None:
    """M=2：前两帧原样进 DiT，后三帧加噪；loss 只覆盖后三帧。"""

    memory_frames = 2
    pipe = _FakePipe()
    input_latents = torch.randn(*LATENT_SHAPE)

    torch.manual_seed(SEED)
    loss = flow_match_sft_loss(
        pipe,
        reference_noise_sigma_max=0.0,
        **_shared_inputs(input_latents, prefix_frames=memory_frames),
    )

    call = pipe.model_fn_calls[-1]
    latents = call["latents"]
    timestep = call["timestep"]
    assert latents.shape == input_latents.shape
    # 记忆帧逐位等于原样的干净潜帧。
    torch.testing.assert_close(
        latents[:, :, :memory_frames],
        input_latents[:, :, :memory_frames],
        rtol=0.0,
        atol=0.0,
    )
    # 目标帧是被加噪的那一段；手算同一段并逐位比较。
    target_latents = input_latents[:, :, memory_frames:]
    noise = torch.randn(target_latents.shape,
        generator=torch.Generator().manual_seed(SEED ^ 0x5DEECE66D))
    expected_noisy = pipe.scheduler.add_noise(target_latents, noise, timestep)
    torch.testing.assert_close(
        latents[:, :, memory_frames:], expected_noisy, rtol=0.0, atol=0.0
    )
    # loss 只覆盖目标帧：手算 mse(pred[:, :, M:], target)。
    training_target = pipe.scheduler.training_target(target_latents, noise, timestep)
    prediction = latents * pipe.dit.scale
    expected = torch.nn.functional.mse_loss(
        prediction[:, :, memory_frames:].float(), training_target.float()
    ) * pipe.scheduler.training_weight(timestep)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
    # 把记忆帧整段进 loss 的那个失效形态会给出别的数：确认两者确实不同，
    # 否则上面那条断言就是空的。
    whole = torch.nn.functional.mse_loss(
        prediction.float(),
        pipe.scheduler.training_target(input_latents, torch.randn_like(input_latents), timestep).float(),
    ) * pipe.scheduler.training_weight(timestep)
    assert not torch.isclose(loss, whole)


def test_a_clean_prefix_wider_than_the_sequence_raises() -> None:
    pipe = _FakePipe()
    with pytest.raises(ValueError, match="prefix_frames must be in"):
        flow_match_sft_loss(
            pipe,
            reference_noise_sigma_max=0.0,
            **_shared_inputs(torch.randn(*LATENT_SHAPE), prefix_frames=LATENT_FRAMES),
        )


def test_a_non_integer_clean_prefix_raises() -> None:
    pipe = _FakePipe()
    inputs = _shared_inputs(torch.randn(*LATENT_SHAPE), prefix_frames=0)
    inputs["prefix_frames"] = 2.0
    with pytest.raises(TypeError, match="prefix_frames must be an int"):
        flow_match_sft_loss(pipe, **inputs, reference_noise_sigma_max=0.0)


# ---------------------------------------------------------------------------
# 3. per-token timestep：原生 1 帧 → N 帧
# ---------------------------------------------------------------------------

TINY_DIT_KWARGS = dict(
    dim=24,
    num_heads=2,
    num_layers=1,
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
RUNTIME_FRAMES = 4
RUNTIME_H, RUNTIME_W = 8, 8
#: patch 2x2 之后每潜帧的 token 数，也就是 vendored 那两行里 ``H * W // 4``。
TOKENS_PER_FRAME = RUNTIME_H * RUNTIME_W // 4
TIMESTEP = 500.0


def _rows_per_frame(latent_frames: int) -> int:
    """时间 MLP 每个潜帧的时间步重复几行（``TIME_MLP_ROWS`` 行起，不按 token 数；11-17 三种卡逐位相同）。"""
    from diffsynth.pipelines.wan_video import TIME_MLP_ROWS

    return -(-TIME_MLP_ROWS // latent_frames)


def _tiny_dit():
    from diffsynth.models.wan_video_dit import WanModel

    dit = WanModel(**TINY_DIT_KWARGS)
    return dit


def _record_timesteps(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """记下 ``model_fn`` 真正送进 ``sinusoidal_embedding_1d`` 的那一串 timestep。"""

    import diffsynth.pipelines.wan_video as pipeline

    real = pipeline.sinusoidal_embedding_1d
    seen: list[torch.Tensor] = []

    def _spy(dim, position):
        seen.append(position.detach().clone())
        return real(dim, position)

    monkeypatch.setattr(pipeline, "sinusoidal_embedding_1d", _spy)
    return seen


def _run_model_fn(dit, *, latent_frames: int = RUNTIME_FRAMES, **overrides):
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    kwargs = dict(
        dit=dit,
        latents=torch.randn(
            1, TINY_DIT_KWARGS["in_dim"], latent_frames, RUNTIME_H, RUNTIME_W
        ),
        timestep=torch.tensor([TIMESTEP]),
        context=torch.zeros(1, 512, 4096),
        temporal_position_mode="global_time_native_rope_v1",
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )
    kwargs.update(overrides)
    return model_fn_wan_video(**kwargs)


def _expected_timestep_vector(prefix_frames: int, latent_frames: int) -> torch.Tensor:
    return torch.concat(
        [
            torch.zeros((prefix_frames, _rows_per_frame(latent_frames))),
            torch.ones((latent_frames - prefix_frames, _rows_per_frame(latent_frames))) * TIMESTEP,
        ]
    ).flatten()


def test_the_native_i2v_one_frame_timestep_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原生 I2V 是「第 0 帧置零、其余为 t」。这条测试把那一串写成改动前的字面表达式。

    N=1 的行为若变了，已有的 TI2V-5B I2V 推理就换了时间条件，而成片上不会立刻看出来。
    """

    seen = _record_timesteps(monkeypatch)
    _run_model_fn(
        _tiny_dit(),
        fuse_vae_embedding_in_latents=True,
        prefix_frames=1,
    )

    assert len(seen) == 1
    old_expression = torch.concat(
        [
            torch.zeros((1, _rows_per_frame(RUNTIME_FRAMES))),
            torch.ones((RUNTIME_FRAMES - 1, _rows_per_frame(RUNTIME_FRAMES))) * TIMESTEP,
        ]
    ).flatten()
    torch.testing.assert_close(seen[0], old_expression, rtol=0.0, atol=0.0)


def test_the_fused_image_unit_declares_its_own_one_frame_prefix() -> None:
    """写干净潜帧的那个单元自己报「写了几帧」。默认给 1 会让别的生产者静默错一帧。"""

    from diffsynth.pipelines.wan_video import WanVideoUnit_ImageEmbedderFused

    unit = WanVideoUnit_ImageEmbedderFused()
    assert "prefix_frames" in unit.output_params

    class _Vae:
        @staticmethod
        def encode(images, **_kwargs):
            return torch.zeros(1, TINY_DIT_KWARGS["in_dim"], 1, 1, 1)

    class _Image:
        @staticmethod
        def resize(size):
            return "resized"

    pipe = SimpleNamespace(
        dit=SimpleNamespace(fuse_vae_embedding_in_latents=True),
        vae=_Vae(),
        device=torch.device("cpu"),
        load_models_to_device=lambda names: None,
        preprocess_image=lambda image: torch.zeros(1, 3, 1, 1),
    )
    out = unit.process(
        pipe,
        _Image(),
        torch.zeros(1, TINY_DIT_KWARGS["in_dim"], 2, 1, 1),
        1,
        1,
        False,
        (1, 1, 1),
        (1, 1, 1),
    )
    assert out["prefix_frames"] == 1
    # 前缀时间步不再是一个可传的量：前缀 token 恒为 0，由 model_fn 按 prefix_frames 写。
    assert "prefix_timesteps" not in unit.output_params
    assert "prefix_timesteps" not in out
    assert out["fuse_vae_embedding_in_latents"] is True


def test_memory_frames_get_timestep_zero_and_the_target_gets_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M=2：前两帧的 token timestep 恒为 0（原生首帧口的 M 帧推广），其余为 t。"""

    seen = _record_timesteps(monkeypatch)
    _run_model_fn(
        _tiny_dit(),
        fuse_vae_embedding_in_latents=True,
        prefix_frames=2,
        reference_offset=INFERENCE_OFFSET,
        memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window(),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        latent_shot_groups=((0,1),(1,2),(2,4)),
        context=torch.zeros(3,512,4096),
    )

    assert len(seen) == 1
    torch.testing.assert_close(
        seen[0], _expected_timestep_vector(2, RUNTIME_FRAMES), rtol=0.0, atol=0.0
    )
    zero_tokens = int((seen[0] == 0.0).sum())
    assert zero_tokens == 2 * _rows_per_frame(RUNTIME_FRAMES)


@pytest.mark.parametrize('fused', [False, True])
def test_no_prefix_and_reference_dropout_use_native_per_token_timestep(monkeypatch, fused) -> None:
    seen = _record_timesteps(monkeypatch)
    _run_model_fn(_tiny_dit(), fuse_vae_embedding_in_latents=fused, prefix_frames=0)
    torch.testing.assert_close(seen[0], _expected_timestep_vector(0, RUNTIME_FRAMES), rtol=0, atol=0)


def test_non_separated_architecture_keeps_scalar_timestep(monkeypatch) -> None:
    seen = _record_timesteps(monkeypatch)
    dit = _tiny_dit()
    dit.seperated_timestep = False
    _run_model_fn(dit, fuse_vae_embedding_in_latents=False, prefix_frames=0)
    assert seen[0].shape == (1,)
    assert float(seen[0][0]) == TIMESTEP


def test_wan21_clean_reference_prefix_uses_zero_time_without_changing_native_architecture(monkeypatch):
    from diffsynth.models.wan_video_dit import WanModel

    seen = _record_timesteps(monkeypatch)
    dit = WanModel(**{**TINY_DIT_KWARGS, "in_dim": 16, "out_dim": 16,
                      "seperated_timestep": False, "require_vae_embedding": True,
                      "require_clip_embedding": True})
    latents = torch.randn(1, 16, RUNTIME_FRAMES, RUNTIME_H, RUNTIME_W, requires_grad=True)
    output = _run_model_fn(
        dit, latents=latents, fuse_vae_embedding_in_latents=True, prefix_frames=2,
        reference_offset=INFERENCE_OFFSET, memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window(), latent_shot_groups=((0, 1), (1, 2), (2, 4)),
        context=torch.randn(3, 5, 4096),
    ).noise
    torch.testing.assert_close(seen[0], _expected_timestep_vector(2, RUNTIME_FRAMES), rtol=0, atol=0)
    assert not dit.seperated_timestep
    assert output.shape == latents.shape
    output[:, :, 2:].square().mean().backward()
    assert torch.isfinite(latents.grad).all()
    for name, parameter in dit.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_a_clean_prefix_without_the_fused_flag_raises() -> None:
    """干净帧没有声明融合口就会被当成噪声去噪。这条必须炸，不能静默。"""

    with pytest.raises(ValueError, match="requires fuse_vae_embedding_in_latents"):
        _run_model_fn(_tiny_dit(), prefix_frames=2)


# ---------------------------------------------------------------------------
# 4. 参考零相位与目标镜相位
# ---------------------------------------------------------------------------


def _memory_groups(memory_frames: int, target_latent_frames: int):
    latent = tuple((i, i + 1) for i in range(memory_frames)) + (
        (memory_frames, memory_frames + target_latent_frames),
    )
    return latent


def test_shot_phase_index_sequence_without_a_prefix_is_the_native_enumeration() -> None:
    """没有干净前缀时必须是 ``range(K)``——这个参数出现之前 ``enumerate`` 给的那一串。"""

    from diffsynth.pipelines.wan_video import shot_phase_index_sequence

    groups = [[0, 2], [2, 4], [4, 6]]
    assert shot_phase_index_sequence(groups, 0, None) == (0, 1, 2)
    with pytest.raises(ValueError, match="without a memory prefix"):
        shot_phase_index_sequence(groups, 0, INFERENCE_OFFSET)


def test_shot_phase_index_sequence_puts_the_target_shot_at_zero() -> None:
    from diffsynth.pipelines.wan_video import shot_phase_index_sequence

    groups = _memory_groups(2, 3)
    assert (
        shot_phase_index_sequence(groups, 2, INFERENCE_OFFSET)
        == (0, 0, 0)
    )
    # 前缀必须落在组边界上：一个组里同时有干净帧和噪声帧时只能拿到一个相位。
    with pytest.raises(ValueError, match="each reference"):
        shot_phase_index_sequence(groups, 3, INFERENCE_OFFSET)
    with pytest.raises(ValueError, match="reference_offset must be a tuple"):
        shot_phase_index_sequence(groups, 2, (1, 15, 26))


def test_the_reference_phase_index_reaches_the_frequency_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """序号必须一路走到 ``shot_phase_adjusted_freqs``，不是算完就丢。"""

    import diffsynth.pipelines.wan_video as pipeline

    real = pipeline.shot_phase_adjusted_freqs
    seen: list[tuple[int, ...]] = []

    def _spy(*args, shot_phase_index, **kwargs):
        seen.append(tuple(shot_phase_index))
        return real(*args, shot_phase_index=shot_phase_index, **kwargs)

    monkeypatch.setattr(pipeline, "shot_phase_adjusted_freqs", _spy)

    latent_groups = _memory_groups(2, 2)
    _run_model_fn(
        _tiny_dit(),
        latent_frames=4,
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        latent_shot_groups=latent_groups,
        context=torch.zeros(len(latent_groups), 512, 4096),
        fuse_vae_embedding_in_latents=True,
        prefix_frames=2,
        reference_offset=INFERENCE_OFFSET,
        memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window(),
    )

    assert seen, "shot_phase_adjusted_freqs 没有被调用：相位这条路没接上"
    assert set(seen) == {(0, 0, 0)}


# ---------------------------------------------------------------------------
# 5. 训练模块的输入面
# ---------------------------------------------------------------------------

T5_LEN, T5_WIDTH = 512, 4096
EMPTY_ROW = 0.25


class _PromptUnit:
    name = "prompt"


class _Pipe(SimpleNamespace):
    """带一个 prompt 单元的假管线：把每个 prompt 编成一行可辨认的 context。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = [_PromptUnit()]
        self.encoded_prompts: list[object] = []

    def unit_runner(self, unit, pipe, shared, positive, negative):
        prompt = positive["prompt"]
        self.encoded_prompts.append(prompt)
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        # 每句独立编码、内容可辨认：同一句编出同一行，不同句编出不同行。
        context = torch.zeros(len(texts), T5_LEN, T5_WIDTH)
        for row, text in enumerate(texts):
            context[row, 0] = EMPTY_ROW + len(text)
        return shared, {**positive, "context": context}, negative


def _memory_module(*, options: MemoryPrefixModelOptions | None = OPTIONS):
    module = object.__new__(Wan22DiffSynthTrainingModule)
    nn.Module.__init__(module)
    object.__setattr__(
        module,
        "_pipe",
        _Pipe(device=torch.device("cpu"), torch_dtype=torch.float32),
    )
    module.device = torch.device("cpu")
    module.dtype = torch.float32
    module.temporal_position_mode = "global_time_msm_fixed_shot_phase_v1"
    module.timestep_boundaries = (0.0, 1.0)
    module.component_mode = WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5
    module.training_cfg = {"use_gradient_checkpointing": False, "seed": 42}
    module.memory_prefix_options = options
    module.last_timing = {}
    module.profile_sync_cuda = False
    module.reference_runtime = None
    return module


def _memory_sample(memory_frames: int, target_latent_frames: int = 2):
    latent_groups = _memory_groups(memory_frames, target_latent_frames)
    target_rgb = (target_latent_frames - 1) * 4 + 1
    raw_groups = tuple((i, i + 1) for i in range(memory_frames)) + (
        (memory_frames, memory_frames + target_rgb),
    )
    sample = {
        "target_storage": "latents",
        "flow_draw_seed": SEED,
        "input_latents": torch.randn(
            1, 2, target_latent_frames, 2, DIM
        ),
        "height": 480,
        "width": 832,
        "num_frames": memory_frames + target_rgb,
        "prompt": _memory_rows(memory_frames) + ["Story: G Now: shot"],
        "case_id": "case_a",
        "shot_id": 1,
        "epoch": 0,
        "memory_frame_count": memory_frames,
        "reference_images": torch.zeros(memory_frames, 3, 480, 832, dtype=torch.uint8),
        "segment_id": "segment_1",
        "memory_shot_ids": tuple(range(memory_frames)),
        "memory_mode": "past_nearest",
    }
    if memory_frames:
        sample["latent_shot_groups"] = latent_groups
        sample["shot_groups"] = raw_groups
        sample["reference_offset"] = list(INFERENCE_OFFSET)
    return sample


def test_prepare_latent_inputs_declares_the_clean_prefix_and_the_fused_flag() -> None:
    module = _memory_module()

    inputs = module._prepare_latent_inputs(_memory_sample(2))

    assert inputs["prefix_frames"] == 2
    assert inputs["fuse_vae_embedding_in_latents"] is True
    assert inputs["reference_offset"] == (
        INFERENCE_OFFSET
    )
    assert inputs["memory_attention_policy"] == "bidirectional_v1"
    assert inputs["memory_attention_window"] == full_window()
    assert inputs["latent_shot_groups"] == ((0, 1), (1, 2), (2, 4))
    assert module.last_timing["memory_frame_count"] == 2


def test_a_dropout_sample_declares_no_clean_prefix() -> None:
    module = _memory_module()

    inputs = module._prepare_latent_inputs(_memory_sample(0))

    assert inputs["prefix_frames"] == 0
    assert inputs["fuse_vae_embedding_in_latents"] is False
    # M==0 的样本没有 shot 组，相位策略也就无从谈起（运行时要求两者同在同缺）。
    assert "latent_shot_groups" not in inputs
    assert "reference_offset" not in inputs
    assert "memory_attention_policy" not in inputs
    assert "memory_attention_window" not in inputs


def test_a_non_memory_arm_declares_a_zero_prefix_and_no_fused_flag() -> None:
    module = _memory_module(options=None)

    inputs = module._prepare_latent_inputs(
        {
            "input_latents": torch.randn(1, 2, 3, 2, DIM),
            "height": 480,
            "width": 832,
            "num_frames": 9,
        }
    )

    assert inputs["prefix_frames"] == 0
    assert inputs["fuse_vae_embedding_in_latents"] is False


def test_a_memory_arm_sample_without_the_count_raises() -> None:
    module = _memory_module()
    sample = _memory_sample(2)
    del sample["memory_frame_count"]

    with pytest.raises(RuntimeError, match="must carry an int memory_frame_count"):
        module._prepare_latent_inputs(sample)


def test_a_non_memory_arm_refuses_a_sample_that_carries_the_count() -> None:
    module = _memory_module(options=None)

    with pytest.raises(RuntimeError, match="model.memory_prefix is"):
        module._prepare_latent_inputs(_memory_sample(2))


def test_the_live_t5_context_encodes_each_memory_row_and_the_target_row() -> None:
    """M个参考组各读固定角色句；目标组仍读其完整目标文字，不替换或截短。"""

    module = _memory_module()
    sample = _memory_sample(2)

    context = module._encode_live_t5_context(sample)

    assert context.shape == (3, T5_LEN, T5_WIDTH)
    rows = _memory_rows(2)
    assert module.pipe.encoded_prompts == [[*rows, "Story: G Now: shot"]]
    assert float(context[0, 0, 0]) == EMPTY_ROW + len(rows[0])
    assert float(context[1, 0, 0]) == EMPTY_ROW + len(rows[1])
    assert float(context[2, 0, 0]) == EMPTY_ROW + len("Story: G Now: shot")


def test_fixed_reference_role_cannot_be_cleared_as_caption_dropout() -> None:
    module = _memory_module()
    sample = _memory_sample(2)
    sample["prompt"][0] = ""
    with pytest.raises(ValueError, match="fixed_role_v1"):
        module._encode_live_t5_context(sample)


def test_a_dropout_sample_gets_exactly_one_context_row() -> None:
    module = _memory_module()

    context = module._encode_live_t5_context(_memory_sample(0))

    assert context.shape == (1, T5_LEN, T5_WIDTH)


# ---------------------------------------------------------------------------
# 6. 端到端：一条记忆样本从 forward 走到 loss，经 vendored model_fn
# ---------------------------------------------------------------------------


class _EndToEndPipe(_Pipe):
    """带 prompt 单元、真 scheduler 与 vendored ``model_fn`` 的假管线。"""

    def __init__(self, dit) -> None:
        from diffsynth.pipelines.wan_video import model_fn_wan_video

        super().__init__(
            dit=dit,
            device=torch.device("cpu"),
            torch_dtype=torch.float32,
            in_iteration_models=("dit",),
            scheduler=_WanLikeScheduler(),
        )
        self._model_fn = model_fn_wan_video
        self.model_fn_calls: list[dict[str, object]] = []

    def model_fn(self, **inputs):
        self.model_fn_calls.append(inputs)
        return self._model_fn(**inputs)


def _end_to_end_module(dit, *, text_dropout_p: float = 0.0):
    module = _memory_module()
    module.profile_sync_cuda = False
    object.__setattr__(module, "_pipe", _EndToEndPipe(dit))
    module.dit = dit
    module.training_cfg = {
        "use_gradient_checkpointing": False,
        "seed": 42,
        "text_dropout_p": text_dropout_p,
    }
    module.reference_runtime = "tiny"
    return module


def _runtime_sample(memory_frames: int = 2, target_latent_frames: int = 2):
    sample = _memory_sample(memory_frames, target_latent_frames)
    sample["segment_id"] = "segment_1"
    sample["input_latents"] = torch.randn(
        1,
        TINY_DIT_KWARGS["in_dim"],
        target_latent_frames,
        RUNTIME_H,
        RUNTIME_W,
    )
    return sample


def test_a_memory_sample_runs_end_to_end_through_the_vendored_runtime(monkeypatch) -> None:
    seen = _record_timesteps(monkeypatch)
    dit = _tiny_dit()
    module = _end_to_end_module(dit)
    sample = _runtime_sample(2)
    sample["input_latents"].requires_grad_()
    loss = module.forward(sample)
    assert loss.ndim == 0 and torch.isfinite(loss)
    call = module.pipe.model_fn_calls[-1]
    assert call["prefix_frames"] == 2
    assert call["fuse_vae_embedding_in_latents"] is True
    assert call["reference_offset"] == INFERENCE_OFFSET
    assert torch.count_nonzero(call["latents"][:, :, :2]) == 0
    assert torch.all(seen[-1].view(call["latents"].shape[2], -1)[:2] == 0)
    assert call["context"].shape[0] == 3
    loss.backward()
    assert sample["input_latents"].grad[:, :, :2].abs().sum() > 0


def test_reference_predictions_never_enter_the_target_loss() -> None:
    pipe = _FakePipe()
    inputs = _shared_inputs(torch.randn(*LATENT_SHAPE),prefix_frames=2)
    torch.manual_seed(SEED)
    loss = flow_match_sft_loss(pipe,**dict(inputs), reference_noise_sigma_max=0.0)
    pipe.dit.forward = lambda *,latents,**_: torch.cat([latents[:,:,:2]*99,latents[:,:,2:]*pipe.dit.scale],dim=2)
    torch.manual_seed(SEED)
    changed = flow_match_sft_loss(pipe,**dict(inputs), reference_noise_sigma_max=0.0)
    assert torch.equal(loss,changed)


def test_runtime_takes_reference_positions_under_native_temporal_mode() -> None:
    # 去镜相位那组：全互读、带参考、整段连续原生时间（10-31 放开，原来只配镜相位）。
    groups = _memory_groups(2,2)
    _run_model_fn(_tiny_dit(),latent_frames=4,latent_shot_groups=groups,
        context=torch.zeros(len(groups),512,4096),fuse_vae_embedding_in_latents=True,
        prefix_frames=2,reference_offset=INFERENCE_OFFSET,
        memory_attention_policy="bidirectional_v1",memory_attention_window=full_window(),
        temporal_position_mode="global_time_native_rope_v1")


def test_joint_target_shots_and_reference_train_in_one_sample() -> None:
    dit = _tiny_dit()
    module = _end_to_end_module(dit)
    sample = _runtime_sample(2,4)
    sample["input_latents"].requires_grad_()
    sample.update(
        num_frames=12,
        shot_groups=((0,1),(1,2),(2,7),(7,12)),
        latent_shot_groups=((0,1),(1,2),(2,4),(4,6)),
        prompt=_memory_rows(2)+["State: Subject 1 holds Subject 3.","State: Subject 2 holds Subject 3."],
    )
    loss = module.forward(sample)
    call = module.pipe.model_fn_calls[-1]
    assert call["prefix_frames"] == 2
    assert call["context"].shape[0] == 4
    assert torch.count_nonzero(call["latents"][:,:,:2]) == 0
    loss.backward()
    assert sample["input_latents"].grad[:, :, :2].abs().sum() > 0


@pytest.fixture(autouse=True)
def selected_reference_encoder(monkeypatch):
    monkeypatch.setattr('mspc.inference.wan22.encode_single_frame',
        lambda runtime, image: (torch.zeros(1, TINY_DIT_KWARGS["in_dim"], 1, RUNTIME_H, RUNTIME_W)
                               if runtime == "tiny" else torch.zeros(1, 2, 1, 2, DIM)))


# ---------------------------------------------------------------------------
# 7. 按镜路由的参考：数据集给的路由在训练入口校验后交给 model_fn
# ---------------------------------------------------------------------------

ROUTED_OPTIONS = dataclasses.replace(OPTIONS, attention_policy="routed_self_only_v1")


def _routed_sample(memory_frames: int, routing):
    sample = _memory_sample(memory_frames)
    if routing is not None:
        sample["reference_routing"] = routing
    return sample


@pytest.mark.parametrize("memory_policy", ["routed_self_only_v1", "routed_first_latent_self_only_v1"])
def test_a_routed_sample_hands_its_validated_routing_to_the_runtime_as_tuples(memory_policy) -> None:
    options = dataclasses.replace(OPTIONS, attention_policy=memory_policy)
    inputs = _memory_module(options=options)._prepare_latent_inputs(_routed_sample(2, [[0, 1]]))
    assert inputs["reference_routing"] == ((0, 1),)
    assert inputs["memory_attention_policy"] == memory_policy


@pytest.mark.parametrize("options,memory_frames,routing,message", [
    (ROUTED_OPTIONS, 2, None, "carried exactly by samples with references"),
    (ROUTED_OPTIONS, 0, [[0]], "carried exactly by samples with references"),
    (OPTIONS, 2, [[0, 1]], "carried exactly by samples with references"),
    (ROUTED_OPTIONS, 2, [[1]], "route every one"),
    (ROUTED_OPTIONS, 2, [[0], [1]], "each of the 1 target shot"),
])
def test_the_training_entry_refuses_routing_that_does_not_fit_the_sample(options, memory_frames, routing,
                                                                         message) -> None:
    with pytest.raises(ValueError, match=message):
        _memory_module(options=options)._prepare_latent_inputs(_routed_sample(memory_frames, routing))


@pytest.mark.parametrize("memory_policy", ["routed_self_only_v1", "routed_first_latent_self_only_v1"])
def test_a_routed_joint_sample_trains_end_to_end_through_the_vendored_runtime(memory_policy) -> None:
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "holocine_first_frame"
    module = _end_to_end_module(dit)
    module.memory_prefix_options = dataclasses.replace(OPTIONS, attention_policy=memory_policy)
    module.temporal_position_mode = "global_time_native_rope_v1"
    module.timestep_boundaries = (0.0, 1.0)
    sample = _runtime_sample(2, 4)
    sample["input_latents"].requires_grad_()
    sample.update(
        num_frames=12,
        shot_groups=((0, 1), (1, 2), (2, 7), (7, 12)),
        latent_shot_groups=((0, 1), (1, 2), (2, 4), (4, 6)),
        prompt=_memory_rows(2) + ["State: Subject 1 holds Subject 3.", "State: Subject 2 holds Subject 3."],
        reference_routing=[[1], [0, 1]],
    )
    loss = module.forward(sample)
    call = module.pipe.model_fn_calls[-1]
    assert call["reference_routing"] == ((1,), (0, 1))
    assert call["memory_attention_policy"] == memory_policy
    loss.backward()
    assert sample["input_latents"].grad[:, :, :2].abs().sum() > 0
