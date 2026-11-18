from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mspc.inference import wan22
from mspc.training.backends import wan22 as wan22_training
from mspc.models import wan22_loader as wan22_runtime
from mspc.models.source_identity import WanVideoPrediction


class _FakePipe:
    device = "cpu"
    height_division_factor = 16
    width_division_factor = 16


def _module_for_prepare() -> wan22_training.Wan22DiffSynthTrainingModule:
    module = object.__new__(wan22_training.Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(module)
    object.__setattr__(module, "_pipe", _FakePipe())
    module.component_mode = wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5
    module.training_cfg = {"flow_shift": 5.0, "use_gradient_checkpointing": True,
                           "activation_checkpointing": "recompute_all"}
    module.pipeline_cfg = {"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)}
    module.temporal_position_mode = "global_time_native_rope_v1"
    module.timestep_boundaries = (0.0, 1.0)
    # ``_prepare_latent_inputs`` 把这一步的记忆帧数并进逐步进度行（与 ``text_dropout``
    # 同一条通道）；真实路径上 ``forward`` 每步先把它清空。
    module.last_timing = {}
    module.profile_sync_cuda = False
    module.profile_timing = False
    return module


def _video(frames: int) -> torch.Tensor:
    return torch.zeros((3, frames, 32, 32), dtype=torch.float32)


def _shot_prompt(now: str = "The cube moves right.") -> str:
    return now


class _FakeDit(torch.nn.Module):
    require_vae_embedding = False
    require_clip_embedding = False
    has_image_input = False
    has_ref_conv = False
    control_adapter = None
    patch_size = (1, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.patch_embedding = torch.nn.Conv3d(16, 2, self.patch_size)
        self.head = torch.nn.Module()
        self.head.head = torch.nn.Linear(2, 16 * 4)
        block = torch.nn.Module()
        self_attn = torch.nn.Module()
        for name in ("q", "k", "v", "o"):
            setattr(self_attn, name, torch.nn.Linear(2, 2, bias=False))
        self_attn.norm_q = torch.nn.LayerNorm(2)
        self_attn.norm_k = torch.nn.LayerNorm(2)
        block.self_attn = self_attn
        block.cross_attn = torch.nn.Linear(2, 2, bias=False)
        block.ffn = torch.nn.Linear(2, 2, bias=False)
        self.blocks = torch.nn.ModuleList([block])



def _model_paths(tmp_path: Path) -> dict[str, object]:
    dit_a = tmp_path / "dit-a.safetensors"
    dit_b = tmp_path / "dit-b.safetensors"
    t5 = tmp_path / "t5.safetensors"
    vae = tmp_path / "vae.safetensors"
    tokenizer = tmp_path / "tokenizer"
    for path in (dit_a, dit_b, t5, vae):
        path.write_bytes(b"weights")
    tokenizer.mkdir()
    return {
        "dit": [str(dit_a), str(dit_b)],
        "t5": str(t5),
        "vae": str(vae),
        "tokenizer": str(tokenizer),
    }


def _install_fake_diffsynth_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[dict[str, object]],
) -> None:
    runtime_root = tmp_path / "runtime"
    package_root = runtime_root / "diffsynth"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "wan_video.py").write_text("", encoding="utf-8")
    (package_root / "diffusion.py").write_text("", encoding="utf-8")
    class ModelConfig:
        def __init__(self, *, path: object, skip_download: bool, computation_dtype=None,
                     computation_device=None) -> None:
            self.path = path
            self.skip_download = skip_download
            self.computation_dtype = computation_dtype
            self.computation_device = computation_device

    class WanVideoPipeline:
        @classmethod
        def from_pretrained(cls, **kwargs: object) -> object:
            calls.append(kwargs)
            config_paths = [config.path for config in kwargs["model_configs"]]  # type: ignore[index]
            dit_count = sum(isinstance(path, list) for path in config_paths)
            has_t5 = any(isinstance(path, str) and path.endswith("t5.safetensors") for path in config_paths)
            has_vae = any(isinstance(path, str) and path.endswith("vae.safetensors") for path in config_paths)
            return SimpleNamespace(
                dit=_FakeDit() if dit_count else None,
                dit2=_FakeDit() if dit_count == 2 else None,
                text_encoder=torch.nn.Linear(1, 1) if has_t5 else None,
                vae=torch.nn.Linear(1, 1) if has_vae else None,
                tokenizer=(
                    object()
                    if kwargs.get("tokenizer_config") is not None
                    else None
                ),
                device=kwargs["device"],
                scheduler=SimpleNamespace(set_timesteps=lambda *args, **kwargs: None),
                use_unified_sequence_parallel=False,
            )

    fake_diffsynth = SimpleNamespace(__file__=str(package_root / "__init__.py"))
    fake_wan_video = SimpleNamespace(
        __file__=str(package_root / "wan_video.py"),
        ModelConfig=ModelConfig,
        WanVideoPipeline=WanVideoPipeline,
        WanVideoUnit_ShapeChecker=type("WanVideoUnit_ShapeChecker", (), {}),
        WanVideoUnit_NoiseInitializer=type("WanVideoUnit_NoiseInitializer", (), {}),
        WanVideoUnit_PromptEmbedder=type("WanVideoUnit_PromptEmbedder", (), {}),
        WanVideoUnit_InputVideoEmbedder=type("WanVideoUnit_InputVideoEmbedder", (), {}),
    )
    fake_loss = SimpleNamespace(
        __file__=str(package_root / "diffusion.py"),
        FlowMatchSFTLoss=lambda *args, **kwargs: torch.zeros(()),
    )

    def fake_import(name: str) -> object:
        if name == "diffsynth":
            return fake_diffsynth
        if name == "diffsynth.pipelines.wan_video":
            return fake_wan_video
        if name == "diffsynth.diffusion":
            return fake_loss
        raise AssertionError(f"unexpected DiffSynth import: {name}")

    monkeypatch.setattr(
        wan22_runtime,
        "_resolve_local_diffsynth_root",
        lambda root: runtime_root,
    )
    monkeypatch.setattr(wan22_runtime.importlib, "import_module", fake_import)


def test_p3_shot_prompt_contract_is_one_trimmed_text_per_shot() -> None:
    assert wan22_runtime.require_shot_prompts([_shot_prompt()], 1) == [_shot_prompt()]
    embedded = "The narrator says Now: move right."
    assert wan22_runtime.require_shot_prompts([embedded], 1) == [embedded]

    for prompt in (" The cube moves right.", "The cube moves right. ", "", "   "):
        with pytest.raises(ValueError, match="trimmed"):
            wan22_runtime.require_shot_prompts([prompt], 1)
    with pytest.raises(ValueError, match="does not match shot count"):
        wan22_runtime.require_shot_prompts([_shot_prompt()], 2)
    with pytest.raises(ValueError, match="one shot prompt per shot"):
        wan22_runtime.require_shot_prompts(_shot_prompt(), 1)


def test_p3_cache_uint8_transport_is_bit_exact() -> None:
    uint8_video = torch.arange(256, dtype=torch.uint8).repeat(3).reshape(3, 1, 16, 16)
    normalized_video = uint8_video.float() / 255 * 2 - 1

    normalized_frames = wan22_training._tensor_to_pil_frames(normalized_video)
    uint8_frames = wan22_training._uint8_tensor_to_pil_frames(uint8_video)

    assert [frame.mode for frame in uint8_frames] == ["RGB"]
    assert [frame.size for frame in uint8_frames] == [(16, 16)]
    assert [frame.tobytes() for frame in uint8_frames] == [
        frame.tobytes() for frame in normalized_frames
    ]
    with pytest.raises(ValueError, match="floating-point"):
        wan22_training._tensor_to_pil_frames(uint8_video)


def test_p3_prepare_allows_uint8_only_for_cache_transport() -> None:
    module = _module_for_prepare()
    uint8_video = torch.zeros((3, 9, 32, 32), dtype=torch.uint8)
    data = {
        "video": uint8_video,
        "prompt": [_shot_prompt()],
        "shot_groups": [[0, 9]],
        "latent_shot_groups": [[0, 3]],
    }

    with pytest.raises(ValueError, match="floating-point"):
        module._prepare_inputs(data)

    shared, positive, negative = module._prepare_inputs(
        data,
        allow_cache_uint8=True,
    )

    assert len(shared["input_video"]) == 9
    assert all(frame.mode == "RGB" and frame.size == (32, 32) for frame in shared["input_video"])
    assert positive == {"prompt": _shot_prompt()}
    assert negative == {}


def test_p3_prepare_k1_strips_all_multishot_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module_for_prepare()
    monkeypatch.setattr(wan22_training, "_tensor_to_pil_frames", lambda value: [object()] * value.shape[1])

    shared, positive, negative = module._prepare_inputs(
        {
            "video": _video(9),
            "prompt": [_shot_prompt()],
            "shot_groups": [[0, 9]],
            "latent_shot_groups": [[0, 3]],
        }
    )

    assert positive == {"prompt": _shot_prompt()}
    assert negative == {}
    assert "shot_groups" not in shared
    assert "latent_shot_groups" not in shared
    assert shared["cfg_merge"] is False
    assert shared["sliding_window_size"] is None
    assert shared["sliding_window_stride"] is None
    assert shared["input_image"] is None
    assert shared["reference_image"] is None
    assert shared["vace_reference_image"] is None
    assert shared["framewise_decoding"] is False


def test_p3_prepare_k2_requires_explicit_groups_and_preserves_per_shot_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module_for_prepare()
    monkeypatch.setattr(wan22_training, "_tensor_to_pil_frames", lambda value: [object()] * value.shape[1])
    expected_shots = ((0, 9), (9, 18))
    expected_latents = ((0, 3), (3, 6))
    fake_wan_video = SimpleNamespace(
        validate_multishot_groups=lambda shots, latents, frames: (
            (asserted_shots := tuple(tuple(item) for item in shots)),
            (asserted_latents := tuple(tuple(item) for item in latents)),
        )
    )
    monkeypatch.setattr(wan22_training.importlib, "import_module", lambda name: fake_wan_video)

    prompts = [_shot_prompt(now="The cube enters from the left."), _shot_prompt(now="The cube leaves at the right.")]
    shared, positive, negative = module._prepare_inputs(
        {
            "video": _video(18),
            "prompt": prompts,
            "shot_groups": [list(item) for item in expected_shots],
            "latent_shot_groups": [list(item) for item in expected_latents],
        }
    )

    assert shared["shot_groups"] == expected_shots
    assert shared["latent_shot_groups"] == expected_latents
    assert positive == {"prompt": prompts}
    assert negative == {}


def test_p3_cached_prepare_uses_the_same_multishot_boundary_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module_for_prepare()
    module.device = torch.device("cpu")
    module.dtype = torch.bfloat16
    calls: list[tuple[object, object, int]] = []
    expected_shots = ((0, 9), (9, 18))
    expected_latents = ((0, 3), (3, 6))

    def validate(shots: object, latents: object, frames: int):
        calls.append((shots, latents, frames))
        return expected_shots, expected_latents

    monkeypatch.setattr(
        wan22_training.importlib,
        "import_module",
        lambda name: SimpleNamespace(validate_multishot_groups=validate),
    )
    prepared = module._prepare_latent_inputs(
        {
            "input_latents": torch.zeros((1, 16, 6, 2, 2)),
            "height": 32,
            "width": 32,
            "num_frames": 18,
            "shot_groups": [list(group) for group in expected_shots],
            "latent_shot_groups": [list(group) for group in expected_latents],
        }
    )

    assert calls == [
        (
            [list(group) for group in expected_shots],
            [list(group) for group in expected_latents],
            18,
        )
    ]
    assert prepared["shot_groups"] == expected_shots
    assert prepared["latent_shot_groups"] == expected_latents


def test_p3_prepare_rejects_implicit_resize_and_bad_k1_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module_for_prepare()
    monkeypatch.setattr(wan22_training, "_tensor_to_pil_frames", lambda value: [object()] * value.shape[1])

    with pytest.raises(ValueError, match="implicit spatial resizing"):
        module._prepare_inputs(
            {
                "video": torch.zeros((3, 9, 31, 32)),
                "prompt": [_shot_prompt()],
                "shot_groups": [[0, 9]],
                "latent_shot_groups": [[0, 3]],
            }
        )
    with pytest.raises(ValueError, match="native Wan 4n\\+1"):
        module._prepare_inputs(
            {
                "video": _video(9),
                "prompt": [_shot_prompt()],
                "shot_groups": [[0, 9]],
                "latent_shot_groups": [[0, 2]],
            }
        )


def test_p3_inference_normalizes_k1_to_native_and_repeats_ordinary_negative_prompt(
) -> None:
    runtime = object.__new__(wan22.Wan22InferenceRuntime)
    runtime.inference_cfg = {"negative_prompt": "low quality"}

    prompt, negative, frames, metadata = runtime._normalize_case(
        prompts=[_shot_prompt()],
        shot_groups=[[0, 9]],
        latent_shot_groups=[[0, 3]],
        memory_frame_count=0,
    )
    assert (prompt, negative, frames, metadata) == (_shot_prompt(), "low quality", 9, {})

    prompts = [_shot_prompt(now="The cube enters."), _shot_prompt(now="The cube exits.")]
    prompt, negative, frames, metadata = runtime._normalize_case(
        prompts=prompts,
        shot_groups=[[0, 9], [9, 18]],
        latent_shot_groups=[[0, 3], [3, 6]],
        memory_frame_count=0,
    )
    assert prompt == prompts
    assert negative == ["low quality", "low quality"]
    assert frames == 18
    assert metadata == {
        "shot_groups": [[0, 9], [9, 18]],
        "latent_shot_groups": [[0, 3], [3, 6]],
    }


def test_p3_inference_rejects_bad_prompt_before_any_pipeline_call() -> None:
    runtime = object.__new__(wan22.Wan22InferenceRuntime)
    runtime.inference_cfg = {"negative_prompt": "low quality"}
    with pytest.raises(ValueError, match="trimmed"):
        runtime._normalize_case(
            prompts=[" untrimmed shot prompt"],
            shot_groups=[[0, 9]],
            latent_shot_groups=[[0, 3]],
            memory_frame_count=0,
        )


def test_p3_runtime_is_local_only_and_has_no_download_escape_hatch() -> None:
    source = inspect.getsource(wan22_training)
    loader_source = inspect.getsource(wan22_runtime.load_pure_t2v_pipeline)

    assert "ShotRPE" not in source
    assert "anchor_image" not in source
    assert "ModelConfig.download" not in loader_source
    assert "skip_download=True" in loader_source
    assert "component_mode" in loader_source
    assert '"redirect_common_files": False' in loader_source
    assert '"use_usp": False' in loader_source
    assert "_resolve_local_diffsynth_root" in loader_source
    for removed in (
        "sha256",
        "pinned_runtime_constants",
        "pinned_cache_runtime_constants",
        "verify_p3_checkpoint_receipt",
        "require_explicit_shot_rpe_checkpoint_keys",
    ):
        assert removed not in inspect.getsource(wan22_runtime)


@pytest.mark.parametrize(
    ("component_mode", "expected_model_keys", "expects_tokenizer"),
    [
        (
            wan22_runtime.WAN22_COMPONENT_MODE_NONCACHED_TRAINING,
            ("dit", "t5", "vae"),
            True,
        ),
        (
            wan22_runtime.WAN22_COMPONENT_MODE_CACHE_ENCODE,
            ("t5", "vae"),
            True,
        ),
        (
            wan22_runtime.WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS,
            ("vae",),
            False,
        ),
        (
            wan22_runtime.WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT,
            ("t5",),
            True,
        ),
        (
            wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5,
            ("dit", "t5"),
            True,
        ),
        (
            wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING,
            ("dit",),
            False,
        ),
    ],
)
def test_p3_loader_component_modes_request_exact_model_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component_mode: str,
    expected_model_keys: tuple[str, ...],
    expects_tokenizer: bool,
) -> None:
    calls: list[dict[str, object]] = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    paths = _model_paths(tmp_path)

    pipe, _ = wan22_runtime.load_pure_t2v_pipeline(
        diffsynth_root=str(tmp_path / "runtime"),
        model_paths=paths,
        self_attention_policy="full",
        torch_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        component_mode=component_mode,
    )

    assert len(calls) == 1
    call = calls[0]
    requested = [config.path for config in call["model_configs"]]  # type: ignore[index]
    for model_config in call["model_configs"]:
        assert model_config.computation_dtype == (
            torch.float32 if isinstance(model_config.path, list) else None
        )
    expected = []
    for key in expected_model_keys:
        if key == "dit":
            expected.append([str(Path(path).resolve()) for path in paths["dit"]])  # type: ignore[index]
        else:
            expected.append(str(Path(paths[key]).resolve()))  # type: ignore[arg-type]
    assert requested == expected
    assert "tokenizer_config" in call
    if expects_tokenizer:
        tokenizer_config = call["tokenizer_config"]
        assert tokenizer_config.path == str(Path(paths["tokenizer"]).resolve())  # type: ignore[attr-defined,arg-type]
    else:
        assert call["tokenizer_config"] is None
    assert call["redirect_common_files"] is False
    assert call["use_usp"] is False
    assert pipe.in_iteration_models == (
        ("dit",) if "dit" in expected_model_keys else tuple()
    )
    assert {type(unit).__name__ for unit in pipe.units} <= {
        "WanVideoUnit_ShapeChecker",
        "WanVideoUnit_NoiseInitializer",
        "WanVideoUnit_PromptEmbedder",
        "WanVideoUnit_InputVideoEmbedder",
    }
    if component_mode == wan22_runtime.WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT:
        assert pipe.dit is None and pipe.vae is None
        assert pipe.text_encoder is not None and pipe.tokenizer is not None
        assert [type(unit).__name__ for unit in pipe.units] == ["WanVideoUnit_PromptEmbedder"]
    if component_mode == wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING:
        assert pipe.dit is not None and pipe.units == []
        assert pipe.text_encoder is None and pipe.tokenizer is None and pipe.vae is None


def test_cached_training_loader_never_opens_text_or_vae_paths(tmp_path, monkeypatch):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    paths = _model_paths(tmp_path)
    paths = {"dit": paths["dit"]}
    pipe, _ = wan22_runtime.load_pure_t2v_pipeline(
        diffsynth_root=str(tmp_path / "runtime"), model_paths=paths,
        self_attention_policy="holocine_first_frame", torch_dtype=torch.bfloat16,
        device=torch.device("cpu"), component_mode=wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING,
    )
    assert pipe.dit is not None
    assert len(calls[0]["model_configs"]) == 1
    assert calls[0]["tokenizer_config"] is None


def test_dual_expert_loader_converts_original_parameters_before_device_transfer(tmp_path, monkeypatch):
    calls = []
    transfers = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    paths = _model_paths(tmp_path)
    low_noise_shard = tmp_path / "low-noise.safetensors"
    low_noise_shard.write_bytes(b"weights")
    fake_wan_video = wan22_runtime.importlib.import_module("diffsynth.pipelines.wan_video")
    original = fake_wan_video.WanVideoPipeline.from_pretrained

    def load_original_fp32(**kwargs):
        pipe = original(**kwargs)
        with torch.no_grad():
            for dit in (pipe.dit, pipe.dit2):
                dit.patch_embedding.weight.fill_(1.003)
                dit.head.head.weight.fill_(1.003)
        return pipe

    def transfer(dit, *args, **kwargs):
        assert not args and kwargs == {"device": torch.device("cuda:0")}
        assert dit.patch_embedding.weight.device.type == "cpu"
        assert dit.patch_embedding.weight.dtype == torch.bfloat16
        assert torch.equal(dit.patch_embedding.weight,
                           torch.full_like(dit.patch_embedding.weight, 1.003))
        assert dit.head.head.weight.dtype == torch.float32
        assert torch.equal(dit.head.head.weight, torch.full_like(dit.head.head.weight, 1.003))
        transfers.append(dit)
        return dit

    monkeypatch.setattr(fake_wan_video.WanVideoPipeline, "from_pretrained", load_original_fp32)
    monkeypatch.setattr(_FakeDit, "to", transfer)
    pipe, _ = wan22_runtime.load_pure_t2v_pipeline(
        diffsynth_root=str(tmp_path / "runtime"), model_paths=paths,
        self_attention_policy="holocine_first_frame", torch_dtype=torch.bfloat16,
        device=torch.device("cuda:0"), low_noise_dit=[str(low_noise_shard)],
    )
    expert_configs = [config for config in calls[0]["model_configs"] if isinstance(config.path, list)]
    assert len(expert_configs) == 2
    assert all(config.computation_device == "cpu" for config in expert_configs)
    assert all(config.computation_dtype == torch.float32 for config in expert_configs)
    assert transfers == [pipe.dit, pipe.dit2]
    assert pipe.in_iteration_models == ("dit",)
    assert pipe.in_iteration_models_2 == ("dit2",)


@pytest.mark.parametrize("extra_component", ["text_encoder", "tokenizer", "vae"])
def test_cached_training_loader_rejects_unrequested_components(tmp_path, monkeypatch, extra_component):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    fake_wan_video = wan22_runtime.importlib.import_module("diffsynth.pipelines.wan_video")
    original = fake_wan_video.WanVideoPipeline.from_pretrained

    def with_extra_component(**kwargs):
        pipe = original(**kwargs)
        setattr(pipe, extra_component, object())
        return pipe

    monkeypatch.setattr(fake_wan_video.WanVideoPipeline, "from_pretrained", with_extra_component)
    with pytest.raises(RuntimeError, match="cached_training forbids loading text_encoder/tokenizer/vae"):
        wan22_runtime.load_pure_t2v_pipeline(
            diffsynth_root=str(tmp_path / "runtime"), model_paths=_model_paths(tmp_path),
            self_attention_policy="full", torch_dtype=torch.bfloat16, device=torch.device("cpu"),
            component_mode=wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING,
        )


@pytest.mark.parametrize("extra_component", ["dit", "vae"])
def test_text_cache_loader_rejects_unrequested_model(tmp_path, monkeypatch, extra_component):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    fake_wan_video = wan22_runtime.importlib.import_module("diffsynth.pipelines.wan_video")
    original = fake_wan_video.WanVideoPipeline.from_pretrained

    def with_extra_model(**kwargs):
        pipe = original(**kwargs)
        setattr(pipe, extra_component, torch.nn.Linear(1, 1))
        return pipe

    monkeypatch.setattr(fake_wan_video.WanVideoPipeline, "from_pretrained", with_extra_model)
    with pytest.raises(RuntimeError, match="cache_encode_text forbids loading DiT/vae"):
        wan22_runtime.load_pure_t2v_pipeline(
            diffsynth_root=str(tmp_path / "runtime"), model_paths=_model_paths(tmp_path),
            self_attention_policy="full", torch_dtype=torch.bfloat16, device=torch.device("cpu"),
            component_mode=wan22_runtime.WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT,
        )


def test_loader_accepts_wan21_pure_t2v_with_unused_image_embedding_defaults(tmp_path, monkeypatch):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(_FakeDit, "require_vae_embedding", True)
    monkeypatch.setattr(_FakeDit, "require_clip_embedding", True)
    pipe, _ = wan22_runtime.load_pure_t2v_pipeline(
        diffsynth_root=str(tmp_path / "runtime"), model_paths=_model_paths(tmp_path),
        self_attention_policy="holocine_first_frame", torch_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert pipe.dit.require_vae_embedding and pipe.dit.require_clip_embedding
    assert pipe.dit.patch_embedding.in_channels == 16
    assert all(block.self_attn.self_attention_policy == "holocine_first_frame" for block in pipe.dit.blocks)


@pytest.mark.parametrize("field", ["has_image_input", "has_ref_conv", "control_adapter"])
def test_loader_rejects_actual_image_or_control_pathways(tmp_path, monkeypatch, field):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(_FakeDit, field, True if field != "control_adapter" else object())
    with pytest.raises(RuntimeError, match="without image or control conditioning"):
        wan22_runtime.load_pure_t2v_pipeline(
            diffsynth_root=str(tmp_path / "runtime"), model_paths=_model_paths(tmp_path),
            self_attention_policy="full", torch_dtype=torch.bfloat16, device=torch.device("cpu"),
        )


def test_loader_rejects_mismatched_input_output_latent_channels(tmp_path, monkeypatch):
    calls = []
    _install_fake_diffsynth_runtime(monkeypatch, tmp_path, calls)
    original = _FakeDit.__init__

    def mismatched(self):
        original(self)
        self.head.head = torch.nn.Linear(2, 8 * 4)

    monkeypatch.setattr(_FakeDit, "__init__", mismatched)
    with pytest.raises(RuntimeError, match="input and output latent channels must match"):
        wan22_runtime.load_pure_t2v_pipeline(
            diffsynth_root=str(tmp_path / "runtime"), model_paths=_model_paths(tmp_path),
            self_attention_policy="full", torch_dtype=torch.bfloat16, device=torch.device("cpu"),
        )


class _FakeTrainingScheduler:
    """The upstream schedule surface the local flow-matching loss consumes."""

    timesteps = torch.arange(4, dtype=torch.float32)

    def add_noise(self, sample, noise, timestep):
        return sample + noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        return torch.ones(())


def _recording_model_fn(seen: list[dict[str, object]]):
    def model_fn(**inputs):
        seen.append(inputs)
        return WanVideoPrediction(torch.zeros_like(inputs["latents"]), None)

    return model_fn


def test_cached_forward_encodes_current_prompts_and_supervises_latents(monkeypatch) -> None:
    seen: list[dict[str, object]] = []
    module = _module_for_prepare()
    module.component_mode = wan22_runtime.WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5
    module.dtype = torch.bfloat16
    module.device = torch.device("cpu")
    object.__setattr__(module, "_pipe", SimpleNamespace(
        device="cpu", torch_dtype=torch.bfloat16,
        scheduler=_FakeTrainingScheduler(), in_iteration_models=(),
        model_fn=_recording_model_fn(seen),
    ))
    prompts = []
    def encode(data):
        prompts.extend(data["prompt"])
        return torch.ones((1, 1, 1), dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_encode_live_t5_context", encode)
    loss = module.forward({
        "target_storage": "latents",
        "flow_draw_seed": 42,
        "input_latents": torch.ones((1, 1, 1, 1, 1)),
        "prompt": ["The cube moves right."], "height": 32, "width": 32,
        "num_frames": 1,
    })
    assert prompts == ["The cube moves right."]
    assert torch.isfinite(loss)
    assert {"latents", "context"} <= set(seen[-1])
    assert seen[-1]["prefix_frames"] == 0


def test_live_t5_context_encode_seeds_cfg_scale_for_unit_runner() -> None:
    seen_shared: list[dict[str, object]] = []

    class PromptUnit:
        name = "prompt"

    class LivePipe:
        def __init__(self) -> None:
            self.device = "cpu"
            self.units = [PromptUnit()]

        def unit_runner(
            self,
            unit: PromptUnit,
            pipe: object,
            shared: dict[str, object],
            positive: dict[str, object],
            negative: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
            seen_shared.append(dict(shared))
            if shared["cfg_scale"] != 1:
                raise AssertionError("live T5 must skip CFG")
            positive["context"] = torch.ones((1, 2, 3), dtype=torch.bfloat16)
            return shared, positive, negative

    live = object.__new__(wan22_training.Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(live)
    object.__setattr__(live, "_pipe", LivePipe())
    live.device = torch.device("cpu")
    live.dtype = torch.bfloat16
    context = live._encode_live_t5_context(
        {
            "prompt": [_shot_prompt()],
            "shot_groups": [[0, 9]],
        }
    )
    assert seen_shared and seen_shared[0]["cfg_scale"] == 1.0
    assert seen_shared[0]["cfg_merge"] is False
    assert torch.equal(context, torch.ones((1, 2, 3), dtype=torch.bfloat16))


def test_p3_checkpoint_exports_full_dit_without_runtime_metadata() -> None:
    module = object.__new__(wan22_training.Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(module)
    module.dit = torch.nn.Linear(2, 1, bias=False)

    exported = module.export_trainable_state_dict(module.state_dict(), remove_prefix="dit.")
    assert set(exported) == {"weight"}
    reloaded = torch.nn.Linear(2, 1, bias=False)
    reloaded.load_state_dict(exported, strict=True)
    assert torch.equal(reloaded.weight, module.dit.weight)
    assert not hasattr(module, "checkpoint_receipt_metadata")


def test_p3_inference_does_not_import_training_backend_private_symbols() -> None:
    import mspc.inference.wan22 as inference_wan22

    source = inspect.getsource(inference_wan22)
    # 推理侧不得伸手去拿训练侧的私有 helper。
    assert "from mspc.training.backends.wan22 import" not in source
    assert "from mspc.models.wan22_loader import" in source


def test_strict_checkpoint_load_rejects_missing_attention_weight() -> None:
    from mspc.training.dit_parameter_freeze import load_dit_state_dict

    class TinyDit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            block = torch.nn.Module()
            self_attn = torch.nn.Module()
            self_attn.q = torch.nn.Linear(2, 2, bias=False)
            block.self_attn = self_attn
            block.ffn = torch.nn.Linear(2, 2, bias=False)
            self.blocks = torch.nn.ModuleList([block])

    model = TinyDit()
    state = dict(model.state_dict())
    assert load_dit_state_dict(model, state) == "full"
    state.pop("blocks.0.self_attn.q.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dit_state_dict(model, state)
    assert "load_dit_state_dict" in inspect.getsource(wan22_training._build_model)
    assert "load_dit_state_dict" in inspect.getsource(wan22.Wan22InferenceRuntime.__init__)
    assert "load_state_dict(state_dict, strict=False)" not in inspect.getsource(
        wan22_training._build_model
    )
    assert "load_state_dict(state_dict, strict=False)" not in inspect.getsource(
        wan22.Wan22InferenceRuntime.__init__
    )


def test_p3_inference_api_has_no_anchor_or_legacy_position_flags() -> None:
    # P3 retired the old ABC renderer path and its three legacy switches:
    # per-case anchor-frame conditioning (``anchor_image``), the local/global
    # Shot-RPE toggle (``use_local_shot_rpe``), and the renderer-selection flag
    # (``conditioning_mode``). ``memory`` is the B0 memory
    # prefix (P8) API: the M clean latent frames this shot carries in front of
    # its own noise together with each memory shot's own text
    # (``MemoryPrefixInput``), ``None`` on the memory-dropout form (the no-memory
    # arm and every story's shot 0). Whether that口 exists at all is not a flag here --
    # it is ``model.memory_prefix`` on the config, so a checkpoint without that
    # arm refuses a non-``None`` value outright. The forbidden-name check below
    # still guards against the three retired flags; only the frozen parameter
    # set grows to admit the memory prefix.
    names = set(inspect.signature(wan22.Wan22InferenceRuntime.generate_case).parameters)
    assert names == {
        "self",
        "prompts",
        "shot_groups",
        "latent_shot_groups",
        # 每镜剧本镜长：存盘前把补到 4n+1 的成片裁回它（MultiShotMaster 上游同一做法）。
        "shot_frame_counts",
        "output_path",
        "seed",
        # B0 记忆前缀（P8）：这一镜前接的 M 张干净记忆潜帧与各记忆镜自己的文字；M=0 的记忆
        # dropout 形态传 None。
        "memory",
        # SDEdit 记忆探针：M=0 的一镜从给定帧按强度重去噪（``SDEditInput``）；普通出片传 None。
        "sdedit",
        # 原生首帧口的潜变量入参：单镜 case 的第 0 潜帧钉成给定帧、时间步 0；不给首帧传 None。
        "first_frame",
        # 推理期首帧早钉：σ 降到它以下时把首帧接回当前噪声层（``_FirstFrameDenoiseState``）。
        "first_frame_pin_until_sigma",
        # 段式出片：留下每镜末帧与其单帧编码，供下一段取参考。
        "capture_history",
        # 目标镜的稀疏布局；没有空间条件时传 None。
        "target_layouts",
        # 规划器 token（PlannerTokenCondition）；没有 token 时传 None。
        "planner_tokens",
    }
    source = inspect.getsource(wan22.Wan22InferenceRuntime)
    for forbidden in ("anchor_image", "use_local_shot_rpe", "conditioning_mode"):
        assert forbidden not in source
