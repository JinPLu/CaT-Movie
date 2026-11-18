"""Public Wan DiffSynth runtime helpers shared by training and inference."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from mspc.core.wan22_contract import WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES

WAN22_COMPONENT_MODE_NONCACHED_TRAINING = "noncached_training"
WAN22_COMPONENT_MODE_CACHE_ENCODE = "cache_encode"
WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS = "cache_encode_latents"
WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT = "cache_encode_text"
WAN22_COMPONENT_MODE_CACHED_TRAINING = "cached_training"
WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5 = "cached_training_live_t5"
WAN22_COMPONENT_MODES = frozenset(
    {
        WAN22_COMPONENT_MODE_NONCACHED_TRAINING,
        WAN22_COMPONENT_MODE_CACHE_ENCODE,
        WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS,
        WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT,
        WAN22_COMPONENT_MODE_CACHED_TRAINING,
        WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5,
    }
)
_P3_ALLOWED_UNIT_NAMES = frozenset(
    {
        "WanVideoUnit_ShapeChecker",
        "WanVideoUnit_NoiseInitializer",
        "WanVideoUnit_PromptEmbedder",
        "WanVideoUnit_InputVideoEmbedder",
    }
)
_UNITS_BY_COMPONENT_MODE = {
    WAN22_COMPONENT_MODE_NONCACHED_TRAINING: frozenset(
        {
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_NoiseInitializer",
            "WanVideoUnit_PromptEmbedder",
            "WanVideoUnit_InputVideoEmbedder",
        }
    ),
    WAN22_COMPONENT_MODE_CACHE_ENCODE: frozenset(
        {
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_NoiseInitializer",
            "WanVideoUnit_PromptEmbedder",
            "WanVideoUnit_InputVideoEmbedder",
        }
    ),
    WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS: frozenset(
        {
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_NoiseInitializer",
            "WanVideoUnit_InputVideoEmbedder",
        }
    ),
    WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5: frozenset({"WanVideoUnit_PromptEmbedder"}),
    WAN22_COMPONENT_MODE_CACHED_TRAINING: frozenset(),
    WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT: frozenset({"WanVideoUnit_PromptEmbedder"}),
}
_FORBIDDEN_PIPELINE_MODELS = (
    "image_encoder",
    "dit2",
    "motion_controller",
    "vace",
    "vace2",
    "vap",
    "animate_adapter",
    "audio_encoder",
)
_MODEL_PATH_KEYS_BY_COMPONENT_MODE = {
    WAN22_COMPONENT_MODE_NONCACHED_TRAINING: ("dit", "t5", "vae", "tokenizer"),
    WAN22_COMPONENT_MODE_CACHE_ENCODE: ("t5", "vae", "tokenizer"),
    WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS: ("vae",),
    WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT: ("t5", "tokenizer"),
    WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5: ("dit", "t5", "tokenizer"),
    WAN22_COMPONENT_MODE_CACHED_TRAINING: ("dit",),
}


def require_path(path: object, label: str, *, directory: bool = False) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} must be a non-empty string path")
    resolved = Path(path)
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {resolved}")
    return resolved.resolve()


def require_component_mode(value: object) -> str:
    if not isinstance(value, str) or value not in WAN22_COMPONENT_MODES:
        raise ValueError(
            "Wan2.2 component_mode must be one of "
            f"{sorted(WAN22_COMPONENT_MODES)}"
        )
    return value


def require_model_paths(
    paths: Mapping[str, Any],
    *,
    component_mode: str = WAN22_COMPONENT_MODE_NONCACHED_TRAINING,
) -> dict[str, Any]:
    if not isinstance(paths, Mapping):
        raise TypeError("model.paths must be a mapping")
    mode = require_component_mode(component_mode)
    required = set(_MODEL_PATH_KEYS_BY_COMPONENT_MODE[mode])
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(
            f"model.paths for component_mode={mode!r} must contain {missing}"
        )
    result: dict[str, Any] = {}
    if "dit" in required:
        dit = paths["dit"]
        if isinstance(dit, (str, bytes)) or not isinstance(dit, (list, tuple)) or not dit:
            raise ValueError("model.paths.dit must be a non-empty list of local shards")
        result["dit"] = [
            str(require_path(path, f"model.paths.dit[{index}]"))
            for index, path in enumerate(dit)
        ]
    if "t5" in required:
        result["t5"] = str(require_path(paths["t5"], "model.paths.t5"))
    if "vae" in required:
        result["vae"] = str(require_path(paths["vae"], "model.paths.vae"))
    if "tokenizer" in required:
        result["tokenizer"] = str(
            require_path(paths["tokenizer"], "model.paths.tokenizer", directory=True)
        )
    return result


def require_temporal_position_mode(value: object) -> str:
    if not isinstance(value, str) or value not in WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES:
        raise ValueError(
            "model.temporal_position_mode must be one of "
            f"{sorted(WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES)}"
        )
    return value


def _module_under(module: object, package_root: Path, label: str) -> None:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise RuntimeError(f"{label} has no import source path")
    resolved = Path(source).resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} was imported from {resolved}, outside required DiffSynth root {package_root}"
        ) from exc


def _resolve_local_diffsynth_root(diffsynth_root: str) -> Path:
    root = require_path(diffsynth_root, "model.diffsynth_root", directory=True)
    package_root = root / "diffsynth"
    if not package_root.is_dir():
        raise FileNotFoundError(
            "model.diffsynth_root must directly contain the materialized diffsynth package: "
            f"{package_root}"
        )
    return root


def load_pure_t2v_pipeline(
    *,
    diffsynth_root: str,
    model_paths: Mapping[str, Any],
    torch_dtype: torch.dtype,
    device: torch.device,
    self_attention_policy: str,
    component_mode: str = WAN22_COMPONENT_MODE_NONCACHED_TRAINING,
    low_noise_dit: list[str] | None = None,
) -> tuple[Any, type[Any]]:
    """``low_noise_dit`` 只给 Wan2.2-A14B 推理：``model_paths["dit"]`` 是高噪专家、它是低噪专家，载成 ``pipe.dit2``，
    去噪循环按 ``switch_DiT_boundary`` 换用（上游 ``WanVideoPipeline.__call__``）。训练每个 run 只载一个专家。"""
    root = _resolve_local_diffsynth_root(diffsynth_root)
    package_root = root / "diffsynth"
    mode = require_component_mode(component_mode)
    paths = require_model_paths(model_paths, component_mode=mode)

    diffsynth = importlib.import_module("diffsynth")
    wan_video = importlib.import_module("diffsynth.pipelines.wan_video")
    _module_under(diffsynth, package_root, "diffsynth")
    _module_under(wan_video, package_root, "diffsynth.pipelines.wan_video")

    ModelConfig = getattr(wan_video, "ModelConfig")
    WanVideoPipeline = getattr(wan_video, "WanVideoPipeline")
    shape_checker_type = getattr(wan_video, "WanVideoUnit_ShapeChecker")
    unit_types = [
        shape_checker_type,
        getattr(wan_video, "WanVideoUnit_NoiseInitializer"),
        getattr(wan_video, "WanVideoUnit_PromptEmbedder"),
        getattr(wan_video, "WanVideoUnit_InputVideoEmbedder"),
    ]
    if {unit_type.__name__ for unit_type in unit_types} != _P3_ALLOWED_UNIT_NAMES:
        raise RuntimeError("unexpected P3 DiffSynth pipeline-unit import surface")

    model_configs = []
    if low_noise_dit is not None and "dit" not in paths:
        raise ValueError("a low-noise expert needs the high-noise expert in model_paths.dit")
    if "dit" in paths:
        # Load the original source before applying selective parameter storage:
        # loading every tensor as BF16 would also round protected FP32 modules.
        model_configs.append(ModelConfig(path=paths["dit"], skip_download=True,
                                         computation_dtype=torch.float32,
                                         computation_device="cpu" if low_noise_dit is not None else device))
    if low_noise_dit is not None:
        model_configs.append(ModelConfig(
            path=[str(require_path(path, f"low_noise_dit[{index}]")) for index, path in enumerate(low_noise_dit)],
            skip_download=True, computation_dtype=torch.float32, computation_device="cpu"))
    if "t5" in paths:
        model_configs.append(ModelConfig(path=paths["t5"], skip_download=True))
    if "vae" in paths:
        model_configs.append(ModelConfig(path=paths["vae"], skip_download=True))
    pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device": device,
        "model_configs": model_configs,
        # DiffSynth's API has a non-None default tokenizer config.
        # Cache encoding of latents must not load an unused text tokenizer.
        "tokenizer_config": None,
        "redirect_common_files": False,
        "use_usp": False,
    }
    if "tokenizer" in paths:
        pretrained_kwargs["tokenizer_config"] = ModelConfig(
            path=paths["tokenizer"],
            skip_download=True,
        )
    pipe = WanVideoPipeline.from_pretrained(**pretrained_kwargs)

    dit_loaded = getattr(pipe, "dit", None) is not None
    text_loaded = getattr(pipe, "text_encoder", None) is not None
    vae_loaded = getattr(pipe, "vae", None) is not None
    tokenizer_loaded = getattr(pipe, "tokenizer", None) is not None
    if mode == WAN22_COMPONENT_MODE_NONCACHED_TRAINING:
        if not (dit_loaded and text_loaded and vae_loaded and tokenizer_loaded):
            raise RuntimeError(
                "DiffSynth did not load the required dit/text_encoder/vae/tokenizer components"
            )
    elif mode == WAN22_COMPONENT_MODE_CACHE_ENCODE:
        if dit_loaded:
            raise RuntimeError("Wan2.2 cache_encode forbids loading the DiT component")
        if not (text_loaded and vae_loaded and tokenizer_loaded):
            raise RuntimeError(
                "DiffSynth did not load the required text_encoder/vae/tokenizer components"
            )
    elif mode == WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS:
        if dit_loaded:
            raise RuntimeError("Wan2.2 cache_encode_latents forbids loading the DiT component")
        if text_loaded or tokenizer_loaded:
            raise RuntimeError(
                "Wan2.2 cache_encode_latents forbids loading text_encoder/tokenizer"
            )
        if not vae_loaded:
            raise RuntimeError("DiffSynth did not load the required vae component")
    elif mode == WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT:
        if dit_loaded or vae_loaded:
            raise RuntimeError("Wan2.2 cache_encode_text forbids loading DiT/vae")
        if not (text_loaded and tokenizer_loaded):
            raise RuntimeError("Wan2.2 cache_encode_text requires text_encoder/tokenizer")
    elif mode == WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5:
        if not dit_loaded:
            raise RuntimeError("DiffSynth did not load the required DiT component")
        if not (text_loaded and tokenizer_loaded):
            raise RuntimeError(
                "Wan2.2 cached_training_live_t5 requires text_encoder/tokenizer"
            )
        if vae_loaded:
            raise RuntimeError("Wan2.2 cached_training_live_t5 forbids loading vae")
    elif mode == WAN22_COMPONENT_MODE_CACHED_TRAINING:
        if not dit_loaded:
            raise RuntimeError("DiffSynth did not load the required DiT component")
        if text_loaded or tokenizer_loaded or vae_loaded:
            raise RuntimeError("Wan2.2 cached_training forbids loading text_encoder/tokenizer/vae")

    experts = [pipe.dit] if low_noise_dit is None else [pipe.dit, getattr(pipe, "dit2", None)]
    if low_noise_dit is not None and pipe.dit2 is None:
        raise RuntimeError("DiffSynth did not load the low-noise expert as dit2")
    for dit in experts if dit_loaded else []:
        from mspc.models.wan22_precision import apply_wan22_parameter_storage
        apply_wan22_parameter_storage(dit)
        if low_noise_dit is not None:
            # Convert each original FP32 expert on the host before GPU residency.
            # Keep both unconverted experts out of the GPU startup peak.
            dit.to(device=device)
        from mspc.core.wan22_contract import WAN22_SELF_ATTENTION_POLICIES
        if self_attention_policy not in WAN22_SELF_ATTENTION_POLICIES:
            raise ValueError(f"unknown self_attention_policy: {self_attention_policy}")
        for block in dit.blocks:
            block.self_attn.self_attention_policy = self_attention_policy
        # Wan2.1 T2V retains True defaults for require_{vae,clip}_embedding,
        # although its actual DiT has no image pathway. Check that pathway and
        # the input/output channel contract rather than the unused defaults.
        if (dit.has_image_input or dit.has_ref_conv
                or dit.control_adapter is not None):
            raise RuntimeError("Wan renderer requires a pure-T2V DiT without image or control conditioning")
        patch_size = getattr(dit, "patch_size", None)
        if isinstance(patch_size, (str, bytes)):
            raise RuntimeError("Wan2.2 P3 requires DiT patch_size=(1, 2, 2)")
        try:
            normalized_patch_size = tuple(patch_size)
        except TypeError as exc:
            raise RuntimeError("Wan2.2 P3 requires DiT patch_size=(1, 2, 2)") from exc
        if normalized_patch_size != (1, 2, 2):
            raise RuntimeError("Wan2.2 P3 requires DiT patch_size=(1, 2, 2)")
        if dit.head.head.out_features != 4 * dit.patch_embedding.in_channels:
            raise RuntimeError("Wan pure-T2V input and output latent channels must match")
    for name in _FORBIDDEN_PIPELINE_MODELS:
        if name == "dit2" and low_noise_dit is not None:
            continue
        if getattr(pipe, name, None) is not None:
            raise RuntimeError(f"Wan2.2 P3 forbids runtime component {name!r}")
    if getattr(pipe, "use_unified_sequence_parallel", False):
        raise RuntimeError("Wan2.2 P3 forbids unified sequence parallelism")

    pipe.in_iteration_models = ("dit",) if dit_loaded else tuple()
    pipe.in_iteration_models_2 = ("dit2",) if low_noise_dit is not None else tuple()
    pipe.post_units = []
    wanted_units = _UNITS_BY_COMPONENT_MODE[mode]
    pipe.units = [
        unit_type()
        for unit_type in unit_types
        if unit_type.__name__ in wanted_units
    ]
    actual_units = {type(unit).__name__ for unit in pipe.units}
    if actual_units != wanted_units:
        raise RuntimeError(
            f"Wan2.2 P3 pipeline units for {mode} are {sorted(actual_units)}, "
            f"expected {sorted(wanted_units)}"
        )
    if actual_units - _P3_ALLOWED_UNIT_NAMES:
        raise RuntimeError("Wan2.2 P3 pipeline units escaped the allow-list")
    return pipe, shape_checker_type


def require_shot_prompts(value: object, shot_count: int) -> list[str]:
    """One rendered text per shot (``multishot.build_prompt_from_fields``): non-empty, trimmed."""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("data.prompt must be one shot prompt per shot")
    prompts = list(value)
    if len(prompts) != shot_count:
        raise ValueError(
            f"prompt count {len(prompts)} does not match shot count {shot_count}"
        )
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
            raise ValueError(f"prompt {index} must be a non-empty trimmed string")
    return prompts


@dataclass(frozen=True)
class VaeOnlyRuntime:
    """只持有VAE的运行时，供在线参考编码、缓存解码与数据生产共用。"""

    device: Any
    dtype: Any
    pipe: Any
    pipeline_cfg: dict[str, Any]


def load_vae_only_runtime(model: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], device: Any) -> VaeOnlyRuntime:
    """按调用方已经解析的唯一模型与编码配置加载VAE。"""
    pipe, _ = load_pure_t2v_pipeline(
        diffsynth_root=model["diffsynth_root"],
        model_paths=model["paths"],
        self_attention_policy=model["self_attention_policy"],
        torch_dtype=torch.bfloat16,
        device=device,
        component_mode=WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS,
    )
    return VaeOnlyRuntime(
        device=device, dtype=torch.bfloat16, pipe=pipe, pipeline_cfg=dict(pipeline_cfg)
    )
