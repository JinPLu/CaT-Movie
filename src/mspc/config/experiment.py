"""Strict experiment configuration contract with validated Python configs."""

from __future__ import annotations

import inspect
import json
import runpy
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from mspc.config.shared_root import SharedRootError, expand_shared_root_token

from mspc.core.wan22_contract import (
    MEMORY_MODES,
    MEMORY_REFERENCE_TEMPORAL_MODE,
    WAN22_CACHE_LAYOUT_LATENTS_ONLY,
    WAN22_CACHE_LAYOUTS,
)

EXPERIMENT_SCHEMA_VERSION = 1
_INTERPOLATION = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")
# Caption-JSON sources compile the same {global_caption, shot0..shotN-1} object;
# the format only chooses the field order in the compiled prompt.
WAN22_DIFFSYNTH_CAPTION_PROMPT_FORMATS = frozenset(
    {
        "caption_json_v1",
    }
)
WAN22_DIFFSYNTH_RAW_PROMPT_FORMAT = "story_now_v1"
#: 一步训练把一条故事怎么送进 DiT：``joint`` 是整故事一次去噪（一个 σ、一次
#: backward），``per_shot`` 是逐镜各抽自己的 σ_k/ε_k、逐镜 forward+backward 累积、
#: 一次 optimizer 更新。两者是不同的目标函数，不是同一条路的两档性能开关，所以
#: wan22_diffsynth 的每一份 training 块都必须显式写出走哪一条，**没有默认值**。
SHOT_FORWARD_MODE_JOINT = "joint"
SHOT_FORWARD_MODE_PER_SHOT = "per_shot"
WAN22_SHOT_FORWARD_MODES = frozenset(
    {SHOT_FORWARD_MODE_JOINT, SHOT_FORWARD_MODE_PER_SHOT}
)
WAN22_ACTIVATION_CHECKPOINTING_POLICIES = frozenset({"recompute_all", "save_self_attention"})
SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE = "shot2story_raw_annotation_v1"
_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?\Z")
#: 同一BlockDataset的显式目标存储：原视频在线VAE，或已编码latent；参考均在线编码。
WAN22_BLOCK_DATASET_TYPES = frozenset({"block_latent_cache", "block_video"})
# Recipe names are declared here and re-exported by
# mspc.training.dit_parameter_freeze, which implements them. The dependency
# only runs training -> config: mspc.training.__init__ imports runner, which
# imports mspc.config, so importing the training owner from here would cycle.
WAN22_DIT_PARAMETER_FREEZE_FULL_DIT_V1 = "full_dit_v1"
#: 冻结原 DiT，只训每个块指定线性层旁的低秩增量（``mspc.models.lora``）。
WAN22_DIT_PARAMETER_FREEZE_LORA_V1 = "lora_v1"
WAN22_DIT_PARAMETER_FREEZE_RECIPES = frozenset(
    {WAN22_DIT_PARAMETER_FREEZE_FULL_DIT_V1, WAN22_DIT_PARAMETER_FREEZE_LORA_V1}
)


# 既有480评测／标注素材的固定工作点；保留其身份，不再限制训练task。
# 新训练媒体必须由task显式给出，任何生产者／reader不得从此常量回退。
WAN22_CACHED_SPACETIME_CONTRACT = MappingProxyType({
    "height": 480, "width": 832, "target_fps": 16,
    "frame_policy": "fps_aware_target_fps", "max_rgb_frames": 308, "vae_temporal_stride": 4,
})


class ExperimentConfigError(ValueError):
    """Raised when an experiment mapping violates the strict schema."""


def _require_exact_keys(
    mapping: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str],
    location: str,
) -> None:
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required - set(optional))
    if missing:
        raise ExperimentConfigError(f"{location} is missing required fields: {missing}")
    if unknown:
        raise ExperimentConfigError(f"{location} contains unknown fields: {unknown}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{location} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ExperimentConfigError(f"{location} keys must be strings")
    return value


def _exact(
    value: Any,
    *,
    location: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    result = _mapping(value, location)
    try:
        _require_exact_keys(
            result, required=required, optional=optional, location=location
        )
    except ValueError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    return result


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExperimentConfigError(f"{location} must be a non-empty trimmed string")
    return value


def _path(value: Any, location: str, base_dir: Path) -> str:
    raw = _string(value, location)
    try:
        raw = expand_shared_root_token(raw, location)
    except SharedRootError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    if _INTERPOLATION.search(raw) or raw == "~" or raw.startswith("~/"):
        raise ExperimentConfigError(
            f"{location} must not use environment or home interpolation"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    # Keep an explicitly selected executable symlink (notably a venv's
    # ``bin/python``). Resolving it to the base interpreter changes sys.prefix
    # and silently discards the selected environment.
    return os.path.abspath(path)


def _absolute_local_path(value: Any, location: str, base_dir: Path) -> str:
    """Parse an explicitly local, absolute filesystem path.

    The Wan2.2 DiffSynth path deliberately has no model-id or download fallback.
    Requiring the original config value to be absolute prevents a URL or an
    implicit relative cache location from being mistaken for a model source.
    """

    raw = _string(value, location)
    try:
        raw = expand_shared_root_token(raw, location)
    except SharedRootError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    if "://" in raw or not Path(raw).is_absolute():
        raise ExperimentConfigError(
            f"{location} must be an absolute local filesystem path"
        )
    return _path(raw, location, base_dir)


def _cuda_device(value: Any, location: str) -> str:
    """Accept an explicit CUDA device only; P3 has no CPU fallback."""

    device = _string(value, location)
    if _CUDA_DEVICE.fullmatch(device) is None:
        raise ExperimentConfigError(
            f"{location} must be 'cuda' or an explicit CUDA index such as 'cuda:0'"
        )
    return device


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ExperimentConfigError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ExperimentConfigError(f"{location} must be >= {minimum}")
    return value


def _number(value: Any, location: str, *, minimum: float | None = None) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ExperimentConfigError(f"{location} must be a finite number")
    if minimum is not None and value < minimum:
        raise ExperimentConfigError(f"{location} must be >= {minimum}")
    return value


def _box_jitter(value: Any, location: str) -> dict[str, float] | None:
    from mspc.training.block_dataset import require_box_jitter
    try:
        return require_box_jitter(value)
    except ValueError as error:
        raise ExperimentConfigError(f"{location}: {error}") from error


def _planner_token_data(value: Any, location: str, base: Path) -> dict:
    """``dataset.planner_tokens``：``content`` 定必填键——规划器特征要标签与预计算特征（口径、层），位置标记对照只要标签。"""
    content = value.get("content") if isinstance(value, Mapping) else None
    validators = {"content": lambda x, q: _one_of(x, q, {"planner_features", "person_index", "person_geometry"}),
                  "labels_path": lambda x, q: _path(x, q, base),
                  "point_jitter": lambda x, q: _dropout_probability(x, q),
                  "point_dropout_p": lambda x, q: _dropout_probability(x, q),
                  "token_dropout_p": lambda x, q: _dropout_probability(x, q)}
    if content == "planner_features":
        validators.update({"features_root": lambda x, q: _path(x, q, base),
                           "feature_layer": lambda x, q: _integer(x, q, minimum=0),
                           "feature_pooling": lambda x, q: _one_of(x, q, {"name", "point", "entry"})})
    elif content == "person_geometry":
        validators.update({"geometry_labels_path": lambda x, q: _path(x, q, base),
                           "max_people": lambda x, q: _integer(x, q, minimum=1)})
    return _fields(value, location=location, validators=validators)


def _layout_fraction(value: Any, location: str) -> float | None:
    if value is None:
        return None
    number = _number(value, location, minimum=0)
    if not 0 < number <= 1:
        raise ExperimentConfigError(f"{location} must be in (0, 1]")
    return float(number)


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise ExperimentConfigError(f"{location} must be a boolean")
    return value


def _probability(value: Any, location: str) -> float:
    number = float(_number(value, location))
    if not 0 <= number <= 1:
        raise ExperimentConfigError(f"{location} must be in [0, 1]")
    return number


def _dropout_probability(value: Any, location: str) -> float:
    number = float(_number(value, location))
    if not 0 <= number < 1:
        raise ExperimentConfigError(f"{location} must be in [0, 1)")
    return number


def _single_complete_sample_batch_size(value: Any, location: str) -> int:
    parsed = _integer(value, location, minimum=1)
    if parsed != 1:
        raise ExperimentConfigError(
            f"{location} must be 1 because the current trainers process one "
            "complete variable-length sample per device"
        )
    return parsed


def _parse_precompute(value: Any, base: Path) -> dict[str, Any]:
    parsed = _fields(
        value,
        location="precompute",
        validators={
            "provenance_path": lambda v, p: _path(v, p, base),
            "encode_batch_size": lambda item, loc: _integer(item, loc, minimum=1),
            "dataset_num_workers": lambda item, loc: _integer(item, loc, minimum=1),
            "prefetch_factor": lambda item, loc: _integer(item, loc, minimum=1),
            "video_decode_num_threads": lambda item, loc: _integer(item, loc, minimum=1),
        },
    )
    return parsed


def _parse_optimization(value: Any) -> dict[str, Any]:
    """Parse the optimization block; readable keys follow lr_scheduler_type."""

    raw = _mapping(value, "optimization")
    validators: dict[str, Callable[[Any, str], Any]] = {
        "learning_rate": lambda item, loc: _number(item, loc),
        "weight_decay": lambda item, loc: _number(item, loc),
        "lr_scheduler_type": lambda item, loc: _one_of(
            item, loc, {"constant", "constant_with_warmup"}
        ),
    }
    if raw.get("lr_scheduler_type") == "constant_with_warmup":
        # Linear warmup owns the whole ramp, so ConstantLR's factor/total_iters
        # would be keys nobody reads; reject them instead of ignoring them.
        # warmup_steps is denominated in optimizer updates, not micro-steps.
        validators["warmup_steps"] = lambda item, loc: _integer(item, loc, minimum=0)
        return _fields(raw, location="optimization", validators=validators)
    validators["lr_scheduler_factor"] = lambda item, loc: _number(item, loc)
    validators["lr_scheduler_total_iters"] = lambda item, loc: _integer(
        item, loc, minimum=1
    )
    parsed = _fields(raw, location="optimization", validators=validators)
    if parsed["lr_scheduler_factor"] != 1.0:
        raise ExperimentConfigError(
            "optimization.lr_scheduler_factor must be exactly 1.0 so the first "
            "optimizer update uses optimization.learning_rate"
        )
    return parsed


def _one_of(value: Any, location: str, allowed: set[str]) -> str:
    parsed = _string(value, location)
    if parsed not in allowed:
        raise ExperimentConfigError(f"{location} must be one of {sorted(allowed)}")
    return parsed


def _nullable_string(value: Any, location: str) -> str | None:
    return None if value is None else _string(value, location)


def _nullable_path(value: Any, location: str, base_dir: Path) -> str | None:
    return None if value is None else _path(value, location, base_dir)


def _strings(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigError(f"{location} must be a non-empty list")
    return [_string(item, f"{location}[{index}]") for index, item in enumerate(value)]


def _string_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list):
        raise ExperimentConfigError(f"{location} must be a list")
    return [_string(item, f"{location}[{index}]") for index, item in enumerate(value)]


def _paths(value: Any, location: str, base_dir: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigError(f"{location} must be a non-empty list")
    return [_path(item, f"{location}[{index}]", base_dir) for index, item in enumerate(value)]


def _absolute_local_paths(
    value: Any, location: str, base_dir: Path
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigError(f"{location} must be a non-empty list")
    return [
        _absolute_local_path(item, f"{location}[{index}]", base_dir)
        for index, item in enumerate(value)
    ]


def _positive_integer_pair(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ExperimentConfigError(f"{location} must be a list of exactly two integers")
    return [
        _integer(item, f"{location}[{index}]", minimum=1)
        for index, item in enumerate(value)
    ]


def _nullable_shot2story_prompt_source(
    value: Any,
    location: str,
    base_dir: Path,
) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = _fields(
        value,
        location=location,
        validators={
            "kind": lambda item, loc: _one_of(
                item,
                loc,
                {SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE},
            ),
            "annotation_json": lambda item, loc: _path(item, loc, base_dir),
            "selected_shot_indices_column": lambda item, loc: _string(item, loc),
        },
    )
    return parsed


def _fields(
    value: Any,
    *,
    location: str,
    validators: Mapping[str, Callable[[Any, str], Any]],
    defaults: Mapping[str, Any] = MappingProxyType({}),
) -> dict[str, Any]:
    if not set(defaults) <= set(validators):
        raise ValueError("field defaults must have matching validators")
    raw = _exact(
        value,
        location=location,
        required=set(validators) - set(defaults),
        optional=set(defaults),
    )
    return {
        key: validator(raw[key] if key in raw else defaults[key], f"{location}.{key}")
        for key, validator in validators.items()
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated canonical experiment mapping."""

    schema_version: int
    source_path: Path
    resolved: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXPERIMENT_SCHEMA_VERSION}")
        if not self.source_path.is_absolute():
            raise ValueError("source_path must be absolute")

    @property
    def kind(self) -> str:
        return str(self.resolved["kind"])

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.resolved)


def canonical_json(mapping: Mapping[str, Any]) -> str:
    """Serialize a mapping deterministically for stable text output."""

    return json.dumps(
        _plain(mapping),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _target_cache_roots(value: Any, location: str, base_dir: Path) -> dict[str, str]:
    roots = _mapping(value, location)
    if not roots:
        raise ExperimentConfigError(f"{location} requires at least one explicit cache source")
    return {_string(key, f"{location} source name"): _path(path, f"{location}.{key}", base_dir)
            for key, path in roots.items()}


def _target_mask_weights(value: Any, location: str, base_dir: Path) -> dict[str, str] | None:
    if value is None:
        return None
    keys = set(_mapping(value, location))
    if keys == {"spatial_root", "weighting"}:
        path_keys = ("spatial_root",)
    elif keys == {"index_path", "root", "weighting"}:
        path_keys = ("index_path", "root")
    else:
        raise ExperimentConfigError(f"{location} must explicitly select original SAM RLE or pooled weight files")
    return _fields(value, location=location, validators={
        **{key: lambda x, q: _path(x, q, base_dir) for key in path_keys},
        "weighting": lambda x, q: _one_of(x, q, {"area_coverage", "touched_cell"}),
    })


def parse_wan22_dit_parameter_freeze(value: Any, location: str = "training.dit_parameter_freeze") -> dict[str, Any]:
    """``full_dit_v1`` 只有 recipe；``lora_v1`` 另写 rank、alpha 与每块要挂的线性层（块内相对名）。"""
    recipe = _one_of(_mapping(value, location).get("recipe"), f"{location}.recipe", WAN22_DIT_PARAMETER_FREEZE_RECIPES)
    if recipe == WAN22_DIT_PARAMETER_FREEZE_FULL_DIT_V1:
        return _fields(value, location=location, validators={"recipe": lambda v, p: v})
    parsed = _fields(value, location=location, validators={
        "recipe": lambda v, p: v,
        "rank": lambda v, p: _integer(v, p, minimum=1),
        "alpha": lambda v, p: _number(v, p, minimum=0),
        "targets": _string_list,
    })
    if not parsed["targets"] or len(set(parsed["targets"])) != len(parsed["targets"]) or parsed["alpha"] <= 0:
        raise ExperimentConfigError(f"{location} needs a positive alpha and non-empty, distinct targets")
    return parsed


def _parse_loss_tracking(value: Any, location: str, base_dir: Path) -> dict[str, Any] | None:
    """``training.loss_tracking``：结构由 ``mspc.training.loss_tracking.LossTracking`` 一处校验；密钥文件路径在这里展开。"""
    if value is None:
        return None
    from mspc.training.loss_tracking import LossTracking

    try:
        LossTracking.from_mapping(value)
    except ValueError as exc:
        raise ExperimentConfigError(f"{location}: {exc}") from exc
    return dict(value, api_key_file=_nullable_path(value["api_key_file"], f"{location}.api_key_file", base_dir))


def _parse_wan22_diffsynth_paths(
    value: Any, base_dir: Path, *, location: str
) -> dict[str, Any]:
    return _fields(
        value,
        location=location,
        validators={
            "dit": lambda v, p: _absolute_local_paths(v, p, base_dir),
            "t5": lambda v, p: _absolute_local_path(v, p, base_dir),
            "vae": lambda v, p: _absolute_local_path(v, p, base_dir),
            "tokenizer": lambda v, p: _absolute_local_path(v, p, base_dir),
        },
    )


def _long_shot_rope(value: Any, location: str) -> dict[str, Any] | None:
    if value is None:
        return None
    raw = _exact(value, location=location, required={"index", "native_latent_frames", "margin"}, optional=set())
    return {"index": _integer(raw["index"], f"{location}.index", minimum=1),
            "native_latent_frames": _integer(raw["native_latent_frames"], f"{location}.native_latent_frames", minimum=1),
            "margin": _number(raw["margin"], f"{location}.margin")}


def _parse_wan22_diffsynth_inference(
    value: Any,
    base_dir: Path,
) -> dict[str, Any]:
    """Parse the no-fallback pure-T2V inference contract for P3."""

    validators: dict[str, Callable[[Any, str], Any]] = {
        "enabled": _boolean,
        "device": _cuda_device,
        "target_height": lambda v, p: _integer(v, p, minimum=1),
        "target_width": lambda v, p: _integer(v, p, minimum=1),
        "fps": lambda v, p: _integer(v, p, minimum=1),
        "seed": _integer,
        "negative_prompt": _string,
        "num_inference_steps": lambda v, p: _integer(v, p, minimum=1),
        "sigma_shift": _number,
        "cfg_scale": _number,
        "video_quality": lambda v, p: _integer(v, p, minimum=1),
        "sampler": lambda v, p: _one_of(v, p, {"flow_match_euler", "flow_unipc_official"}),
        "test_csv_path": lambda v, p: _nullable_path(v, p, base_dir),
        "output_dir": lambda v, p: _nullable_path(v, p, base_dir),
        "smoke_case_id": _nullable_string,
        "weight_source": lambda v, p: _one_of(
            v,
            p,
            {
                "pretrained_model_paths",
                "full_dit_checkpoint",
                "training_output_checkpoint",
                "expert_lora_checkpoints",
            },
        ),
        "dit_checkpoint": lambda v, p: _nullable_path(v, p, base_dir),
        # Wan2.2-A14B 两个去噪专家（``mspc.models.noise_expert``）：model.paths.dit 是高噪专家，这里给低噪专家、两份 LoRA 导出、
        # 训练时的 LoRA 配方与切换边界；只配 weight_source='expert_lora_checkpoints'。
        "experts": lambda v, p: None if v is None else _fields(v, location=p, validators={
            "boundary": lambda x, q: _number(x, q, minimum=0),
            "low_noise_dit": lambda x, q: _absolute_local_paths(x, q, base_dir),
            "high_noise_lora": lambda x, q: _path(x, q, base_dir),
            "low_noise_lora": lambda x, q: _path(x, q, base_dir),
            "lora": parse_wan22_dit_parameter_freeze,
        }),
        # 长单镜时间 RoPE 修正的候选入口（语义见 mspc.inference.wan22.LongShotRope）；缺省不改位置编码。
        "long_shot_rope": _long_shot_rope,
    }
    defaults: dict[str, Any] = {
        "smoke_case_id": None,
        "long_shot_rope": None,
        "experts": None,
    }
    parsed = _fields(
        value,
        location="inference",
        validators=validators,
        defaults=defaults,
    )
    if (parsed["weight_source"] == "expert_lora_checkpoints") != (parsed["experts"] is not None):
        raise ExperimentConfigError("inference.experts goes with weight_source='expert_lora_checkpoints' and only with it")
    if parsed["experts"] is not None:
        experts = parsed["experts"]
        if (not 0 < experts["boundary"] < 1 or experts["lora"]["recipe"] != WAN22_DIT_PARAMETER_FREEZE_LORA_V1
                or parsed["dit_checkpoint"] is not None):
            raise ExperimentConfigError("inference.experts needs a boundary in (0, 1), a lora_v1 recipe "
                                        "and dit_checkpoint=null")
    operands = ("test_csv_path", "output_dir")
    if parsed["enabled"]:
        if parsed["weight_source"] != "training_output_checkpoint":
            missing = [key for key in operands if parsed[key] is None]
            if missing:
                raise ExperimentConfigError(
                    f"inference.enabled=true requires non-null operands: {missing}"
                )
        if parsed["weight_source"] == "full_dit_checkpoint":
            if parsed["dit_checkpoint"] is None:
                raise ExperimentConfigError(
                    "inference.weight_source='full_dit_checkpoint' requires "
                    "a non-null dit_checkpoint"
                )
        elif parsed["weight_source"] == "training_output_checkpoint":
            non_null = [
                key
                for key in (
                    "output_dir",
                    "dit_checkpoint",
                )
                if parsed[key] is not None
            ]
            if non_null:
                raise ExperimentConfigError(
                    "inference.weight_source='training_output_checkpoint' "
                    f"requires runtime-resolved null operands: {non_null}"
                )
        elif parsed["weight_source"] == "expert_lora_checkpoints":
            pass
        elif parsed["dit_checkpoint"] is not None:
            raise ExperimentConfigError(
                "inference.weight_source='pretrained_model_paths' requires null "
                "dit_checkpoint"
            )
    else:
        non_null = [
            key
            for key in (
                "test_csv_path",
                "output_dir",
                "smoke_case_id",
                "dit_checkpoint",
            )
            if parsed[key] is not None
        ]
        if non_null:
            raise ExperimentConfigError(
                f"inference.enabled=false requires null operands: {non_null}"
            )
    return parsed


def _parse_wan22_m0_mapping(value: Any, source: Path) -> ExperimentConfig:
    base = source.parent
    root = _exact(
        value,
        location="config",
        required={
            "schema_version", "kind", "runtime", "launch", "model",
            "generation", "execution", "m0", "scoring",
        },
    )
    version = _integer(root["schema_version"], "schema_version")
    if version != EXPERIMENT_SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"unsupported schema_version {version}; expected {EXPERIMENT_SCHEMA_VERSION}"
        )
    if root["kind"] != "wan22_m0_baseline":
        raise ExperimentConfigError("kind must be 'wan22_m0_baseline'")
    runtime = _fields(root["runtime"], location="runtime", validators={
        "repo_root": lambda v, p: _path(v, p, base),
        "python": lambda v, p: _path(v, p, base),
        "hf_cache": lambda v, p: _path(v, p, base),
        "ffmpeg": lambda v, p: _path(v, p, base),
        "offline_hf": _boolean,
        "cuda_visible_devices": _nullable_string,
    })
    launch = _fields(root["launch"], location="launch", validators={
        "num_workers": lambda v, p: _integer(v, p, minimum=1),
        "worker_index": lambda v, p: _integer(v, p, minimum=0),
    })
    if launch["worker_index"] >= launch["num_workers"]:
        raise ExperimentConfigError("launch.worker_index must be less than launch.num_workers")
    model = _fields(root["model"], location="model", validators={
        "official_generate_py": lambda v, p: _path(v, p, base),
        "ckpt_dir": lambda v, p: _path(v, p, base),
        "task": lambda v, p: _one_of(v, p, {"ti2v-5B"}),
    })
    generation = _fields(root["generation"], location="generation", validators={
        "size": lambda v, p: _string(v, p),
        "frame_num": lambda v, p: _integer(v, p, minimum=1),
        "sample_steps": lambda v, p: _integer(v, p, minimum=1),
        "sample_shift": _number,
        "sample_guide_scale": _number,
        "base_seed": lambda v, p: _integer(v, p, minimum=0),
        "offload_model": _boolean,
        "convert_model_dtype": _boolean,
        "t5_cpu": _boolean,
    })
    execution = _fields(root["execution"], location="execution", validators={
        "mode": lambda v, p: _one_of(v, p, {"dry_run", "generate", "build_manifest_only"}),
        "skip_existing": _boolean,
        "collision_policy": lambda v, p: _one_of(v, p, {"error", "replace"}),
    })
    m0 = _fields(root["m0"], location="m0", validators={
        "run_tag": _string,
        "generator_id": _string,
        "benchmark_track": _string,
        "eval_subset": lambda v, p: _path(v, p, base),
        "output_dir": lambda v, p: _path(v, p, base),
    })
    scoring = _fields(root["scoring"], location="scoring", validators={
        key: (lambda v, p: _path(v, p, base))
        for key in (
            "eval_root", "generator_spec_json", "generated_manifest", "vllm_config",
            "pass1_dir", "pass2_dir", "summary_json",
        )
    } | {
        "generator_id": _string,
        "manifest_generator": _string,
        "max_tokens": lambda v, p: _integer(v, p, minimum=1),
    })
    normalized = {
        "schema_version": version,
        "kind": "wan22_m0_baseline",
        "runtime": runtime,
        "launch": launch,
        "model": model,
        "generation": generation,
        "execution": execution,
        "m0": m0,
        "scoring": scoring,
    }
    return ExperimentConfig(
        schema_version=version,
        source_path=source,
        resolved=_freeze(normalized),
    )


def _require_wan22_spacetime_contract(
    task: Mapping[str, Any], *, location: str
) -> None:
    if task["height"] % 32 or task["width"] % 32:
        raise ExperimentConfigError(f"{location} pixels must be divisible by the Wan token grid (32)")
    if task["frame_policy"] != "fps_aware_target_fps" or task["vae_temporal_stride"] != 4:
        raise ExperimentConfigError(f"{location} requires explicit fps-aware sampling and temporal stride 4")


def parse_experiment_mapping(
    value: Any, *, source_path: str | Path
) -> ExperimentConfig:
    """Validate an already-decoded mapping without importing executable config."""

    source = Path(source_path).resolve(strict=False)
    raw_root = _mapping(value, "config")
    if raw_root.get("kind") == "wan22_m0_baseline":
        return _parse_wan22_m0_mapping(value, source)
    base = source.parent
    required = {
        "schema_version", "kind", "identity", "task", "runtime", "environment",
        "nccl", "launch", "trainer", "model", "dataset", "training", "pipeline_inputs",
        "optimization", "dataloader", "timing", "output", "inference", "checkpoint",
    }
    root = _exact(value, location="config", required=required, optional={"precompute"})
    version = _integer(root["schema_version"], "schema_version")
    if version != EXPERIMENT_SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"unsupported schema_version {version}; expected {EXPERIMENT_SCHEMA_VERSION}"
        )
    if root["kind"] != "train":
        raise ExperimentConfigError("kind must be 'train'")

    identity = _fields(
        root["identity"], location="identity", validators={
            key: (lambda v, p: _string(v, p))
            for key in ("model_key", "dataset_key", "run_name", "experiment_key")
        }
    )
    trainer = _string(root["trainer"], "trainer")
    from mspc.core.component_registry import (
        DATASET_REGISTRY,
        MODEL_REGISTRY,
        TRAINER_REGISTRY,
        validate_model_dataset_compatibility,
        validate_trainer_type,
    )

    validate_trainer_type(trainer)
    from mspc.config.model import parse_model

    model = parse_model(root["model"], base)
    model_entry = MODEL_REGISTRY.get(model["type"])
    task = _fields(root["task"], location="task", validators={
        "paradigm": lambda v, p: _string(v, p),
        "height": lambda v, p: _integer(v, p, minimum=1),
        "width": lambda v, p: _integer(v, p, minimum=1),
        "frame_policy": lambda v, p: _string(v, p),
        "max_rgb_frames": lambda v, p: _integer(v, p, minimum=1),
        "target_fps": lambda v, p: _integer(v, p, minimum=1),
        "vae_temporal_stride": lambda v, p: _integer(v, p, minimum=1),
    })
    if model["type"] == "wan22_diffsynth" and task is not None:
        _require_wan22_spacetime_contract(task, location="task")
    if (
        model_entry.task_paradigm is not None
        and task is not None
        and task["paradigm"] != model_entry.task_paradigm
    ):
        raise ExperimentConfigError(
            f"{model['type']} task.paradigm must be {model_entry.task_paradigm!r}"
        )
    runtime = _fields(root["runtime"], location="runtime", validators={
        # No "accelerate": it is a console script inside the env that "python"
        # names, so declaring it separately describes one environment twice and
        # invites the two halves to disagree.  mspc.runs.launch takes it from
        # the interpreter that is actually running.
        "python": lambda v, p: _path(v, p, base),
        "distributed_timeout_seconds": lambda v, p: _integer(v, p, minimum=1),
        "deepspeed": lambda v, p: None if v is None else _fields(v, location=p, validators={
            "zero_optimization": lambda x, q: _fields(x, location=q, validators={
                "stage": lambda z, r: _integer(z, r, minimum=0),
                "contiguous_gradients": _boolean,
                "reduce_bucket_size": lambda z, r: _integer(z, r, minimum=1),
                "allgather_bucket_size": lambda z, r: _integer(z, r, minimum=1),
                "overlap_comm": _boolean,
            }),
            "gradient_clipping": lambda x, q: _number(x, q, minimum=0),
            "wall_clock_breakdown": _boolean,
            "fp16": lambda x, q: _fields(x, location=q, validators={"enabled": _boolean}),
            "bf16": lambda x, q: _fields(x, location=q, validators={"enabled": _boolean}),
            "torch_autocast": lambda x, q: _fields(x, location=q, validators={
                "enabled": _boolean,
                "dtype": lambda z, r: _one_of(z, r, {"bfloat16"}),
                "lower_precision_safe_modules": _string_list,
            }),
        }),
        "hf_cache": lambda v, p: _path(v, p, base),
        "offline_hf": lambda v, p: _boolean(v, p),
    })
    if runtime["deepspeed"] is not None and runtime["deepspeed"]["zero_optimization"]["stage"] != 2:
        raise ExperimentConfigError("runtime.deepspeed.zero_optimization.stage must be 2 for this training path")
    if model["type"] == "wan22_diffsynth" and runtime["deepspeed"] is not None:
        ds = runtime["deepspeed"]
        if ds["fp16"]["enabled"] or ds["bf16"]["enabled"]:
            raise ExperimentConfigError("DeepSpeed whole-model casts must be disabled to preserve Wan protected FP32 modules")
        if not ds["torch_autocast"]["enabled"] or ds["torch_autocast"]["lower_precision_safe_modules"]:
            raise ExperimentConfigError("Wan DiT requires BF16 torch_autocast without lower-precision communication overrides")
    environment = _fields(root["environment"], location="environment", validators={
        key: (lambda v, p: _string(v, p))
        for key in ("tokenizers_parallelism", "python_unbuffered", "wandb_mode")
    })
    nccl = _fields(root["nccl"], location="nccl", validators={
        key: (lambda v, p: _string(v, p))
        for key in ("debug", "async_error_handling", "device_max_connections", "blocking_wait", "buffsize", "nthreads")
    })
    launch = _fields(root["launch"], location="launch", validators={
        "train_entrypoint": lambda v, p: _path(v, p, base),
        "infer_entrypoint": lambda v, p: _path(v, p, base),
        "train_pythonpath": lambda v, p: _paths(v, p, base),
        "infer_pythonpath": lambda v, p: _paths(v, p, base),
        "cuda_visible_devices": _nullable_string,
    })
    if model["type"] == "wan22_diffsynth":
        diffsynth_root = model["diffsynth_root"]
        missing_pythonpaths = [
            name
            for name in ("train_pythonpath", "infer_pythonpath")
            if diffsynth_root not in launch[name]
        ]
        if missing_pythonpaths:
            raise ExperimentConfigError(
                "wan22_diffsynth model.diffsynth_root must appear exactly in "
                f"launch entries: {missing_pythonpaths}"
            )
    raw_dataset = _mapping(root["dataset"], "dataset")
    dataset_type = _string(raw_dataset.get("type"), "dataset.type")
    try:
        dataset_entry = DATASET_REGISTRY.get(dataset_type)
    except ValueError as exc:
        raise ExperimentConfigError(
            "dataset.type must be one of "
            f"{sorted(DATASET_REGISTRY.known_types())}"
        ) from exc
    if dataset_type == "multishot_csv":
        dataset = _fields(root["dataset"], location="dataset", validators={
            "type": lambda v, p: _string(v, p),
            "schema": lambda v, p: _string(v, p), "prompt_format": lambda v, p: _string(v, p),
            "metadata_path": lambda v, p: _path(v, p, base),
            "shot2story_prompt_source": lambda v, p: _nullable_shot2story_prompt_source(
                v,
                p,
                base,
            ),
            "video_column": lambda v, p: _string(v, p), "caption_column": _nullable_string,
            "shot_groups_column": lambda v, p: _string(v, p), "case_id_column": _nullable_string,
            "case_id_prefix": lambda v, p: _string(v, p),
            "repeat": lambda v, p: _integer(v, p, minimum=1),
            "cache_layout": lambda v, p: (
                None if v is None else _one_of(v, p, WAN22_CACHE_LAYOUTS)
            ),
        }, defaults={"shot2story_prompt_source": None, "cache_layout": None})
    elif dataset_type in WAN22_BLOCK_DATASET_TYPES:
        block_validators = {
            "type": lambda v, p: _string(v, p),
            "members_path": lambda v, p: _path(v, p, base),
            "sampling_manifest_path": lambda v, p: _path(v, p, base),
            "text_admission_path": lambda v, p: None if v is None else _path(v, p, base),
            "segment_capacity": lambda v, p: _fields(v, location=p, validators={
                "max_frames": lambda x, q: _integer(x, q, minimum=1),
            }),
            "reference_row_dropout_p": lambda v, p: None if v is None else _dropout_probability(v, p),
            "reference_policy": lambda v, p: _one_of(v, p, {"random_frame"}),
            "reference_crops": lambda v, p: None if v is None else _fields(v, location=p, validators={
                "path": lambda x, q: _path(x, q, base),
            }),
            "workload_cap": lambda v, p: None if v is None else _fields(v, location=p, validators={
                "latent_frames": lambda x, q: _integer(x, q, minimum=1),
                "attention_frame_pairs": lambda x, q: _integer(x, q, minimum=1),
            }),
            "minimum_layout_shot_fraction": lambda v, p: _layout_fraction(v, p),
            "box_dropout_p": lambda v, p: None if v is None else _dropout_probability(v, p),
            "picture_type_dropout_p": lambda v, p: None if v is None else _dropout_probability(v, p),
            "box_jitter": lambda v, p: _box_jitter(v, p),
            "planner_tokens": lambda v, p: None if v is None else _planner_token_data(v, p, base),
            "reference_entities": lambda v, p: None if v is None else _fields(v, location=p, validators={
                "labels_path": lambda x, q: _path(x, q, base),
            }),
        }
        if dataset_type == "block_latent_cache":
            block_validators.update({
                "cache_provenance_path": lambda v, p: _path(v, p, base),
                "target_index_path": lambda v, p: _path(v, p, base),
                "target_roots": lambda v, p: _target_cache_roots(v, p, base),
            })
        else:
            block_validators["target_latent_geometry"] = lambda v, p: _fields(v, location=p, validators={
                "channels": lambda x, q: _integer(x, q, minimum=1),
                "spatial_stride": lambda x, q: _integer(x, q, minimum=1),
            })
        block_validators["target_mask_weights"] = lambda v, p: _target_mask_weights(v, p, base)
        dataset = _fields(root["dataset"], location="dataset", validators=block_validators,
            defaults={"minimum_layout_shot_fraction": None, "box_dropout_p": None, "box_jitter": None,
                     "planner_tokens": None, "reference_entities": None, "picture_type_dropout_p": None,
                     "text_admission_path": None, "reference_crops": None, "workload_cap": None})
    else:
        raise ExperimentConfigError(
            "dataset.type must be one of "
            f"{sorted(DATASET_REGISTRY.known_types())}"
        )
    validate_model_dataset_compatibility(model["type"], dataset["type"])
    if (
        dataset["type"] == "multishot_csv"
        and dataset["schema"] != "scene_window_v1"
        and dataset.get("shot2story_prompt_source") is None
        and dataset["caption_column"] is None
    ):
        raise ExperimentConfigError(
            "caption_column is required without dataset.shot2story_prompt_source"
        )
    if (
        model_entry.prompt_format is not None
        and dataset["type"] not in WAN22_BLOCK_DATASET_TYPES
        and dataset["prompt_format"] != model_entry.prompt_format
    ):
        raise ExperimentConfigError(
            f"{model['type']} dataset.prompt_format must be "
            f"{model_entry.prompt_format!r}"
        )
    if model["type"] == "wan22_diffsynth":
        if dataset["type"] in WAN22_BLOCK_DATASET_TYPES:
            pass
        elif dataset.get("schema") == "scene_window_v1":
            if (dataset["caption_column"] is not None
                    or dataset.get("shot2story_prompt_source") is not None
                    or dataset["cache_layout"] != WAN22_CACHE_LAYOUT_LATENTS_ONLY):
                raise ExperimentConfigError(
                    "scene_window_v1 requires caption_column=null, no shot2story_prompt_source, "
                    "and cache_layout=latents_only"
                )
        elif dataset.get("shot2story_prompt_source") is not None:
            if dataset["caption_column"] is not None:
                raise ExperimentConfigError(
                    "Wan2.2 raw-source data must set dataset.caption_column to null"
                )
            if dataset["prompt_format"] != WAN22_DIFFSYNTH_RAW_PROMPT_FORMAT:
                raise ExperimentConfigError(
                    "Wan2.2 raw-source data requires dataset.prompt_format="
                    f"{WAN22_DIFFSYNTH_RAW_PROMPT_FORMAT!r}"
                )
        else:
            if dataset["caption_column"] is None:
                raise ExperimentConfigError(
                    "caption_column is required without dataset.shot2story_prompt_source"
                )
            if dataset["prompt_format"] not in WAN22_DIFFSYNTH_CAPTION_PROMPT_FORMATS:
                raise ExperimentConfigError(
                    "Wan2.2 caption-JSON data dataset.prompt_format must be one of "
                    f"{sorted(WAN22_DIFFSYNTH_CAPTION_PROMPT_FORMATS)}, "
                    f"got {dataset['prompt_format']!r}"
                )
    wan22_cached = (
        model["type"] == "wan22_diffsynth"
        and dataset["type"] in WAN22_BLOCK_DATASET_TYPES
    )
    if wan22_cached:
        # 参考只由模型选项与文字dropout声明；RGB在CPU worker读取，VAE留在主进程。
        memory_prefix_given = model.get("memory_prefix") is not None
        reference_leaves = {
            "model.memory_prefix": memory_prefix_given,
            "dataset.reference_row_dropout_p": dataset["reference_row_dropout_p"] is not None,
        }
        if len(set(reference_leaves.values())) != 1:
            raise ExperimentConfigError(
                "the reference frames of segment training are declared by two leaves "
                f"together (all set or all null); got {reference_leaves}"
            )
        spatial_layout = model.get("spatial_layout")
        if spatial_layout is not None and dataset["box_dropout_p"] is None:
            raise ExperimentConfigError(
                "dataset.box_dropout_p is required when model.spatial_layout is set")
        # 同一数据选项还控制目标布局文字的整组丢弃；纯文字渲染器无结构CA模块也消费它。
        # 区域模式训练时框必须抖动（模拟规划误差）；旧加分模式不抖动，以复现已交训练。
        region_mode = spatial_layout is not None and spatial_layout["mode"] != "text_cross_attention_bias_v1"
        if region_mode != (dataset["box_jitter"] is not None):
            raise ExperimentConfigError(
                "dataset.box_jitter is required exactly for the region layout modes")
        # 规划器 token：模型侧（连接器）与数据侧（标签、特征）成对出现，读同一层。
        planner_model, planner_data = model.get("planner_tokens"), dataset["planner_tokens"]
        if (planner_model is None) != (planner_data is None):
            raise ExperimentConfigError("model.planner_tokens and dataset.planner_tokens are set together or not at all")
        if planner_model is not None:
            if planner_model["content"] != planner_data["content"]:
                raise ExperimentConfigError("model and dataset planner_tokens.content differ")
            if planner_model["content"] == "planner_features" and planner_model["feature_layer"] != planner_data["feature_layer"]:
                raise ExperimentConfigError("model and dataset planner_tokens.feature_layer differ")
            if planner_model["content"] == "person_geometry" and planner_model["max_people"] != planner_data["max_people"]:
                raise ExperimentConfigError("model and dataset planner_tokens.max_people differ")
            # token 只放本镜一处，只走全互读（点控制两组，统筹 10-31）；镜首帧接口那组不带 token。
            if model["self_attention_policy"] != "full":
                raise ExperimentConfigError("planner tokens go with self_attention_policy full")
        # 参考位置与目标时间模式必须显式相容；偏移取值已由 MemoryPrefixModelOptions 校验。
        if runtime["deepspeed"] is None:
            raise ExperimentConfigError("block training requires runtime.deepspeed")
        if memory_prefix_given:
            from mspc.core.wan22_contract import (FRAME_ENTITY_ROW_POLICIES,
                                                  MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1)
            frame_entities = model["memory_prefix"]["row_policy"] in FRAME_ENTITY_ROW_POLICIES
            if (model["memory_prefix"]["row_policy"] == MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1
                    and model["source_identity"] is not None):
                raise ExperimentConfigError("source identity routing recompiles character-only reference rows; "
                                            "it does not combine with shot-object reference rows")
            if frame_entities != (dataset["reference_entities"] is not None):
                raise ExperimentConfigError("frame-entity reference rows and dataset.reference_entities must be declared together")
            # 带参考时两种时间位置都可以：镜相位（原来各组）或整段连续原生时间（去镜相位那组）；镜首帧接口只配原生时间。
            allowed_modes = ({"global_time_native_rope_v1"} if model["self_attention_policy"] == "first_frame_interface"
                             else {MEMORY_REFERENCE_TEMPORAL_MODE, "global_time_native_rope_v1"})
            if model["temporal_position_mode"] not in allowed_modes:
                raise ExperimentConfigError(
                    f"model.memory_prefix references under self_attention_policy={model['self_attention_policy']!r} "
                    f"belong to model.temporal_position_mode in {sorted(allowed_modes)}; got "
                    f"{model['temporal_position_mode']!r}"
                )
        if memory_prefix_given and model["self_attention_policy"] == "first_frame_interface":
            raise ExperimentConfigError(
                "first_frame_interface does not implement bidirectional reference attention; "
                "replay its historical runs with their frozen code revision")
        if dataset["reference_entities"] is not None and not memory_prefix_given:
            raise ExperimentConfigError("reference_entities require actual memory-prefix RGB rows")
        if model["source_identity"] is not None and (
                not memory_prefix_given or dataset["reference_entities"] is None
                or model["self_attention_policy"] != "full" or model["spatial_layout"] is not None):
            raise ExperimentConfigError("source_identity requires frame entity captions and full ordinary Wan attention")
    training_validators: dict[str, Callable[[Any, str], Any]] = {
        "seed": lambda v, p: _integer(v, p, minimum=0),
        "task": lambda v, p: _string(v, p), "mixed_precision": lambda v, p: _string(v, p),
        "scheduler_train_timesteps": lambda v, p: _integer(v, p, minimum=1), "flow_shift": lambda v, p: _number(v, p),
        "num_epochs": lambda v, p: _integer(v, p, minimum=1), "save_steps": lambda v, p: _integer(v, p, minimum=1),
        "max_train_steps": lambda v, p: None if v is None else _integer(v, p, minimum=1),
        "gradient_accumulation_steps": lambda v, p: _integer(v, p, minimum=1),
        "use_gradient_checkpointing": _boolean, "find_unused_parameters": _boolean,
    }
    if model["type"] == "wan22_diffsynth":
        # 一步训练把整故事一次去噪，还是逐镜各抽 σ_k 逐镜反传累积。两者是不同的目标
        # 函数（联合损失对整故事全部潜 token 取均值，逐镜是 Σ n_k L_k / Σ n_k），也是
        # 消融表里「范式对照」那一行的唯一变量，所以每一份 wan22_diffsynth 配方都要
        # 自己写出来；没有默认值，不写即报错。
        training_validators["shot_forward_mode"] = lambda v, p: _one_of(
            v, p, WAN22_SHOT_FORWARD_MODES
        )
        training_validators["activation_checkpointing"] = lambda v, p: _one_of(
            v, p, WAN22_ACTIVATION_CHECKPOINTING_POLICIES
        )
        if "self_attention_cache_layers" in root["training"]:
            training_validators["self_attention_cache_layers"] = (
                lambda v, p: None if v is None else _integer(v, p, minimum=1)
            )
        if "workload_cost" in root["training"]:
            training_validators["workload_cost"] = lambda v, p: None if v is None else _fields(
                v, location=p, validators={
                    "linear_seconds_per_latent": lambda x, q: _number(x, q, minimum=0),
                    "attention_seconds_per_frame_pair": lambda x, q: _number(x, q, minimum=0),
                })
        training_validators["dit_parameter_freeze"] = parse_wan22_dit_parameter_freeze
        # 训练标量的外部展示（SwanLab）：项目名、实验名、模式（online／offline）与密钥文件四个键全写；null 是不接外部展示。
        # 本地逐步日志（logs/metrics.jsonl 与进度行）不受这个键影响，始终是事实来源。
        training_validators["loss_tracking"] = lambda v, p: _parse_loss_tracking(v, p, base)
    if dataset["type"] in WAN22_BLOCK_DATASET_TYPES:
        training_validators["step_unit"] = lambda v, p: _one_of(v, p, {"optimizer_update"})
        if "state_checkpoint" in root["training"]:
            training_validators["state_checkpoint"] = lambda v, p: _fields(v, location=p, validators={
                "keep": lambda x, q: _integer(x, q, minimum=1),
            })
        # 训练中的留出验证 loss（``mspc.training.validation_loss``）：留出窗的成员目录、每隔多少次更新量一遍、每窗用几组
        # 固定的噪声与时间步。必须写出来；null 是不量。
        training_validators["validation"] = lambda v, p: None if v is None else _fields(v, location=p, validators={
            "members_path": lambda x, q: _path(x, q, base),
            "sampling_manifest_path": lambda x, q: _path(x, q, base),
            "sample_budget": lambda x, q: _integer(x, q, minimum=1),
            "text_admission_path": lambda x, q: None if x is None else _path(x, q, base),
            "every_steps": lambda x, q: _integer(x, q, minimum=1),
            "draws": lambda x, q: _integer(x, q, minimum=1),
            **({"intermediate_panel": lambda x, q: _fields(x, location=q, validators={
                "sample_budget": lambda y, r: _integer(y, r, minimum=1),
                "draws": lambda y, r: _integer(y, r, minimum=1),
            })} if isinstance(v, Mapping) and "intermediate_panel" in v else {}),
            # 规划器 token 版：验证集自己的读历史标签（与预计算特征）；其余 token 设置沿用 dataset.planner_tokens。
            "planner_tokens": lambda x, q: None if x is None else _fields(x, location=q, validators={
                "labels_path": lambda y, r: _path(y, r, base),
                **({"features_root": lambda y, r: _path(y, r, base)} if "features_root" in x else {}),
                **({"geometry_labels_path": lambda y, r: _path(y, r, base)}
                   if dataset.get("planner_tokens") is not None
                   and dataset["planner_tokens"]["content"] == "person_geometry" else {}),
            }),
            "reference_entities": lambda x, q: None if x is None else _fields(x, location=q, validators={
                "labels_path": lambda y, r: _path(y, r, base),
            }),
        }, defaults={"planner_tokens": None, "reference_entities": None, "text_admission_path": None})
    training_defaults: dict[str, Any] = {}
    if model["type"] == "wan22_diffsynth":
        training_defaults["loss_tracking"] = None
    training = _fields(
        root["training"],
        location="training",
        validators=training_validators,
        defaults=training_defaults,
    )
    workload_cost = training.get("workload_cost")
    cache_layers = training.get("self_attention_cache_layers")
    if cache_layers is not None:
        if cache_layers > 30:
            raise ExperimentConfigError("training.self_attention_cache_layers must be <= 30")
        if training["activation_checkpointing"] != "save_self_attention":
            raise ExperimentConfigError(
                "training.self_attention_cache_layers requires activation_checkpointing='save_self_attention'"
            )
    if workload_cost is not None:
        if not any(value > 0 for value in workload_cost.values()):
            raise ExperimentConfigError("training.workload_cost needs at least one positive coefficient")
        if (dataset["type"] not in WAN22_BLOCK_DATASET_TYPES
                or model["self_attention_policy"] not in ("holocine_first_frame", "first_frame_communication")):
            raise ExperimentConfigError("training.workload_cost requires calibrated first-frame block targets "
                                        "(holocine_first_frame or first_frame_communication)")
    # 更新次数是训练预算唯一来源；拓扑确定后推导精确样本数。
    if dataset["type"] in WAN22_BLOCK_DATASET_TYPES:
        validation = training["validation"]
        if validation is not None and "intermediate_panel" in validation:
            for field in ("sample_budget", "draws"):
                if validation["intermediate_panel"][field] > validation[field]:
                    raise ExperimentConfigError(
                        f"training.validation.intermediate_panel.{field} exceeds the full panel")
        if (validation is not None and dataset["planner_tokens"] is not None
                and dataset["planner_tokens"]["content"] == "person_geometry"):
            geometry_validation = validation["planner_tokens"]
            if geometry_validation is None:
                raise ExperimentConfigError("person_geometry requires explicit validation geometry labels")
            if geometry_validation["geometry_labels_path"] == dataset["planner_tokens"]["geometry_labels_path"]:
                raise ExperimentConfigError("validation geometry cannot reuse the training geometry label file")
        if validation is not None and ((dataset["reference_entities"] is not None)
                                       != (validation["reference_entities"] is not None)):
            raise ExperimentConfigError("training and validation require their own explicit reference entity label paths")
        if validation is not None and ((dataset["text_admission_path"] is not None)
                                       != (validation["text_admission_path"] is not None)):
            raise ExperimentConfigError("training and validation require their own explicit text admission paths")
        if (validation is not None and dataset["text_admission_path"] is not None
                and dataset["text_admission_path"] == validation["text_admission_path"]):
            raise ExperimentConfigError("validation text admission cannot reuse the training admission file")
        if (validation is not None and dataset["reference_entities"] is not None
                and dataset["reference_entities"]["labels_path"] == validation["reference_entities"]["labels_path"]):
            raise ExperimentConfigError("validation reference entities cannot reuse the training label file")
        if training["shot_forward_mode"] != "joint":
            raise ExperimentConfigError(
                "block datasets denoise a whole segment at once; "
                f"training.shot_forward_mode must be 'joint', got {training['shot_forward_mode']!r}"
            )
        if training["max_train_steps"] is None:
            raise ExperimentConfigError("block training requires training.max_train_steps")
        if training["num_epochs"] != 1:
            raise ExperimentConfigError(
                "block datasets derive their sample budget from max_train_steps; "
                f"training.num_epochs must be 1, got {training['num_epochs']}"
            )
    if model["type"] == "wan22_diffsynth":
        if training["task"] != "sft":
            raise ExperimentConfigError(
                "wan22_diffsynth training.task must be 'sft'"
            )
        if training["mixed_precision"] != "no":
            raise ExperimentConfigError(
                "wan22_diffsynth training.mixed_precision must be 'no'; DeepSpeed torch_autocast owns BF16 operations"
            )
        if training["scheduler_train_timesteps"] != 1000:
            raise ExperimentConfigError(
                "wan22_diffsynth training.scheduler_train_timesteps must be 1000"
            )
        if training["flow_shift"] != 5.0:
            raise ExperimentConfigError("wan22_diffsynth training.flow_shift must be 5.0")
        if training["find_unused_parameters"]:
            raise ExperimentConfigError(
                "wan22_diffsynth training.find_unused_parameters must be false"
            )
        if (training["activation_checkpointing"] == "save_self_attention"
                and not training["use_gradient_checkpointing"]):
            raise ExperimentConfigError(
                "training.activation_checkpointing='save_self_attention' requires "
                "training.use_gradient_checkpointing=true"
            )
    pipeline = root["pipeline_inputs"]
    if model["type"] == "wan22_diffsynth":
        if pipeline is None:
            raise ExperimentConfigError("pipeline_inputs must be a mapping for wan22_diffsynth")
        pipeline = _fields(pipeline, location="pipeline_inputs", validators={
            "tiled": _boolean, "tile_size": _positive_integer_pair, "tile_stride": _positive_integer_pair,
        })
    if (model_entry.requires_pipeline_inputs is True) != (pipeline is not None):
        if not wan22_cached:
            raise ExperimentConfigError(
                "pipeline_inputs must be a mapping for DiffSynth models and null otherwise"
            )
    optimization = _parse_optimization(root["optimization"])
    dataloader = _fields(
        root["dataloader"],
        location="dataloader",
        validators={
            "per_device_batch_size": _single_complete_sample_batch_size,
            "split_batches": _boolean, "drop_last": _boolean,
            "dataset_num_workers": lambda v, p: _integer(v, p, minimum=0), "prefetch_factor": lambda v, p: _integer(v, p, minimum=1),
            "video_decode_num_threads": lambda v, p: _integer(v, p, minimum=1),
            "persistent_workers": _boolean, "pin_memory": _boolean,
        },
    )
    if dataloader["split_batches"]:
        raise ExperimentConfigError(
            "dataloader.split_batches must be false so every process receives "
            "one complete sample"
        )
    if not dataloader["drop_last"]:
        raise ExperimentConfigError(
            "dataloader.drop_last must be true so every optimizer micro-step "
            "contains one complete sample from every process"
        )
    timing = _fields(root["timing"], location="timing", validators={
        "log_steps": lambda v, p: _integer(v, p, minimum=1), "log_every": lambda v, p: _integer(v, p, minimum=0), "sync_cuda": _boolean,
    })
    output = _fields(root["output"], location="output", validators={
        "root": lambda v, p: _path(v, p, base),
    })
    checkpoint = (
        None
        if root["checkpoint"] is None
        else _path(root["checkpoint"], "checkpoint", base)
    )
    # checkpoint=null 从 model.paths 中明确列出的原生权重初始化；参考条件不决定训练起点。

    inference = _parse_wan22_diffsynth_inference(root["inference"], base)
    if wan22_cached and inference["enabled"]:
        raise ExperimentConfigError(
            "block training uses an explicit storyboard panel for inference via renderer_checkpoint_diagnostics; "
            "the legacy CSV inference.enabled entry is unsupported"
        )

    is_wan22_cache_source = (
        model["type"] == "wan22_diffsynth"
        and dataset["type"] == "multishot_csv"
        and dataset.get("cache_layout") in WAN22_CACHE_LAYOUTS
    )
    raw_precompute = root.get("precompute")
    if is_wan22_cache_source:
        if raw_precompute is None:
            raise ExperimentConfigError(
                "Wan2.2 cache-source configs require precompute "
                "(encode_batch_size and decode DataLoader knobs)"
            )
        precompute = _parse_precompute(raw_precompute, base)
    else:
        if raw_precompute is not None:
            raise ExperimentConfigError(
                "precompute is only valid on Wan2.2 multishot_csv cache-source configs"
            )
        precompute = None
    normalized = {
        "schema_version": version, "kind": "train", "identity": identity, "task": task,
        "runtime": runtime, "environment": environment, "nccl": nccl, "launch": launch,
        "trainer": trainer,
        "model": model, "dataset": dataset, "training": training,
        "pipeline_inputs": pipeline, "optimization": optimization, "dataloader": dataloader,
        "timing": timing, "output": output, "inference": inference, "checkpoint": checkpoint,
        "precompute": precompute,
    }
    return ExperimentConfig(
        schema_version=version,
        source_path=source,
        resolved=_freeze(normalized),
    )


TRAIN_EXPERIMENT_FIELDS = frozenset({
    "schema_version", "kind", "identity", "task", "runtime", "environment",
    "nccl", "launch", "trainer", "model", "dataset", "training", "pipeline_inputs",
    "optimization", "dataloader", "timing", "output", "inference", "checkpoint",
    "precompute",
})
_PYTHON_CONFIG_IGNORED = frozenset(
    {
        "__builtins__",
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__annotations__",
    }
)


def _python_config_like_keys(namespace: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key, value in namespace.items():
        if key in _PYTHON_CONFIG_IGNORED or key.startswith("_"):
            continue
        if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
            continue
        if inspect.isbuiltin(value):
            continue
        keys.add(key)
    return keys


def _load_python_experiment(source: Path) -> Any:
    try:
        namespace = runpy.run_path(
            str(source),
            run_name="__mspc_experiment_config__",
        )
    except Exception as exc:
        raise ExperimentConfigError(
            f"failed to execute Python config: {source}"
        ) from exc
    unknown = sorted(
        _python_config_like_keys(namespace) - TRAIN_EXPERIMENT_FIELDS - {"EXPERIMENT_CONFIG"}
    )
    if unknown:
        raise ExperimentConfigError(
            f"Python config contains unknown top-level keys: {unknown}"
        )
    if "EXPERIMENT_CONFIG" in namespace:
        return namespace["EXPERIMENT_CONFIG"]
    return {
        key: namespace[key]
        for key in TRAIN_EXPERIMENT_FIELDS
        if key in namespace
    }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a Python, YAML, or JSON config and apply the same strict schema."""

    try:
        source = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExperimentConfigError(f"config path cannot be resolved: {path}") from exc
    if not source.is_file():
        raise ExperimentConfigError(f"config path is not a file: {source}")
    if source.suffix == ".py":
        raw = _load_python_experiment(source)
    else:
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExperimentConfigError(f"failed to read config: {source}") from exc
    if source.suffix == ".json":
        try:
            raw = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise ExperimentConfigError(f"invalid JSON config: {source}") from exc
    elif source.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("YAML experiment configs require PyYAML") from exc

        class UniqueLoader(yaml.SafeLoader):
            pass

        def construct(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
            return _reject_duplicate_pairs(
                [(loader.construct_object(k, deep=deep), loader.construct_object(v, deep=deep)) for k, v in node.value]
            )

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
        try:
            raw = yaml.load(text, Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ExperimentConfigError(f"invalid YAML config: {source}") from exc
    elif source.suffix != ".py":
        raise ExperimentConfigError(
            "config extension must be .py, .json, .yaml, or .yml"
        )
    return parse_experiment_mapping(raw, source_path=source)


def _reject_duplicate_pairs(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentConfigError(f"duplicate config key: {key!r}")
        result[key] = value
    return result
