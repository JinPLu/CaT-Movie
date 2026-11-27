"""Per-family experiment model section parsers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mspc.config.experiment import (
    ExperimentConfigError,
    _absolute_local_path,
    _fields,
    _mapping,
    _one_of,
    _parse_wan22_diffsynth_paths,
    _string,
)
from mspc.core.wan22_contract import (MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1,
                                      ROUTED_MEMORY_ATTENTION_POLICIES,
                                      WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES, WAN22_SELF_ATTENTION_POLICIES)
from mspc.models.memory_prefix_options import MemoryPrefixModelOptions
from mspc.models.planner_tokens import PlannerTokenOptions
from mspc.models.first_frame_tokens import FirstFrameTwinOptions, ShotRegisterOptions
from mspc.models.spatial_layout_inputs import SpatialLayoutOptions
from mspc.models.source_identity import SourceIdentityOptions


@dataclass(frozen=True)
class Wan22DiffSynthModel:
    type: str
    diffsynth_root: str
    temporal_position_mode: str
    self_attention_policy: str
    paths: Mapping[str, Any]
    #: 完整离线训练context缓存的manifest；None保留原live文字路径。
    text_context_cache: str | None = None
    memory_prefix: Mapping[str, Any] | None = None
    spatial_layout: Mapping[str, Any] | None = None
    planner_tokens: Mapping[str, Any] | None = None
    source_identity: Mapping[str, Any] | None = None
    #: 首帧组附加 token（``mspc.models.first_frame_tokens``）：孪生帧与镜 register 二选一，只配首帧通信；None 为不加。
    first_frame_twin: Mapping[str, Any] | None = None
    shot_registers: Mapping[str, Any] | None = None
    #: 训练与推理都从目标文字里省掉的部分（``shot_text.TARGET_TEXT_OMISSIONS``）。
    omit_target_text: tuple[str, ...] = ()
    #: Wan2.2-A14B 的去噪专家（``mspc.models.noise_expert``）：这个 run 训哪个专家、边界在哪；None 为单模型底座。
    expert: Mapping[str, Any] | None = None


def _parse_wan22_memory_prefix(value: Any, location: str) -> dict[str, Any]:
    """Parse ``model.memory_prefix``：整块交给 :class:`MemoryPrefixModelOptions`。

    ``null`` 表示这条臂不存在，由调用方在进这里之前处理。对象一旦出现，全部键都要在、
    取值都要合法——缺键或越界在 carrier 里炸，这里不再枚举、不再写第二份范围检查。
    """

    try:
        return asdict(
            MemoryPrefixModelOptions.from_mapping(_mapping(value, location), location)
        )
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(str(error)) from error


def _parse_omit_target_text(value: Any, location: str) -> tuple[str, ...]:
    from mspc.data.shot_text import TARGET_TEXT_OMISSIONS
    if (not isinstance(value, (list, tuple)) or len(set(value)) != len(value)
            or not all(isinstance(item, str) and item in TARGET_TEXT_OMISSIONS for item in value)):
        raise ExperimentConfigError(f"{location} must list distinct items of {sorted(TARGET_TEXT_OMISSIONS)}")
    return tuple(value)


def _parse_spatial_layout(value: Any, location: str) -> dict[str, Any]:
    try:
        return SpatialLayoutOptions.from_mapping(_mapping(value, location)).to_mapping()
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(f"{location}: {error}") from error


def _parse_planner_tokens(value: Any, location: str) -> dict[str, Any]:
    try:
        options = asdict(PlannerTokenOptions.from_mapping(_mapping(value, location), location))
        return {key: options[key] for key in value}
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(str(error)) from error


def _parse_first_frame_twin(value: Any, location: str) -> dict[str, Any]:
    try:
        return asdict(FirstFrameTwinOptions.from_mapping(_mapping(value, location), location))
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(str(error)) from error


def _parse_shot_registers(value: Any, location: str) -> dict[str, Any]:
    try:
        return asdict(ShotRegisterOptions.from_mapping(_mapping(value, location), location))
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(str(error)) from error


def _parse_noise_expert(value: Any, location: str) -> dict[str, Any]:
    from mspc.models.noise_expert import NoiseExpertOptions
    try:
        return asdict(NoiseExpertOptions.from_mapping(_mapping(value, location)))
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(f"{location}: {error}") from error


def _parse_source_identity(value: Any, location: str) -> dict[str, Any]:
    try:
        return asdict(SourceIdentityOptions.from_mapping(_mapping(value, location)))
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError(f"{location}: {error}") from error


VENDORED_DIFFSYNTH_ROOT_PARTS = ("third_party", "wan22_runtime")


def _vendored_diffsynth_root(value: Any, location: str, base_dir: Path) -> str:
    """Accept only the in-repository vendored DiffSynth tree.

    方案 A：整个项目只留一棵执行树 ``<checkout>/third_party/wan22_runtime``。
    共享盘上的 ``.runtime/wan22`` 物化是只读归档，不再是可声明的执行树。

    执行面上 checkout 的绝对前缀各不相同（主服务器
    ``${MSPC_SHARED_ROOT}/Storyboard``、训练集群
    ``${MSPC_SHARED_ROOT}/Storyboard``、本机 ``/Users/.../Storyboard``），所以判据
    只能落在末两级路径上，不能硬编码任何一台机器的前缀。
    """

    resolved = _absolute_local_path(value, location, base_dir)
    if Path(resolved).parts[-2:] != VENDORED_DIFFSYNTH_ROOT_PARTS:
        expected = "/".join(VENDORED_DIFFSYNTH_ROOT_PARTS)
        raise ExperimentConfigError(
            f"{location} must be the vendored execution tree <checkout>/{expected}, "
            f"got {resolved!r}; the .runtime/wan22 materialization is a read-only "
            "archive"
        )
    return resolved


def parse_wan22_diffsynth_model(
    raw: Mapping[str, Any], base_dir: Path
) -> dict[str, Any]:
    parsed = _fields(
        raw,
        location="model",
        validators={
            "type": _string,
            "diffsynth_root": lambda v, p: _vendored_diffsynth_root(v, p, base_dir),
            "temporal_position_mode": lambda v, p: _one_of(
                v, p, WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES
            ),
            "self_attention_policy": lambda v, p: _one_of(
                v, p, WAN22_SELF_ATTENTION_POLICIES
            ),
            "paths": lambda v, p: _parse_wan22_diffsynth_paths(v, base_dir, location=p),
            "text_context_cache": lambda v, p: (
                None if v is None else _absolute_local_path(v, p, base_dir)
            ),
            "memory_prefix": lambda v, p: (
                None if v is None else _parse_wan22_memory_prefix(v, p)
            ),
            "spatial_layout": lambda v, p: None if v is None else _parse_spatial_layout(v, p),
            "planner_tokens": lambda v, p: None if v is None else _parse_planner_tokens(v, p),
            "source_identity": lambda v, p: None if v is None else _parse_source_identity(v, p),
            "first_frame_twin": lambda v, p: None if v is None else _parse_first_frame_twin(v, p),
            "shot_registers": lambda v, p: None if v is None else _parse_shot_registers(v, p),
            "omit_target_text": _parse_omit_target_text,
            "expert": lambda v, p: None if v is None else _parse_noise_expert(v, p),
        },
        defaults={
            "expert": None,
            "omit_target_text": [],
            "text_context_cache": None,
            "memory_prefix": None,
            "spatial_layout": None,
            "planner_tokens": None,
            "source_identity": None,
            "first_frame_twin": None,
            "shot_registers": None,
        },
    )
    if parsed["planner_tokens"] is not None and parsed["spatial_layout"] is not None:
        raise ExperimentConfigError("model.planner_tokens replaces model.spatial_layout; set it to null")
    if (parsed["self_attention_policy"] in ("holocine_first_frame", "holocine_own_shot_only", "first_frame_communication")
            and parsed["temporal_position_mode"] != "global_time_native_rope_v1"):
        raise ExperimentConfigError(f"{parsed['self_attention_policy']} requires native RoPE")
    # 参考注意力与 DiT 自注意力的唯一配对检查：holocine / 首帧通信的参考只能按镜路由；按镜路由的参考配这两者
    # （以及 MultiShotMaster 式全互读消融）；路由这组不带规划器 token、源身份路由与空间布局。
    routed = (parsed["memory_prefix"] is not None
              and parsed["memory_prefix"]["attention_policy"] in ROUTED_MEMORY_ATTENTION_POLICIES)
    if (parsed["memory_prefix"] is not None
            and parsed["self_attention_policy"] in ("holocine_first_frame", "holocine_own_shot_only",
                                                    "first_frame_communication")
            and not routed):
        raise ExperimentConfigError(f"{parsed['self_attention_policy']} reads references only under "
                                    f"model.memory_prefix.attention_policy in {ROUTED_MEMORY_ATTENTION_POLICIES!r}")
    if routed and parsed["self_attention_policy"] not in ("holocine_first_frame", "holocine_own_shot_only",
                                                          "first_frame_communication", "full"):
        raise ExperimentConfigError(f"routed reference policies {ROUTED_MEMORY_ATTENTION_POLICIES!r} need "
                                    "holocine_first_frame, first_frame_communication or full self-attention")
    if (routed and parsed["memory_prefix"]["attention_policy"] == MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1
            and parsed["self_attention_policy"] != "holocine_first_frame"):
        raise ExperimentConfigError("first-latent reference queries require holocine_first_frame self-attention")
    for key in ("first_frame_twin", "shot_registers"):
        if parsed[key] is not None and parsed["self_attention_policy"] != "first_frame_communication":
            raise ExperimentConfigError(f"model.{key} belongs to first_frame_communication self-attention")
    if parsed["first_frame_twin"] is not None and parsed["shot_registers"] is not None:
        raise ExperimentConfigError("model.first_frame_twin and model.shot_registers are separate experiments")
    if routed and any(parsed[key] is not None for key in ("planner_tokens", "source_identity", "spatial_layout")):
        raise ExperimentConfigError("routed reference attention takes no planner tokens, source identity routing "
                                    "or spatial layout")
    return asdict(Wan22DiffSynthModel(**parsed))


MODEL_PARSE_REGISTRY: dict[str, Callable[[Mapping[str, Any], Path], dict[str, Any]]] = {
    "wan22_diffsynth": parse_wan22_diffsynth_model,
}


def parse_model(value: Any, base_dir: Path) -> dict[str, Any]:
    """Validate and normalize the experiment ``model`` section."""

    raw = _mapping(value, "model")
    if "type" not in raw:
        raise ExperimentConfigError("model is missing required fields: ['type']")
    model_type = _string(raw["type"], "model.type")
    try:
        parser = MODEL_PARSE_REGISTRY[model_type]
    except KeyError as exc:
        raise ExperimentConfigError(
            "model.type must be one of "
            f"{sorted(MODEL_PARSE_REGISTRY)}"
        ) from exc
    return parser(raw, base_dir)
