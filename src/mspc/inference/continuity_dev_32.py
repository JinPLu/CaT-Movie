"""Build Wan2.2 T2V case inputs from the continuity_dev_32 / main_table_96 benchmark JSON."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from mspc.config.shared_root import expand_shared_root_token
from mspc.config.experiment import WAN22_CACHED_SPACETIME_CONTRACT
from mspc.data.frame_planner import (
    pad_rgb_length_to_vae,
    rgb_to_latent_shot_groups,
    validate_shot_groups,
)
from mspc.data.multishot import (
    PROMPT_FORMAT_STORY_NOW_V1,
    RENDERED_PROMPT_CONTRACT,
    build_prompts_from_fields,
)
from mspc.core.wan22_contract import (
    GLOBAL_TIME_NATIVE_ROPE_V1,
)


#: 题面每镜可带的第二份文字：本镜文字＋设定、不带关系行。拼装过设定的题面
#: （``story_visual_setting.build_visual_setting_benchmark``）每镜都写；这一镜被当作记忆帧喂入时，
#: 仅保留历史题面的来源元数据；当前参考消费者不读它，只使用固定角色句。
MEMORY_ROW_PROMPT_FIELD = "memory_row_prompt"

# schema_version strings are data-contract values and stay unchanged.
CONTINUITY_DEV_32_BENCHMARK_SCHEMA = "bernini_continuous_planner_dev32_v1"
MAIN_TABLE_96_BENCHMARK_SCHEMA = "bernini_continuous_planner_formal96_v1"
#: 真实留出池的开发故事基准（首轮视觉线索验证的分母来源）。这里只登记字符串与白名单，
#: 这份 JSON 自身的字段校验模块另有归属。
REAL_HOLDOUT_96_BENCHMARK_SCHEMA = "real_holdout_96_v1"
#: 任意一份语料 case 清单编出的评测子集，同一套
#: case 形状。它不是留出集（训练见没见过未知），读数只能按「池内诊断」报；收进渲染链是为了
#: 事件中段扩充集的依从探针。
CORPUS_EVAL_SUBSET_BENCHMARK_SCHEMA = "corpus_eval_subset_v1"
#: 论文报告集：约 100 条手写剧本，Subject N＋属性，全部硬切，另带 subjects/locations/
#: present_subjects/events 供剧本编译断言。v1（2026-09-24 起替换 main_table_96 与 real_holdout_96
#: 的报告职能，冻结于 ``9e0e2d53``）；v2（09-30 整体重写，每镜另带景别、机位与镜界类型）替换 v1 的
#: 报告职能。这里只读它们与其他 benchmark 共有的 case 形状；严格校验在
#: ``mspc.benchmark.script_state_bench``。
SCRIPT_STATE_BENCH_V1_SCHEMA = "script_state_bench_v1"
SCRIPT_STATE_BENCH_V2_SCHEMA = "script_state_bench_v2"
#: 文字方位服从探针：从 v2 的主体
#: 与地点编出的两镜「场景」，每镜是独立出片的一个探针，只比两镜的文字差别。这里只登记字符串；题面由
#: ``mspc.evaluation.spatial_layout_probe`` 编，不进任何报告集。
SPATIAL_TEXT_OBEDIENCE_PROBE_SCHEMA = "spatial_text_obedience_probe_v1"
ACCEPTED_BENCHMARK_SCHEMAS = frozenset(
    {
        CONTINUITY_DEV_32_BENCHMARK_SCHEMA,
        MAIN_TABLE_96_BENCHMARK_SCHEMA,
        REAL_HOLDOUT_96_BENCHMARK_SCHEMA,
        CORPUS_EVAL_SUBSET_BENCHMARK_SCHEMA,
        SCRIPT_STATE_BENCH_V1_SCHEMA,
        SCRIPT_STATE_BENCH_V2_SCHEMA,
        SPATIAL_TEXT_OBEDIENCE_PROBE_SCHEMA,
    }
)
CONTINUITY_DEV_32_GENERATE_SCHEMA = "wan22_controlled32_generate_v1"
#: manifest 里 ``shots[].frame_count`` 是成片自己的逐镜帧数。字符串是数据契约值，不随语义改名：
#: 2026-10-05 起成片按 :func:`crop_frames_to_shot_lengths` 裁回剧本镜长，这个数就是剧本镜长；
#: 此前成片与这个数都是补到 4n+1 之后的长度。
CONTINUITY_DEV_32_FRAME_COUNT_SOURCE = "generated_vae_aligned"
WAN_BASELINE_ARM_ID = "wan_baseline"
WAN_BASELINE_WEIGHT_SOURCE = "pretrained_model_paths"
WAN_BASELINE_GENERATION_PARADIGM = "single_shot_text_to_video_concatenated"
MULTISHOT_GENERATION_PARADIGM = "wan22_diffsynth_pure_t2v_multishot_continuity_dev_32"
#: 逐镜自回归 + 冻结 VLM 线索的主方法 B（第一章/二章）：镜 k 生成完取第 0 帧、缩到
#: 224²、喂给镜 k+1..K，独立生成后 ffmpeg 拼接。与 ``WAN_BASELINE_GENERATION_PARADIGM``
#: 同属「逐镜生成 + 拼接」这一族，但 wan_baseline 镜间零条件（连文字都各镜独立），本档
#: 镜间靠冻结 VLM 线索块（09-24 前还有共享的 ``Story:{global}`` 文字段）；两者分属不同 arm_id
#: 与不同 code path，判据互不覆盖（见 ``generate_continuity_dev_32`` 的路由）。
GENERATION_PARADIGM_PER_SHOT_CONCAT = "per_shot_concat"
#: 参考只消费已生成镜的末帧单帧编码；无参考作为条件丢弃形态。
MEMORY_ARM_NONE = "none"
# 历史离线诊断的来源名仍供已有分析工具读取，不能作为当前渲染任务的memory_arm。
MEMORY_ARM_ORACLE_PREV = "oracle_prev"
MEMORY_ARM_SAME_STORY_OTHER = "same_story_other"
MEMORY_ARM_OTHER_STORY = "other_story"
MEMORY_ARMS = (MEMORY_ARM_NONE,)
REAL_LATENT_MEMORY_ARMS = (
    MEMORY_ARM_ORACLE_PREV,
    MEMORY_ARM_SAME_STORY_OTHER,
    MEMORY_ARM_OTHER_STORY,
)
ACCEPTED_MEMORY_ARMS = MEMORY_ARMS
GENERATION_PATH_MEMORY_PREFIX = "b0_memory_prefix_per_shot_concat"
#: 给定每镜首帧出片。冻结原生
#: TI2V-5B 逐镜出片，每镜第 0 潜帧经原生首帧口钉住（``Wan22InferenceRuntime.generate_case(first_frame=...)``），
#: 只差这一帧从哪来：
#:
#: - ``real_first``：本镜真片第 0 潜帧（latent 缓存里训练读的那一份，不经解码再编码）——A1；
#: - ``real_first_continuation_prev_last``：镜界 k−1→k 在真片上是续拍（两镜真片第 0 潜帧的
#:   ``aligned_minus_shuffled_cosine`` 超过 ``TAU_FIRST_FIRST``，:func:`real_continuation_boundaries`）
#:   时，镜 k 取本臂上一镜成片末帧的单帧编码；其余镜同 ``real_first``——A2；
#: - ``other_story_first``：另一个故事（:func:`other_story_donors`）第 0 镜的真片第 0 潜帧，文字仍是本镜
#:   的——X0 的冲突臂，量首帧口是否压过文字。
#: - ``real_first_coarse``：本镜真片第 0 潜帧先单帧解码，像素长宽各盒式降到 1/8 再双三次升回，
#:   用同一 ``encode_single_frame`` 钉住——粗画布臂 C。倍数写死，不做扫描。
#:
#: 「不给首帧」（A0）就是 ``wan_baseline`` 那一档：同一 runtime、同一逐镜拼接、同一 seed。
FIRST_FRAME_ARM_REAL = "real_first"
FIRST_FRAME_ARM_CONTINUATION_PREV_LAST = "real_first_continuation_prev_last"
FIRST_FRAME_ARM_OTHER_STORY = "other_story_first"
FIRST_FRAME_ARM_EVENT = "event_first"
FIRST_FRAME_ARM_COARSE = "real_first_coarse"
COARSE_CANVAS_FACTOR = 8
FIRST_FRAME_ARMS = (
    FIRST_FRAME_ARM_REAL,
    FIRST_FRAME_ARM_CONTINUATION_PREV_LAST,
    FIRST_FRAME_ARM_OTHER_STORY,
    FIRST_FRAME_ARM_EVENT,
    FIRST_FRAME_ARM_COARSE,
)
GENERATION_PATH_FIRST_FRAME = "native_first_frame_per_shot_concat"
#: 本轮五档载体只用 all_readable：没有版本路由，条件块内不做逐行掩码。
PER_SHOT_CUE_ARM = "all_readable"
#: 线索帧几何（§1.4.1 shot_first_frame_v1）：自生成前缀帧沿用同一套数字，方便与训练侧
#: 的真实首帧在编码器输入分布上对齐；两者是不同的生产者，只共享这三个常量。
CUE_FRAME_SIZE = 224
CUE_FRAME_JPEG_QUALITY = 95
MSM_RENDER_CONTRACT = {
    "target_height": 480,
    "target_width": 832,
    "num_inference_steps": 50,
    "sigma_shift": 5.0,
    "cfg_scale": 5.0,
}
MSM_RENDER_FIELDS = tuple(MSM_RENDER_CONTRACT)


def load_continuity_dev_32_benchmark(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"continuity_dev_32 benchmark is missing: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("continuity_dev_32 benchmark must be an object")
    schema = raw.get("schema_version")
    if schema not in ACCEPTED_BENCHMARK_SCHEMAS:
        raise ValueError(
            "benchmark.schema_version must be one of "
            f"{sorted(ACCEPTED_BENCHMARK_SCHEMAS)!r}; got {schema!r}"
        )
    if raw.get("evaluation_only") is not True:
        raise ValueError("continuity_dev_32 benchmark.evaluation_only must be true")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("continuity_dev_32 benchmark.cases must be a non-empty array")
    return source, raw


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_source_metadata(value: object, label: str) -> dict[str, Any] | None:
    """``real_holdout_96_v1`` 独有的真实源片路径与逐镜帧区间（S1/S2 的物质基础）。

    ``source_metadata`` 这个字段名在两份 benchmark 上装着**两种不同的东西**：
    ``main_table_96``（controlled64 从无真实视频）拿它记「这条受控题面的设计出处」
    （``source_kind``/``source_dataset``/…，没有 ``video_path``）；``real_holdout_96_v1``
    拿它记真实源片。两份 benchmark 共用同一个读取函数，这里只认后一种形状——
    ``video_path`` 键不存在就返回 ``None``（S0 三档因此不受影响），键存在但形状不对
    才当场抛，不静默放过一条写错的真实源片记录。

    **实测（09-08）：这两个字段与它们指向的源片自身帧序不自洽**——抽查 12 条
    dev-96，11 条 ``max(source_shot_groups)`` 越过源片实际帧数。
    它们只在 benchmark 内部自洽（``source_shot_groups`` 每段长度恰好等于对应
    ``shots[].frame_count``），``real_holdout_96.py`` 的 schema 校验用的正是这条内部
    自洽性，不依赖源片。**因此不得用这两个字段的值去源片切帧**；这里仍然解析、校验它们，
    只是为了保留「这条 case 有没有真实视频」这个判据（``is not None``）以及
    ``real_holdout_96.py`` 自己的 schema 契约。
    """

    if value is None or not isinstance(value, Mapping) or "video_path" not in value:
        return None
    video_path = _text(value.get("video_path"), f"{label}.video_path")
    groups = value.get("source_shot_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{label}.source_shot_groups must be a non-empty array")
    parsed_groups: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        if (
            not isinstance(group, list)
            or len(group) != 2
            or any(type(item) is not int for item in group)
            or group[0] < 0
            or group[1] <= group[0]
        ):
            raise ValueError(
                f"{label}.source_shot_groups[{index}] must be a [start, end) integer pair "
                "with start < end"
            )
        parsed_groups.append((group[0], group[1]))
    return {"video_path": video_path, "source_shot_groups": parsed_groups}


def _optional_entity_descriptions(value: object, label: str) -> list[str] | None:
    """Validate an optional per-case entity roster, if the benchmark carries one.

    Neither ``dev32.json`` nor ``formal96_candidate.json`` defines
    ``entity_descriptions`` on any case today (checked: 0/32 and 0/96) -- this
    still reads the field so a future benchmark revision that adds real rosters
    is picked up automatically, instead of the field being silently dropped
    here regardless of what the JSON contains.
    """

    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array of strings when present")
    return [_text(item, f"{label}[{index}]") for index, item in enumerate(value)]


def wan22_shot_plan_from_frame_counts(
    frame_counts: Sequence[int],
    *,
    vae_temporal_stride: int = int(WAN22_CACHED_SPACETIME_CONTRACT["vae_temporal_stride"]),
    max_rgb_frames: int = int(WAN22_CACHED_SPACETIME_CONTRACT["max_rgb_frames"]),
    label: str,
) -> dict[str, Any]:
    if not frame_counts:
        raise ValueError(f"{label}: frame_counts must be non-empty")
    aligned: list[int] = []
    for index, count in enumerate(frame_counts):
        frames = _positive_int(count, f"{label}.frame_counts[{index}]")
        aligned.append(pad_rgb_length_to_vae(frames, vae_temporal_stride))
    shot_groups: list[list[int]] = []
    cursor = 0
    for length in aligned:
        shot_groups.append([cursor, cursor + length])
        cursor += length
    if cursor > max_rgb_frames:
        raise ValueError(
            f"{label}: aligned train_rgb_frames={cursor} exceeds max_rgb_frames={max_rgb_frames}"
        )
    validate_shot_groups(shot_groups, label)
    latent_shot_groups = rgb_to_latent_shot_groups(shot_groups, vae_temporal_stride)
    return {
        "shot_groups": shot_groups,
        "latent_shot_groups": latent_shot_groups,
        "train_rgb_frames": cursor,
        "train_latent_frames": latent_shot_groups[-1][-1],
        "aligned_rgb_lengths": aligned,
        "vae_temporal_stride": vae_temporal_stride,
        "max_rgb_frames": max_rgb_frames,
    }


def crop_frames_to_shot_lengths(
    frames: Sequence[Any],
    *,
    shot_groups: Sequence[Sequence[int]],
    shot_frame_counts: Sequence[int],
    vae_temporal_stride: int = int(WAN22_CACHED_SPACETIME_CONTRACT["vae_temporal_stride"]),
    label: str,
) -> list[Any]:
    """把按 4n+1 补齐后生成（或解码）的帧裁回每镜剧本镜长，次序即镜序。

    与 MultiShotMaster 上游同一做法：每镜补到 4n+1 再去噪，存盘前只留每镜前
    ``shot_frame_counts[i]`` 帧（上游 ``util.get_user_wanted_frames``；本项目此前把补齐后的长度当
    镜长写进 manifest）。
    ``shot_groups`` 是补齐后的 RGB 区间，从 0 起、首尾相接、末端等于 ``len(frames)``；
    每镜剧本镜长补齐后必须正好等于该区间长度，否则是调用方把两份镜长对错了。
    """

    if len(shot_groups) != len(shot_frame_counts) or not shot_groups:
        raise ValueError(
            f"{label}: {len(shot_frame_counts)} shot frame count(s) for "
            f"{len(shot_groups)} shot group(s)"
        )
    cursor = 0
    kept: list[Any] = []
    for index, (group, count) in enumerate(zip(shot_groups, shot_frame_counts)):
        start, end = int(group[0]), int(group[1])
        if start != cursor:
            raise ValueError(f"{label}: shot_groups[{index}] starts at {start}, expected {cursor}")
        frames_in_shot = _positive_int(count, f"{label}.shot_frame_counts[{index}]")
        if pad_rgb_length_to_vae(frames_in_shot, vae_temporal_stride) != end - start:
            raise ValueError(
                f"{label}: shot {index} has {frames_in_shot} script frame(s), which pad to "
                f"{pad_rgb_length_to_vae(frames_in_shot, vae_temporal_stride)}, but its "
                f"generated span is {end - start}"
            )
        kept.extend(frames[start : start + frames_in_shot])
        cursor = end
    if cursor != len(frames):
        raise ValueError(f"{label}: shot_groups end at {cursor}, got {len(frames)} frame(s)")
    return kept


def script_shot_frame_counts(case_inputs: Mapping[str, Any]) -> list[int]:
    """一份 case 输入（:func:`case_inputs_from_benchmark_case` 或 :func:`single_shot_inputs_from_case`）
    每镜的剧本镜长，次序即镜序。"""

    return [int(shot["frame_count"]) for shot in case_inputs["shots"]]


def case_inputs_from_benchmark_case(
    case: Mapping[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    case_id = _text(case.get("case_id"), "case_id")
    row_label = label or case_id
    story_prompt = _text(case.get("story_prompt"), f"{row_label}.story_prompt")
    shots = case.get("shots")
    shot_count = _positive_int(case.get("shot_count"), f"{row_label}.shot_count")
    if not isinstance(shots, list) or len(shots) != shot_count:
        raise ValueError(f"{row_label}.shots must contain shot_count entries")
    shot_ids: list[str] = []
    shot_prompts: list[str] = []
    memory_row_prompts: list[str] = []
    frame_counts: list[int] = []
    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise ValueError(f"{row_label}.shots[{index}] must be an object")
        shot_id = _text(shot.get("shot_id"), f"{row_label}.shots[{index}].shot_id")
        if shot_id in shot_ids:
            raise ValueError(f"{row_label} has duplicate shot_id={shot_id!r}")
        shot_ids.append(shot_id)
        shot_prompts.append(_text(shot.get("prompt"), f"{row_label}.shots[{index}].prompt"))
        if MEMORY_ROW_PROMPT_FIELD in shot:
            memory_row_prompts.append(
                _text(shot[MEMORY_ROW_PROMPT_FIELD], f"{row_label}.shots[{index}].{MEMORY_ROW_PROMPT_FIELD}")
            )
        frame_counts.append(
            _positive_int(shot.get("frame_count"), f"{row_label}.shots[{index}].frame_count")
        )
    if memory_row_prompts and len(memory_row_prompts) != len(shots):
        raise ValueError(
            f"{row_label}: {len(memory_row_prompts)} of {len(shots)} shots carry "
            f"{MEMORY_ROW_PROMPT_FIELD!r}; an assembled benchmark writes it on every shot"
        )
    plan = wan22_shot_plan_from_frame_counts(frame_counts, label=row_label)
    event_sources = case.get("event_first_source_shots")
    if event_sources is not None:
        if not isinstance(event_sources, list) or len(event_sources) != shot_count:
            raise ValueError(f"{row_label}: event_first_source_shots must cover every shot")
        for check, source in enumerate(event_sources):
            if source is not None and (type(source) is not int or not 0 <= source < check):
                raise ValueError(f"{row_label}: invalid event source {source} for shot {check}")
    prompts = build_prompts_from_fields(shot_prompts, prompt_format=PROMPT_FORMAT_STORY_NOW_V1)
    return {
        "case_id": case_id,
        "prompts": prompts,
        "story_prompt": story_prompt,
        "shot_prompts": shot_prompts,
        "event_first_source_shots": event_sources,
        "shot_ids": shot_ids,
        "target_fps": int(WAN22_CACHED_SPACETIME_CONTRACT["target_fps"]),
        "max_rgb_frames": plan["max_rgb_frames"],
        "vae_temporal_stride": plan["vae_temporal_stride"],
        "shot_groups": plan["shot_groups"],
        "latent_shot_groups": plan["latent_shot_groups"],
        "train_rgb_frames": plan["train_rgb_frames"],
        "train_latent_frames": plan["train_latent_frames"],
        "entity_descriptions": _optional_entity_descriptions(
            case.get("entity_descriptions"), f"{row_label}.entity_descriptions"
        ),
        "continuity": case.get("continuity") if isinstance(case.get("continuity"), Mapping) else {},
        "source_metadata": _optional_source_metadata(
            case.get("source_metadata"), f"{row_label}.source_metadata"
        ),
        "shots": [
            {
                "shot_id": shot_id,
                "prompt": prompt,
                "frame_count": length,
                # 只有拼装过设定的题面（``story_visual_setting.build_visual_setting_benchmark``）带它。
                **(
                    {MEMORY_ROW_PROMPT_FIELD: memory_row_prompts[index]}
                    if memory_row_prompts
                    else {}
                ),
            }
            for index, (shot_id, prompt, length) in enumerate(
                zip(shot_ids, shot_prompts, frame_counts)
            )
        ],
    }


def iter_continuity_dev_32_cases(
    benchmark: Mapping[str, Any],
    *,
    case_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = benchmark.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark.cases must be a non-empty array")
    wanted = None if case_ids is None else list(case_ids)
    if wanted is not None:
        if len(set(wanted)) != len(wanted):
            raise ValueError("case_ids must be unique")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"benchmark.cases[{index}] must be an object")
        case_id = _text(row.get("case_id"), f"benchmark.cases[{index}].case_id")
        if case_id in seen:
            raise ValueError(f"benchmark.cases has duplicate case_id={case_id!r}")
        seen.add(case_id)
        if wanted is not None and case_id not in wanted:
            continue
        selected.append(case_inputs_from_benchmark_case(row))
    if wanted is not None:
        missing = [case_id for case_id in wanted if case_id not in {row["case_id"] for row in selected}]
        if missing:
            raise ValueError(f"continuity_dev_32 case_ids not in benchmark: {missing}")
        order = {case_id: index for index, case_id in enumerate(wanted)}
        selected.sort(key=lambda row: order[row["case_id"]])
    if not selected:
        raise ValueError("no continuity_dev_32 cases selected")
    return selected


def require_native_wan_baseline_floor(
    train_config: Any,
    *,
    weight_source: str,
    arm_id: str,
) -> None:
    """Fail loud if the paper wan_baseline row is pointed at an MSM architecture."""

    if arm_id != WAN_BASELINE_ARM_ID:
        return
    mode = train_config.resolved["model"]["temporal_position_mode"]
    if mode != GLOBAL_TIME_NATIVE_ROPE_V1:
        raise ValueError(
            "wan_baseline requires model.temporal_position_mode="
            f"{GLOBAL_TIME_NATIVE_ROPE_V1!r}; got {mode!r}"
        )
    if weight_source != WAN_BASELINE_WEIGHT_SOURCE:
        raise ValueError(
            "wan_baseline requires weight_source="
            f"{WAN_BASELINE_WEIGHT_SOURCE!r}; got {weight_source!r}"
        )


def single_shot_inputs_from_case(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split one story into independent native-Wan K=1 cases."""

    case_id = _text(case.get("case_id"), "case_id")
    prompts = case.get("prompts")
    shots = case.get("shots")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{case_id}: prompts must be a non-empty list")
    if not isinstance(shots, list) or len(shots) != len(prompts):
        raise ValueError(f"{case_id}: shots must align with prompts")
    rows: list[dict[str, Any]] = []
    for index, (shot, prompt) in enumerate(zip(shots, prompts)):
        if not isinstance(shot, Mapping):
            raise ValueError(f"{case_id}.shots[{index}] must be an object")
        shot_id = _text(shot.get("shot_id"), f"{case_id}.shots[{index}].shot_id")
        shot_prompt = _text(shot.get("prompt"), f"{case_id}.shots[{index}].prompt")
        frame_count = _positive_int(
            shot.get("frame_count"), f"{case_id}.shots[{index}].frame_count"
        )
        plan = wan22_shot_plan_from_frame_counts(
            [frame_count],
            label=f"{case_id}.shot[{index}]",
        )
        rows.append(
            {
                "case_id": f"{case_id}::{shot_id}",
                "prompts": [prompt],
                "story_prompt": _text(case.get("story_prompt"), f"{case_id}.story_prompt"),
                "shot_prompts": [shot_prompt],
                "shot_ids": [shot_id],
                "target_fps": case["target_fps"],
                "max_rgb_frames": plan["max_rgb_frames"],
                "vae_temporal_stride": plan["vae_temporal_stride"],
                "shot_groups": plan["shot_groups"],
                "latent_shot_groups": plan["latent_shot_groups"],
                "train_rgb_frames": plan["train_rgb_frames"],
                "train_latent_frames": plan["train_latent_frames"],
                "shots": [
                    {
                        "shot_id": shot_id,
                        "prompt": shot_prompt,
                        "frame_count": frame_count,
                    }
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 主方法 B：逐镜自回归 + 冻结 VLM 线索。镜 k 生成完取第 0 帧、缩到 224²、喂给
# 镜 k+1..K；ffmpeg 拼接独立生成的逐镜 mp4。S0 全部自生成；S1/S2 的镜 1 是真实片段
# 原样保留，差别只在镜 2..K 的前缀来源（自生成 vs 真实）。
# ---------------------------------------------------------------------------


def extract_first_frame_jpeg(
    video_path: Path, *, output_path: Path, frame_size: int = CUE_FRAME_SIZE
) -> Path:
    """自回归推理专用：镜 k 生成完，取它的第 0 帧，缩到 ``frame_size`` 见方，落成 JPEG。

    几何与训练侧 ``shot_first_frame_v1``（224²、BICUBIC）同形，但这里读的是**推理刚
    生成的输出**，不是训练语料——两者是不同的生产者，不共用代码，只共享这三个几何
    常量（§1.4.1）。
    """

    import imageio
    from PIL import Image

    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise FileNotFoundError(f"generated shot video is missing or empty: {video_path}")
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    image = Image.fromarray(frame).convert("RGB").resize(
        (frame_size, frame_size), Image.BICUBIC
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=CUE_FRAME_JPEG_QUALITY)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"failed to write the extracted first frame: {output_path}")
    return output_path


def decode_real_shot_frames_from_latent_cache(
    *,
    case: Mapping[str, Any],
    latent_cache_dir: Path,
    runtime: Any,
    shot_index: int,
) -> list[Any]:
    """从训练用的 latent 缓存解出**第 ``shot_index`` 镜**的 RGB 帧，不落盘。

    这是本项目「真实视频」的唯一解码口：S1/S2 的镜 1
    （:func:`decode_real_shot_zero_from_latent_cache` 在这之上只多一步存 mp4）与
    ``tools/audit/measure_source_vs_reconstruction_plan_gap.py`` 的重建侧共用它，
    所以两处拿到的「重建帧」按构造就是同一件东西，不会有第二份解码实现漂移。

    **权威依据（实测，09-08）**：``real_holdout_96_v1.json`` 的 ``source_metadata``
    与它指向的源片自身帧序不自洽——抽查 12 条 dev-96，11 条 ``max(source_shot_groups)``
    越过源片实际帧数（例如 ``ci_vid:test:000005689`` 源片仅 78 帧却要取到帧 131）。
    与 benchmark 自洽的是这份 latent 缓存：``outputs/datasets/latent_cache/
    hq_pool_205k_full_v1`` 下 ``<case_id>.safetensors`` 的 ``input_latents`` 时间维
    长度，逐条等于该 case 每镜 ``(frame_count-1)//4+1`` 之和；因此本项目把它当成
    「真实视频」的权威来源。

    切片区间直接复用 ``case["latent_shot_groups"][shot_index]``——调用方（benchmark 走
    ``case_inputs_from_benchmark_case``，审计工具走 latent 缓存 manifest）已经用
    ``wan22_shot_plan_from_frame_counts``/``rgb_to_latent_shot_groups`` 按各自的
    ``shots[].frame_count`` 算过一遍，这里不再另起一份换算。VAE 解码用调用方已经加载
    好的那一份（``runtime.pipe.vae``），几何（480×832）与
    ``MSM_RENDER_CONTRACT`` 相同。写盘帧率只读 ``runtime.inference_cfg["fps"]``。
    取帧 ``target_fps`` 与出片帧率同为 ``WAN22_CACHED_SPACETIME_CONTRACT`` 的 16，解码结果不需要
    再缩放。
    """

    shot_latents = read_real_shot_latents_from_latent_cache(
        case=case, latent_cache_dir=latent_cache_dir, shot_index=shot_index
    ).to(device=runtime.device, dtype=runtime.dtype)

    decoded = runtime.pipe.vae.decode(
        shot_latents,
        device=runtime.device,
        tiled=runtime.pipeline_cfg["tiled"],
        tile_size=tuple(runtime.pipeline_cfg["tile_size"]),
        tile_stride=tuple(runtime.pipeline_cfg["tile_stride"]),
    )
    frames = list(runtime.pipe.vae_output_to_video(decoded))
    # 缓存按补齐后的镜长编码；与生成臂同一口径裁回剧本镜长。
    return crop_frames_to_shot_lengths(
        frames,
        shot_groups=[[0, len(frames)]],
        shot_frame_counts=[case["shots"][shot_index]["frame_count"]],
        label=f"{case['case_id']}: decoded real shot {shot_index}",
    )


#: 真片参照行的臂名与出片路径。它不是一条生成臂：没有权重、没有噪声、没有 seed，每镜就是
#: 训练用 latent 缓存里那一镜的 VAE 解码。
REAL_FOOTAGE_ARM_ID = "real_footage_latent_decode"
GENERATION_PATH_REAL_FOOTAGE = "latent_cache_vae_decode_per_shot_concat"


def write_real_footage_reference(
    *,
    runtime: Any,
    benchmark_path: Path,
    latent_cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """benchmark 每条故事的真片参照：逐镜从 latent 缓存解码成 mp4、硬切拼接、写评分清单。

    为什么走 latent 缓存而不是源片：``real_holdout_96`` 的源片 74/96 不在主服务器盘上，在的 22 条
    声明帧数又超出源片时长；与
    benchmark 自洽的真片只有这份缓存（:func:`decode_real_shot_frames_from_latent_cache`
    的 docstring）。解码就是那一个口，所以这一行与 ``oracle_prev`` 记忆读的是同一份潜变量。

    与出片同形的三件事：逐镜 mp4 写在 ``shots/<case_id>/<序号>_<shot_id>.mp4``，用
    ``wan22.save_video`` 按 ``runtime.inference_cfg`` 的 fps／quality 存；拼接用逐镜出片同一个
    ``core.video.concat_shot_videos``；清单行由 ``default_manifest_row`` 给，
    ``generation_paradigm`` 是 ``per_shot_concat``（逐镜解码再硬切，字面为真）。清单行**不写**
    ``generation_seed``：这里没有采样，打分时给的 seed 只是 ``case_id`` 里的标签。

    VAE 的缩放是一次潜变量→RGB 的有损重建：这一行是「VAE 重建的真片」，不是源片。
    """

    from mspc.inference.sampler import default_manifest_row
    from mspc.inference.wan22 import save_video
    from mspc.core.video import concat_shot_videos

    if output_dir.exists():
        raise FileExistsError(f"real footage reference directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _, benchmark = load_continuity_dev_32_benchmark(benchmark_path)
    manifest: list[dict[str, Any]] = []
    for case in iter_continuity_dev_32_cases(benchmark):
        output_path = output_dir / f"{case['case_id']}.mp4"
        shots_dir = output_dir / "shots" / output_path.stem
        shot_paths: list[Path] = []
        for shot_index, shot_id in enumerate(case["shot_ids"]):
            frames = decode_real_shot_frames_from_latent_cache(
                case=case,
                latent_cache_dir=latent_cache_dir,
                runtime=runtime,
                shot_index=shot_index,
            )
            shot_path = shots_dir / f"{shot_index:02d}_{shot_id}.mp4"
            save_video(
                frames,
                shot_path,
                fps=runtime.inference_cfg["fps"],
                quality=runtime.inference_cfg["video_quality"],
            )
            shot_paths.append(shot_path)
        concat_shot_videos(shot_paths, output_path=output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"real footage video was not written: {output_path}")
        manifest.append(
            default_manifest_row(
                case_id=case["case_id"],
                output_path=output_path,
                generation_paradigm=GENERATION_PARADIGM_PER_SHOT_CONCAT,
                case_inputs=case,
                source_row=None,
                extra_fields={
                    "arm_id": REAL_FOOTAGE_ARM_ID,
                    "shot_ids": case["shot_ids"],
                    "shots": case["shots"],
                    "frame_count_source": CONTINUITY_DEV_32_FRAME_COUNT_SOURCE,
                    "generation_path": GENERATION_PATH_REAL_FOOTAGE,
                    "latent_cache_dir": str(latent_cache_dir),
                    "shot_video_paths": [str(path) for path in shot_paths],
                },
            )
        )
    manifest_path = output_dir / "generated_eval_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "arm_id": REAL_FOOTAGE_ARM_ID,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "case_count": len(manifest),
    }


def read_real_shot_latents_from_latent_cache(
    *,
    case: Mapping[str, Any],
    latent_cache_dir: Path,
    shot_index: int,
) -> Any:
    """第 ``shot_index`` 镜的真实 latent：``[1, C, F_shot, H, W]``，留在 CPU、保持缓存 dtype。

    权威来源与切片契约见 :func:`decode_real_shot_frames_from_latent_cache` 的 docstring
    （那里在此之上解码成 RGB）。两处共用这一份读取与核对。
    """

    if type(shot_index) is not int or shot_index < 0:
        raise ValueError(f"shot_index must be a non-negative int; got {shot_index!r}")
    case_id = case["case_id"]
    cache_path = Path(latent_cache_dir) / "latents" / f"{case_id}.safetensors"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"{case_id}: real shot-{shot_index} latent cache is missing: {cache_path}"
        )

    from safetensors import safe_open

    with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
        if "input_latents" not in set(handle.keys()):
            raise ValueError(f"{case_id}: {cache_path} does not carry an input_latents tensor")
        input_latents = handle.get_tensor("input_latents")
    expected_latent_frames = case["train_latent_frames"]
    if input_latents.shape[2] != expected_latent_frames:
        raise ValueError(
            f"{case_id}: cached input_latents has {input_latents.shape[2]} latent frames "
            f"along dim 2, expected {expected_latent_frames} from this benchmark case's "
            "shots[].frame_count"
        )
    latent_groups = case["latent_shot_groups"]
    if shot_index >= len(latent_groups):
        raise ValueError(
            f"{case_id}: shot_index {shot_index} is out of range; this case has "
            f"{len(latent_groups)} shot(s)"
        )
    latent_start, latent_end = latent_groups[shot_index]
    return input_latents[:, :, latent_start:latent_end]


def decode_real_shot_zero_from_latent_cache(
    *,
    case: Mapping[str, Any],
    latent_cache_dir: Path,
    runtime: Any,
    output_path: Path,
) -> Path:
    """S1/S2 镜 1 的真实片段：从训练用的 latent 缓存解码后存成 mp4。

    这是 S1「真实镜 1 原样保留」与 S2 教师强制真实前缀的物质基础，取代了原先按
    ``source_metadata.video_path``/``source_shot_groups`` 拿 ffmpeg 切源片的路径。
    解码本身归 :func:`decode_real_shot_frames_from_latent_cache`（那里有权威依据与
    切片契约），本函数只多一步落盘。
    """

    frames = decode_real_shot_frames_from_latent_cache(
        case=case,
        latent_cache_dir=latent_cache_dir,
        runtime=runtime,
        shot_index=0,
    )

    from mspc.inference.wan22 import save_video

    save_video(
        frames,
        output_path,
        fps=runtime.inference_cfg["fps"],
        quality=runtime.inference_cfg["video_quality"],
    )
    return output_path




#: 「同一段源视频」按语料的目录版式判，一种版式一行：``(语料目录名, 其下的层数, 取哪一层当
#: 源视频)``。异故事记忆不能与本条同源——同一段源视频切出的两条故事共享人物与场景，拿它当
#: 「另一个故事」就把同源相似记成了异故事读数。
#:
#: - ci_vid：``ci_vid/videos/chunk_<n>/<源视频>/<序号>/<文件>``。同一源视频目录下的不同序号是
#:   同源不同切片（``real_holdout_96`` 里就有一对：``bbc1046a…/18`` 与 ``bbc1046a…/3``）；
#:   ``chunk_<n>`` 只是存储分桶，两段不同源视频可以落进同一个桶。
#: - cine250k：``cine250k/raw_videos/<源视频>.mp4``，故事是其中一个时间窗。
#:
#: 不在表里的版式直接抛：猜错同源判据会让「不同源」这句话悄悄失效。
SOURCE_VIDEO_LAYOUTS: dict[str, tuple[int, int]] = {
    "ci_vid": (5, 2),
    "cine250k": (2, 1),
}


def source_video_identity(case: Mapping[str, Any]) -> str:
    """这条故事切自哪段源视频：``"<语料>/<源视频>"``。判据见 :data:`SOURCE_VIDEO_LAYOUTS`。"""

    case_id = _text(case.get("case_id"), "case_id")
    metadata = case.get("source_metadata")
    if not isinstance(metadata, Mapping) or "video_path" not in metadata:
        raise ValueError(
            f"{case_id}: has no source_metadata.video_path, so its source video is unknown "
            "and an other-story donor cannot be checked against it"
        )
    parts = PurePosixPath(_text(metadata["video_path"], f"{case_id}.video_path")).parts
    for corpus, (depth, source_level) in SOURCE_VIDEO_LAYOUTS.items():
        if corpus not in parts:
            continue
        tail = parts[parts.index(corpus) + 1 :]
        if len(tail) != depth:
            break
        return f"{corpus}/{PurePosixPath(tail[source_level]).stem}"
    raise ValueError(
        f"{case_id}: video_path {metadata['video_path']!r} matches no layout in "
        f"SOURCE_VIDEO_LAYOUTS ({sorted(SOURCE_VIDEO_LAYOUTS)})"
    )


def other_story_donors(
    cases: Sequence[Mapping[str, Any]], *, offset: int
) -> dict[str, Mapping[str, Any]]:
    """异故事记忆的 donor 表：``case_id -> donor case``，覆盖 ``cases`` 的每一条。

    ``cases`` 是 benchmark 的**完整**声明顺序（不是本次分片）。第 ``i`` 条的 donor 是从
    ``(i + offset) mod n`` 起往后第一条与它**不同源**（:func:`source_video_identity`）的故事，
    因此不会自配，也不会配到同源的另一个切片。donor 只由位置与来源决定，所以分片、``--limit``
    与渲染次序都不改变任何一条的 donor。跳过同源时两条故事可能共用一个 donor。
    """

    count = len(cases)
    if type(offset) is not int or not 1 <= offset < count:
        raise ValueError(
            f"memory_donor_offset must be an integer in 1..{count - 1} for {count} cases; "
            f"got {offset!r}"
        )
    sources = [source_video_identity(case) for case in cases]
    donors: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(cases):
        for step in range(count):
            position = (index + offset + step) % count
            if sources[position] != sources[index]:
                donors[case["case_id"]] = cases[position]
                break
        else:
            raise ValueError(
                f"{case['case_id']}: every case in the benchmark shares its source video "
                f"{sources[index]!r}; there is no other-story donor"
            )
    return donors


def nearest_past_shots(shot_index: int, frames_max: int) -> tuple[int, ...]:
    """目标镜之前、离它最近的至多 ``frames_max`` 镜，按镜序（升序）。

    与训练地板池（窗口清单 ``memory_mode=past_nearest``：只取过去、最近的至多 3 镜）同一
    规则；按镜序排，最后一个是离目标最近的那一镜——原生跨步位置策略
    这里只给出历史镜序，不拥有渲染器的位置坐标。
    """

    if type(shot_index) is not int or shot_index < 0:
        raise ValueError(f"shot_index must be a non-negative int; got {shot_index!r}")
    if type(frames_max) is not int or frames_max < 1:
        raise ValueError(f"frames_max must be a positive int; got {frames_max!r}")
    return tuple(range(max(0, shot_index - frames_max), shot_index))


def deduplicated_past_shots(latents: Sequence[Any], frames_max: int) -> tuple[int, ...]:
    """去重记忆库：按镜序逐镜收第 0 潜帧，与已收任一镜的相似度超过 ``TAU_FIRST_FIRST`` 就不收；
    返回库里最近的至多 ``frames_max`` 镜，按镜序（升序）。"""

    from mspc.data.latent_pair_similarity import TAU_FIRST_FIRST, aligned_minus_shuffled_cosine

    if type(frames_max) is not int or frames_max < 1:
        raise ValueError(f"frames_max must be a positive int; got {frames_max!r}")
    frames = [latent[0, :, 0].float().cpu().numpy() for latent in latents]
    kept: list[int] = []
    for shot, frame in enumerate(frames):
        if all(aligned_minus_shuffled_cosine(frames[other], frame) <= TAU_FIRST_FIRST for other in kept):
            kept.append(shot)
    return tuple(kept[-frames_max:])




def decode_memory_latent_frame(runtime: Any, latent: Any) -> Any:
    """一张记忆潜帧 ``[1, C, 1, H, W]`` → 一张 RGB 图（VAE 单帧解码，渲染器自己的那个 VAE）。"""

    decoded = runtime.pipe.vae.decode(
        latent.to(device=runtime.device, dtype=runtime.dtype),
        device=runtime.device,
        tiled=runtime.pipeline_cfg["tiled"],
        tile_size=tuple(runtime.pipeline_cfg["tile_size"]),
        tile_stride=tuple(runtime.pipeline_cfg["tile_stride"]),
    )
    frames = list(runtime.pipe.vae_output_to_video(decoded))
    if len(frames) != 1:
        raise ValueError(f"one memory latent frame decoded to {len(frames)} RGB frames")
    return frames[0]




def decode_memory_frame_from_latent_cache(
    *,
    case: Mapping[str, Any],
    latent_cache_dir: Path,
    runtime: Any,
    shot_index: int,
) -> Any:
    """``oracle_prev`` 臂在镜 ``shot_index`` 离目标最近的那一帧记忆（上一镜真片第 0 潜帧），
    解码成一张 RGB 图。

    潜帧读的是训练用的 latent 缓存（与出片同一片真片潜帧），这里只多一次单帧 VAE 解码，
    所以读图写手看到的就是渲染器拿到的那一帧，而不是另从源片、帧包或 2 fps 首帧里再找
    一张「差不多」的图。镜 0 没有记忆，调用方不该为它要图。
    """

    if type(shot_index) is not int or shot_index < 0:
        raise ValueError(f"shot_index must be a non-negative int; got {shot_index!r}")
    if shot_index == 0:
        raise ValueError(
            f"{case['case_id']}: shot {shot_index} has no memory frame (M=0); "
            "only shots after the first one carry a previous-shot memory"
        )
    memory = read_real_shot_latents_from_latent_cache(
        case=case, latent_cache_dir=latent_cache_dir, shot_index=shot_index - 1
    )[:, :, 0:1]
    return decode_memory_latent_frame(runtime, memory)




def render_memory_prefix_shots_concat(
    *,
    runtime: Any,
    case: Mapping[str, Any],
    memory_arm: str,
    latent_cache_dir: Path | None,
    output_path: Path,
    seed: int,
    shot_seed_stride: int,
    manifest_dir: Path,
    donor_case: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """旧dev32适配器仅保留无参考地板；编译参考直接由runtime.generate_case消费。"""

    from mspc.core.video import generate_story_by_single_shot_concat

    if memory_arm != MEMORY_ARM_NONE:
        raise ValueError("dev32 only accepts memory_arm='none'; reference segments use compiled rows through Wan22InferenceRuntime.generate_case")
    if latent_cache_dir is not None or donor_case is not None:
        raise ValueError("the no-reference floor does not take a reference cache or donor")
    readouts: list[dict[str, Any]] = []

    def generate_shot(*, shot_inputs: Mapping[str, Any], output_path: Path, seed: int) -> None:
        readout = runtime.generate_case(
            prompts=shot_inputs["prompts"],
            shot_groups=shot_inputs["shot_groups"],
            latent_shot_groups=shot_inputs["latent_shot_groups"],
            shot_frame_counts=script_shot_frame_counts(shot_inputs),
            output_path=output_path,
            seed=seed,
            memory=None,
        )
        readouts.append({**readout, "memory_frames": []})

    generate_story_by_single_shot_concat(
        shot_inputs=single_shot_inputs_from_case(case),
        output_path=output_path,
        seed=seed,
        shot_seed_stride=shot_seed_stride,
        generate_shot=generate_shot,
    )
    return readouts


def real_continuation_boundaries(real_first_latents: Sequence[Any]) -> tuple[bool, ...]:
    """镜 k 与镜 k−1 在真片上是不是「同一机位接着拍」：两镜真片第 0 潜帧的
    ``aligned_minus_shuffled_cosine`` 超过 ``TAU_FIRST_FIRST``（``real_holdout_96`` 干净硬切的 95 分位，
    与去重记忆臂同一把尺子）。第 0 镜没有上一镜，恒为 ``False``。"""

    from mspc.data.latent_pair_similarity import TAU_FIRST_FIRST, aligned_minus_shuffled_cosine

    frames = [latent[0, :, 0].float().cpu().numpy() for latent in real_first_latents]
    return tuple(
        index > 0
        and aligned_minus_shuffled_cosine(frames[index - 1], frame) > TAU_FIRST_FIRST
        for index, frame in enumerate(frames)
    )


def write_fed_first_frame(
    *, runtime: Any, case_id: str, shot_index: int, latent: Any, manifest_dir: Path
) -> str:
    """喂进首帧口的那一张潜帧经 VAE 单帧解码落成 PNG，返回绝对路径（与清单里成片的
    ``output_path`` 同一写法，分片清单合并时不需要改写）。

    量首帧口服从（成片各帧像不像喂入帧）比的就是这张图；它是渲染器实际拿到的那一帧，不是
    另从源片或 2 fps 帧包里找的一张。"""

    path = (
        Path(manifest_dir).resolve()
        / "first_frame"
        / case_id.replace(":", "_")
        / f"shot{shot_index:02d}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    decode_memory_latent_frame(runtime, latent).save(path)
    return str(path)


def coarse_canvas_image(image):
    """真帧像素长宽各降到 1/8，再双三次升回原尺寸。倍数写死为 8。"""

    from PIL import Image

    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("coarse canvas requires one RGB image from the same VAE decode")
    factor = COARSE_CANVAS_FACTOR
    width, height = image.size
    if width % factor or height % factor:
        raise ValueError(
            f"coarse canvas factor {factor} requires both sides divisible by {factor}; got {width}x{height}"
        )
    small = image.resize((width // factor, height // factor), Image.BOX)
    if small.size != (width // factor, height // factor):
        raise RuntimeError("box downsample did not land on 1/8 of each side")
    restored = small.resize((width, height), Image.BICUBIC)
    if restored.size != (width, height) or restored.mode != "RGB":
        raise RuntimeError("bicubic restore did not return the original RGB size")
    return restored


def render_first_frame_shots_concat(
    *,
    runtime: Any,
    case: Mapping[str, Any],
    first_frame_arm: str,
    first_frame_pin_until_sigma: float,
    latent_cache_dir: Path,
    output_path: Path,
    seed: int,
    shot_seed_stride: int,
    manifest_dir: Path,
    donor_case: Mapping[str, Any] | None = None,
    baseline_video: Path | None = None,
) -> list[dict[str, Any]]:
    """首帧臂的逐镜渲染：按镜序渲整条故事，每镜第 0 潜帧经原生首帧口钉住，硬切拼接。

    与 :func:`render_memory_prefix_shots_concat` 同族（镜级输入由 :func:`single_shot_inputs_from_case`
    切，写盘与拼接归 ``core.video.generate_story_by_single_shot_concat``）；次序写死为镜序，因为
    ``real_first_continuation_prev_last`` 的续拍镜要上一镜的成片末帧。返回每镜一条读出，次序即镜序，
    记下喂入帧的来源与解码 PNG。
    """

    from mspc.core.video import generate_story_by_single_shot_concat

    if first_frame_arm not in FIRST_FRAME_ARMS:
        raise ValueError(
            f"first_frame_arm must be one of {list(FIRST_FRAME_ARMS)}; got {first_frame_arm!r}"
        )
    if (first_frame_arm == FIRST_FRAME_ARM_OTHER_STORY) != (donor_case is not None):
        raise ValueError(
            f"first_frame_arm={FIRST_FRAME_ARM_OTHER_STORY!r} needs a donor case and no other "
            f"arm takes one; got first_frame_arm={first_frame_arm!r}"
        )
    if donor_case is not None and donor_case["case_id"] == case["case_id"]:
        raise ValueError(f"{case['case_id']}: the other-story first frame cannot be this case")
    if (first_frame_arm == FIRST_FRAME_ARM_EVENT) != (baseline_video is not None):
        raise ValueError("event_first requires an A0 baseline video; other arms refuse it")
    event_sources = case.get("event_first_source_shots")
    if first_frame_arm == FIRST_FRAME_ARM_EVENT and event_sources is None:
        raise ValueError(f"{case['case_id']}: event_first requires compiled event sources")
    per_shot = single_shot_inputs_from_case(case)
    shot_index_by_id = {row["shot_ids"][0]: index for index, row in enumerate(per_shot)}
    real_first = [
        read_real_shot_latents_from_latent_cache(
            case=case, latent_cache_dir=latent_cache_dir, shot_index=index
        )[:, :, 0:1]
        for index in range(len(per_shot))
    ]
    continuation = real_continuation_boundaries(real_first)
    donor_first = (
        None
        if donor_case is None
        else read_real_shot_latents_from_latent_cache(
            case=donor_case, latent_cache_dir=latent_cache_dir, shot_index=0
        )[:, :, 0:1]
    )
    readouts: list[dict[str, Any]] = []

    def generate_shot(*, shot_inputs: Mapping[str, Any], output_path: Path, seed: int) -> None:
        shot_index = shot_index_by_id[shot_inputs["shot_ids"][0]]
        if len(readouts) != shot_index:
            raise RuntimeError(
                f"{case['case_id']}: first-frame arms render a story in shot order; shot "
                f"{shot_index} came up with {len(readouts)} shot(s) already rendered"
            )
        if first_frame_arm == FIRST_FRAME_ARM_EVENT:
            source_index = event_sources[shot_index]
            if source_index is None:
                clip = baseline_video.parent / "shots" / baseline_video.stem / f"{shot_index:02d}_{shot_inputs['shot_ids'][0]}.mp4"
                if not clip.is_file():
                    raise FileNotFoundError(clip)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(clip, output_path)
                readouts.append({"source": "a0_reused_clip", "source_video": str(clip),
                                 "source_shot_index": shot_index,
                                 "real_continuation": continuation[shot_index]})
                return
            fed = real_first[source_index]
            source = {"source": "event_first", "source_case_id": case["case_id"],
                      "source_shot_index": source_index}
        elif first_frame_arm == FIRST_FRAME_ARM_OTHER_STORY:
            fed, source = donor_first, {"source": "other_story", "source_case_id": donor_case["case_id"], "source_shot_index": 0}
        elif first_frame_arm == FIRST_FRAME_ARM_COARSE:
            from mspc.inference.wan22 import encode_single_frame

            rgb = decode_memory_latent_frame(runtime, real_first[shot_index])
            coarse = coarse_canvas_image(rgb)
            fed = encode_single_frame(runtime, coarse)
            source = {
                "source": "real_first_coarse",
                "source_case_id": case["case_id"],
                "source_shot_index": shot_index,
                "coarse_factor": COARSE_CANVAS_FACTOR,
                "downsample": "pil_box",
                "upsample": "pil_bicubic",
            }
        elif first_frame_arm == FIRST_FRAME_ARM_CONTINUATION_PREV_LAST and continuation[shot_index]:
            fed = runtime.last_generated_last_latent
            if fed is None:
                raise RuntimeError(
                    f"{case['case_id']} shot {shot_index}: the previous shot's last-frame "
                    "encoding was not kept"
                )
            source = {"source": "generated_prev_last", "source_case_id": case["case_id"], "source_shot_index": shot_index - 1}
        else:
            fed, source = real_first[shot_index], {"source": "real_first", "source_case_id": case["case_id"], "source_shot_index": shot_index}
        png = write_fed_first_frame(
            runtime=runtime,
            case_id=case["case_id"],
            shot_index=shot_index,
            latent=fed,
            manifest_dir=manifest_dir,
        )
        readout = runtime.generate_case(
            prompts=shot_inputs["prompts"],
            shot_groups=shot_inputs["shot_groups"],
            latent_shot_groups=shot_inputs["latent_shot_groups"],
            shot_frame_counts=script_shot_frame_counts(shot_inputs),
            output_path=output_path,
            seed=seed,
            first_frame=fed,
            first_frame_pin_until_sigma=first_frame_pin_until_sigma,
        )
        readouts.append(
            {
                **readout,
                **source,
                "real_continuation": continuation[shot_index],
                "first_frame_png": png,
            }
        )

    generate_story_by_single_shot_concat(
        shot_inputs=per_shot,
        output_path=output_path,
        seed=seed,
        shot_seed_stride=shot_seed_stride,
        generate_shot=generate_shot,
    )
    return readouts


def require_render_fps(raw: Mapping[str, Any], *, label: str) -> int:
    """出片帧率只读配置。缺字段、布尔值或非正整数都立即报错，不再另有合同常量。"""
    if "fps" not in raw:
        raise ValueError(f"{label}.fps is required")
    value = raw["fps"]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}.fps must be a positive integer")
    return value


def parse_msm_render_params(raw: Mapping[str, Any], *, label: str = "render") -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for field, expected in MSM_RENDER_CONTRACT.items():
        if field not in raw:
            raise ValueError(f"{label}.{field} is required")
        value = raw[field]
        if type(expected) is int:
            if type(value) is not int:
                raise ValueError(f"{label}.{field} must be an integer")
        elif type(expected) is float:
            if type(value) not in {int, float} or isinstance(value, bool):
                raise ValueError(f"{label}.{field} must be a number")
            value = float(value)
        if value != expected:
            raise ValueError(
                f"{label}.{field} must be the MSM render contract value "
                f"{expected!r}; got {value!r}"
            )
        parsed[field] = value
    parsed["fps"] = require_render_fps(raw, label=label)
    return parsed


def load_continuity_dev_32_generate_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"continuity_dev_32 generate config is missing: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("continuity_dev_32 generate config must be an object")
    if raw.get("schema_version") != CONTINUITY_DEV_32_GENERATE_SCHEMA:
        raise ValueError(
            f"schema_version must be {CONTINUITY_DEV_32_GENERATE_SCHEMA!r}"
        )
    extra = sorted(set(raw) - {
        "schema_version",
        "arm_id",
        "train_config",
        "benchmark",
        "weight_source",
        "generation_seed",
        "checkpoint",
        "output_dir",
        "case_ids",
        # 参考来源：``none`` / null
        # （不写 = 这一次不是记忆臂）。它不进物化出的 ``inference`` 块——臂的机制参数
        # （帧数上界、dropout、相位策略）归 train config 的 ``model.memory_prefix``，与训练
        # 侧共用同一份；这里只说这一次出片的记忆从哪来。
        "memory_arm",
        # 异故事记忆的错位步长：第 i 条故事从声明顺序的第 i+offset 条起取第一条不同源的故事
        # 当 donor（:func:`other_story_donors`）。只有 ``other_story`` 这一档要它。
        "memory_donor_offset",
        # 首帧臂（:data:`FIRST_FRAME_ARMS`）：不写 = 这一次不是首帧臂。与记忆臂一样只说
        # 这一次每镜第 0 潜帧从哪来；``other_story_first`` 的 donor 错位沿用 ``memory_donor_offset``
        # （同一张 donor 表，:func:`other_story_donors`）。
        "first_frame_arm",
        "first_frame_pin_until_sigma",
        "first_frame_baseline_manifest",
        "generation_paradigm",
        # 逐镜出片时第 i 镜的种子是 generation_seed + shot_seed_stride·i。不写 = null，即每镜都用
        # generation_seed（09-26 之前的全部逐镜出片都是这样）。只有逐镜拼接范式读它。
        "shot_seed_stride",
        *MSM_RENDER_FIELDS,
        "fps",
    })
    if extra:
        raise ValueError(f"continuity_dev_32 generate config has unknown fields: {extra}")
    base = source.parent

    def _optional_path(value: object, label: str) -> Path | None:
        """Resolve one path field: ``${MSPC_SHARED_ROOT}`` first, then relative-to-config.

        The token has to be expanded *before* the absolute test, or a value that
        is absolute once expanded looks relative and gets glued onto the config's
        own directory -- which is how the B carriers first produced
        ``configs/infer/wan22_5b/p6/media/datasets/...``.  Expanding here rather
        than at the one call site that needed it keeps every path field in this
        config speaking the same dialect as the rest of the project, and keeps a
        shared-disk absolute path out of a committed config, which is the shape
        that already cost this project a training job on the metadata CSV.
        """

        if value is None:
            return None
        raw_path = Path(
            expand_shared_root_token(_text(value, label), label)
        ).expanduser()
        return raw_path if raw_path.is_absolute() else (base / raw_path).resolve()

    case_ids = raw.get("case_ids")
    if case_ids is not None:
        if not isinstance(case_ids, list) or not case_ids:
            raise ValueError("case_ids must be a non-empty array when provided")
        case_ids = [_text(item, f"case_ids[{index}]") for index, item in enumerate(case_ids)]
    seed = raw.get("generation_seed")
    if type(seed) is not int:
        raise ValueError("generation_seed must be an integer")
    # 记忆来源：不写等于 null，即这一次不是记忆臂。取值必须在册，不猜、不回退到某一档。
    memory_arm = raw.get("memory_arm")
    if memory_arm is not None:
        memory_arm = _text(memory_arm, "memory_arm")
        if memory_arm not in ACCEPTED_MEMORY_ARMS:
            raise ValueError(
                f"memory_arm must be one of {list(ACCEPTED_MEMORY_ARMS)} or null; got {memory_arm!r}"
            )
    donor_offset = raw.get("memory_donor_offset")
    if donor_offset is not None and (type(donor_offset) is not int or donor_offset <= 0):
        raise ValueError(f"memory_donor_offset must be a positive integer; got {donor_offset!r}")
    first_frame_arm = raw.get("first_frame_arm")
    if first_frame_arm is not None:
        first_frame_arm = _text(first_frame_arm, "first_frame_arm")
        if first_frame_arm not in FIRST_FRAME_ARMS:
            raise ValueError(
                f"first_frame_arm must be one of {list(FIRST_FRAME_ARMS)} or null; got {first_frame_arm!r}"
            )
        if memory_arm is not None:
            raise ValueError("first_frame_arm and memory_arm are two different arms")
    baseline_manifest = _optional_path(raw.get("first_frame_baseline_manifest"), "first_frame_baseline_manifest")
    if (first_frame_arm == FIRST_FRAME_ARM_EVENT) != (baseline_manifest is not None):
        raise ValueError("event_first requires first_frame_baseline_manifest; other arms refuse it")
    pin_sigma = raw.get("first_frame_pin_until_sigma")
    if first_frame_arm is not None:
        if type(pin_sigma) not in (int, float) or not 0 <= pin_sigma <= 1:
            raise ValueError("first_frame_arm requires explicit first_frame_pin_until_sigma in [0, 1]")
    elif pin_sigma is not None:
        raise ValueError("first_frame_pin_until_sigma requires first_frame_arm")
    takes_donor = (
        memory_arm == MEMORY_ARM_OTHER_STORY or first_frame_arm == FIRST_FRAME_ARM_OTHER_STORY
    )
    if takes_donor != (donor_offset is not None):
        raise ValueError(
            f"memory_arm={MEMORY_ARM_OTHER_STORY!r} or first_frame_arm="
            f"{FIRST_FRAME_ARM_OTHER_STORY!r} requires memory_donor_offset, and no other arm takes one"
        )
    # 哪种去噪序列（逐镜拼接 vs 联合）。没有默认值——不写就是 null，路由退回既有的
    # 联合/wan_baseline 判据（见 generate_continuity_dev_32），不是「不给就当某一档」。
    generation_paradigm = raw.get("generation_paradigm")
    if generation_paradigm is not None:
        generation_paradigm = _text(generation_paradigm, "generation_paradigm")
        if generation_paradigm != GENERATION_PARADIGM_PER_SHOT_CONCAT:
            raise ValueError(
                "generation_paradigm must be "
                f"{GENERATION_PARADIGM_PER_SHOT_CONCAT!r} or null; got "
                f"{generation_paradigm!r}"
            )
    shot_seed_stride = raw.get("shot_seed_stride")
    if shot_seed_stride is not None:
        if type(shot_seed_stride) is not int or shot_seed_stride < 0:
            raise ValueError(f"shot_seed_stride must be a non-negative integer or null; got {shot_seed_stride!r}")
        if generation_paradigm != GENERATION_PARADIGM_PER_SHOT_CONCAT:
            raise ValueError("shot_seed_stride only applies to generation_paradigm=per_shot_concat")
    return {
        "source_path": source,
        "arm_id": _text(raw.get("arm_id"), "arm_id"),
        "train_config": _optional_path(raw.get("train_config"), "train_config"),
        "benchmark": _optional_path(raw.get("benchmark"), "benchmark"),
        "weight_source": _text(raw.get("weight_source"), "weight_source"),
        "generation_seed": seed,
        "checkpoint": _optional_path(raw.get("checkpoint"), "checkpoint"),
        "output_dir": _optional_path(raw.get("output_dir"), "output_dir"),
        "case_ids": case_ids,
        "memory_arm": memory_arm,
        "memory_donor_offset": donor_offset,
        "first_frame_arm": first_frame_arm,
        "first_frame_pin_until_sigma": pin_sigma,
        "first_frame_baseline_manifest": baseline_manifest,
        "generation_paradigm": generation_paradigm,
        "shot_seed_stride": 0 if shot_seed_stride is None else shot_seed_stride,
        "render": parse_msm_render_params(raw, label="generate config"),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_continuity_dev_32_inference_config(
    train_config_path: Path,
    *,
    output_dir: Path,
    weight_source: str,
    checkpoint: Path | None,
    seed: int,
    render: Mapping[str, Any],
) -> Path:
    from mspc.config import load_experiment_config

    parsed_render = parse_msm_render_params(render, label="render")
    train_config = load_experiment_config(train_config_path)
    payload = train_config.to_dict()
    placeholder = output_dir / "continuity-dev-32-schema-placeholder.csv"
    placeholder.write_text("case_id\ncontinuity_dev_32_placeholder\n", encoding="utf-8")
    inference = payload["inference"]
    inference["enabled"] = True
    inference["output_dir"] = str(output_dir)
    inference["test_csv_path"] = str(placeholder)
    inference["weight_source"] = weight_source
    inference["seed"] = seed
    inference["dit_checkpoint"] = None if checkpoint is None else str(checkpoint)
    # 训练形态配置里的 inference.fps 必须等于取帧 target_fps，不能改成出片帧率。
    # 出片帧率只来自调用方交来的 render.fps，在加载之后写入 runtime。
    inference.update({key: value for key, value in parsed_render.items() if key != "fps"})
    config_path = output_dir / "continuity-dev-32-inference-config.json"
    _write_json(config_path, payload)
    return config_path


def generate_continuity_dev_32(
    *,
    train_config_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    weight_source: str,
    seed: int,
    checkpoint: Path | None = None,
    case_ids: Sequence[str] | None = None,
    arm_id: str,
    render: Mapping[str, Any] | None = None,
    memory_arm: str | None = None,
    memory_donor_offset: int | None = None,
    latent_cache_dir: Path | None = None,
    generation_paradigm: str | None = None,
    shot_seed_stride: int,
    generate_case=None,
    first_frame_arm: str | None = None,
    first_frame_pin_until_sigma: float | None = None,
    first_frame_baseline_manifest: Path | None = None,
) -> dict[str, Any]:
    from mspc.config import load_experiment_config
    from mspc.inference.sampler import default_manifest_row
    from mspc.inference.wan22 import Wan22InferenceRuntime, build_config
    from mspc.core.video import generate_story_by_single_shot_concat

    if output_dir.exists():
        raise FileExistsError(f"continuity_dev_32 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _, benchmark = load_continuity_dev_32_benchmark(benchmark_path)
    cases = iter_continuity_dev_32_cases(benchmark, case_ids=case_ids)
    # 异故事 donor 在**整份** benchmark 上一次算死：本次只渲其中几条（分片、case_ids）时，
    # 每条的 donor 与整臂一次渲完时相同。
    donors: dict[str, Mapping[str, Any]] = {}
    takes_donor = (
        memory_arm == MEMORY_ARM_OTHER_STORY or first_frame_arm == FIRST_FRAME_ARM_OTHER_STORY
    )
    if takes_donor != (memory_donor_offset is not None):
        raise ValueError(
            f"memory_arm={MEMORY_ARM_OTHER_STORY!r} or first_frame_arm="
            f"{FIRST_FRAME_ARM_OTHER_STORY!r} requires memory_donor_offset, and no "
            f"other arm takes one; got memory_arm={memory_arm!r}, first_frame_arm="
            f"{first_frame_arm!r}, memory_donor_offset={memory_donor_offset!r}"
        )
    if takes_donor:
        donors = other_story_donors(
            iter_continuity_dev_32_cases(benchmark, case_ids=None),
            offset=memory_donor_offset,
        )
    train_config = load_experiment_config(train_config_path)
    require_native_wan_baseline_floor(
        train_config,
        weight_source=weight_source,
        arm_id=arm_id,
    )
    # 路由只由「这一档读不读线索」决定。``generation_paradigm`` 不是路由开关，而是这条臂
    # 对自己去噪序列的**声明**：``wan_baseline`` 与三档 B 同属逐镜拼接，两者都写
    # ``per_shot_concat``，报表据此把切点两列标成记录列（切点按构造产生，不是能力）。
    # 早先的写法拿 ``generation_paradigm`` 当路由、并禁止它与 wan_baseline 同时出现，
    # 那会给后来者留一个陷阱：谁老实给 wan_baseline 补上它字面为真的那个声明，谁就撞上
    # 一个 raise。
    baseline_path = arm_id == WAN_BASELINE_ARM_ID
    # dev32仅保留无参考地板路由；ReferenceRows通过runtime直接消费。
    memory_path = memory_arm is not None
    if memory_path:
        if memory_arm not in ACCEPTED_MEMORY_ARMS:
            raise ValueError(
                f"memory_arm must be one of {list(ACCEPTED_MEMORY_ARMS)}; got {memory_arm!r}"
            )
        if baseline_path:
            raise ValueError(
                "the memory prefix arm is its own arm: it does not share a run with the "
                f"zero-conditioning wan_baseline floor (arm_id={arm_id!r})"
            )
        if generate_case is not None:
            raise ValueError(
                "the memory prefix arm renders through Wan22InferenceRuntime itself: "
                "the model memory-prefix contract must be present, which an "
                "injected generate_case cannot establish"
            )
        if (memory_arm in REAL_LATENT_MEMORY_ARMS) != (latent_cache_dir is not None):
            raise ValueError(
                f"memory arms {list(REAL_LATENT_MEMORY_ARMS)} read a real latent frame "
                "from the training latent cache, so they need latent_cache_dir and no "
                "other arm takes one"
            )
    # 首帧臂：同一逐镜拼接族，每镜第 0 潜帧经原生首帧口钉住；三档都从 latent 缓存读真片潜帧。
    baseline_videos = {}
    if (first_frame_arm == FIRST_FRAME_ARM_EVENT) != (first_frame_baseline_manifest is not None):
        raise ValueError("event_first requires first_frame_baseline_manifest; other arms refuse it")
    if first_frame_baseline_manifest is not None:
        baseline_rows = json.loads(Path(first_frame_baseline_manifest).read_text())
        for row in baseline_rows:
            cid = row["case_id"]
            if cid in baseline_videos:
                raise ValueError(f"duplicate baseline case: {cid}")
            baseline_videos[cid] = Path(row["video_path"])
    first_frame_path = first_frame_arm is not None
    if first_frame_path:
        if first_frame_arm not in FIRST_FRAME_ARMS:
            raise ValueError(
                f"first_frame_arm must be one of {list(FIRST_FRAME_ARMS)}; got {first_frame_arm!r}"
            )
        if baseline_path or memory_path:
            raise ValueError(
                "the first-frame arm is its own arm: it does not share a run with the "
                f"wan_baseline floor or a memory arm (arm_id={arm_id!r}, "
                f"memory_arm={memory_arm!r})"
            )
        if generate_case is not None:
            raise ValueError(
                "the first-frame arm renders through Wan22InferenceRuntime itself: a "
                "continuation shot's first frame is the previous shot's last-frame encoding"
            )
        if latent_cache_dir is None:
            raise ValueError(
                "first-frame arms read real first latent frames from the training latent "
                "cache, so they need latent_cache_dir"
            )
    elif not memory_path and latent_cache_dir is not None:
        raise ValueError(
            f"latent_cache_dir belongs to one of {list(REAL_LATENT_MEMORY_ARMS)} or a "
            f"first-frame arm; got memory_arm={memory_arm!r}"
        )
    runtime = None
    if generate_case is None:
        if render is None:
            raise ValueError("MSM render params are required when generating videos")
        config_path = materialize_continuity_dev_32_inference_config(
            train_config_path,
            output_dir=output_dir,
            weight_source=weight_source,
            checkpoint=checkpoint,
            seed=seed,
            render=render,
        )
        config = load_experiment_config(config_path)
        model_cfg, _, inference_cfg, pipeline_cfg = build_config(config)
        inference_cfg = dict(inference_cfg)
        inference_cfg["fps"] = require_render_fps(render, label="render")
        runtime = Wan22InferenceRuntime(
            model_cfg=model_cfg,
            inference_cfg=inference_cfg,
            pipeline_cfg=pipeline_cfg,
            dit_ckpt=config.resolved["inference"]["dit_checkpoint"],
        )
    if baseline_path:
        manifest_generation_paradigm = WAN_BASELINE_GENERATION_PARADIGM
    elif generation_paradigm == GENERATION_PARADIGM_PER_SHOT_CONCAT:
        manifest_generation_paradigm = GENERATION_PARADIGM_PER_SHOT_CONCAT
    else:
        manifest_generation_paradigm = MULTISHOT_GENERATION_PARADIGM
    manifest: list[dict[str, Any]] = []
    for case in cases:
        output_path = output_dir / f"{case['case_id']}.mp4"
        story_provenance: dict[str, Any] = {}
        if baseline_path:
            shot_inputs = single_shot_inputs_from_case(case)

            def generate_shot(
                *,
                shot_inputs: Mapping[str, Any],
                output_path: Path,
                seed: int,
            ) -> None:
                if generate_case is None:
                    if runtime is None:
                        raise RuntimeError("continuity_dev_32 runtime was not constructed")
                    runtime.generate_case(
                        prompts=shot_inputs["prompts"],
                        shot_groups=shot_inputs["shot_groups"],
                        latent_shot_groups=shot_inputs["latent_shot_groups"],
                        shot_frame_counts=script_shot_frame_counts(shot_inputs),
                        output_path=output_path,
                        seed=seed,
                    )
                else:
                    generate_case(
                        case_id=shot_inputs["case_id"],
                        case_inputs=shot_inputs,
                        output_path=output_path,
                        seed=seed,
                    )

            generate_story_by_single_shot_concat(
                shot_inputs=shot_inputs,
                output_path=output_path,
                seed=seed,
                shot_seed_stride=shot_seed_stride,
                generate_shot=generate_shot,
            )
        elif memory_path:
            if runtime is None:
                raise RuntimeError("continuity_dev_32 runtime was not constructed")
            donor_case = donors.get(case["case_id"])
            story_provenance = {
                "memory_arm": memory_arm,
                "memory_shots": render_memory_prefix_shots_concat(
                    runtime=runtime,
                    case=case,
                    memory_arm=memory_arm,
                    latent_cache_dir=latent_cache_dir,
                    output_path=output_path,
                    seed=seed,
                    shot_seed_stride=shot_seed_stride,
                    manifest_dir=output_dir,
                    donor_case=donor_case,
                ),
            }
            if donor_case is not None:
                story_provenance["memory_donor_case_id"] = donor_case["case_id"]
        elif first_frame_path:
            if runtime is None:
                raise RuntimeError("continuity_dev_32 runtime was not constructed")
            donor_case = donors.get(case["case_id"])
            story_provenance = {
                "first_frame_arm": first_frame_arm,
                "first_frame_shots": render_first_frame_shots_concat(
                    runtime=runtime,
                    case=case,
                    first_frame_arm=first_frame_arm,
                    first_frame_pin_until_sigma=first_frame_pin_until_sigma,
                    baseline_video=baseline_videos[case["case_id"]] if first_frame_arm == FIRST_FRAME_ARM_EVENT else None,
                    latent_cache_dir=latent_cache_dir,
                    output_path=output_path,
                    seed=seed,
                    shot_seed_stride=shot_seed_stride,
                    manifest_dir=output_dir,
                    donor_case=donor_case,
                ),
            }
            if donor_case is not None:
                story_provenance["first_frame_donor_case_id"] = donor_case["case_id"]
        elif generate_case is None:
            if runtime is None:
                raise RuntimeError("continuity_dev_32 runtime was not constructed")
            story_provenance = runtime.generate_case(
                prompts=case["prompts"],
                shot_groups=case["shot_groups"],
                latent_shot_groups=case["latent_shot_groups"],
                shot_frame_counts=script_shot_frame_counts(case),
                output_path=output_path,
                seed=seed,
            )
        else:
            generate_case(
                case_id=case["case_id"],
                case_inputs=case,
                output_path=output_path,
                seed=seed,
            )
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"continuity_dev_32 video was not written: {output_path}")
        extra_fields = {
            "arm_id": arm_id,
            "generation_seed": seed,
            "shot_ids": case["shot_ids"],
            "shots": case["shots"],
            "frame_count_source": CONTINUITY_DEV_32_FRAME_COUNT_SOURCE,
            "prompt_contract": RENDERED_PROMPT_CONTRACT,
            **story_provenance,
        }
        if baseline_path:
            extra_fields["generation_path"] = "single_shot_concat"
        elif memory_path:
            extra_fields["generation_path"] = GENERATION_PATH_MEMORY_PREFIX
        elif first_frame_path:
            extra_fields["generation_path"] = GENERATION_PATH_FIRST_FRAME
        manifest.append(
            default_manifest_row(
                case_id=case["case_id"],
                output_path=output_path,
                generation_paradigm=manifest_generation_paradigm,
                case_inputs=case,
                source_row=None,
                extra_fields=extra_fields,
            )
        )
    manifest_path = output_dir / "generated_eval_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "arm_id": arm_id,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "case_count": len(manifest),
        "generation_seed": seed,
    }


def parser():
    import argparse

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--config", required=True, type=Path)
    arg_parser.add_argument("--checkpoint", type=Path)
    arg_parser.add_argument("--output-dir", type=Path)
    arg_parser.add_argument("--case-id", action="append", default=[])
    return arg_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    job = load_continuity_dev_32_generate_config(args.config)
    checkpoint = args.checkpoint or job["checkpoint"]
    output_dir = args.output_dir or job["output_dir"]
    if output_dir is None:
        raise ValueError("output_dir is required")
    if job["train_config"] is None or job["benchmark"] is None:
        raise ValueError("train_config and benchmark are required")
    if job["weight_source"] == "full_dit_checkpoint" and checkpoint is None:
        raise ValueError("full_dit_checkpoint requires --checkpoint or config.checkpoint")
    if job["weight_source"] == "pretrained_model_paths" and checkpoint is not None:
        raise ValueError("pretrained_model_paths requires a null checkpoint")
    payload = generate_continuity_dev_32(
        train_config_path=job["train_config"],
        benchmark_path=job["benchmark"],
        output_dir=Path(output_dir),
        weight_source=job["weight_source"],
        seed=job["generation_seed"],
        checkpoint=checkpoint,
        case_ids=args.case_id or job["case_ids"],
        arm_id=job["arm_id"],
        render=job["render"],
        memory_arm=job["memory_arm"],
        memory_donor_offset=job["memory_donor_offset"],
        generation_paradigm=job["generation_paradigm"],
        shot_seed_stride=job["shot_seed_stride"],
        first_frame_arm=job["first_frame_arm"],
        first_frame_pin_until_sigma=job["first_frame_pin_until_sigma"],
        first_frame_baseline_manifest=job["first_frame_baseline_manifest"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
