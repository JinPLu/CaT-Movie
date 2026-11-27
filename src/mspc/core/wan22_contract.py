"""Shared Wan2.2 MSPC experiment contracts.

This module intentionally contains only mode constants shared by the strict
config parser, training backend, and inference runtime.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence


GLOBAL_TIME_NATIVE_ROPE_V1: Final = "global_time_native_rope_v1"
GLOBAL_TIME_MSM_FIXED_SHOT_PHASE_V1: Final = (
    "global_time_msm_fixed_shot_phase_v1"
)
T2V_CONDITIONING_MODE: Final = "t2v"
PER_SHOT_TI2V_FIRST_FRAME_CONDITIONING_MODE: Final = (
    "per_shot_ti2v_first_frame"
)
FIXED_METHOD_VALIDATION_DATA_SCOPE: Final = "fixed_method_validation"
QUALITY_ADMITTED_DATA_SCOPE: Final = "quality_admitted"

#: DiT 自注意力：full 为联合镜头互读；holocine_first_frame 为各镜全部潜帧读首潜帧池与路由参考；
#: first_frame_communication 为只有首潜帧读外部信息（本镜全部＋各镜首潜帧池＋路由参考），其余潜帧只读本镜。
#: holocine_own_shot_only 是 holocine_first_frame 的镜间读取切除（推理时的注意力切除对照）：首潜帧池只留本镜首潜帧
#: （本镜首帧照旧重复一份），每镜只读本镜全部潜帧＋路由参考，不读别的镜。
#: 旧首帧接口的具体可见性见各模型 owner。配置解析与模型加载只认这一份。
WAN22_SELF_ATTENTION_POLICIES: Final[frozenset[str]] = frozenset(
    {"full", "holocine_first_frame", "holocine_own_shot_only", "first_frame_communication", "first_frame_interface"}
)
WAN22_DIFFSYNTH_TEMPORAL_POSITION_MODES: Final[frozenset[str]] = frozenset(
    {
        GLOBAL_TIME_NATIVE_ROPE_V1,
        GLOBAL_TIME_MSM_FIXED_SHOT_PHASE_V1,
    }
)
WAN22_FLOW_MATCH_SHIFT: Final = 5.0
WAN22_CONDITIONING_MODES: Final[frozenset[str]] = frozenset(
    {
        T2V_CONDITIONING_MODE,
        PER_SHOT_TI2V_FIRST_FRAME_CONDITIONING_MODE,
    }
)
WAN22_TRAINING_DATA_SCOPES: Final[frozenset[str]] = frozenset(
    {
        FIXED_METHOD_VALIDATION_DATA_SCOPE,
        QUALITY_ADMITTED_DATA_SCOPE,
    }
)

WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT: Final = "latents_and_context"
WAN22_CACHE_LAYOUT_LATENTS_ONLY: Final = "latents_only"
WAN22_CACHE_LAYOUTS: Final[frozenset[str]] = frozenset(
    {
        WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
        WAN22_CACHE_LAYOUT_LATENTS_ONLY,
    }
)
WAN22_CONTEXT_SOURCE_EMBEDDED: Final = "embedded"
WAN22_CONTEXT_SOURCE_LIVE_T5: Final = "live_t5"
WAN22_CONTEXT_SOURCE_CONTEXT_CACHE: Final = "context_cache"
WAN22_CONTEXT_SOURCES: Final[frozenset[str]] = frozenset(
    {
        WAN22_CONTEXT_SOURCE_EMBEDDED,
        WAN22_CONTEXT_SOURCE_LIVE_T5,
        WAN22_CONTEXT_SOURCE_CONTEXT_CACHE,
    }
)

#: 记忆前缀这条臂的选帧策略，与窗口 manifest 的 ``memory_mode`` 字段同一组名字
#: （生产端 ``tools/data/build_event_memory_windows.py``）。
#:
#: - ``similarity``：B0 抬基线用的相似配对——记忆镜取同故事的其他镜（含未来镜，已被
#:   ``past_nearest`` 取代，只留给 B0 复现）；
#: - ``past_nearest``：阶段 A 地板池——目标镜之前、离它最近的至多 3 镜，按镜序；
#: - ``event``：阶段 B 事件臂——记忆镜取事件前、该实体在场的镜，目标镜取事件后镜；
#: - ``event_matched_control``：阶段 B 对照臂——同一批目标镜，记忆取最近过去镜、镜数与事件
#:   臂相同；
#: - ``ledger``：账本驱动选帧——按实体当前版本所在的最近一镜取帧。
#:
#: 它们是几份**不同的训练／推理配对**，不是同一条路的几档，所以
#: ``dataset.memory_mode`` 必须显式写，没有默认值；数据集只读与它相等的那些行。
MEMORY_MODE_SIMILARITY: Final = "similarity"
MEMORY_MODE_PAST_NEAREST: Final = "past_nearest"
MEMORY_MODE_EVENT: Final = "event"
MEMORY_MODE_EVENT_MATCHED_CONTROL: Final = "event_matched_control"
MEMORY_MODE_LEDGER: Final = "ledger"
MEMORY_MODES: Final[tuple[str, ...]] = (
    MEMORY_MODE_SIMILARITY,
    MEMORY_MODE_PAST_NEAREST,
    MEMORY_MODE_EVENT,
    MEMORY_MODE_EVENT_MATCHED_CONTROL,
    MEMORY_MODE_LEDGER,
)

#: 记忆 token 在 DiT 自注意力里能看谁。字面量与 vendored
#: ``diffsynth.models.wan_video_dit.MEMORY_ATTENTION_POLICIES`` 是同一组（行为的 owner 在
#: 那里），config 与运行时不得各写一份。``bidirectional_v1``（配 ``full``）：参考与目标在层内全双向交互；
#: ``routed_self_only_v1``（配 ``holocine_first_frame``，带 ``reference_routing``）：每个目标镜只读路由给它的参考，
#: 参考只读自己（``SelfAttention.routed_shot_attention``）。
#: 干净参考只在模型输入及每次去噪更新后保持，不冻结层内参考表示。
MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1: Final = "bidirectional_v1"
MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1: Final = "routed_self_only_v1"
MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1: Final = "routed_first_latent_self_only_v1"
#: ``window_shared_self_only_v1``：窗内每个目标镜读全部参考（参考是共用的「首帧」），参考只读自己。注意力实现与
#: ``routed_self_only_v1`` 同一份，只是路由给每镜全部参考（``mspc.data.reference_routing.reference_readers``）。
#: 渲染器补差距实验的候选入口。
MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1: Final = "window_shared_self_only_v1"
ROUTED_MEMORY_ATTENTION_POLICIES: Final[tuple[str, ...]] = (
    MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1,
    MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1,
    MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1,
)
MEMORY_ATTENTION_POLICIES: Final[tuple[str, ...]] = (
    MEMORY_ATTENTION_POLICY_BIDIRECTIONAL_V1,
    *ROUTED_MEMORY_ATTENTION_POLICIES,
)

#: 参考只带固定角色句，不声明未经逐帧核验的身份、服装、场所或源镜动作。
MEMORY_ROW_ROLE_TEXT: Final = (
    "Reference still from another shot of this story; not the current shot."
)
MEMORY_ROW_POLICY_FIXED_ROLE_V1: Final = "fixed_role_v1"
MEMORY_ROW_POLICY_FRAME_ENTITIES_V1: Final = "frame_entities_v1"
#: 在逐帧确认可见的人物之后，再写参考源镜在场物体的外貌短语（训练取块标注、推理取计划，同一定义）。
MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1: Final = "frame_entities_with_shot_objects_v1"
MEMORY_ROW_POLICIES: Final[tuple[str, ...]] = (MEMORY_ROW_POLICY_FIXED_ROLE_V1, MEMORY_ROW_POLICY_FRAME_ENTITIES_V1,
                                              MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1)
#: 参考行带逐帧可见人物ID、因而训练要逐样本身份标签、推理要读图确认的行写法。
FRAME_ENTITY_ROW_POLICIES: Final[frozenset[str]] = frozenset(
    {MEMORY_ROW_POLICY_FRAME_ENTITIES_V1, MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1})
MEMORY_ROW_VISIBLE_IDENTITIES: Final = " Visible identities: "
MEMORY_ROW_SHOT_OBJECTS: Final = " Objects in that shot: "
#: 参考源镜的地点，与目标行「Setting: …」同一写法。
MEMORY_ROW_SETTING: Final = " Setting: "


def memory_prefix_case_layout(
    *,
    target_prompts: Sequence[str],
    target_shot_groups: Sequence[Sequence[int]],
    target_latent_shot_groups: Sequence[Sequence[int]],
    memory_rows: Sequence[str],
    row_policy: str,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """已编译参考行＋联合目标段的唯一文字与几何合同。

    参考一张一组（1 RGB / 1 latent），参考文字按显式row_policy核验；目标分组整体偏移 M。
    目标文字不包装、不规范化；无参考时目标文字与分组逐字节、逐坐标不变。
    """

    for name, rows in (("target_prompts", target_prompts), ("memory_rows", memory_rows)):
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError(f"{name} must be a sequence of strings")
        if any(not isinstance(row, str) for row in rows):
            raise TypeError(f"{name} must contain only strings")
    require_memory_rows(memory_rows, row_policy=row_policy)
    if not target_prompts:
        raise ValueError("target_prompts must contain at least one shot")
    groups = []
    for name, raw_groups in (
        ("target_shot_groups", target_shot_groups),
        ("target_latent_shot_groups", target_latent_shot_groups),
    ):
        if len(raw_groups) != len(target_prompts):
            raise ValueError(f"{name} must have one group per target prompt")
        parsed = []
        cursor = 0
        for group in raw_groups:
            if len(group) != 2 or any(type(value) is not int for value in group):
                raise ValueError(f"{name} entries must be integer [start, end] pairs")
            start, end = group
            if start != cursor or end <= start:
                raise ValueError(f"{name} must be contiguous nonempty groups starting at zero")
            parsed.append((start, end))
            cursor = end
        groups.append(tuple(parsed))
    for rgb, latent in zip(*groups):
        frames = rgb[1] - rgb[0]
        if (frames - 1) % 4 or latent[1] - latent[0] != (frames - 1) // 4 + 1:
            raise ValueError("target RGB and latent groups must obey the Wan 4n+1 contract")
    count = len(memory_rows)
    reference_groups = tuple((index, index + 1) for index in range(count))
    return (
        tuple(memory_rows) + tuple(target_prompts),
        reference_groups + tuple((start + count, end + count) for start, end in groups[0]),
        reference_groups + tuple((start + count, end + count) for start, end in groups[1]),
    )


def require_memory_rows(rows: Sequence[str], *, row_policy: str) -> None:
    """参考角色与已核逐帧ID，不接受整镜caption冒称实际源帧事实。"""
    if row_policy not in MEMORY_ROW_POLICIES:
        raise ValueError(f"row_policy must be one of {list(MEMORY_ROW_POLICIES)}; got {row_policy!r}")
    for index, row in enumerate(rows):
        if not isinstance(row, str):
            raise TypeError(f"memory row {index} must be a string; got {type(row).__name__}")
        prefixes = {MEMORY_ROW_POLICY_FIXED_ROLE_V1: (),
                    MEMORY_ROW_POLICY_FRAME_ENTITIES_V1: (MEMORY_ROW_VISIBLE_IDENTITIES, MEMORY_ROW_SETTING),
                    MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1: (MEMORY_ROW_VISIBLE_IDENTITIES, MEMORY_ROW_SETTING,
                                                                           MEMORY_ROW_SHOT_OBJECTS)}[row_policy]
        valid = (row == MEMORY_ROW_ROLE_TEXT or any(row.startswith(MEMORY_ROW_ROLE_TEXT + prefix) for prefix in prefixes)) and (
            MEMORY_ROW_SHOT_OBJECTS in prefixes or MEMORY_ROW_SHOT_OBJECTS not in row)
        if not valid:
            raise ValueError(
                f"memory row {index} is not a {row_policy!r} memory row "
                f"(role sentence {MEMORY_ROW_ROLE_TEXT!r}): {row!r}"
            )


#: 按源时间升序的 M 张参考占 -5M, ..., -5；空间在原网格（dh=dw=0）或整块移出目标网格（11-05 起最终配方取一个网格的高和宽）。
#: 训练与推理显式声明同一参考坐标合同；干净首帧的原生 TI2V 路径没有参考偏移。
#: 目标时间和 MSM 镜相位仍从 0 开始；独立参考编号向量已随逐张时间槽设计删除。
MEMORY_REFERENCE_TEMPORAL_MODE: Final = GLOBAL_TIME_MSM_FIXED_SHOT_PHASE_V1

#: 一个生成窗口最多多少帧。渲染器训练窗口只受它限制、镜数不封顶。
SEGMENT_MAX_FRAMES: Final = 228
#: 公开分镜表的时间坐标；不同原生媒体只派生执行帧数，不改变共同计划或源窗。
STORYBOARD_FPS: Final = 16
STORYBOARD_MAX_SHOT_FRAMES: Final = 1 + (SEGMENT_MAX_FRAMES - 1) // 4 * 4
#: 旧的至多 4 镜窗口。出片不再读它（11-06 起窗口镜数是每个条件里渲染器训练时的值：最终联合配方不限、逐镜 1、
#: 旧渲染器 4）；还读它的是事件日志编译器切段与 ``training.block_dataset.pack_span``——后者定义了论文基准统计里
#: "跨段依赖"的段（``tools/analysis/benchmark_long_range_dependencies.py`` 等），改它要连同那些统计一起重定。
SEGMENT_MAX_SHOTS: Final = 4


def require_compiler_capacity(capacity: Mapping[str, int]) -> None:
    """训练窗口容量只有这一处；配置必须交出同一帧上限，不能另切。"""
    expected = {"max_frames": SEGMENT_MAX_FRAMES}
    if (not isinstance(capacity, Mapping) or set(capacity) != set(expected)
            or any(type(capacity[name]) is not int for name in expected)
            or any(capacity[name] != expected[name] for name in expected)):
        raise ValueError(
            "segment_capacity must equal the compiler dependency segment capacity "
            f"{expected}"
        )


def frame_counts_over_segment_cap(frame_counts: Sequence[int]) -> tuple[int, ...]:
    """单镜超过帧上限的下标。这是帧数检查，不是另一套切段。"""
    over: list[int] = []
    for index, count in enumerate(frame_counts):
        if type(count) is not int or count <= 0:
            raise ValueError(f"shot {index} frame count {count!r} is invalid")
        if count > SEGMENT_MAX_FRAMES:
            over.append(index)
    return tuple(over)


def noise_memory_prefix(memory_latents, sigmas, *, generator_seed: int):
    """离线记忆位置探针的连续噪声混合 ``(1−σ)m+σε``；新段训练不用此路径。

    每张参考一个 sigma，独立 CPU generator 不改变全局 RNG；float32 混合后恢复
    输入 dtype。该数学函数保留给 run_memory_position_probe 的显式诊断输入。
    """

    import torch

    if not isinstance(memory_latents, torch.Tensor) or memory_latents.ndim != 5:
        raise ValueError("memory_latents must be a [1, C, M, H, W] tensor")
    frames = int(memory_latents.shape[2])
    sigmas = torch.as_tensor(sigmas, dtype=torch.float64)
    if sigmas.shape != (frames,):
        raise ValueError(
            f"sigmas must carry one value per memory frame ({frames}); got {tuple(sigmas.shape)}"
        )
    if bool((sigmas < 0).any()) or bool((sigmas >= 1).any()):
        raise ValueError(f"memory sigmas must lie in [0, 1); got {sigmas.tolist()}")
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    noise = torch.randn(
        tuple(memory_latents.shape), generator=generator, dtype=torch.float32
    ).to(memory_latents.device)
    sigma = sigmas.to(device=memory_latents.device, dtype=torch.float32).view(1, 1, frames, 1, 1)
    return ((1.0 - sigma) * memory_latents.float() + sigma * noise).to(memory_latents.dtype)
