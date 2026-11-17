"""``script_state_bench``：论文报告集的严格校验。

四份集共用一条校验路径，差别全在一张规则表 :data:`RULES`（按 ``schema_version`` 取）：

- v1（冻结于 ``9e0e2d53``，读数留作历史）与 v2（09-30 整体重写，副表）：2–5 镜手写剧本，整条一次出片；
- 主表 A（``script_state_bench_long_a_v1``）：6–12 镜真片反构，镜长取真片、事件可以没有、带真片出处；
- 主表 B（``script_state_bench_long_b_v1``）：在 A 的剧本上注入 1–2 处状态变化，无真片参照。

6–12 镜的故事超过渲染链单段上限，要分段出片；这里只钉单镜不超单段上限，并在统计里报按单段上限
（≤308 帧、≤5 镜）至少要切几段（:func:`mspc.benchmark.long_story_holdout.min_segments`）。

这份集是手写剧本，没有真片；它要考的是「事件后的状态穿过硬切」，并且换人、换景、隔镜回访时
仍然成立。断言编译器从同一份 JSON 读 ``subjects``/``locations``/``present_subjects``/``events``，
所以这里钉的是让断言无歧义的那些结构约束：

1. **生成契约**：``evaluation_only``、fps 按规则表（手写集冻结于 15，长故事 A／B 取时空契约 16）、seeds ``[42]``；镜长按规则表的 ``frames``：手写集事件镜
   ``frame_count`` 恰为 :data:`EVENT_SHOT_FRAMES`、其余镜恰为 :data:`NON_EVENT_SHOT_FRAMES`，整条不超渲染链
   ``max_rgb_frames`` 并能过 ``continuity_dev_32.case_inputs_from_benchmark_case``（渲染链读 benchmark 的唯一
   入口）；A 每镜等于真片该镜时长换成的 4n+1 帧（``source_span`` 与 ``source.fps``）；B 必须与配对 A 的镜数及逐镜帧数相同，事件在原时长内完成，
   不另加 81 帧下限。长故事单镜不超 ``max_rgb_frames``。
2. **标签闭合**：``story_prompt`` 逐字等于由 ``subjects`` 编出的训练同款主体锚
   ``Subject 1: …; Subject 2: ….``；每镜 prompt 里出现的 ``Subject N`` 恰好是该镜
   ``present_subjects``，并逐字含该镜地点的 ``attributes``（回访同一地点时描述一致）。
3. **事件可判**：事件数在规则表的 ``events`` 区间内；一镜至多一个事件（事件镜只演一个动作）；
   ``prior_state``／``post_state`` 各不超 :data:`STATE_MAX_WORDS` 词；事件主体在事件镜在场；``post_state``
   里点名的别的主体，在该状态仍成立、且事件主体在场的后续镜里也必须在场；每个事件的主体之后至少再出现
   一次；规则表要求时，每条故事至少有一处「状态穿过换人、换景或隔镜回访的硬切」
   （:func:`crosses_context_change`）。
4. **标签可导出**：``tags`` 恰好等于由结构导出的集合（见 :func:`derive_tags`），不手填。
5. **不与旧集重叠**：任何 prompt 与参照集（DEV32、main_table_96；v2 另加 v1）的任一 prompt 不共享
   :data:`OVERLAP_NGRAM` 词的连续片段。
6. **构图与镜界**（v2、A、B）：每镜 ``shot_size``、``camera_angle`` 与非首镜的 ``cut_type`` 取训练标注同一份词表
   （``mspc.data.cut_type_labels``：Wan2.2 开源扩写器的景别、拍摄角度原词，加平视；镜界五类）；
   非首镜的 ``cut_type`` 必须等于由地点、在场与构图导出的类型（:func:`derive_cut_type`），回机位另记
   ``returns_to``；非续拍的镜界景别或机位至少一项与上一镜不同；prompt 以
   ``<景别>, <机位（首字母小写）>, <地点 attributes 原文>. `` 开头，续拍前面再加 :data:`CONTINUATION_PREFIX`。
7. **v2 全集配额**：各镜界类型的占比不低于 :data:`CUT_TYPE_MIN_SHARE`，任一景别不超过
   :data:`MAX_SHOT_SIZE_SHARE`。A 的镜界照真片，不设配额，只报分布。
8. **画面方位与朝向**（A、B）：每镜 ``facing`` 是镜头正对的在场主体（看不到正脸、或是空镜时 ``null``），
   并入构图（:func:`framing`），正反打因此导出成换机位而不是续拍。每镜 ``blocking`` 是在场主体从左到右的顺序（≥2 个、互不相同），画面里分不出
   左右时写 ``null``；有顺序的镜，prompt 要写出 :data:`LEFT_PHRASE` 与 :data:`RIGHT_PHRASE`。判法是检测框
   排左右（空间尺子坏例验收已过），不进判官。
9. **出处**（A）：``source`` 记片、场景、源视频、帧率、预卷与裁切框，每镜 ``source_span`` 是源视频里的
   半开帧区间；B 的 ``derived_from`` 是它改写的 A 故事。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dataclasses import dataclass

from mspc.benchmark.long_story_holdout import RENDER_FPS, min_segments, render_frame_count
from mspc.data.cut_type_labels import CAMERA_ANGLES, CUT_TYPES, SHOT_SIZES
from mspc.inference.continuity_dev_32 import (
    SCRIPT_STATE_BENCH_V1_SCHEMA,
    SCRIPT_STATE_BENCH_V2_SCHEMA,
    WAN22_CACHED_SPACETIME_CONTRACT,
    case_inputs_from_benchmark_case,
    load_continuity_dev_32_benchmark,
)

#: 手写集 v1／v2 冻结于 15 fps（镜长 45／81 帧按 15 fps 定），不随时空契约改；长故事 A／B 的帧率是 :data:`RENDER_FPS`。
SHORT_SET_FPS = 15
GENERATION_SEEDS = (42,)
#: 非事件镜：训练镜长中位数（real_holdout_96 的 336 个真镜，同一 205k 池、同一 latent 缓存）；4n+1，n=11。
NON_EVENT_SHOT_FRAMES = 45
#: 事件镜：试点里 3 s 演不完一个动作的终态（09-24），加到约 5.4 s；4n+1，n=20。
EVENT_SHOT_FRAMES = 81
#: 整条故事的 RGB 总帧数上限，取渲染链自己的契约值。
MAX_STORY_FRAMES = int(WAN22_CACHED_SPACETIME_CONTRACT["max_rgb_frames"])
#: 事件前后状态的词数上限：短、可见、能二值判。
STATE_MAX_WORDS = 8
SCRIPT_STATE_BENCH_LONG_A_SCHEMA = "script_state_bench_long_a_v1"
SCRIPT_STATE_BENCH_LONG_B_SCHEMA = "script_state_bench_long_b_v1"
#: 手写集取 45／81；A 取真片时长；B 在配对校验中逐镜复用 A 的帧数。
FRAMES_EVENT_RULE, FRAMES_SOURCE, FRAMES_PAIRED_A = "event_rule", "source", "paired_a"
DEFAULT_LONG_B_INJECTIONS = 1
MAX_LONG_B_INJECTIONS = 2


@dataclass(frozen=True)
class BenchRules:
    """一份集的结构规则。字段都是显式的，没有默认值。"""

    #: 清单 ``fps`` 必须等于它，也是整条故事时长统计的帧率。
    fps: int
    shot_range: tuple[int, int]
    max_subjects: int
    max_animate_subjects: int
    max_locations: int
    #: 每镜带 ``shot_size``／``camera_angle``，非首镜带 ``cut_type``，prompt 以构图开头。
    framing: bool
    frames: str
    #: 事件数区间（含两端）。
    events: tuple[int, int]
    #: 每条故事至少一处状态穿过换人、换景或隔镜回访。
    require_crossing: bool
    #: 整条故事一次出片：总帧数不超 ``max_rgb_frames``，并过渲染链读入口。
    single_pass: bool
    #: 全集镜界与景别配额（:func:`validate_quotas`）。
    quotas: bool
    #: 非续拍的镜界必须换景别或机位（手写集的「切镜带来新信息」；真片正反打常是同景别，不要求）。
    cut_changes_framing: bool
    #: 每镜带 ``blocking``。
    blocking: bool
    #: 允许一镜没有任何主体在场（真片的空镜、人群镜、风景镜）。
    allow_empty_cast: bool
    case_extra_fields: frozenset[str]
    shot_extra_fields: frozenset[str]


SOURCE_FIELDS = frozenset(
    {"film_key", "scene_id", "video_id", "source_video_path", "fps", "container_preroll_frames", "crop_box"}
)
#: 手写集与长故事的规则表；校验只从这里读差别。
RULES = {
    SCRIPT_STATE_BENCH_V1_SCHEMA: BenchRules(
        fps=SHORT_SET_FPS, shot_range=(2, 5), max_subjects=4, max_animate_subjects=3, max_locations=3, framing=False,
        frames=FRAMES_EVENT_RULE, events=(1, 5), require_crossing=True, single_pass=True, quotas=False,
        cut_changes_framing=False, blocking=False, allow_empty_cast=False, case_extra_fields=frozenset(),
        shot_extra_fields=frozenset(),
    ),
    SCRIPT_STATE_BENCH_V2_SCHEMA: BenchRules(
        fps=SHORT_SET_FPS, shot_range=(2, 5), max_subjects=4, max_animate_subjects=3, max_locations=3, framing=True,
        frames=FRAMES_EVENT_RULE, events=(1, 5), require_crossing=True, single_pass=True, quotas=True,
        cut_changes_framing=True, blocking=False, allow_empty_cast=False, case_extra_fields=frozenset(),
        shot_extra_fields=frozenset(),
    ),
    SCRIPT_STATE_BENCH_LONG_A_SCHEMA: BenchRules(
        fps=RENDER_FPS, shot_range=(6, 12), max_subjects=5, max_animate_subjects=4, max_locations=3, framing=True,
        frames=FRAMES_SOURCE, events=(0, 3), require_crossing=False, single_pass=False, quotas=False,
        cut_changes_framing=False, blocking=True, allow_empty_cast=True, case_extra_fields=frozenset({"source"}),
        shot_extra_fields=frozenset({"blocking", "facing", "source_span"}),
    ),
    SCRIPT_STATE_BENCH_LONG_B_SCHEMA: BenchRules(
        fps=RENDER_FPS, shot_range=(6, 12), max_subjects=5, max_animate_subjects=4, max_locations=3, framing=True,
        frames=FRAMES_PAIRED_A, events=(1, 5), require_crossing=True, single_pass=False, quotas=False,
        cut_changes_framing=False, blocking=True, allow_empty_cast=True,
        case_extra_fields=frozenset({"derived_from", "injected_event_ids"}), shot_extra_fields=frozenset({"blocking", "facing"}),
    ),
}
#: 有画面方位的镜，prompt 里要写出的左右短语（与文字方位服从检查的写法一致）。
LEFT_PHRASE = "left side of the frame"
RIGHT_PHRASE = "right side of the frame"
SHOT_PROMPT_WORDS = (20, 100)
#: ``Story: {story_prompt} Now: {shot prompt}`` 的词数上限，给文字设定／账本臂在 umT5 512 里留余量。
STORY_NOW_MAX_WORDS = 180
OVERLAP_NGRAM = 8
SUBJECT_KINDS = ("person", "animal", "object")
ANIMATE_KINDS = frozenset({"person", "animal"})
TAG_ORDER = (
    "cast_change",
    "location_change",
    "revisit",
    "location_revisit",
    "subject_exit",
    "attribute_change",
    "object_state_change",
    "ownership_transfer",
)

MANIFEST_FIELDS = frozenset(
    {"schema_version", "benchmark_id", "evaluation_only", "fps", "generation_seeds", "cases"}
)
CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "shot_count",
        "story_prompt",
        "subjects",
        "locations",
        "shots",
        "events",
        "tags",
    }
)
SUBJECT_FIELDS = frozenset({"subject_id", "kind", "attributes"})
LOCATION_FIELDS = frozenset({"location_id", "attributes"})
SHOT_FIELDS = frozenset({"shot_id", "prompt", "location_id", "present_subjects", "frame_count"})
SCHEMAS = tuple(RULES)

CONTINUATION, SAME_SCENE_NEW_CAMERA, RETURN_TO_SETUP, SAME_SCENE_CAST_CHANGE, SCENE_CHANGE = CUT_TYPES
#: v2 全集镜界配额：每类镜界至少占全部镜界的比例。
CUT_TYPE_MIN_SHARE = {
    SAME_SCENE_NEW_CAMERA: 0.35,
    RETURN_TO_SETUP: 0.08,
    CONTINUATION: 0.08,
    SAME_SCENE_CAST_CHANGE: 0.08,
    SCENE_CHANGE: 0.08,
}
#: v2 全集任一景别的镜数占比上限（v1 有 278/350 镜是中景）。
MAX_SHOT_SIZE_SHARE = 0.40
#: 续拍镜 prompt 的开头（续拍写 Continuous shot 且构图不变）。
CONTINUATION_PREFIX = "Continuous shot. "
#: 过肩镜头：前景是一个人的肩背，画面里至少要有两个人。
OVER_THE_SHOULDER = "Over-the-shoulder shot"
FRAMING_FIELDS = frozenset({"shot_size", "camera_angle"})
EVENT_FIELDS = frozenset({"event_id", "shot_id", "subject_id", "prior_state", "post_state"})

_SUBJECT_REF = re.compile(r"\bSubject (\d+)\b")
_LOCATION_REF = re.compile(r"\bLocation \d+\b")
_WORD = re.compile(r"[a-z0-9']+")


def _exact_fields(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != fields:
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - set(value))}, "
            f"unknown={sorted(set(value) - fields)}"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _subject_refs(text: str) -> set[str]:
    return {f"Subject {number}" for number in _SUBJECT_REF.findall(text)}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + OVERLAP_NGRAM]) for i in range(len(words) - OVERLAP_NGRAM + 1)}


def roster_prompt(subjects: Sequence[Mapping[str, Any]]) -> str:
    """训练同款静态主体锚：``Subject 1: attrs; Subject 2: attrs.``。"""

    return "; ".join(f"{row['subject_id']}: {row['attributes']}" for row in subjects) + "."


def derive_tags(case: Mapping[str, Any]) -> list[str]:
    """由镜序、在场表与事件导出 ``tags``（按 :data:`TAG_ORDER` 排序）。

    - ``cast_change``／``location_change``：某对相邻镜的在场集合／地点不同；
    - ``revisit``：某主体在场、离开至少一镜、再回来；``location_revisit`` 同理对地点；
    - ``subject_exit``：某主体在某镜之后再不出现（之后各镜都是「不该出现」）；
    - 事件三类按 ``post_state`` 判：点名别的主体 → ``ownership_transfer``；否则主体是
      object → ``object_state_change``，是 person/animal → ``attribute_change``。
    """

    shots = case["shots"]
    casts = [frozenset(shot["present_subjects"]) for shot in shots]
    locations = [shot["location_id"] for shot in shots]
    count = len(shots)
    tags: set[str] = set()
    for index in range(count - 1):
        if casts[index] != casts[index + 1]:
            tags.add("cast_change")
        if locations[index] != locations[index + 1]:
            tags.add("location_change")
    kinds = {row["subject_id"]: row["kind"] for row in case["subjects"]}
    for subject_id in kinds:
        seen = [index for index in range(count) if subject_id in casts[index]]
        if any(later - earlier > 1 for earlier, later in zip(seen, seen[1:])):
            tags.add("revisit")
        if seen[-1] < count - 1:
            tags.add("subject_exit")
    for location_id in set(locations):
        seen = [index for index in range(count) if locations[index] == location_id]
        if any(later - earlier > 1 for earlier, later in zip(seen, seen[1:])):
            tags.add("location_revisit")
    for event in case["events"]:
        if _subject_refs(event["post_state"]) - {event["subject_id"]}:
            tags.add("ownership_transfer")
        elif kinds[event["subject_id"]] == "object":
            tags.add("object_state_change")
        else:
            tags.add("attribute_change")
    return [tag for tag in TAG_ORDER if tag in tags]


def shot_frame_count(case: Mapping[str, Any], shot_id: str) -> int:
    """该镜应有的帧数：有事件的镜 :data:`EVENT_SHOT_FRAMES`，其余 :data:`NON_EVENT_SHOT_FRAMES`。"""

    if any(event["shot_id"] == shot_id for event in case["events"]):
        return EVENT_SHOT_FRAMES
    return NON_EVENT_SHOT_FRAMES


def state_holding_shots(case: Mapping[str, Any], event: Mapping[str, Any]) -> list[int]:
    """事件 ``post_state`` 在哪些后续镜（0 起下标）可判：事件镜之后、同主体下一事件镜之前、主体在场。"""

    shot_index = {shot["shot_id"]: index for index, shot in enumerate(case["shots"])}
    start = shot_index[event["shot_id"]]
    stop = len(case["shots"])
    for other in case["events"]:
        other_index = shot_index[other["shot_id"]]
        if other["subject_id"] == event["subject_id"] and other_index > start:
            stop = min(stop, other_index)
    return [
        index
        for index in range(start + 1, stop)
        if event["subject_id"] in case["shots"][index]["present_subjects"]
    ]


def crosses_context_change(shots: Sequence[Mapping[str, Any]], at: int, later: int) -> bool:
    """事件镜 ``at`` 到判定镜 ``later`` 之间（含 ``later``）是否有镜换了地点或在场集合。

    这把三种情形都算作「状态要穿过的变化」：判定镜本身换人／换景；以及隔镜回访——中间镜
    换过人或景，判定镜又回到与事件镜相同的人与景。
    """

    origin = (shots[at]["location_id"], set(shots[at]["present_subjects"]))
    return any(
        (shots[index]["location_id"], set(shots[index]["present_subjects"])) != origin
        for index in range(at + 1, later + 1)
    )


def framing(shot: Mapping[str, Any]) -> tuple[str | None, ...]:
    """一镜的构图：（景别, 机位）；长故事集另加 ``facing``（镜头正对谁）。

    真片的正反打常是同景别、同机位（两个过肩镜都是中近景），只差镜头对着谁；不加这一项，反打会被
    导出成续拍。``facing`` 是长故事集的必填字段（规则表 ``shot_extra_fields``），手写集没有这个键。
    """

    if "facing" in shot:
        return shot["shot_size"], shot["camera_angle"], shot["facing"]
    return shot["shot_size"], shot["camera_angle"]


def derive_cut_type(shots: Sequence[Mapping[str, Any]], index: int) -> tuple[str, str | None]:
    """第 ``index`` 镜（非首镜）相对上一镜的镜界类型，由地点、在场与构图导出；回机位另给回到的镜号。

    - 地点变了：``scene_change``；
    - 同地点、在场集合变了：``same_scene_cast_change``；
    - 同地点、同在场、构图与上一镜相同：``continuation``；
    - 同地点、同在场、构图与本段同地点连续镜里更早一镜（不是上一镜）相同：``return_to_setup``，
      回到最近的那一镜；那一镜的在场必须与本镜相同，否则类型含糊，直接拒收；
    - 其余（本段同地点里没用过的构图）：``same_scene_new_camera``。
    """

    previous, shot = shots[index - 1], shots[index]
    if shot["location_id"] != previous["location_id"]:
        return SCENE_CHANGE, None
    if shot["present_subjects"] != previous["present_subjects"]:
        return SAME_SCENE_CAST_CHANGE, None
    if framing(shot) == framing(previous):
        return CONTINUATION, None
    start = index - 1
    while start > 0 and shots[start - 1]["location_id"] == shot["location_id"]:
        start -= 1
    earlier = [back for back in range(start, index - 1) if framing(shots[back]) == framing(shot)]
    if not earlier:
        return SAME_SCENE_NEW_CAMERA, None
    setup = shots[earlier[-1]]
    if setup["present_subjects"] != shot["present_subjects"]:
        raise ValueError(
            f"{shot['shot_id']} repeats the framing of {setup['shot_id']} with a different cast; "
            "the cut type is ambiguous"
        )
    return RETURN_TO_SETUP, setup["shot_id"]


def shot_prompt_head(shot: Mapping[str, Any], location_attributes: str, cut_type: str | None) -> str:
    """v2 一镜 prompt 必须的开头：``<景别>, <机位>, <地点 attributes>. ``，续拍前加 :data:`CONTINUATION_PREFIX`。

    景别、机位都是词表原词；机位在句中，首字母小写（``Medium shot, low angle shot, …``）。
    """

    angle = shot["camera_angle"]
    head = f"{shot['shot_size']}, {angle[0].lower()}{angle[1:]}, {location_attributes}. "
    return CONTINUATION_PREFIX + head if cut_type == CONTINUATION else head


def _shot_fields(raw: object, index: int, rules: BenchRules) -> frozenset[str]:
    fields = SHOT_FIELDS | rules.shot_extra_fields
    if not rules.framing:
        return fields
    fields |= FRAMING_FIELDS
    if index == 0:
        return fields
    cut_type = raw.get("cut_type") if isinstance(raw, dict) else None
    return fields | {"cut_type"} | ({"returns_to"} if cut_type == RETURN_TO_SETUP else set())


def _validate_framing(
    case: Mapping[str, Any], label: str, location_text: Mapping[str, str], rules: BenchRules
) -> None:
    kinds = {row["subject_id"]: row["kind"] for row in case["subjects"]}
    shots = case["shots"]
    for index, shot in enumerate(shots):
        shot_label = f"{label}.shots[{index}]"
        if shot["shot_size"] not in SHOT_SIZES:
            raise ValueError(f"{shot_label}.shot_size must be one of {SHOT_SIZES}")
        if shot["camera_angle"] not in CAMERA_ANGLES:
            raise ValueError(f"{shot_label}.camera_angle must be one of {CAMERA_ANGLES}")
        if shot["camera_angle"] == OVER_THE_SHOULDER and sum(
            kinds[subject] == "person" for subject in shot["present_subjects"]
        ) < 2:
            raise ValueError(f"{shot_label}: an over-the-shoulder shot needs two present persons")
        cut_type = None
        if index > 0:
            if shot["cut_type"] not in CUT_TYPES:
                raise ValueError(f"{shot_label}.cut_type must be one of {CUT_TYPES}")
            derived, returns_to = derive_cut_type(shots, index)
            if shot["cut_type"] != derived:
                raise ValueError(
                    f"{shot_label}.cut_type is {shot['cut_type']!r} but location, cast and framing make it "
                    f"{derived!r}"
                )
            if derived == RETURN_TO_SETUP and shot["returns_to"] != returns_to:
                raise ValueError(f"{shot_label}.returns_to must be {returns_to!r}")
            if (
                rules.cut_changes_framing
                and derived != CONTINUATION
                and framing(shot) == framing(shots[index - 1])
            ):
                raise ValueError(f"{shot_label}: a {derived} cut must change the shot size or camera angle")
            cut_type = derived
        head = shot_prompt_head(shot, location_text[shot["location_id"]], cut_type)
        if not shot["prompt"].startswith(head):
            raise ValueError(f"{shot_label}.prompt must open with {head!r}")


def _validate_blocking(shot: Mapping[str, Any], label: str) -> None:
    blocking = shot["blocking"]
    if blocking is None:
        return
    present = shot["present_subjects"]
    if (
        not isinstance(blocking, list)
        or len(blocking) < 2
        or len(set(blocking)) != len(blocking)
        or not set(blocking) <= set(present)
    ):
        raise ValueError(f"{label}.blocking must be null or >=2 distinct present subjects, left to right")
    if LEFT_PHRASE not in shot["prompt"] or RIGHT_PHRASE not in shot["prompt"]:
        raise ValueError(f"{label}: a shot with blocking must say {LEFT_PHRASE!r} and {RIGHT_PHRASE!r}")


def _validate_source(case: Mapping[str, Any], label: str) -> float:
    source = _exact_fields(case["source"], SOURCE_FIELDS, f"{label}.source")
    for key in ("film_key", "scene_id", "video_id", "source_video_path"):
        _text(source[key], f"{label}.source.{key}")
    if not isinstance(source["fps"], (int, float)) or source["fps"] <= 0:
        raise ValueError(f"{label}.source.fps must be positive")
    box = source["crop_box"]
    if not (isinstance(box, list) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]):
        raise ValueError(f"{label}.source.crop_box must be [x0, y0, x1, y1]")
    previous_end = None
    for index, shot in enumerate(case["shots"]):
        span = shot["source_span"]
        if not (isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span) and span[0] < span[1]):
            raise ValueError(f"{label}.shots[{index}].source_span must be [start, end) source frames")
        if previous_end is not None and span[0] < previous_end:
            raise ValueError(f"{label}.shots[{index}].source_span must follow the previous shot")
        previous_end = span[1]
    return float(source["fps"])


def expected_frame_count(case: Mapping[str, Any], shot: Mapping[str, Any], rules: BenchRules) -> int | None:
    """规则表给出的帧数；B 返回 ``None``，逐镜帧数由必需的 A/B 配对校验负责。"""

    if rules.frames == FRAMES_EVENT_RULE:
        return shot_frame_count(case, shot["shot_id"])
    if rules.frames == FRAMES_SOURCE:
        start, end = shot["source_span"]
        return render_frame_count((end - start) / float(case["source"]["fps"]))
    return None


def _validate_frames(case: Mapping[str, Any], label: str, rules: BenchRules) -> None:
    shots = case["shots"]
    for index, shot in enumerate(shots):
        count = shot["frame_count"]
        if type(count) is not int or count < 5 or count % 4 != 1 or count > MAX_STORY_FRAMES:
            raise ValueError(f"{label}.shots[{index}].frame_count must be 4n+1 and <= {MAX_STORY_FRAMES}")
        expected = expected_frame_count(case, shot, rules)
        if expected is not None and count != expected:
            raise ValueError(f"{label}.shots[{index}].frame_count must be {expected}")
    if not rules.single_pass:
        return
    planned = [shot["frame_count"] for shot in shots]
    if sum(planned) > MAX_STORY_FRAMES:
        raise ValueError(f"{label}: {sum(planned)} frames exceed max_rgb_frames={MAX_STORY_FRAMES}")
    inputs = case_inputs_from_benchmark_case(case)
    if [end - start for start, end in inputs["shot_groups"]] != planned:
        raise ValueError(f"{label}: frame counts were padded by the renderer plan")


def _validate_case(case: dict[str, Any], label: str, schema: str) -> None:
    rules = RULES[schema]
    if case["schema_version"] != schema:
        raise ValueError(f"{label}.schema_version must equal {schema!r}")
    shot_count = case["shot_count"]
    low, high = rules.shot_range
    if type(shot_count) is not int or not low <= shot_count <= high:
        raise ValueError(f"{label}.shot_count must be an integer in [{low}, {high}]")

    subjects = case["subjects"]
    if not isinstance(subjects, list) or not 1 <= len(subjects) <= rules.max_subjects:
        raise ValueError(f"{label}.subjects must list 1..{rules.max_subjects} subjects")
    for index, raw in enumerate(subjects):
        row = _exact_fields(raw, SUBJECT_FIELDS, f"{label}.subjects[{index}]")
        if row["subject_id"] != f"Subject {index + 1}":
            raise ValueError(f"{label}.subjects[{index}].subject_id must be 'Subject {index + 1}'")
        if row["kind"] not in SUBJECT_KINDS:
            raise ValueError(f"{label}.subjects[{index}].kind must be one of {SUBJECT_KINDS}")
        _text(row["attributes"], f"{label}.subjects[{index}].attributes")
        if _subject_refs(row["attributes"]) or ";" in row["attributes"]:
            raise ValueError(f"{label}.subjects[{index}].attributes must not name labels or use ';'")
    subject_ids = [row["subject_id"] for row in subjects]
    if sum(row["kind"] in ANIMATE_KINDS for row in subjects) > rules.max_animate_subjects:
        raise ValueError(f"{label} has more than {rules.max_animate_subjects} person/animal subjects")
    if case["story_prompt"] != roster_prompt(subjects):
        raise ValueError(f"{label}.story_prompt must equal the roster compiled from subjects")

    locations = case["locations"]
    if not isinstance(locations, list) or not 1 <= len(locations) <= rules.max_locations:
        raise ValueError(f"{label}.locations must list 1..{rules.max_locations} locations")
    location_text: dict[str, str] = {}
    for index, raw in enumerate(locations):
        row = _exact_fields(raw, LOCATION_FIELDS, f"{label}.locations[{index}]")
        if row["location_id"] != f"Location {index + 1}":
            raise ValueError(f"{label}.locations[{index}].location_id must be 'Location {index + 1}'")
        location_text[row["location_id"]] = _text(row["attributes"], f"{label}.locations[{index}].attributes")

    shots = case["shots"]
    if not isinstance(shots, list) or len(shots) != shot_count:
        raise ValueError(f"{label}.shots must contain shot_count entries")
    used_subjects: set[str] = set()
    used_locations: set[str] = set()
    for index, raw in enumerate(shots):
        shot_label = f"{label}.shots[{index}]"
        shot = _exact_fields(raw, _shot_fields(raw, index, rules), shot_label)
        if shot["shot_id"] != f"s{index + 1:02d}":
            raise ValueError(f"{shot_label}.shot_id must be 's{index + 1:02d}'")
        if shot["location_id"] not in location_text:
            raise ValueError(f"{shot_label}.location_id is not a declared location")
        used_locations.add(shot["location_id"])
        present = shot["present_subjects"]
        if (
            not isinstance(present, list)
            or not (present or rules.allow_empty_cast)
            or not set(present) <= set(subject_ids)
            or present != sorted(set(present), key=subject_ids.index)
        ):
            raise ValueError(
                f"{shot_label}.present_subjects must be a duplicate-free subset of subjects in roster order"
                f"{'' if rules.allow_empty_cast else ', non-empty'}"
            )
        used_subjects.update(present)
        prompt = _text(shot["prompt"], f"{shot_label}.prompt")
        if _subject_refs(prompt) != set(present):
            raise ValueError(
                f"{shot_label}.prompt names {sorted(_subject_refs(prompt))} but present_subjects is "
                f"{present}"
            )
        if _LOCATION_REF.search(prompt):
            raise ValueError(f"{shot_label}.prompt must describe the location, not name its label")
        if location_text[shot["location_id"]] not in prompt:
            raise ValueError(f"{shot_label}.prompt must contain its location attributes verbatim")
        words = len(prompt.split())
        if not SHOT_PROMPT_WORDS[0] <= words <= SHOT_PROMPT_WORDS[1]:
            raise ValueError(f"{shot_label}.prompt has {words} words, outside {SHOT_PROMPT_WORDS}")
        if len(case["story_prompt"].split()) + words + 2 > STORY_NOW_MAX_WORDS:
            raise ValueError(f"{shot_label}: Story/Now prompt exceeds {STORY_NOW_MAX_WORDS} words")
        if rules.blocking:
            _validate_blocking(shot, shot_label)
        if "facing" in rules.shot_extra_fields and not (shot["facing"] is None or shot["facing"] in present):
            raise ValueError(f"{shot_label}.facing must be null or one of the present subjects")
    if used_subjects != set(subject_ids):
        raise ValueError(f"{label}: subjects never present in any shot: {sorted(set(subject_ids) - used_subjects)}")
    if used_locations != set(location_text):
        raise ValueError(f"{label}: locations never used: {sorted(set(location_text) - used_locations)}")
    if rules.framing:
        _validate_framing(case, label, location_text, rules)
    if "source" in rules.case_extra_fields:
        _validate_source(case, label)
    if "derived_from" in rules.case_extra_fields:
        _text(case["derived_from"], f"{label}.derived_from")

    events = case["events"]
    low, high = rules.events
    if not isinstance(events, list) or not low <= len(events) <= high:
        raise ValueError(f"{label}.events must list {low}..{high} events")
    shot_index = {shot["shot_id"]: index for index, shot in enumerate(shots)}
    previous = -1
    event_shots: set[str] = set()
    crosses_change = False
    for index, raw in enumerate(events):
        event_label = f"{label}.events[{index}]"
        event = _exact_fields(raw, EVENT_FIELDS, event_label)
        if event["event_id"] != f"e{index + 1}":
            raise ValueError(f"{event_label}.event_id must be 'e{index + 1}'")
        if event["shot_id"] not in shot_index:
            raise ValueError(f"{event_label}.shot_id is not a shot of this case")
        at = shot_index[event["shot_id"]]
        if at < previous:
            raise ValueError(f"{event_label}: events must be ordered by shot")
        previous = at
        subject_id = event["subject_id"]
        if subject_id not in shots[at]["present_subjects"]:
            raise ValueError(f"{event_label}.subject_id must be present in {event['shot_id']}")
        if event["shot_id"] in event_shots:
            raise ValueError(f"{event_label}: one event per shot")
        event_shots.add(event["shot_id"])
        prior = _text(event["prior_state"], f"{event_label}.prior_state")
        post = _text(event["post_state"], f"{event_label}.post_state")
        for name, state in (("prior_state", prior), ("post_state", post)):
            if len(state.split()) > STATE_MAX_WORDS:
                raise ValueError(f"{event_label}.{name} has more than {STATE_MAX_WORDS} words")
        if prior == post:
            raise ValueError(f"{event_label}: prior_state and post_state must differ")
        for state in (prior, post):
            unknown = _subject_refs(state) - set(subject_ids)
            if unknown or subject_id in _subject_refs(state):
                raise ValueError(f"{event_label}: states may only name other declared subjects")
        if not any(subject_id in shot["present_subjects"] for shot in shots[at + 1 :]):
            raise ValueError(f"{event_label}: {subject_id} never appears after the event")
        holding = state_holding_shots(case, event)
        for later in holding:
            missing = _subject_refs(post) - set(shots[later]["present_subjects"])
            if missing:
                raise ValueError(
                    f"{event_label}: post_state needs {sorted(missing)} present in "
                    f"{shots[later]['shot_id']}"
                )
            if crosses_context_change(shots, at, later):
                crosses_change = True
    if rules.require_crossing and not crosses_change:
        raise ValueError(f"{label}: no event state is carried across a cast or location change")

    tags = case["tags"]
    if tags != derive_tags(case):
        raise ValueError(f"{label}.tags must equal the derived tags {derive_tags(case)}")
    _validate_frames(case, label, rules)



def validate_long_story_pair(a: dict[str, Any], b: dict[str, Any]) -> None:
    """校验 B 对 A 的真实配对：同镜长，1–2 个注入，其他事件与骨架保留。

    ``injected_event_ids`` 指向 B 中新增或改写的事件，和继承 A 的事件分开计数。
    事件 ID 可因插入而顺延；继承关系比较实际字段，不把 ID 顺延误判成状态改动。
    这只校验结构与文字差分；原片明显违背 B、动作能在原时长完成仍需看帧记录。
    """

    for case, schema in ((a, SCRIPT_STATE_BENCH_LONG_A_SCHEMA), (b, SCRIPT_STATE_BENCH_LONG_B_SCHEMA)):
        _exact_fields(case, CASE_FIELDS | RULES[schema].case_extra_fields, case["case_id"])
        _validate_case(case, case["case_id"], schema)
    label = b["case_id"]
    if b["derived_from"] != a["case_id"]:
        raise ValueError(f"{label}: derived_from does not match paired A")
    if b["shot_count"] != a["shot_count"]:
        raise ValueError(f"{label}: A/B shot_count must match")
    for field in ("subjects", "locations", "story_prompt"):
        if b[field] != a[field]:
            raise ValueError(f"{label}: A/B {field} must match; inject a state through an event")
    injected_ids = b["injected_event_ids"]
    if (not isinstance(injected_ids, list)
            or not DEFAULT_LONG_B_INJECTIONS <= len(injected_ids) <= MAX_LONG_B_INJECTIONS
            or not all(isinstance(value, str) for value in injected_ids)
            or len(set(injected_ids)) != len(injected_ids)):
        raise ValueError(f"{label}: injected_event_ids must list 1..2 distinct B event IDs")
    by_id = {event["event_id"]: event for event in b["events"]}
    if not set(injected_ids) <= set(by_id):
        raise ValueError(f"{label}: injected_event_ids names an absent B event")
    injections = [by_id[event_id] for event_id in injected_ids]
    injected_shots = {event["shot_id"] for event in injections}
    a_events = {event["shot_id"]: event for event in a["events"]}
    b_events = {event["shot_id"]: event for event in b["events"]}

    def content(event: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in event.items() if key != "event_id"}

    for shot_id in set(a_events) | set(b_events):
        before, after = a_events.get(shot_id), b_events.get(shot_id)
        if shot_id in injected_shots:
            if before is not None and content(before) == content(after):
                raise ValueError(f"{label}.{shot_id}: declared injection is unchanged from A")
        elif before is None or after is None or content(before) != content(after):
            raise ValueError(f"{label}.{shot_id}: an inherited A event was removed or changed without injection")
    for a_shot, b_shot in zip(a["shots"], b["shots"], strict=True):
        shot_id = a_shot["shot_id"]
        if a_shot["frame_count"] != b_shot["frame_count"]:
            raise ValueError(f"{label}.{shot_id}: A/B frame_count must match")
        for key, value in a_shot.items():
            if key in {"source_span", "prompt"}:
                continue
            if b_shot[key] != value:
                raise ValueError(f"{label}.{shot_id}: A/B shot field {key} must match")
        changed = a_shot["prompt"] != b_shot["prompt"]
        if shot_id in injected_shots and not changed:
            raise ValueError(f"{label}.{shot_id}: injected event must change its shot prompt")
        if shot_id not in injected_shots and changed:
            raise ValueError(f"{label}.{shot_id}: only injected event-shot prompts may change")


def reference_prompts(benchmarks: Iterable[Mapping[str, Any]]) -> list[str]:
    """旧集里所有 story/shot prompt，供重叠检查。"""

    prompts: list[str] = []
    for benchmark in benchmarks:
        for case in benchmark["cases"]:
            prompts.append(case["story_prompt"])
            prompts.extend(shot["prompt"] for shot in case["shots"])
    return prompts


def validate_script_state_bench(
    payload: object,
    *,
    forbidden_prompts: Sequence[str],
    paired_a: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """严格校验整份清单，返回原 payload；``forbidden_prompts`` 是不得重叠的旧集 prompt。"""

    if not forbidden_prompts:
        raise ValueError("forbidden_prompts must hold the reference benchmark prompts")
    manifest = _exact_fields(payload, MANIFEST_FIELDS, "manifest")
    schema = manifest["schema_version"]
    if schema not in SCHEMAS:
        raise ValueError(f"manifest.schema_version must be one of {SCHEMAS}")
    _text(manifest["benchmark_id"], "manifest.benchmark_id")
    if manifest["evaluation_only"] is not True:
        raise ValueError("manifest.evaluation_only must be true")
    if manifest["fps"] != RULES[schema].fps:
        raise ValueError(f"manifest.fps must be {RULES[schema].fps}")
    if manifest["generation_seeds"] != list(GENERATION_SEEDS):
        raise ValueError(f"manifest.generation_seeds must be {list(GENERATION_SEEDS)}")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")

    forbidden: set[tuple[str, ...]] = set()
    for prompt in forbidden_prompts:
        forbidden |= _ngrams(prompt)
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for index, raw in enumerate(cases):
        case = _exact_fields(raw, CASE_FIELDS | RULES[schema].case_extra_fields, f"cases[{index}]")
        case_id = _text(case["case_id"], f"cases[{index}].case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id {case_id!r}")
        seen_ids.add(case_id)
        _validate_case(case, case_id, schema)
        for text in [case["story_prompt"], *(shot["prompt"] for shot in case["shots"])]:
            shared = _ngrams(text) & forbidden
            if shared:
                raise ValueError(f"{case_id}: prompt overlaps a reference benchmark on {sorted(shared)[0]}")
        for shot in case["shots"]:
            if shot["prompt"] in seen_prompts:
                raise ValueError(f"{case_id}.{shot['shot_id']}: duplicate shot prompt in this benchmark")
            seen_prompts.add(shot["prompt"])
    if RULES[schema].quotas:
        validate_quotas(cases)
    if schema == SCRIPT_STATE_BENCH_LONG_B_SCHEMA:
        if paired_a is None:
            raise ValueError("long B validation requires paired_a; equal shot timing cannot be checked alone")
        validated_a = validate_script_state_bench(paired_a, forbidden_prompts=forbidden_prompts)
        if validated_a["schema_version"] != SCRIPT_STATE_BENCH_LONG_A_SCHEMA:
            raise ValueError("paired_a must use the long A schema")
        by_id = {case["case_id"]: case for case in validated_a["cases"]}
        used_a: set[str] = set()
        for case in cases:
            source_id = case["derived_from"]
            if source_id not in by_id or source_id in used_a:
                raise ValueError(f"{case['case_id']}: derived_from must bind to one distinct paired A case")
            used_a.add(source_id)
            validate_long_story_pair(by_id[source_id], case)
    return manifest


def validate_quotas(cases: Sequence[Mapping[str, Any]]) -> None:
    """v2 全集配额：各镜界类型占比不低于 :data:`CUT_TYPE_MIN_SHARE`，任一景别不超过 :data:`MAX_SHOT_SIZE_SHARE`。"""

    cuts = Counter(shot["cut_type"] for case in cases for shot in case["shots"][1:])
    boundaries = sum(cuts.values())
    for cut_type, share in CUT_TYPE_MIN_SHARE.items():
        if cuts[cut_type] < share * boundaries:
            raise ValueError(
                f"{cut_type} covers {cuts[cut_type]}/{boundaries} boundaries, below the {share:.0%} quota"
            )
    sizes = Counter(shot["shot_size"] for case in cases for shot in case["shots"])
    shots = sum(sizes.values())
    for size, count in sizes.items():
        if count > MAX_SHOT_SIZE_SHARE * shots:
            raise ValueError(f"{size} covers {count}/{shots} shots, above the {MAX_SHOT_SIZE_SHARE:.0%} cap")


def load_script_state_bench(
    path: str | Path, *, references: Sequence[str | Path], paired_a: str | Path | None = None
) -> dict[str, Any]:
    """读盘并校验；``references`` 是不得重叠的参照集 JSON 路径（DEV32、main_table_96；v2 另加 v1）。

    本集直接按 JSON 读（长故事集不走渲染链的整条读入口，结构全由本模块校验）；参照集仍经渲染链读入口。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    loaded = [load_continuity_dev_32_benchmark(reference)[1] for reference in references]
    source_a = None if paired_a is None else json.loads(Path(paired_a).read_text(encoding="utf-8"))
    return validate_script_state_bench(
        payload, forbidden_prompts=reference_prompts(loaded), paired_a=source_a
    )


def coverage_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """覆盖统计：K 分布、标签计数、事件数、相邻镜界类型与可判的（事件, 镜）对数。"""

    cases = manifest["cases"]
    boundaries: Counter[str] = Counter()
    checks = 0
    checks_across_change = 0
    for case in cases:
        shots = case["shots"]
        for left, right in zip(shots, shots[1:]):
            same_location = left["location_id"] == right["location_id"]
            same_cast = left["present_subjects"] == right["present_subjects"]
            boundaries[
                f"{'same' if same_location else 'new'}_location/{'same' if same_cast else 'new'}_cast"
            ] += 1
        shot_index = {shot["shot_id"]: index for index, shot in enumerate(shots)}
        for event in case["events"]:
            at = shot_index[event["shot_id"]]
            for later in state_holding_shots(case, event):
                checks += 1
                if crosses_context_change(shots, at, later):
                    checks_across_change += 1
    summary: dict[str, Any] = {
        "cases": len(cases),
        "shot_count": dict(sorted(Counter(case["shot_count"] for case in cases).items())),
        "shots": sum(case["shot_count"] for case in cases),
        "frames": sum(shot["frame_count"] for case in cases for shot in case["shots"]),
        "events": sum(len(case["events"]) for case in cases),
        "tags": {tag: sum(tag in case["tags"] for case in cases) for tag in TAG_ORDER},
        "boundaries": dict(sorted(boundaries.items())),
        "state_checks": checks,
        "state_checks_across_change": checks_across_change,
    }
    rules = RULES[manifest["schema_version"]]
    if not rules.single_pass:
        segments = [min_segments([shot["frame_count"] for shot in case["shots"]]) for case in cases]
        summary["min_segments"] = dict(sorted(Counter(segments).items()))
        summary["story_seconds"] = sorted(
            round(sum(shot["frame_count"] for shot in case["shots"]) / rules.fps, 1) for case in cases
        )
    if rules.blocking:
        summary["shots_with_blocking"] = sum(
            shot["blocking"] is not None for case in cases for shot in case["shots"]
        )
    if not rules.framing:
        return summary

    def shares(counts: Counter[str], order: Sequence[str]) -> dict[str, dict[str, Any]]:
        total = sum(counts.values())
        return {key: {"count": counts[key], "share": round(counts[key] / total, 3)} for key in order}

    later = [shot for case in cases for shot in case["shots"][1:]]
    summary["cut_types"] = shares(Counter(shot["cut_type"] for shot in later), CUT_TYPES)
    summary["cut_types_by_shot_count"] = {
        count: dict(
            Counter(
                shot["cut_type"] for case in cases if case["shot_count"] == count for shot in case["shots"][1:]
            )
        )
        for count in sorted({case["shot_count"] for case in cases})
    }
    every = [shot for case in cases for shot in case["shots"]]
    summary["shot_sizes"] = shares(Counter(shot["shot_size"] for shot in every), SHOT_SIZES)
    summary["camera_angles"] = shares(Counter(shot["camera_angle"] for shot in every), CAMERA_ANGLES)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--paired-a", type=Path, help="校验长 B 时必需的 A 清单")
    args = parser.parse_args(argv)
    manifest = load_script_state_bench(args.benchmark, references=args.reference, paired_a=args.paired_a)
    print(json.dumps(coverage_summary(manifest), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
