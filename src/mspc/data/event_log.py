"""规范原子事实→event_log_v3→逐镜全文；严格校验，不修补证据。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES, SEGMENT_MAX_SHOTS
from mspc.data.state_facts import Anchor, AtomicFact, PredicateRegistry
from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET

UNKNOWN = "unknown"
MAX_REFERENCES = 4
REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs/data/state_predicates_v2.yaml"
MOVEMENTS = (
    "static", "pan left", "pan right", "tilt up", "tilt down",
    "push in", "pull out", "tracking", "handheld", "unknown",
)


#: 景别、机位角度、运镜的统一词表：日志取值 → 写进文字的词。
#: 景别与机位角度的词取自 Wan2.2 官方提示词改写指令（wan/utils/system_prompt.py::T2V_A14B_EN_SYS_PROMPT，原文：Medium shot、
#: Medium close-up shot、Wide shot、Medium wide shot、Close-up shot、Extreme close-up shot、Extreme wide shot；Over-the-shoulder shot、
#: Low angle shot、High angle shot、Dutch angle shot、Aerial shot、Overhead shot）。平视（eye level）与静止（static）是缺省，值为 None
#: 表示不写。运镜在官方 T2V 指令里没有词表，写法按 I2V 指令示例的措辞（the camera pans／moves／pushes forward／pulls back）。
SHOT_SIZE_PHRASES = {
    "extreme wide": "Extreme wide shot", "wide": "Wide shot", "medium wide": "Medium wide shot", "medium": "Medium shot",
    "medium close-up": "Medium close-up shot", "close-up": "Close-up shot", "extreme close-up": "Extreme close-up shot",
}
CAMERA_ANGLE_PHRASES = {
    "eye level": None, "high": "High angle shot", "low": "Low angle shot", "overhead": "Overhead shot",
    "over-the-shoulder": "Over-the-shoulder shot", "dutch": "Dutch angle shot", "aerial": "Aerial shot",
}
MOVEMENT_PHRASES = {
    "static": None, "pan left": "camera pans left", "pan right": "camera pans right", "tilt up": "camera tilts up",
    "tilt down": "camera tilts down", "push in": "camera pushes in", "pull out": "camera pulls back",
    "tracking": "tracking shot", "handheld": "handheld camera",
}
SHOT_FIELD_PHRASES = {"shot_size": SHOT_SIZE_PHRASES, "camera_angle": CAMERA_ANGLE_PHRASES, "movement": MOVEMENT_PHRASES}


def _vocabulary_key(value: str) -> str:
    """词表比较用的键：小写、连字符当空格、去掉词尾的 shot／angle（``High-angle shot``、``high angle``、``high`` 同一个键）。"""
    words = re.sub(r"[\s_-]+", " ", value.strip().lower()).split()
    if words and words[-1] == "shot":
        words.pop()
    if words and words[-1] in ("angle", "view") and len(words) > 1:
        words.pop()
    return " ".join(words)


#: 景别与机位角度是自由文本字段，评分前要先过词表；运镜是枚举（schema 的 Literal），不需要。
_CANONICAL = {field: {_vocabulary_key(key): key for key in SHOT_FIELD_PHRASES[field]} for field in ("shot_size", "camera_angle")}


def canonical_shot_field(field: str, value: str) -> str:
    """景别／机位角度取值过统一词表：词表里的写法（``Medium shot``、``medium shot``、``medium``）都回到日志取值 ``medium``；
    不在词表里的值只做小写与空白整理，仍交自由文本裁决。规划器评分与写手表用的都是这一个函数。"""
    table = _CANONICAL[field]
    key = _vocabulary_key(value)
    return table.get(key, " ".join(value.strip().lower().split()))


def shot_field_phrase(field: str, value: str) -> str | None:
    """日志取值 → 写进文字的词；缺省值（平视、静止）与 unknown 返回 None；词表外的值原样返回（不替它改写）。"""
    if value == UNKNOWN:
        return None
    table = SHOT_FIELD_PHRASES[field]
    canonical = canonical_shot_field(field, value) if field in _CANONICAL else value
    return table[canonical] if canonical in table else value


def registry() -> PredicateRegistry:
    return PredicateRegistry.load(REGISTRY_PATH)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Subject(StrictModel):
    subject_id: str
    kind: Literal["person", "animal", "object", "unknown"]
    appearance: str


class Location(StrictModel):
    location_id: str
    appearance: str
    anchors: list[Anchor]


class Shot(StrictModel):
    shot_id: str
    location_id: str
    present_subjects: list[str]
    visible_entities: list[str]
    screen_facts: list[AtomicFact]
    frame_count: int = Field(gt=0)
    shot_size: str
    camera_angle: str
    movement: Literal["static", "pan left", "pan right", "tilt up", "tilt down", "push in", "pull out", "tracking", "handheld", "unknown"]
    action: str


class Event(StrictModel):
    event_id: str
    shot_id: str
    before: AtomicFact | None
    after: AtomicFact | None
    on_screen: bool | Literal["unknown"]


#: 事件日志 schema 名的唯一定义处；结构不兼容的改动在这里升版（v3：删 segments/style/cut_type/参考键）。
EVENT_LOG_SCHEMA = "event_log_v3"


class EventLog(StrictModel):
    schema_version: Literal[EVENT_LOG_SCHEMA] = Field(alias="schema")
    subjects: list[Subject]
    locations: list[Location]
    initial_state: list[AtomicFact]
    shots: list[Shot]
    events: list[Event]


@dataclass(frozen=True)
class Issue:
    rule: int
    path: str
    message: str


class EventLogError(ValueError):
    def __init__(self, issues: Sequence[Issue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"rule {i.rule} {i.path}: {i.message}" for i in issues))


def fact_key(fact: AtomicFact) -> tuple[str, ...]:
    if fact.family == "relation":
        targets = fact.object_or_value if isinstance(fact.object_or_value, list) else [fact.object_or_value]
        return (fact.subject, fact.predicate_or_axis, *targets)
    return (fact.subject, fact.predicate_or_axis)


def event_key(event: Event) -> tuple[str, ...]:
    return fact_key(event.before if event.before is not None else event.after)


def fact_value(fact: AtomicFact):
    return fact.object_or_value


def same_fact(a: AtomicFact | None, b: AtomicFact | None) -> bool:
    if a is None or b is None:
        return a is b
    return (a.subject, a.family, a.predicate_or_axis, a.object_or_value) == (
        b.subject, b.family, b.predicate_or_axis, b.object_or_value)


FRAME_POSITIONS = ("first", "middle", "last")


@dataclass(frozen=True)
class FrameEvidence:
    """历史镜某一帧（first/middle/last）的证据；result_states 键同 fact_key，只含该帧可见的已知场景事实。"""
    shot_id: str
    position: Literal["first", "middle", "last"]
    accepted: bool
    pixel_available: bool
    result_states: Mapping[tuple[str, ...], AtomicFact]
    visible_subjects: frozenset[str]


@dataclass(frozen=True)
class CompiledSegment:
    span: tuple[int, int]
    reference_shot_ids: tuple[str, ...]
    reference_rows: tuple[str, ...]
    target_rows: tuple[str, ...]
    token_counts: tuple[int, ...]

    @property
    def prompts(self) -> tuple[str, ...]:
        return self.reference_rows + self.target_rows


def parse_event_log(raw: Mapping) -> EventLog:
    try:
        return EventLog.model_validate(raw)
    except ValidationError as exc:
        raise EventLogError([Issue(1, str(e["loc"]), e["msg"]) for e in exc.errors()]) from exc


def _known(fact: AtomicFact) -> bool:
    return fact.evidence != UNKNOWN and fact.object_or_value != UNKNOWN


def endpoints(fact: AtomicFact) -> set[str]:
    ids = {fact.subject}
    if fact.family == "relation":
        ids.update(fact.object_or_value if isinstance(fact.object_or_value, list) else [fact.object_or_value])
    if fact.family == "screen" and fact.predicate_or_axis == "gaze" and _known(fact):
        ids.add(str(fact.object_or_value))
    if fact.family == "screen" and fact.predicate_or_axis == "orientation" and str(fact.object_or_value).startswith("toward Subject "):
        ids.add(str(fact.object_or_value).removeprefix("toward "))
    return ids


def _bound_to_old_place(fact: AtomicFact, old: str, new: str, reg: PredicateRegistry) -> bool:
    if old == new:
        return False
    return (fact.predicate_or_axis == "environment" or any(
        x == old or x.startswith(old + "/") for x in endpoints(fact)))


def _timeline(log: EventLog):
    """shot_span 给出声明/观测坐标；镜末已知事实由保持规则延续，unknown 不覆盖。

    返回 (starts, ends, priors, carried)：carried 是本镜声明之前、只由保持规则延续来的状态。

    同键后续已知观测更新当前事实；只有 events 声明演出的变化，不从遮挡生成事件。
    镜末应用本镜全部事件；持久事实随人跨场，旧地点锚点关系一旦换场失效。
    """
    reg = registry()
    state: dict[tuple[str, ...], AtomicFact] = {}
    starts, ends, priors, carried = {}, {}, {}, {}
    previous = UNKNOWN
    for i, shot in enumerate(log.shots):
        state = {k: f.model_copy(update={"evidence":"carried"}) for k, f in state.items()
                 if f.predicate_or_axis not in reg.nonpersistent_predicates}
        for k, f in list(state.items()):
            if _bound_to_old_place(f, previous, shot.location_id, reg):
                del state[k]
        carried[shot.shot_id] = dict(state)
        for f in log.initial_state:
            if f.shot_span[0] == i and _known(f):
                state[fact_key(f)] = f
        priors[shot.shot_id] = dict(state)
        for e in log.events:
            if e.shot_id == shot.shot_id and e.on_screen is not True:
                _apply(state, e)
        starts[shot.shot_id] = dict(state)
        for e in log.events:
            if e.shot_id == shot.shot_id and e.on_screen is True:
                _apply(state, e)
        ends[shot.shot_id] = dict(state)
        previous = shot.location_id
    return starts, ends, priors, carried


def _apply(state: dict, event: Event):
    key = event_key(event)
    if event.after is None:
        state.pop(key, None)
    elif _known(event.after):
        state[key] = event.after


def snapshots(log: EventLog, *, end: bool = False) -> dict[str, dict[tuple[str, ...], AtomicFact]]:
    start, finish, _, _ = _timeline(log)
    return finish if end else start


#: 规则 5（主体外观不得含可变状态）拒收的词表，唯一定义处；评测剧本转日志（benchmark/long_story_event_log）从这里导入。
MUTABLE_STATE_WORDS = re.compile(
    r"\b(?:holds|holding|carrying|wet|opened|closed|sitting|seated|standing|walking|running|lying|operating|handling|launching)\b"
    r"|拿着|持有|穿上|湿了|站着|坐着|躺着", re.I)


def validate_event_log(log: EventLog) -> None:
    issues: list[Issue] = []
    def bad(rule, path, message):
        issues.append(Issue(rule, path, message))
    reg = registry()
    for collection, field, prefix, padded in ((log.subjects, "subject_id", "Subject ", False),
            (log.locations, "location_id", "Location ", False), (log.shots, "shot_id", "s", True),
            (log.events, "event_id", "e", False)):
        for i, item in enumerate(collection, 1):
            expected = f"{prefix}{i:02d}" if padded else f"{prefix}{i}"
            if getattr(item, field) != expected:
                bad(1, field, f"expected {expected}, got {getattr(item, field)}")
    for subject in log.subjects:
        if MUTABLE_STATE_WORDS.search(subject.appearance):
            bad(5, subject.subject_id, "mutable state in static appearance")
    subjects = {s.subject_id: s for s in log.subjects}
    locations = {l.location_id for l in log.locations}
    anchors = {a.id: (l.location_id, a) for l in log.locations for a in l.anchors}
    all_ids = set(subjects) | locations | set(anchors)
    if len(anchors) != sum(len(l.anchors) for l in log.locations):
        bad(1, "locations", "duplicate anchor")
    for l in log.locations:
        for a in l.anchors:
            if not re.fullmatch(re.escape(l.location_id) + r"/Anchor [1-9]\d*", a.id):
                bad(1, a.id, "anchor must belong to its registered location")
    shots = {s.shot_id: s for s in log.shots}
    idx = {s.shot_id: i for i, s in enumerate(log.shots)}
    for s in log.shots:
        if s.location_id not in locations | {UNKNOWN}:
            bad(1, s.shot_id, "undeclared location")
        if len(set(s.present_subjects)) != len(s.present_subjects) or not set(s.present_subjects) <= set(subjects):
            bad(1, s.shot_id, "duplicate or undeclared present subject")
        if len(set(s.visible_entities)) != len(s.visible_entities) or not set(s.visible_entities) <= all_ids:
            bad(1, s.shot_id, "duplicate or undeclared visible entity")
        if not (set(s.visible_entities) & set(subjects)) <= set(s.present_subjects):
            bad(1, s.shot_id, "visible entity must be present")
        if any(x in anchors and anchors[x][0] != s.location_id for x in s.visible_entities):
            bad(1, s.shot_id, "visible anchor belongs to another location")
    all_facts = [(f, "initial_state") for f in log.initial_state]
    all_facts += [(f, s.shot_id) for s in log.shots for f in s.screen_facts]
    all_facts += [(f, e.event_id) for e in log.events for f in (e.before, e.after) if f is not None]
    for f, path in all_facts:
        p, v = f.predicate_or_axis, f.object_or_value
        if f.subject not in all_ids:
            bad(1, path, "unregistered fact subject")
        if len(f.shot_span) != 2 or f.shot_span[0] < 0 or f.shot_span[0] > f.shot_span[1] or f.shot_span[1] >= len(shots):
            bad(1, path, "invalid shot_span")
            continue
        if v == UNKNOWN and f.evidence is not None:
            bad(2, path, "unknown observation cannot overwrite observed/carried fact")
        if f.family == "screen":
            start, stop = f.shot_span
            if start != stop or f.scope != f"screen@{start}" or f.evidence == "carried" or path not in shots or idx[path] != start:
                bad(3, path, "screen fact must belong only to its shot and cannot carry")
            if path not in shots or f.subject not in subjects or f.subject not in shots[path].visible_entities:
                bad(3, path, "screen subject must be visibly present")
            if p not in reg.screen_axes:
                bad(3, path, "unregistered screen axis")
            elif p == "bbox":
                if not isinstance(v, list) or len(v) != 4 or not all(type(x) is int and 0 <= x <= 1000 for x in v) or not (v[0] < v[2] and v[1] < v[3]):
                    bad(3, path, "invalid normalized bbox")
            elif _known(f):
                if p == "gaze":
                    if v not in all_ids or v == f.subject:
                        bad(3, path, "unregistered/self gaze target")
                elif p == "orientation" and str(v).startswith("toward Subject "):
                    if str(v).removeprefix("toward ") not in subjects or str(v).removeprefix("toward ") == f.subject:
                        bad(3, path, "unregistered/self orientation target")
                elif v not in reg.screen_axes[p]:
                    bad(3, path, "invalid screen value")
            continue
        if f.scope != "scene" or path in shots:
            bad(3, path, "persistent facts must have scene scope")
        if f.family == "state":
            if p not in reg.state_axes or not isinstance(v, str) or not v.strip() or v != v.strip():
                bad(3, path, "unregistered state axis or invalid value")
            if (p == "environment") != (f.subject in locations) or f.subject in anchors:
                bad(3, path, "state axis subject kind mismatch")
        else:
            if reg.relation_registration(p) != "registered":
                bad(3, path, "unregistered predicate cannot enter persistent facts")
            targets = v if isinstance(v, list) else [v]
            if not targets or not all(isinstance(t, str) and t in all_ids for t in targets):
                bad(1, path, "unregistered relation object")
            if f.subject in targets or len(set(map(str, targets))) != len(targets):
                bad(3, path, "self/duplicate relation")
            if (p == "between" and len(targets) != 2) or (p != "between" and len(targets) != 1):
                bad(3, path, "between needs two objects; others need one")
            if p in reg.nonpersistent_predicates and (f.evidence == "carried" or f.shot_span[0] != f.shot_span[1]):
                bad(3, path, "touch cannot persist across shots")
            for t in [f.subject, *targets]:
                if isinstance(t, str) and t in anchors and p in reg.anchor_orientation_predicates and anchors[t][1].front_definition is None:
                    bad(3, path, "anchor has no defined front")
    # Per-shot initial slot conflicts are errors; unknown observations are not negative assertions.
    for i in range(len(log.shots)):
        known = {}
        for f in log.initial_state:
            if len(f.shot_span) == 2 and f.shot_span[0] <= i <= f.shot_span[1] and _known(f):
                k = fact_key(f)
                if k in known and not same_fact(known[k], f):
                    bad(4, f"initial_state@{i}", "conflicting same-key facts")
                known[k] = f
    seen = set()
    for e in log.events:
        if e.shot_id not in shots or (e.before is None and e.after is None):
            bad(1, e.event_id, "event requires a known shot and at least one endpoint")
            continue
        if any(f.family == "screen" or not _known(f) for f in (e.before, e.after) if f is not None):
            bad(2, e.event_id, "event requires confirmed scene facts; unknown is not deletion")
        if e.before is not None and e.after is not None and fact_key(e.before) != fact_key(e.after):
            bad(2, e.event_id, "replacement must keep one fact key; transfer uses delete/add")
        if same_fact(e.before, e.after):
            bad(2, e.event_id, "no fact change")
        k = (e.shot_id, event_key(e))
        if k in seen:
            bad(4, e.event_id, "duplicate same-shot fact-key event")
        seen.add(k)
        if e.on_screen is True:
            required = set().union(*(endpoints(f) for f in (e.before, e.after) if f is not None))
            if not required <= set(shots[e.shot_id].visible_entities):
                bad(3, e.event_id, "both event endpoints must be visibly verifiable")
        for f in (e.before, e.after):
            if f is not None and len(f.shot_span) == 2 and not f.shot_span[0] <= idx[e.shot_id] <= f.shot_span[1]:
                bad(2, e.event_id, "event fact span excludes event shot")
    if issues:
        raise EventLogError(issues)
    starts, ends, priors, _ = _timeline(log)
    for e in log.events:
        if not same_fact(priors[e.shot_id].get(event_key(e)), e.before):
            bad(2, e.event_id, "before differs from known opening fact")
    for sid, state in [*starts.items(), *ends.items()]:
        graph: dict[str, set[str]] = {}
        for f in state.values():
            if f.family == "relation" and f.predicate_or_axis in reg.containment_predicates:
                graph.setdefault(f.subject, set()).add(str(f.object_or_value))
        def visit(node, path):
            if node in path:
                bad(3, sid, "containment cycle")
                return
            for t in graph.get(node, ()):
                visit(t, path | {node})
        for node in graph:
            visit(node, set())
    if issues:
        raise EventLogError(issues)


def _name(log: EventLog, entity: str) -> str:
    for loc in log.locations:
        for a in loc.anchors:
            if a.id == entity:
                return f"the {a.description} ({a.id})"
    return entity


def fact_clause(log: EventLog, f: AtomicFact) -> str:
    s, p, v = _name(log, f.subject), f.predicate_or_axis, f.object_or_value
    if not _known(f):
        return ""
    if f.family == "state":
        return f"{s} wears {v}" if p == "costume" else f"{s} is {v}"
    if f.family == "relation":
        targets = v if isinstance(v, list) else [v]
        names = [_name(log, t) for t in targets]
        verb = {"in": "is inside", "on": "is on", "under": "is under", "attached to": "is attached to",
                "in front of": "is in front of", "behind": "is behind", "next to": "is next to",
                "against": "is against", "at": "is at", "between": "is between"}.get(p, p)
        return f"{s} {verb} {' and '.join(names)}"
    if p == "bbox":
        return ""
    if p == "horizontal":
        return f"{s} is in the center" if v == "center" else f"{s} is on the {v}"
    if p == "depth":
        return f"{s} is in the {v}"
    if p == "gaze":
        return f"{s} looks at {_name(log, str(v))}"
    facing = {"toward camera": "the camera", "away": "away from the camera", "screen-left": "screen left", "screen-right": "screen right"}.get(str(v), str(v))
    return f"{s} is facing {facing}"


def visible_fact(shot: Shot, f: AtomicFact) -> bool:
    visible = set(shot.visible_entities)
    if f.family == "state" and f.predicate_or_axis == "environment":
        return f.subject == shot.location_id
    return endpoints(f) <= visible


def state_clause(log: EventLog, shot: Shot, state: Mapping[tuple[str, ...], AtomicFact]) -> list[str]:
    return [text for _, f in sorted(state.items()) if visible_fact(shot, f) and (text := fact_clause(log, f))]


def _negative_clause(log: EventLog, fact: AtomicFact, *, first: bool = False) -> str:
    clause = fact_clause(log, fact)
    s = _name(log, fact.subject)
    rest = clause.removeprefix(s + " ")
    if first:
        if rest.startswith("is "):
            return f"{s} is not {rest.removeprefix('is ')}"
        if rest.startswith("stands "):
            return f"{s} does not stand {rest.removeprefix('stands ')}"
        verbs = {"holds":"hold", "wears":"wear", "carries":"carry", "touches":"touch"}
        verb, tail = rest.split(" ", 1)
        return f"{s} does not {verbs[verb]} {tail}"
    if rest.startswith("is "):
        return f"{s} is no longer {rest.removeprefix('is ')}"
    if rest.startswith("stands "):
        return f"{s} no longer stands {rest.removeprefix('stands ')}"
    return f"{s} no longer {rest}"


def state_text(log: EventLog, shot: Shot, start: Mapping, end: Mapping, *, transition: bool) -> str:
    before, after = state_clause(log, shot, start), state_clause(log, shot, end)
    parts = before
    if transition:
        cb, ca = [x for x in before if x not in after], [x for x in after if x not in before]
        parts = [x for x in before if x in after]
        additions = [e.after for e in log.events if e.shot_id == shot.shot_id and e.on_screen is True and e.before is None]
        cb += [_negative_clause(log, f, first=True) for f in additions if visible_fact(shot, f)]
        if cb:
            parts.append(f"{' and '.join(cb)} at first")
        if ca:
            parts.append(f"by the end of the shot {' and '.join(ca)}")
        # Explicit relation deletion is witnessed, not inferred from an absent endpoint.
        deleted = [e.before for e in log.events if e.shot_id == shot.shot_id and e.on_screen is True and e.after is None]
        negatives = [_negative_clause(log, f) for f in deleted if visible_fact(shot, f)]
        if negatives:
            parts.append(f"by the end of the shot {' and '.join(negatives)}")
    return f"State: {'; '.join(parts)}." if parts else ""


_HORIZONTAL_ORDER = {"left": 0, "center": 1, "right": 2}
#: ``space_text`` 写出的开头句的格式标记（唯一定义处）；训练的快照层丢弃时与 State 句一起删。句内不会出现
#: 「句号＋空白」（外观短语取第一句，``appearance_phrase``），所以句尾就是第一个后面接空格或结尾的句号。
SPACE_SENTENCE = re.compile(
    r"(?:On the (?:left|right) side of the frame|In the center of the frame).*?\.(?: |$)", re.S)
_ARTICLES = ("a ", "an ", "the ")
_HEAD_NOUN_END = re.compile(r",| with | in | wearing | of | on ")


def appearance_phrase(attributes: str) -> str:
    """剧本外观描述 -> 句中指人的外观短语：取第一句、去句末标点、首词小写；没有冠词的按首字母补 ``a``／``an``。
    用的就是 ``Design`` 句里那段描述，不另行改写。S-first 臂（跨镜布局判别）与编译器的空间句共用这一处。"""
    first = re.split(r"(?<=[.!?])\s+", attributes.strip())[0].rstrip(".!? ").strip()
    if not first:
        raise ValueError("subject attributes are empty")
    phrase = first[0].lower() + first[1:]
    if phrase.startswith(_ARTICLES):
        return phrase
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def _head_noun(appearance: str) -> str:
    """外观描述的头名词：第一个修饰成分（逗号、with、in、wearing、of、on）之前的最后一个词；不是英文词就没有。"""
    words = _HEAD_NOUN_END.split(appearance.strip().rstrip("."), maxsplit=1)[0].split()
    return words[-1].lower() if words and re.fullmatch(r"[A-Za-z][A-Za-z-]*", words[-1]) else ""


def space_text(log: EventLog, shot: Shot) -> str:
    """每镜目标行开头的空间句：按画面从左到右，每个人物或动物用其外观描述的
    整句（:func:`appearance_phrase`，与 S-first 臂逐字相同）指人，写左右；该人有前后（只在两人都绑定时才有）就连同
    前后，面朝镜头就补一句。

    措辞同 S-first 臂的首句（``On the left side of the frame, in the background, is a man with short dark hair,
    wearing a brown jacket; …``）。10-10 先用过空间抽取链的短属性词指人，逐镜服从率 0.741，S-first 的整句外观描述
    0.940（跨镜布局判别 S-compiled 臂），所以改回整句。外观未知，或同一句里两人的外观短语相同（读者分不出谁是谁）
    时退回 ``Subject k`` 加外观头名词。物体的屏幕事实与其余屏幕轴（朝向某人、视线、侧向）不进文字。
    """
    subjects = {s.subject_id: s for s in log.subjects}
    horizontal: dict[str, str] = {}
    for fact in shot.screen_facts:
        if fact.predicate_or_axis == "horizontal" and _known(fact) and subjects[fact.subject].kind in ("person", "animal"):
            horizontal.setdefault(fact.subject, str(fact.object_or_value))
    if not horizontal:
        return ""
    value = {(f.subject, f.predicate_or_axis): str(f.object_or_value) for f in shot.screen_facts if _known(f)}
    order = {s.subject_id: i for i, s in enumerate(log.subjects)}
    phrases = {sid: appearance_phrase(subjects[sid].appearance) for sid in horizontal if subjects[sid].appearance != UNKNOWN}
    clauses = []
    for sid in sorted(horizontal, key=lambda sid: (_HORIZONTAL_ORDER[horizontal[sid]], order[sid])):
        phrase = phrases.get(sid)
        if phrase is not None and list(phrases.values()).count(phrase) == 1:
            who = phrase
        else:
            noun = "" if subjects[sid].appearance == UNKNOWN else _head_noun(subjects[sid].appearance)
            who = f"{sid}, the {noun}" if noun else sid
        where = "in the center of the frame" if horizontal[sid] == "center" else f"on the {horizontal[sid]} side of the frame"
        depth = f", in the {value[(sid, 'depth')]}," if (sid, "depth") in value else ""
        facing = ", facing the camera" if value.get((sid, "orientation")) == "toward camera" else ""
        clauses.append(f"{where}{depth} is {who}{facing}")
    text = "; ".join(clauses) + "."
    return text[0].upper() + text[1:]


def design_text(log: EventLog, shot: Shot) -> str:
    entries = [f"{s.subject_id}: {s.appearance}" for s in log.subjects
               if s.subject_id in shot.visible_entities and s.appearance != UNKNOWN]
    entries += [f"{l.location_id}: {l.appearance}" for l in log.locations
                if l.location_id == shot.location_id and l.appearance != UNKNOWN]
    return f"Design: {'; '.join(entries)}." if entries else ""


def _result_shot_index(log: EventLog, event: Event) -> int:
    """结果镜是变化后状态第一次可见的那一镜；看不见就只绑事件镜。"""
    start = next(i for i, shot in enumerate(log.shots) if shot.shot_id == event.shot_id)
    if event.after is None and event.before is None:
        return start
    for index in range(start, len(log.shots)):
        shot = log.shots[index]
        if event.after is not None:
            visible = visible_fact(shot, event.after)
        else:
            visible = endpoints(event.before) <= set(shot.visible_entities)
        if visible:
            return index
    return start


def atomic_units(log: EventLog) -> tuple[tuple[int, int], ...]:
    """闭区间。事件镜到结果镜不能从中间切开；其余镜各自成块。"""
    count = len(log.shots)
    if count == 0:
        return ()
    shot_index = {shot.shot_id: i for i, shot in enumerate(log.shots)}
    cover = [-1] * count
    for event in log.events:
        start = shot_index[event.shot_id]
        end = _result_shot_index(log, event)
        for index in range(start, end + 1):
            cover[index] = max(cover[index], end)
    units: list[tuple[int, int]] = []
    index = 0
    while index < count:
        if cover[index] >= index:
            end = cover[index]
            cursor = index
            while cursor <= end:
                if cover[cursor] > end:
                    end = cover[cursor]
                cursor += 1
            units.append((index, end))
            index = end + 1
        else:
            units.append((index, index))
            index += 1
    return tuple(units)


def dependency_segment_spans(log: EventLog) -> tuple[tuple[int, int], ...]:
    """半开区间。事件镜到结果镜不拆；每段不超过帧数和镜数上限，按顺序贪心装。"""
    units = atomic_units(log)
    for start, end in units:
        frames = sum(log.shots[i].frame_count for i in range(start, end + 1))
        label = f"{log.shots[start].shot_id}:{log.shots[end].shot_id}"
        if frames > SEGMENT_MAX_FRAMES or end - start + 1 > SEGMENT_MAX_SHOTS:
            raise EventLogError([Issue(6, label, "依赖段无法装入")])
    spans: list[tuple[int, int]] = []
    current: list[int] | None = None
    for start, end in units:
        frames = sum(log.shots[i].frame_count for i in range(start, end + 1))
        shots = end - start + 1
        if current is None:
            current = [start, end + 1, frames, shots]
            continue
        cur_start, cur_end, cur_frames, cur_shots = current
        if cur_frames + frames <= SEGMENT_MAX_FRAMES and cur_shots + shots <= SEGMENT_MAX_SHOTS:
            current = [cur_start, end + 1, cur_frames + frames, cur_shots + shots]
        else:
            spans.append((cur_start, cur_end))
            current = [start, end + 1, frames, shots]
    if current is not None:
        spans.append((current[0], current[1]))
    return tuple(spans)


def frame_evidence(log: EventLog, shot_id: str, position: str, visible: Sequence[str]) -> FrameEvidence:
    """某镜 first/middle/last 帧的证据：first 取起点快照，last 取终点快照，middle 只取首尾相同的事实。

    只收该帧端点全部可见的已知场景事实；screen 类与 unknown 不收。visible 是该帧可见实体名单。
    """
    if position not in FRAME_POSITIONS:
        raise ValueError(f"frame position must be one of {FRAME_POSITIONS}; got {position!r}")
    shots = {shot.shot_id: shot for shot in log.shots}
    if shot_id not in shots:
        raise ValueError(f"unknown shot {shot_id}")
    seen = frozenset(visible)
    ids = {s.subject_id for s in log.subjects} | {l.location_id for l in log.locations} | {a.id for l in log.locations for a in l.anchors}
    if not seen <= ids:
        raise ValueError(f"{shot_id} {position}: visible entities must be registered IDs")
    starts, ends = snapshots(log), snapshots(log, end=True)
    start, end = starts[shot_id], ends[shot_id]
    if position == "first":
        facts = start
    elif position == "last":
        facts = end
    else:
        facts = {key: fact for key, fact in start.items() if same_fact(end.get(key), fact)}
    frame = shots[shot_id].model_copy(update={"visible_entities": sorted(seen)})
    states = {key: fact for key, fact in facts.items()
              if fact.family != "screen" and _known(fact) and visible_fact(frame, fact)}
    return FrameEvidence(shot_id, position, True, True, states, seen)


def _validated_reference_sources(log: EventLog, span: tuple[int, int], references: Sequence[FrameEvidence]) -> list[Shot]:
    """参考帧的校验（现行与第二版编译共用）：张数、必须是本段之前各镜的不同已接受帧、可见实体已登记、事实是带精确键的可见场景事实。
    返回每张参考帧对应的镜（可见实体换成该帧的可见名单）。"""
    start = span[0]
    if len(references) > MAX_REFERENCES:
        raise EventLogError([Issue(6, f"{span}", f"at most {MAX_REFERENCES} references")])
    ids = {s.subject_id for s in log.subjects} | {l.location_id for l in log.locations} | {a.id for l in log.locations for a in l.anchors}
    shots = {s.shot_id: s for s in log.shots}
    indices = {s.shot_id: i for i, s in enumerate(log.shots)}
    seen_frames = set()
    sources = []
    for frame in references:
        if (frame.shot_id not in indices or indices[frame.shot_id] >= start or frame.position not in FRAME_POSITIONS
                or (frame.shot_id, frame.position) in seen_frames
                or frame.accepted is not True or frame.pixel_available is not True):
            raise EventLogError([Issue(6, frame.shot_id, "reference must be a distinct accepted frame of an earlier shot")])
        seen_frames.add((frame.shot_id, frame.position))
        if not frame.visible_subjects <= ids:
            raise EventLogError([Issue(6, frame.shot_id, "unregistered visible frame entity")])
        source = shots[frame.shot_id].model_copy(update={"visible_entities": sorted(frame.visible_subjects)})
        for key, fact in frame.result_states.items():
            if (not isinstance(fact, AtomicFact) or key != fact_key(fact) or fact.family == "screen"
                    or not _known(fact) or not visible_fact(source, fact)):
                raise EventLogError([Issue(6, frame.shot_id, "frame facts must be known visible scene facts with exact keys")])
        sources.append(source)
    return sources


def compile_event_log(log: EventLog, *, span: tuple[int, int], references: Sequence[FrameEvidence],
                      token_counter: Callable[[str], int]) -> CompiledSegment:
    """编译一段：span 为半开镜区间（由 dependency_segment_spans 给出），references 由调用方选定。

    编译器不挑帧，只描述传入的参考帧：参考行写该帧当时的 Design 与 State，
    目标行写现在的文字和逐参考说明。
    """
    validate_event_log(log)
    start, end = span
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(log.shots):
        raise ValueError(f"span must be a nonempty half-open shot range; got {span!r}")
    sources = _validated_reference_sources(log, span, references)
    starts, ends = snapshots(log), snapshots(log, end=True)
    number = {s.shot_id: i + 1 for i, s in enumerate(log.shots)}
    refs, targets = [], []
    for n, (frame, source) in enumerate(zip(references, sources), 1):
        # 参考行只写该帧当时的样子，不写计划里现在的状态。
        refs.append(" ".join(x for x in ("Reference still from an earlier shot of this story; not the current shot.",
            f"Reference {n}, the {frame.position} frame of shot {number[frame.shot_id]}.", design_text(log, source),
            state_text(log, source, frame.result_states, frame.result_states, transition=False)) if x))
    for target in log.shots[start:end]:
        target_id = target.shot_id
        prefix = ", ".join(x for x in (target.shot_size, target.camera_angle, target.movement) if x != UNKNOWN)
        parts = [space_text(log, target), prefix + "." if prefix else "", target.action if target.action != UNKNOWN else "",
            design_text(log, target),
            state_text(log, target, starts[target_id], ends[target_id], transition=any(e.shot_id == target_id and e.on_screen is True for e in log.events))]
        for n, frame in enumerate(references, 1):
            borrowed = [s.subject_id for s in log.subjects if s.subject_id in target.visible_entities
                        and s.subject_id in frame.visible_subjects and s.appearance != UNKNOWN]
            if not borrowed:
                continue
            current = [f for f in starts[target_id].values() if endpoints(f) & set(borrowed) and visible_fact(target, f)]
            changes = [fact_clause(log, f) for f in current if not same_fact(frame.result_states.get(fact_key(f)), f)]
            # No blanket 'nothing changed': unobserved/expired/hidden facts cannot warrant it.
            since = f"; current facts from text: {' and '.join(changes)}" if changes else ""
            parts.append(f"Reference {n} is the {frame.position} frame of shot {number[frame.shot_id]}; "
                f"take {' and '.join(f'{s} appearance' for s in borrowed)} from it; possession, location and object condition follow the current text{since}.")
        targets.append(" ".join(x for x in parts if x))
    counts = tuple(token_counter(t) for t in (*refs, *targets))
    errors = []
    for i, count in enumerate(counts):
        if type(count) is not int or count < 0:
            raise ValueError("token_counter must return a nonnegative int")
        if count > UMT5_PROMPT_TOKEN_BUDGET:
            errors.append(Issue(7, f"{span}.prompts[{i}]", f"{count} tokens exceeds 512"))
    if errors:
        raise EventLogError(errors)
    return CompiledSegment((start, end), tuple(f.shot_id for f in references), tuple(refs), tuple(targets), counts)


# ---------------------------------------------------------------------------
# 规划器监督：声明表、起因/可推出、实体对齐、逐条比对与第 1 档合并
# ---------------------------------------------------------------------------

ENTRY_KINDS = ("appearance", "shot", "presence", "fact", "screen", "event")
SHOT_FIELDS = ("location_id", "shot_size", "camera_angle", "movement", "action")
# 剧本不写的镜字段缺省值：平视、静止不写，规划器按缺省补；地点与上一镜相同也不写。
# 取值来自地板 v10 的 156 窗：eye level 377/440 镜、static 319/440 镜、非首镜里与上一镜同地点 253/284。
SHOT_FIELD_DEFAULTS = {"camera_angle": "eye level", "movement": "static"}
LABEL_CLASSES = ("stated", "derivable", "spatial", "free")
ALIGN_MIN_SCORE = 0.2


@dataclass(frozen=True)
class FactEntry:
    """声明表的一条：role 为 cause（起因，写手要写）、derivable（可推出，写手不写）或 spatial（空间类，不进写手表）。

    编号从 1 起：写手表条目（cause/derivable）按镜序在前、连续编号；空间条目排在其后，不进写手表。

    kind=appearance：target 为实体编号；shot：target 为镜字段名；presence：target 为在场主体；
    fact：本镜 initial_state 已知声明；screen：本镜已知屏幕事实，即“空间类”（位置、朝向、视线；
    bbox 不编译，不进表）；event：事件。
    """
    number: int
    shot_id: str
    kind: Literal["appearance", "shot", "presence", "fact", "screen", "event"]
    role: Literal["cause", "derivable", "spatial"]
    text: str
    target: str = ""
    fact: AtomicFact | None = None
    event: Event | None = None


def entity_ids(log: EventLog) -> list[str]:
    return ([s.subject_id for s in log.subjects] + [l.location_id for l in log.locations]
            + [a.id for l in log.locations for a in l.anchors])


def _first_appearance(log: EventLog) -> dict[str, int]:
    """实体首次出现的镜：在场、可见或作为本镜地点；从未出镜的实体取首次被事实或事件引用的镜，再无则 0。"""
    first: dict[str, int] = {}
    for i, shot in enumerate(log.shots):
        for entity in (*shot.present_subjects, *shot.visible_entities, shot.location_id):
            first.setdefault(entity, i)
    index = {s.shot_id: i for i, s in enumerate(log.shots)}
    referenced = [(f.shot_span[0], f) for f in log.initial_state]
    referenced += [(index[e.shot_id], f) for e in log.events for f in (e.before, e.after) if f is not None]
    for i, fact in sorted(referenced, key=lambda item: item[0]):
        for entity in endpoints(fact):
            first.setdefault(entity, i)
    return {entity: first.get(entity, 0) for entity in entity_ids(log)}


# 写手表不写空间：body 状态值里描述画面方位、朝向镜头或画面左右的短语从表文字里剥掉，方位仍由空间类计分。
# 规则取自地板 v10 的 166 窗真实 body 取值（520 种写法，73 种含这类短语）；“to the left of Subject 3”这类相对主体的位置、
# “out of frame”这类出入画、“his right fist”这类身体部位不剥。
_SCREEN_SIDE = r"(?:left|right|center|centre|middle|top|bottom)"
SCREEN_PHRASES = re.compile("|".join([
    rf"\b(?:on|at|to|in|from) the (?:far )?{_SCREEN_SIDE}(?: (?:edge|side|corner))?(?: of the (?:frame|screen|row))?"
    r"(?![\w-])(?!\s+(?:(?:edge|side|corner)\s+)?of\b)",
    r"\bat (?:the )?(?:center|centre)\b(?!\s+of\b)",
    r"\bat the edge of the frame\b",
    r"\bin the (?:foreground|background)\b",
    r"\b(?:with )?(?:(?:his|her|its|their) )?back (?:partly )?(?:turned )?to(?:wards?)? (?:the )?camera\b",
    r"\bfacing (?:towards? )?(?:the )?camera\b",
    r"\b(?:towards?|to|at|past|close to|closer to) (?:the )?camera\b",
    r"\bfacing (?:screen[- ])?(?:left|right)\b",
    r"\b(?:(?:towards?|to) )?screen[- ](?:left|right)\b",
    r"\bin (?:side )?profile\b",
]), re.IGNORECASE)


def screen_free(value: str) -> str:
    """剥掉画面方位与朝向短语，再收拾悬空的逗号和连接词；整句都是方位时返回空串。"""
    text = re.sub(r"\s+", " ", SCREEN_PHRASES.sub(" ", value))
    text = re.sub(r"\s+([,;])", r"\1", text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\s+(?:and|with|while)$", "", re.sub(r"^(?:and|with|while)\s+", "", text.strip(" ,;"), flags=re.I), flags=re.I)
    return text


def _table_fact(fact: AtomicFact) -> AtomicFact | None:
    """写手表里的这条声明：body 值剥掉画面方位；剥空了（只说方位）返回 None，由空间类计分。"""
    if fact.family != "state" or fact.predicate_or_axis != "body" or not isinstance(fact.object_or_value, str):
        return fact
    value = screen_free(fact.object_or_value)
    return fact.model_copy(update={"object_or_value": value}) if value else None


def _table_event(event: Event) -> tuple[AtomicFact | None, AtomicFact | None] | None:
    """事件进写手表的前后态：body 值剥掉画面方位；一侧剥空或剥完只剩方位变化时返回 None，归空间类。"""
    sides = tuple(None if f is None else _table_fact(f) for f in (event.before, event.after))
    if any(f is not None and t is None for f, t in zip((event.before, event.after), sides)):
        return None
    if None not in sides and sides[0].object_or_value == sides[1].object_or_value and not same_fact(event.before, event.after):
        return None
    return sides


def _event_text(log: EventLog, event: Event, before: AtomicFact | None, after: AtomicFact | None) -> str:
    when = {True: "on screen", False: "off screen", UNKNOWN: "visibility unknown"}[event.on_screen]
    before_text = fact_clause(log, before) if before is not None else "(absent)"
    after_text = fact_clause(log, after) if after is not None else "(no longer true)"
    return f"event ({when}): {before_text} -> {after_text}"


def fact_roles(log: EventLog) -> tuple[FactEntry, ...]:
    """逐镜列出全部声明并分起因/可推出。

    起因：值与上一镜按保持规则延续来的值不同（含此前没有），或该事实的主体在本镜首次出现。
    可推出：只是复述延续值的 scene 声明。外观只在主体首次出场那镜登记为起因，之后外观不变不再登记
    （不进写手表、不计分）。事件、在场、景别、动作没有延续值，逐镜档每镜都算起因。
    机位角度与运镜取缺省值（SHOT_FIELD_DEFAULTS：平视、静止）时算可推出；地点与上一镜相同算可推出
    （贴近评测时规划器拿到的逐镜剧本）。
    空间事实（kind=screen：位置、朝向、视线）是“空间类”：剧本输入不写这类细节，
    所以不进写手表，role 为 spatial、排在编号末尾；由规划器从剧本推出，在目标日志里单列计分，不并入写明/需推出主分。
    body 状态值里的画面方位短语（on the left、in the foreground、facing the camera 等，见 SCREEN_PHRASES）在表文字里剥掉；
    剥空的声明、以及剥掉后只剩方位变化的事件，同样归空间类（与「剧本不写空间」一致，方位仍在空间类计分）。
    """
    validate_event_log(log)
    _, _, _, carried = _timeline(log)
    first = _first_appearance(log)
    kinds = {s.subject_id: s.kind for s in log.subjects}
    entries: list[FactEntry] = []

    def add(shot_id, kind, role, text, **extra):
        entries.append(FactEntry(0, shot_id, kind, role, script_text(log, text), **extra))

    for i, shot in enumerate(log.shots):
        sid = shot.shot_id
        for subject in log.subjects:
            if first[subject.subject_id] == i and subject.appearance != UNKNOWN:
                add(sid, "appearance", "cause", f"{subject.subject_id} ({kinds[subject.subject_id]}) looks like: {subject.appearance}",
                    target=subject.subject_id)
        for location in log.locations:
            if first[location.location_id] == i and location.appearance != UNKNOWN:
                add(sid, "appearance", "cause", f"{location.location_id} is: {location.appearance}", target=location.location_id)
        for name in SHOT_FIELDS:
            value = getattr(shot, name)
            if value != UNKNOWN:
                label = {"location_id": "takes place in", "shot_size": "shot size:", "camera_angle": "camera angle:",
                         "movement": "camera movement:", "action": "action:"}[name]
                unchanged_place = name == "location_id" and i > 0 and value == log.shots[i - 1].location_id
                role = "derivable" if SHOT_FIELD_DEFAULTS.get(name) == value or unchanged_place else "cause"
                # 景别、机位、运镜写表里的词表词（缺省的平视、静止没有词，写日志取值，标 DO-NOT-WRITE）。
                shown = (shot_field_phrase(name, value) or value) if name in SHOT_FIELD_PHRASES else value
                add(sid, "shot", role, f"{label} {shown}", target=name)
        for subject in shot.present_subjects:
            add(sid, "presence", "cause", f"{subject} is in the shot", target=subject)
        seen: set[tuple] = set()
        for fact in log.initial_state:
            if fact.shot_span[0] != i or not _known(fact):
                continue
            key = fact_key(fact)
            if (key, str(fact.object_or_value)) in seen:
                continue
            seen.add((key, str(fact.object_or_value)))
            restated = same_fact(carried[sid].get(key), fact) and first[fact.subject] != i
            table = _table_fact(fact)
            if table is None:
                add(sid, "fact", "spatial", fact_clause(log, fact), fact=fact)
            else:
                add(sid, "fact", "derivable" if restated else "cause", fact_clause(log, table), fact=fact)
        for fact in shot.screen_facts:
            if _known(fact) and (text := fact_clause(log, fact)):
                add(sid, "screen", "spatial", text, fact=fact)
        for event in log.events:
            if event.shot_id == sid:
                sides = _table_event(event)
                add(sid, "event", "spatial" if sides is None else "cause",
                    _event_text(log, event, *(sides or (event.before, event.after))), event=event)
    ordered = [e for e in entries if e.role != "spatial"] + [e for e in entries if e.role == "spatial"]
    return tuple(replace(e, number=n) for n, e in enumerate(ordered, 1))


def table_entries(entries: Sequence[FactEntry]) -> tuple[FactEntry, ...]:
    """写手表与核对表只含起因与可推出条目；空间条目不进表。"""
    return tuple(e for e in entries if e.role != "spatial")


def _require_numbers(entries: Sequence[FactEntry], numbers, label: str) -> set[int]:
    known = {e.number for e in entries}
    chosen = set(numbers)
    if any(type(n) is not int for n in chosen) or not chosen <= known:
        raise ValueError(f"{label} must be entry numbers of this log's fact table")
    return chosen


def derivable(log: EventLog, stated) -> tuple[dict, dict]:
    """只保留已写明的起因（scene 声明与事件）重跑时间线，返回各镜 (起点快照, 终点快照)。"""
    entries = fact_roles(log)
    chosen = _require_numbers(entries, stated, "stated")
    keep_facts = {id(e.fact) for e in entries if e.number in chosen and e.kind == "fact" and e.role == "cause"}
    keep_events = {id(e.event) for e in entries if e.number in chosen and e.kind == "event"}
    reduced = log.model_copy(update={
        "initial_state": [f for f in log.initial_state if id(f) in keep_facts],
        "events": [e for e in log.events if id(e) in keep_events]})
    starts, ends, _, _ = _timeline(reduced)
    return starts, ends


def entry_labels(log: EventLog, written) -> dict[str, list[int]]:
    """四类互斥且覆盖全部条目（空间类单列、不并入主分）。

    写明＝核对确认写到的条目；
    需推出＝只由写明的起因重跑时间线后，与金标本镜起点快照相等且本镜可见的 scene 声明；
    以及没写的缺省机位角度、缺省运镜，和从已知地点延续来的同地点镜（前面最近一次地点变化写明了才算）；
    空间＝全部空间条目（kind=screen，以及 body 值只说画面方位的声明与事件）：写手表里没有它们，由规划器从剧本推出，单列计分；
    自由＝其余条目。
    written 只能取写手表条目。
    """
    entries = fact_roles(log)
    stated = _require_numbers(table_entries(entries), written, "written")
    reduced, _ = derivable(log, stated)
    gold = snapshots(log)
    shots = {s.shot_id: s for s in log.shots}
    derived = []
    place_known = False  # 读者此刻是否知道当前地点：写明了地点，或由已知地点延续而来
    for e in entries:
        if e.kind == "shot" and e.target == "location_id":
            place_known = e.number in stated or (e.role == "derivable" and place_known)
            if e.number not in stated and place_known:
                derived.append(e.number)
            continue
        if e.kind == "shot" and e.role == "derivable" and e.number not in stated:
            derived.append(e.number)  # 平视、静止：没写就按缺省
            continue
        if e.number in stated or e.kind != "fact" or e.role == "spatial":
            continue
        key = fact_key(e.fact)
        value = gold[e.shot_id].get(key)
        if value is not None and visible_fact(shots[e.shot_id], value) and same_fact(reduced[e.shot_id].get(key), value):
            derived.append(e.number)
    spatial = [e.number for e in entries if e.role == "spatial"]
    free = [e.number for e in entries if e.number not in stated and e.number not in set(derived) and e.role != "spatial"]
    return {"stated": sorted(stated), "derivable": derived, "spatial": spatial, "free": free}


def require_labels(entries: Sequence[FactEntry], labels: Mapping[str, Sequence[int]]) -> dict[str, set[int]]:
    if not isinstance(labels, Mapping) or set(labels) != set(LABEL_CLASSES):
        raise ValueError(f"labels must contain exactly {LABEL_CLASSES}")
    sets = {name: _require_numbers(entries, labels[name], name) for name in LABEL_CLASSES}
    if sum(map(len, sets.values())) != len(entries) or set().union(*sets.values()) != {e.number for e in entries}:
        raise ValueError("labels must partition the whole fact table")
    return sets


# --- 实体对齐 ---------------------------------------------------------------

_SCRIPT_ID = re.compile(r"\b(?:Subject|Character|Location) [1-9]\d*(?:/Anchor [1-9]\d*)?")
SCRIPT_CHARACTER_KINDS = ("person", "animal")


def script_name(subject_id: str, kind: str) -> str:
    """剧本里的称呼：人和动物写 ``Character N``，N 就是日志里 ``Subject N`` 的 N；物体仍写 ``Subject N``。"""
    return "Character " + subject_id.removeprefix("Subject ") if kind in SCRIPT_CHARACTER_KINDS else subject_id


def script_text(log: EventLog, text: str) -> str:
    """把一段用日志编号写的文字里人和动物的 ``Subject N`` 换成剧本里的 ``Character N``。"""
    kinds = {s.subject_id: s.kind for s in log.subjects}
    return re.sub(r"\bSubject [1-9]\d*", lambda m: script_name(m.group(0), kinds.get(m.group(0), "object")), text)


def script_named_ids(script: str) -> set[str]:
    """剧本里逐字点名的 Subject N / Character N / Location N / Location N/Anchor M，一律以日志里的编号返回
    （``Character N`` 就是 ``Subject N``）。"""
    return {"Subject " + found.removeprefix("Character ") if found.startswith("Character ") else found
            for found in _SCRIPT_ID.findall(script)}


def _entity_class(log: EventLog, entity: str) -> tuple[str, ...]:
    for s in log.subjects:
        if s.subject_id == entity:
            return ("subject", s.kind)
    for l in log.locations:
        if l.location_id == entity:
            return ("location",)
        if any(a.id == entity for a in l.anchors):
            return ("anchor",)
    raise ValueError(f"unregistered entity {entity}")


def _entity_text(log: EventLog, entity: str) -> str:
    for s in log.subjects:
        if s.subject_id == entity:
            return s.appearance
    for l in log.locations:
        if l.location_id == entity:
            return l.appearance
        for a in l.anchors:
            if a.id == entity:
                return a.description
    raise ValueError(f"unregistered entity {entity}")


def shot_frame_ranges(log: EventLog) -> list[range]:
    out, start = [], 0
    for shot in log.shots:
        out.append(range(start, start + shot.frame_count))
        start += shot.frame_count
    return out


def _entity_frames(log: EventLog) -> dict[str, set[int]]:
    frames = {entity: set() for entity in entity_ids(log)}
    for shot, span in zip(log.shots, shot_frame_ranges(log)):
        for entity in {*shot.present_subjects, *shot.visible_entities, shot.location_id} - {UNKNOWN}:
            frames[entity].update(span)
    return frames


def _words(text: str) -> list[str]:
    """英文按词、中日韩按单字切分；unknown 视为空。"""
    return [] if text == UNKNOWN else re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]", text.casefold())


def word_f1(a: str, b: str) -> float:
    from collections import Counter
    x, y = Counter(_words(a)), Counter(_words(b))
    overlap = sum((x & y).values())
    return 0.0 if not overlap else 2 * overlap / (sum(x.values()) + sum(y.values()))


def alignment_score(gold: EventLog, predicted: EventLog, gold_id: str, predicted_id: str) -> float:
    """0.5×在场 Jaccard（按帧）＋0.5×外观词 F1；不同类为 0。"""
    if _entity_class(gold, gold_id) != _entity_class(predicted, predicted_id):
        return 0.0
    g, p = _entity_frames(gold)[gold_id], _entity_frames(predicted)[predicted_id]
    jaccard = len(g & p) / len(g | p) if g | p else 0.0
    return 0.5 * jaccard + 0.5 * word_f1(_entity_text(gold, gold_id), _entity_text(predicted, predicted_id))


def align_entities(gold: EventLog, predicted: EventLog, named) -> dict[str, str]:
    """金标编号 → 预测编号。named 中两边都登记且同类的编号直接对齐；其余同类内匈牙利匹配，分数低于 0.2 不对齐。"""
    from scipy.optimize import linear_sum_assignment
    gold_ids, pred_ids = entity_ids(gold), set(entity_ids(predicted))
    alignment = {e: e for e in sorted(set(named)) if e in gold_ids and e in pred_ids
                 and _entity_class(gold, e) == _entity_class(predicted, e)}
    rest_gold = [e for e in gold_ids if e not in alignment]
    rest_pred = [e for e in entity_ids(predicted) if e not in set(alignment.values())]
    for cls in sorted({_entity_class(gold, e) for e in rest_gold}):
        rows = [e for e in rest_gold if _entity_class(gold, e) == cls]
        cols = [e for e in rest_pred if _entity_class(predicted, e) == cls]
        if not cols:
            continue
        scores = [[alignment_score(gold, predicted, g, p) for p in cols] for g in rows]
        for r, c in zip(*linear_sum_assignment(scores, maximize=True)):
            if scores[r][c] >= ALIGN_MIN_SCORE:
                alignment[rows[r]] = cols[c]
    return alignment


# --- 逐条比对 ---------------------------------------------------------------

class MissingVerdict(KeyError):
    """自由文本对没有裁决；调用方先取裁决表再比对，这里不猜等价。"""


@dataclass(frozen=True)
class EntryOutcome:
    number: int
    status: Literal["match", "contradiction", "missing"]
    #: 判为等价且写法不同的自由文本：((层, 键…), 金标值, 预测值)。
    rewrites: tuple[tuple[tuple, str, str], ...] = ()


def entry_outcomes(gold: EventLog, predicted: EventLog, alignment: Mapping[str, str],
                   verdicts: Mapping[tuple[str, str], bool]) -> tuple[EntryOutcome, ...]:
    """金标声明表逐条对预测日志判 match/contradiction/missing。

    比较时间点是金标每镜的首帧和末帧（执行计划 P5）：首帧读预测里含该帧那一镜的起点快照，末帧读含该帧那一镜的
    终点快照；事实、镜字段、在场、空间事实、事件都在这两个时间点各比一次再合并（任一矛盾即矛盾，全部匹配才匹配，
    其余算漏）。外观不分时间点。本镜有镜内（on_screen）事件改了某条声明的键时，这条声明只比首帧（改前值），
    末帧的状态由事件条目（前后状态）计分，不重复扣分。
    编号、枚举逐字比（编号先经对齐映射）；自由文本（state 值、外观、景别、机位、动作）逐字相同即等价，
    否则查 verdicts[(金标值, 预测值)]，缺裁决抛 MissingVerdict。未登记/unknown 算 missing，不算矛盾。
    """
    validate_event_log(gold)
    validate_event_log(predicted)
    ids = set(entity_ids(gold))

    def mapped(entity: str) -> str | None:
        if entity.startswith("toward Subject "):
            target = alignment.get(entity.removeprefix("toward "))
            return None if target is None else "toward " + target
        return alignment.get(entity) if entity in ids else entity

    def map_key(key: tuple) -> tuple | None:
        out = [mapped(key[0]), key[1], *(mapped(t) for t in key[2:])]
        return None if None in out else tuple(out)

    def equal_text(layer: tuple, g: str, p: str, rewrites: list) -> bool:
        if g == p:
            return True
        if (g, p) not in verdicts:
            raise MissingVerdict((g, p))
        if verdicts[(g, p)] is True:
            if (layer, g, p) not in rewrites:
                rewrites.append((layer, g, p))
            return True
        return False

    g_starts, g_ends, _, _ = _timeline(gold)
    p_starts, p_ends, _, _ = _timeline(predicted)
    g_frames, p_frames = shot_frame_ranges(gold), shot_frame_ranges(predicted)
    g_index = {s.shot_id: i for i, s in enumerate(gold.shots)}

    def p_shot(frame: int):
        for shot, span in zip(predicted.shots, p_frames):
            if frame in span:
                return shot
        return None

    def at(shot_id: str, point: str):
        span = g_frames[g_index[shot_id]]
        return p_shot(span[0] if point == "first" else span[-1])

    def compare_fact(g: AtomicFact | None, key: tuple, snapshot: Mapping | None, rewrites: list) -> str:
        """g=None 表示该键在此时点应不存在。"""
        mk = map_key(key)
        if snapshot is None or mk is None:
            return "missing"
        p = snapshot.get(mk)
        if g is None:
            return "contradiction" if p is not None else "match"
        if g.family == "relation":
            if p is not None:
                return "match"
            rival = [f for k, f in snapshot.items() if f.family == "relation" and k[:2] == mk[:2]]
            return "contradiction" if rival else "missing"
        if p is None:
            return "missing"
        return "match" if equal_text(("fact", *key), str(g.object_or_value), str(p.object_or_value), rewrites) else "contradiction"

    def combine(statuses: Sequence[str]) -> str:
        if "contradiction" in statuses:
            return "contradiction"
        return "match" if statuses and all(s == "match" for s in statuses) else "missing"

    def shot_status(e: FactEntry, point: str, rewrites: list) -> str:
        shot = at(e.shot_id, point)
        g_value = getattr(gold.shots[g_index[e.shot_id]], e.target)
        p_value = UNKNOWN if shot is None else getattr(shot, e.target)
        if p_value == UNKNOWN:
            return "missing"
        if e.target == "location_id":
            return "match" if mapped(g_value) == p_value else ("missing" if mapped(g_value) is None else "contradiction")
        if e.target == "movement":
            return "match" if g_value == p_value else "contradiction"
        if e.target in ("shot_size", "camera_angle") and canonical_shot_field(e.target, g_value) == canonical_shot_field(e.target, p_value):
            return "match"  # 两边先过统一词表再比：``medium`` 与 ``Medium shot`` 同一个值
        return "match" if equal_text(("shot", e.shot_id, e.target), g_value, p_value, rewrites) else "contradiction"

    def screen_status(e: FactEntry, point: str) -> str:
        shot, target = at(e.shot_id, point), alignment.get(e.fact.subject)
        found = [] if shot is None or target is None else [
            f for f in shot.screen_facts if f.subject == target and f.predicate_or_axis == e.fact.predicate_or_axis and _known(f)]
        if not found:
            return "missing"
        value = e.fact.object_or_value
        want = mapped(value) if e.fact.predicate_or_axis in ("gaze", "orientation") else value
        return "missing" if want is None else ("match" if any(f.object_or_value == want for f in found) else "contradiction")

    def status_at(e: FactEntry, point: str, rewrites: list) -> str:
        shot = at(e.shot_id, point)
        if e.kind == "shot":
            return shot_status(e, point, rewrites)
        if e.kind == "presence":
            target = alignment.get(e.target)
            return "missing" if shot is None or target is None else (
                "match" if target in shot.present_subjects else "contradiction")
        if e.kind == "screen":
            return screen_status(e, point)
        key = fact_key(e.fact)
        gold_snapshots, predicted_snapshots = (g_starts, p_starts) if point == "first" else (g_ends, p_ends)
        return compare_fact(gold_snapshots[e.shot_id].get(key), key,
                            None if shot is None else predicted_snapshots[shot.shot_id], rewrites)

    def changed_in_shot(e: FactEntry) -> bool:
        return e.kind == "fact" and any(ev.shot_id == e.shot_id and ev.on_screen is True and event_key(ev) == fact_key(e.fact)
                                        for ev in gold.events)

    outcomes = []
    for e in fact_roles(gold):
        rewrites: list = []
        if e.kind == "appearance":
            target = alignment.get(e.target)
            p_text = _entity_text(predicted, target) if target is not None else UNKNOWN
            status = "missing" if p_text == UNKNOWN else (
                "match" if equal_text(("appearance", e.target), _entity_text(gold, e.target), p_text, rewrites) else "contradiction")
        elif e.kind == "event":
            event = e.event
            key = event_key(event)
            first, last = at(e.shot_id, "first"), at(e.shot_id, "last")
            items = [compare_fact(event.after, key, None if last is None else p_ends[last.shot_id], rewrites)]
            if event.on_screen is True and event.before is not None:
                items.append(compare_fact(event.before, key, None if first is None else p_starts[first.shot_id], rewrites))
            if event.after is None and items[0] == "match" and len(items) == 1:
                items[0] = "missing"  # 不存在本身不能证明删过；要有起点在场佐证
            status = combine(items)
        else:
            points = ("first",) if changed_in_shot(e) else ("first", "last")
            status = combine([status_at(e, point, rewrites) for point in points])
        outcomes.append(EntryOutcome(e.number, status, tuple(rewrites) if status == "match" else ()))
    return tuple(outcomes)


# --- 联合第 1 档：规划器日志与真值合并 ---------------------------------------

def _event_log_raw(log: EventLog) -> dict:
    return log.model_dump(by_alias=True, mode="json")


def merge_planned_log(truth: EventLog, planned: EventLog, labels: Mapping[str, Sequence[int]],
                      verdicts: Mapping[tuple[str, str], bool], *, script: str) -> tuple[EventLog, dict]:
    """逐镜档规划器日志与真值合并（镜头锁定，不符报错）。先实体对齐（script 里逐字点名的编号直接对齐，
    其余在同类内匈牙利匹配），再按真值声明表逐条：

    同键同值→用规划器写法；值矛盾→回退真值；多写→删；漏写的写明或需推出→补真值；漏写的自由或空间→保持漏写（空间类由规划器自己推出，它漏了就是漏了）。
    合并从真值出发构造，所以多写天然不进。规划器写法按 (键, 真值) 全局替换，保证时间线前后一致。
    任何偏离真值的改动若使日志不再通过 validate_event_log，就退回真值并记入 reverted（例如被保留事件
    依赖的自由声明）。不变式：合并后编译出的每个从句，要么在真值编译结果里，要么是裁决等价的改写，
    由 merge_invariant_violations 检查。
    """
    validate_event_log(truth)
    validate_event_log(planned)
    if [(s.shot_id, s.frame_count) for s in planned.shots] != [(s.shot_id, s.frame_count) for s in truth.shots]:
        raise ValueError("merge accepts only shot-tier planner logs with the locked shot ids and frame counts")
    entries = fact_roles(truth)
    classes = require_labels(entries, labels)
    alignment = align_entities(truth, planned, script_named_ids(script))
    outcomes = {o.number: o for o in entry_outcomes(truth, planned, alignment, verdicts)}
    rewrites: dict[tuple, str] = {}
    changes: list[tuple] = []
    for e in entries:
        outcome = outcomes[e.number]
        if outcome.status == "match":
            for layer, gold_value, planned_value in outcome.rewrites:
                if (layer, gold_value) not in rewrites:
                    rewrites[(layer, gold_value)] = planned_value
                    changes.append(("rewrite", layer, gold_value, planned_value))
        elif outcome.status == "missing" and (e.number in classes["free"] or e.number in classes["spatial"]):
            changes.append(("omit", e))
    raw = _event_log_raw(truth)
    fact_ids = {id(f): i for i, f in enumerate(truth.initial_state)}
    event_ids = {id(ev): i for i, ev in enumerate(truth.events)}

    def build(applied: Sequence[tuple]) -> dict:
        out = {**raw, "subjects": [dict(s) for s in raw["subjects"]],
               "locations": [dict(l) for l in raw["locations"]],
               "shots": [dict(s, present_subjects=list(s["present_subjects"]), visible_entities=list(s["visible_entities"]),
                              screen_facts=list(s["screen_facts"])) for s in raw["shots"]]}
        drop_facts, drop_events, drop_screen = set(), set(), set()
        value_map: dict[tuple, str] = {}
        for change in applied:
            if change[0] == "rewrite":
                _, layer, gold_value, planned_value = change
                if layer[0] == "fact":
                    value_map[(tuple(layer[1:]), gold_value)] = planned_value
                elif layer[0] == "appearance":
                    for item in out["subjects"]:
                        if item["subject_id"] == layer[1]:
                            item["appearance"] = planned_value
                    for item in out["locations"]:
                        if item["location_id"] == layer[1]:
                            item["appearance"] = planned_value
                else:
                    _, shot_id, field = layer
                    next(s for s in out["shots"] if s["shot_id"] == shot_id)[field] = planned_value
                continue
            entry = change[1]
            shot = next(s for s in out["shots"] if s["shot_id"] == entry.shot_id)
            if entry.kind == "appearance":
                for item in out["subjects"] + out["locations"]:
                    if item.get("subject_id", item.get("location_id")) == entry.target:
                        item["appearance"] = UNKNOWN
            elif entry.kind == "shot":
                shot[entry.target] = UNKNOWN
            elif entry.kind == "presence":
                shot["present_subjects"] = [x for x in shot["present_subjects"] if x != entry.target]
                shot["visible_entities"] = [x for x in shot["visible_entities"] if x != entry.target]
            elif entry.kind == "fact":
                drop_facts.add(fact_ids[id(entry.fact)])
            elif entry.kind == "screen":
                drop_screen.add((entry.shot_id, entry.fact.subject, entry.fact.predicate_or_axis))
            else:
                drop_events.add(event_ids[id(entry.event)])

        def fact(item):
            if item is None:
                return None
            parsed = AtomicFact.model_validate(item)
            planned_value = value_map.get((fact_key(parsed), str(parsed.object_or_value)))
            return item if planned_value is None else dict(item, object_or_value=planned_value)

        out["initial_state"] = [fact(f) for i, f in enumerate(raw["initial_state"]) if i not in drop_facts]
        kept = [dict(ev, before=fact(ev["before"]), after=fact(ev["after"])) for i, ev in enumerate(raw["events"]) if i not in drop_events]
        out["events"] = [dict(ev, event_id=f"e{n}") for n, ev in enumerate(kept, 1)]
        for shot in out["shots"]:
            shot["screen_facts"] = [f for f in shot["screen_facts"]
                                    if (shot["shot_id"], f["subject"], f["predicate_or_axis"]) not in drop_screen]
        return out

    applied: list[tuple] = []
    reverted: list[tuple] = []
    for change in changes:
        try:
            validate_event_log(parse_event_log(build([*applied, change])))
        except EventLogError:
            reverted.append(change)
            continue
        applied.append(change)
    merged = parse_event_log(build(applied))
    validate_event_log(merged)

    def describe(change):
        if change[0] == "rewrite":
            return {"change": "rewrite", "layer": list(change[1]), "truth": change[2], "planned": change[3]}
        return {"change": "omit", "entry": change[1].number, "kind": change[1].kind}
    report = {"alignment": alignment, "outcomes": {n: o.status for n, o in outcomes.items()},
              "applied": [describe(c) for c in applied], "reverted": [describe(c) for c in reverted],
              "rewrites": [(c[2], c[3]) for c in applied if c[0] == "rewrite"]}
    return merged, report


def _clause_atoms(rows: Sequence[str]) -> set[str]:
    atoms = set()
    for row in rows:
        for part in re.split(r"(?<=\.)\s+|;\s+", row):
            part = part.strip().removesuffix(".").removeprefix("by the end of the shot ").removesuffix(" at first")
            for atom in part.split(" and "):
                if atom.strip():
                    atoms.add(atom.strip())
    return atoms


def merge_invariant_violations(truth: EventLog, merged: EventLog, rewrites: Sequence[tuple[str, str]],
                               token_counter: Callable[[str], int]) -> list[str]:
    """合并日志编译出的从句里，既不在真值编译结果、也不能由裁决等价的改写还原成真值从句的那些。"""
    def rows(log):
        return [row for span in dependency_segment_spans(log)
                for row in compile_event_log(log, span=span, references=(), token_counter=token_counter).target_rows]
    truth_atoms = _clause_atoms(rows(truth))
    restored = []
    for row in rows(merged):
        for gold_value, planned_value in rewrites:
            row = row.replace(planned_value, gold_value)
        restored.append(row)
    return sorted(_clause_atoms(restored) - truth_atoms)
