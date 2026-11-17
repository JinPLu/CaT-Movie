"""状态层 v2 的规范生产端；不接入 event_log 或渲染器。

原子事实保留七元组，登记标记是旁表，不把开放谓词静默改名。
镜号从 0 开始，shot_span 为闭区间；bbox 的 frame_id 指向显式关键帧。
未知是观测缺口，不能据此删除已知事实。验证失败直接报错。
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer
import yaml


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Entity(StrictRecord):
    id: str
    kind: Literal["person", "object", "animal"]
    static_appearance: str

    @model_validator(mode="after")
    def valid_id(self):
        if not re.fullmatch(r"Subject [1-9]\d*", self.id):
            raise ValueError("entity id must be Subject N")
        if not self.static_appearance.strip():
            raise ValueError("static appearance must be explicit; no mutable state")
        return self


class Anchor(StrictRecord):
    id: str
    description: str
    front_definition: str | None


class Location(StrictRecord):
    id: str
    description: str
    anchors: list[Anchor]

    @model_validator(mode="after")
    def valid_location(self):
        if not re.fullmatch(r"Location [1-9]\d*", self.id):
            raise ValueError("location id must be Location N")
        if not 0 <= len(self.anchors) <= 3:
            raise ValueError("each location allows 0–3 visible fixed anchors; never invent unseen anchors")
        for a in self.anchors:
            if not re.fullmatch(re.escape(self.id) + r"/Anchor [1-9]\d*", a.id):
                raise ValueError("anchor id must be Location N/Anchor N")
            if not a.description.strip() or a.front_definition == "":
                raise ValueError("empty anchor description/front definition")
        return self


class Shot(StrictRecord):
    index: int
    location: str
    present: list[str]


class EvidenceFrame(StrictRecord):
    id: str
    shot: int
    proxy_frame_index: int
    width: int
    height: int
    present: list[str]


class Keyframe(EvidenceFrame):
    phase: Literal["before", "after"]


class AtomicFact(StrictRecord):
    subject: str
    family: Literal["state", "relation", "screen"]
    predicate_or_axis: str
    object_or_value: str | list[str] | list[int]
    scope: str
    shot_span: list[int] = Field(
        min_length=2, max_length=2,
        description="从0开始的闭区间[start,end]，必须恰好两个镜号；单镜写[0,0]而非[0]。",
    )
    evidence: Literal["observed", "carried", "unknown"] | None = None

    @model_validator(mode="after")
    def explicit_evidence_for_known_value(self):
        if self.object_or_value == "unknown":
            if "evidence" in self.model_fields_set:
                raise ValueError("unknown observation cannot carry an evidence field")
        elif self.evidence is None:
            raise ValueError("known value requires explicit evidence")
        return self

    @model_serializer(mode="wrap")
    def serialize_fact(self, handler):
        value = handler(self)
        if self.object_or_value == "unknown":
            value.pop("evidence", None)
        return value

    def atomic_tuple(self) -> tuple:
        return tuple(getattr(self, k) for k in self.__class__.model_fields)


class FactRecord(StrictRecord):
    fact: AtomicFact
    registration: Literal["registered", "unregistered", "not_applicable"]
    frame_id: str | None
    evidence_frames: list[str]
    coverage: str | None
    source: str | None


class PredicateRegistry(StrictRecord):
    schema_version: Literal["state_predicates_v2"]
    state_axes: list[str]
    relation_groups: dict[str, list[str]]
    anchor_orientation_predicates: list[str]
    portable_predicates: list[str]
    nonpersistent_predicates: list[str]
    containment_predicates: list[str]
    screen_axes: dict[str, list[str] | str]
    evidence: list[str]
    coverage: dict[str, str]
    relation_coverage: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> PredicateRegistry:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def relation_registration(self, phrase: str) -> str:
        if not phrase.strip() or phrase != phrase.strip():
            raise ValueError("predicate must be a nonempty exact phrase")
        return "registered" if any(phrase in x for x in self.relation_groups.values()) else "unregistered"


class FactWindow(StrictRecord):
    schema_version: Literal["state_facts_v2"]
    window_id: str
    entities: list[Entity]
    locations: list[Location]
    shots: list[Shot]
    sampled_frames: list[EvidenceFrame]
    keyframes: list[Keyframe]
    records: list[FactRecord]


def validate_bbox_value(value: object) -> None:
    """完整窗口与候选审计共用坐标合同；字符串坐标不是合法列表。"""
    if not isinstance(value, list) or len(value) != 4 or not all(type(x) is int and 0 <= x <= 1000 for x in value):
        raise ValueError("bbox must be integer xyxy in 0–1000")
    if not value[0] < value[2] or not value[1] < value[3]:
        raise ValueError("bbox must have positive area")


def validate_window(window: FactWindow, registry: PredicateRegistry) -> None:
    """校验注册、作用域、朝向、关键帧和逐镜包含环；不推测或修补像素事实。"""
    entities = {e.id: e for e in window.entities}
    locations = {l.id: l for l in window.locations}
    anchors = {a.id: (l.id, a) for l in window.locations for a in l.anchors}
    all_ids = set(entities) | set(locations) | set(anchors)
    if len(all_ids) != len(window.entities) + len(window.locations) + sum(len(l.anchors) for l in window.locations):
        raise ValueError("duplicate registry id")
    shots = {s.index: s for s in window.shots}
    if set(shots) != set(range(len(window.shots))) or len(shots) != len(window.shots):
        raise ValueError("shots must be unique consecutive indices from zero")
    frames = {f.id: f for f in window.sampled_frames}
    keys = {f.id: f for f in window.keyframes}
    if len(frames) != len(window.sampled_frames) or len(keys) != len(window.keyframes):
        raise ValueError("duplicate frame id")
    for f in keys.values():
        if f.id not in frames or f.model_dump(exclude={"phase"}) != frames[f.id].model_dump():
            raise ValueError("keyframe must match a sampled evidence frame")
    for s in window.shots:
        if s.location not in locations or len(s.present) != len(set(s.present)) or not set(s.present) <= set(entities):
            raise ValueError("shot has unregistered location/entity or duplicate presence")
    for f in frames.values():
        if f.shot not in shots or f.proxy_frame_index < 0 or min(f.width, f.height) <= 0:
            raise ValueError("invalid keyframe coordinates")
        if len(f.present) != len(set(f.present)) or not set(f.present) <= set(shots[f.shot].present):
            raise ValueError("keyframe presence must be declared in shot")
    graphs: dict[tuple[int, str | None], dict[str, set[str]]] = {}
    for record in window.records:
        fact = record.fact
        p, value = fact.predicate_or_axis, fact.object_or_value
        if fact.subject not in all_ids:
            raise ValueError("unregistered subject")
        if len(fact.shot_span) != 2 or not all(type(i) is int for i in fact.shot_span):
            raise ValueError("shot_span must be two integer indices")
        start, end = fact.shot_span
        if start > end or start not in shots or end not in shots:
            raise ValueError("invalid shot_span")
        if not record.evidence_frames or not set(record.evidence_frames) <= set(frames):
            raise ValueError("explicit sampled evidence frames required")
        if any(not start <= frames[f].shot <= end for f in record.evidence_frames) and fact.evidence != "carried":
            raise ValueError("evidence frame outside shot_span")
        if fact.evidence == "carried" and any(frames[f].shot > end for f in record.evidence_frames):
            raise ValueError("carried evidence cannot come from future shots")
        if value == "unknown" and fact.evidence is not None:
            raise ValueError("unknown observation cannot overwrite observed/carried fact")
        if fact.scope not in {"scene", f"screen@{start}"}:
            raise ValueError("invalid scope")
        if fact.family == "screen":
            if start != end or fact.scope != f"screen@{start}" or fact.evidence == "carried":
                raise ValueError("screen facts are observed/unknown in one shot, never carried")
            if fact.subject not in entities or fact.subject not in shots[start].present:
                raise ValueError("screen subject must be present registered entity")
            if record.registration != "not_applicable" or record.coverage is not None:
                raise ValueError("screen is not a seventh-class scene fact")
            if p not in registry.screen_axes:
                raise ValueError("unregistered screen axis")
            if p == "bbox":
                if record.frame_id not in keys:
                    raise ValueError("bbox requires selected keyframe")
                f = frames[record.frame_id]
                if f.shot != start or fact.subject not in f.present or record.frame_id not in record.evidence_frames:
                    raise ValueError("bbox entity/frame mismatches")
                validate_bbox_value(value)
                if fact.evidence != "observed":
                    raise ValueError("unknown bbox belongs in undecidable items, not invented coordinates")
            elif not isinstance(value, str):
                raise ValueError("screen value must be text")
            elif value != "unknown":
                allowed = registry.screen_axes[p]
                if p == "gaze":
                    if value not in all_ids or value == fact.subject:
                        raise ValueError("gaze target must be distinct registered id")
                elif p == "orientation" and value.startswith("toward Subject "):
                    target = value.removeprefix("toward ")
                    if target not in entities or target == fact.subject:
                        raise ValueError("orientation target must be distinct registered entity")
                elif value not in allowed:
                    raise ValueError("invalid screen value")
            if record.frame_id is not None and (record.frame_id not in frames or frames[record.frame_id].shot != start):
                raise ValueError("screen frame id mismatch")
            continue
        if fact.scope != "scene":
            raise ValueError("state/relation must have scene scope")
        if record.registration == "not_applicable" or record.coverage not in set(registry.coverage.values()):
            raise ValueError("scene fact requires registration and seven-class coverage")
        if fact.family == "state":
            if p not in registry.state_axes or record.registration != "registered" or record.coverage != registry.coverage[p]:
                raise ValueError("invalid state axis/coverage")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("state value must be nonempty open phrase")
            if (p == "environment") != (fact.subject in locations):
                raise ValueError("environment belongs to location; other axes to entities")
            if fact.subject in anchors:
                raise ValueError("mutable anchor state needs an entity registration")
        else:
            if record.registration != registry.relation_registration(p):
                raise ValueError("relation registration mismatch; no silent synonym mapping")
            if record.coverage == registry.coverage["interpersonal"] and not record.source:
                raise ValueError("interpersonal relation requires explicit evidence source")
            if record.registration == "registered" and record.coverage != registry.relation_coverage[p]:
                raise ValueError("relation coverage mismatches registered predicate")
            if not isinstance(value, (str, list)):
                raise ValueError("relation requires registry id(s)")
            targets = value if isinstance(value, list) else [value]
            if not targets or not all(isinstance(t, str) and t in all_ids for t in targets):
                raise ValueError("unregistered relation target")
            if fact.subject in targets or len(targets) != len(set(targets)):
                raise ValueError("self/duplicate relation")
            if (p == "between" and len(targets) != 2) or (p != "between" and len(targets) != 1):
                raise ValueError("between requires two endpoints; other relations one")
            if p in registry.nonpersistent_predicates and (fact.evidence == "carried" or start != end):
                raise ValueError("contact cannot automatically persist across shots")
            for target in targets:
                if target in anchors:
                    loc, anchor = anchors[target]
                    if any(shots[i].location != loc for i in range(start, end + 1)):
                        raise ValueError("anchor relationship expired after location change")
                    if p in registry.anchor_orientation_predicates and anchor.front_definition is None:
                        raise ValueError("anchor has no defined front; use next to / at")
                if p in registry.containment_predicates and fact.evidence != "unknown":
                    for i in range(start, end + 1):
                        graphs.setdefault((i, record.frame_id), {}).setdefault(fact.subject, set()).add(target)
        if record.frame_id is not None and (record.frame_id not in frames or not start <= frames[record.frame_id].shot <= end):
            raise ValueError("fact frame id mismatch")
        if fact.family == "state" and p == "environment" and any(
            shots[i].location != fact.subject for i in range(start, end + 1)
        ):
            raise ValueError("environment cannot persist to another location")
    for (shot, frame), graph in graphs.items():
        merged = {s: set(ts) for s, ts in graphs.get((shot, None), {}).items()}
        for s, ts in graph.items():
            merged.setdefault(s, set()).update(ts)
        def visit(node: str, path: set[str]) -> None:
            if node in path:
                raise ValueError("containment cycle")
            for target in merged.get(node, ()):
                visit(target, path | {node})
        for subject in merged:
            visit(subject, set())


def carry_facts(records: list[FactRecord], *, from_location: str, to_location: str,
                next_shot: int, registry: PredicateRegistry) -> list[FactRecord]:
    """延续既有事实；unknown 不写入事实状态，屏幕与瞬时接触不继承。

    地点变化只删除与旧地点/锚点相关的关系及环境；人与物 holds/wears/carries
    随人走。函数不读取模型输出，不生成事件；返回值仍须 validate_window。
    """
    result = []
    for r in records:
        f = r.fact
        if f.family == "screen" or f.object_or_value == "unknown" or f.evidence == "unknown" or f.predicate_or_axis in registry.nonpersistent_predicates:
            continue
        values = f.object_or_value if isinstance(f.object_or_value, list) else [f.object_or_value]
        bound_ids = [f.subject] + [v for v in values if isinstance(v, str)]
        if from_location != to_location and (f.predicate_or_axis == "environment" or any(
            x == from_location or x.startswith(from_location + "/") for x in bound_ids
        )):
            continue
        result.append(r.model_copy(update={"fact": f.model_copy(update={
            "shot_span": [next_shot, next_shot], "evidence": "carried"
        }), "frame_id": None}))
    return result
