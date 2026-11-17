"""旧 event_log_v3 规划器四项指标：可编译率（逐层）、事实 F1（写明/需推出分开，空间类另报）、跨镜编号一致、要求保住率。

只做离线计算，不调用任何模型。自由文本是否等价由调用方传入裁决表（键为 (金标值, 预测值)）；
先用 :func:`free_text_pairs` 列出需要裁决的文本对，取得裁决后再算 F1 与保住率。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

from mspc.data.event_log import (
    ALIGN_MIN_SCORE, EventLog, EventLogError, entity_ids, shot_frame_ranges, alignment_score, compile_event_log,
    dependency_segment_spans, entry_outcomes, fact_roles, parse_event_log, require_labels, validate_event_log,
)
from mspc.data.event_planner_contract import normalize_planner_answer, planner_lock_issues

COMPILE_LAYERS = ("json", "schema", "structure", "input_lock", "dependency_segments", "compile")
#: 事实 F1 分开报的类；空间类（位置、朝向、视线）单列，不并入写明/需推出主分。
FACT_CLASSES = ("stated", "derivable", "spatial")


def compile_layers(raw: str, request: Mapping[str, Any], *, token_counter: Callable[[str], int]) -> dict[str, Any]:
    """JSON → schema → 结构 → 输入锁 → 依赖分段 → 编译（每条目标行 umT5 ≤512），停在第一层失败处。"""
    result: dict[str, Any] = {"passed": [], "failed_layer": None, "issues": [], "event_log": None}

    def fail(layer: str, issues: list) -> dict[str, Any]:
        result.update(failed_layer=layer, issues=issues)
        return result

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail("json", [{"rule": "json", "message": str(exc)}])
    result["passed"].append("json")
    payload = normalize_planner_answer(payload)  # 值为 unknown 的事实带 evidence 不拒收，丢掉再校验
    try:
        log = parse_event_log(payload)
    except EventLogError as exc:
        return fail("schema", [asdict(issue) for issue in exc.issues])
    result["passed"].append("schema")
    result["event_log"] = payload
    try:
        validate_event_log(log)
    except EventLogError as exc:
        return fail("structure", [asdict(issue) for issue in exc.issues])
    result["passed"].append("structure")
    issues = planner_lock_issues(request, payload)
    if issues:
        return fail("input_lock", [{"rule": "input_lock", "message": text} for text in issues])
    result["passed"].append("input_lock")
    try:
        spans = dependency_segment_spans(log)
    except EventLogError as exc:
        return fail("dependency_segments", [asdict(issue) for issue in exc.issues])
    result["passed"].append("dependency_segments")
    try:
        for span in spans:
            compile_event_log(log, span=span, references=(), token_counter=token_counter)
    except EventLogError as exc:
        return fail("compile", [asdict(issue) for issue in exc.issues])
    result["passed"].append("compile")
    return result


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def compilability(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """逐层通过率；分母是全部请求（含传输失败，传输失败的行传 passed=[]）。"""
    return {layer: _rate(sum(layer in row["passed"] for row in results), len(results)) for layer in COMPILE_LAYERS}


def _structured(predicted: Mapping[str, Any] | None) -> EventLog | None:
    if predicted is None:
        return None
    log = parse_event_log(predicted)
    validate_event_log(log)
    return log


class _PairRecorder(dict):
    """遍历一次比对，只记下需要裁决的自由文本对。"""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: set[tuple[str, str]] = set()

    def __contains__(self, pair: object) -> bool:
        self.pairs.add(pair)  # type: ignore[arg-type]
        return True

    def __getitem__(self, pair: tuple[str, str]) -> bool:
        return False


def free_text_pairs(gold: EventLog, predicted: EventLog, alignment: Mapping[str, str]) -> list[tuple[str, str]]:
    recorder = _PairRecorder()
    entry_outcomes(gold, predicted, alignment, recorder)
    return sorted(recorder.pairs)


def fact_f1(gold: EventLog, predicted: EventLog | None, labels: Mapping[str, Sequence[int]],
            alignment: Mapping[str, str], verdicts: Mapping[tuple[str, str], bool]) -> dict[str, dict[str, int]]:
    """写明、需推出、空间三类分别计数：match、contradiction、gold。预测不可结构化时全部算漏（只进召回分母）。"""
    classes = require_labels(fact_roles(gold), labels)
    outcomes = {} if predicted is None else {o.number: o.status for o in entry_outcomes(gold, predicted, alignment, verdicts)}
    return {name: {"match": sum(outcomes.get(n) == "match" for n in classes[name]),
                   "contradiction": sum(outcomes.get(n) == "contradiction" for n in classes[name]),
                   "gold": len(classes[name])} for name in FACT_CLASSES}


def pool_fact_f1(rows: Sequence[Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, Any]]:
    """合并计数后算：精确＝匹配/(匹配＋矛盾)，召回＝匹配/该类金标数；多写不计分。"""
    out = {}
    for name in FACT_CLASSES:
        match = sum(row[name]["match"] for row in rows)
        contradiction = sum(row[name]["contradiction"] for row in rows)
        gold = sum(row[name]["gold"] for row in rows)
        precision = match / (match + contradiction) if match + contradiction else None
        recall = match / gold if gold else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        out[name] = {"match": match, "contradiction": contradiction, "gold": gold,
                     "precision": precision, "recall": recall, "f1": f1}
    return out


def _present(shot, entity: str) -> bool:
    return entity in shot.present_subjects or entity in shot.visible_entities or entity == shot.location_id


def id_consistency(gold: EventLog, predicted: EventLog | None, alignment: Mapping[str, str]) -> dict[str, Any]:
    """金标中出现于 ≥2 镜的实体：在它每个出现镜的首帧和末帧，对齐实体是否在预测的对应镜里在场。

    另报重复登记数：未被对齐、但与某金标实体同类分数 ≥0.2 的预测实体个数。
    """
    frames_g = shot_frame_ranges(gold)
    eligible = [e for e in entity_ids(gold) if sum(_present(shot, e) for shot in gold.shots) >= 2]
    points, kept = 0, 0
    entities_all = 0
    for entity in eligible:
        target = alignment.get(entity)
        ok = True
        for shot, span in zip(gold.shots, frames_g):
            if not _present(shot, entity):
                continue
            for frame in (span[0], span[-1]):
                points += 1
                hit = False
                if predicted is not None and target is not None:
                    for p_shot, p_span in zip(predicted.shots, shot_frame_ranges(predicted)):
                        if frame in p_span:
                            hit = _present(p_shot, target)
                kept += hit
                ok &= hit
        entities_all += ok
    duplicates = 0
    if predicted is not None:
        aligned = set(alignment.values())
        for candidate in entity_ids(predicted):
            if candidate not in aligned and any(
                    alignment_score(gold, predicted, entity, candidate) >= ALIGN_MIN_SCORE for entity in entity_ids(gold)):
                duplicates += 1
    return {"eligible_entities": len(eligible), "time_points": _rate(kept, points),
            "entities_present_throughout": _rate(entities_all, len(eligible)), "duplicate_registrations": duplicates}


def requirement_retention(gold: EventLog, predicted: EventLog | None, labels: Mapping[str, Sequence[int]],
                          alignment: Mapping[str, str], verdicts: Mapping[tuple[str, str], bool],
                          request: Mapping[str, Any]) -> dict[str, Any]:
    """写明事实＋写明事件＋事件先后＋锁定字段（含总帧数），逐条记满足与否；空间类不在其中。
    rate 为本样本满足比例，perfect 为全部保住。"""
    entries = fact_roles(gold)
    classes = require_labels(entries, labels)
    outcomes = {} if predicted is None else {o.number: o.status for o in entry_outcomes(gold, predicted, alignment, verdicts)}
    requirements = [{"kind": e.kind, "entry": e.number, "satisfied": outcomes.get(e.number) == "match"}
                    for e in entries if e.number in classes["stated"]]
    index = {s.shot_id: i for i, s in enumerate(gold.shots)}
    stated_events = [e for e in entries if e.kind == "event" and e.number in classes["stated"]]

    def predicted_shot(shot_id: str) -> int | None:
        last = shot_frame_ranges(gold)[index[shot_id]][-1]
        for i, span in enumerate(shot_frame_ranges(predicted)):
            if last in span:
                return i
        return None

    for a, b in zip(stated_events, stated_events[1:]):
        if index[a.shot_id] == index[b.shot_id]:
            continue
        satisfied = (predicted is not None and outcomes.get(a.number) == "match" and outcomes.get(b.number) == "match"
                     and None not in (predicted_shot(a.shot_id), predicted_shot(b.shot_id))
                     and predicted_shot(a.shot_id) < predicted_shot(b.shot_id))
        requirements.append({"kind": "event_order", "entry": [a.number, b.number], "satisfied": satisfied})
    counts = [] if predicted is None else [shot.frame_count for shot in predicted.shots]
    requirements.append({"kind": "total_frames", "entry": None, "satisfied": predicted is not None and sum(counts) == request["total_frames"]})
    if "shots" in request:
        have = {} if predicted is None else {shot.shot_id: (i, shot.frame_count) for i, shot in enumerate(predicted.shots)}
        for i, shot in enumerate(request["shots"]):
            requirements.append({"kind": "locked_shot", "entry": shot["shot_id"],
                                 "satisfied": have.get(shot["shot_id"]) == (i, shot["frame_count"])})
    satisfied = sum(item["satisfied"] for item in requirements)
    return {"requirements": requirements, "rate": satisfied / len(requirements), "perfect": satisfied == len(requirements)}


def retention_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Overall＝各样本满足比例的平均，Perfect＝全部保住的样本比例（WanPE 的两项）。"""
    if not rows:
        raise ValueError("retention summary needs at least one sample")
    return {"samples": len(rows), "overall": sum(row["rate"] for row in rows) / len(rows),
            "perfect": _rate(sum(row["perfect"] for row in rows), len(rows))}


def structured_prediction(result: Mapping[str, Any]) -> EventLog | None:
    """compile_layers 的结果过了结构层就给出日志对象，否则 None（后续指标按全漏计）。"""
    return _structured(result["event_log"]) if "structure" in result["passed"] else None
