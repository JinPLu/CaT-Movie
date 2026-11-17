"""规划器四项指标：真实 v3 日志与真实失败原答复；预测是对真实日志的显式改动，不是模型成绩。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.event_log import align_entities, entry_labels, fact_roles, parse_event_log
from mspc.data.event_planner_contract import planner_request
from mspc.evaluation.event_planner_metrics import (
    COMPILE_LAYERS, compilability, compile_layers, fact_f1, free_text_pairs, id_consistency, pool_fact_f1,
    requirement_retention, retention_summary, structured_prediction,
)

FIXTURES = Path(__file__).parents[2] / "fixtures"
REAL = json.loads((FIXTURES / "event_log_real_slice/04.json").read_text())["log"]
REFERENCE = json.loads((FIXTURES / "event_log_v3_reference/p1a-01.json").read_text())["event_log"]
FAILURES = {r["name"]: r for r in json.loads((FIXTURES / "planner_call/contract_real_failures.json").read_text())["responses"]}
COSTUME = ("a white long-sleeved top and dark shorts", "a white long-sleeve shirt with dark shorts")


def words(text):
    return len(text.split())


def request(log=REAL):
    return planner_request(script="Subject 1 freezes in Location 1.", event_log=log,
                           target_start_shot_id="s01", history_shot_ids=())


def test_real_gold_passes_every_layer_and_failures_stop_at_their_layer():
    gold = compile_layers(json.dumps(REAL), request(), token_counter=words)
    assert gold["passed"] == list(COMPILE_LAYERS) and gold["failed_layer"] is None
    assert compile_layers(json.dumps(REAL)[:200], request(), token_counter=words)["failed_layer"] == "json"
    span = json.loads((FIXTURES / "planner_call/9b_real_span_failure.json").read_text())["raw_content"]
    assert compile_layers(span, request(), token_counter=words)["failed_layer"] == "schema"
    plain = compile_layers(FAILURES["r2_27b_plain"]["raw_content"], request(), token_counter=words)
    assert plain["failed_layer"] in {"schema", "structure"}
    moved = deepcopy(REAL)
    moved["shots"][1]["frame_count"] += 4
    locked = compile_layers(json.dumps(moved), request(), token_counter=words)
    assert locked["failed_layer"] == "input_lock" and locked["passed"] == ["json", "schema", "structure"]
    long = compile_layers(json.dumps(REAL), request(), token_counter=lambda text: 600)
    assert long["failed_layer"] == "compile" and "dependency_segments" in long["passed"]
    rates = compilability([gold, locked, long, {"passed": []}])
    assert rates["json"] == {"numerator": 3, "denominator": 4, "rate": 0.75}
    assert rates["compile"]["numerator"] == 1


def test_real_reference_window_breaks_the_duration_rule():
    # GPT 参照窗的帧数不是 4n+1；它能结构化，但过不了时长规则，说明这条规则对旧参照日志不成立。
    result = compile_layers(json.dumps(REFERENCE), request(REFERENCE), token_counter=words)
    assert result["failed_layer"] == "input_lock" and "4n+1" in result["issues"][0]["message"]
    assert structured_prediction(result) is not None


def variant():
    data = deepcopy(REFERENCE)
    for fact in data["initial_state"]:
        if fact["predicate_or_axis"] == "costume":
            fact["object_or_value"] = COSTUME[1]
    for fact in data["shots"][1]["screen_facts"]:
        if fact["subject"] == "Subject 1" and fact["predicate_or_axis"] == "horizontal":
            fact["object_or_value"] = "left"
    return parse_event_log(data)


def labels_all_causes(log):
    return entry_labels(log, [e.number for e in fact_roles(log) if e.role == "cause"])


def test_real_fact_f1_counts_stated_derivable_and_spatial_separately():
    gold = parse_event_log(REFERENCE)
    labels = labels_all_causes(gold)
    alignment = align_entities(gold, gold, set())
    self_counts = fact_f1(gold, gold, labels, alignment, {})
    assert self_counts["stated"]["match"] == self_counts["stated"]["gold"] == len(labels["stated"])
    # 需推出：三处衣服、两处持钻复述，加三镜与上一镜相同的地点；空间类 27 条另计，不并入前两类。
    assert self_counts["derivable"]["match"] == len(labels["derivable"]) == 8
    assert self_counts["spatial"]["match"] == self_counts["spatial"]["gold"] == len(labels["spatial"]) == 27
    predicted = variant()
    alignment = align_entities(gold, predicted, set())
    assert free_text_pairs(gold, predicted, alignment) == [COSTUME]
    accepted = fact_f1(gold, predicted, labels, alignment, {COSTUME: True})
    # s02 的左右位置改了：矛盾记在空间类，不进写明/需推出主分。
    assert accepted["stated"]["contradiction"] == accepted["derivable"]["contradiction"] == 0
    assert accepted["spatial"]["contradiction"] == 1
    rejected = fact_f1(gold, predicted, labels, alignment, {COSTUME: False})
    assert rejected["derivable"]["contradiction"] == 3 and rejected["stated"]["contradiction"] == 1
    missing = fact_f1(gold, None, labels, {}, {})
    assert missing["stated"] == {"match": 0, "contradiction": 0, "gold": len(labels["stated"])}
    assert missing["spatial"] == {"match": 0, "contradiction": 0, "gold": 27}
    pooled = pool_fact_f1([accepted, missing])
    stated = pooled["stated"]
    assert stated["precision"] == 1.0
    assert stated["recall"] == stated["match"] / (2 * len(labels["stated"]))
    spatial = pooled["spatial"]
    assert spatial["precision"] == 26 / 27 and spatial["recall"] == 26 / 54


def test_real_id_consistency_and_duplicate_registration():
    gold = parse_event_log(REFERENCE)
    same = id_consistency(gold, gold, align_entities(gold, gold, set()))
    assert same["time_points"]["rate"] == 1 and same["duplicate_registrations"] == 0
    data = deepcopy(REFERENCE)
    data["shots"][1]["present_subjects"].remove("Subject 3")
    data["shots"][1]["visible_entities"].remove("Subject 3")
    data["shots"][1]["screen_facts"] = [f for f in data["shots"][1]["screen_facts"] if f["subject"] != "Subject 3"]
    data["subjects"].append({"subject_id": "Subject 4", "kind": "object", "appearance": "light rectangular wooden box"})
    data["shots"][1]["present_subjects"].append("Subject 4")
    predicted = parse_event_log(data)
    result = id_consistency(gold, predicted, align_entities(gold, predicted, set()))
    assert result["time_points"]["numerator"] == result["time_points"]["denominator"] - 2
    assert result["duplicate_registrations"] == 1
    assert id_consistency(gold, None, {})["time_points"]["numerator"] == 0


def test_real_requirement_retention_overall_and_perfect():
    gold = parse_event_log(REAL)
    labels = labels_all_causes(gold)
    alignment = align_entities(gold, gold, set())
    perfect = requirement_retention(gold, gold, labels, alignment, {}, request())
    kinds = {item["kind"] for item in perfect["requirements"]}
    assert {"total_frames", "locked_shot"} <= kinds and perfect["perfect"]
    moved = deepcopy(REAL)
    moved["shots"][1]["frame_count"] += 4
    shifted = parse_event_log(moved)
    # 帧数一改，后面各镜的首末帧落到预测的别的镜上；需要裁决的文本对全判不等价。
    verdicts = {pair: False for pair in free_text_pairs(gold, shifted, alignment)}
    broken = requirement_retention(gold, shifted, labels, alignment, verdicts, request())
    assert not broken["perfect"] and 0 < broken["rate"] < 1
    assert [item["entry"] for item in broken["requirements"] if not item["satisfied"] and item["kind"] == "locked_shot"] == ["s02"]
    nothing = requirement_retention(gold, None, labels, {}, {}, request())
    assert nothing["rate"] == 0
    summary = retention_summary([perfect, broken, nothing])
    assert summary["perfect"]["numerator"] == 1
    assert summary["overall"] == pytest.approx((1 + broken["rate"]) / 3)


def test_real_retention_ignores_the_spatial_class_but_keeps_total_frames():
    gold = parse_event_log(REFERENCE)
    labels = labels_all_causes(gold)
    predicted = variant()  # s02 的左右位置与金标矛盾（空间类）；写明事实、事件、帧数都保住
    alignment = align_entities(gold, predicted, set())
    kept = requirement_retention(gold, predicted, labels, alignment, {COSTUME: True}, request(REFERENCE))
    assert kept["perfect"] and kept["rate"] == 1
    kinds = [item["kind"] for item in kept["requirements"]]
    assert len(kinds) == len(labels["stated"]) + kinds.count("total_frames") + kinds.count("locked_shot")
    assert kinds.count("total_frames") == 1 and "screen" not in kinds
    short = deepcopy(REFERENCE)
    short["shots"][0]["frame_count"] -= 1
    broken = requirement_retention(gold, parse_event_log(short), labels, align_entities(gold, parse_event_log(short), set()),
                                   {COSTUME: True}, request(REFERENCE))
    assert not broken["perfect"]
    assert [item["kind"] for item in broken["requirements"] if not item["satisfied"] and item["kind"] in ("total_frames", "locked_shot")] \
        == ["total_frames", "locked_shot"]



def with_evidence_on_unknown(log, evidence="observed"):
    """显式改动：给日志里每个值为 unknown 的事实补上 evidence 字段，复现链路验证里被 schema 层拒收的写法。"""
    data = deepcopy(log)
    facts = list(data["initial_state"])
    facts += [f for shot in data["shots"] for f in shot["screen_facts"]]
    facts += [f for e in data["events"] for f in (e["before"], e["after"]) if f is not None]
    unknown = [f for f in facts if f["object_or_value"] == "unknown"]
    for fact in unknown:
        fact["evidence"] = evidence
    return data, len(unknown)


def test_real_log_with_evidence_on_unknown_values_passes_schema_but_other_rules_still_reject():
    from mspc.data.event_log import EventLogError
    data, count = with_evidence_on_unknown(REAL)
    # 04 号窗只有 initial_state 里的 unknown；再补一条真实失败形状的屏幕事实（SFT 答复里的 gaze unknown＋observed）。
    data["shots"][0]["screen_facts"].append({"subject": "Subject 1", "family": "screen", "predicate_or_axis": "gaze",
        "object_or_value": "unknown", "scope": "screen@0", "shot_span": [0, 0], "evidence": "observed"})
    with pytest.raises(EventLogError):
        parse_event_log(data)  # 直接解析仍然拒收：归一只在规划器答复入口
    for evidence in ("observed", "unknown"):
        data, _ = with_evidence_on_unknown(data, evidence)
        result = compile_layers(json.dumps(data), request(), token_counter=words)
        assert result["failed_layer"] is None and "schema" in result["passed"] and "compile" in result["passed"], result["issues"]
        assert all("evidence" not in f for f in result["event_log"]["initial_state"] if f["object_or_value"] == "unknown")
    # 其余拒收规则不受影响：已知值缺 evidence 仍在 schema 层被拒；空事件仍在 structure 层被拒。
    known = deepcopy(REAL)
    known["initial_state"][0].pop("evidence")
    assert compile_layers(json.dumps(known), request(), token_counter=words)["failed_layer"] == "schema"
    noop = deepcopy(REFERENCE)
    noop["events"][0]["after"] = deepcopy(noop["events"][0]["before"])
    assert compile_layers(json.dumps(noop), request(REFERENCE), token_counter=words)["failed_layer"] == "structure"
    assert count >= 1
