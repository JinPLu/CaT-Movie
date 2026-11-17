"""依赖分段与按调用方给定参考帧编译：夹具来自 lsh-001-b / lsh-006-b 的真实失败日志。"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mspc.data.event_log import (
    EventLogError, FrameEvidence, compile_event_log, dependency_segment_spans, fact_key,
    parse_event_log,
)
from mspc.data.state_facts import AtomicFact
from tests.support.paths import REPO_ROOT

# 固件里的 schema 字符串已按 event_log_v3 改写（2026-10-09），非逐字原文；其余内容未动。
FIXTURE = REPO_ROOT / "tests/fixtures/segment_chain_real_failure/lsh_segment_fail.json"


def cases():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {case["case_id"]: case for case in payload["cases"]}


def frames(case):
    evidence = []
    for row in case["observations"]:
        facts = [AtomicFact.model_validate(fact) for fact in row["result_states"]]
        evidence.append(FrameEvidence(
            row["shot_id"], "last", row["accepted"], True,
            {fact_key(fact): fact for fact in facts}, frozenset(row["visible_subjects"]),
        ))
    return evidence


def test_real_failures_pack_in_order_without_cutting_on_location():
    loaded = cases()
    first = parse_event_log(loaded["lsh-001-b"]["event_log"])
    second = parse_event_log(loaded["lsh-006-b"]["event_log"])
    # lsh-001-b 地点交替，事件与结果同在 s01。贪心装成 s01–s04（168 帧）和 s05–s06（150 帧）。
    assert dependency_segment_spans(first) == ((0, 4), (4, 6))
    assert [[shot.shot_id for shot in first.shots[start:end]]
            for start, end in dependency_segment_spans(first)] == [
                ["s01", "s02", "s03", "s04"], ["s05", "s06"]]
    # lsh-006-b 的 s10 换地点，但 17+33+73 帧、3 镜装得进，所以 s09–s11 同段。
    assert dependency_segment_spans(second) == ((0, 4), (4, 8), (8, 11))


def test_compiler_describes_exactly_the_given_real_observations():
    case = cases()["lsh-001-b"]
    log = parse_event_log(case["event_log"])
    observed = {frame.shot_id: frame for frame in frames(case)}
    (start, end) = dependency_segment_spans(log)[1]
    given = [observed["s04"], observed["s02"]]
    compiled = compile_event_log(log, span=(start, end), references=given,
        token_counter=lambda text: len(text.split()))
    assert compiled.span == (4, 6)
    assert compiled.reference_shot_ids == ("s04", "s02")
    assert [row.split(" Design: ")[0].split("shot. ")[1] for row in compiled.reference_rows] == [
        "Reference 1, the last frame of shot 4.", "Reference 2, the last frame of shot 2."]
    assert all(row.endswith("State: Subject 1 wears no cap.") for row in compiled.reference_rows)
    assert compiled.target_rows[1].endswith("Reference 2 is the last frame of shot 2; take Subject 4 appearance "
        "from it; possession, location and object condition follow the current text.")
    # 前段漏画不再由编译器记账；没给的帧不出现。
    assert "shot 1" not in " ".join(compiled.prompts)
    # 参考必须早于本段，不能取本段或之后的镜。
    with pytest.raises(EventLogError, match="earlier shot"):
        compile_event_log(log, span=(start, end), references=[observed["s01"], FrameEvidence(
            "s05", "last", True, True, {}, frozenset())], token_counter=lambda text: len(text.split()))


def test_event_and_result_stay_in_one_segment_when_the_change_is_seen_later():
    case = deepcopy(cases()["lsh-006-b"])
    log = case["event_log"]
    event = log["events"][0]
    event["shot_id"] = "s08"
    event["on_screen"] = "unknown"
    by_id = {shot["shot_id"]: shot for shot in log["shots"]}
    by_id["s08"]["visible_entities"] = ["Subject 2"]
    by_id["s08"]["present_subjects"] = ["Subject 1", "Subject 2"]
    by_id["s09"]["visible_entities"] = ["Subject 1", "Subject 2"]
    by_id["s09"]["present_subjects"] = ["Subject 1", "Subject 2"]
    spans = dependency_segment_spans(parse_event_log(log))
    together = [span for span in spans if 7 in range(*span) or 8 in range(*span)]
    assert together == [(7, 11)]
    assert (4, 7) in spans


def test_cross_location_event_and_result_stay_together_when_they_fit():
    case = deepcopy(cases()["lsh-006-b"])
    log = case["event_log"]
    event = log["events"][0]
    event["shot_id"] = "s09"
    event["on_screen"] = "unknown"
    by_id = {shot["shot_id"]: shot for shot in log["shots"]}
    by_id["s09"]["visible_entities"] = ["Subject 2"]
    by_id["s09"]["present_subjects"] = ["Subject 1", "Subject 2"]
    by_id["s10"]["visible_entities"] = ["Subject 1", "Subject 3"]
    by_id["s10"]["present_subjects"] = ["Subject 1", "Subject 3"]
    spans = dependency_segment_spans(parse_event_log(log))
    assert [span for span in spans if 8 in range(*span)] == [(8, 11)]


def test_atomic_event_result_block_over_the_frame_cap_is_rejected():
    case = deepcopy(cases()["lsh-006-b"])
    log = case["event_log"]
    event = log["events"][0]
    event["shot_id"] = "s09"
    event["on_screen"] = "unknown"
    by_id = {shot["shot_id"]: shot for shot in log["shots"]}
    by_id["s09"]["visible_entities"] = ["Subject 2"]
    by_id["s09"]["present_subjects"] = ["Subject 1", "Subject 2"]
    by_id["s09"]["frame_count"] = 120
    by_id["s10"]["visible_entities"] = ["Subject 1", "Subject 3"]
    by_id["s10"]["present_subjects"] = ["Subject 1", "Subject 3"]
    by_id["s10"]["frame_count"] = 120
    with pytest.raises(EventLogError, match="依赖段无法装入"):
        dependency_segment_spans(parse_event_log(log))
