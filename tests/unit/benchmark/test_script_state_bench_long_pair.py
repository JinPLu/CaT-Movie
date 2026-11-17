"""从已看帧的 lsh-049 餐桌窗构造 A/B：验证等镜长与注入/继承的接口。

帧区间来自候选原片；负例只改配对字段。结构测试不充当 A/B 视觉验收。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mspc.benchmark.script_state_bench import (
    SCRIPT_STATE_BENCH_LONG_A_SCHEMA as A_SCHEMA,
    SCRIPT_STATE_BENCH_LONG_B_SCHEMA as B_SCHEMA,
    SOURCE_FIELDS,
    derive_cut_type,
    derive_tags,
    roster_prompt,
    shot_prompt_head,
    validate_script_state_bench,
)

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ["a reference benchmark prompt from an unrelated scene"]


def _long_pair() -> tuple[dict, dict]:
    source = next(row for row in json.loads((ROOT / "configs/benchmark/script_state_bench_long_holdout_v2_candidate_r3.json").read_text())["stories"] if row["story_id"] == "lsh-049")
    subjects = [
        {"subject_id": "Subject 1", "kind": "person", "attributes": "curly-haired woman in a white blouse and pearl necklace"},
        {"subject_id": "Subject 2", "kind": "person", "attributes": "bald man with glasses in a dark suit and striped tie"},
        {"subject_id": "Subject 3", "kind": "person", "attributes": "dark-haired woman in a red jacket"},
        {"subject_id": "Subject 4", "kind": "person", "attributes": "dark-haired man in a gray jacket"},
    ]
    location = "a wood-paneled dining room with a candlelit dinner table"
    # 可见主要人物；边缘背景宾客不是本 fixture 的身份跟踪主体。
    casts = [[1, 2, 3, 4], [2, 3, 4], [1], [2, 3, 4], [3, 4], [1], [2], [1, 2, 3, 4], [2, 3, 4], [1], [2, 3, 4], [1]]
    shots = []
    for index, (original, cast) in enumerate(zip(source["shots"], casts, strict=True)):
        shot = {"shot_id": f"s{index + 1:02d}", "location_id": "Location 1", "present_subjects": [f"Subject {value}" for value in cast], "frame_count": original["render_frames"], "shot_size": "Medium shot", "camera_angle": "Eye-level shot", "blocking": None, "facing": f"Subject {cast[0]}", "source_span": [original["start_frame"], original["end_frame"]]}
        shots.append(shot)
        cut = None
        if index:
            cut, returns_to = derive_cut_type(shots, index)
            shot["cut_type"] = cut
            if returns_to is not None:
                shot["returns_to"] = returns_to
        cast_text = " and ".join(shot["present_subjects"])
        shot["prompt"] = shot_prompt_head(shot, location, cut) + f"{cast_text} converse beside the flowers and tableware while the dinner conversation continues during moment {index + 1}."
    a = {"schema_version": A_SCHEMA, "case_id": "lsh-049-a", "shot_count": 12, "story_prompt": roster_prompt(subjects), "subjects": subjects, "locations": [{"location_id": "Location 1", "attributes": location}], "shots": shots, "events": [], "tags": [], "source": {key: source[key] for key in SOURCE_FIELDS}}
    a["tags"] = derive_tags(a)
    b = copy.deepcopy(a)
    b.update(schema_version=B_SCHEMA, case_id="lsh-049-b", derived_from=a["case_id"], injected_event_ids=["e1"])
    del b["source"]
    for shot in b["shots"]:
        del shot["source_span"]
    b["events"] = [{"event_id": "e1", "shot_id": "s03", "subject_id": "Subject 1", "prior_state": "wearing a red heart badge", "post_state": "without the red heart badge"}]
    b["shots"][2]["prompt"] += " Subject 1 removes the red heart badge."
    b["tags"] = derive_tags(b)
    def manifest(case, schema):
        return {"schema_version": schema, "benchmark_id": schema, "evaluation_only": True, "fps": 16, "generation_seeds": [42], "cases": [case]}
    return manifest(a, A_SCHEMA), manifest(b, B_SCHEMA)


def _validate(a, b):
    return validate_script_state_bench(b, forbidden_prompts=REFERENCE, paired_a=a)


def test_original_short_event_duration_survives_pair_validation():
    a, b = _long_pair()
    assert b["cases"][0]["shots"][2]["frame_count"] == 25
    assert _validate(a, b) is b


def test_b_without_a_cannot_be_validated():
    _, b = _long_pair()
    with pytest.raises(ValueError, match="requires paired_a"):
        validate_script_state_bench(b, forbidden_prompts=REFERENCE)


def test_b_cannot_extend_event_shot_to_81_frames():
    a, b = _long_pair()
    b["cases"][0]["shots"][2]["frame_count"] = 81
    with pytest.raises(ValueError, match="frame_count must match"):
        _validate(a, b)


def test_pair_binding_is_checked():
    a, b = _long_pair()
    b["cases"][0]["derived_from"] = "another-story"
    with pytest.raises(ValueError, match="derived_from"):
        _validate(a, b)


@pytest.mark.parametrize("ids", [[], ["e1", "e1"], ["e1", "e2", "e3"], ["e9"]])
def test_injection_ids_are_explicit_b_event_references(ids):
    a, b = _long_pair()
    b["cases"][0]["injected_event_ids"] = ids
    with pytest.raises(ValueError, match="injected_event_ids"):
        _validate(a, b)


def test_inherited_events_do_not_count_towards_two_injections():
    a, b = _long_pair()
    inherited = [
        {"shot_id": "s01", "subject_id": "Subject 2", "prior_state": "standing beside the chair", "post_state": "seated at the table"},
        {"shot_id": "s02", "subject_id": "Subject 3", "prior_state": "holding a dinner napkin", "post_state": "without the dinner napkin"},
        {"shot_id": "s04", "subject_id": "Subject 4", "prior_state": "holding a glass", "post_state": "without the glass"},
    ]
    a["cases"][0]["events"] = [{"event_id": f"e{i+1}", **event} for i, event in enumerate(inherited)]
    a["cases"][0]["tags"] = derive_tags(a["cases"][0])
    events = inherited + b["cases"][0]["events"] + [{"shot_id": "s06", "subject_id": "Subject 1", "prior_state": "without the red heart badge", "post_state": "wearing a red heart badge"}]
    events.sort(key=lambda row: row["shot_id"])
    b["cases"][0]["events"] = [{**event, "event_id": f"e{i+1}"} for i, event in enumerate(events)]
    b["cases"][0]["injected_event_ids"] = ["e3", "e5"]
    b["cases"][0]["shots"][5]["prompt"] += " Subject 1 attaches the red heart badge."
    b["cases"][0]["tags"] = derive_tags(b["cases"][0])
    assert len(b["cases"][0]["events"]) == 5
    _validate(a, b)
    b["cases"][0]["events"][0]["post_state"] = "leaning against the table"
    with pytest.raises(ValueError, match="inherited A event"):
        _validate(a, b)


def test_b_does_not_restate_post_state_in_later_prompt():
    a, b = _long_pair()
    b["cases"][0]["shots"][5]["prompt"] += " Subject 1 is without the red heart badge."
    with pytest.raises(ValueError, match="only injected event-shot prompts"):
        _validate(a, b)


def test_b_cannot_drop_a_shot():
    a, b = _long_pair()
    case = b["cases"][0]
    case["shots"].pop()
    case["shot_count"] -= 1
    case["tags"] = derive_tags(case)
    with pytest.raises(ValueError, match="shot_count must match"):
        _validate(a, b)
