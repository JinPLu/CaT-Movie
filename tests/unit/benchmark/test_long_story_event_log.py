"""C9：评测剧本确定性转 event_log_v3，与 lsh-001-b、lsh-006-b 手工日志逐字段对拍。

手工日志（segment-inference-20261008 的 inputs/prepare.py 产出，已随共享改动 X 迁到 v3）用的是修订 3 之前
15 fps 的帧数；转换按合同改用修订 3 冻结帧数，所以对拍时 frame_count 单独与 r3 比，其余字段逐字相等。
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mspc.benchmark.long_story_event_log import change_values, event_log_from_script, r3_frame_counts
from tests.support.paths import REPO_ROOT

PAIRS = REPO_ROOT / "tests/fixtures/long_story_event_log/dev30_b_lsh001_lsh006.json"
HAND = REPO_ROOT / "tests/fixtures/segment_chain_real_failure/lsh_segment_fail.json"
CANDIDATE = REPO_ROOT / "configs/benchmark/script_state_bench_long_holdout_v2_candidate_r3.json"


@pytest.mark.parametrize("story", ["lsh-001", "lsh-006"])
def test_conversion_equals_hand_log_field_by_field_with_r3_frame_counts(story):
    pair = next(p for p in json.loads(PAIRS.read_text())["pairs"] if p["story_id"] == story)
    hand = next(c for c in json.loads(HAND.read_text())["cases"] if c["case_id"] == f"{story}-b")["event_log"]
    counts = r3_frame_counts(json.loads(CANDIDATE.read_text()), story)
    log = event_log_from_script(pair["context_b"], pair["change"], counts)
    assert [s["frame_count"] for s in log["shots"]] == counts
    expected = deepcopy(hand)
    for shot, count in zip(expected["shots"], counts):
        shot["frame_count"] = count
    assert log == expected


@pytest.mark.parametrize("prior,post,expected", [
    ("wearing the checkered cap", "bareheaded", ("costume", "the checkered cap", "no cap")),
    ("not wearing sunglasses", "wearing dark sunglasses", ("costume", "no sunglasses", "dark sunglasses")),
    ("wearing the dark vest", "dark vest removed, white shirt exposed", ("costume", "the dark vest", "no vest")),
    ("holding the handgun", "empty-handed, handgun released onto the floor",
     ("body", "holding the handgun", "empty-handed, handgun released onto the floor")),
    ("hair tied in a ponytail", "hair loose with ponytail undone",
     ("costume", "hair tied in a ponytail", "hair loose with ponytail undone")),
])
def test_change_value_rule_on_real_dev30_change_records(prior, post, expected):
    assert change_values({"prior": prior, "post": post}) == expected


def test_frame_count_mismatch_and_bad_event_shot_raise():
    pair = json.loads(PAIRS.read_text())["pairs"][0]
    with pytest.raises(ValueError, match="frozen frame counts"):
        event_log_from_script(pair["context_b"], pair["change"], [37])
    counts = r3_frame_counts(json.loads(CANDIDATE.read_text()), pair["story_id"])
    with pytest.raises(ValueError, match="event_shot"):
        event_log_from_script(pair["context_b"], dict(pair["change"], event_shot=99), counts)


STATE_IN_APPEARANCE = REPO_ROOT / "tests/fixtures/long_story_event_log/dev30_b_state_in_appearance.json"


@pytest.mark.parametrize("story,subject,appearance,state,span", [
    ("lsh-010", "Subject 1", "A group of soldiers wearing green camouflage uniforms and helmets.", "holding rifles", [0, 0]),
    ("lsh-020", "Subject 9", "Woman with short grey-brown hair and a white apron.", "standing", [0, 0]),
    ("lsh-024", "Subject 3", "Male, older, grey hair, wearing a maroon robe.", "seated in a wheelchair", [2, 2]),
    ("lsh-024", "Subject 6", "Female, dark hair, wearing a dark jacket.", "holding a child", [2, 2]),
    ("lsh-028", "Subject 4", "A white man with short brown hair, wearing a light grey t-shirt.",
     "standing slightly behind Subject 2", [1, 1]),
    ("lsh-029", "Subject 5", "A man, wearing a white shirt, dark trousers, and suspenders.",
     "standing in the background", [0, 0]),
])
def test_state_words_leave_the_appearance_for_a_first_shot_initial_state(story, subject, appearance, state, span):
    pair = next(p for p in json.loads(STATE_IN_APPEARANCE.read_text())["pairs"] if p["story_id"] == story)
    log = event_log_from_script(pair["context_b"], pair["change"],
                                r3_frame_counts(json.loads(CANDIDATE.read_text()), story))
    assert next(s for s in log["subjects"] if s["subject_id"] == subject)["appearance"] == appearance
    fact = next(f for f in log["initial_state"] if f["subject"] == subject)
    assert (fact["predicate_or_axis"], fact["object_or_value"], fact["shot_span"]) == ("body", state, span)


@pytest.mark.parametrize("story,state,prior,post", [
    ("lsh-020", "seated", "holding a white teacup", "empty-handed with the teacup resting on the table"),
    ("lsh-023", "standing on four legs", "mouth empty", "holding a large bundle of straw in its mouth"),
])
def test_posture_of_the_changed_subject_joins_both_ends_of_the_body_change(story, state, prior, post):
    pair = next(p for p in json.loads(STATE_IN_APPEARANCE.read_text())["pairs"] if p["story_id"] == story)
    log = event_log_from_script(pair["context_b"], pair["change"],
                                r3_frame_counts(json.loads(CANDIDATE.read_text()), story))
    event = log["events"][0]
    assert event["before"]["object_or_value"] == f"{state}, {prior}"
    assert event["after"]["object_or_value"] == f"{state}, {post}"
    assert [f for f in log["initial_state"] if f["subject"] == pair["change"]["subject_id"]] == [event["before"]]


def test_state_word_list_has_one_owner_and_the_split_clears_it_from_every_dev30_subject():
    from mspc.benchmark import long_story_event_log
    from mspc.data import event_log

    assert long_story_event_log.MUTABLE_STATE_WORDS is event_log.MUTABLE_STATE_WORDS
    pairs = json.loads(STATE_IN_APPEARANCE.read_text())["pairs"]
    flagged = [s["attributes"] for p in pairs for s in p["context_b"]["subjects"] if event_log.MUTABLE_STATE_WORDS.search(s["attributes"])]
    assert len(flagged) >= 6  # 固件里这批真实描述原样会被规则 5 拒收
    for pair in pairs:
        log = event_log_from_script(pair["context_b"], pair["change"],
                                    r3_frame_counts(json.loads(CANDIDATE.read_text()), pair["story_id"]))
        assert not any(event_log.MUTABLE_STATE_WORDS.search(s["appearance"]) for s in log["subjects"])
