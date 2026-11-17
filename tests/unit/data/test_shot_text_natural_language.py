"""真实训练块暴露的自然语言重复、主体绑定与首次可见状态回归。"""
import copy
import json
import re
from pathlib import Path

import pytest

from mspc.data.shot_text import shot_text_with_entity_spans


CASES = {row["key"]: row for row in json.loads((
    Path(__file__).parents[2] / "fixtures/block_annotation_v18/natural-text-failures.json"
).read_text())["cases"]}


def _compile(key):
    row = CASES[key]
    return row, shot_text_with_entity_spans(row["annotation"], row["shot_index"])


def _action(row, compiled):
    source = f"shots[{row['shot_index']}].action"
    start = min(span["start"] for span in compiled["entity_spans"] if span["source"] == source)
    return compiled["text"][start:].split("By the end of the shot, ", 1)[0].strip()


@pytest.mark.parametrize("story_id,index,names", [
    ("cinedance-014083-b0001-shot01", 5, ("Character 1", "Character 5")),
    ("film-004441-b0001-shot01", 5, ("Character 1", "Character 4")),
    ("film-007998-b0006-shot01", 11, ("Character 1", "Character 3")),
    ("film-004064-b0008-shot01", 6, ("Character 1", "Character 3", "Character 4")),
])
def test_real_compound_subjects_keep_identity_with_base_form_action_verbs(story_id, index, names):
    stories = json.loads((Path(__file__).parents[3] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    annotation = next(row["storyboard"] for row in stories if row["story_id"] == story_id)
    compiled = shot_text_with_entity_spans(annotation, index)
    action = _action({"shot_index": index}, compiled)
    for name in names:
        assert name + ", " in action
    assert ",," not in action and ",." not in action


def test_real_rabbits_register_existing_clothes_once():
    _, compiled = _compile("panel-125")
    registry = compiled["text"].split("Setting: ", 1)[0]
    assert registry.count("wearing blue denim jacket") == 1
    assert registry.count("wearing grey shirt and dark green jacket") == 1
    assert "2: rabbit" in registry and "3: rabbit" in registry


def test_real_action_binds_its_subject_and_preserves_plain_object_ids():
    row, compiled = _compile("panel-021")
    action = _action(row, compiled)
    assert action.startswith("Character 3, ")
    assert " speaks while looking towards Character 1 and Character 2." in action
    assert ",." not in action and ",," not in action


def test_real_compound_and_subordinate_subjects_keep_the_possessive_owner():
    row, compiled = _compile("panel-026")
    action = _action(row, compiled)
    assert action.startswith("Character 1, ")
    assert "and Character 6, " in action
    assert "while Character 5, " in action
    assert " touches Character 4's arm." in action
    assert ",." not in action and ",," not in action


def test_real_passive_clause_does_not_treat_the_object_before_as_as_a_subject():
    row = {**CASES["panel-091"], "shot_index": 4}
    compiled = shot_text_with_entity_spans(row["annotation"], row["shot_index"])
    action = _action(row, compiled)
    assert action.startswith("Character 2, ")
    assert "is restrained by Character 3 and Character 4 as Character 5, the soldiers and people, around them raise their spears and shout." in action


def test_real_group_short_name_does_not_drop_its_second_member():
    row, compiled = _compile("panel-145")
    group = next(entry for entry in row["annotation"]["characters"] if entry["name"] == "Character 3")
    assert group["kind"] == "group"
    mention = _action(row, compiled).split("while Character 3", 1)[1].split(" stands", 1)[0]
    # 原ID已足够指组；若附短称，必须仍指向女人和男人两位成员。
    assert not mention or ("woman" in mention and "man" in mention)
    assert "man with glasses" in compiled["text"].split("Setting: ", 1)[0]


def test_real_color_conjunction_keeps_its_mane_noun():
    row, compiled = _compile("panel-188")
    action = _action(row, compiled)
    assert " talks, puts a hand to her head, and gestures with both hands." in action
    if "purple" in action:
        assert "purple and teal mane" in action or "dark purple wings" in action
    assert ",," not in action and ",." not in action


def test_real_existing_behind_relation_is_not_repeated_from_layout():
    _, compiled = _compile("panel-091")
    assert compiled["text"].count("behind Character 2") == 1


def test_real_absent_before_state_is_not_given_a_visible_layout():
    row, compiled = _compile("panel-193")
    opening = compiled["text"][:compiled["text"].index(_action(row, compiled))]
    absent_sentences = [sentence for sentence in opening.split(". ") if "not present" in sentence]
    assert absent_sentences
    assert all(not re.search(r"\b(?:behind|in front of|to the left|to the right|unconscious)\b", sentence)
               for sentence in absent_sentences)
    action = _action(row, compiled)
    assert "carries an unconscious Character 4 through the water past Character 1 and Character 2." in action


@pytest.mark.parametrize("key", list(CASES))
def test_real_single_setting_and_entity_spans_preserve_annotation(key):
    row = CASES[key]
    annotation = row["annotation"]
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, row["shot_index"])
    text = compiled["text"]
    assert len(annotation["places"]) == 1
    assert "Setting: " in text
    assert not re.search(r"\bLocation [1-9]\d*\b", text)
    names = {entry["name"] for kind in ("characters", "objects", "places") for entry in annotation[kind]}
    assert compiled["entity_spans"]
    for span in compiled["entity_spans"]:
        assert span["name"] in names
        assert 0 <= span["start"] < span["end"] <= len(text)
        assert text[span["start"]:span["end"]] and span["source"]
    assert annotation == before
