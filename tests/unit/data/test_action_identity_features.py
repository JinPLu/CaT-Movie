"""真实动作执行者使用一个完整且同镜排他的已有属性；未知时保完整事实。"""
import copy
import json
import re
from pathlib import Path

import pytest

from mspc.data.shot_text import _opening_shot, _short_identity_choices, shot_text_with_entity_spans, without_target_text, resolve_action_references


ROOT = Path(__file__).parents[2] / "fixtures/block_annotation_v18"
STAGE = json.loads((ROOT / "stage-text-overflow.json").read_text())["cases"]
OVERFLOW = json.loads((ROOT / "action-identity-overflow.json").read_text())["cases"]


def _annotation(case):
    if "annotation" in case:
        annotation = case["annotation"]
    else:
        reference = case["annotation_fixture"]
        annotation = next(row["annotation"] for row in json.loads((ROOT / reference["path"]).read_text())["cases"]
                          if row["case_id"] == reference["case_id"])
    omitted = {name for key, name in (("layout_text_dropped", "layouts"), ("picture_type_text_dropped", "picture_type"))
               if case.get("metadata", {}).get(key)}
    return without_target_text(annotation, frozenset(omitted))


def _choices(annotation, index):
    shot, sources, _ = _opening_shot(annotation, index)
    return _short_identity_choices(annotation, shot, sources, f"shots[{index}]")


def test_real_spiderverse_executor_uses_skin_without_repeating_the_second_hair_feature():
    case = next(case for case in OVERFLOW if case["sample_index"] == 93281)
    choices = _choices(_annotation(case), case["shot_index"])
    assert choices["Character 1"][0].text == "the young man with dark skin"
    assert choices["Character 1"][1] == "unique_feature"


def test_real_shared_dark_hair_selects_the_given_current_shirt_instead_of_an_ambiguous_trait():
    case = next(case for case in OVERFLOW if case["split"] == "validation")
    choices = _choices(_annotation(case), case["shot_index"])
    assert choices["Character 1"][0].text == "the man wearing a light blue button-down shirt"
    assert choices["Character 2"][0].text == "the man with a beard"
    assert choices["Character 5"][0].text == "the man with a mustache"


def test_real_dark_spiky_hair_covers_dark_hair_so_the_given_glasses_identify_the_boy():
    case = next(case for case in OVERFLOW if case["sample_index"] == 6)
    choices = _choices(_annotation(case), case["shot_index"])
    assert choices["Character 4"][0].text == "the young boy with glasses"


def test_real_identical_people_keep_their_whole_facts_without_a_fabricated_unique_attribute():
    case = next(case for case in STAGE if case["sample_index"] == 70110)
    choices = _choices(case["annotation"], case["shot_index"])
    for name in ("Character 5", "Character 7", "Character 8"):
        assert choices[name][1] == "short_shared_feature"
        assert choices[name][0].text == "the man with a mustache"
        assert "flat cap" in shot_text_with_entity_spans(case["annotation"], case["shot_index"])["text"]
    assert choices["Character 6"][0].text == "the man wearing a dark coat"


def test_real_comma_separated_hair_adjectives_do_not_become_a_bare_long_feature():
    stories = json.loads((ROOT.parents[2] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    annotation = next(row["storyboard"] for row in stories if row["story_id"] == "film-003551-b0002-shot05")
    choices = _choices(annotation, 0)
    assert choices["Character 1"][0].text == "the woman with long, wavy reddish-brown hair"


@pytest.mark.parametrize("case", OVERFLOW, ids=lambda row: f"{row['split']}-{row['sample_index']}")
def test_action_identity_choices_keep_full_group_facts_in_the_fourteen_actual_overflow_rows(case):
    annotation = _annotation(case)
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, case["shot_index"], references=case["references"])
    text = compiled["text"]
    assert annotation == before
    action = annotation["shots"][case["shot_index"]]["action"]
    if "Character " not in action:
        assert resolve_action_references(annotation, case["shot_index"])["action"] in text
    for name, (identity, why) in _choices(annotation, case["shot_index"]).items():
        character = next(row for row in annotation["characters"] if row["name"] == name)
        if character["kind"] == "group":
            assert why == "group_preserved"
            assert character["appearance"].removeprefix("a ").removeprefix("an ") in text
            assert identity.text.startswith("the ")
