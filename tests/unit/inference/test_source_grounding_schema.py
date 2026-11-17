"""真实源读取坏字段经过生成语法约束；视觉正确性不在此测试范围。"""
from copy import deepcopy
import json

import pytest

from mspc.inference.reference_pointing import grounded_reference_entities, source_grounding_schema
from tests.support.paths import REPO_ROOT

JsonSchemaParser = pytest.importorskip("lmformatenforcer").JsonSchemaParser
ANNOTATION = json.loads((REPO_ROOT / "tests/fixtures/history_reading_labels/film-006492-b0000.json").read_text())["annotation"]
REFERENCES = [[1, 12]]


def accepted(value, *, raw=False):
    parser = JsonSchemaParser(source_grounding_schema(ANNOTATION, REFERENCES))
    text = value if raw else json.dumps(value)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def answer():
    return {"references": [{"reference": 1, "characters": [
        {"name": "Character 1", "point_2d": [450, 500]},
        {"name": "Character 4", "point_2d": None}], "unknown_characters": ["Character 2"]}]}


def test_observed_prose_missing_field_unknown_objects_and_null_array_are_blocked():
    valid = answer()
    assert accepted(valid)
    assert not accepted("Based on the images: " + json.dumps(valid), raw=True)
    assert not accepted(json.dumps(valid)[:-1], raw=True)
    missing = deepcopy(valid)
    del missing["references"][0]["unknown_characters"]
    assert not accepted(missing)
    objects = deepcopy(valid)
    objects["references"][0]["unknown_characters"] = [{"name": "Character 2", "point_2d": [450, 500]}]
    assert not accepted(objects)
    null_array = deepcopy(valid)
    null_array["references"][0]["characters"][1]["point_2d"] = [None, None]
    assert not accepted(null_array)


def test_names_group_null_coordinate_range_and_reference_count_are_constrained():
    valid = answer()
    valid["references"][0]["characters"][0]["point_2d"] = [0, 1000]
    assert accepted(valid)
    wrong = deepcopy(valid)
    wrong["references"][0]["characters"][0]["point_2d"] = [0, 1001]
    assert not accepted(wrong)
    wrong = deepcopy(valid)
    wrong["references"][0]["characters"][0]["name"] = "Character 99"
    assert not accepted(wrong)
    wrong = deepcopy(valid)
    wrong["references"][0]["characters"][1]["point_2d"] = [500, 500]
    assert not accepted(wrong)
    wrong = deepcopy(valid)
    wrong["references"].append(deepcopy(wrong["references"][0]))
    assert not accepted(wrong)


def test_cross_list_identity_conflict_is_resolved_by_the_consumer_as_ambiguous():
    conflict = answer()
    conflict["references"][0]["unknown_characters"] = ["Character 1"]
    assert accepted(conflict)
    row, = grounded_reference_entities(conflict, annotation=ANNOTATION, references=REFERENCES)
    assert [entity["name"] for entity in row["entities"]] == ["Character 4"]
    assert row["unknown_characters"] == ["Character 1"] and row["ambiguous_characters"] == ["Character 1"]
