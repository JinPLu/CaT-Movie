"""真实阶段长行的同谓语去重；每个实体完整编号、来源span与用途保护。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.shot_text import shot_text_with_entity_spans, without_target_text


CASES = json.loads((Path(__file__).parents[2] / "fixtures/block_annotation_v18/stage-text-overflow.json").read_text())["cases"]


def _compile(case):
    annotation = without_target_text(case["annotation"], frozenset(case["omit_target_text"]))
    return shot_text_with_entity_spans(annotation, case["shot_index"], references=case["references"])



def test_real_walking_group_keeps_each_source_identity_and_the_three_distinct_positions():
    case = next(case for case in CASES if case["sample_index"] == 3052)
    compiled = _compile(case)
    text = compiled["text"]
    clause = "Character 1, Character 2, Character 6, Character 7, Character 8, Character 9, Character 10, Character 11 and Character 12 are walking towards the camera."
    assert clause in text
    left = text.index(clause)
    members = [span for span in compiled["entity_spans"] if left <= span["start"] < left + len(clause)]
    assert [(span["name"], text[span["start"]:span["end"]], span["source"]) for span in members] == [
        (f"Character {number}", f"Character {number}", f"shots[2].present[{number - 1}].name")
        for number in (1, 2, 6, 7, 8, 9, 10, 11, 12)
    ]
    for sentence in (
        "Character 3 is walking towards the camera to the left of Character 1.",
        "Character 4 is walking towards the camera behind Character 2.",
        "Character 5 is walking towards the camera behind Character 3.",
    ):
        assert sentence in text


def test_real_desk_registers_every_present_object_and_keeps_static_placements():
    case = next(case for case in CASES if case["sample_index"] == 42073)
    compiled = _compile(case)
    text = compiled["text"]
    registry = compiled["text_sections"]["registry"]
    for line in ("Object 6: tissue box with yellow and blue stripes.", "Object 7: gold vase with red flowers.",
                 "Object 9: framed portraits of children on the wall."):
        assert line in registry
    assert "Object 1, Object 2, Object 5, Object 6 and Object 7 are on the desk." in text
    assert {"Object 6", "Object 7", "Object 9"} <= {span["name"] for span in compiled["entity_spans"]}
    assert "Character 1 sits on Object 10." in text


def test_real_partial_clothing_repeat_retains_the_complete_outfit_and_ambiguous_conjunction():
    case = next(case for case in CASES if case["sample_index"] == 70110)
    compiled = _compile(case)
    text = compiled["text"]
    first = text.split(" Character 2:", 1)[0]
    assert first == "Character 1: elderly man with a grey beard, wearing grey jacket, yellow trousers, a dark hat."
    registry_six = text.split("Character 6: ", 1)[1].split(" Character 7: ", 1)[0]
    assert all(fact in registry_six for fact in ("dark coat", "white cravat", "dark hat"))
    assert "wearing a dark hat and coat" not in registry_six  # 两件dark衣物均在稳定outfit，才可去整段重复。
    outfit = [span for span in compiled["entity_spans"] if span["source"] == "characters[0].outfit"]
    assert len(outfit) == 1
    assert text[outfit[0]["start"]:outfit[0]["end"]] == "wearing grey jacket, yellow trousers, a dark hat"


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case['split']}-{case['sample_index']}")
def test_actual_stage_entities_keep_full_labels_and_nonoverlapping_source_spans(case):
    before = copy.deepcopy(case["annotation"])
    compiled = _compile(case)
    text = compiled["text"]
    assert case["annotation"] == before
    labels = [span for span in compiled["entity_spans"] if span["source"].endswith(".name")]
    assert labels and all(text[span["start"]:span["end"]] == span["name"] for span in labels)
    assert set(compiled["text_sections"]) == {"registry", "style", "camera", "opening", "action", "ending"}
    spans = sorted(compiled["entity_spans"], key=lambda span: (span["start"], span["end"]))
    assert all(0 <= span["start"] < span["end"] <= len(text) for span in spans)
    for index, first in enumerate(spans):
        for second in spans[index + 1:]:
            if first["start"] < second["end"] and second["start"] < first["end"]:
                assert (first["name"], first["start"], first["end"]) == (second["name"], second["start"], second["end"])


def test_real_many_person_registry_keeps_multidigit_names_and_literal_hat_text_separate():
    case = next(case for case in CASES if case["sample_index"] == 82142)
    before = copy.deepcopy(case["annotation"])
    compiled = _compile(case)
    text = compiled["text"]
    names = [(span["name"], text[span["start"]:span["end"]]) for span in compiled["entity_spans"]
             if span["source"].startswith(("characters[", "objects[")) and span["source"].endswith(".name")]
    assert names == [(f"Character {number}", f"Character {number}") for number in range(1, 15)] + [
        (f"Object {number}", f"Object {number}") for number in range(1, 9)]
    assert "Character 14: " in text and "Object 8: " in text
    assert "yellow crown hat with '2000' written on it" in text
    assert not any("2000" in name for name, _ in names)
    for wearer, hat in ((3, 5), (2, 6), (4, 7), (7, 8)):
        assert any(sentence.startswith(f"Character {wearer} ") and f"wearing Object {hat}" in sentence
                   for sentence in text.split(". "))
    assert case["annotation"] == before


def test_real_partial_stable_clothes_remove_only_the_fully_retained_tail_facts():
    case = next(case for case in CASES if case["split"] == "validation")
    text = _compile(case)["text"]
    assert "1: man with dark hair, wearing light blue button-down shirt, dark pants." in text
    assert "3: man, wearing dark police uniform with a tie and badge." in text
    assert "4: man with glasses, wearing light grey polo shirt with dark stripes and light pants." in text
    assert "6: woman, wearing dark police uniform with a tie and badge." in text


def test_real_spiderverse_keeps_registered_features_without_a_reference_mapping_tail():
    case = next(case for case in CASES if case["sample_index"] == 93281)
    compiled = _compile(case)
    text = compiled["text"]
    assert "4: man with dark hair and a beard, wearing long coat over a red suit." in text
    assert "7: woman with dark skin and an afro hairstyle, wearing patterned outfit with a bag." in text
    assert "black suit with a red spider emblem on the chest and back" in text
    assert text.endswith(compiled["text_sections"]["ending"] or compiled["text_sections"]["action"])
