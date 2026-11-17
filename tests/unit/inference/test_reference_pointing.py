"""读历史第二步：在真实块 film-006492-b0000 第 3–4 镜上，提示里图片占位数＝参考数＋末帧一张、参考按标签时刻编号；
标签作答能过第二步的核对，其「名字＋镜首点」条目与目标镜在场的人一一对应（null 照写）；没有历史时没有图片。"""
import json

import pytest

from mspc.data.history_reading_labels import LabelRules, build_labels
from mspc.inference.planner_token_features import entry_spans
from mspc.inference.reference_pointing import (answer_from_labels, step_two_content, target_people,
                                               validate_answer)
from tests.support.paths import REPO_ROOT

CASE = json.loads((REPO_ROOT / "tests/fixtures/history_reading_labels/film-006492-b0000.json").read_text())
RULES = LabelRules(reference_min_area=0.02, weak_area=0.05, start_min_area=0.005, cut_margin=4,
                   max_references=6)


def _labels():
    return build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], CASE["span"], RULES)


def _content(labels):
    span = labels["span"]
    return step_two_content(CASE["annotation"], span=span, references=labels["moments"],
                            last_shot=max(labels["history"]) if labels["history"] else None,
                            target_prompts=[f"prompt {i}" for i in range(*span)], needed=labels["needed"])


def test_images_are_the_references_then_the_last_history_frame():
    labels = _labels()
    content = _content(labels)
    texts = [part["text"] for part in content if part["type"] == "text"]
    assert sum(part["type"] == "image" for part in content) == len(labels["moments"]) + 1
    for number, (shot, frame) in enumerate(labels["moments"], start=1):
        assert f"Reference {number} (Shot {shot + 1}, {frame / 16:.1f} s):" in texts
    assert f"Last frame of Shot {max(labels['history']) + 1}:" in texts
    assert "uncertain" not in texts[-1].lower()
    assert f"There are {len(labels['moments'])} reference images" in texts[-1]
    assert "is not a reference" in texts[-1]


def test_label_answer_passes_the_check_and_yields_one_entry_per_present_character():
    labels = _labels()
    answer = answer_from_labels(labels)
    targets = target_people(CASE["annotation"], labels["span"])
    _, dropped = validate_answer(json.loads(answer), references=len(labels["moments"]), needed=labels["needed"],
                                 targets=targets)
    assert dropped == []
    spans = [s for s in entry_spans(answer) if s[0] == "shot"]
    assert [(shot, name) for _, shot, name, _ in spans] == [(shot, n) for shot, names in sorted(targets.items())
                                                           for n in names]
    assert any(json.loads(answer[slice(*by["entry"])])["point_2d"] is None for *_, by in spans)


def test_no_history_means_no_images():
    labels = build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], [0, 1], RULES)
    content = _content(labels)
    assert not any(part["type"] == "image" for part in content)
    assert "There are no reference images" in content[-1]["text"]


def test_check_writes_a_missing_target_entry_as_null_and_records_it():
    labels = _labels()
    answer = json.loads(answer_from_labels(labels))
    missing = answer["shots"][0]["characters"][0]["name"]
    answer["shots"][0]["characters"] = answer["shots"][0]["characters"][1:]
    kept, dropped = validate_answer(answer, references=len(labels["moments"]), needed=labels["needed"],
                                    targets=target_people(CASE["annotation"], labels["span"]))
    assert kept["shots"][0]["characters"][0] == {"name": missing, "point_2d": None}
    assert [d["name"] for d in dropped] == [missing]


def test_check_drops_a_name_pointed_at_twice_in_one_reference():
    labels = _labels()
    answer = json.loads(answer_from_labels(labels))
    row = answer["references"][0]
    row["characters"] = row["characters"] + row["characters"][:1]
    twice = row["characters"][0]["name"]
    kept, dropped = validate_answer(answer, references=len(labels["moments"]), needed=labels["needed"],
                                    targets=target_people(CASE["annotation"], labels["span"]))
    assert twice not in [c["name"] for c in kept["references"][0]["characters"]]
    assert {d["reason"] for d in dropped} == {"named more than once"}


def test_check_drops_all_pointing_when_references_are_misnumbered():
    labels = _labels()
    answer = json.loads(answer_from_labels(labels))
    answer["references"] = answer["references"][::-1] if len(answer["references"]) > 1 else []
    kept, dropped = validate_answer(answer, references=len(labels["moments"]), needed=labels["needed"],
                                    targets=target_people(CASE["annotation"], labels["span"]))
    assert [r["characters"] for r in kept["references"]] == [[] for _ in labels["moments"]]
    assert dropped[0]["where"] == "references"
    assert kept["shots"] == answer["shots"]


def test_final_source_grounding_preserves_actual_moments_and_explicit_unknowns():
    from mspc.inference.reference_pointing import grounded_reference_entities
    from mspc.data.shot_text import reference_text_with_entity_spans
    labels = _labels()
    old = json.loads(answer_from_labels(labels))
    answer = {"references": [{**row, "unknown_characters": []} for row in old["references"]]}
    rows = grounded_reference_entities(answer, annotation=CASE["annotation"], references=labels["moments"])
    assert [(row["shot"] - 1, row["frame"]) for row in rows] == list(map(tuple, labels["moments"]))
    assert all(row["alignment_cells"] is None for row in rows)
    for row, original in zip(rows, old["references"]):
        assert {e["name"] for e in row["entities"]} == {e["name"] for e in original["characters"]}
    name = CASE["annotation"]["characters"][0]["name"]
    answer["references"][0] = {"reference": 1, "characters": [], "unknown_characters": [name]}
    rows = grounded_reference_entities(answer, annotation=CASE["annotation"], references=labels["moments"])
    assert rows[0]["entities"] == [] and rows[0]["unknown_characters"] == [name]
    assert reference_text_with_entity_spans(
        rows[0]["entities"], setting=None, characters=CASE["annotation"]["characters"])["entity_spans"] == []


def test_final_source_grounding_rejects_old_numbers_and_marks_conflicting_identity_ambiguous():
    from mspc.inference.reference_pointing import grounded_reference_entities
    labels = _labels()
    old = json.loads(answer_from_labels(labels))
    answer = {"references": [{**row, "unknown_characters": []} for row in old["references"]]}
    answer["references"][0]["reference"] = 2
    with pytest.raises(ValueError, match="exact final RGB"):
        grounded_reference_entities(answer, annotation=CASE["annotation"], references=labels["moments"])
    answer["references"][0]["reference"] = 1
    name = CASE["annotation"]["characters"][0]["name"]
    answer["references"][0]["characters"] = [{"name": name, "point_2d": None}]
    answer["references"][0]["unknown_characters"] = [name]
    row = grounded_reference_entities(answer, annotation=CASE["annotation"], references=labels["moments"])[0]
    assert name not in {entity["name"] for entity in row["entities"]}
    assert row["unknown_characters"] == [name] and row["ambiguous_characters"] == [name]
    answer["references"][0]["characters"] = [{"name": "Not registered", "point_2d": None}]
    answer["references"][0]["unknown_characters"] = []
    with pytest.raises(ValueError, match="registered name"):
        grounded_reference_entities(answer, annotation=CASE["annotation"], references=labels["moments"])


def test_one_name_given_to_two_people_keeps_the_reference_without_that_identity():
    from mspc.inference.reference_pointing import grounded_reference_entities
    # test92 cinedance-014481-b0000-shot01 g7 raw answer, 2026-11-03; two dark-haired passers-by both named Character 6.
    annotation = {"characters": [
        {"name": "Character 1", "kind": "person", "appearance": "a man with a beard and short dark hair"},
        {"name": "Character 2", "kind": "person", "appearance": "a bald man with a beard"},
        {"name": "Character 3", "kind": "person", "appearance": "a man with long wavy hair and a beard"},
        {"name": "Character 4", "kind": "person",
         "appearance": "a person wearing a black hooded garment and a black face mask"},
        {"name": "Character 5", "kind": "person", "appearance": "a woman with dark hair"},
        {"name": "Character 6", "kind": "person", "appearance": "a person with dark hair"}]}
    answer = {"references": [
        {"reference": 1, "characters": [{"name": "Character 5", "point_2d": [150, 550]},
                                        {"name": "Character 6", "point_2d": [440, 350]},
                                        {"name": "Character 6", "point_2d": [500, 350]}], "unknown_characters": []},
        {"reference": 2, "characters": [{"name": "Character 3", "point_2d": [250, 450]}], "unknown_characters": []}]}
    first, second = grounded_reference_entities(answer, annotation=annotation, references=[[1, 72], [3, 34]])
    assert [(entity["name"], entity["point_2d"]) for entity in first["entities"]] == [("Character 5", [150, 550])]
    assert first["unknown_characters"] == ["Character 6"] and first["ambiguous_characters"] == ["Character 6"]
    assert [entity["name"] for entity in second["entities"]] == ["Character 3"] and second["ambiguous_characters"] == []


def test_source_grounding_unknown_schema_example_matches_strict_consumer():
    from mspc.inference.reference_pointing import grounded_reference_entities, source_grounding_content
    name = CASE["annotation"]["characters"][0]["name"]
    example = {"unknown_characters": [name]}
    content = source_grounding_content(CASE["annotation"], [[1, 12]])
    assert json.dumps(example) in content[-1]["text"]
    answer = {"references": [{"reference": 1, "characters": [], **example}]}
    rows = grounded_reference_entities(answer, annotation=CASE["annotation"], references=[[1, 12]])
    assert rows[0]["unknown_characters"] == [name] and rows[0]["entities"] == []


def test_source_grounding_hints_give_every_registered_character_the_outfit_of_each_reference_shot():
    import re
    from mspc.inference.reference_pointing import source_grounding_content
    # 真实开镜对齐计划：Character 1 第 2 镜穿登记的西装，第 6 镜改为 shirtless；Character 2 第 2 镜不在场。
    plan = json.loads((REPO_ROOT / "tests/fixtures/opening_alignment/film-000127-b0000-shot01.json").read_text())["plan"]
    content = source_grounding_content(plan, [[1, 20], [5, 30]])
    assert [part["type"] for part in content] == ["text", "text", "image", "text", "text", "image", "text", "text"]
    hints = []
    for number, part in ((1, content[3]), (2, content[6])):
        prefix = f"Identification hints for Reference {number}: "
        assert part["text"].startswith(prefix)
        hints.append({row["name"]: row for row in json.loads(part["text"].removeprefix(prefix))})
    names = [entry["name"] for entry in plan["characters"]]
    assert [list(rows) for rows in hints] == [names, names]
    assert hints[0]["Character 1"]["description"] == (
        "a man with a shaved head, mustache, and medium build, wearing a grey suit, a pink shirt, and a purple tie")
    assert hints[1]["Character 1"]["description"] == (
        "a man with a shaved head, mustache, and medium build, wearing shirtless and dark trousers")
    assert "Character 2" not in {entry["name"] for entry in plan["shots"][1]["present"]}
    assert hints[0]["Character 2"]["description"].endswith("wearing " + plan["characters"][1]["outfit"])
    assert all(row["kind"] == "person" for rows in hints for row in rows.values())
    texts = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert re.search(r"\[\s*\d+\s*,\s*\d+\s*\]", texts) is None
    assert "500" not in texts
    assert '"point_2d": [x, y]' in content[-1]["text"] and "0-1000" in content[-1]["text"]


def test_source_grounding_hint_for_a_group_without_outfit_is_its_appearance():
    from mspc.inference.reference_pointing import source_grounding_content
    content = source_grounding_content(CASE["annotation"], [[1, 12]])
    rows = json.loads(content[3]["text"].split(": ", 1)[1])
    group = next(row for row in rows if row["name"] == "Character 4")
    assert group == {"name": "Character 4", "kind": "group", "description": "people walking outside"}


def test_source_grounding_rejects_observed_unknown_object_without_coercion():
    from mspc.inference.reference_pointing import grounded_reference_entities
    # H dev30 cinedance-010863-b0000-shot01/g2 raw unknown entry, 2026-11-01T20:27Z.
    answer = {"references": [{"reference": 1, "characters": [],
                              "unknown_characters": [{"name": "Character 1", "point_2d": [450, 500]}]}]}
    assert "Character 1" in {row["name"] for row in CASE["annotation"]["characters"]}
    with pytest.raises(ValueError, match="unknown source identities must be unique registered names"):
        grounded_reference_entities(answer, annotation=CASE["annotation"], references=[[1, 12]])
    assert answer["references"][0]["unknown_characters"] == [{"name": "Character 1", "point_2d": [450, 500]}]
