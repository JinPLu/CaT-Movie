"""真实v18行的新登记／镜内时间线／参考用途合同。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.shot_text import shot_text, shot_text_with_entity_spans, resolve_action_references


ROOT = Path(__file__).parents[2] / "fixtures"
BLOCKS = [row["annotation"] for row in map(json.loads,
    (ROOT / "sampling_ledger/block-annotations.jsonl").read_text().splitlines()) if row["annotation"]]
GAPS = {row["story_id"]: row["annotation"] for row in json.loads(
    (ROOT / "block_annotation_v18/test92-compiler-information-gaps.json").read_text())}


def test_real_hat_change_stays_out_of_stable_registry_and_after_the_action():
    annotation = copy.deepcopy(BLOCKS[1])
    before = copy.deepcopy(annotation)
    text = shot_text(annotation, 1)
    registry, timeline = text.split("2D animation.", 1)
    assert "Character 1:" in registry
    assert "red collar" not in registry and "straw hat" not in registry
    opening, action_and_end = timeline.split(resolve_action_references(annotation, 1)["action"], 1)
    assert "wearing a red collar" in opening and "wearing a red collar and a straw hat" not in opening
    assert "By the end of the shot, Character 1 wears a red collar and a straw hat" in action_and_end
    assert annotation == before


def test_real_release_keeps_known_relation_without_guessing_a_visible_endpoint():
    annotation = GAPS["film-005223-b0001-shot01"]
    text = shot_text(annotation, 0)
    assert "holds Object 1" in text
    assert text.endswith("Character 1 has released Object 1.")
    assert "no longer" not in text and "empty-handed" not in text
    missing = copy.deepcopy(annotation)
    missing["shots"][0]["changes"][0].pop("before")
    with pytest.raises(ValueError, match="prior held entity"):
        shot_text(missing, 0)


def test_real_release_uses_the_same_entitys_visible_state_ending_once():
    row = json.loads((ROOT / "block_annotation_v18/holding-release-visible-ending.json").read_text())
    annotation = row["annotation"]
    before = copy.deepcopy(annotation)
    text = shot_text(annotation, 0)
    assert text.endswith("By the end of the shot, Character 1 is rubbing face.")
    assert "has released" not in text
    assert "then lowers it and rubs his face" in text
    assert annotation == before


def test_real_matching_hair_uses_existing_shot_clothing_in_explicit_action_ids_only():
    annotation = json.loads((ROOT / "planner_source_v18/film-003415-b0000.json").read_text())["annotation"]
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 0)
    text = compiled["text"]
    registry, timeline = text.split("Setting: ", 1)
    assert "dark vest" in registry and "suspenders" in registry
    assert "Character 1, the man wearing dark trousers with suspenders," in timeline
    assert "Character 2, the man wearing dark vest," in timeline
    spans = [span for span in compiled["entity_spans"] if span["source"] == "characters[1].outfit"]
    assert len([span for span in spans if text[span["start"]:span["end"]] ==
                "dark vest"]) == 1
    assert annotation == before

    judge = GAPS["cinedance-009077-b0004-shot01"]
    assert shot_text(judge, 0).endswith(judge["shots"][0]["action"])


def test_real_entering_character_is_described_on_first_appearance_without_asserting_rgb_zero_presence():
    stories = json.loads((ROOT.parents[1] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    annotation = next(row["storyboard"] for row in stories if row["story_id"] == "cinedance-013630-b0000-shot01")
    before = copy.deepcopy(annotation)
    text = shot_text(annotation, 1)
    assert "entering through the door" in text
    assert "enters the room through the door." in text
    assert text.endswith("By the end of the shot, Object 1 is open.")
    assert annotation == before


def test_real_same_hair_uses_center_noun_and_an_existing_distinctive_second_feature():
    stories = json.loads((ROOT.parents[1] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    annotation = next(row["storyboard"] for row in stories if row["story_id"] == "cinedance-012125-b0000-shot01")
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 0)
    assert "Character 1, the man with a medium build, sits at a table" in compiled["text"]
    assert "Character 2, the man with a lean build," in compiled["text"]
    for span in compiled["entity_spans"]:
        assert 0 <= span["start"] < span["end"] <= len(compiled["text"])
    assert annotation == before


def test_real_possessive_reference_keeps_its_owner_and_color_conjunction_keeps_the_garment():
    stories = json.loads((ROOT.parents[1] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    by_id = {row["story_id"]: row["storyboard"] for row in stories}
    compiled = shot_text_with_entity_spans(by_id["cinedance-011581-b0002-shot01"], 2)
    assert "Character 1's white shirt" in compiled["text"]
    action_mentions = [compiled["text"][span["start"]:span["end"]] for span in compiled["entity_spans"]
                       if span["source"] == "shots[2].action"]
    assert action_mentions == ["Character 1"]
    text = shot_text(by_id["cinedance-011901-b0000-shot01"], 0)
    assert "Character 1, the young man wearing a blue and white varsity jacket," in text


def test_real_hat_opening_uses_explicit_before_without_mutating_the_given_board():
    stories = json.loads((ROOT.parents[1] / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
    annotation = copy.deepcopy(next(row["storyboard"] for row in stories
                                   if row["story_id"] == "film-001594-b0006-shot04"))
    original = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 3)
    opening = compiled["text"].split("By the end of the shot, ", 1)[0]
    assert "wearing a light purple polo shirt" in opening and "wearing a hat with lures on it" not in opening
    assert "wears a light purple polo shirt and a hat with lures" in compiled["text"].split("By the end of the shot, ", 1)[1]
    assert {entry["field"] for entry in compiled["opening_conflicts"]} == {"outfit"}
    assert compiled["opening_conflicts"][0]["before_source"] == "shots[3].changes[0].before"
    assert annotation == original

    # 明确空衣着不能把registry默认衣着/帽子重新填回来；没有before则只保留已给起态，不猜。
    annotation["shots"][3]["changes"][0]["before"] = ""
    empty = shot_text(annotation, 3).split("By the end of the shot, ", 1)[0]
    assert "light purple polo shirt" not in empty
    annotation["shots"][3]["changes"][0].pop("before")
    unknown = shot_text_with_entity_spans(annotation, 3)
    assert "wearing a hat with lures on it" in unknown["text"].split("By the end of the shot, ", 1)[0]
    assert not unknown["opening_conflicts"]


def test_explicit_empty_holding_before_overrides_the_supplied_after_possession_and_spans():
    annotation = copy.deepcopy(GAPS["film-005223-b0001-shot01"])
    annotation["shots"][0]["changes"] = [{"name": "Character 1", "what": "holding",
                                         "before": [], "after": ["Object 1"]}]
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 0)
    assert "holding Object 1" not in compiled["text"].split("By the end of the shot, ", 1)[0]
    assert compiled["text"].endswith("By the end of the shot, Character 1 holds Object 1.")
    assert compiled["opening_conflicts"][0]["before"] == []
    assert annotation == before


def test_real_crowded_scene_registers_every_present_object_once():
    case = json.loads((ROOT / "block_annotation_v18/layout-token-overflow.json").read_text())["cases"][0]
    annotation = case["annotation"]
    before = copy.deepcopy(annotation)
    output = shot_text_with_entity_spans(annotation, 0)
    # 在场11件登记物体全部写登记描述，包括沙发、桌椅、灯这类只列在场、没有状态的陈设。
    objects = [span["name"] for span in output["entity_spans"]
               if span["source"].startswith("objects[") and span["source"].endswith(".name")]
    assert objects == [f"Object {number}" for number in range(1, 12)]
    assert len([entry for entry in annotation["shots"][0]["present"] if entry["name"].startswith("Object ")]) == 11
    assert "Object 2: brown leather couch. " in output["text_sections"]["registry"]
    assert annotation == before
    for number in range(1, 8):
        assert f"Character {number}: " in output["text"].split("Object 1: ", 1)[0]
    first, action = output["text"].split("A group of people are gathered in an office lounge. ", 1)
    assert "Character 7 is to the right of Character 6, holding Object 1 and Object 4" in first
    assert "sit at a table" in action and output["text"].count("wearing green cardigan") == 1
    assert action.count("Character 7, the woman with short black hair,") == 1
    assert "with Character 7 smoking a cigar" in action


def test_actual_reference_slots_protect_entities_without_adding_target_text_or_spans():
    annotation = BLOCKS[0]
    names = [entry["name"] for entry in annotation["shots"][0]["present"]
             if entry["name"].startswith("Character ")]
    references = [{"index": 2, "entities": names}, {"index": 5, "entities": []}]
    output = shot_text_with_entity_spans(annotation, 0, references=references)
    text = output["text"]
    assert text == shot_text(annotation, 0)
    assert text == shot_text(annotation, 0, references=references)
    spans = [span for span in output["entity_spans"] if span["source"].startswith("references[")]
    assert spans == []
    assert all(text[span["start"]:span["end"]] for span in output["entity_spans"])

    unknown = shot_text(annotation, 0, references=[{"index": n, "entities": []} for n in range(1, 11)])
    assert unknown == shot_text(annotation, 0)


@pytest.mark.parametrize("references", [
    [{"index": 0, "entities": []}],
    [{"index": 1, "entities": []}, {"index": 1, "entities": []}],
    [{"index": 1, "entities": ["Character 999"]}],
    [{"index": 1, "entities": "Character 1"}],
    [{"index": 1, "entities": [], "source_shot_objects": []}],
])
def test_invalid_reference_metadata_cannot_turn_into_visual_facts(references):
    with pytest.raises(ValueError):
        shot_text(BLOCKS[0], 0, references=references)
