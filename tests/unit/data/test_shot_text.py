"""v18 原生块与旧真实窗离线转换后的逐字对拍。"""
import copy
import json
import re
from pathlib import Path

import pytest

from mspc.core.wan22_contract import MEMORY_ROW_ROLE_TEXT, require_memory_rows
from mspc.data.shot_text import (_composition, _third, opening_frame_text, reference_text,
                                reference_setting, reference_text_with_entity_spans, shot_object_phrases, shot_script_text, shot_text,
                                shot_text_with_entity_spans, without_target_text)
from mspc.data.storyboard import align_board_openings, align_opening_with_changes, from_annotation

LEGACY_ANNOTATION = json.loads((Path(__file__).parents[2] / "fixtures/window_annotation_v15/trial-0132.json").read_text())["annotation"]
# 仅验收旧文字的离线转换；生产编译器只读 outfit。
ANNOTATION = copy.deepcopy(LEGACY_ANNOTATION)
for _shot in ANNOTATION["shots"]:
    for _entry in _shot["present"]:
        if "clothing" in _entry:
            _entry["outfit"] = _entry.pop("clothing")
    for _change in _shot["changes"]:
        if _change["what"] == "clothing":
            _change["what"] = "outfit"
RAW_ANNOTATION = copy.deepcopy(ANNOTATION)
# 生产里文字只读经过对齐的标注（标注修补与规划编译都先调 align_board_openings）。
align_board_openings(ANNOTATION, {e["name"]: e.get("outfit") for e in ANNOTATION["characters"]})
BLOCK_RECORDS = [row for row in map(json.loads,
    (Path(__file__).parents[2] / "fixtures/sampling_ledger/block-annotations.jsonl").read_text().splitlines())
    if row["annotation"] is not None]
BLOCKS = [row["annotation"] for row in BLOCK_RECORDS]


@pytest.mark.parametrize("index,prefix", [(0, "Live-action footage."), (1, "2D animation.")])
def test_real_picture_type_reaches_all_text_consumers(index, prefix):
    annotation = copy.deepcopy(BLOCKS[index])
    assert (prefix + " ") in shot_text(annotation, 0)
    assert opening_frame_text(annotation, 0).startswith(prefix + " ")


def test_real_picture_type_survives_annotation_to_inference_without_moving_entity_spans():
    annotation = copy.deepcopy(BLOCKS[0])
    counts = [37] * len(annotation["shots"])
    board = from_annotation(annotation, counts, start=0, end=len(counts))
    assert board["shots"][0]["picture_type"] == annotation["shots"][0]["picture_type"]
    assert shot_text(board, 0) == shot_text(annotation, 0)
    assert reference_text() == MEMORY_ROW_ROLE_TEXT
    compiled = shot_text_with_entity_spans(board, 0)
    for span in compiled["entity_spans"]:
        assert compiled["text"][span["start"]:span["end"]]


def test_unspecified_picture_type_does_not_invent_a_medium_and_unknown_value_fails():
    annotation = copy.deepcopy(BLOCKS[0])
    annotation["shots"][0].pop("picture_type")
    assert "Live-action footage." not in shot_text(annotation, 0)
    annotation["shots"][0]["picture_type"] = "photorealistic_guessed_by_agent"
    with pytest.raises(ValueError, match="picture_type"):
        shot_text(annotation, 0)


def test_native_blocks_compile_including_a_title_card_without_place():
    for record in BLOCK_RECORDS:
        annotation = record["annotation"]
        for index in range(len(annotation["shots"])):
            assert shot_text(annotation, index)
            if record["shot_filter"][index]["status"] == "可用":
                require_memory_rows([reference_text()], row_policy="fixed_role_v1")
    assert "SNOOPY'S SUMMERTIME FUN LIST" in shot_text(BLOCKS[1], 0)
    omitted = copy.deepcopy(BLOCKS[0])
    omitted["shots"][0]["place"] = "none"
    assert "In none" not in shot_text(omitted, 0)


def test_real_entity_spans_keep_source_ids_through_repeated_mentions_and_equal_appearances():
    annotation = copy.deepcopy(ANNOTATION)
    visible_object = next(entry['name'] for entry in annotation['shots'][1]['present'] if entry['name'].startswith('Object '))
    next(entry for entry in annotation['objects'] if entry['name'] == visible_object)['appearance'] = annotation['places'][0]['appearance']
    compiled = shot_text_with_entity_spans(annotation, 1)
    assert compiled['text'] == shot_text(annotation, 1)
    spans = compiled['entity_spans']
    assert all(0 <= span['start'] < span['end'] <= len(compiled['text']) and span['source'] for span in spans)
    thing = visible_object
    place = annotation['places'][0]['name']
    assert {span['name'] for span in spans} >= {thing, place, 'Character 3'}
    object_spans = [span for span in spans if span['name'] == thing]
    assert all(span['name'] != place for span in object_spans)
    assert any(span['source'].endswith('.action') and span['name'] == 'Character 3' for span in spans)


def test_layout_writes_composition_and_positions_in_wan_style_without_templates_or_numbers():
    # 真实第 3 镜两人同在场；框两锚不动。构图词接在机位后，后出现的人写对先出现者的位置，不写大小档与数字。
    annotation = copy.deepcopy(ANNOTATION)
    base = shot_text(annotation, 2)
    still = lambda box: [{"time": 0.0, "visibility": "visible", "box": box},
                         {"time": 1.0, "visibility": "visible", "box": box}]
    annotation["layouts"] = [{"shot": 3, "entities": [
        {"name": "Character 1", "anchors": still([0.1, 0.2, 0.4, 0.9])},
        {"name": "Character 2", "anchors": still([0.6, 0.3, 0.9, 0.7])}]}]
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 2)
    text = compiled["text"]
    assert annotation == before
    assert "symmetrical composition." in text.split(" In ")[0]
    assert "to the right of Character 1" in text
    assert "Layout:" not in text and "Near to far" not in text
    assert text.replace(", symmetrical composition", "").replace(" to the right of Character 1", "") == base
    layout_spans = [span for span in compiled["entity_spans"] if span["source"].startswith("layouts")]
    assert [text[span["start"]:span["end"]] for span in layout_spans] == ["Character 1"]


@pytest.mark.parametrize("boxes,word", [
    ([[0.05, 0.2, 0.3, 0.9]], "left-heavy composition"),
    ([[0.7, 0.2, 0.95, 0.9]], "right-heavy composition"),
    ([[0.4, 0.1, 0.6, 0.9]], "center composition"),
    ([[0.27, 0.1, 0.5, 0.9]], "left-heavy composition"),
    ([[0.5, 0.1, 0.73, 0.9]], "right-heavy composition"),
    ([[0.35, 0.2, 0.5, 0.9], [0.5, 0.2, 0.65, 0.9]], "center composition"),
    ([[0.1, 0.2, 0.3, 0.9], [0.7, 0.2, 0.9, 0.9]], "symmetrical composition"),
    ([[0.1, 0.2, 0.3, 0.9], [0.7, 0.6, 0.8, 0.9]], "left-heavy composition"),
    ([[0.2, 0.2, 0.4, 0.9], [0.85, 0.5, 0.95, 0.9]], "balanced composition"),
    ([[0.05, 0.2, 0.2, 0.9], [0.45, 0.2, 0.55, 0.9], [0.8, 0.2, 0.95, 0.9]], "balanced composition"),
])
def test_composition_follows_the_area_weighted_horizontal_balance(boxes, word):
    assert _composition(boxes) == word


def test_composition_needs_a_placed_character():
    assert _composition([]) is None


def test_character_who_changes_side_is_placed_at_the_end_of_the_shot():
    annotation = copy.deepcopy(ANNOTATION)
    annotation["layouts"] = [{"shot": 3, "entities": [{"name": "Character 1", "anchors": [
        {"time": 0.0, "visibility": "visible", "box": [0.05, 0.2, 0.3, 0.9]},
        {"time": 1.0, "visibility": "visible", "box": [0.7, 0.2, 0.95, 0.9]}]}]}]
    text = shot_text(annotation, 2)
    assert "left-heavy composition." in text
    assert "Character 1 is on the right side of the frame." in text.split("By the end of the shot, ")[1]


def test_a_small_drift_across_a_third_line_is_not_an_end_position():
    annotation = copy.deepcopy(ANNOTATION)
    annotation["layouts"] = [{"shot": 3, "entities": [{"name": "Character 1", "anchors": [
        {"time": 0.0, "visibility": "visible", "box": [0.55, 0.2, 0.75, 0.9]},
        {"time": 1.0, "visibility": "visible", "box": [0.58, 0.2, 0.78, 0.9]}]}]}]
    assert "side of the frame" not in shot_text(annotation, 2)


def test_late_entries_objects_and_absent_people_are_not_placed():
    annotation = copy.deepcopy(ANNOTATION)
    present = [entry["name"] for entry in annotation["shots"][2]["present"]]
    absent = next(entry["name"] for entry in annotation["characters"] if entry["name"] not in present)
    annotation["layouts"] = [{"shot": 3, "entities": [
        {"name": "Character 1", "anchors": [{"time": 0.8, "visibility": "visible", "box": [0.05, 0.2, 0.3, 0.9]}]},
        {"name": absent, "anchors": [{"time": 0.0, "visibility": "visible", "box": [0.7, 0.2, 0.95, 0.9]}]},
        {"name": annotation["objects"][0]["name"],
         "anchors": [{"time": 0.0, "visibility": "visible", "box": [0.4, 0.2, 0.6, 0.9]}]}]}]
    assert shot_text(annotation, 2) == shot_text(ANNOTATION, 2)


@pytest.mark.parametrize("case", json.loads((Path(__file__).parents[2]
    / "fixtures/block_annotation_v18/layout-token-overflow.json").read_text())["cases"],
    ids=lambda case: ",".join(map(str, case["sample_indices"])))
def test_real_crowded_layouts_compile_to_wan_phrases_about_present_characters(case):
    # 3 条真实多人行：旧模板超过 umT5 512；新写法只剩构图词与相邻两人的左右、标注里的前后。
    annotation = copy.deepcopy(case["annotation"])
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 0)
    text = compiled["text"]
    assert annotation == before
    assert "Layout:" not in text and "Near to far" not in text
    assert re.search(r"(?:center|balanced|symmetrical|left-heavy|right-heavy) composition\.", text.split(" In ")[0])
    present = {entry["name"] for entry in annotation["shots"][0]["present"]}
    named = re.findall(r"(?:to the (?:left|right) of|in front of|behind) (Character \d+)", text)
    assert named and set(named) <= present
    assert text.startswith("Character 1: ")
    for span in compiled["entity_spans"]:
        assert text[span["start"]:span["end"]] == span["name"] or not span["source"].startswith("layouts")


@pytest.mark.parametrize("x,word", [(0.0, "left"), (0.333, "left"), (1 / 3, "center"), (0.5, "center"),
                                    (0.666, "center"), (2 / 3, "right"), (1.0, "right")])
def test_layout_thirds_split_left_center_right(x, word):
    assert _third(x) == word


def test_real_animal_outfit_uses_registry_then_change_then_shot_override():
    annotation = BLOCKS[1]
    first = shot_text(annotation, 1)
    assert "is walking and dancing, wearing a red collar," in first
    assert "By the end of the shot, Character 1 wears a red collar and a straw hat" in first
    assert "stands, wearing a red collar and a straw hat, looking angry" in shot_text(annotation, 3)
    assert reference_text() == MEMORY_ROW_ROLE_TEXT


def test_outfit_change_matching_registry_default_uses_explicit_before_on_a_copy():
    # 在真实换装镜上核登记表默认值也参与开镜状态比较：开镜默认穿着已是变化后，未对齐就报错，对齐后开镜写 before。
    annotation = copy.deepcopy(BLOCKS[1])
    shot = annotation["shots"][1]
    shot["changes"] = [{"name": "Character 1", "what": "outfit",
                        "before": "a red collar and a straw hat", "after": "a red collar"}]
    original = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 1)
    assert compiled["opening_conflicts"]
    assert "red collar and a straw hat" in compiled["text"].split("By the end of the shot,")[0]
    assert annotation == original
    align_opening_with_changes(shot, {e["name"]: e.get("outfit") for e in annotation["characters"]}, {})
    text = shot_text(annotation, 1)
    assert "a red collar and a straw hat" in text.split("By the end of the shot,")[0]
    assert text.endswith("By the end of the shot, Character 1 wears a red collar.")


def test_real_outfit_change_already_starting_with_wearing_uses_one_verb():
    annotation = json.loads((Path(__file__).parents[2]
        / "fixtures/block_annotation_v18/film-002869-b0001.json").read_text())["annotation"]
    assert annotation["shots"][4]["changes"][0]["after"] == "wearing a beige cap"
    text = shot_text(annotation, 4)
    assert text.endswith("By the end of the shot, Character 1 wears a beige cap.")


def test_shot_tier_uses_registries_and_camera_action_only():
    text = shot_script_text(BLOCKS[0])
    assert text.startswith("SUBJECTS\n- Character 1 (person): a young woman with dark hair tied back and bangs, "
                           "wearing a white fuzzy sweater with a pink collar\n")
    assert "\nLOCATIONS\n- Location 1: a classroom" in text
    assert "\nSHOTS\ns01: Medium close-up shot. Character 1 slowly opens" in text
    assert "\ns02: Medium shot, pan. Character 1 walks" in text
    assert "peeking through the doorway" not in text and "By the end of the shot," not in text

FOREST = "In a dense green forest with a small stream, "
PULLER = ("Character 3, a man with dark curly hair and a medium build, is standing in the stream, pulling, "
          "wearing a light shirt and dark overalls, holding a rope or strap attached to the cow.")


def test_target_row_runs_camera_opening_state_action_then_end_of_shot():
    text = shot_text(ANNOTATION, 1)
    registry, timeline = text.split("Wide shot, high angle shot. ", 1)
    assert "Character 3: man with dark curly hair and a medium build" in registry
    assert "Setting: dense green forest with a small stream" in registry
    assert "wearing light shirt, dark overalls" in registry
    assert text.count("wearing light shirt, dark overalls") == 1
    opening, ending = timeline.split("By the end of the shot, ", 1)
    assert opening.startswith("Character 3 stands")
    assert "holding Object 5" in opening and "Object 4 lies still in the stream" in opening
    assert "Character 3, the man with dark curly hair, struggles to drag a dead cow out of a stream in the forest." in opening
    assert ending == "Object 4 is being pulled through the water."


def test_real_opening_frame_keeps_initial_state_without_advancing_the_event():
    text = opening_frame_text(ANNOTATION, 1)
    assert text.startswith(
        "Wide shot, high angle shot. " + FOREST + PULLER
        + " A dead brown and white cow is lying in the stream.")
    assert "struggles to drag" in shot_text(ANNOTATION, 1)
    assert "By the end of the shot," in shot_text(ANNOTATION, 1)


def test_real_first_frame_preserves_the_cat_and_static_object_positions():
    fixture = Path(__file__).parents[2] / "fixtures/storydiffusion/cinedance-014127-b0000-shot01.json"
    board = json.loads(fixture.read_text())["storyboard"]
    text = opening_frame_text(board, 0)
    assert "A cat, sitting on top of a trash can" in text
    assert "A green trash can, standing next to a wooden fence" in text
    assert "Black trash bags, on the ground next to the trash can" in text
    assert "White trash bags, on the ground next to the fence" in text
    turn = opening_frame_text(board, 9)
    assert "looking at the woman" in turn
    assert "turning away and walking" in shot_text(board, 9)


def test_default_camera_words_are_left_out_and_held_objects_are_written_by_appearance():
    text = shot_text(ANNOTATION, 2)
    assert "Medium shot. Character 1 is standing and gesturing." in text
    assert text.count("wearing grey coat, a brown hat") == 1
    assert "Object 3: large two-man hand saw" in text and "holding Object 3." in text
    for banned in ("Design:", "State:", "eye-level", "static"):
        assert banned not in text


def test_a_change_already_written_into_the_opening_record_uses_before_immutably():
    # 第 4 镜：changes 记牛「lying in the stream → being dragged through the water」，但开镜记录里已经是后者。
    # 旧排版悄悄跳过这条变化，镜末状态被写在开头；现在未对齐的输入报错，对齐后开镜写 before、镜末写 after。
    shot = RAW_ANNOTATION["shots"][3]
    assert shot["changes"][0]["after"] == next(e for e in shot["present"] if e["name"] == "Object 4")["state"]
    original = copy.deepcopy(RAW_ANNOTATION)
    compiled = shot_text_with_entity_spans(RAW_ANNOTATION, 3)
    assert compiled["opening_conflicts"] == [{
        "name": "Object 4", "field": "state", "source": "shots[3].present[1].state",
        "value": "being dragged through the water", "before_source": "shots[3].changes[0].before",
        "before": "lying in the stream"}]
    assert RAW_ANNOTATION == original
    text = shot_text(ANNOTATION, 3)
    assert "lies in the stream" in text.split("By the end of the shot,")[0]
    assert text.endswith("is being dragged through the water.") and text.count("is being dragged through the water") == 1


def test_a_character_named_in_the_action_but_absent_from_the_shot_gets_no_appearance():
    # trial-0132 的动作句没有点到不在场的人；这里把第 4 镜的动作句改成点到 Character 1（他只在第 1、3 镜），
    # 其余字段是真实返回。全量标注回来后换成真实行。
    annotation = copy.deepcopy(ANNOTATION)
    annotation["shots"][3]["action"] = "Character 3 pulls the cow while shouting to Character 1."
    text = shot_text(annotation, 3)
    assert "shouting to Character 1." in text and "long dark hair" not in text


def test_reference_role_has_no_unverified_whole_shot_facts():
    text = reference_text()
    require_memory_rows([text], row_policy="fixed_role_v1")
    assert text == MEMORY_ROW_ROLE_TEXT


def test_reference_rows_name_the_source_shot_objects_with_the_target_text_phrase():
    annotation = next(block for block in BLOCKS if any(
        entry["name"] in {o["name"] for o in block["objects"]} for shot in block["shots"] for entry in shot["present"]))
    index = next(i for i, shot in enumerate(annotation["shots"]) if shot_object_phrases(annotation, i))
    phrases = shot_object_phrases(annotation, index)
    appearance = {o["name"]: o["appearance"] for o in annotation["objects"]}
    assert list(phrases) == [appearance[e["name"]] for e in annotation["shots"][index]["present"] if e["name"] in appearance]
    plain = reference_text_with_entity_spans([{"name": "Character 1"}], setting=None, characters=annotation["characters"])
    row = reference_text_with_entity_spans([{"name": "Character 1"}], setting=None, characters=annotation["characters"], shot_objects=phrases)
    assert row["text"] == plain["text"] + " Objects in that shot: " + "; ".join(phrases) + "."
    assert row["entity_spans"] == plain["entity_spans"]
    only_objects = reference_text_with_entity_spans([], setting=None, characters=annotation["characters"], shot_objects=phrases)["text"]
    assert only_objects == MEMORY_ROW_ROLE_TEXT + " Objects in that shot: " + "; ".join(phrases) + "."
    require_memory_rows([row["text"], only_objects, MEMORY_ROW_ROLE_TEXT],
                        row_policy="frame_entities_with_shot_objects_v1")
    with pytest.raises(ValueError, match="frame_entities_v1"):
        require_memory_rows([only_objects], row_policy="frame_entities_v1")


def test_reference_visible_characters_each_include_the_registered_appearance_and_its_span():
    characters = BLOCKS[0]['characters'][:2]
    entities = [{'name': row['name']} for row in characters]
    compiled = reference_text_with_entity_spans(entities, setting=None, characters=characters)
    descriptions = [re.sub(r'^an? ', '', row['appearance']).strip().rstrip('.') for row in characters]
    assert compiled['text'] == MEMORY_ROW_ROLE_TEXT + ' Visible identities: ' + '; '.join(
        row['name'] + ': ' + appearance for row, appearance in zip(characters, descriptions, strict=True)) + '.'
    assert [(span['name'], compiled['text'][span['start']:span['end']], span['source'])
            for span in compiled['entity_spans']] == [
        item for i, (row, appearance) in enumerate(zip(characters, descriptions, strict=True)) for item in (
            (row['name'], row['name'], f'reference_entities[{i}].name'),
            (row['name'], appearance, f'characters[{i}].appearance'))]


def test_reference_registry_is_required_and_visible_ids_cannot_use_a_missing_registration():
    with pytest.raises(TypeError, match='characters'):
        reference_text_with_entity_spans([])
    with pytest.raises(ValueError, match='exist in the character registry'):
        reference_text_with_entity_spans([{'name': 'Character 999'}], setting=None, characters=BLOCKS[0]['characters'])


def test_omitted_target_text_is_one_rule_for_training_members_and_plans():
    annotation = copy.deepcopy(BLOCKS[0])
    annotation["shots"][0]["picture_type"] = "live_action"
    annotation["layouts"] = [{"shot": 1, "entities": [{"name": annotation["characters"][0]["name"], "anchors": [
        {"time": 0.0, "visibility": "visible", "box": [0.1, 0.1, 0.4, 0.9]}]}]}]
    full = shot_text(annotation, 0)
    plain = shot_text(without_target_text(annotation, frozenset({"layouts", "picture_type"})), 0)
    assert "Live-action footage." in full and "composition." in full
    assert "Live-action" not in plain and "composition" not in plain
    assert "layouts" in annotation and annotation["shots"][0]["picture_type"] == "live_action"
    with pytest.raises(ValueError, match="omitted target text"):
        without_target_text(annotation, frozenset({"actions"}))


def test_mixed_live_action_cg_label_does_not_declare_the_whole_shot_cg():
    # 真实Ted2切片中，CG熊与真人/法庭同框；粗枚举不能保证全画面动画。
    annotation = copy.deepcopy(BLOCKS[0])
    annotation["shots"][0]["picture_type"] = "animation_3d_or_game"
    assert "The shot includes computer-generated imagery." in shot_text(annotation, 0)


def test_every_real_shot_compiles():
    for i, camera in enumerate(["Medium shot, handheld", "Wide shot, high angle shot", "Medium shot", "Medium wide shot"]):
        assert camera + ". " in shot_text(ANNOTATION, i)


def test_a_held_person_is_written_by_name():
    # 真实返回里人可以抱着或牵着另一个人（cinedance-009198-w0003：抱着婴儿）
    annotation = copy.deepcopy(ANNOTATION)
    holder = annotation["shots"][0]["present"][1]
    holder["holding"] = [annotation["shots"][0]["present"][0]["name"], *holder["holding"]]
    assert "holding Character 1 and " in shot_text(annotation, 0)


@pytest.mark.parametrize("outfit,expected,banned", [
    # 首批窗的真实返回：cinedance-009034-w0002 第 2 镜、一条「none visible above the blanket」、149 条 unknown。
    ("shirtless, wet hair", "is standing and gesturing, shirtless, wet hair.", "wearing shirtless"),
    ("none visible above the blanket", "is standing and gesturing.", "none visible"),
    ("unknown", "is standing and gesturing.", "unknown"),
])
def test_an_outfit_value_that_is_not_a_garment_is_not_written_after_wearing(outfit, expected, banned):
    annotation = copy.deepcopy(ANNOTATION)
    shot = annotation["shots"][2]
    entry = next(e for e in shot["present"] if e["name"] == "Character 1")
    entry["outfit"] = outfit
    shot["changes"] = [{"name": "Character 1", "what": "outfit", "before": "a grey coat and a brown hat", "after": outfit}]
    entry["outfit"] = "a grey coat and a brown hat"
    text = shot_text(annotation, 2)
    assert banned not in text
    assert ("By the end of the shot, Character 1 is shirtless, wet hair." in text) == outfit.startswith("shirtless")
    entry["outfit"], shot["changes"] = outfit, []
    text = shot_text(annotation, 2)
    assert "Character 1 " + expected in text and banned not in text


COMPILER_GAPS = {case["story_id"]: case["annotation"] for case in json.loads((Path(__file__).parents[2]
    / "fixtures/block_annotation_v18/test92-compiler-information-gaps.json").read_text())}


def test_test92_object_posture_reaches_video_and_opening_frame_once():
    board = COMPILER_GAPS["cinedance-009077-b0004-shot01"]
    video, opening = shot_text(board, 0), opening_frame_text(board, 0)
    for name, phrase, posture in [("Object 1", "a black leather chair", "behind Character 1"),
                                  ("Object 2", "a microphone on a stand", "in front of Character 1"),
                                  ("Object 7", "a laptop or monitor", "in front of Character 1")]:
        assert video.count(name.split()[1] + ": " + phrase.removeprefix("a ")) == 1
        assert posture + "." in video
        assert ("Object 1 is behind Character 1." if name == "Object 1" else
                "Object 2 and Object 7 are in front of Character 1.") in video
        assert opening.count(phrase.capitalize() + ", " + posture + ".") == 1
    assert "A judge speaks from behind his bench." not in opening_frame_text(board, 0)


def test_test92_release_relation_and_its_source_identity_reach_video_only():
    board = COMPILER_GAPS["film-005223-b0001-shot01"]
    compiled = shot_text_with_entity_spans(board, 0)
    assert compiled["text"].endswith("By the end of the shot, Character 1 has released Object 1.")
    assert "empty-handed" not in compiled["text"] and "no longer" not in compiled["text"]
    assert "no longer holds" not in opening_frame_text(board, 0)
    assert any(span["name"] == "Object 1" and span["source"] == "shots[0].changes[0].before"
               for span in compiled["entity_spans"])


def test_reference_rows_add_the_source_shot_setting_with_the_target_text_wording():
    # 参考行写源镜地点，与目标行登记段同一短语；多地点块带编号，单地点块不带。
    multi = next(block for block in BLOCKS if len(block["places"]) > 1
                 and any(shot.get("place") not in (None, "none", "unknown") for shot in block["shots"]))
    single = next(block for block in BLOCKS if len(block["places"]) == 1
                  and block["places"][0]["appearance"] and block["shots"][0].get("place") not in (None, "none", "unknown"))
    for annotation in (multi, single):
        index = next(i for i, shot in enumerate(annotation["shots"]) if shot.get("place") not in (None, "none", "unknown"))
        setting = reference_setting(annotation, index)
        assert "Setting: " + setting in shot_text(annotation, index)
        assert setting.startswith(annotation["shots"][index]["place"] + ", ") == (annotation is multi)
        row = reference_text_with_entity_spans([{"name": "Character 1"}], setting=setting, characters=annotation["characters"])
        plain = reference_text_with_entity_spans([{"name": "Character 1"}], setting=None, characters=annotation["characters"])
        assert row["text"] == plain["text"] + " Setting: " + setting.rstrip(".") + "."
        assert row["entity_spans"] == plain["entity_spans"]
        only_place = reference_text_with_entity_spans([], setting=setting, characters=annotation["characters"])["text"]
        assert only_place == MEMORY_ROW_ROLE_TEXT + " Setting: " + setting.rstrip(".") + "."
        require_memory_rows([row["text"], only_place, MEMORY_ROW_ROLE_TEXT], row_policy="frame_entities_v1")
