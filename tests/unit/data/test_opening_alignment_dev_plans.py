"""开镜对齐（10-30 收束）在开发集真实计划上：「开头即终态」那几道题所在的镜，旧编排的开镜句已写终态，新规则改对；
近义写法（sitting down 对 sitting、kneeling on the floor 对 kneeling、standing up straight 对 standing straight）按实词认出；
计划没写变化、动作暗含下跪的镜把开镜姿态挪成变化。夹具是出片用过的编译计划（去掉 layouts）。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.shot_text import shot_text, shot_text_with_entity_spans
from mspc.data.storyboard import align_board_openings, posture_words

ROOT = Path(__file__).parents[2] / "fixtures/opening_alignment"


def _plan(story):
    return json.loads((ROOT / f"{story}.json").read_text())["plan"]


def _aligned(story):
    plan = copy.deepcopy(_plan(story))
    fixes = align_board_openings(plan, {e["name"]: e.get("outfit") for e in plan["characters"]})
    return plan, fixes


def _opening(text):
    return text.partition("By the end of the shot, ")[0]


@pytest.mark.parametrize("story, shot, name, end_state", [
    ("film-000127-b0000-shot01", 10, "Character 1", "is sitting down"),
    ("film-000127-b0000-shot01", 10, "Character 2", "is kneeling on the floor"),
    ("film-000129-b0000-shot01", 6, "Character 2", "is sitting down at the desk"),
    ("cinedance-013630-b0000-shot01", 6, "Character 1", "is standing up straight"),
])
def test_opening_no_longer_states_the_end_posture(story, shot, name, end_state):
    original = _plan(story)
    raw = copy.deepcopy(original)
    compiled = shot_text_with_entity_spans(raw, shot - 1)
    assert any(entry['name'] == name and entry['field'] == 'posture' for entry in compiled['opening_conflicts'])
    assert raw == original
    assert end_state not in _opening(compiled['text'])
    plan, fixes = _aligned(story)
    assert end_state not in _opening(shot_text(plan, shot - 1))
    assert any(f["shot"] == shot and f["name"] == name and f["rule"] == "开镜值取变化前" for f in fixes)


def test_implied_kneel_moves_the_posture_into_a_change():
    plan, fixes = _aligned("cinedance-010863-b0000-shot01")
    implied = {f["name"]: f for f in fixes if f["rule"] == "动作暗含的姿态变化"}
    assert set(implied) == {"Character 2", "Character 3", "Character 4"} and {f["shot"] for f in implied.values()} == {7}
    text = shot_text(plan, 6)
    opening, marker, ending = text.partition("By the end of the shot, ")
    assert marker and "is kneeling" not in opening
    for name in implied:
        assert f"{name} is kneeling" in ending
    assert implied["Character 2"]["to"] is None  # 上一镜是「looking shocked」，不是身体姿势，变化前不写


def test_posture_words_drop_fillers_and_keep_content():
    assert posture_words("is sitting down") == ("sit",)
    assert posture_words("kneeling on the floor") == ("kneel", "floor")
    assert posture_words("standing up straight") == posture_words("standing straight") == ("stand",)


def test_alignment_is_idempotent_on_the_dev_plans():
    for path in ROOT.glob("*.json"):
        plan, _ = _aligned(path.stem)
        assert align_board_openings(plan, {e["name"]: e.get("outfit") for e in plan["characters"]}) == []


def test_implied_change_goes_to_the_person_the_sentence_names():
    from mspc.data.storyboard import align_opening_with_changes

    shot = {"shot": 2, "present": [{"name": "Character 1", "posture": "sitting at a table"},
                                   {"name": "Character 2", "posture": "sitting"}],
            "action": "Character 1 sits at a table. Character 2 approaches and sits down.", "changes": []}
    fixes = align_opening_with_changes(shot, {}, {"Character 1": "standing", "Character 2": "standing"})
    assert [f["name"] for f in fixes] == ["Character 2"]
    assert shot["present"][0]["posture"] == "sitting at a table" and "posture" not in shot["present"][1]
