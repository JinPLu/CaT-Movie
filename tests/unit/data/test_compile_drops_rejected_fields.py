"""规划原答编译逐字段丢弃：夹具是底座零样本写开发集时被拒收的真实原答（10-31，旧合同：带参考镜号与人物框）。
只有这些字段走，其余不动；旧合同的参考镜号与人物框、以及规划器不再写的人物关系在编译时整项丢掉并记进报告。"""
import json

import pytest

from mspc.data.storyboard import (CHANGE_ERROR, HOLDING_ERROR, PLANNER_BOXES_DROPPED, PLANNER_REFERENCES_DROPPED,
                                  PLANNER_RELATIONS_DROPPED, UNREGISTERED_ERROR, compile_storyboard)
from tests.support.paths import REPO_ROOT

OLD_CONTRACT = {PLANNER_BOXES_DROPPED, PLANNER_REFERENCES_DROPPED}
NOT_PLANNED = OLD_CONTRACT | {PLANNER_RELATIONS_DROPPED}
# film-000129 被拒收的那一项在人物框里；规划器不再写框，整份框丢掉后没有别的错。
CASES = {"cinedance-012520-b0000-shot01": {HOLDING_ERROR},
         "film-000129-b0000-shot01": set(),
         "film-004402-b0002-shot02": {CHANGE_ERROR}}


def _raw(story):
    path = REPO_ROOT / "tests/fixtures/planner_rejected_answers" / f"{story}.json"
    return json.loads(path.read_text())["raw_storyboard"]


def _semantic(report):
    return [d for d in report["dropped_fields"] if d["reason"] not in NOT_PLANNED]


@pytest.mark.parametrize("story, reasons", CASES.items())
def test_rejected_answer_compiles_with_the_bad_field_recorded(story, reasons):
    raw = _raw(story)
    board, report = compile_storyboard(raw, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert {d["reason"] for d in _semantic(report)} == reasons
    assert {d["reason"] for d in report["dropped_fields"]} >= OLD_CONTRACT
    assert "layouts" not in board and all("references" not in shot and "relations" not in shot for shot in board["shots"])
    assert len(board["shots"]) == len(raw["shots"]) and len(board["characters"]) == len(raw["characters"])


def test_an_unregistered_person_leaves_the_structure_and_the_prose_stays():
    raw = _raw("cinedance-011901-b0000-shot01")
    board, report = compile_storyboard(raw, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    unregistered = [d for d in report["dropped_fields"] if d["reason"] == UNREGISTERED_ERROR]
    assert [d["path"] for d in unregistered] == ["shots[0].present[2]"] and unregistered[0]["value"]["name"] == "Character 3"
    assert "Character 3" not in json.dumps(board)
    assert [shot["action"] for shot in board["shots"]] == [shot["action"] for shot in raw["shots"]]


def test_every_structural_field_naming_an_unregistered_entity_goes():
    raw = _raw("film-000129-b0000-shot01")
    ghost = "Character 9"
    first = raw["shots"][0]
    holder = next(entry for entry in first["present"] if entry["name"].startswith("Character"))
    first["present"] = [*first["present"], {"name": ghost}]
    holder["holding"] = [ghost]
    first["changes"] = [*first["changes"], {"name": ghost, "what": "posture", "before": "sitting", "after": "standing"}]
    first["relations"] = [{"subject": holder["name"], "relation": "left_of", "object": ghost}]
    first["ending_sides"] = [{"name": ghost, "side": "right"}]
    board, report = compile_storyboard(raw, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert sum(d["reason"] == UNREGISTERED_ERROR for d in report["dropped_fields"]) == 4
    assert {"path": "shots[0].relations", "value": [{"subject": holder["name"], "relation": "left_of", "object": ghost}],
            "reason": PLANNER_RELATIONS_DROPPED} in report["dropped_fields"]
    assert ghost not in json.dumps(board)


def test_only_the_change_on_an_absent_person_goes():
    raw = _raw("film-004402-b0002-shot02")
    board, report = compile_storyboard(raw, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert [d["value"]["name"] for d in _semantic(report)] == ["Character 1", "Character 1"]
    before = sum(len(shot["changes"]) for shot in raw["shots"])
    assert sum(len(shot["changes"]) for shot in board["shots"]) == before - 2


def test_a_clean_plan_loses_only_the_old_contract_fields():
    clean = json.loads((REPO_ROOT / "tests/fixtures/opening_alignment/film-000127-b0000-shot01.json").read_text())["plan"]
    _, report = compile_storyboard(clean, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert _semantic(report) == []


def test_relations_go_whole_and_an_off_list_composition_and_side_go():
    raw = _raw("film-000129-b0000-shot01")
    shot = raw["shots"][0]
    people = [entry["name"] for entry in shot["present"] if entry["name"].startswith("Character")]
    shot["composition"] = "diagonal composition"
    shot["relations"] = [{"subject": people[0], "relation": "left_of", "object": people[0]},
                         {"subject": people[0], "relation": "beside", "object": people[-1]}]
    shot["ending_sides"] = [{"name": people[0], "side": "offscreen"}]
    board, report = compile_storyboard(raw, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert [d["path"] for d in report["dropped_fields"] if d["reason"] == PLANNER_RELATIONS_DROPPED] == ["shots[0].relations"]
    assert [d["path"] for d in _semantic(report)] == ["shots[0].composition", "shots[0].ending_sides[0]"]
    assert not any(key in board["shots"][0] and board["shots"][0][key] for key in ("composition", "relations", "ending_sides"))
