"""规划器文本检查：真实开发集原答（公开 9B 与语义微调各一份）上的程序检查，布局与回看规则，判官回答的折算。"""
from copy import deepcopy
import json

import pytest

from mspc.evaluation.planner_text_checks import (
    grade, judge_prompt, layout_units, plan_forms, program_checks, reference_units, semantic_checkpoints,
    semantic_requirements, summarize,
)
from tests.support.paths import REPO_ROOT

FIXTURE = json.loads((REPO_ROOT / "tests/fixtures/planner_text_checks/cinedance-010673-b0001-shot01.json").read_text())
REQUEST = next(story for story in json.loads((REPO_ROOT / "configs/benchmark/story_requests_v4.json").read_text())["stories"]
               if story["story_id"] == FIXTURE["story_id"])["inputs"]["shots"]["text"]


def test_program_checks_on_real_answers_count_presence_from_the_request_not_the_answer():
    public = program_checks(FIXTURE["records"]["public_qwen"], "shots", REQUEST, FIXTURE["checkpoints"])
    tuned = program_checks(FIXTURE["records"]["sft_qwen"], "shots", REQUEST, FIXTURE["checkpoints"])
    assert public["contract"] == {"json_parsed": True, "compiled_valid": True, "contract_accepted": True, "truncated": False}
    assert public["checks"]["shot_count_kept"] == {"pass": 1, "applicable": 1}
    # 请求点名 9 个物体、4 个人、1 个地点；分母与哪个规划器无关。
    assert public["checks"]["request_entities_registered"]["applicable"] == 14
    presence_total = sum(len(row["present"]) for row in FIXTURE["checkpoints"]["presence"])
    assert public["checks"]["request_presence_kept"]["applicable"] == presence_total == tuned["checks"]["request_presence_kept"]["applicable"]
    # 公开 9B 第 6、7 镜没有把床（Object 4）写进在场，第 10 镜没有画（Object 2）。
    assert public["checks"]["request_presence_kept"]["pass"] == presence_total - 3
    assert tuned["checks"]["request_presence_kept"]["pass"] == presence_total
    assert "layout_legal" not in public["checks"] and "reference_legal" not in public["checks"]


def test_holding_counts_as_presence_and_a_missing_raw_answer_keeps_only_the_contract():
    record = deepcopy(FIXTURE["records"]["sft_qwen"])
    shot = record["raw_storyboard"]["shots"][1]
    shot["present"] = [entry for entry in shot["present"] if entry["name"] != "Object 3"]
    assert any("Object 3" in entry.get("holding", []) for entry in shot["present"])
    assert program_checks(record, "shots", REQUEST, FIXTURE["checkpoints"])["checks"]["request_presence_kept"]["pass"] == \
        program_checks(FIXTURE["records"]["sft_qwen"], "shots", REQUEST, FIXTURE["checkpoints"])["checks"]["request_presence_kept"]["pass"]
    failed = {"status": "planning_failed", "structure_status": "json_error", "finish_reason": "length", "raw_storyboard": None}
    result = program_checks(failed, "shots", REQUEST, FIXTURE["checkpoints"])
    assert result["contract"] == {"json_parsed": False, "compiled_valid": False, "contract_accepted": False, "truncated": True}
    assert "checks" not in result


def _board_with_layouts():
    board = deepcopy(FIXTURE["records"]["sft_qwen"]["raw_storyboard"])
    board["layouts"] = [
        {"shot": 1, "entities": [
            {"name": "Character 1", "anchors": [{"time": 0.0, "visibility": "visible", "box": [0.1, 0.1, 0.5, 0.9]},
                                                {"time": 1.0, "visibility": "not_visible", "box": None}]},
            {"name": "Character 3", "anchors": [{"time": 0.2, "visibility": "visible", "box": [0.1, 0.1, 0.5, 0.9]}]}]},
        {"shot": 2, "entities": [
            {"name": "Character 2", "anchors": [{"time": 0.5, "visibility": "visible", "box": [0.6, 0.1, 0.5, 1.2]}]},
            {"name": "Character 1", "anchors": [{"time": 0.1, "visibility": "visible", "box": [0.1, 0.1, 0.5, 0.9]}] * 3}]},
    ]
    return board


def test_layout_rules_box_inside_frame_two_anchors_registered_and_present():
    units = {(unit["shot"], unit["entity"]): unit for unit in layout_units(_board_with_layouts())}
    assert units[(1, "Character 1")]["pass"]
    assert units[(1, "Character 3")]["reasons"] == ["not_present"]
    assert units[(2, "Character 2")]["reasons"] == ["box"]
    assert units[(2, "Character 1")]["reasons"] == ["anchor_count", "anchor_time"]


def test_reference_rules_point_back_to_an_earlier_shot_where_the_entity_appeared():
    board = deepcopy(FIXTURE["records"]["sft_qwen"]["raw_storyboard"])
    for shot in board["shots"]:
        shot["references"] = []
    board["shots"][2]["references"] = [{"name": "Character 1", "shot": 1}, {"name": "Character 2", "shot": 3},
                                       {"name": "Character 3", "shot": 1}]
    board["shots"][4]["references"] = [{"name": "Character 1", "shot": 1}]
    units = reference_units(board)
    assert [(unit["shot"], unit["entity"], unit["pass"], unit["most_recent"]) for unit in units] == [
        (3, "Character 1", True, False), (3, "Character 2", False, False), (3, "Character 3", False, False),
        (5, "Character 1", True, False)]


def test_judge_answers_fold_into_kept_order_roles_and_conflicts():
    requirements = semantic_requirements("shots", FIXTURE["checkpoints"])
    answers = [{"id": row["requirement_id"], "status": "stated", "shot": row["shot"],
                "roles": "correct" if row["dimension"] == "event" else "not_applicable"} for row in requirements]
    answers[0] = {**answers[0], "shot": 2}
    answers[1] = {**answers[1], "status": "contradicted", "roles": "wrong"}
    graded = grade("shots", requirements, {"checkpoints": answers, "contradictions": [{"shot": 4, "plan": "x", "request": "y"}]})
    events = sum(row["dimension"] == "event" for row in requirements)
    assert graded["event_kept"] == {"pass": events - 1, "applicable": events}
    assert graded["event_order"] == {"pass": events - 2, "applicable": events}
    assert graded["contradicted"] == 1 and graded["fabricated_conflicts"] == 1
    assert grade("shots", requirements, {"checkpoints": answers[1:], "contradictions": []})["measured"] is False


def test_written_checkpoints_split_events_and_results_and_order_is_checked_by_plan_shots():
    rows = semantic_checkpoints([
        {"question": "Does the man hand the case to the other man?", "doer": "the man", "receiver": "the other man",
         "result": "Does the other man hold the case?"},
        {"question": "Do they ride in a vehicle?", "doer": "the two men", "receiver": None, "result": None}])
    assert [row["requirement_id"] for row in rows] == ["e01", "e01r", "e02"]
    answer = {"checkpoints": [{"id": "e01", "status": "stated", "shot": 3, "roles": "correct"},
                              {"id": "e01r", "status": "absent", "shot": None, "roles": "not_applicable"},
                              {"id": "e02", "status": "stated", "shot": 2, "roles": "correct"}], "contradictions": []}
    graded = grade("synopsis", rows, answer)
    assert graded["event_order"] == {"pass": 0, "applicable": 1}
    assert graded["state_kept"] == {"pass": 0, "applicable": 1}


def test_holdout_writer_checkpoints_keep_semantic_scores_and_report_unmeasured_shot_order():
    failure = json.loads((REPO_ROOT / "tests/fixtures/planner_text_checks/holdout-cinedance-009700-b0002-missing-shot.json").read_text())
    graded = grade(failure["tier"], failure["requirements"], failure["answer"])
    assert graded["measured"] is True
    assert graded["event_kept"] == graded["state_kept"] == graded["roles_correct"] == {"pass": 2, "applicable": 2}
    assert graded["contradicted"] == graded["fabricated_conflicts"] == 0
    assert "event_order" not in graded and "state_order" not in graded
    assert graded["unmeasured"] == {
        "event_order": {"checkpoints": 2, "requirement_ids": ["e01", "e02"],
                        "reason": "frozen_checkpoint_missing_request_shot"},
        "state_order": {"checkpoints": 2, "requirement_ids": ["e01r", "e02r"],
                        "reason": "frozen_checkpoint_missing_request_shot"}}
    story = failure["story_id"]
    program = {("shots", story): {"status": "completed", "contract": {
        "json_parsed": True, "compiled_valid": True, "contract_accepted": True, "truncated": False},
        "compile_repairs": {"holding_presence_added": 0, "opening_aligned_with_changes": 0}, "checks": {}}}
    summary = summarize({story: {}}, {("shots", story): {}}, program, {("shots", story, "raw"): graded})["shots"]["judge_raw"]
    assert summary["event_kept"]["pass"] == 2 and "event_order" not in summary
    assert summary["unmeasured"]["event_order"] == {
        "checkpoints": 2, "plans": 1, "reason_counts": {"frozen_checkpoint_missing_request_shot": 2}}


def test_shot_order_only_counts_frozen_checkpoints_with_explicit_shot_numbers():
    requirements = deepcopy(semantic_requirements("shots", FIXTURE["checkpoints"]))
    answers = [{"id": row["requirement_id"], "status": "stated", "shot": row["shot"],
                "roles": "correct" if row["dimension"] == "event" else "not_applicable"} for row in requirements]
    event = next(row for row in requirements if row["dimension"] == "event")
    state = next(row for row in requirements if row["dimension"] == "state")
    del event["shot"], state["shot"]
    graded = grade("shots", requirements, {"checkpoints": answers, "contradictions": []})
    for metric, dimension in (("event_order", "event"), ("state_order", "state")):
        applicable = sum(row["dimension"] == dimension for row in requirements) - 1
        assert graded[metric] == {"pass": applicable, "applicable": applicable}
        assert graded["unmeasured"][metric]["checkpoints"] == 1


def test_compiled_text_is_judged_for_every_compiled_plan():
    public, tuned = FIXTURE["records"]["public_qwen"], FIXTURE["records"]["sft_qwen"]
    assert set(plan_forms(tuned)) == ({"raw", "compiled"} if tuned["status"] == "completed" else {"raw"})
    forms = plan_forms(public)
    assert set(forms) == {"raw", "compiled"} and forms["compiled"].startswith("Shot 1: ")
    system, prompt = judge_prompt(REQUEST, forms["raw"], "raw", semantic_requirements("shots", FIXTURE["checkpoints"]), "shots")
    assert "json" in system and "EXAMPLE JSON OUTPUT" in system and REQUEST not in system
    assert "request shot 3, by the end of the shot" in prompt and REQUEST in prompt


def test_deepseek_call_ledger_keeps_receipts_checks_schema_and_never_resends(tmp_path, monkeypatch):
    import io
    import json as js
    from mspc.evaluation import planner_text_checks as checks
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            body = js.loads(request.data)
            calls.append(body)
            assert request.headers["Authorization"] == "Bearer test-key" and body["model"] == "deepseek-flash"
            assert [m["role"] for m in body["messages"]] == ["system", "user"]
            assert "temperature" not in body and body["reasoning_effort"] == "high" and body["response_format"] == {"type": "json_object"}
            prompt = body["messages"][1]["content"]
            if "blank" in prompt:
                content = "   "
            elif "bad" in prompt:
                content = '{"checkpoints": [{"id": "q01"}], "contradictions": []}'
            else:
                content = '{"checkpoints": [], "contradictions": []}'
            return io.BytesIO(js.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                        "usage": {"total_tokens": 9}}).encode())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    ledger = checks.CallLedger(tmp_path / "calls", checks.EFFORT)
    ledger.opener = FakeOpener()
    first = ledger.call("p/shots/raw/s1", checks.JUDGE_SYSTEM, "good prompt", checks.JUDGE_SCHEMA)
    assert first["status"] == "answered" and first["result"] == {"checkpoints": [], "contradictions": []}
    assert "test-key" not in js.dumps(first) and first["user"] == "good prompt"
    again = ledger.call("p/shots/raw/s1", checks.JUDGE_SYSTEM, "good prompt", checks.JUDGE_SCHEMA)
    assert again == first and len(calls) == 1
    failed = ledger.call("p/shots/raw/s2", checks.JUDGE_SYSTEM, "bad prompt", checks.JUDGE_SCHEMA)
    assert failed["status"] == "format_failed" and "result" not in failed
    assert len(calls) == 2
    assert ledger.call("p/shots/raw/s2", checks.JUDGE_SYSTEM, "bad prompt", checks.JUDGE_SCHEMA)["status"] == "format_failed" and len(calls) == 2
    blank = ledger.call("p/shots/raw/s3", checks.JUDGE_SYSTEM, "blank prompt", checks.JUDGE_SCHEMA)
    assert blank["status"] == "blank_answer" and len(calls) == 3
    ledger.call("p/shots/raw/s3", checks.JUDGE_SYSTEM, "blank prompt", checks.JUDGE_SCHEMA)
    assert len(calls) == 4  # 空回答下次重发
    with pytest.raises(ValueError, match="different messages"):  # 换了计划却沿用同名回执，不能拿旧判词充数
        ledger.call("p/shots/raw/s1", checks.JUDGE_SYSTEM, "another plan", checks.JUDGE_SCHEMA)
    assert len(calls) == 4
