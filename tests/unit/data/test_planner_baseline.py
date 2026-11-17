"""零样本基线的评估与逐镜档写手：真实 v3 日志与原生字幕；模型回复由脚本化的假 complete 给出，只验接线与规则。"""
import json
from pathlib import Path

import pytest

from mspc.data.event_log import entry_labels, fact_roles, parse_event_log, script_name, validate_event_log
from mspc.data.planner_baseline import (
    STORYBOARD_LABEL_INSTRUCTIONS, STORYBOARD_WRITER_INSTRUCTIONS, _WRITER_EXAMPLE_OUTPUTS, _writer_example_logs, _writer_example_spatial,
    WriterCallRecorder, complete_shot_tier_call, drawn_spatial_entries, fact_table, format_shot_tier,
    load_shot_tier_rewrite_config, native_leakage,
    parse_storyboard_labels, parse_storyboard_writer, run_storyboard_writer, scalar_counts, storyboard_label_messages,
    storyboard_writer_messages, storyboard_writer_retry_messages, storyboard_writer_summary, wilson,
    write_storyboard_script,
)

ROOT = Path(__file__).resolve().parents[3]
REAL = json.loads((ROOT / "tests/fixtures/event_log_real_slice/05.json").read_text())["log"]
KITCHEN = json.loads((ROOT / "tests/fixtures/event_log_real_slice/04.json").read_text())["log"]
P1A01 = json.loads((ROOT / "tests/fixtures/event_log_v3_reference/p1a-01.json").read_text())["event_log"]
NONE = frozenset()


def test_real_log_unknown_denominator():
    scalar = scalar_counts(REAL)
    assert scalar["unknown"] > 0
    assert scalar["total"] == sum(v for k, v in scalar.items() if k.startswith("total:"))
    assert scalar["unknown"] == sum(v for k, v in scalar.items() if k.startswith("unknown:"))


def test_native_copy_audit_on_actual_captions():
    captions = [s["action"] for s in REAL["shots"]]
    leak = native_leakage(captions[0], captions)
    assert leak["exact_copied_sentences"]
    assert leak["copied_8word_ratio"] == 1
    assert native_leakage("The athlete dives and then swims.", captions)["copied_8word_positions"] == 0


def test_wilson_interval_math():
    low, high = wilson(50, 100)
    assert abs((low + high) / 2 - 0.5) < 1e-10
    assert low < 0.5 < high
    assert wilson(0, 0) is None


def test_real_short_subtitle_quote_is_not_hidden_by_eight_word_threshold():
    from mspc.data.planner_baseline import quoted_annotation_leakage
    source = json.loads((ROOT / "tests/fixtures/planner/native_subtitle_quotes.json").read_text())
    detected = quoted_annotation_leakage("Luck", source["native_captions"])
    assert any(q["text"] == "Luck" and q["exact_copied"] and
               q["context_class"] == "subtitle_or_dialogue_annotation" for q in detected["quotes"])
    assert detected["exact_copied_candidate_count"] > 0
    assert native_leakage("Luck", source["native_captions"])["copied_8word_positions"] == 0


def test_invented_writer_examples_are_valid_logs_with_both_marks():
    for raw in _writer_example_logs():
        log = parse_event_log(raw)
        validate_event_log(log)
        roles = {e.role for e in fact_roles(log)}
        assert roles == {"cause", "derivable", "spatial"}  # 空间事实单列，不进表


def test_real_writer_request_marks_causes_for_writer_and_checker():
    log = parse_event_log(KITCHEN)
    messages = storyboard_writer_messages(log, spatial=NONE)
    assert messages[0]["content"].startswith(STORYBOARD_WRITER_INSTRUCTIONS)
    assert messages[0]["content"].count("EXAMPLE OUTPUT") == 2
    table = fact_table(fact_roles(log), marks=True, spatial=NONE)
    assert messages[1]["content"].endswith(table)
    assert "WRITE" in table and "[1] WRITE Character 1 (person) looks like" in table
    check = storyboard_label_messages(log, "s01: Subject 1 freezes.", spatial=NONE)
    assert check[0]["content"] == STORYBOARD_LABEL_INSTRUCTIONS
    assert check[1]["content"].split("\n\nSCRIPT\n")[0].endswith(table)  # 核对表带标记，用来判复述
    retry = storyboard_writer_retry_messages(log, ["Subject 1 eats cereal"], spatial=NONE)
    assert retry[1]["content"].endswith("- Subject 1 eats cereal")
    with pytest.raises(ValueError):
        storyboard_writer_retry_messages(log, [], spatial=NONE)


def test_invented_examples_show_the_input_habits_of_the_evaluation_scripts():
    """示例按评测逐镜剧本的写法：缺省机位、静止、未换地点不写；地点用描述不用编号；只有抽中的镜写方位；
    景别、机位、运镜用统一词表的词（Medium shot、Low angle shot、camera pushes in），方位写在人物前面。"""
    tables = []
    for index, raw in enumerate(_writer_example_logs()):
        entries = fact_roles(parse_event_log(raw))
        tables.append(fact_table(entries, marks=True, spatial=_writer_example_spatial(index, entries)))
    assert "DO-NOT-WRITE camera angle: eye level" in tables[0] and "DO-NOT-WRITE camera movement: static" in tables[0]
    assert "WRITE camera movement: camera pushes in" in tables[0] and "WRITE camera angle: Low angle shot" in tables[1]
    assert "WRITE shot size: Wide shot" in tables[0] and "WRITE shot size: Medium close-up shot" in tables[1]
    assert all(table.count("DO-NOT-WRITE takes place in Location 1") == 1 for table in tables)
    # 示例一首镜是抽中写方位的镜，人物左右一条空间条目列在该镜下；示例二没有抽中的镜。
    first_shot = tables[0].split("SHOT s02")[0]
    assert "WRITE Character 1 is on the left" in first_shot and "of the frame" not in tables[0] + tables[1]
    prompts = [shot["prompt"] for output in _WRITER_EXAMPLE_OUTPUTS for shot in output["shots"]]
    assert not any("Location" in prompt or "eye level" in prompt or "static" in prompt for prompt in prompts)
    assert "On the left, Character 1" in prompts[0] and not any("of the frame" in prompt for prompt in prompts)


def test_real_window_spatial_facts_enter_the_writer_table_only_for_drawn_shots():
    log = parse_event_log(P1A01)
    entries = fact_roles(log)
    table = fact_table(entries, marks=True, spatial=NONE)
    # p1a-01 有 27 条真实空间事实（位置、深度、朝向、视线）；没抽中的镜，写手表和核对表里一条也没有。
    assert sum(e.kind == "screen" for e in entries) == 27
    assert not any(word in table for word in ("of the frame", "foreground", "facing", "looks at", "bbox"))
    assert storyboard_writer_messages(log, spatial=NONE)[1]["content"].endswith(table)
    spatial = next(e for e in entries if e.kind == "screen")
    with pytest.raises(ValueError, match="written"):
        parse_storyboard_labels(json.dumps({"written": [spatial.number], "restated": [], "added_facts": []}), entries, spatial=NONE)
    causes = [e.number for e in entries if e.role == "cause"]
    classes = entry_labels(log, causes)
    assert len(classes["spatial"]) == 27 and set(classes["stated"]) == set(causes)  # 空间类单列，不并入写明
    # 抽签只看 (种子, 窗, 镜)：比例 1 抽中全部有人物左右的镜，比例 0 一镜不抽；同一键重复调用结果相同。
    # 抽中的镜只写人物的左右：前后、朝向、视线和物体（p1a-01 的电钻、盒子）的方位一律不写，留给规划器推。
    everything = drawn_spatial_entries(log, entries, case_id="p1a-01", fraction=1.0, seed=42)
    person_lateral = {e.number for e in entries if e.kind == "screen" and e.fact.subject == "Subject 1"
                      and e.fact.predicate_or_axis == "horizontal"}
    assert everything == person_lateral and len(everything) == 4
    assert sum(e.kind == "screen" and e.fact.predicate_or_axis == "horizontal" for e in entries) > len(everything)  # 物体左右也有
    assert drawn_spatial_entries(log, entries, case_id="p1a-01", fraction=0.0, seed=42) == NONE
    some = drawn_spatial_entries(log, entries, case_id="p1a-01", fraction=0.5, seed=42)
    assert some == drawn_spatial_entries(log, entries, case_id="p1a-01", fraction=0.5, seed=42) and some <= everything
    with pytest.raises(ValueError, match="fraction"):
        drawn_spatial_entries(log, entries, case_id="p1a-01", fraction=1.5, seed=42)
    # 抽中的空间条目列在所属镜下、标 WRITE，核对可以把它们记为写到；标签里仍在空间类，另记 spatial_written。
    one_shot = frozenset(e.number for e in entries if e.number in everything and e.shot_id == "s02")
    drawn_table = fact_table(entries, marks=True, spatial=one_shot)
    block = drawn_table.split("SHOT s02")[1].split("SHOT s03")[0]
    assert all(f"[{n}] WRITE " in block for n in one_shot) and len(drawn_table.splitlines()) == len(table.splitlines()) + len(one_shot)
    complete, _ = scripted(log, shot_written=causes + sorted(one_shot)[:2])
    result = write_storyboard_script(log, complete, spatial=one_shot)
    assert result["spatial_written"] == sorted(one_shot)[:2] and result["spatial_write"] == sorted(one_shot)
    assert result["labels"] == entry_labels(log, causes) and set(one_shot) <= set(result["labels"]["spatial"])
    with pytest.raises(ValueError, match="spatial"):
        fact_table(entries, marks=True, spatial=frozenset({causes[0]}))


def writer_output(log, prompt="Subject 1 stands in Location 1."):
    return {"subjects": [{"subject_id": script_name(s.subject_id, s.kind), "kind": s.kind, "attributes": s.appearance} for s in log.subjects],
            "locations": [{"location_id": l.location_id, "attributes": l.appearance} for l in log.locations],
            "shots": [{"shot_id": s.shot_id, "prompt": prompt} for s in log.shots]}


def test_real_writer_parse_requires_the_log_ids():
    log = parse_event_log(KITCHEN)
    good = writer_output(log)
    script = parse_storyboard_writer(json.dumps(good), log)
    assert format_shot_tier(script).startswith("SUBJECTS\n- Character 1 (person):")
    bad = writer_output(log)
    bad["shots"] = bad["shots"][:-1]
    with pytest.raises(ValueError, match="shots"):
        parse_storyboard_writer(json.dumps(bad), log)
    with pytest.raises(ValueError, match="written"):
        parse_storyboard_labels(json.dumps({"written": [999], "restated": [], "added_facts": []}), fact_roles(log), spatial=NONE)


def scripted(log, *, shot_added=(), shot_written=None, shot_restated=()):
    """按系统提示分派的假模型：写手写全部起因；shot_added 依次给出每次核对报的表外事实。

    shot_written／shot_restated 覆盖核对报的写到与复述编号。
    """
    entries = fact_roles(log)
    causes = [e.number for e in entries if e.role == "cause"]
    shot_checks = list(shot_added)
    calls = []

    def complete(messages):
        system = messages[0]["content"]
        calls.append(system[:40])
        if system.startswith(STORYBOARD_WRITER_INSTRUCTIONS):
            return json.dumps(writer_output(log))
        if system == STORYBOARD_LABEL_INSTRUCTIONS:
            added = shot_checks.pop(0) if shot_checks else []
            return json.dumps({"written": causes if shot_written is None else shot_written,
                               "restated": list(shot_restated), "added_facts": added})
        raise AssertionError(system[:80])
    return complete, calls


def test_real_script_gets_labels_from_the_checker_and_the_log():
    log = parse_event_log(KITCHEN)
    complete, calls = scripted(log)
    result = write_storyboard_script(log, complete, spatial=NONE)
    assert result["status"] == "kept" and result["text"] == format_shot_tier(result["script"])
    entries = fact_roles(log)
    causes = [e.number for e in entries if e.role == "cause"]
    assert result["labels"] == entry_labels(log, causes) and result["restated"] == []
    assert len(calls) == 2  # 写、核


def test_real_added_fact_rewrites_once_then_discards_the_window():
    log = parse_event_log(KITCHEN)
    complete, calls = scripted(log, shot_added=[["Subject 1 eats cereal"]])
    once = write_storyboard_script(log, complete, spatial=NONE)
    assert once["status"] == "kept" and once["attempts"] == 2 and len(calls) == 4
    complete, calls = scripted(log, shot_added=[["Subject 1 eats cereal"], ["Subject 1 eats cereal"]])
    twice = write_storyboard_script(log, complete, spatial=NONE)
    assert twice["status"] == "discarded" and twice["reason"] == "added_facts_after_retry" and "labels" not in twice


def test_writer_summary_rates_share_one_denominator():
    """复述率分母只算经过核对的窗；丢弃按原因另报。

    p1a-01 是带 DO-NOT-WRITE 条目的真实 v3 窗（23 条起因、5 条可推出）。
    """
    log = parse_event_log(P1A01)
    entries = fact_roles(log)
    causes = [e.number for e in entries if e.role == "cause"]
    derivable = [e.number for e in entries if e.role == "derivable"]
    assert derivable
    total = len(derivable)
    runs = [scripted(log, shot_written=causes + derivable[:2], shot_restated=derivable[:1]),
            scripted(log, shot_added=[["Subject 1 eats cereal"], ["Subject 1 eats cereal"]]),
            scripted(log)]
    rows = []
    for index, (complete, _) in enumerate(runs):
        result = write_storyboard_script(log, complete, spatial=NONE)
        rows.append({"case_id": f"w{index}", "derivable_entries": total, **result, "leakage": {"exact_copied_sentences": []}})
    summary = storyboard_writer_summary(rows)
    assert (summary["added_fact_rate"]["numerator"], summary["added_fact_rate"]["denominator"]) == (1, 3)
    assert (summary["restated_entry_rate"]["numerator"], summary["restated_entry_rate"]["denominator"]) == (1, 3 * total)
    assert summary["discarded"] == {"added_facts_after_retry": 1} and summary["kept"] == 2 and summary["errors"] == 0


def test_real_writer_run_writes_rows_and_summary(tmp_path):
    row = {"case_id": "ci_vid:train:000000753", "status": "accepted", "event_log": KITCHEN,
           "native_captions": [s["action"] for s in KITCHEN["shots"]]}
    windows = tmp_path / "windows.jsonl"
    windows.write_text(json.dumps(row) + "\n" + json.dumps({**row, "case_id": "x", "status": "rejected"}) + "\n")
    complete, _ = scripted(parse_event_log(KITCHEN))
    run_storyboard_writer(windows_path=windows, output=tmp_path / "out", config={"model": "test"}, expected_windows=1,
                          spatial_fraction=0.23, spatial_seed=42, complete=complete, spent=lambda: "0",
                          recorder=WriterCallRecorder(tmp_path / "out/attempts"))
    written = [json.loads(line) for line in (tmp_path / "out/writer.jsonl").read_text().splitlines()]
    assert len(written) == 1 and "exact_copied_sentences" in written[0]["leakage"] and written[0]["status"] == "kept"
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["kept"] == 1
    assert summary["added_fact_rate"]["numerator"] == 0
    with pytest.raises(ValueError, match="count"):
        run_storyboard_writer(windows_path=windows, output=tmp_path / "again", config={}, expected_windows=2,
                              spatial_fraction=0.23, spatial_seed=42, complete=complete, spent=lambda: "0",
                              recorder=WriterCallRecorder(tmp_path / "again/attempts"))


def test_real_writer_run_resumes_after_a_connection_error_and_skips_written_windows(tmp_path):
    base = {"status": "accepted", "event_log": KITCHEN, "native_captions": [s["action"] for s in KITCHEN["shots"]]}
    windows = tmp_path / "windows.jsonl"
    windows.write_text("".join(json.dumps({**base, "case_id": f"w{i}"}) + "\n" for i in range(3)))
    log = parse_event_log(KITCHEN)
    args = dict(windows_path=windows, output=tmp_path / "out", config={"model": "test"}, expected_windows=3,
                spatial_fraction=0.23, spatial_seed=42, spent=lambda: "0", recorder=WriterCallRecorder(tmp_path / "out/attempts"))
    complete, calls = scripted(log)

    def drops_the_connection_in_the_second_window(messages):
        if len(calls) == 2 * 1 + 1:  # 第一窗两次调用写完，第二窗写手之后断开
            raise ConnectionError("incomplete chunked read")
        return complete(messages)

    with pytest.raises(ConnectionError):
        run_storyboard_writer(complete=drops_the_connection_in_the_second_window, **args)
    first = [json.loads(line) for line in (tmp_path / "out/writer.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in first] == ["w0"] and not (tmp_path / "out/summary.json").exists()
    # 参数不同不能续跑；目录里没有写手运行也不能当续跑。
    with pytest.raises(ValueError, match="different config"):
        run_storyboard_writer(complete=complete, **{**args, "spatial_seed": 7})
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no writer run"):
        run_storyboard_writer(complete=complete, **{**args, "output": tmp_path / "empty",
                                                    "recorder": WriterCallRecorder(tmp_path / "empty/attempts")})
    resumed_complete, resumed_calls = scripted(log)
    assert run_storyboard_writer(complete=resumed_complete, **args) == 1
    rows = [json.loads(line) for line in (tmp_path / "out/writer.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in rows] == ["w0", "w1", "w2"] and rows[0] == first[0]
    assert len(resumed_calls) == 2 * 2  # 只补了后两窗，已写完的第一窗没有重发
    assert json.loads((tmp_path / "out/summary.json").read_text())["windows"] == 3
    again, again_calls = scripted(log)
    assert run_storyboard_writer(complete=again, **args) == 3 and again_calls == []  # 全部写完后再跑不发任何调用
    # 每窗一个 attempts 目录；假 complete 不经客户端，所以目录里只有 attempt.json。第二窗断开那次留着 running 的现场，
    # 续跑重发的是 -retry1。
    states = [json.loads(p.read_text()) for p in sorted((tmp_path / "out/attempts").glob("*/attempt.json"))]
    assert [(st["attempt_id"], st["window_id"], st["status"], st["calls"]) for st in states] == [
        ("w0", "w0", "succeeded", []), ("w1", "w1", "running", []), ("w1-retry1", "w1", "succeeded", []), ("w2", "w2", "succeeded", [])]


def recorded_writer(tmp_path, log, *, break_at=None):
    """真 OpenRouterClient＋假传输：脚本化模型的答复包成 SDK 响应体，请求原样经传输层，落盘走 WriterCallRecorder。"""
    from mspc.data.openrouter_client import build_openrouter_client
    config = load_shot_tier_rewrite_config(ROOT / "configs/planner/storyboard_writer.json")
    recorder = WriterCallRecorder(tmp_path / "out/attempts")
    model, calls = scripted(log)

    def transport(**request):
        if break_at is not None and len(calls) == break_at:
            raise ConnectionError("incomplete chunked read")
        number = len(calls)
        text = model(request["messages"])
        return {"id": f"gen-{number}", "model": request["model"], "provider": "Google AI Studio",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}}

    client = build_openrouter_client(
        api_key="test-key", pricing={key: config[key] for key in ("base_url", "model")} | {
            "hard_cap_usd": config["budget_hard_cap_usd"], "input_usd_per_mtok": config["input_usd_per_mtok"],
            "output_usd_per_mtok": config["output_usd_per_mtok"]},
        precall_prompt_tokens=config["precall_prompt_tokens"], precall_completion_tokens=config["precall_completion_tokens"],
        timeout_seconds=config["timeout_seconds"], transport=transport, recorder=recorder)

    def complete(messages):
        return complete_shot_tier_call(client, messages, model=config["model"], temperature=config["temperature"],
                                       max_tokens=config["max_tokens"], request_options=config["request_options"])

    windows = tmp_path / "windows.jsonl"
    windows.write_text(json.dumps({"case_id": "ci_vid:train:000000753", "status": "accepted", "event_log": KITCHEN,
                                   "native_captions": [s["action"] for s in KITCHEN["shots"]]}) + "\n")
    args = dict(windows_path=windows, output=tmp_path / "out", config={"model": "test"}, expected_windows=1, spatial_fraction=0.23,
                spatial_seed=42, complete=complete, recorder=recorder, spent=lambda: client.cost_counter.snapshot()["spent_usd"])
    return config, args, calls


def test_writer_records_each_request_and_raw_answer_in_the_annotation_attempt_layout(tmp_path):
    log = parse_event_log(KITCHEN)
    config, args, _ = recorded_writer(tmp_path, log)
    run_storyboard_writer(**args)
    attempt = tmp_path / "out/attempts/ci_vid-train-000000753"
    state = json.loads((attempt / "attempt.json").read_text())
    assert (state["attempt_id"], state["window_id"], state["status"], state["error"]) == (
        "ci_vid-train-000000753", "ci_vid:train:000000753", "succeeded", None)
    assert [call["call"] for call in state["calls"]] == ["01-write", "02-label"]
    assert state["calls"][0]["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
    assert (state["calls"][0]["request_id"], state["calls"][0]["finish_reason"], state["calls"][0]["provider"]) == ("gen-0", "stop", "Google AI Studio")
    assert sorted(path.name for path in attempt.iterdir()) == sorted(
        ["attempt.json"] + [f"{call['call']}_{part}.json" for call in state["calls"] for part in ("request", "response", "raw")])
    # 请求是实际交给传输层的那份：模型、采样、reasoning／provider 选项和消息逐项相同。
    entries = fact_roles(log)
    spatial = drawn_spatial_entries(log, entries, case_id="ci_vid:train:000000753", fraction=0.23, seed=42)
    request = json.loads((attempt / "01-write_request.json").read_text())
    assert request == {"model": config["model"], "temperature": config["temperature"], "max_tokens": config["max_tokens"],
                       "response_format": {"type": "json_object"}, "messages": storyboard_writer_messages(log, spatial=spatial),
                       **config["request_options"]}
    # 原始答复：模型原文一字不差，响应体带 id、用量与耗时。
    raw = json.loads((attempt / "01-write_raw.json").read_text())
    assert json.loads(raw["text"]) == writer_output(log)
    response = json.loads((attempt / "01-write_response.json").read_text())
    assert response["error"] is None and response["elapsed_seconds"] >= 0
    assert response["response"]["id"] == "gen-0" and response["response"]["choices"][0]["message"]["content"] == raw["text"]
    row = json.loads((tmp_path / "out/writer.jsonl").read_text())
    assert row["script"] == json.loads(raw["text"]) and row["text"] == format_shot_tier(row["script"])
    label = json.loads(json.loads((attempt / "02-label_raw.json").read_text())["text"])
    assert row["written"] == label["written"]


def test_writer_records_a_failed_call_and_a_restarted_window_gets_a_retry_directory(tmp_path):
    log = parse_event_log(KITCHEN)
    _, args, _ = recorded_writer(tmp_path, log, break_at=1)  # 写手之后，第二次调用（核对）断开
    with pytest.raises(ConnectionError):
        run_storyboard_writer(**args)
    attempt = tmp_path / "out/attempts/ci_vid-train-000000753"
    state = json.loads((attempt / "attempt.json").read_text())
    assert state["status"] == "running" and [call["call"] for call in state["calls"]] == ["01-write", "02-label"]
    assert state["calls"][1]["error"] == "incomplete chunked read" and state["calls"][1]["request_id"] is None
    failed = json.loads((attempt / "02-label_response.json").read_text())
    assert failed["error"] == "incomplete chunked read" and failed["response"] is None
    assert (attempt / "02-label_request.json").is_file() and not (attempt / "02-label_raw.json").exists()
    assert (tmp_path / "out/writer.jsonl").read_text() == ""
    _, resumed, _ = recorded_writer(tmp_path, log)  # 同一输出目录续跑：这一窗没写完，重发，目录接 -retry1
    run_storyboard_writer(**resumed)
    retry = tmp_path / "out/attempts/ci_vid-train-000000753-retry1"
    assert json.loads((retry / "attempt.json").read_text())["status"] == "succeeded" and len(list(retry.glob("*_raw.json"))) == 2
    assert json.loads((attempt / "attempt.json").read_text())["status"] == "running"  # 上次的现场原样留着


def test_writer_config_follows_the_official_gemini_3_8_flash_values_and_the_loader_takes_explicit_temperature(tmp_path):
    """写手配置取官方值（Gemini 3 指南：温度保持默认 1.0；3.8 Flash 思考档只有 low／medium／high，缺省 medium，minimal 不支持；
    思考页：开思考时不要设小的 max_output_tokens，模型输出上限 65536）。冻结的评测反写配方仍是温度 0，加载器两者都收。"""
    writer = load_shot_tier_rewrite_config(ROOT / "configs/planner/storyboard_writer.json")
    assert (writer["temperature"], writer["request_options"]["reasoning"]) == (1.0, {"effort": "medium"})
    assert writer["max_tokens"] == 65536 == writer["precall_completion_tokens"]
    assert load_shot_tier_rewrite_config(ROOT / "configs/planner/shot_tier_rewrite.json")["temperature"] == 0  # 冻结配方（温度 0）照样加载
    for bad in (True, "1.0", -0.1, 2.5, None):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({**writer, "temperature": bad}))
        with pytest.raises(ValueError, match="temperature"):
            load_shot_tier_rewrite_config(path)


def test_writer_recorder_must_point_at_the_run_output_attempts(tmp_path):
    log = parse_event_log(KITCHEN)
    _, args, _ = recorded_writer(tmp_path, log)
    with pytest.raises(ValueError, match="recorder must write to"):
        run_storyboard_writer(**{**args, "recorder": WriterCallRecorder(tmp_path / "elsewhere")})


MISSES = {row["case_id"]: row for row in
          json.loads((ROOT / "tests/fixtures/planner/writer_first_run_checker_misses.json").read_text())["rows"]}


def test_real_state_implied_by_a_write_action_is_written_not_restated():
    """首跑 R02：s02 的 WRITE 动作句已蕴含两条 DO-NOT-WRITE 状态；它们算写到（标签进写明），复述只取核对器另报的。"""
    row = MISSES["ci_vid:train:000236318"]
    log = parse_event_log(row["event_log"])
    entries = fact_roles(log)
    implied = [e.number for e in entries if e.role == "derivable" and e.kind == "fact" and e.shot_id == "s02"]
    action = next(e for e in entries if e.shot_id == "s02" and e.target == "action")
    assert len(implied) == 2 and "stirs the ice water" in action.text
    assert sorted(row["run1_checks"]["shot"]["restated"]) == implied  # 旧口径：写到的 DO-NOT-WRITE 全算复述
    table = [e.number for e in entries if e.role != "spatial"]
    check = parse_storyboard_labels(json.dumps({"written": table, "restated": [], "added_facts": []}), entries, spatial=NONE)
    assert check["restated"] == [] and set(implied) <= set(check["written"])
    assert set(implied) <= set(entry_labels(log, check["written"])["stated"])
    restated = parse_storyboard_labels(json.dumps({"written": table, "restated": implied[:1], "added_facts": []}), entries, spatial=NONE)
    assert restated["restated"] == implied[:1]
    with pytest.raises(ValueError, match="DO-NOT-WRITE"):
        parse_storyboard_labels(json.dumps({"written": table, "restated": [action.number], "added_facts": []}), entries, spatial=NONE)
    with pytest.raises(ValueError, match="also be written"):
        parse_storyboard_labels(json.dumps({"written": [action.number], "restated": implied[:1], "added_facts": []}), entries, spatial=NONE)
    script = storyboard_label_messages(log, "s01: Subject 1 stirs the ice water.", spatial=NONE)[1]["content"]
    assert "DO-NOT-WRITE" in script and script.endswith("SCRIPT\ns01: Subject 1 stirs the ice water.")


def test_real_dev30_animal_left_right_is_drawn_but_its_depth_and_facing_are_not():
    from mspc.benchmark.long_story_event_log import event_log_from_script, r3_frame_counts
    from mspc.evaluation.cross_shot_layout import with_screen_facts
    pair = next(p for p in json.loads((ROOT / "tests/fixtures/long_story_event_log/dev30_b_state_in_appearance.json").read_text())["pairs"]
                if p["story_id"] == "lsh-023")
    candidate = json.loads((ROOT / "configs/benchmark/script_state_bench_long_holdout_v2_candidate_r3.json").read_text())
    raw = event_log_from_script(pair["context_b"], pair["change"], r3_frame_counts(candidate, "lsh-023"))
    layouts = [{} for _ in raw["shots"]]
    layouts[0] = {"Subject 1": {"horizontal": "left", "depth": "foreground", "orientation": "toward camera"}}
    log = parse_event_log(with_screen_facts(raw, layouts, axes=("horizontal", "depth", "orientation")))
    entries = fact_roles(log)
    assert [e.fact.predicate_or_axis for e in entries if e.role == "spatial"] == ["horizontal", "depth", "orientation"]
    drawn = drawn_spatial_entries(log, entries, case_id="lsh-023", fraction=1.0, seed=42)
    assert [e.fact.predicate_or_axis for e in entries if e.number in drawn] == ["horizontal"]


SPATIAL_EVENT_WINDOWS = json.loads((ROOT / "tests/fixtures/planner/writer_spatial_event_windows.json").read_text())["rows"]


@pytest.mark.parametrize("window", SPATIAL_EVENT_WINDOWS, ids=lambda w: w["case_id"])
def test_real_windows_with_spatial_event_entries_draw_person_lateral_only_and_build_requests(window):
    """10-10 夜写手第二跑的 156 个接受窗里有 5 窗的声明表带空间事件条目（kind=event，fact 为 None）；抽签曾对它们抛 AttributeError。
    抽中的只能是人物与动物的左右屏幕事实；空间事件和只说画面方位的声明不进候选，写手与核对请求都能构造。"""
    log = parse_event_log(window["event_log"])
    entries = fact_roles(log)
    assert any(e.role == "spatial" and e.kind == "event" and e.fact is None for e in entries)
    kinds = {s.subject_id: s.kind for s in log.subjects}
    lateral = {e.number for e in entries if e.role == "spatial" and e.kind == "screen" and e.fact.predicate_or_axis == "horizontal"
               and kinds[e.fact.subject] in ("person", "animal")}
    assert lateral and drawn_spatial_entries(log, entries, case_id=window["case_id"], fraction=1.0, seed=42) == lateral
    assert drawn_spatial_entries(log, entries, case_id=window["case_id"], fraction=0.0, seed=42) == NONE
    drawn = drawn_spatial_entries(log, entries, case_id=window["case_id"], fraction=0.23, seed=42)  # 第二跑的参数
    assert drawn <= lateral
    user = storyboard_writer_messages(log, spatial=drawn)[1]["content"]
    assert all(f"[{n}] WRITE " in user for n in drawn)
    assert not any(f"[{e.number}] " in user for e in entries if e.role == "spatial" and e.number not in drawn)
    assert storyboard_label_messages(log, "script", spatial=drawn)[1]["content"].count("WRITE ") >= len(drawn)
    complete, _ = scripted(log)
    assert write_storyboard_script(log, complete, spatial=drawn)["status"] == "kept"


def test_second_run_draw_parameters_on_the_five_spatial_event_windows_are_pinned():
    drawn = {}
    for window in SPATIAL_EVENT_WINDOWS:
        log = parse_event_log(window["event_log"])
        drawn[window["case_id"]] = sorted(drawn_spatial_entries(log, fact_roles(log), case_id=window["case_id"], fraction=0.23, seed=42))
    assert drawn == {"ci_vid:train:000251205": [], "full:cine250k:cine250k:386644107_170": [],
                     "ci_vid:train:000016693": [34, 43], "full:cine250k:cine250k:254121744_324": [],
                     "full:cine250k:cine250k:26787554_21": [38]}
