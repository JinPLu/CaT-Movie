"""规划器监督相关的 event_log 函数：声明表、起因/可推出、实体对齐、逐条比对、第 1 档合并。

日志一律取真实 v3 固件；预测日志是在真实日志上做的显式改动（改写、矛盾、漏写、多写），
用来检验比对与合并规则，不是模型输出。word_f1 是纯函数，用合成字符串。
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.event_log import (
    MissingVerdict, align_entities, derivable, entity_ids, entry_labels, entry_outcomes, fact_key, fact_roles,
    merge_invariant_violations, merge_planned_log, parse_event_log, script_named_ids, snapshots, table_entries,
    validate_event_log, word_f1,
)
from mspc.evaluation.event_planner_metrics import free_text_pairs

FIXTURES = Path(__file__).parents[2] / "fixtures"
VALID = ["p1a-01", "p1a-03", "p1a-06", "p1a-10", "p1a-12", "p1a-25", "p1a-35"]


def raw(name="p1a-01"):
    return json.loads((FIXTURES / "event_log_v3_reference" / f"{name}.json").read_text())["event_log"]


def words(text):
    return len(text.split())


def test_real_roles_restated_carried_fact_is_derivable_and_new_value_is_cause():
    log = parse_event_log(raw())
    entries = fact_roles(log)
    assert [e.number for e in entries] == list(range(1, len(entries) + 1))
    # 写手表条目连续编号在前，空间事实（屏幕事实）排在末尾、不进写手表。p1a-01：20 条 WRITE、8 条 DO-NOT-WRITE（5 条状态复述、
    # 3 镜地点与上一镜相同）、27 条空间事实。
    table = table_entries(entries)
    assert [e.number for e in table] == list(range(1, len(table) + 1))
    assert {e.kind for e in entries if e.role == "spatial"} == {"screen"} and len(entries) - len(table) == 27
    assert (sum(e.role == "cause" for e in entries), sum(e.role == "derivable" for e in entries)) == (20, 8)
    costume = [e for e in entries if e.kind == "fact" and e.fact.predicate_or_axis == "costume"]
    # 真实窗把同一件衣服在四镜各声明一次：首镜是起因，后三镜只是复述延续值。
    assert [(e.shot_id, e.role) for e in costume] == [("s01", "cause"), ("s02", "derivable"), ("s03", "derivable"), ("s04", "derivable")]
    assert all(e.role == "cause" for e in entries if e.kind in ("presence", "event"))  # 没有延续值，每镜都算起因
    # 四镜同一地点：首镜写，后三镜与上一镜相同、不写。
    place = [e for e in entries if e.kind == "shot" and e.target == "location_id"]
    assert [(e.shot_id, e.role) for e in place] == [("s01", "cause"), ("s02", "derivable"), ("s03", "derivable"), ("s04", "derivable")]
    causes = [e.number for e in entries if e.role == "cause"]
    assert {e.number for e in place[1:]} <= set(entry_labels(log, causes)["derivable"])
    # 首镜地点没写明，后面的“同一地点”就无从推出，归自由。
    unplaced = entry_labels(log, [n for n in causes if n != place[0].number])
    assert {e.number for e in place} <= set(unplaced["free"])
    assert all(e.role == "spatial" for e in entries if e.kind == "screen")
    assert [e.kind for e in entries if e.kind == "event"] == ["event"]
    # s04 的盒子状态：值与此前延续值不同（此前没有），算起因。
    box = next(e for e in entries if e.kind == "fact" and e.fact.subject == "Subject 3")
    assert (box.shot_id, box.role) == ("s04", "cause")
    # bbox 不编译，不进表。
    assert not any("bbox" in e.text for e in entries)


def test_real_appearance_is_listed_only_where_the_entity_first_appears():
    entries = fact_roles(parse_event_log(raw()))
    appearance = [(e.shot_id, e.target, e.role) for e in entries if e.kind == "appearance"]
    # 之后各镜外观不变，不再登记：不进写手表，也不计分。
    assert appearance == [("s01", "Subject 1", "cause"), ("s01", "Subject 2", "cause"),
                          ("s01", "Subject 3", "cause"), ("s01", "Location 1", "cause")]


def test_real_spatial_facts_stay_out_of_the_writer_table_but_keep_their_own_class():
    log = parse_event_log(raw())
    entries = fact_roles(log)
    spatial = [e for e in entries if e.kind == "screen"]
    # 位置、朝向、视线不进写手表，也不分起因/复述；bbox 不编译。
    assert spatial and all(e.role == "spatial" for e in spatial)
    assert min(e.number for e in spatial) > len(table_entries(entries))
    assert not any(e.role == "spatial" for e in table_entries(entries))
    assert {e.fact.predicate_or_axis for e in spatial} == {"horizontal", "depth", "orientation", "gaze"}


@pytest.mark.parametrize("name", VALID)
def test_real_derivable_reproduces_gold_snapshots_when_all_causes_are_stated(name):
    log = parse_event_log(raw(name))
    entries = fact_roles(log)
    starts, ends = derivable(log, [e.number for e in entries if e.role == "cause"])
    gold_starts, gold_ends = snapshots(log), snapshots(log, end=True)
    for shot in log.shots:
        for key, fact in gold_starts[shot.shot_id].items():
            assert key in starts[shot.shot_id]
            assert starts[shot.shot_id][key].object_or_value == fact.object_or_value
        assert set(ends[shot.shot_id]) == set(gold_ends[shot.shot_id])


def test_real_labels_partition_table_and_follow_written_causes():
    log = parse_event_log(raw())
    entries = fact_roles(log)
    causes = [e.number for e in entries if e.role == "cause"]
    labels = entry_labels(log, causes)
    assert sorted(labels["stated"] + labels["derivable"] + labels["spatial"] + labels["free"]) == sorted(e.number for e in entries)
    # 空间类单列：全部空间事实都在里面（写手没写它们，由规划器从剧本推出），写明和需推出主分里没有。
    screen = {e.number for e in entries if e.kind == "screen"}
    assert set(labels["spatial"]) == screen and not (screen & (set(labels["stated"]) | set(labels["derivable"])))
    assert labels["free"] == []
    assert entry_labels(log, [])["spatial"] == labels["spatial"]  # 写没写都一样
    with pytest.raises(ValueError):
        entry_labels(log, [next(iter(screen))])  # 空间事实不在写手表里，不能算写明
    restated = [e.number for e in entries if e.role == "derivable"]
    assert labels["derivable"] == restated  # 起因写全，复述的可见事实都能推出
    # 没写首镜的衣服，后面复述的衣服就推不出，归自由。
    first_costume = next(e.number for e in entries if e.kind == "fact" and e.fact.predicate_or_axis == "costume")
    partial = entry_labels(log, [n for n in causes if n != first_costume])
    costume = [e.number for e in entries if e.kind == "fact" and e.fact.predicate_or_axis == "costume"]
    assert set(costume) <= set(partial["free"])
    with pytest.raises(ValueError):
        entry_labels(log, [len(entries) + 1])


def test_real_invisible_restated_fact_is_free_not_derivable():
    log = parse_event_log(raw("p1a-03"))
    entries = fact_roles(log)
    labels = entry_labels(log, [e.number for e in entries if e.role == "cause"])
    holds = next(e for e in entries if e.shot_id == "s04" and e.kind == "fact" and e.fact.predicate_or_axis == "holds")
    assert holds.role == "derivable" and holds.number in labels["free"]  # 球不在 s04 可见名单里


def test_word_f1_counts_tokens_and_cjk_characters():
    assert word_f1("red umbrella", "a red umbrella") == pytest.approx(0.8)
    assert word_f1("unknown", "red") == 0
    assert word_f1("黑白足球", "黑色足球") == pytest.approx(0.75)


@pytest.mark.parametrize("name", VALID)
def test_real_self_alignment_is_identity_and_everything_matches(name):
    log = parse_event_log(raw(name))
    alignment = align_entities(log, log, set())
    assert alignment == {e: e for e in entity_ids(log)}
    assert {o.status for o in entry_outcomes(log, log, alignment, {})} == {"match"}


def test_real_renumbered_prediction_aligns_by_presence_and_appearance():
    data = raw("p1a-01")
    swapped = json.loads(json.dumps(data).replace("Subject 2", "Subject X").replace("Subject 3", "Subject 2").replace("Subject X", "Subject 3"))
    swapped["subjects"] = sorted(swapped["subjects"], key=lambda s: s["subject_id"])
    gold, predicted = parse_event_log(data), parse_event_log(swapped)
    validate_event_log(predicted)
    assert align_entities(gold, predicted, set())["Subject 2"] == "Subject 3"
    # 剧本点名的编号直接对齐，不看分数。
    named = align_entities(gold, predicted, script_named_ids("Subject 2 picks up the drill."))
    assert named["Subject 2"] == "Subject 2"


SCRIPT = "Subject 1 holds Subject 2 beside Subject 3 in Location 1."  # 点名全部编号：对齐走直接对应


def predicted_variant():
    """真实 p1a-01 上的显式改动：衣服换说法、s02 人物左右位置矛盾、s03 删一条屏幕事实、多写一条盒子状态。"""
    data = deepcopy(raw())
    for fact in data["initial_state"]:
        if fact["predicate_or_axis"] == "costume":
            fact["object_or_value"] = "a white long-sleeve shirt with dark shorts"
    for fact in data["shots"][1]["screen_facts"]:
        if fact["subject"] == "Subject 1" and fact["predicate_or_axis"] == "horizontal":
            fact["object_or_value"] = "left"
    data["shots"][2]["screen_facts"] = [f for f in data["shots"][2]["screen_facts"] if f["predicate_or_axis"] != "gaze"]
    data["initial_state"].append({"subject": "Subject 3", "family": "state", "predicate_or_axis": "object_condition",
                                  "object_or_value": "empty", "scope": "scene", "shot_span": [0, 0], "evidence": "observed"})
    return data


def test_real_outcomes_need_verdicts_for_free_text_and_classify_changes():
    gold, predicted = parse_event_log(raw()), parse_event_log(predicted_variant())
    validate_event_log(predicted)
    alignment = align_entities(gold, predicted, set())
    with pytest.raises(MissingVerdict):
        entry_outcomes(gold, predicted, alignment, {})
    pair = ("a white long-sleeved top and dark shorts", "a white long-sleeve shirt with dark shorts")
    entries = {e.number: e for e in fact_roles(gold)}
    outcomes = {o.number: o for o in entry_outcomes(gold, predicted, alignment, {pair: True})}
    costume = [n for n, e in entries.items() if e.kind == "fact" and e.fact.predicate_or_axis == "costume"]
    assert all(outcomes[n].status == "match" for n in costume)
    assert outcomes[costume[0]].rewrites == ((("fact", "Subject 1", "costume"), *pair),)
    horizontal = next(n for n, e in entries.items() if e.shot_id == "s02" and e.kind == "screen"
                      and e.fact.subject == "Subject 1" and e.fact.predicate_or_axis == "horizontal")
    assert outcomes[horizontal].status == "contradiction"
    gaze = next(n for n, e in entries.items() if e.shot_id == "s03" and e.kind == "screen" and e.fact.predicate_or_axis == "gaze")
    assert outcomes[gaze].status == "missing"
    rejected = {o.number: o.status for o in entry_outcomes(gold, predicted, alignment, {pair: False})}
    assert all(rejected[n] == "contradiction" for n in costume)


def written_all_causes(log):
    """写手写全了起因时的标签：空间事实单列在 spatial。"""
    return entry_labels(log, [e.number for e in fact_roles(log) if e.role == "cause"])


def test_real_merge_follows_five_rules_and_keeps_invariant():
    truth, planned = parse_event_log(raw()), parse_event_log(predicted_variant())
    entries = fact_roles(truth)
    gaze = next(e.number for e in entries if e.shot_id == "s03" and e.kind == "screen" and e.fact.predicate_or_axis == "gaze")
    labels = written_all_causes(truth)
    assert gaze in labels["spatial"]
    pair = ("a white long-sleeved top and dark shorts", "a white long-sleeve shirt with dark shorts")
    merged, report = merge_planned_log(truth, planned, labels, {pair: True}, script=SCRIPT)
    validate_event_log(merged)
    # 同键同值→规划器写法（全局一致替换）
    assert {f.object_or_value for f in merged.initial_state if f.predicate_or_axis == "costume"} == {pair[1]}
    # 值矛盾→真值；多写→删
    assert [f.model_dump() for f in merged.shots[1].screen_facts] == [f.model_dump() for f in truth.shots[1].screen_facts]
    assert not any(f.subject == "Subject 3" and f.object_or_value == "empty" for f in merged.initial_state)
    # 漏写的空间键→保持漏写（规划器自己推出，它漏了就是漏了）；规划器写到的空间键照常保留
    assert not any(f.predicate_or_axis == "gaze" for f in merged.shots[2].screen_facts)
    assert [f.model_dump() for f in merged.shots[0].screen_facts] == [f.model_dump() for f in truth.shots[0].screen_facts]
    assert report["rewrites"] == [pair]
    assert merge_invariant_violations(truth, merged, report["rewrites"], words) == []
    # 漏写的写明键→补真值（合并只要求标签划分全表；这里人为把一条空间事实标成写明来检验这条规则）
    stated = dict(labels, stated=sorted(labels["stated"] + [gaze]), spatial=[n for n in labels["spatial"] if n != gaze])
    kept, _ = merge_planned_log(truth, planned, stated, {pair: True}, script=SCRIPT)
    assert any(f.predicate_or_axis == "gaze" for f in kept.shots[2].screen_facts)


def test_real_merge_without_equivalences_only_emits_truth_clauses():
    truth, planned = parse_event_log(raw()), parse_event_log(predicted_variant())
    entries = fact_roles(truth)
    labels = {"stated": [], "derivable": [], "spatial": [], "free": [e.number for e in entries]}
    pair = ("a white long-sleeved top and dark shorts", "a white long-sleeve shirt with dark shorts")
    merged, report = merge_planned_log(truth, planned, labels, {pair: False}, script=SCRIPT)
    assert report["rewrites"] == []
    assert merge_invariant_violations(truth, merged, [], words) == []


def swap_subjects_2_and_3(data):
    swapped = json.loads(json.dumps(data).replace("Subject 2", "Subject X").replace("Subject 3", "Subject 2").replace("Subject X", "Subject 3"))
    swapped["subjects"] = sorted(swapped["subjects"], key=lambda s: s["subject_id"])
    return swapped


def test_real_merge_aligns_named_ids_directly_and_the_rest_by_hungarian_matching():
    truth = parse_event_log(raw())
    planned = parse_event_log(swap_subjects_2_and_3(raw()))
    labels = written_all_causes(truth)

    def merge(script):
        alignment = align_entities(truth, planned, script_named_ids(script))
        verdicts = {pair: False for pair in free_text_pairs(truth, planned, alignment)}
        return merge_planned_log(truth, planned, labels, verdicts, script=script)[1]

    # 剧本没点名：预测把盒子和电钻的编号对调了，匈牙利匹配按在场与外观把它们对回来。
    report = merge("A man works in a room.")
    assert report["alignment"]["Subject 2"] == "Subject 3" and report["alignment"]["Subject 3"] == "Subject 2"
    assert report["alignment"]["Subject 1"] == "Subject 1"
    # 剧本逐字点名 Subject 2：直接按编号对齐（哪怕外观对不上），其余仍走匹配。
    named = merge("Subject 2 picks up the drill.")
    assert named["alignment"]["Subject 2"] == "Subject 2" and named["alignment"]["Subject 1"] == "Subject 1"


def test_real_merge_keeps_free_declaration_an_event_depends_on():
    truth = parse_event_log(raw())
    planned_raw = deepcopy(raw())
    planned_raw["initial_state"] = [f for f in planned_raw["initial_state"] if f["subject"] != "Subject 3"]
    planned_raw["events"] = []
    planned = parse_event_log(planned_raw)
    validate_event_log(planned)
    entries = fact_roles(truth)
    box = next(e.number for e in entries if e.kind == "fact" and e.fact.subject == "Subject 3")
    event = next(e.number for e in entries if e.kind == "event")
    labels = {"stated": [event], "derivable": [], "spatial": [], "free": [e.number for e in entries if e.number != event]}
    merged, report = merge_planned_log(truth, planned, labels, {}, script=SCRIPT)
    validate_event_log(merged)
    assert len(merged.events) == 1
    # 盒子开头状态是自由键且规划器漏写，但被保留的事件依赖它，删了日志不合法，退回真值并记账。
    assert any(f.subject == "Subject 3" for f in merged.initial_state)
    assert {"change": "omit", "entry": box, "kind": "fact"} in report["reverted"]


def test_real_merge_rejects_unlocked_shots():
    truth = parse_event_log(raw())
    changed = deepcopy(raw())
    changed["shots"][0]["frame_count"] += 4
    planned = parse_event_log(changed)
    entries = fact_roles(truth)
    with pytest.raises(ValueError, match="locked shot"):
        merge_planned_log(truth, planned, {"stated": [], "derivable": [], "spatial": [], "free": [e.number for e in entries]}, {},
                          script=SCRIPT)


def test_real_fact_keys_of_relations_include_targets():
    log = parse_event_log(raw())
    holds = next(e for e in fact_roles(log) if e.kind == "fact" and e.fact.predicate_or_axis == "holds")
    assert fact_key(holds.fact) == ("Subject 1", "holds", "Subject 2")


FLOOR = {row["case_id"]: row["event_log"] for row in
         json.loads((FIXTURES / "planner/writer_first_run_checker_misses.json").read_text())["rows"]}


def test_real_body_values_lose_screen_phrases_only():
    """地板 v10 的真实 body 取值：画面方位、朝向镜头剥掉；相对主体的位置、出入画、身体部位左右保留。"""
    from mspc.data.event_log import screen_free
    cases = {
        "standing on the left": "standing", "standing in the middle": "standing", "standing in profile": "standing",
        "standing with back partly to camera": "standing", "facing the camera and talking": "talking",
        "standing with back to camera, reaching": "standing, reaching", "sitting on the bed facing right": "sitting on the bed",
        "looking screen-right at Subject 3": "looking at Subject 3", "partially out of frame on the left": "partially out of frame",
        "standing to the right of Subject 3 with its back to camera": "standing to the right of Subject 3",
        "facing the camera": "", "in the foreground": "",
        "walking to the right side of the room": "walking to the right side of the room",
        "standing at the left side of Subject 4": "standing at the left side of Subject 4",
        "standing with his right fist raised": "standing with his right fist raised",
        "holding Subject 2 to his left near his face": "holding Subject 2 to his left near his face",
        "head lowered below the frame": "head lowered below the frame",
    }
    assert {value: screen_free(value) for value in cases} == cases


def test_real_writer_table_has_no_screen_phrases_and_pure_orientation_events_go_spatial():
    from mspc.data.event_log import SCREEN_PHRASES
    wedding = parse_event_log(FLOOR["full:cine250k:cine250k:326318830_30"])
    entries = fact_roles(wedding)
    table = [e for e in table_entries(entries) if e.kind in ("fact", "event")]
    assert not any(SCREEN_PHRASES.search(e.text) for e in table)
    assert any(e.text == "Character 1 is standing" and e.role == "cause" for e in table)  # 原值 standing on the left
    assert any(e.text == "event (on screen): Character 3 is standing -> Character 3 is turned" for e in table)
    beach = parse_event_log(FLOOR["full:cine250k:cine250k:386644107_170"])
    entries = fact_roles(beach)
    turned = [e for e in entries if e.kind == "event" and e.shot_id == "s03"]
    assert [e.role for e in turned] == ["spatial", "spatial"]  # holding hands -> facing the camera：剥掉后只剩朝向
    assert all("facing the camera" in e.text for e in turned)
    causes = [e.number for e in entries if e.role == "cause"]
    labels = entry_labels(beach, causes)
    assert {e.number for e in turned} <= set(labels["spatial"])
    assert sorted(n for ns in labels.values() for n in ns) == [e.number for e in entries]


def test_real_default_camera_fields_are_derivable_and_shot_size_stays_a_cause():
    """首跑 R08（四镜全是 eye level、static、同一地点）：缺省机位与运镜不写，景别每镜写；没写的缺省值算需推出。"""
    log = parse_event_log(FLOOR["full:cine250k:cine250k:326318830_30"])
    entries = fact_roles(log)
    fields = {(e.shot_id, e.target): e for e in entries if e.kind == "shot"}
    assert {e.text for e in fields.values() if e.target == "camera_angle"} == {"camera angle: eye level"}
    assert {e.role for e in fields.values() if e.target in ("camera_angle", "movement")} == {"derivable"}
    assert {e.role for e in fields.values() if e.target in ("shot_size", "action")} == {"cause"}
    assert [fields[(s.shot_id, "location_id")].role for s in log.shots] == ["cause", "derivable", "derivable", "derivable"]
    labels = entry_labels(log, [e.number for e in entries if e.role == "cause"])
    assert {e.number for e in fields.values() if e.role == "derivable"} <= set(labels["derivable"])
    # 动镜头照写：首跑 R04 两镜是 pull out、pan right。
    moving = parse_event_log(FLOOR["ci_vid:train:000044398"])
    assert [e.role for e in fact_roles(moving) if e.target == "movement"] == ["cause", "cause"]
