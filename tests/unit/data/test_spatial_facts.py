import itertools
import json

import numpy as np
import pytest

from mspc.data import spatial_facts as sf
from tests.support.paths import REPO_ROOT


def test_layout_and_key_positions():
    assert sf.layout_positions(41) == (12, 20, 28)
    assert sf.layout_positions(37) == (11, 18, 25)
    assert sf.layout_positions(5) == (1, 2, 3)
    assert sf.key_positions(41) == (0, 12, 20, 28, 40)
    with pytest.raises(ValueError):
        sf.layout_positions(3)


def test_assign_one_to_one_matches_brute_force_and_respects_floor():
    rng = np.random.default_rng(1)
    for _ in range(30):
        rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        gain = {(r, c): float(rng.uniform(0, 1)) for r in range(rows) for c in range(cols) if rng.uniform() < 0.7}
        result = sf.assign_one_to_one(gain, min_gain=0.0)
        assert len(set(result.values())) == len(result)
        slots = list(range(cols)) + [None] * rows
        best = max(sum(gain.get((r, c), -1e9) for r, c in enumerate(choice) if c is not None)
                   for choice in itertools.permutations(slots, rows)
                   if all(c is None or (r, c) in gain for r, c in enumerate(choice)))
        assert sum(gain[(r, c)] for r, c in result.items()) == pytest.approx(best)
    assert sf.assign_one_to_one({("a", 1): 0.2}, min_gain=0.35) == {}


def test_phrase_binding_needs_three_frames_dominance_and_distinct_phrases():
    phrases = {"Subject 1": "old man in grey suit", "Subject 2": "woman in red dress"}
    votes = {"Subject 1": {"person#0": 20, "person#1": 1}, "Subject 2": {"person#1": 18}}
    assert sf.phrase_binding(votes, phrases) == {"Subject 1": "person#0", "Subject 2": "person#1"}
    # 只触发 2 帧：不算。
    assert sf.phrase_binding({"Subject 1": {"person#0": 2}}, {"Subject 1": "man"}) == {}
    # 同一短语在两个实例上都触发、相差不到 2 倍：不唯一，交兜底。
    assert sf.phrase_binding({"Subject 1": {"person#0": 9, "person#1": 7}}, {"Subject 1": "man"}) == {}
    # 两个主体短语相同：都不用。
    same = {"Subject 1": "man", "Subject 2": "man"}
    assert sf.phrase_binding({"Subject 1": {"person#0": 9}, "Subject 2": {"person#1": 9}}, same) == {}


def test_phrase_binding_on_real_dev30_votes():
    # 长集 dev lsh-001 的 16 fps 逐帧计票：s02 两个短语在两人上逐帧都触发（65/65），不能靠短语分。
    phrases = {"Subject 2": "bald man in grey suit", "Subject 3": "man in grey suit"}
    s02 = {"Subject 2": {"person#1": 65, "person#0": 65}, "Subject 3": {"person#1": 65, "person#0": 65}}
    assert sf.phrase_binding(s02, phrases) == {}
    # s04：两个短语都以 2 倍以上占优落在 person#1 上，同一实例被两人认领，两人都交兜底。
    s04 = {"Subject 2": {"person#1": 40, "person#0": 14}, "Subject 3": {"person#1": 49, "person#0": 23}}
    assert sf.phrase_binding(s04, phrases) == {}
    assert sf.phrase_binding({"Subject 1": {"person#0": 37}}, {"Subject 1": "man in yellow jacket"}) == \
        {"Subject 1": "person#0"}


def test_short_phrase_keeps_age_head_and_first_colored_garment():
    attributes = "A bald middle-aged man with a grey beard, wearing a charcoal suit and a white shirt."
    assert sf.short_phrase(attributes, "person") == "middle-aged bald man in charcoal suit"
    assert sf.short_phrase("A small brown dog with floppy ears.", "animal") == "dog"
    assert sf.short_phrase("A figure in a long coat.", "person") == "person"


def test_horizontal_labels_and_majority():
    assert sf.horizontal_labels({"a": 100.0}, width=832) == {"a": "left"}
    assert sf.horizontal_labels({"a": 416.0}, width=832) == {"a": "center"}
    assert sf.horizontal_labels({"a": 500.0, "b": 100.0, "c": 300.0}, width=832) == {
        "b": "left", "c": "center", "a": "right"}
    assert sf.majority(["x", None, "x"]) == "x"
    assert sf.majority(["x", None, "y"]) is None


def _rect(h, w, y0, y1, x0, x1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_contact_difference_reads_disparity_on_both_sides_of_the_contact():
    a = _rect(60, 80, 10, 50, 10, 40)  # 前景
    b = _rect(60, 80, 10, 50, 40, 70)  # 被挡者，与 a 在 x=40 相接
    disparity = np.zeros((60, 80))
    disparity[a] = 5.0
    disparity[b] = 2.0
    assert sf.contact_difference(a, b, disparity) == pytest.approx(3.0)
    assert sf.contact_difference(b, a, disparity) == pytest.approx(-3.0)
    far = _rect(60, 80, 10, 50, 60, 75)
    assert sf.contact_difference(_rect(60, 80, 10, 50, 0, 20), far, disparity) is None


def test_depth_relations_gap_from_medians_direction_from_contact():
    medians = {"S1": 0.9, "S2": 0.3, "S3": 0.2}
    # S1–S3 相接、相接处 S3 更近（遮挡）：有无看中位差（0.35 ≥ 0.1），方向听相接处。
    relations = sf.depth_relations(medians, {("S1", "S3"): -0.6}, frame_range=2.0, gap=0.1)
    assert relations == {("S1", "S2"): "front", ("S1", "S3"): "behind"}
    assert sf.depth_labels(relations, ["S1", "S2", "S3"]) == {"S3": "foreground", "S1": "background",
                                                              "S2": "background"}
    # 中位差不够：没有前后，不论相接处差多少。
    assert sf.depth_relations({"S1": 0.9, "S2": 0.85}, {("S1", "S2"): 1.5}, frame_range=2.0, gap=0.1) == {}
    assert sf.depth_labels({}, ["S1", "S2"]) == {}


def _pose(nose_x, *, face=7.0, body=7.0, back=False):
    xy = np.zeros((133, 2))
    scores = np.zeros(133)
    scores[23:91] = face  # 68 个脸点（RTMW 的 SimCC 原始峰值尺度）
    scores[[5, 6, 11, 12]] = body
    xy[5], xy[6] = ((80, 120), (160, 120)) if back else ((160, 120), (80, 120))  # 正面时人体左肩在画面右侧
    xy[23] = (100, 50)
    xy[39] = (140, 50)
    xy[53] = (nose_x, 55)
    return xy, scores


def test_orientation_rule_and_mirror_equivariance():
    kw = {"box_width": 120.0, "face_visible": 3.0, "face_back": 6.0, "lateral": 0.15}
    assert sf.orientation_from_keypoints(*_pose(121), **kw) == "toward camera"
    assert sf.orientation_from_keypoints(*_pose(132), **kw) == "screen-right"
    assert sf.orientation_from_keypoints(*_pose(105), **kw) == "screen-left"
    assert sf.orientation_from_keypoints(*_pose(120, face=2.0), **kw) == "away"
    assert sf.orientation_from_keypoints(*_pose(120, face=2.0, body=2.0), **kw) is None
    # 肩序是背面、脸点又不够清楚：背影；肩序背面但脸点很清楚（回头看镜头）：按脸判。
    assert sf.orientation_from_keypoints(*_pose(121, face=5.0, back=True), **kw) == "away"
    assert sf.orientation_from_keypoints(*_pose(121, face=7.0, back=True), **kw) == "toward camera"
    xy, scores = _pose(132)
    flipped = xy.copy()
    flipped[:, 0] = 832 - xy[:, 0]
    flipped[[23, 39]] = flipped[[39, 23]]  # 镜像后左右下颌点互换（模型按解剖左右出点）
    assert sf.orientation_from_keypoints(flipped, scores, **kw) == sf.MIRROR_ORIENTATION["screen-right"]


def test_identity_fusion_and_count_limited_matching():
    a, b = sf.fit_logistic([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.25, 0.75], [0, 0, 0, 1, 1, 1, 1, 0])
    assert a > 0
    calibration = {"prior": -1.0, "paths": {"face": (10.0, -6.0), "reid": (8.0, -5.0)}}
    assert sf.fused_log_odds({}, calibration) is None
    high = sf.fused_log_odds({"face": 0.9, "reid": 0.8}, calibration)
    low = sf.fused_log_odds({"face": 0.2}, calibration)
    assert high > 0 > low
    log_odds = {("x", "p"): 3.0, ("x", "q"): 1.0, ("y", "q"): 2.0, ("y", "p"): -1.0}
    assert sf.cross_shot_matches(log_odds, limit=2) == {"x": "p", "y": "q"}
    assert sf.cross_shot_matches(log_odds, limit=1) == {"x": "p"}
    assert sf.unit_mean([[3.0, 0.0], [0.0, 1.0]]) == pytest.approx([2 ** -0.5, 2 ** -0.5])


# ---------------------------------------------------------------------------
# 真实切片：地板 floor-0000 的 v10 编译事件日志（renderer-floor-v9-reannotate-20261009/compile-v10/event_logs.jsonl 第 1 行）
# ---------------------------------------------------------------------------

REAL = REPO_ROOT / "tests" / "fixtures" / "spatial_facts_real"


def _floor0000():
    return json.loads((REAL / "floor0000_v10_event_log.json").read_text(encoding="utf-8"))["event_log"]


def test_short_phrase_on_real_v10_subjects_with_shot_clothing():
    log = _floor0000()
    appearance = {s["subject_id"]: s["appearance"] for s in log["subjects"]}
    clothing = {f["subject"]: f["object_or_value"] for f in log["initial_state"]
                if f["predicate_or_axis"] == "costume" and f["shot_span"][0] == 0}
    assert sf.short_phrase(appearance["Subject 1"] + ", wearing " + clothing["Subject 1"], "person") == "bald man"
    assert sf.short_phrase(appearance["Subject 2"] + ", wearing " + clothing["Subject 2"], "person") == \
        "woman in black cardigan"


def test_replace_screen_facts_on_real_v10_log_keeps_objects_and_validates():
    from mspc.data.event_log import parse_event_log, validate_event_log

    log = _floor0000()
    layouts = [{"Subject 1": {"horizontal": "left", "depth": "background", "orientation": "screen-right"},
                "Subject 2": {"horizontal": "right", "depth": "foreground", "orientation": "toward camera"}},
               {"Subject 2": {"horizontal": "center", "depth": None, "orientation": "screen-left"}},
               {}]
    merged = sf.replace_screen_facts(log, layouts)
    validate_event_log(parse_event_log(merged))
    s01 = [(f["subject"], f["predicate_or_axis"], f["object_or_value"]) for f in merged["shots"][0]["screen_facts"]]
    assert s01[:6] == [("Subject 1", "horizontal", "left"), ("Subject 2", "horizontal", "right"),
                       ("Subject 1", "depth", "background"), ("Subject 2", "depth", "foreground"),
                       ("Subject 1", "orientation", "screen-right"), ("Subject 2", "orientation", "toward camera")]
    # 物体（盘、杯、叉、刀）的 qwen 屏幕事实原样保留；人物的 gaze 去掉；人物在本镜没有工具链事实时也不留 qwen 的。
    assert [f["subject"] for f in merged["shots"][0]["screen_facts"][6:]] == ["Subject 3", "Subject 4", "Subject 5", "Subject 6"]
    assert not any(f["predicate_or_axis"] == "gaze" for s in merged["shots"] for f in s["screen_facts"])
    assert [f["subject"] for f in merged["shots"][2]["screen_facts"]] == ["Subject 3", "Subject 4", "Subject 5", "Subject 6"]
    # 其余字段不动。
    for key in ("subjects", "locations", "initial_state", "events"):
        assert merged[key] == log[key]
    assert [{k: v for k, v in s.items() if k != "screen_facts"} for s in merged["shots"]] == \
        [{k: v for k, v in s.items() if k != "screen_facts"} for s in log["shots"]]
    with pytest.raises(ValueError, match="not present people or animals"):
        sf.replace_screen_facts(log, [{"Subject 3": {"horizontal": "left"}}, {}, {}])
    with pytest.raises(ValueError, match="unsupported screen values"):
        sf.replace_screen_facts(log, [{"Subject 1": {"orientation": "toward Subject 2"}}, {}, {}])


def test_endpoint_positions_and_votes():
    assert sf.endpoint_positions(41) == {"first": (0, 1, 2, 3, 4), "last": (36, 37, 38, 39, 40)}
    assert sf.endpoint_positions(3) == {"first": (0, 1, 2), "last": (0, 1, 2)}
    # 首端 5 帧：S2 入画前两帧只有 S1（单人按三等分记 center），之后两人并排；过半（≥3 帧）才写。
    frames = [{"S1": 400.0}, {"S1": 400.0}, {"S1": 300.0, "S2": 600.0}, {"S1": 300.0, "S2": 620.0},
              {"S1": 310.0, "S2": 640.0}]
    assert sf.endpoint_horizontal(frames, width=832) == {"S1": "left", "S2": "right"}
    assert sf.endpoint_horizontal([{"S1": 100.0}, {}, {}, {"S1": 100.0}, {}], width=832) == {}


def test_merged_screen_layout_keeps_horizontal_depth_and_only_toward_camera():
    votes = {"Subject 1": "left", "Subject 2": "right"}
    frame = {"Subject 1": {"horizontal": "left", "depth": "foreground", "orientation": "screen-right"},
             "Subject 2": {"horizontal": "right", "depth": "background", "orientation": "toward camera"},
             "Subject 3": {"horizontal": "center", "depth": None, "orientation": "away"}}
    assert sf.merged_screen_layout(votes, frame) == {
        "Subject 1": {"horizontal": "left", "depth": "foreground"},
        "Subject 2": {"horizontal": "right", "depth": "background", "orientation": "toward camera"}}
    # 投票没过半、首帧只有侧向：这个主体什么都不并。
    assert sf.merged_screen_layout({}, {"Subject 1": {"horizontal": "left", "depth": None,
                                                    "orientation": "screen-left"}}) == {}


def test_merged_layout_goes_into_real_v10_log_and_compiles_space_sentence():
    from mspc.data.event_log import compile_event_log, parse_event_log, space_text, validate_event_log

    log = _floor0000()
    # 本链在 floor-0000 上的首端事实（地板放量 10-10 实跑：两人左右、首帧朝向都是侧向，没有前后）。
    first = {"Subject 1": {"horizontal": "left", "depth": None, "orientation": "screen-right"},
             "Subject 2": {"horizontal": "right", "depth": None, "orientation": "screen-left"}}
    layouts = [sf.merged_screen_layout({"Subject 1": "left", "Subject 2": "right"}, first),
               sf.merged_screen_layout({"Subject 2": "center"}, {"Subject 2": {"orientation": "toward camera"}}), {}]
    merged = sf.replace_screen_facts(log, layouts)
    parsed = parse_event_log(merged)
    validate_event_log(parsed)
    people = {"Subject 1", "Subject 2"}
    assert [(f["subject"], f["predicate_or_axis"], f["object_or_value"]) for f in merged["shots"][0]["screen_facts"]
            if f["subject"] in people] == [("Subject 1", "horizontal", "left"), ("Subject 2", "horizontal", "right")]
    assert [(f["predicate_or_axis"], f["object_or_value"]) for f in merged["shots"][1]["screen_facts"]
            if f["subject"] in people] == [("horizontal", "center"), ("orientation", "toward camera")]
    # 第 1 镜两人左右；第 2 镜 Subject 2 居中且面朝镜头；第 3 镜没有可并的事实就没有空间句。
    assert [space_text(parsed, shot) for shot in parsed.shots] == [
        "On the left side of the frame is a bald man with glasses and a grey beard; "
        "on the right side of the frame is a woman with short grey hair.",
        "In the center of the frame is a woman with short grey hair, facing the camera.", ""]
    compiled = compile_event_log(parsed, span=(0, 3), references=(), token_counter=lambda text: len(text.split()))
    assert compiled.target_rows[1].startswith(
        "In the center of the frame is a woman with short grey hair, facing the camera. close-up, eye level, static.")
    assert not any("Screen:" in row for row in compiled.target_rows)
    assert all(row.index("Design:") > row.index("static.") for row in compiled.target_rows)  # 空间句→前缀→动作→Design
