"""跨镜空间布局判别的纯函数：取帧、绑定指派、oracle 布局、四道题的关系与判据、S 臂 Screen 事实、N 臂纯删除核对。

语料类函数（S 臂填 screen 事实、删方位核对）用真实 dev 30 B 剧本切片（lsh-001、lsh-006 固件）；几何与指派是纯数学，
用合成输入。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mspc.benchmark.long_story_event_log import event_log_from_script, r3_frame_counts
from mspc.data.event_log import compile_event_log, parse_event_log
from mspc.data.spatial_facts import layout_positions
from mspc.evaluation import cross_shot_layout as csl
from tests.support.paths import REPO_ROOT

PAIRS = REPO_ROOT / "tests/fixtures/long_story_event_log/dev30_b_lsh001_lsh006.json"
CANDIDATE = REPO_ROOT / "configs/benchmark/script_state_bench_long_holdout_v2_candidate_r3.json"


def _pair(story: str) -> dict:
    return next(p for p in json.loads(PAIRS.read_text())["pairs"] if p["story_id"] == story)


# ---------------------------------------------------------------------------
# 取帧
# ---------------------------------------------------------------------------


def test_source_offsets_shift_by_shot_start():
    assert csl.source_offsets(269, 325) == tuple(269 + o for o in layout_positions(56))


def test_detector_phrase_drops_trailing_period_and_inner_separators():
    assert csl.detector_phrase("A woman with long hair. She wears red.") == "A woman with long hair, She wears red"


# ---------------------------------------------------------------------------
# 指派与跨镜认人
# ---------------------------------------------------------------------------


def test_cross_shot_mapping_follows_embedding_similarity():
    first = {"Subject 1": [1.0, 0.0, 0.0], "Subject 2": [0.0, 1.0, 0.0]}
    second = {"Subject 1": [0.1, 0.99, 0.0], "Subject 2": [0.98, 0.05, 0.0]}
    assert csl.cross_shot_mapping(first, second) == {"Subject 1": "Subject 2", "Subject 2": "Subject 1"}


def test_bound_tracks_follow_sam_track_identity_across_frames():
    frames = [
        {"instances": [{"concept": "person", "track_id": 0, "box": [10, 0, 50, 100], "mask_key": "f0a"},
                       {"concept": "person", "track_id": 1, "box": [400, 0, 450, 100], "mask_key": "f0b"}]},
        {"instances": [{"concept": "person", "track_id": 1, "box": [390, 0, 440, 100], "mask_key": "f1b"}]},
        {"instances": [{"concept": "person", "track_id": 0, "box": [20, 0, 60, 100], "mask_key": "f2a"},
                       {"concept": "person", "track_id": 1, "box": [380, 0, 430, 100], "mask_key": "f2b"}]},
    ]
    tracks = csl.bound_tracks(frames, {"Subject 1": "person#1", "Subject 2": "person#0"},
                              {"f0b": [1.0, 0.0], "f1b": [1.0, 0.0], "f0a": [0.0, 1.0]})
    assert tracks["boxes"][1] == {"Subject 1": [390.0, 0.0, 440.0, 100.0], "Subject 2": None}
    assert tracks["mask_keys"][2] == {"Subject 1": "f2b", "Subject 2": "f2a"}
    assert csl.pair_left(tracks, "Subject 1", "Subject 2", width=832) == "Subject 2"
    assert csl.median_center(tracks, "Subject 2") == pytest.approx(35.0)
    assert np.allclose(tracks["embeddings"]["Subject 1"], [1.0, 0.0])


def test_separable_uses_kind_and_color_words():
    man = {"kind": "person", "attributes": "A bald man wearing a grey, high-collared suit."}
    short = {"kind": "person", "attributes": "Short adult man wearing a grey suit."}
    yellow = {"kind": "person", "attributes": "Heavyset man, yellow jacket, green suspenders."}
    dog = {"kind": "animal", "attributes": "A grey dog."}
    assert not csl.separable(man, short)
    assert csl.separable(man, yellow)
    assert csl.separable(man, dog)
    woman = {"kind": "person", "attributes": "A woman with dark skin, wearing a yellow spaghetti-strap top."}
    bearded = {"kind": "person", "attributes": "Man with dark skin, short dark hair, a beard and a dark T-shirt."}
    assert csl.separable(woman, bearded)


# ---------------------------------------------------------------------------
# S 臂：真实剧本填 screen 事实并经编译器出开头的空间句（10-10 起取代末尾的 Screen 句）
# ---------------------------------------------------------------------------


def test_screen_facts_compile_into_leading_space_sentence_on_real_dev30_log():
    pair = _pair("lsh-001")
    raw = event_log_from_script(pair["context_b"], pair["change"],
                                r3_frame_counts(json.loads(CANDIDATE.read_text()), "lsh-001"))
    layouts = [{} for _ in raw["shots"]]
    layouts[1] = {"Subject 2": {"horizontal": "right", "depth": "background", "orientation": "screen-left"},
                  "Subject 3": {"horizontal": "left", "depth": "foreground", "orientation": "screen-right"}}
    filled = csl.with_screen_facts(raw, layouts, axes=("horizontal", "depth", "orientation"))
    assert raw["shots"][1]["screen_facts"] == []
    log = parse_event_log(filled)
    compiled = compile_event_log(log, span=(0, len(log.shots)), references=[], token_counter=lambda text: len(text.split()))
    # 从左到右、外观短语指人、左右与前后同句；侧向朝向不进文字（只写面朝镜头）。
    assert compiled.target_rows[1].startswith(
        "On the left side of the frame, in the foreground, is a short adult man with a shaved head, wearing a grey suit; "
        "on the right side of the frame, in the background, is a bald man with a thin face, wearing a grey, high-collared suit. "
        "A medium shot in Location 2.")
    # 与 S-first 臂的首句逐字相同（同一份外观短语函数）：没有朝向时，编译器的空间句就是 layout_first_sentence。
    from mspc.data.event_log import space_text
    attributes = {s["subject_id"]: s["attributes"] for s in _pair("lsh-001")["context_b"]["subjects"]}
    assert space_text(log, log.shots[1]) == csl.layout_first_sentence(layouts[1], attributes)
    assert not any("Screen:" in row for row in compiled.target_rows)
    assert not compiled.target_rows[0].startswith(("On the", "In the center"))
    only_lr = csl.with_screen_facts(raw, layouts, axes=("horizontal",))
    assert [f["predicate_or_axis"] for f in only_lr["shots"][1]["screen_facts"]] == ["horizontal", "horizontal"]


def test_screen_facts_reject_subjects_not_visible_in_the_shot():
    pair = _pair("lsh-001")
    raw = event_log_from_script(pair["context_b"], pair["change"],
                                r3_frame_counts(json.loads(CANDIDATE.read_text()), "lsh-001"))
    layouts = [{} for _ in raw["shots"]]
    layouts[0] = {"Subject 2": {"horizontal": "left"}}
    with pytest.raises(ValueError):
        csl.with_screen_facts(raw, layouts, axes=("horizontal",))


# ---------------------------------------------------------------------------
# Q1 关系、Q2 朝向、Q4 构图
# ---------------------------------------------------------------------------


def test_pair_relation_two_paths():
    ident = {"A": "A", "B": "B"}
    swap = {"A": "B", "B": "A"}
    common = dict(left_first="A", a="A", b="B")
    assert csl.pair_relation(left_second="A", separable_text=True, mapping=ident, **common)["relation"] == "same"
    assert csl.pair_relation(left_second="B", separable_text=True, mapping=ident, **common)["relation"] == "reversed"
    assert csl.pair_relation(left_second="A", separable_text=True, mapping=swap, **common)["reason"] == "binding_disagree"
    assert csl.pair_relation(left_second="A", separable_text=False, mapping=swap, **common)["relation"] == "reversed"
    assert csl.pair_relation(left_second=None, separable_text=True, mapping=ident, **common)["relation"] is None


def test_orientation_unflip_and_relation():
    assert csl.unflip("screen-left", True) == "screen-right"
    assert csl.unflip("toward camera", True) == "toward camera"
    assert csl.orientation_relation("screen-left", "screen-right") == "opposite"
    assert csl.orientation_relation("screen-left", "toward camera") is None


def test_crop_flip_is_deterministic_and_roughly_balanced():
    flags = [csl.crop_flipped(42, f"lsh-{i:03d}|s01|Subject 1") for i in range(400)]
    assert flags == [csl.crop_flipped(42, f"lsh-{i:03d}|s01|Subject 1") for i in range(400)]
    assert 150 < sum(flags) < 250


def test_composition_rules():
    head_a, head_b = [0, 0, 10, 20], [0, 0, 10, 30]
    r = csl.head_log_ratio(head_a, head_b)
    assert r == pytest.approx(math.log(1.5))
    assert csl.composition_item(r) == "tighter"
    assert csl.composition_item(0.15) is None
    assert csl.composition_holds("tighter", 0.3) is True
    assert csl.composition_holds("tighter", -0.3) is False
    assert csl.composition_holds("looser", 0.005) is False
    assert csl.composition_tool_direction(0.05) == "same"
    assert csl.composition_tool_direction(0.15) is None


# ---------------------------------------------------------------------------
# Q3 叠加层
# ---------------------------------------------------------------------------


def test_static_overlay_mask_marks_pixels_that_never_change():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (40, 60)).astype(np.uint8) for _ in range(5)]
    for frame in frames:
        frame[30:35, 10:50] = 200  # 烧录字幕条
    mask = csl.static_overlay_mask(frames)
    assert mask[32, 30] and not mask[5, 5]
    background = np.full((40, 60), 255, np.uint8)
    assert csl.suppress_overlay(background, mask)[32, 30] == 0
    with pytest.raises(ValueError):
        csl.static_overlay_mask(frames[:4])


# ---------------------------------------------------------------------------
# N 臂纯删除核对（真实剧本句）
# ---------------------------------------------------------------------------


def test_pure_deletion_on_real_dev30_prompt():
    original = _pair("lsh-001")["context_b"]["shots"][1]["prompt"]
    assert csl.is_pure_deletion(original, original.replace(" in Location 2", ""))
    assert not csl.is_pure_deletion(original, original.replace("kneeling", "standing"))
    assert csl.deleted_words(original, original.replace(" in Location 2", "")) == ["in", "Location", "2"]


def test_pure_deletion_rejects_new_words_spelled_from_deleted_letters():
    assert not csl.is_pure_deletion("Subject 1 is nothing like him", "Subject 1 is not like him")


# ---------------------------------------------------------------------------
# VLM 回答解析
# ---------------------------------------------------------------------------


def test_parse_binding_marks_shared_boxes_unjudgeable():
    text = json.dumps({"matches": [{"subject": "Subject 1", "evidence": "", "box": 2},
                                   {"subject": "Subject 2", "evidence": "", "box": 2},
                                   {"subject": "Subject 3", "evidence": "", "box": 0}]})
    assert csl.parse_binding(text, ["Subject 1", "Subject 2", "Subject 3"], 2) == {
        "Subject 1": None, "Subject 2": None, "Subject 3": None}
    ok = json.dumps({"matches": [{"subject": "Subject 1", "evidence": "", "box": 1}]})
    assert csl.parse_binding(ok, ["Subject 1"], 1) == {"Subject 1": 1}


def test_parse_orientation_checks_order_and_values():
    good = json.dumps({"images": [{"image": 1, "evidence": "", "orientation": "away"},
                                  {"image": 2, "evidence": "", "orientation": "insufficient_evidence"}]})
    assert csl.parse_orientation(good, 2) == ["away", "insufficient_evidence"]
    bad = json.dumps({"images": [{"image": 2, "evidence": "", "orientation": "away"}]})
    with pytest.raises(ValueError):
        csl.parse_orientation(bad, 1)


# ---------------------------------------------------------------------------
# 第 1 步：固定读法、预写判定、提及顺序、前后
# ---------------------------------------------------------------------------


def test_difference_reading_follows_frozen_thresholds():
    assert csl.difference_reading(0.20, 0.05, 0.33) == "≫"
    assert csl.difference_reading(0.20, -0.01, 0.40) == "未决"       # 差够大但区间含 0
    assert csl.difference_reading(0.03, 0.01, 0.05) == "未决"        # 区间不含 0 但不到 0.05
    assert csl.difference_reading(0.02, -0.06, 0.09) == "≈"
    assert csl.difference_reading(-0.04, -0.12, 0.03) == "≈"
    assert csl.difference_reading(-0.08, -0.15, -0.01) == "明显更差"
    assert csl.difference_reading(-0.08, -0.20, 0.02) == "未决"


def test_step1_verdict_table():
    assert csl.step1_verdict("≫", "≈", ["≈"]) == "文字够"
    assert csl.step1_verdict("≫", "未决", ["未决"]) == "文字够"
    assert csl.step1_verdict("≫", "明显更差", ["≈"]) == "部分"
    assert csl.step1_verdict("≫", "≈", ["明显更差"]) == "未决"
    assert csl.step1_verdict("≈", "明显更差", ["≈"]) == "文字不够"
    assert csl.step1_verdict("未决", "≈", []) == "未决"
    assert csl.step1_verdict("明显更差", "明显更差", []) == "未决"
    with pytest.raises(ValueError):
        csl.step1_verdict(">>", "≈", [])


def test_first_mentioned_on_real_dev30_prompt():
    # 长集 dev lsh-001 s02 的 N 臂逐镜文字（节选到 Design 句之前）。
    prompt = ("A medium shot in Location 2. Subject 3 stands before Subject 2, who is kneeling. Subject 3 holds Subject 4 up "
              "towards Subject 2. Subject 2 takes the bottle, looks at it, and then embraces Subject 3.")
    assert csl.first_mentioned(prompt, "Subject 2", "Subject 3") == "Subject 3"
    assert csl.first_mentioned(prompt, "Subject 3", "Subject 2") == "Subject 3"
    assert csl.first_mentioned(prompt, "Subject 2", "Subject 1") is None
    # 编号按整词比：Subject 1 不被 Subject 10 触发。
    assert csl.first_mentioned("Subject 10 waves at Subject 2. Subject 1 enters.", "Subject 1", "Subject 2") == "Subject 2"


def test_depth_front_needs_one_foreground_and_one_background():
    layout = {"Subject 1": {"depth": "foreground"}, "Subject 2": {"depth": "background"}, "Subject 3": {"depth": None}}
    assert csl.depth_front(layout, "Subject 1", "Subject 2") == "Subject 1"
    assert csl.depth_front(layout, "Subject 2", "Subject 1") == "Subject 1"
    assert csl.depth_front(layout, "Subject 1", "Subject 3") is None
    assert csl.depth_front(layout, "Subject 1", "Subject 9") is None


def test_appearance_phrase_keeps_the_script_description_and_adds_an_article():
    # 长集 dev 的真实描述：两句的只取第一句；没有冠词的补 a／an。
    assert csl.appearance_phrase("A man with short, dark hair, wearing a dark suit, white shirt, and dark tie. "
                                 "His face is glistening with sweat.") == \
        "a man with short, dark hair, wearing a dark suit, white shirt, and dark tie"
    assert csl.appearance_phrase("Older seated woman with grey swept-back hair, wearing a black dress.") == \
        "an older seated woman with grey swept-back hair, wearing a black dress"
    assert csl.appearance_phrase("Giraffe.") == "a giraffe"
    with pytest.raises(ValueError):
        csl.appearance_phrase(" . ")


def test_layout_first_sentence_on_real_dev30_gold():
    attributes = {row["subject_id"]: row["attributes"] for row in _pair("lsh-001")["context_b"]["subjects"]}
    # lsh-001 s02 的金标（真片）：Subject 3 在左、Subject 2 在右；朝向不进首句。
    s02 = {"Subject 2": {"horizontal": "right", "orientation": "screen-left"},
           "Subject 3": {"horizontal": "left", "orientation": "screen-right"}}
    sentence = csl.layout_first_sentence(s02, attributes)
    assert sentence == ("On the left side of the frame is a short adult man with a shaved head, wearing a grey suit; "
                        "on the right side of the frame is a bald man with a thin face, wearing a grey, high-collared suit.")
    assert "Subject" not in sentence
    s01 = {"Subject 1": {"horizontal": "center", "orientation": "toward camera"}}
    assert csl.layout_first_sentence(s01, attributes) == (
        "In the center of the frame is a heavyset man with a round face, yellow jacket, green suspenders and white polo shirt.")
    with_depth = {"Subject 2": {"horizontal": "left", "depth": "background"},
                  "Subject 3": {"horizontal": "right", "depth": "foreground"}}
    assert csl.layout_first_sentence(with_depth, attributes) == (
        "On the left side of the frame, in the background, is a bald man with a thin face, wearing a grey, high-collared suit; "
        "on the right side of the frame, in the foreground, is a short adult man with a shaved head, wearing a grey suit.")
    assert csl.layout_first_sentence({}, attributes) == ""
    assert csl.layout_first_sentence({"Subject 1": {"orientation": "away"}}, attributes) == ""
    with pytest.raises(ValueError):
        csl.layout_first_sentence({"Subject 1": {"horizontal": "middle"}}, attributes)


def test_numbered_space_references_reads_only_the_leading_space_sentence():
    # 新编译器的开头空间句：短属性词撞词时退回 ``Subject k, the <头名词>``；后文散文里的编号不算。
    prompt = ("On the left side of the frame is Subject 4, the man; on the right side of the frame, in the background, is "
              "Subject 2, the man. A medium close-up of Subject 2. Subject 1 enters.")
    assert csl.numbered_space_references(prompt) == frozenset({"Subject 4", "Subject 2"})
    assert csl.numbered_space_references("On the left side of the frame is a woman in yellow top; on the right side of the "
                                         "frame is a man in dark t-shirt. Subject 2 looks at Subject 1.") == frozenset()
    assert csl.numbered_space_references("A medium shot in Location 2. Subject 3 stands before Subject 2.") is None
