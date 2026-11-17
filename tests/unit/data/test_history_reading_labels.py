"""读历史标签：真实块 film-006492-b0000 第 3–4 镜（第 1–2 镜为历史）。这一块里 Character 2 在历史镜的穿着手持
与目标开镜对不上（退为只供身份），目标镜有人镜首没有掩码（写 null）；另核硬条件（参考面积门槛）、弱参考标记、
择优与避开切点。门槛取试训用的那组：参考 ≥2%、5% 以下记弱、镜首 ≥0.5%、切点前后 4 帧。"""
import json

import numpy as np
import pytest

from mspc.data.history_reading_labels import (LabelRules, ShotMasks, box_center, build_labels, closing_state,
                                              opening_state)
from tests.support.paths import REPO_ROOT

RULES = LabelRules(reference_min_area=0.02, weak_area=0.05, start_min_area=0.005, cut_margin=4,
                   max_references=6)
CASE = json.loads((REPO_ROOT / "tests/fixtures/history_reading_labels/film-006492-b0000.json").read_text())


def _labels():
    return build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], CASE["span"], RULES)


def test_real_block_labels_pick_each_needed_person_once_inside_history():
    labels = _labels()
    assert labels["history"] == [0, 1]
    assert sorted(p["name"] for p in labels["picks"]) == sorted(set(p["name"] for p in labels["picks"]))
    masks = ShotMasks(CASE["masks"])
    for pick in labels["picks"]:
        assert pick["shot_index"] in labels["history"]
        assert RULES.cut_margin <= pick["frame"] < CASE["frame_counts"][pick["shot_index"]] - RULES.cut_margin
        mask = masks.mask(pick["shot_index"], pick["name"], pick["frame"])
        assert mask is not None and mask.mean() >= RULES.reference_min_area
        assert pick["weak"] == (mask.mean() < RULES.weak_area)
        x, y = pick["point_2d"]
        assert mask[min(479, y * 480 // 1000), min(831, x * 832 // 1000)]


def test_real_block_marks_identity_only_when_no_history_state_matches():
    labels = _labels()
    pick = next(p for p in labels["picks"] if p["name"] == "Character 2")
    a = CASE["annotation"]
    target = next(i for i in range(*CASE["span"]) if any(e["name"] == "Character 2" for e in a["shots"][i]["present"]))
    want = opening_state(a, "Character 2", target)
    assert not pick["state_match"]
    assert all(closing_state(a, "Character 2", i) != want for i in labels["history"]
               if opening_state(a, "Character 2", i) is not None)


def test_real_block_writes_null_for_people_without_a_first_frame_mask():
    labels = _labels()
    masks = ShotMasks(CASE["masks"])
    nulls = [(s["shot_index"], c["name"]) for s in labels["starts"] for c in s["characters"] if c["point_2d"] is None]
    assert nulls
    for index, name in nulls:
        mask = masks.mask(index, name, 0)
        assert mask is None or mask.mean() < RULES.start_min_area


def test_preference_takes_the_larger_unclipped_view():
    # 两帧都满足硬条件：后一帧更大但碰边（罚半），前一帧居中；择优取分高者。
    big_edge = np.zeros((480, 832), bool)
    big_edge[0:300, 0:300] = True
    small_center = np.zeros((480, 832), bool)
    small_center[100:300, 300:500] = True
    from mspc.data.history_reading_labels import frame_facts, preference

    class Fake(ShotMasks):
        def __init__(self):
            self.cache = {}

        def names(self, index):
            return ["A"]

        def mask(self, index, name, frame):
            return big_edge if frame == 8 else small_center

    facts = {f: frame_facts(Fake(), 0, "A", f, {"A"}, RULES.reference_min_area) for f in (4, 8)}
    assert facts[8]["border"] and not facts[4]["border"]
    assert preference(facts[8]) == pytest.approx(0.5 * big_edge.mean())
    assert box_center(small_center) == [481, 417]  # 框 x 300–500、y 100–300 的中心


def test_small_views_do_not_qualify_as_references():
    from mspc.data.history_reading_labels import frame_facts

    tiny = np.zeros((480, 832), bool)
    tiny[0:60, 0:100] = True  # 约 1.5% 的画面

    class Fake(ShotMasks):
        def __init__(self):
            self.cache = {}

        def names(self, index):
            return ["A"]

        def mask(self, index, name, frame):
            return tiny

    assert frame_facts(Fake(), 0, "A", 4, {"A"}, RULES.reference_min_area) is None
    assert frame_facts(Fake(), 0, "A", 4, {"A"}, RULES.start_min_area) is not None


def test_rules_reject_inverted_thresholds():
    with pytest.raises(ValueError):
        LabelRules(reference_min_area=0.05, weak_area=0.02, start_min_area=0.005, cut_margin=4, max_references=6)


def test_cap_keeps_the_best_moments_and_names_who_lost_a_reference():
    from dataclasses import replace

    # 同一块第 8–9 镜：两个所需角色各在一个时刻。
    full = build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], [7, 9], RULES)
    assert len(full["moments"]) == 2 and full["dropped_by_cap"] == []
    capped = build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], [7, 9],
                          replace(RULES, max_references=1))
    assert len(capped["moments"]) == 1
    best = max(full["picks"], key=lambda p: p["preference"])
    assert capped["moments"] == [[best["shot_index"], best["frame"]]]
    assert sorted(capped["dropped_by_cap"]) == sorted(p["name"] for p in full["picks"]
                                                      if [p["shot_index"], p["frame"]] != capped["moments"][0])


def test_labels_carry_the_rules_they_were_built_with():
    from dataclasses import asdict

    assert _labels()["rules"] == asdict(RULES)
