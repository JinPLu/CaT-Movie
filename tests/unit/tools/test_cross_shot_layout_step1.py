"""跨镜布局第 1 步读数的工具侧：逐单元行、配对差与对天花板的率差（纯计算，合成输入）。"""
from __future__ import annotations

import pytest

from tools.run.cross_shot_layout import cluster_rate_gap, paired_difference, step1_block, step1_comparison, step1_rows

IDENTITY = {"prior": -0.8, "paths": {"dino": [10.0, -6.0]}}


def _measure(left_box, right_box, *, swap_embeddings=False, bound=True):
    """两人镜：Subject 1 的框在 ``left_box``，Subject 2 在 ``right_box``；嵌入是两条正交向量（可对调）。"""

    e1, e2 = ([1.0, 0.0], [0.0, 1.0]) if not swap_embeddings else ([0.0, 1.0], [1.0, 0.0])
    boxes = {"Subject 1": left_box if bound else None, "Subject 2": right_box}
    return {"binding": {"Subject 1": "person#0" if bound else None, "Subject 2": "person#1"},
            "source": {"Subject 1": "phrase" if bound else "vlm", "Subject 2": "phrase"},
            "tracks": {"boxes": [boxes] * 3, "embeddings": {"Subject 1": e1 if bound else None, "Subject 2": e2}},
            "vectors": {"Subject 1": {"dino": e1} if bound else {}, "Subject 2": {"dino": e2}},
            "layout": {"Subject 1": {"depth": "foreground"}, "Subject 2": {"depth": "background"}} if bound else {},
            "mid_boxes": boxes, "all_mid_boxes": {"person#0": left_box, "person#1": right_box}, "mid_position": 5}


LEFT, RIGHT = [50.0, 100.0, 250.0, 400.0], [550.0, 100.0, 750.0, 400.0]
WINDOWS = {"width": 832, "windows": {"w1": {
    "subjects": [{"subject_id": "Subject 1", "kind": "person", "attributes": "A man in a blue suit."},
                 {"subject_id": "Subject 2", "kind": "person", "attributes": "A woman in a red dress."}],
    "shots": [{"shot_id": sid, "location_id": "Location 1", "animate": ["Subject 1", "Subject 2"]} for sid in ("s01", "s02", "s03")]}}}
GOLD = {"w1": {sid: {"layout": {"Subject 1": {"horizontal": h1, "depth": "foreground"}, "Subject 2": {"horizontal": h2, "depth": "background"}},
                     "binding_wrong": []}
               for sid, h1, h2 in (("s01", "left", "right"), ("s02", "left", "right"), ("s03", "right", "left"))}}
PROMPTS = {"w1": ["Subject 1 greets Subject 2.", "Subject 2 answers Subject 1.", "Subject 1 leaves Subject 2."]}


def _rows(measures):
    return step1_rows(WINDOWS, measures, GOLD, {}, PROMPTS, IDENTITY)


def test_step1_rows_relations_identity_rules_and_obedience():
    measures = {("w1", "s01"): _measure(LEFT, RIGHT), ("w1", "s02"): _measure(LEFT, RIGHT, swap_embeddings=True),
                ("w1", "s03"): _measure(RIGHT, LEFT)}
    rows = _rows(measures)
    pairs = {tuple(row["shots"]): row for row in rows["pairs"]}
    # s01→s03：生成片把两人对调了，金标也对调，一致；两路一致。
    assert pairs[("s01", "s03")]["strict"] == "reversed" == pairs[("s01", "s03")]["gold"]
    # s01→s02：镜内指认说同序，跨镜外观说两人对调，两路不一致，主口径判不了；只用镜内指认时算同序。
    assert pairs[("s01", "s02")]["strict"] is None and pairs[("s01", "s02")]["strict_reason"] == "binding_disagree"
    assert pairs[("s01", "s02")]["path1"] == "same" and pairs[("s01", "s02")]["fused"] is None
    assert pairs[("s01", "s03")]["layer"] == "same_run" and pairs[("s01", "s02")]["layer"] == "adjacent"
    block = step1_block(rows)
    assert block["q1"]["strict"]["judgeable"]["num"] == 1 and block["q1"]["gold_defined"] == 3
    assert block["q1"]["path1"]["consistent"]["num"] == 3
    # 逐镜：三镜的工具左边那位都与金标相同；先提到的人在左只有 s01（s02 先提到 Subject 2，她在右；s03 先提到的 Subject 1 在右）。
    assert block["per_shot"]["obeys_oracle"]["num"] == 3
    assert block["per_shot"]["first_mentioned_on_left"]["num"] == 1 and block["per_shot"]["first_mentioned_on_left"]["den"] == 3
    assert block["presence"]["script_present_bound"]["rate"] == 1.0
    assert block["depth"]["same_front_among_bound"]["num"] == 3


def test_step1_rows_unbound_subject_is_unjudgeable_and_absent():
    measures = {("w1", "s01"): _measure(LEFT, RIGHT), ("w1", "s02"): _measure(LEFT, RIGHT, bound=False),
                ("w1", "s03"): _measure(RIGHT, LEFT)}
    rows = _rows(measures)
    pairs = {tuple(row["shots"]): row for row in rows["pairs"]}
    assert pairs[("s01", "s02")]["strict"] is None and pairs[("s01", "s02")]["strict_reason"] == "order_undecided"
    block = step1_block(rows)
    assert block["presence"]["script_present_bound"]["num"] == 5 and block["presence"]["script_present_bound"]["den"] == 6
    assert block["per_shot"]["decided"]["num"] == 2
    assert block["depth"]["both_bound"] == 2


def test_paired_difference_and_gap_readings():
    def units(story, values):
        return [{"unit": f"{story}|{k}", "story": story, "ok": bool(v)} for k, v in enumerate(values)]

    better = [row for s in range(8) for row in units(f"w{s}", [1, 1, 1, 1])]
    worse = [row for s in range(8) for row in units(f"w{s}", [1, 0, 0, 0])]
    diff = paired_difference(better, worse, usable=lambda r: True, value=lambda r: r["ok"])
    assert diff["units"] == 32 and diff["difference"] == pytest.approx(0.75) and diff["reading"] == "≫"
    same = paired_difference(better, better, usable=lambda r: True, value=lambda r: r["ok"])
    assert same["difference"] == 0 and same["reading"] == "≈"
    # 只在两边都可用的单元上配对。
    half = paired_difference(better, worse, usable=lambda r: r["unit"].endswith("|0"), value=lambda r: r["ok"])
    assert half["units"] == 8 and half["difference"] == 0
    gap = cluster_rate_gap(worse, better, value=lambda r: r["ok"])
    assert gap["difference"] == pytest.approx(-0.75) and gap["reading"] == "明显更差"
    assert cluster_rate_gap(better, better[:16], value=lambda r: r["ok"])["reading"] == "≈"


def test_step1_comparison_pairs_units_across_arms():
    obedient = {("w1", "s01"): _measure(LEFT, RIGHT), ("w1", "s02"): _measure(LEFT, RIGHT), ("w1", "s03"): _measure(RIGHT, LEFT)}
    stuck = {("w1", sid): _measure(LEFT, RIGHT) for sid in ("s01", "s02", "s03")}
    comparison = step1_comparison(_rows(obedient), _rows(stuck))
    # 不听话的臂三镜都把 Subject 1 放左边：s01→s03、s02→s03 与金标（对调）不一致。
    assert comparison["q1_strict"]["units"] == 3
    assert comparison["q1_strict"]["first_rate"] == 1.0 and comparison["q1_strict"]["second_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert comparison["per_shot_obeys_oracle"]["second_rate"] == pytest.approx(2 / 3, abs=1e-4)
