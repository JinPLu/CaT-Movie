"""出片时的规划器 token（点控制两组）：在真实块 film-006492-b0000 第 8–9 镜上，标签镜首点经推理端的
``token_condition`` 放置，与训练端（同一个 ``token_placement``）逐项一致；计划镜首框取中心；移点改那人那镜的点、
换 token 只交换那一镜两人的内容；作答只有镜首段时取特征的条目顺序；第一步作答的核对与提示。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mspc.data.history_reading_labels import LabelRules, build_labels
from mspc.data.shot_targets import latent_frames
from mspc.inference.planner_token_features import entry_spans
from mspc.inference.planner_token_reading import (PlannerTokenReader, TokenEdit, apply_token_edits, person_numbers,
                                                  swap_rows, token_condition)
from mspc.inference.reference_pointing import (answer_from_starts, plan_starts, step_one_content, step_one_moments,
                                               target_people)
from mspc.models.planner_tokens import point_cell
from tests.support.paths import REPO_ROOT

CASE = json.loads((REPO_ROOT / "tests/fixtures/history_reading_labels/film-006492-b0000.json").read_text())
RULES = LabelRules(reference_min_area=0.02, weak_area=0.05, start_min_area=0.005, cut_margin=4, max_references=6)
SPAN = (7, 9)


def _labels():
    return build_labels(CASE["annotation"], CASE["masks"], CASE["frame_counts"], CASE["usable"], list(SPAN), RULES)


def test_inference_places_tokens_like_training():
    # 这一块镜首没有掩码（标签点都是 null），按标签的人与镜给点，只核放置。
    labels = _labels()
    starts = [(start["shot_index"], c["name"], [300 + 100 * k, 400])
              for k, start in enumerate(labels["starts"]) for c in start["characters"]]
    starts.append((SPAN[0], "Character 9", None))
    lengths = [latent_frames(CASE["frame_counts"][i]) for i in range(*SPAN)]
    features = np.arange(len(starts) * 4, dtype=np.float32).reshape(len(starts), 4)
    condition = token_condition(starts, features, span=SPAN, latent_lengths=lengths)
    first = [0, lengths[0]]
    expected = [(k, index - SPAN[0], first[index - SPAN[0]], *point_cell(point))
                for k, (index, _, point) in enumerate(starts) if point is not None]
    assert list(condition.shot_tokens) == expected
    assert condition.features.shape == (len(starts), 4)


def test_the_answer_has_only_start_entries_in_start_order():
    starts = [(7, "Character 1", [100, 200]), (7, "Character 2", None), (8, "Character 1", [900, 500])]
    answer = answer_from_starts(starts)
    assert json.loads(answer) == {"shots": [
        {"shot": 8, "characters": [{"name": "Character 1", "point_2d": [100, 200]},
                                   {"name": "Character 2", "point_2d": None}]},
        {"shot": 9, "characters": [{"name": "Character 1", "point_2d": [900, 500]}]}]}
    assert [(section, key, name) for section, key, name, _ in entry_spans(answer)] == [
        ("shot", 8, "Character 1"), ("shot", 8, "Character 2"), ("shot", 9, "Character 1")]


def test_plan_starts_take_only_the_visible_time_zero_anchor_box_center():
    board = {**CASE["annotation"], "layouts": [{"shot": 8, "entities": [
        {"name": name, "anchors": [{"time": 0.5, "visibility": "visible", "box": [0.0, 0.0, 0.2, 0.2]},
                                   {"time": 0.0, "visibility": "visible", "box": [0.1, 0.2, 0.3, 0.6]}]}
        for name in target_people(CASE["annotation"], SPAN)[8][:1]]}]}
    starts = plan_starts(board, SPAN)
    names = target_people(CASE["annotation"], SPAN)
    assert [(i, n) for i, n, _ in starts] == [(shot - 1, n) for shot, people in sorted(names.items()) for n in people]
    assert starts[0][2] == [200, 400] and all(p is None for _, _, p in starts[1:])


@pytest.mark.parametrize("opening", [
    None,
    {"time": 0.0, "visibility": "not_visible", "box": None},
    {"time": 0.0, "visibility": "visible", "box": None},
])
def test_late_visible_anchor_cannot_create_an_opening_token(opening):
    name = target_people(CASE["annotation"], SPAN)[8][0]
    # 实际 real-slice member53 的两锚时间与框；该人物在镜中后段才可见。
    late = [{"time": 0.37, "visibility": "visible", "box": [0.79, 0.88, 0.88, 0.96]},
            {"time": 1.0, "visibility": "visible", "box": [0.69, 0.6, 0.89, 0.95]}]
    anchors = late if opening is None else [opening, late[0]]
    board = {**CASE["annotation"], "layouts": [{"shot": 8, "entities": [{"name": name, "anchors": anchors}]}]}
    starts = plan_starts(board, SPAN)
    assert all(point is None for _, _, point in starts)
    assert token_condition(starts, person_numbers(starts), span=SPAN, latent_lengths=[3, 3]) is None


def test_real_late_entry_layout_keeps_its_events_without_opening_points():
    case = json.loads((REPO_ROOT / "tests/fixtures/block_annotation_v18/layout-token-overflow.json").read_text())["cases"][2]
    board = case["annotation"]
    starts = {(shot, name): point for shot, name, point in plan_starts(board, (0, len(board["shots"])))}
    layout = board["layouts"][0]
    shot = layout["shot"] - 1
    late_names = [entity["name"] for entity in layout["entities"]
                  if entity["name"].startswith("Character") and entity["anchors"][0]["time"] > 0]
    assert late_names == ["Character 4", "Character 5"]
    assert all(starts[(shot, name)] is None for name in late_names)
    assert starts[(shot, "Character 1")] == [710, 590]
    assert starts[(shot, "Character 2")] == [660, 440]
    # 仍用同一份实际annotation编译完整文字，后段人物与运动没有被删成空布局。
    from mspc.data.shot_text import shot_text
    text = shot_text(board, shot)
    assert late_names[0] in text and late_names[1] in text


def test_move_and_swap_touch_only_the_named_shot():
    starts = [(2, "Character 1", [100, 100]), (2, "Character 2", [800, 100]), (3, "Character 1", [500, 500])]
    moved, applied = apply_token_edits(starts, [TokenEdit(kind="move", names=("Character 1",), shot=2,
                                                          point=(800, 100))])
    assert moved[0] == (2, "Character 1", [800, 100]) and moved[2] == starts[2]
    assert applied[0]["moved_from"] == [100, 100]
    with pytest.raises(ValueError, match="no start entry"):
        apply_token_edits(starts, [TokenEdit(kind="move", names=("Character 3",), shot=2, point=(1, 1))])
    numbers = person_numbers(starts)
    assert numbers.tolist() == [1, 2, 1]
    swapped = swap_rows(numbers, starts, [TokenEdit(kind="swap", names=("Character 1", "Character 2"), shot=2)])
    assert swapped.tolist() == [2, 1, 1]
    features = np.arange(3, dtype=np.float32)[:, None].repeat(2, 1)
    assert swap_rows(features, starts, [TokenEdit(kind="swap", names=("Character 1", "Character 2"),
                                                  shot=2)])[:, 0].tolist() == [1, 0, 2]
    with pytest.raises(ValueError, match="does not have"):
        swap_rows(features, starts, [TokenEdit(kind="swap", names=("Character 1", "Character 3"), shot=2)])
    edit = TokenEdit(kind="move", names=("A",), shot=2, point=(100, 200))
    assert TokenEdit.from_mapping(json.loads(json.dumps(edit.to_mapping()))) == edit


def test_remove_drops_the_token_and_all_removed_means_no_tokens():
    starts = [(2, "Character 1", [100, 100]), (2, "Character 2", [800, 100]), (3, "Character 1", [500, 500])]
    removed, applied = apply_token_edits(starts, [TokenEdit(kind="remove", names=("Character 2",), shot=2)])
    assert removed[1] == (2, "Character 2", None) and removed[0] == starts[0] and applied[0]["moved_from"] == [800, 100]
    everything = [TokenEdit(kind="remove", names=(n,), shot=i) for i, n, _ in starts]
    cleared, _ = apply_token_edits(starts, everything)
    assert token_condition(cleared, person_numbers(cleared), span=(2, 4), latent_lengths=[3, 3]) is None
    with pytest.raises(ValueError, match="no point"):
        TokenEdit(kind="remove", names=("A",), shot=2, point=(1, 1))


def test_step_one_check_merges_moments_and_drops_bad_picks():
    frames = {0: 40, 1: 32}
    moments, picks, dropped = step_one_moments({"picks": [{"name": "A", "shot": 1, "time": 1.0},
                                                          {"name": "B", "shot": 1, "time": 1.0}]},
                                               needed=["A", "B"], history_frames=frames, cut_margin=4, max_moments=6)
    assert moments == [[0, 16]] and [p["name"] for p in picks] == ["A", "B"] and dropped == []
    # 切点前后 4 帧不选：第 0 帧收进第 4 帧，末尾收进倒数第 5 帧；原选帧照记。
    moments, picks, _ = step_one_moments({"picks": [{"name": "A", "shot": 1, "time": 0.0},
                                                    {"name": "B", "shot": 2, "time": 2.0}]},
                                         needed=["A", "B"], history_frames=frames, cut_margin=4, max_moments=6)
    assert moments == [[0, 4], [1, 27]] and [p["chosen_frame"] for p in picks] == [0, 31]
    short = step_one_moments({"picks": [{"name": "A", "shot": 1, "time": 0.0}]}, needed=["A"],
                             history_frames={0: 6}, cut_margin=4, max_moments=6)
    assert short[0] == [[0, 3]] and short[1][0]["short_shot"] and not picks[0]["short_shot"]
    # 不合格的 pick 丢掉并记录，这个人这一段没有参考；不报错。
    for bad in ({"picks": [{"name": "A", "shot": 3, "time": 0.5}]},
                {"picks": [{"name": "A", "shot": 2, "time": 9.0}]},
                {"picks": []},
                {"picks": [{"name": "A", "shot": 1, "time": 0.5, "point_2d": [1, 2]}]},
                "not an object"):
        moments, picks, dropped = step_one_moments(bad, needed=["A"], history_frames=frames, cut_margin=4, max_moments=6)
        assert moments == [] and picks == [] and dropped[-1]["reason"] == "no usable pick; no reference this segment"


def test_step_one_keeps_the_first_moments_in_answer_order_up_to_the_cap():
    frames = {index: 80 for index in range(6)}
    picks = {"picks": [{"name": name, "shot": shot, "time": 2.0}
                       for name, shot in (("A", 6), ("B", 2), ("C", 2), ("D", 4), ("E", 1), ("F", 3))]}
    moments, kept, dropped = step_one_moments(picks, needed=list("ABCDEF"), history_frames=frames, cut_margin=4,
                                              max_moments=4)
    # 同一刻不多占名额；第五个不同时刻起按作答顺序丢掉，那个人这一段没有参考；时刻仍按镜序排。
    assert moments == [[0, 32], [1, 32], [3, 32], [5, 32]]
    assert [p["name"] for p in kept] == list("ABCDE")
    assert [d["value"] for d in dropped if d["reason"] == "more than 4 moments"] == [picks["picks"][5]]
    assert [d["value"] for d in dropped if d["reason"].startswith("no usable pick")] == ["F"]


def test_step_one_consolidates_each_shot_by_entity_votes_before_applying_capacity():
    answer = {"picks": [{"name": "A", "shot": 1, "time": 0.5},
                        {"name": "B", "shot": 1, "time": 1.0},
                        {"name": "C", "shot": 1, "time": 1.0},
                        {"name": "D", "shot": 2, "time": 0.5}]}
    moments, kept, dropped = step_one_moments(answer, needed=list("ABCD"), history_frames={0: 80, 1: 80},
                                              cut_margin=4, max_moments=2)
    assert moments == [[0, 16], [1, 8]]
    assert [row["name"] for row in kept] == list("BCD")
    assert all(row["source_binding_requires_repointing"] for row in kept)
    discarded = next(row for row in dropped if row["reason"] == "one_reference_frame_per_shot")
    assert discarded["value"] == answer["picks"][0]
    assert discarded["candidate"]["frame"] == 8 and discarded["selected_frame"] == 16
    assert discarded["entity_votes"] == 1 and discarded["selected_entity_votes"] == 2
    assert discarded["source_binding_requires_repointing"]
    assert not any(row["name"] == "A" for row in kept)


def test_step_one_equal_votes_keep_the_first_answered_frame_not_the_first_video_frame():
    moments, kept, dropped = step_one_moments({"picks": [{"name": "A", "shot": 1, "time": 1.5},
                                                          {"name": "B", "shot": 1, "time": 0.5}]},
        needed=["A", "B"], history_frames={0: 80}, cut_margin=4, max_moments=10)
    assert moments == [[0, 24]] and [row["name"] for row in kept] == ["A"]
    assert dropped[0]["reason"] == "one_reference_frame_per_shot" and dropped[0]["selected_frame"] == 24


@pytest.mark.parametrize("capacity", [0, 11, True])
def test_step_one_reference_capacity_matches_the_renderer(capacity):
    with pytest.raises(ValueError, match="1..10"):
        step_one_moments({"picks": []}, needed=[], history_frames={}, cut_margin=4, max_moments=capacity)


def test_select_moments_needs_no_model_call_without_needed_characters_or_history():
    reader = PlannerTokenReader.__new__(PlannerTokenReader)
    reader.cut_margin = 4
    reader.processor = SimpleNamespace(video_processor=None)
    reader._generate = lambda *args, **kwargs: pytest.fail("no model call without needed characters or history")
    common = dict(span=(1, 2), target_prompts=["next"], max_moments=4)
    empty = {"needed": [], "moments": [], "step_one_picks": []}
    assert reader.select_moments(CASE["annotation"], history={0: np.zeros((8, 2, 2, 3))}, needed=[], **common) == empty
    assert reader.select_moments(CASE["annotation"], history={}, needed=[], **common) == empty


def test_step_one_prompt_has_one_clip_per_history_shot():
    labels = _labels()
    content = step_one_content(CASE["annotation"], span=SPAN,
                               history_frames={i: CASE["frame_counts"][i] for i in labels["history"]},
                               target_prompts=[f"prompt {i}" for i in range(*SPAN)], needed=labels["needed"])
    assert sum(part["type"] == "video" for part in content) == len(labels["history"])
    assert f"Shot {SPAN[0]} (" in "".join(p.get("text", "") for p in content)
