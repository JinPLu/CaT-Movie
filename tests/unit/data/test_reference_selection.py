"""推理端按分镜选参考帧（锚视角）：每个在场实体取外观不变期间同框人少、景别清楚、最早的一镜，地点取人少、景别远、最早的
一镜；去重、上限、按镜序；取帧位置。

分镜用开发集 30 个故事的分镜表（``configs/benchmark/benchmark_storyboards_v4.json``），窗按 128 源帧切、镜数不限。"""
import json

import pytest

from mspc.benchmark.layout_baseline import execution_storyboard
from mspc.comparison_methods.prompts import adapt_storyboard
from mspc.data.reference_routing import reference_readers
from mspc.data.reference_selection import anchor_key, place_key, reference_position, select_references
from tests.support.paths import REPO_ROOT

DEV = [story for story in json.loads((REPO_ROOT / "configs/benchmark/benchmark_storyboards_v4.json").read_text())["stories"]
       if story["split"] == "dev"]


def _board(story_id):
    return execution_storyboard(next(story["storyboard"] for story in DEV if story["story_id"] == story_id))


def _names(board, shot):
    return [entry["name"] for entry in board["shots"][shot]["present"]]


def _anchor(board, name, target, before=None):
    """``name`` 在窗起点 ``before``（默认就是目标镜）之前出现过的镜里，按锚视角排序键取的那一镜。"""
    seen = [index for index in range(target if before is None else before) if name in _names(board, index)]
    return min(seen, key=lambda index: anchor_key(board, candidate=index, target=target, name=name))


def test_a_character_reads_its_view_with_fewest_other_people_not_its_latest_appearance():
    # 抽帧复查的 cinedance-014617 第 5 镜：Character 2 最近一次出现在第 4 镜（与另两人同框的中景，生成时他已走样），
    # 更早的第 2 镜是他单人的中近景；地点取同地点同框人最少的最早一镜，也是第 2 镜。
    board = _board("cinedance-014617-b0002-shot01")
    assert _names(board, 3) == ["Character 2", "Character 1", "Character 8"] and _names(board, 1) == ["Character 2"]
    assert _names(board, 4) == ["Character 2"]
    assert select_references(board, span=(4, 5), candidates=range(4), frames_max=10) == [1]


def test_equally_clean_views_go_to_the_earliest_so_the_anchor_stays_fixed_across_windows():
    # cinedance-011050：Character 1 在第 6 镜、第 11 镜都是单人特写；第 11 镜与第 12 镜两窗都读第 6 镜，不读刚生成的第 11 镜。
    board = _board("cinedance-011050-b0000-shot01")
    assert _names(board, 5) == _names(board, 10) == ["Character 1"]
    assert _anchor(board, "Character 1", 10) == _anchor(board, "Character 1", 11) == 5
    picked = select_references(board, span=(11, 12), candidates=range(11), frames_max=10)
    assert 5 in picked and 10 not in picked


def test_views_from_before_a_recorded_outfit_change_rank_last():
    # film-000127：Character 1 在第 3 镜换装（grey suit → pink shirt and dark trousers）；第 4 镜他的锚是第 3 镜，
    # 不是更早、单人特写但衣着过时的第 2 镜。第 5 镜再换装后，第 6 镜只读第 5 镜（远景，但衣着是对的）。
    board = _board("film-000127-b0000-shot01")
    assert [change["what"] for change in board["shots"][2]["changes"]] == ["outfit"] and _names(board, 1) == ["Character 1"]
    assert _anchor(board, "Character 1", 3) == 2
    assert _anchor(board, "Character 1", 5) == 4


def test_picking_something_up_does_not_move_the_anchor():
    # cinedance-012473 第 2 镜 Character 1 拿起物体（手持物变化，人在走廊远处、与两人同框）；第 3 镜仍读第 1 镜他的单人视角。
    board = _board("cinedance-012473-b0000-shot01")
    assert {change["what"] for change in board["shots"][1]["changes"] if change["name"] == "Character 1"} == {"holding"}
    assert _anchor(board, "Character 1", 2) == 0


def test_place_view_is_the_same_place_shot_with_fewest_people_then_the_widest():
    # cinedance-011050 第 11 镜回到 Location 1：第 2、3 镜都只有物体、没人，第 2 镜特写比第 3 镜大特写远，取第 2 镜。
    board = _board("cinedance-011050-b0000-shot01")
    assert [board["shots"][index]["camera"]["shot_size"] for index in (1, 2)] == ["close-up shot", "extreme close-up shot"]
    place = [index for index in range(10) if board["shots"][index]["place"] == "Location 1"]
    assert min(place, key=lambda index: place_key(board, candidate=index)) == 1
    assert select_references(board, span=(10, 11), candidates=range(10), frames_max=10) == [1, 5]


def test_references_are_collected_by_target_then_present_then_place_deduplicated_and_capped():
    # cinedance-012429 第 6–10 镜一窗：第 6 镜 Character 5 → 第 2 镜（单人特写）、地点 → 第 4 镜（单人中近景，比特写远）；
    # 第 7 镜 Character 3 → 第 1 镜（只出现过这一次）、Object 1 → 第 5 镜（同框一人）；其余重复。
    board = _board("cinedance-012429-b0002-shot01")
    assert select_references(board, span=(5, 10), candidates=range(5), frames_max=10) == [0, 1, 2, 3, 4]
    # 上限 3：按收集顺序留前三张（第 2、4、1 镜），按镜序返回。
    assert select_references(board, span=(5, 10), candidates=range(5), frames_max=3) == [0, 1, 3]
    assert select_references(board, span=(5, 10), candidates=[2, 3], frames_max=10) == [2, 3]


def test_every_entity_seen_before_its_window_gets_a_view_routed_to_its_target_on_every_dev_window():
    """开发集 30 个故事每一窗：目标镜里在场、窗前出现过的登记实体，选中的参考里至少一张有它，且按训练同一个读者分派
    路由给这个目标镜（上限 10 张在开发集上从不截断）。"""
    windows = 0
    for story in DEV:
        board = execution_storyboard(story["storyboard"])
        registered = {entry["name"] for entry in [*board["characters"], *board.get("objects", ())]}
        for segment in adapt_storyboard(board, family="wan_segment", count_tokens=lambda text: len(text.split()),
                                        limits=dict(max_shots=None, max_frames=128, max_tokens=512)):
            start, end = segment["start_shot"], segment["end_shot"]
            picked = select_references(board, span=(start, end), candidates=range(start), frames_max=10)
            assert picked == sorted(set(picked)) and len(picked) <= 10 and all(index < start for index in picked)
            if not picked:
                continue
            routing, _ = reference_readers("routed_self_only_v1", board["shots"], reference_shots=picked,
                                           target_shots=range(start, end))
            for target, entry in zip(range(start, end), routing):
                for name in set(_names(board, target)) & registered:
                    if any(name in _names(board, index) for index in range(start)):
                        assert _anchor(board, name, target, start) in picked
                        assert any(name in _names(board, picked[slot]) for slot in entry)
            windows += 1
    assert windows == 117


@pytest.mark.parametrize("frames_max", [0, 11, True])
def test_capacity_is_an_int_from_one_to_ten(frames_max):
    with pytest.raises(ValueError, match="1..10"):
        select_references(_board("cinedance-012429-b0002-shot01"), span=(5, 10), candidates=range(5),
                          frames_max=frames_max)


@pytest.mark.parametrize("span,candidates", [((5, 10), [5]), ((5, 10), [1, 1]), ((5, 10), [10]), ((5, 5), [1]),
                                             ((5, 11), [1])])
def test_candidates_must_be_unique_shots_outside_the_target_span(span, candidates):
    with pytest.raises(ValueError):
        select_references(_board("cinedance-012429-b0002-shot01"), span=span, candidates=candidates, frames_max=10)


def test_frame_position_is_the_last_frame_only_for_an_outfit_or_state_change():
    # film-000129：第 4 镜只有手持物变化、第 6、8 镜只有姿态变化，都取中帧；第 5 镜换装取末帧。
    board = _board("film-000129-b0000-shot01")
    assert [[change["what"] for change in shot["changes"]] for shot in board["shots"][3:8]] == [
        ["holding"], ["outfit"], ["posture"], [], ["posture"]]
    assert [reference_position(shot) for shot in board["shots"]] == ["middle"] * 4 + ["last"] + ["middle"] * 7
    # film-000123 第 3 镜「转身看身后」：只有姿态变化，取中帧（末帧是背影）。
    assert [change["what"] for change in _board("film-000123-b0001-shot01")["shots"][2]["changes"]] == ["posture"]
    assert reference_position(_board("film-000123-b0001-shot01")["shots"][2]) == "middle"
