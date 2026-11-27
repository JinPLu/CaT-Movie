"""参考按镜路由的规则：给与参考源镜共享在场实体的目标镜，一镜都不共享时给全部目标镜；不按场所。"""
import json

import pytest

from mspc.data.reference_routing import route_references, select_past_reference_shots
from tests.support.paths import REPO_ROOT


def _real_block_annotation():
    return json.loads((REPO_ROOT / "tests/fixtures/sampling_ledger/block-annotations.jsonl")
                      .read_text().splitlines()[1])["annotation"]


def _shots(*rows):
    return [{"shot": number, "present": [{"name": name} for name in names], **extra}
            for number, (names, extra) in enumerate(rows, 1)]


def test_real_block_routes_by_shared_entity_and_sends_an_unshared_object_shot_to_every_target():
    # 真实块标注：镜1只有物体；目标镜2（Character 1、2）与镜3（Character 2）；参考取自镜1、4、5。
    annotation = _real_block_annotation()
    routing, reasons = route_references(annotation["shots"], reference_shots=[0, 3, 4], target_shots=range(1, 3))
    assert routing == ((0, 1, 2), (0,))
    assert reasons == ("all", "entity", "entity")


def test_a_shared_place_without_a_shared_entity_goes_to_every_target():
    shots = _shots((["Object 1"], {"place": "Location 2"}), (["Character 1"], {"place": "Location 1"}),
                   (["Character 2"], {"place": "Location 2"}))
    assert route_references(shots, reference_shots=[0], target_shots=[1, 2]) == (((0,), (0,)), ("all",))


def test_every_reference_reaches_some_target_and_each_shot_lists_references_in_order():
    shots = _shots((["Character 1"], {}), (["Character 2"], {}), (["Object 1"], {}),
                   (["Character 1", "Character 2"], {}), (["Character 2"], {}))
    routing, reasons = route_references(shots, reference_shots=[0, 1, 2], target_shots=[3, 4])
    assert routing == ((0, 1, 2), (1, 2))
    assert reasons == ("entity", "entity", "all")
    assert set().union(*routing) == {0, 1, 2}


@pytest.mark.parametrize("references,targets", [([0], []), ([5], [1]), ([0], [True])])
def test_routing_refuses_missing_targets_and_indices_outside_the_annotation(references, targets):
    with pytest.raises(ValueError, match="reference routing needs"):
        route_references(_shots((["Character 1"], {}), (["Character 1"], {})), reference_shots=references,
                         target_shots=targets)


def test_past_selector_only_uses_eligible_history_and_never_fills_unrelated_references():
    annotation = _real_block_annotation()
    # 真实块首镜只有Object，目标镜2要人物；有容量也不能补这张无关历史。
    assert select_past_reference_shots(annotation, span=(1, 2), eligible_past_shots=[0], frames_max=10) == []
    # 目标镜4/5需要的人物只在镜2共享，镜1虽可用但不相关。
    assert select_past_reference_shots(annotation, span=(3, 5), eligible_past_shots=[0, 1], frames_max=10) == [1]
    assert select_past_reference_shots(annotation, span=(3, 5), eligible_past_shots=[0], frames_max=10) == []
    assert select_past_reference_shots(annotation, span=(0, 2), eligible_past_shots=[], frames_max=10) == []


def test_actual_reference_slots_filter_verified_ids_without_inventing_visibility_from_present():
    from mspc.data.reference_routing import reference_slot_metadata
    shots = _shots((["Character 1"], {}), (["Character 2"], {}))
    sources = [{"entities": [{"name": "Character 1", "identity_status": "registered"}]},
               {"entities": [{"name": "Character 2", "identity_status": "unresolved"}]}]
    routed = reference_slot_metadata(shots, target_shots=[0, 1], reference_count=2,
                                     reference_entities=sources, routing=((0,), (1,)))
    assert routed == (({"index": 1, "entities": ["Character 1"]},), ({"index": 2, "entities": []},))
    # 非路由模型确实读全部slots；目标present里有Character2不证明源帧认出了Character2。
    full = reference_slot_metadata(shots, target_shots=[0, 1], reference_count=2,
                                   reference_entities=sources, routing=None)
    assert full[1] == ({"index": 1, "entities": []}, {"index": 2, "entities": []})
    unknown = reference_slot_metadata(shots, target_shots=[0, 1], reference_count=1,
                                      reference_entities=None, routing=None)
    assert unknown == (({"index": 1, "entities": []},), ({"index": 1, "entities": []},))
    assert reference_slot_metadata(shots, target_shots=[0, 1], reference_count=0,
                                    reference_entities=None, routing=None) == ((), ())


@pytest.mark.parametrize("past", [[0, 3], [4], [1, 1], [True]])
def test_past_selector_rejects_inside_future_duplicate_or_noninteger_candidate(past):
    with pytest.raises(ValueError, match="unique usable shots before"):
        select_past_reference_shots(_real_block_annotation(), span=(3, 5), eligible_past_shots=past, frames_max=10)


def test_window_shared_policy_gives_every_target_all_references_on_the_real_block():
    from mspc.data.reference_routing import reference_readers

    annotation = json.loads((REPO_ROOT / "tests/fixtures/sampling_ledger/block-annotations.jsonl")
                            .read_text().splitlines()[1])["annotation"]
    routing, reasons = reference_readers("window_shared_self_only_v1", annotation["shots"],
                                         reference_shots=[0, 3, 4], target_shots=range(1, 3))
    assert routing == ((0, 1, 2), (0, 1, 2))
    assert reasons == ("window", "window", "window")
    routed = reference_readers("routed_self_only_v1", annotation["shots"], reference_shots=[0, 3, 4], target_shots=range(1, 3))
    assert routed == route_references(annotation["shots"], reference_shots=[0, 3, 4], target_shots=range(1, 3))
    with pytest.raises(ValueError):
        reference_readers("bidirectional_v1", annotation["shots"], reference_shots=[0], target_shots=[1])


def _shot(names, place="Location 1", size="medium", angle="eye_level"):
    return {"present": [{"name": n} for n in names], "place": place, "camera": {"shot_size": size, "angle": angle}}


def test_distinct_camera_first_prefers_other_setups_and_keeps_one_same_setup_only_as_fallback():
    from mspc.data.reference_routing import distinct_camera_first
    shots = [
        _shot(["Character 1"]),                       # 0 同机位、共有人物 1
        _shot(["Character 1"], size="close_up"),      # 1 异机位、共有人物 1
        _shot(["Character 2"]),                       # 2 同机位、共有人物 2（人物 2 没有异机位候选）
        _shot(["Character 9"]),                       # 3 无关镜
        _shot(["Character 1"]),                       # 4 目标镜：人物 1 中景
        _shot(["Character 2"]),                       # 5 目标镜：人物 2 中景
    ]
    chosen = distinct_camera_first(shots, candidates=[0, 1, 2, 3], target_shots=[4, 5], count=4, limit=10,
                                   same_camera_fallback=True, fill_unrelated=True)
    assert 1 in chosen and 2 in chosen and 0 not in chosen   # 人物 1 有异机位就不给同机位；人物 2 只能兜底
    assert 3 in chosen                                       # 空位用无关镜补回张数
    assert chosen == sorted(chosen)


def test_distinct_camera_only_never_gives_a_same_setup_reference_and_fills_with_unrelated_shots():
    from mspc.data.reference_routing import distinct_camera_first
    shots = [
        _shot(["Character 1"]),                       # 0 同机位、共有人物 1
        _shot(["Character 1"], size="close_up"),      # 1 异机位、共有人物 1
        _shot(["Character 2"]),                       # 2 同机位、共有人物 2（兜底时会被补进来）
        _shot(["Character 9"]),                       # 3 无关镜
        _shot(["Character 8"]),                       # 4 无关镜
        _shot(["Character 1"]),                       # 5 目标镜
        _shot(["Character 2"]),                       # 6 目标镜
    ]
    chosen = distinct_camera_first(shots, candidates=[0, 1, 2, 3, 4], target_shots=[5, 6], count=3, limit=10,
                                   same_camera_fallback=False, fill_unrelated=True)
    assert chosen == [1, 3, 4]                               # 不给同机位，名额用无关镜补回


def test_distinct_camera_shared_only_drops_same_setup_and_unrelated_to_empty():
    from mspc.data.reference_routing import distinct_camera_first
    shots = [
        _shot(["Character 1"]),                       # 0 同机位、共有人物 1
        _shot(["Character 1"], size="close_up"),      # 1 异机位、共有人物 1
        _shot(["Character 2"]),                       # 2 同机位、共有人物 2
        _shot(["Character 9"]),                       # 3 无关镜
        _shot(["Character 1"]),                       # 4 目标镜
        _shot(["Character 2"]),                       # 5 目标镜
    ]
    assert distinct_camera_first(shots, candidates=[0, 1, 2, 3], target_shots=[4, 5], count=4, limit=10,
                                 same_camera_fallback=False, fill_unrelated=False) == [1]
    assert distinct_camera_first(shots, candidates=[0, 2, 3], target_shots=[4, 5], count=4, limit=10,
                                 same_camera_fallback=False, fill_unrelated=False) == []
    with pytest.raises(TypeError, match="explicit booleans"):
        distinct_camera_first(shots, candidates=[0, 1], target_shots=[4], count=2, limit=10,
                              same_camera_fallback=False, fill_unrelated=1)


def test_camera_setup_reads_a_missing_angle_as_eye_level():
    from mspc.data.reference_routing import camera_setup
    written = _shot(["Character 1"], angle="eye-level shot")
    omitted = {**written, "camera": {"shot_size": "medium"}}
    assert camera_setup(omitted) == camera_setup(written) != camera_setup(_shot(["Character 1"], angle="high angle shot"))
    assert camera_setup({**written, "camera": {"angle": "eye-level shot"}}) is None


def test_distinct_camera_first_treats_unknown_setups_as_same_and_respects_the_limit():
    from mspc.data.reference_routing import camera_setup, distinct_camera_first
    unknown = {"present": [{"name": "Character 1"}], "place": None, "camera": {}}
    assert camera_setup(unknown) is None
    shots = [unknown] + [_shot(["Character 1"], angle=f"a{i}") for i in range(12)] + [_shot(["Character 1"])]
    target = len(shots) - 1
    chosen = distinct_camera_first(shots, candidates=list(range(target)), target_shots=[target], count=12, limit=10,
                                   same_camera_fallback=True, fill_unrelated=True)
    assert len(chosen) == 10 and 0 not in chosen
