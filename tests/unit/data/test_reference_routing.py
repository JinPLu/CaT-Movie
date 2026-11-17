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
