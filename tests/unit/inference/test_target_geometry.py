"""实际目标布局的同帧几何编译；不把晚锚或未知前后提前到镜首。"""
import threading
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mspc.inference.planner_token_reading import person_numbers, token_condition
from mspc.inference.reference_pointing import plan_geometry, plan_starts
from mspc.models.planner_tokens import geometry_token_features
from tests.support.paths import REPO_ROOT


def test_real_target_plan_geometry_and_points_share_frame_zero():
    fixture = json.loads((REPO_ROOT / "tests/fixtures/block_annotation_v18/layout-token-overflow.json").read_text())
    board = deepcopy(fixture["cases"][2]["annotation"])
    layout = board["layouts"][0]
    layout["front_of"] = [{"near": "Character 1", "far": "Character 2"},
                          {"near": "Character 1", "far": "Character 4"}]
    span = (0, len(board["shots"]))
    starts, rows = plan_starts(board, span), plan_geometry(board, span)
    row = rows[layout["shot"] - 1]
    entities = {entity["name"]: entity for entity in row["characters"]}
    assert entities["Character 1"]["size_valid"]
    assert entities["Character 1"]["size"] == pytest.approx([0.36, 0.82])
    assert entities["Character 4"] == {
        "name": "Character 4", "point_2d": None, "size": None, "size_valid": False}
    assert row["front_of"] == [{"near": "Character 1", "far": "Character 2"}]
    assert all(record["frame"] == 0 for record in rows)
    packed = geometry_token_features(starts, rows, 64)
    condition = token_condition(starts, person_numbers(starts), span=span,
                                latent_lengths=[2] * len(board["shots"]), geometry_rows=rows, max_people=64)
    points = token_condition(starts, person_numbers(starts), span=span, latent_lengths=[2] * len(board["shots"]))
    assert condition.shot_tokens == points.shot_tokens
    assert torch.equal(condition.geometry, packed)
    assert layout["entities"][2]["anchors"][0]["time"] == 0.49  # 原晚锚仍在完整计划中


def test_missing_opening_geometry_preserves_unknowns_and_no_token():
    board = {"characters": [{"name": "Character 1", "kind": "person"}],
             "shots": [{"present": [{"name": "Character 1"}]}],
             "layouts": [{"shot": 1, "entities": [{"name": "Character 1", "anchors": [
                 {"time": 0.37, "visibility": "visible", "box": [0.79, 0.88, 0.88, 0.96]}]}]}]}
    starts, rows = plan_starts(board, (0, 1)), plan_geometry(board, (0, 1))
    assert rows[0]["characters"][0]["size"] is None
    assert rows[0]["front_of"] == []
    assert token_condition(starts, person_numbers(starts), span=(0, 1), latent_lengths=[2],
                           geometry_rows=rows, max_people=64) is None


def test_group_keeps_the_same_target_point_and_size_but_has_no_known_front_of():
    fixture = json.loads((REPO_ROOT / "tests/fixtures/block_annotation_v18/layout-token-overflow.json").read_text())
    board = deepcopy(fixture["cases"][2]["annotation"])
    next(entity for entity in board["characters"] if entity["name"] == "Character 2")["kind"] = "group"
    board["layouts"][0]["front_of"] = [{"near": "Character 1", "far": "Character 2"}]
    span = (0, len(board["shots"]))
    starts, rows = plan_starts(board, span), plan_geometry(board, span)
    shot = board["layouts"][0]["shot"] - 1
    row = rows[shot]
    group = next(entity for entity in row["characters"] if entity["name"] == "Character 2")
    assert group["point_2d"] is not None and group["size_valid"] and group["size"] is not None
    assert row["front_of"] == []
    geometry = token_condition(starts, person_numbers(starts), span=span, latent_lengths=[2] * len(board["shots"]),
                               geometry_rows=rows, max_people=64)
    points = token_condition(starts, person_numbers(starts), span=span, latent_lengths=[2] * len(board["shots"]))
    assert geometry.shot_tokens == points.shot_tokens
    group_index = next(i for i, (target, name, _) in enumerate(starts) if target == shot and name == "Character 2")
    assert torch.count_nonzero(geometry.geometry[group_index, 3:]) == 0


def test_geometry_requires_its_explicit_person_capacity():
    with pytest.raises(ValueError, match="supplied together"):
        token_condition([], torch.empty(0, dtype=torch.long), span=(0, 1), latent_lengths=[2], geometry_rows=[])


def _generate_real_geometry_story(output, edits):
    """真实十镜布局与完整分段入口；CPU runtime double只捕获接线，不生成或判视觉质量。"""
    from mspc.comparison_methods.prompts import adapt_storyboard
    from mspc.inference.segment_story import generate_story
    from mspc.benchmark.layout_baseline import execution_storyboard
    # 旧合同的真实规划记录：执行计划去掉参考镜号（由在场名单决定），人物框照留。
    board = execution_storyboard(json.loads(
        (REPO_ROOT / "tests/fixtures/planner/layout_planner_record_two_anchors_real.json").read_text())["storyboard"])
    runtime = SimpleNamespace(lock=threading.Lock(), model_cfg={"self_attention_policy": "full",
        "memory_prefix": {"row_policy": "frame_entities_v1"},
        "planner_tokens": {"content": "person_geometry", "max_people": 64}})
    calls = []

    def generate(**kw):
        calls.append(kw)
        runtime.last_generated_shot_videos = [[np.asarray([[[len(calls), shot, frame]]], dtype=np.uint8)
            for frame in range(count)] for shot, count in enumerate(kw["shot_frame_counts"])]
        runtime.last_generated_shot_frames = [{"middle": frames[len(frames) // 2]}
                                              for frames in runtime.last_generated_shot_videos]
        return {}

    runtime.generate_case = generate
    runtime.encode_single_frame = lambda image: torch.full((1, 1, 1, 1, 1), float(image[0, 0, 0]))
    segments = adapt_storyboard(board, family="wan_segment",
        limits={"max_shots": 4, "max_frames": 228, "max_tokens": 512}, count_tokens=lambda _: 1)
    records = generate_story(runtime, annotation=board, shot_ids=[f"s{i + 1:02d}" for i in range(len(board["shots"]))],
        segments=segments, output=output, token_counter=lambda _: 1,
        frames_max=10,
        seed=42, segment_seed_stride=100, on_segment=lambda _: None,
        history_policy="recent_shots", token_edits=edits)
    return calls, records


@pytest.mark.parametrize("kind", ["move", "remove", "swap"])
def test_generate_story_applies_geometry_token_edits_without_changing_other_inputs(tmp_path, kind):
    from mspc.inference.planner_token_reading import TokenEdit
    from mspc.models.planner_tokens import point_cell
    base_calls, base_records = _generate_real_geometry_story(tmp_path / "base", None)
    edit = TokenEdit(kind=kind, names=("Character 1", "Character 2") if kind == "swap" else ("Character 1",),
                     shot=1, point=(200, 300) if kind == "move" else None)
    calls, records = _generate_real_geometry_story(tmp_path / kind, {1: (edit,)})
    assert len(calls) == len(base_calls) == 6
    assert base_calls[1]["memory"] is not None  # 编辑落在有自产历史的第二段
    before, after = base_calls[1]["planner_tokens"], calls[1]["planner_tokens"]
    assert before.features.tolist() == [2, 1]
    if kind == "move":
        assert torch.equal(after.features, before.features) and torch.equal(after.geometry, before.geometry)
        assert after.shot_tokens[0] == before.shot_tokens[0]
        assert after.shot_tokens[1][:3] == before.shot_tokens[1][:3]
        assert after.shot_tokens[1][3:] == point_cell([200, 300])
        assert records[1]["planner_tokens"]["geometry"][0]["characters"][1]["point_2d"] == [200, 300]
    elif kind == "remove":
        assert torch.equal(after.features, before.features) and torch.equal(after.geometry, before.geometry)
        assert after.shot_tokens == (before.shot_tokens[0],)
        assert after.features.shape == before.features.shape == (2,)  # 不删全行内容，只不放该token
        assert records[1]["planner_tokens"]["geometry"][0]["characters"][1]["point_2d"] is None
    else:
        assert after.features.tolist() == [1, 2]
        assert after.shot_tokens == before.shot_tokens  # 坐标与RoPE不换位
        assert torch.equal(after.geometry, before.geometry[[1, 0]])  # ID与完整几何内容同步换
        assert after.geometry[0, 3 + 1] == 1 and after.geometry[1, 3 + 0] == -1  # 关系列不换
    for index, (base, edited) in enumerate(zip(base_calls, calls)):
        assert edited["prompts"] == base["prompts"] and edited["seed"] == base["seed"]
        assert edited["shot_frame_counts"] == base["shot_frame_counts"]
        if base["memory"] is not None:
            assert torch.equal(edited["memory"].latents, base["memory"].latents)
            assert edited["memory"].shot_texts == base["memory"].shot_texts
            assert edited["memory"].reference_routing == base["memory"].reference_routing
        if index != 1 and base["planner_tokens"] is not None:
            assert edited["planner_tokens"].shot_tokens == base["planner_tokens"].shot_tokens
            assert torch.equal(edited["planner_tokens"].geometry, base["planner_tokens"].geometry)
    before_rows, after_rows = base_records[1]["planner_tokens"]["geometry"], records[1]["planner_tokens"]["geometry"]
    assert before_rows[0]["front_of"] == after_rows[0]["front_of"]
    assert [entity["size"] for entity in before_rows[0]["characters"]] == [
        entity["size"] for entity in after_rows[0]["characters"]]
