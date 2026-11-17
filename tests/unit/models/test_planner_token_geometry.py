"""第五组的几何内容：同ID／token数／RoPE，已知有向关系与未知分离，零初始化与可训练性。"""
from copy import deepcopy

import pytest
import torch

from mspc.models.planner_tokens import (
    PLANNER_CONNECTOR, PersonEmbedding, PersonGeometryEmbedding, PlannerTokenCondition, PlannerTokenOptions,
    attach_planner_connector, geometry_token_features, planner_token_states,
)
from tests.unit.models.test_memory_prefix_attention import _tiny_dit
from tests.unit.models.test_planner_tokens import H, MEMORY, OFFSET, TARGET, W, _groups, _latents
from tests.support.memory_prefix import full_window


def _starts_and_rows():
    starts = [(2, "Character 1", [100, 200]), (2, "Character 2", [900, 200]),
              (2, "Character 3", None), (3, "Character 1", [500, 500])]
    rows = [
        {"shot_index": 2, "frame": 0, "characters": [
            {"name": "Character 1", "point_2d": [100, 200], "size": [0.2, 0.4], "size_valid": True},
            {"name": "Character 2", "point_2d": [900, 200], "size": [0.1, 0.3], "size_valid": True},
            {"name": "Character 3", "point_2d": None, "size": None, "size_valid": False}],
         "front_of": [{"near": "Character 1", "far": "Character 2"}]},
        {"shot_index": 3, "frame": 0, "characters": [
            {"name": "Character 1", "point_2d": [500, 500], "size": [0.3, 0.5], "size_valid": True}],
         "front_of": []}]
    return starts, rows


def test_geometry_uses_signed_person_slots_and_explicit_unknowns_without_closure():
    starts, rows = _starts_and_rows()
    rows[0]["front_of"].append({"near": "Character 2", "far": "Character 3"})
    packed = geometry_token_features(starts, rows, 64)
    assert packed.shape == (4, 131) and packed.dtype == torch.float32
    assert torch.equal(packed[0, :3], torch.tensor([0.2, 0.4, 1.0]))
    assert packed[0, 3 + 1] == 1 and packed[1, 3 + 0] == -1
    assert packed[0, 3 + 64 + 1] == packed[1, 3 + 64 + 0] == 1
    assert packed[1, 3 + 2] == 1 and packed[2, 3 + 1] == -1
    assert packed[0, 3 + 2] == packed[0, 3 + 64 + 2] == 0  # 不补1→3传递边
    assert torch.count_nonzero(packed[2, :3]) == 0  # 无尺寸不伪造为小人物
    assert torch.count_nonzero(packed[3, 3:]) == 0  # 关系不跨镜


@pytest.mark.parametrize("change, message", [
    ("late_frame", "frame-zero"), ("point_mismatch", "same frame-zero"),
    ("identity_order", "identity and row order"), ("contradictory_pair", "consistent pairs"),
    ("unknown_size_value", "explicitly known or null"),
])
def test_geometry_rejects_changed_time_identity_or_unknown_semantics(change, message):
    starts, rows = _starts_and_rows()
    if change == "late_frame":
        rows[0]["frame"] = 4
    elif change == "point_mismatch":
        rows[0]["characters"][0]["point_2d"] = [200, 200]
    elif change == "identity_order":
        rows[0]["characters"].reverse()
    elif change == "contradictory_pair":
        rows[0]["front_of"].append({"near": "Character 2", "far": "Character 1"})
    else:
        rows[0]["characters"][2]["size"] = [0.1, 0.1]
    with pytest.raises(ValueError, match=message):
        geometry_token_features(starts, rows, 64)


def test_empty_people_shots_are_explicit_and_make_no_geometry_tokens():
    assert geometry_token_features([], [{"shot_index": 0, "frame": 0, "characters": [], "front_of": []}], 64).shape == (0, 131)


def test_geometry_initialization_preserves_rng_and_matches_the_same_id_embedding():
    starts, rows = _starts_and_rows()
    geometry = geometry_token_features(starts, rows, 8)
    torch.manual_seed(42)
    expected = torch.randn(32)
    torch.manual_seed(42)
    richer = PersonGeometryEmbedding(8, 24)
    assert torch.equal(torch.randn(32), expected)
    points = PersonEmbedding(8, 24)
    with torch.no_grad():
        points.table.weight.normal_()
        richer.table.weight.copy_(points.table.weight)
    people = torch.tensor([1, 2, 3, 1])
    assert torch.equal(richer(people, geometry), points(people))
    assert torch.count_nonzero(richer.geometry_projection.weight) == 0
    richer(people, geometry).square().mean().backward()
    assert torch.isfinite(richer.geometry_projection.weight.grad).all()
    assert torch.count_nonzero(richer.geometry_projection.weight.grad) > 0
    assert torch.isfinite(richer.table.weight.grad).all() and torch.count_nonzero(richer.table.weight.grad) > 0


def test_geometry_has_an_explicit_model_content_and_requires_its_tensor():
    options = PlannerTokenOptions.from_mapping({"content": "person_geometry", "max_people": 64, "learning_rate": 1e-4})
    assert options.max_people == 64
    connector = PersonGeometryEmbedding(64, 24)
    condition = PlannerTokenCondition(features=torch.tensor([1]), shot_tokens=((0, 0, 0, 1, 2),))
    with pytest.raises(ValueError, match="requires explicit geometry"):
        planner_token_states(connector, condition, torch.float32)
    malformed = torch.zeros(1, 131)
    malformed[0, 3 + 64 + 1] = 1  # 已知关系不能用0冒充tie
    with pytest.raises(ValueError, match="signed known relations"):
        connector(torch.tensor([1]), malformed)


def _condition(with_geometry):
    from mspc.inference.planner_token_reading import person_numbers, token_condition
    starts, rows = _starts_and_rows()
    return token_condition(starts, person_numbers(starts), span=(2, 4), latent_lengths=(2, 1),
                           **({"geometry_rows": rows, "max_people": 8} if with_geometry else {}))


def _forward(dit, condition, *, checkpoint=False, source=None):
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    return model_fn_wan_video(
        dit=dit, latents=_latents(), timestep=torch.tensor([500.0]),
        context=torch.randn(4, 16, 4096, generator=torch.Generator().manual_seed(7)),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1", latent_shot_groups=_groups(),
        fuse_vae_embedding_in_latents=True, prefix_frames=MEMORY, reference_offset=OFFSET,
        memory_attention_policy="bidirectional_v1", memory_attention_window=full_window(),
        planner_tokens=condition, source_identity_condition=source,
        use_gradient_checkpointing=checkpoint, use_gradient_checkpointing_offload=False)


def test_zero_projection_keeps_point_token_count_positions_and_model_output_exact():
    point_dit = _tiny_dit()
    attach_planner_connector(point_dit, PlannerTokenOptions(content="person_index", max_people=8, learning_rate=1e-4))
    geometry_dit = _tiny_dit()
    attach_planner_connector(geometry_dit, PlannerTokenOptions(content="person_geometry", max_people=8, learning_rate=1e-4))
    missing, unexpected = geometry_dit.load_state_dict(point_dit.state_dict(), strict=False)
    assert missing == ["planner_connector.geometry_projection.weight", "planner_connector.geometry_projection.bias"]
    assert unexpected == []
    point, richer = _condition(False), _condition(True)
    assert point.count == richer.count == 3 and point.shot_tokens == richer.shot_tokens
    with torch.no_grad():
        assert torch.equal(_forward(point_dit, point).noise, _forward(geometry_dit, richer).noise)


def test_joint_source_geometry_checkpoint_backward_and_reload():
    from mspc.models.source_identity import SourceIdentityCondition, SourceIdentityRow
    dit = _tiny_dit().train()
    options = PlannerTokenOptions(content="person_geometry", max_people=8, learning_rate=1e-4)
    attach_planner_connector(dit, options)
    row = SourceIdentityRow(entity_names=("Character 1", "Character 2"), entity_tokens=((1,), (2,)),
                            common_tokens=(0, 3), points=((0, 1, 2), (1, 3, 4)),
                            alignment_cells=((0, 1, 3), (1, 3, 5)))
    source = SourceIdentityCondition(rows=(row, row), alignment_layer=0)
    condition = _condition(True)
    prediction = _forward(dit, condition, checkpoint=True, source=source)
    assert prediction.noise.shape == (1, 48, MEMORY + TARGET, H, W)
    assert prediction.source_alignment_loss is not None
    loss = prediction.noise[:, :, MEMORY:].float().square().mean() + 0.01 * prediction.source_alignment_loss
    loss.backward()
    connector = getattr(dit, PLANNER_CONNECTOR)
    for parameter in connector.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(connector.geometry_projection.weight.grad) > 0
    assert torch.count_nonzero(connector.table.weight.grad) > 0
    with torch.no_grad():
        connector.geometry_projection.weight.add_(connector.geometry_projection.weight.grad, alpha=-1e-4)
    restored = _tiny_dit().train()
    attach_planner_connector(restored, options)
    restored.load_state_dict(deepcopy(dit.state_dict()))
    with torch.no_grad():
        assert torch.equal(_forward(dit, condition, source=source).noise,
                           _forward(restored, condition, source=source).noise)
