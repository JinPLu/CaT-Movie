"""目标点 token 与干净参考共用全双向视频注意力；点只读自己，视频读取点。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tests.support.memory_prefix import INFERENCE_OFFSET, full_window

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from mspc.models.planner_tokens import (  # noqa: E402
    PLANNER_CONNECTOR, PlannerTokenCondition, PlannerTokenLayout, PlannerTokenOptions, attach_planner_connector,
    box_point, planner_self_attention, planner_token_freqs, point_cell, token_placement,
)
from tests.unit.models.test_memory_prefix_attention import TINY_DIT_KWARGS, _tiny_dit  # noqa: E402
from diffsynth.pipelines.wan_video import TIME_MLP_ROWS  # noqa: E402

MEMORY, TARGET, H, W = 2, 3, 30, 52  # 潜变量 30×52 → 15×26 格，与 832×480 出片相同
OFFSET = (-5, 0, 0)
OPTIONS = PlannerTokenOptions(content="planner_features", feature_dim=12, feature_layer=19, learning_rate=1e-4)


def _groups():
    return ((0, 1), (1, 2), (2, 4), (4, 5))


def _forward(dit, latents, *, planner_tokens=None, seed=1):
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    torch.manual_seed(seed)
    with torch.no_grad():
        return model_fn_wan_video(
            dit=dit, latents=latents[:, :, MEMORY:], timestep=torch.tensor([500.0]),
            context=torch.randn(2, 16, 4096),
            temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
            latent_shot_groups=((0, 2), (2, 3)),
            fuse_vae_embedding_in_latents=True, prefix_frames=0,
            planner_tokens=planner_tokens,
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False).noise


def _latents(seed=0):
    return torch.randn(1, 48, MEMORY + TARGET, H, W, generator=torch.Generator().manual_seed(seed))


def _condition(features=None):
    features = torch.randn(2, OPTIONS.feature_dim, generator=torch.Generator().manual_seed(3)) if features is None else features
    # 目标两镜：第 0 镜从目标第 0 个潜帧起，第 1 镜从第 2 个起。
    return PlannerTokenCondition(features=features, shot_tokens=((0, 0, 0, 1, 2), (1, 1, 2, 3, 0)))


def test_point_cell_uses_the_15_by_26_grid():
    assert point_cell([0, 0]) == (0, 0)
    assert point_cell([999, 999]) == (14, 25)
    assert point_cell([1000, 1000]) == (14, 25)
    assert point_cell([500, 500]) == (7, 13)
    with pytest.raises(ValueError):
        point_cell([1001, 0])


def test_box_center_matches_the_label_definition():
    import numpy as np
    from mspc.data.history_reading_labels import box_center

    mask = np.zeros((480, 832), bool)
    mask[96:288, 208:624] = True
    box = [208 / 832, 96 / 480, 624 / 832, 288 / 480]
    assert box_point(box) == box_center(mask) == [500, 400]
    with pytest.raises(ValueError):
        box_point([0.5, 0.2, 0.4, 0.3])


def test_placement_skips_null_and_dropped_points():
    lookup = {(3, "A"): 0, (3, "B"): 1, (4, "A"): 2}
    placed = token_placement(starts=[(3, "A", [100, 100]), (3, "B", None), (4, "A", [900, 900])], span_start=3,
                             latent_starts=[0, 5], shot_lookup=lookup,
                             move_point=lambda shot, name, point: None if shot == 4 else point)
    assert placed == {"shot_tokens": [(0, 0, 0, 1, 2)]}
    assert token_placement(starts=[(3, "B", None)], span_start=3, latent_starts=[0], shot_lookup=lookup) is None


def test_options_require_every_field():
    with pytest.raises(ValueError):
        PlannerTokenOptions.from_mapping({"content": "planner_features", "feature_dim": 4096, "feature_layer": 19})
    assert PlannerTokenOptions.from_mapping({"content": "planner_features", "feature_dim": 4096, "feature_layer": 19,
                                             "learning_rate": 1e-4}).feature_layer == 19
    with pytest.raises(ValueError):
        PlannerTokenOptions.from_mapping({"content": "person_index", "max_people": 32, "feature_dim": 4096,
                                          "learning_rate": 1e-4})
    assert PlannerTokenOptions.from_mapping({"content": "person_index", "max_people": 32,
                                             "learning_rate": 1e-4}).max_people == 32


def test_rope_puts_tokens_on_their_video_cell_with_the_shot_phase():
    from diffsynth.pipelines.wan_video import memory_prefix_position_freqs
    from diffsynth.models.wan_video_dit import shot_phase_adjusted_freqs

    dit = _tiny_dit()
    h, w = H // 2, W // 2
    tokens_per_frame = h * w
    frames = MEMORY + TARGET
    base = memory_prefix_position_freqs(dit, frames, h, w, MEMORY, OFFSET)
    video = shot_phase_adjusted_freqs(base, _groups(), frames, tokens_per_frame, shot_phase_index=(0, 0, 0, 1))
    freqs = planner_token_freqs(dit, _condition(), video_freqs=video, latent_frames=frames,
                                tokens_per_frame=tokens_per_frame, prefix_frames=MEMORY, reference_offset=OFFSET,
                                shot_phase=True)

    def video_index(latent_frame, row, col):
        return latent_frame * tokens_per_frame + row * w + col

    # 目标第 0 镜（全局潜帧 2）的 (1,2)；目标第 1 镜（全局潜帧 4）的 (3,0)，带镜序相位。
    assert freqs.shape[0] == 2
    assert torch.allclose(freqs[0, 0], video[video_index(2, 1, 2), 0])
    assert torch.allclose(freqs[1, 0], video[video_index(4, 3, 0), 0])


def test_visibility_of_tokens_references_and_targets():
    layout = PlannerTokenLayout(video_tokens=5 * 4, shot_token_shots=(0, 1))
    total = 5 * 4 + 2

    def attn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q, k, v = (torch.randn(1, total, 6, generator=torch.Generator().manual_seed(s)) for s in (1, 2, 3))
    out = planner_self_attention(attn, q, k, v, layout=layout)
    assert torch.equal(out[:, 20:22], v[:, 20:22])  # token 只读自己
    changed_v = v.clone()
    changed_v[:, 4:8] += 1.0  # 改第 1 张参考帧
    again = planner_self_attention(attn, q, k, changed_v, layout=layout)
    assert not torch.allclose(again[:, 0:4], out[:, 0:4])  # 参考帧彼此互读
    assert not torch.allclose(again[:, 4:8], out[:, 4:8])
    assert not torch.allclose(again[:, 8:20], out[:, 8:20])
    target_changed = v.clone()
    target_changed[:, 8:20] += 1.0
    targets_read_back = planner_self_attention(attn, q, k, target_changed, layout=layout)
    assert not torch.allclose(targets_read_back[:, 0:8], out[:, 0:8])  # 参考也读目标
    token_changed = v.clone()
    token_changed[:, 20] += 1.0
    read = planner_self_attention(attn, q, k, token_changed, layout=layout)
    assert not torch.allclose(read[:, 0:8], out[:, 0:8])
    assert not torch.allclose(read[:, 8:20], out[:, 8:20])
    assert torch.equal(read[:, 21], out[:, 21])  # 点不读另一个点


def test_planner_attention_matches_dense_visibility_outputs_and_gradients():
    layout = PlannerTokenLayout(video_tokens=5, shot_token_shots=(0, 1))
    q, k, v = (torch.randn(1, 7, 6, dtype=torch.float64, requires_grad=True) for _ in range(3))
    attn = torch.nn.functional.scaled_dot_product_attention
    actual = planner_self_attention(attn, q, k, v, layout=layout)
    allowed = torch.zeros(7, 7, dtype=torch.bool)
    allowed[:5] = True
    allowed[5, 5] = allowed[6, 6] = True
    expected = attn(q, k, v, attn_mask=allowed)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    for got, want in zip(actual_grads, expected_grads):
        assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(actual_grads[0][:, 5:]) == 0
    assert torch.count_nonzero(actual_grads[1][:, 5:]) > 0  # 点key经视频query获得梯度


def test_without_tokens_the_forward_is_unchanged_bit_for_bit():
    dit = _tiny_dit()
    before = _forward(dit, _latents())
    attach_planner_connector(dit, OPTIONS)
    after = _forward(dit, _latents())
    assert torch.equal(before, after)


def test_targets_read_tokens_once_the_connector_is_nonzero():
    dit = _tiny_dit()
    attach_planner_connector(dit, OPTIONS)
    connector = getattr(dit, PLANNER_CONNECTOR)
    zero_a = _forward(dit, _latents(), planner_tokens=_condition())
    zero_b = _forward(dit, _latents(), planner_tokens=_condition(torch.randn(2, OPTIONS.feature_dim)))
    assert torch.equal(zero_a, zero_b)  # 末层零初始化：特征不同、输出相同
    torch.nn.init.normal_(connector.outer.weight)
    a = _forward(dit, _latents(), planner_tokens=_condition())
    b = _forward(dit, _latents(), planner_tokens=_condition(torch.randn(2, OPTIONS.feature_dim)))
    assert a.shape == zero_a.shape and not torch.allclose(a[:, :, MEMORY:], b[:, :, MEMORY:])


def test_warm_start_from_a_renderer_without_the_connector():
    from mspc.training.dit_parameter_freeze import load_dit_state_dict

    state = {k: v.clone() for k, v in _tiny_dit().state_dict().items()}
    target = _tiny_dit()
    attach_planner_connector(target, OPTIONS)
    assert load_dit_state_dict(target, state) == "warm_start_plus_new_channel"
    assert torch.count_nonzero(getattr(target, PLANNER_CONNECTOR).outer.weight) == 0
    assert load_dit_state_dict(target, {k: v.clone() for k, v in target.state_dict().items()}) == "full"


def test_planner_tokens_refuse_the_first_frame_interface():
    dit = _tiny_dit()
    attach_planner_connector(dit, OPTIONS)
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "first_frame_interface"
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    with pytest.raises(ValueError, match="full self-attention"), torch.no_grad():
        model_fn_wan_video(
            dit=dit, latents=_latents()[:, :, MEMORY:], timestep=torch.tensor([500.0]), context=torch.randn(2, 16, 4096),
            temporal_position_mode="global_time_native_rope_v1", latent_shot_groups=((0, 2), (2, 3)),
            fuse_vae_embedding_in_latents=True, prefix_frames=0,
            planner_tokens=_condition(), use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)


def test_planner_tokens_need_the_15_by_26_grid():
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    dit = _tiny_dit()
    attach_planner_connector(dit, OPTIONS)
    latents = torch.randn(1, 48, MEMORY + TARGET, 8, 8)
    with pytest.raises(ValueError, match="15x26"):
        model_fn_wan_video(
            dit=dit, latents=latents, timestep=torch.tensor([500.0]), context=torch.randn(4, 16, 4096),
            temporal_position_mode="global_time_msm_fixed_shot_phase_v1", latent_shot_groups=_groups(),
            fuse_vae_embedding_in_latents=True, prefix_frames=MEMORY, reference_offset=OFFSET,
            memory_attention_policy="bidirectional_v1", memory_attention_window=full_window(),
            planner_tokens=_condition(),
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)


def test_cross_attention_keeps_its_backend_when_tokens_are_present():
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    dit = _tiny_dit()
    attach_planner_connector(dit, OPTIONS)
    seen = []
    dit.blocks[0].cross_attn.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((args[0].shape[1], kwargs.get("text_attention_backend"))),
        with_kwargs=True)
    torch.manual_seed(1)
    with torch.no_grad():
        model_fn_wan_video(
            dit=dit, latents=_latents()[:, :, MEMORY:], timestep=torch.tensor([500.0]), context=torch.randn(2, 16, 4096),
            temporal_position_mode="global_time_msm_fixed_shot_phase_v1", latent_shot_groups=((0, 2), (2, 3)),
            fuse_vae_embedding_in_latents=True, prefix_frames=0,
            planner_tokens=_condition(), text_attention_backend="sdpa",
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    # 只有视频 token 进交叉注意力，且显式后端传到了。
    assert seen == [(TARGET * (H // 2) * (W // 2), "sdpa")]


def test_person_marker_tokens_start_at_zero_and_learn_per_person():
    from mspc.models.planner_tokens import PersonEmbedding

    dit = _tiny_dit()
    attach_planner_connector(dit, PlannerTokenOptions(content="person_index", max_people=8, learning_rate=1e-4))
    table = getattr(dit, PLANNER_CONNECTOR)
    assert isinstance(table, PersonEmbedding) and torch.count_nonzero(table.table.weight) == 0
    people = PlannerTokenCondition(features=torch.tensor([1, 2]), shot_tokens=((0, 0, 0, 1, 2), (1, 1, 2, 3, 0)))
    zero = _forward(dit, _latents(), planner_tokens=people)
    torch.nn.init.normal_(table.table.weight)
    swapped = PlannerTokenCondition(features=torch.tensor([2, 1]), shot_tokens=people.shot_tokens)
    a = _forward(dit, _latents(), planner_tokens=people)
    b = _forward(dit, _latents(), planner_tokens=swapped)
    assert zero.shape == a.shape and not torch.allclose(a[:, :, MEMORY:], b[:, :, MEMORY:])
    with pytest.raises(ValueError, match="1..8"):
        table(torch.tensor([9]))


@pytest.mark.parametrize("source_points", [False, True])
@pytest.mark.parametrize("target_points", [False, True])
def test_source_and_target_point_interfaces_share_the_reference_prefix(source_points, target_points):
    dit = _tiny_dit()
    attach_planner_connector(dit, PlannerTokenOptions(content="person_index", max_people=8, learning_rate=1e-4))
    condition = PlannerTokenCondition(features=torch.tensor([1, 2]), shot_tokens=_condition().shot_tokens)
    source = None
    if source_points:
        from mspc.models.source_identity import SourceIdentityCondition, SourceIdentityRow
        row = SourceIdentityRow(entity_names=("Character 1",), entity_tokens=((1, 2),),
                                common_tokens=(0, 3), points=((0, 1, 2),), alignment_cells=None)
        source = SourceIdentityCondition(rows=(row, row), alignment_layer=0)
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    with torch.no_grad():
        prediction = model_fn_wan_video(
            dit=dit, latents=_latents(), timestep=torch.tensor([500.0]), context=torch.randn(4, 16, 4096),
            temporal_position_mode="global_time_msm_fixed_shot_phase_v1", latent_shot_groups=_groups(),
            fuse_vae_embedding_in_latents=True, prefix_frames=MEMORY, reference_offset=OFFSET,
            memory_attention_policy="bidirectional_v1", memory_attention_window=full_window(),
            planner_tokens=condition if target_points else None, source_identity_condition=source,
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    assert prediction.noise.shape == _latents().shape
    assert torch.isfinite(prediction.noise).all()


def test_person_embedding_initialization_preserves_the_global_noise_rng():
    from mspc.models.planner_tokens import PersonEmbedding
    torch.manual_seed(42)
    expected_noise = torch.randn(1, 48, 3, 8, 8)
    torch.manual_seed(42)
    PersonEmbedding(max_people=8, dim=24)
    assert torch.equal(torch.randn(1, 48, 3, 8, 8), expected_noise)


def test_target_only_loss_trains_point_embeddings_with_reference_prefix_and_checkpointing():
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    dit = _tiny_dit().train()
    attach_planner_connector(dit, PlannerTokenOptions(content="person_index", max_people=8, learning_rate=1e-4))
    latents = _latents().requires_grad_()
    condition = PlannerTokenCondition(features=torch.tensor([1, 2]), shot_tokens=_condition().shot_tokens)
    prediction = model_fn_wan_video(
        dit=dit, latents=latents, timestep=torch.tensor([500.0]), context=torch.randn(4, 16, 4096),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1", latent_shot_groups=_groups(),
        fuse_vae_embedding_in_latents=True, prefix_frames=MEMORY, reference_offset=OFFSET,
        memory_attention_policy="bidirectional_v1", memory_attention_window=full_window(),
        planner_tokens=condition, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
    prediction.noise[:, :, MEMORY:].float().square().mean().backward()
    gradient = getattr(dit, PLANNER_CONNECTOR).table.weight.grad
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[1:3]) > 0
    assert torch.count_nonzero(gradient[3:]) == 0
    assert torch.isfinite(latents.grad).all() and torch.count_nonzero(latents.grad[:, :, :MEMORY]) > 0


def test_clean_point_timestep_mlp_keeps_fp32_outside_amp():
    dit = _tiny_dit()
    attach_planner_connector(dit, OPTIONS)
    seen = []
    for name in ("time_embedding", "time_projection"):
        getattr(dit, name).register_forward_hook(lambda module, args, output, name=name:
            seen.append((name, args[0].dtype, output.dtype, output.shape[1], torch.is_autocast_enabled("cpu"))))
    _forward(dit, _latents().bfloat16(), planner_tokens=_condition())
    assert seen == [
        ("time_embedding", torch.float32, torch.float32, TARGET * -(-TIME_MLP_ROWS // TARGET), False),
        ("time_projection", torch.float32, torch.float32, TARGET * -(-TIME_MLP_ROWS // TARGET), False),
        ("time_embedding", torch.float32, torch.float32, 2, False),
        ("time_projection", torch.float32, torch.float32, 2, False),
    ]
