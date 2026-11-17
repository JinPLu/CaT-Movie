import math
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mspc.models.source_identity import (
    SourceIdentityCondition, SourceIdentityRow, WanVideoBlockPrediction, compile_source_identity_condition,
    routed_source_attention, source_alignment_loss,
)


@pytest.mark.parametrize("visible", [("Character 1",), ("Character 1", "Character 2"), ()])
@pytest.mark.parametrize("shot_objects", [(), ("a glass display case",)])
def test_actual_registry_reference_spans_align_for_one_frame_with_multiple_registered_characters(visible, shot_objects):
    from mspc.data.shot_text import reference_text_with_entity_spans
    from tests.unit.models.test_spatial_layout_inputs import ActualFastTokenizer
    annotation = json.loads((Path(__file__).parents[2] /
        "fixtures/block_annotation_v18/film-002869-b0001.json").read_text())["annotation"]
    characters = annotation["characters"]
    record = {"entities": [{"name": name, "point_cell": None} for name in visible], "alignment_cells": None}
    compiled = reference_text_with_entity_spans(record["entities"], setting=None, characters=characters, shot_objects=shot_objects)
    condition = compile_source_identity_condition(ActualFastTokenizer(), [compiled["text"]], [record],
        characters=characters, shot_objects=[shot_objects], settings=[None], alignment_layer=14, max_alignment_cells_per_reference=64)
    row = condition.rows[0]
    assert row.entity_names == visible and len(row.entity_tokens) == len(visible)
    assert all(tokens for tokens in row.entity_tokens)
    assert all(set(a).isdisjoint(b) for i, a in enumerate(row.entity_tokens) for b in row.entity_tokens[i + 1:])
    assert not set(row.common_tokens).intersection(token for ids in row.entity_tokens for token in ids)
    assert {span["source"] for span in compiled["entity_spans"] if span["source"].endswith("appearance")} == {
        f"characters[{i}].appearance" for i, entry in enumerate(characters) if entry["name"] in visible}
    if visible:
        changed = [{**entry, "appearance": characters[1]["appearance"]} if i == 0 else entry
                   for i, entry in enumerate(characters)]
        with pytest.raises(ValueError, match="actual consumed caption"):
            compile_source_identity_condition(ActualFastTokenizer(), [compiled["text"]], [record],
                characters=changed, shot_objects=[shot_objects], settings=[None], alignment_layer=14,
                max_alignment_cells_per_reference=64)


def row(*, labels=True):
    return SourceIdentityRow(("Character 1", "Character 2"), ((0, 1), (3, 4, 5)), (2, 6),
        ((0, 0, 0), (1, 0, 1)), ((0, 1, 1), (1, 1, 2)) if labels else None)


def attention(q, k, v):
    def heads(x):
        return x.reshape(1, x.shape[1], 2, 4).transpose(1, 2)
    out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
    return out.transpose(1, 2).flatten(2)


def test_entity_logsumexp_loss_equals_true_per_head_attention_competition():
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(1, 390, 8, generator=generator, requires_grad=True)
    k = torch.randn(1, 7, 8, generator=generator, requires_grad=True)
    actual = source_alignment_loss(q, k, value_dtype=torch.float32, num_heads=2, row=row())
    cells = [27, 28]
    logits = torch.matmul(q[:, cells].reshape(1, 2, 2, 4).transpose(1, 2),
                          k.reshape(1, 7, 2, 4).transpose(1, 2).transpose(-1, -2)) / math.sqrt(4)
    probabilities = logits.softmax(-1)
    score = torch.stack((probabilities[..., [0, 1]].mean(-1),
                         probabilities[..., [3, 4, 5]].mean(-1)), dim=-1)
    posterior = score / score.sum(-1, keepdim=True)
    expected = -torch.stack((posterior[0, :, 0, 0].log().mean(),
                            posterior[0, :, 1, 1].log().mean())).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert q.grad[:, cells].abs().sum() > 0
    assert torch.count_nonzero(q.grad[:, [0, 1]]) == 0


def test_point_rows_read_only_own_complete_span_and_common_tokens():
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(1, 390, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 7, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 7, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    out = routed_source_attention(attention, q, k, v, num_heads=2, row=row(labels=False))
    expected = attention(q, k, v)
    expected = expected.index_copy(1, torch.tensor([0]), attention(q[:, :1], k[:, [0, 1, 2, 6]], v[:, [0, 1, 2, 6]]))
    expected = expected.index_copy(1, torch.tensor([1]), attention(q[:, 1:2], k[:, [2, 3, 4, 5, 6]], v[:, [2, 3, 4, 5, 6]]))
    torch.testing.assert_close(out.hidden_states, expected)
    assert out.source_alignment_loss is None
    out.hidden_states.square().mean().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


def test_alignment_only_uses_seeded_nonpoint_cells():
    with pytest.raises(ValueError, match="exclude every"):
        SourceIdentityRow(("Character 1",), ((0,),), (), ((0, 0, 0),), ((0, 0, 0),))
    with pytest.raises(ValueError, match="unseeded"):
        SourceIdentityRow(("Character 1", "Character 2"), ((0,), (1,)), (), ((0, 0, 0),), ((1, 1, 0),))
    one = SourceIdentityRow(("Character 1",), ((0,),), (1,), ((0, 0, 0),), ((0, 1, 0),))
    assert source_alignment_loss(torch.ones(1, 390, 8), torch.ones(1, 2, 8),
                                 value_dtype=torch.float32, num_heads=2, row=one) is None


@pytest.mark.parametrize("labels", [False, True])
def test_named_prediction_aux_survives_the_same_gradient_checkpoint(labels):
    generator = torch.Generator().manual_seed(7)
    initial = [torch.randn(1, n, 8, generator=generator) for n in (390, 7, 7)]
    outputs, gradients = [], []
    for use_checkpoint in (False, True):
        q, k, v = [x.clone().requires_grad_() for x in initial]
        def forward(q, k, v):
            result = routed_source_attention(attention, q, k, v, num_heads=2, row=row(labels=labels))
            return WanVideoBlockPrediction(result.hidden_states, result.source_alignment_loss)
        out = checkpoint(forward, q, k, v, use_reentrant=False) if use_checkpoint else forward(q, k, v)
        loss = out.hidden_states.square().mean()
        if out.source_alignment_loss is not None:
            loss = loss + .01 * out.source_alignment_loss
        loss.backward()
        outputs.append(loss.detach())
        gradients.append([x.grad for x in (q, k, v)])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for a, b in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_only_declared_middle_layer_receives_answers_and_last_ca_is_not_routed():
    condition = SourceIdentityCondition((row(),), 14)
    assert condition.for_layer(14, 30).rows[0].alignment_cells is not None
    assert condition.for_layer(13, 30).rows[0].alignment_cells is None
    assert condition.for_layer(13, 30).rows[0].points == row().points
    assert condition.for_layer(29, 30) is None


def cross_attention_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
    from diffsynth.models.wan_video_dit import CrossAttention
    return CrossAttention(8, 2)


def test_actual_wan_cross_attention_propagates_seeded_midlayer_aux_and_flow_gradients():
    module = cross_attention_module()
    x = torch.randn(1, 394, 8, requires_grad=True)
    context = torch.randn(2, 7, 8, requires_grad=True)
    prediction = module(x, context, [(0, 390), (390, 394)],
                        source_identity_condition=SourceIdentityCondition((row(),), 14))
    assert prediction.hidden_states.shape == x.shape
    assert prediction.source_alignment_loss.ndim == 0
    (prediction.hidden_states.square().mean() + .01 * prediction.source_alignment_loss).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(context.grad).all()
    assert module.q.weight.grad.abs().sum() > 0 and module.k.weight.grad.abs().sum() > 0


def test_unknown_source_identity_keeps_actual_reference_ca_unchanged():
    module = cross_attention_module()
    x, context = torch.randn(1, 394, 8), torch.randn(2, 7, 8)
    empty = SourceIdentityRow((), (), tuple(range(7)), (), None)
    baseline = module(x, context, [(0, 390), (390, 394)])
    unknown = module(x, context, [(0, 390), (390, 394)],
                     source_identity_condition=SourceIdentityCondition((empty,), 14))
    torch.testing.assert_close(baseline.hidden_states, unknown.hidden_states, rtol=0, atol=0)
    assert unknown.source_alignment_loss is None


def test_middle_ca_aux_reads_nonpoint_queries_after_point_seed_and_self_attention():
    from dataclasses import replace
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
    from diffsynth.models.wan_video_dit import DiTBlock
    torch.manual_seed(40)
    earlier = DiTBlock(False, 8, 2, 16)
    middle = DiTBlock(False, 8, 2, 16)
    x, context = torch.randn(1, 780, 8), torch.randn(2, 7, 8)
    freqs, time = torch.ones(780, 1, 2, dtype=torch.complex64), torch.zeros(1, 6, 8)
    kwargs = dict(shot_token_ranges=[(0, 390), (390, 780)],
                  latent_shot_groups=[(0, 1), (1, 2)], tokens_per_frame=390,
                  memory_attention_policy="bidirectional_v1", prefix_frames=1)
    nonpoint_queries, losses = [], []
    for seed_row in (row(), replace(row(), points=((0, 0, 1), (1, 0, 0)))):
        condition = SourceIdentityCondition((seed_row,), 14)
        previous = earlier(x, context, time, freqs,
                           source_identity_condition=condition.for_layer(13, 30), **kwargs)
        handle = middle.norm3.register_forward_hook(
            lambda module, inputs, output: nonpoint_queries.append(output[:, [27, 28]].detach()))
        try:
            prediction = middle(previous.hidden_states, context, time, freqs,
                                source_identity_condition=condition.for_layer(14, 30), **kwargs)
        finally:
            handle.remove()
        losses.append(prediction.source_alignment_loss)
    # 相同RGB特征和文字，只交换输入点与ID的对应；实际中层非点Q随之改变。
    assert torch.count_nonzero(nonpoint_queries[0] - nonpoint_queries[1]) > 0
    losses[0].backward()
    assert earlier.cross_attn.v.weight.grad.abs().sum() > 0
    assert earlier.cross_attn.o.weight.grad.abs().sum() > 0


def test_actual_pipeline_training_and_inference_share_noise_and_checkpoint_aux_gradients():
    import copy
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    from tests.support.memory_prefix import full_window
    torch.manual_seed(25)
    model = WanModel(dim=8, in_dim=4, out_dim=4, ffn_dim=16, text_dim=8, freq_dim=8,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=3,
        has_image_input=False, seperated_timestep=True,
        require_clip_embedding=False, require_vae_embedding=False)
    checkpointed = copy.deepcopy(model)
    latent = torch.randn(1, 4, 2, 30, 52)
    context = torch.randn(2, 7, 8)
    condition = SourceIdentityCondition((row(),), 1)
    inputs = dict(latents=latent, context=context, timestep=torch.tensor([500.]),
        temporal_position_mode="global_time_msm_fixed_shot_phase_v1",
        latent_shot_groups=((0, 1), (1, 2)), fuse_vae_embedding_in_latents=True,
        prefix_frames=1, reference_offset=(-5, 0, 0), memory_attention_policy="bidirectional_v1",
        memory_attention_window=full_window())
    training = model_fn_wan_video(dit=model, source_identity_condition=condition, **inputs)
    inference = model_fn_wan_video(dit=model,
        source_identity_condition=SourceIdentityCondition((row(labels=False),), 1), **inputs)
    torch.testing.assert_close(training.noise, inference.noise, rtol=0, atol=0)
    assert training.source_alignment_loss.ndim == 0 and inference.source_alignment_loss is None
    actual = model_fn_wan_video(dit=checkpointed, source_identity_condition=condition,
                               use_gradient_checkpointing=True, **inputs)
    torch.testing.assert_close(actual.noise, training.noise, rtol=0, atol=0)
    torch.testing.assert_close(actual.source_alignment_loss, training.source_alignment_loss, rtol=0, atol=0)
    for result in (training, actual):
        (result.noise[:, :, 1:].square().mean() + .01 * result.source_alignment_loss).backward()
    for expected, observed in zip(model.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(observed.grad, expected.grad, rtol=0, atol=0)
