"""全DiT训练与checkpoint往返，拒绝缺失或错形状权重。"""
from pathlib import Path
import torch
from torch import nn
import pytest
from safetensors.torch import save_file, load_file
from mspc.training.dit_parameter_freeze import (
    FULL_DIT_V1, apply_named_dit_freeze, load_dit_state_dict, require_dit_parameter_freeze,
)


def _tiny_dit():
    from tests.unit.models.test_memory_prefix_attention import _tiny_dit as make
    return make()


def test_full_dit_trains_every_backbone_parameter():
    dit = _tiny_dit()
    for p in dit.parameters(): p.requires_grad_(False)
    partition = apply_named_dit_freeze(dit, {"recipe": FULL_DIT_V1})
    assert partition["frozen"] == ()
    assert set(partition["trainable"]) == set(dict(dit.named_parameters()))
    assert all(p.requires_grad for p in dit.parameters())


def test_native_structure_checkpoint_round_trip_preserves_output(tmp_path: Path):
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    dit = _tiny_dit()
    torch.manual_seed(41)
    x = torch.randn(1, 48, 3, 8, 8)
    context = torch.randn(1, 5, 4096)
    timestep = torch.tensor([500.])
    with torch.no_grad():
        expected = model_fn_wan_video(dit=dit, latents=x, timestep=timestep, context=context).noise
    path = tmp_path / "full.safetensors"
    save_file(dit.state_dict(), str(path))
    restored = _tiny_dit()
    for p in restored.parameters(): p.data.zero_()
    assert load_dit_state_dict(restored, load_file(str(path))) == "full"
    assert dit.state_dict().keys() == restored.state_dict().keys()
    with torch.no_grad():
        actual = model_fn_wan_video(dit=restored, latents=x, timestep=timestep, context=context).noise
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape"])
def test_incomplete_or_mismatched_checkpoint_is_rejected(fault):
    dit = _tiny_dit()
    state = dict(dit.state_dict())
    key = "blocks.0.self_attn.q.weight"
    if fault == "missing": del state[key]
    elif fault == "extra": state["extra.weight"] = torch.zeros(1)
    else: state[key] = state[key][:-1]
    with pytest.raises(RuntimeError):
        load_dit_state_dict(dit, state)


def test_full_dit_rejects_empty_model():
    with pytest.raises(RuntimeError, match="no DiT parameters"):
        apply_named_dit_freeze(nn.Module(), {"recipe": FULL_DIT_V1})


def test_invalid_freeze_contract_is_rejected():
    for invalid in (None, {}, {"recipe": "missing"}, {"recipe": FULL_DIT_V1, "extra": True}):
        with pytest.raises(ValueError): require_dit_parameter_freeze(invalid)
