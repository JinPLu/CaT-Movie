from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_dit import SelfAttention
from mspc.evaluation.renderer_diagnostics import own_shot_attention, sampled_attention_mass


@pytest.mark.parametrize("policy", ["full", "holocine_first_frame"])
def test_single_shot_intervention_is_identical_and_other_shot_cannot_change_output(policy):
    torch.manual_seed(3)
    module = SelfAttention(24, 2).double()
    module.self_attention_policy = policy
    x = torch.randn(1, 12, 24, dtype=torch.float64)
    angles = torch.randn(12, 1, 6, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angles), angles)
    expected = module(x, freqs, latent_shot_groups=[(0, 4)], tokens_per_frame=3)
    actual = own_shot_attention(module, x, freqs, [(0, 4)], 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original = own_shot_attention(module, x, freqs, [(0, 2), (2, 4)], 3)
    changed = x.clone()
    changed[:, 6:] += 10
    isolated = own_shot_attention(module, changed, freqs, [(0, 2), (2, 4)], 3)
    torch.testing.assert_close(original[:, :6], isolated[:, :6], rtol=0, atol=0)
    assert not torch.equal(original[:, 6:], isolated[:, 6:])


def test_sampled_mass_preserves_reference_mask_and_holocine_duplicate_count():
    module = SelfAttention(24, 2)
    with torch.no_grad():
        for p in (module.q.weight, module.q.bias, module.k.weight, module.k.bias):
            p.zero_()
    x = torch.randn(1, 12, 24)
    freqs = torch.ones(12, 1, 6, dtype=torch.complex64)
    groups = [(0, 1), (1, 4)]
    mass = sampled_attention_mass(module, x, freqs, groups, 3, "memory_self_only_v1", 1, own_shot=False)
    torch.testing.assert_close(torch.tensor(mass), torch.tensor([[1., 0.], [.25, .75]]))
    module.self_attention_policy = "holocine_first_frame"
    mass = sampled_attention_mass(module, x, freqs, groups, 3, None, 0, own_shot=False)
    torch.testing.assert_close(torch.tensor(mass), torch.tensor([[2/3, 1/3], [1/5, 4/5]]))
    cut = sampled_attention_mass(module, x, freqs, groups, 3, None, 0, own_shot=True)
    torch.testing.assert_close(torch.tensor(cut), torch.eye(2))
