"""Wan FlowMatch shift 只认 ``WAN22_FLOW_MATCH_SHIFT``，漏传不得落到第二个数字。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from mspc.core.wan22_contract import WAN22_FLOW_MATCH_SHIFT

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402


def test_set_timesteps_wan_refuses_a_missing_shift() -> None:
    with pytest.raises(TypeError, match="requires shift"):
        FlowMatchScheduler.set_timesteps_wan(num_inference_steps=4)


def test_wan_set_timesteps_without_shift_uses_the_contract_constant() -> None:
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(4, training=False)
    expected, _ = FlowMatchScheduler.set_timesteps_wan(
        num_inference_steps=4,
        denoising_strength=1.0,
        shift=WAN22_FLOW_MATCH_SHIFT,
    )
    assert torch.equal(scheduler.sigmas, expected)
