"""本地 UniPC 与钉住的官方 ``fm_solvers_unipc.py`` 在 CPU 上同一步。"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

from mspc.inference.flow_unipc import (
    FLOW_UNIPC_OFFICIAL,
    FlowUniPCMultistepScheduler,
    FlowUniPCOfficial,
    require_sampler,
)
from tests.support.paths import REPO_ROOT

OFFICIAL_PATH = REPO_ROOT / "tests/fixtures/wan22_1ea34ff_fm_solvers_unipc.py"


def _official_class():
    from mspc.inference import flow_unipc as port

    for name in (
        "diffusers",
        "diffusers.configuration_utils",
        "diffusers.schedulers",
        "diffusers.schedulers.scheduling_utils",
        "diffusers.utils",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    config_mod = sys.modules["diffusers.configuration_utils"]
    config_mod.ConfigMixin = port.ConfigMixin
    config_mod.register_to_config = port.register_to_config
    schedulers = sys.modules["diffusers.schedulers.scheduling_utils"]
    schedulers.KarrasDiffusionSchedulers = port.KarrasDiffusionSchedulers
    schedulers.SchedulerMixin = port.SchedulerMixin
    schedulers.SchedulerOutput = port.SchedulerOutput
    utils = sys.modules["diffusers.utils"]
    utils.deprecate = port.deprecate
    utils.is_scipy_available = lambda: False
    spec = importlib.util.spec_from_file_location("official_wan22_fm_solvers_unipc", OFFICIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


def _pair():
    official_cls = _official_class()
    kwargs = dict(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        final_sigmas_type="zero",
    )
    official = official_cls(**kwargs)
    local = FlowUniPCMultistepScheduler(**kwargs)
    return official, local


def test_sigma_grid_starts_at_the_official_training_endpoint() -> None:
    _official, local = _pair()
    assert local.sigma_max == pytest.approx(0.999)
    assert local.sigma_min == 0.0
    local.set_timesteps(50, device="cpu", shift=5)
    assert local.sigmas[-1].item() == 0.0
    assert len(local.timesteps) == 50
    assert local.timesteps.dtype == torch.int64
    # linspace(0.999, 0, 51)[:-1], then shift 5. The first node is 0.999.
    first = 5 * 0.999 / (1 + 4 * 0.999)
    assert local.sigmas[0].item() == pytest.approx(first, rel=1e-6)


def test_updates_match_the_pinned_official_solver() -> None:
    official, local = _pair()
    adapted = FlowUniPCOfficial()
    official.set_timesteps(50, device="cpu", shift=5)
    local.set_timesteps(50, device="cpu", shift=5)
    adapted.set_timesteps(50, denoising_strength=1.0, shift=5, device="cpu")
    assert torch.equal(local.sigmas, official.sigmas)
    assert torch.equal(local.timesteps, official.timesteps)
    assert torch.equal(adapted.sigmas, official.sigmas)
    assert torch.equal(adapted.timesteps, official.timesteps)
    generator = torch.Generator().manual_seed(20261027)
    sample = torch.randn(2, 4, 3, 5, 7, generator=generator)
    for index in range(len(official.timesteps)):
        model_output = torch.randn(sample.shape, generator=generator)
        timestep = official.timesteps[index]
        official_next = official.step(model_output, timestep, sample, return_dict=False)[0]
        local_next = local.step(model_output, local.timesteps[index], sample, return_dict=False)[0]
        adapted_next = adapted.step(model_output, adapted.timesteps[index], sample)
        assert torch.allclose(local_next, official_next, rtol=1e-5, atol=1e-5)
        assert torch.allclose(adapted_next, official_next, rtol=1e-5, atol=1e-5)
        sample = official_next


def test_sampler_name_is_explicit() -> None:
    assert require_sampler(FLOW_UNIPC_OFFICIAL) == FLOW_UNIPC_OFFICIAL
    with pytest.raises(ValueError, match="sampler"):
        require_sampler("unipc")
    with pytest.raises(ValueError, match="denoising_strength"):
        FlowUniPCOfficial().set_timesteps(4, denoising_strength=0.5, shift=5)
