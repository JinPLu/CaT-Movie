"""A14B LoRA 与去噪专家时间段：挂上即与原模型同输出、只训低秩参数且常驻 FP32；专家时间段与官方边界一致。"""
import pytest
import torch

from mspc.models.lora import LORA_PARAMETER_SUFFIXES, LoRALinear, inject_lora
from mspc.models.noise_expert import (NoiseExpertOptions, expert_timestep_index_range,
                                      index_range_as_boundaries)
from mspc.models.wan22_precision import apply_wan22_parameter_storage, verify_wan22_parameter_storage
from mspc.training.dit_parameter_freeze import apply_named_dit_freeze, require_dit_trainable

TARGETS = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
           "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"]
LORA = {"recipe": "lora_v1", "rank": 4, "alpha": 4, "targets": TARGETS}


def _tiny_dit():
    from tests.unit.models.test_memory_prefix_attention import _tiny_dit as make
    return make()


def _forward(dit):
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    torch.manual_seed(7)
    x = torch.randn(1, 48, 3, 8, 8)
    context = torch.randn(1, 5, 4096)
    with torch.no_grad():
        return model_fn_wan_video(dit=dit, latents=x, timestep=torch.tensor([500.]), context=context).noise


def test_lora_starts_at_the_original_output_and_trains_only_low_rank_parameters():
    dit = _tiny_dit()
    expected = _forward(dit)
    partition = apply_named_dit_freeze(dit, LORA)
    assert torch.equal(_forward(dit), expected)
    assert len(partition["trainable"]) == len(dit.blocks) * len(TARGETS) * 2
    assert all(name.endswith(LORA_PARAMETER_SUFFIXES) for name in partition["trainable"])
    require_dit_trainable(dit.train(), LORA)
    next(p for n, p in dit.named_parameters() if n.endswith(".base.weight")).requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainability"):
        require_dit_trainable(dit, LORA)


def test_lora_reads_the_frozen_resolved_config_shape():
    from types import MappingProxyType
    frozen = MappingProxyType({**LORA, "targets": tuple(TARGETS)})
    partition = apply_named_dit_freeze(_tiny_dit(), frozen)
    assert len(partition["trainable"]) == 2 * len(TARGETS) * 2


def test_lora_parameters_stay_fp32_beside_bf16_storage():
    dit = _tiny_dit()
    apply_wan22_parameter_storage(dit)
    apply_named_dit_freeze(dit, LORA)
    verify_wan22_parameter_storage(dit)
    module = dit.blocks[0].self_attn.q
    assert isinstance(module, LoRALinear)
    assert module.lora_down.weight.dtype == torch.float32 and module.base.weight.dtype == torch.bfloat16


def test_lora_rejects_a_target_that_is_not_linear():
    with pytest.raises(ValueError, match="nn.Linear"):
        inject_lora(_tiny_dit(), rank=2, alpha=2.0, targets=["ffn.1"])


def _training_timesteps():
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(1000, training=True, shift=5.0)
    return scheduler.timesteps


def test_expert_ranges_split_the_training_table_at_the_official_boundary():
    timesteps = _training_timesteps()
    high = expert_timestep_index_range(timesteps, NoiseExpertOptions("high", 0.875), 1000)
    low = expert_timestep_index_range(timesteps, NoiseExpertOptions("low", 0.875), 1000)
    assert high[0] == 0 and high[1] == low[0] and low[1] == len(timesteps)
    assert bool((timesteps[:high[1]] >= 875).all()) and bool((timesteps[low[0]:] < 875).all())
    # DiffSynth 官方 T2V-A14B 高噪脚本按 shift 5 写 max_timestep_boundary=0.417。
    assert high[1] == 417
    for index_range in (high, low):
        lo, hi = index_range_as_boundaries(index_range, len(timesteps))
        assert (int(lo * len(timesteps)), int(hi * len(timesteps))) == index_range


def test_expert_options_need_exact_fields():
    with pytest.raises(ValueError):
        NoiseExpertOptions.from_mapping({"noise_level": "mid", "boundary": 0.875})
    with pytest.raises(ValueError):
        NoiseExpertOptions.from_mapping({"noise_level": "high", "boundary": 1})


def test_lora_export_round_trips_into_a_fresh_mounted_dit():
    from mspc.models.lora import load_lora_state_dict
    trained = _tiny_dit()
    apply_named_dit_freeze(trained, LORA)
    for name, parameter in trained.named_parameters():
        if name.endswith(LORA_PARAMETER_SUFFIXES):
            parameter.data.normal_(0, 0.02)
    export = {name: parameter.detach().clone() for name, parameter in trained.named_parameters()
              if parameter.requires_grad}
    fresh = _tiny_dit()
    fresh.load_state_dict({k: v for k, v in trained.state_dict().items() if not k.endswith(LORA_PARAMETER_SUFFIXES)},
                          strict=False)
    inject_lora(fresh, rank=LORA["rank"], alpha=float(LORA["alpha"]), targets=TARGETS)
    assert load_lora_state_dict(fresh, export) == len(export)
    assert torch.equal(_forward(fresh), _forward(trained))
    with pytest.raises(ValueError, match="differ"):
        load_lora_state_dict(fresh, dict(list(export.items())[1:]))
