"""真实去噪循环在专家边界选择对应 CFG，参考回填逐步核两支调用。"""
from types import SimpleNamespace
import sys

import pytest
import torch

from mspc.config.experiment import ExperimentConfigError, parse_experiment_mapping
from mspc.inference.wan22 import _CleanPrefixDenoiseState
from tests.unit.config.test_experiment_config import experiment_mapping
from tests.unit.models.test_lora_and_noise_expert import TARGETS
from tests.support.paths import REPO_ROOT

RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.pipelines.wan_video import WanVideoPipeline


def _config():
    raw = experiment_mapping()
    raw["inference"].update(weight_source="expert_lora_checkpoints", dit_checkpoint=None,
        cfg_scale={"high_noise": 4.0, "low_noise": 3.0}, experts={
            "boundary": 0.875, "low_noise_dit": ["/srv/models/low.safetensors"],
            "high_noise_lora": "/srv/high-step1000.safetensors",
            "low_noise_lora": "/srv/low-step1000.safetensors",
            "lora": {"recipe": "lora_v1", "rank": 128, "alpha": 128, "targets": TARGETS},
        })
    return raw


def test_expert_cfg_requires_both_named_scales_and_two_experts():
    raw = _config()
    parsed = parse_experiment_mapping(raw, source_path="/srv/config.json").resolved
    assert dict(parsed["inference"]["cfg_scale"]) == {"high_noise": 4.0, "low_noise": 3.0}
    del raw["inference"]["cfg_scale"]["low_noise"]
    with pytest.raises(ExperimentConfigError, match="low_noise"):
        parse_experiment_mapping(raw, source_path="/srv/config.json")
    raw = experiment_mapping()
    raw["inference"]["cfg_scale"] = {"high_noise": 4.0, "low_noise": 3.0}
    with pytest.raises(ExperimentConfigError, match="requires inference.experts"):
        parse_experiment_mapping(raw, source_path="/srv/config.json")


@pytest.mark.parametrize("guidance,expected_predictions,calls", [
    ({"high_noise": 4.0, "low_noise": 3.0}, [13., 13., 10.], 6),
    ({"high_noise": 1.0, "low_noise": 3.0}, [4., 4., 10.], 4),
    ({"high_noise": 4.0, "low_noise": 1.0}, [13., 13., 4.], 5),
    (5.0, [16., 16., 16.], 6),
])
def test_real_denoise_loop_uses_expert_cfg_and_prefix_counts(guidance, expected_predictions, calls):
    class Scheduler(FlowMatchScheduler):
        def __init__(self):
            self.timesteps = torch.tensor([900., 875., 874.])
            self.predictions = []

        def set_timesteps(self, *args, **kwargs):
            pass

        def step(self, prediction, timestep, sample, **kwargs):
            self.predictions.append(float(prediction.mean()))
            return sample - prediction

    high, low = object(), object()
    seen = []
    pipe = SimpleNamespace(device="cpu", torch_dtype=torch.float32, dit=high, dit2=low, vace2=None,
        units=[object()], post_units=[], in_iteration_models=["dit"], in_iteration_models_2=["dit2"],
        check_resize_height_width=lambda h, w, f: (h, w), load_models_to_device=lambda _: None,
        scheduler=Scheduler(), vae=SimpleNamespace(decode=lambda latents, **kwargs: latents))

    def unit_runner(unit, pipe, shared, positive, negative):
        shared["latents"] = torch.zeros(1, 16, 2, 2, 2)
        positive["positive"] = True
        negative["positive"] = False
        return shared, positive, negative

    def model_fn(**kwargs):
        seen.append((float(kwargs["timestep"]), kwargs["dit"], kwargs["positive"]))
        return SimpleNamespace(noise=torch.full_like(kwargs["latents"], 4. if kwargs["positive"] else 1.))

    pipe.unit_runner, pipe.model_fn = unit_runner, model_fn
    reference = torch.full((1, 16, 1, 2, 2), 7.)
    controller = _CleanPrefixDenoiseState(pipe=pipe, original_model_fn=model_fn, cfg_scale=guidance,
        expert_boundary=0.875, clean_latents=reference, forward_kwargs={"prefix_frames": 1})
    controller.install()
    result = WanVideoPipeline.__call__(pipe, prompt="x", negative_prompt="", height=8, width=8,
        num_frames=5, cfg_scale=guidance, switch_DiT_boundary=0.875, num_inference_steps=3,
        sampler="flow_match_euler", output_type="floatpoint", progress_bar_cmd=lambda x: x)
    controller.assert_finished()
    controller.restore()
    assert pipe.scheduler.predictions == expected_predictions
    assert len(seen) == calls
    assert all(dit is (high if t >= 875 else low) for t, dit, _ in seen)
    assert torch.equal(result[:, :, :1], reference)
    expected_cfg = ([guidance["high_noise"], guidance["high_noise"], guidance["low_noise"]]
                    if isinstance(guidance, dict) else [guidance] * 3)
    assert pipe.denoising_steps == [
        {"timestep": t, "expert": expert, "cfg_scale": cfg,
         "model_calls": 1 if cfg == 1.0 else 2}
        for t, expert, cfg in zip([900., 875., 874.],
                                 ["high_noise", "high_noise", "low_noise"], expected_cfg)
    ]
    assert sum(step["model_calls"] for step in pipe.denoising_steps) == len(seen)
