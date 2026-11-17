"""首帧通信的配置投影、逐镜文字隔离与 checkpoint 梯度接线。"""

import copy
from pathlib import Path

import pytest
import torch

from mspc.config.experiment import parse_experiment_mapping
from tests.unit.config.test_experiment_config import experiment_mapping
from tests.unit.models.test_memory_prefix_attention import _tiny_dit


@pytest.mark.parametrize("policy", ["full", "holocine_first_frame"])
def test_attention_policy_survives_training_and_inference_projection(policy, tmp_path):
    from mspc.inference.wan22 import build_config
    from mspc.training.backends.wan22 import configure_args
    from mspc.training.training_args import build_common_training_args

    raw = experiment_mapping()
    raw["model"]["self_attention_policy"] = policy
    raw["dataset"]["case_id_column"] = "case_id"
    # 此处只验配置投影；权重路径是接口要求，内容不在这一步读取。
    for name in ("dit", "t5", "vae"):
        path = tmp_path / (name + ".safetensors")
        path.touch()
        raw["model"]["paths"][name] = [str(path)] if name == "dit" else str(path)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    raw["model"]["paths"]["tokenizer"] = str(tokenizer)
    config = parse_experiment_mapping(raw, source_path=Path("/tmp/holocine.json"))
    args = configure_args(build_common_training_args(config), config, checkpoint=None)
    assert args.model_cfg["self_attention_policy"] == policy
    model_cfg, _, _, _ = build_config(config)
    assert model_cfg["self_attention_policy"] == policy


def _holocine_dit():
    dit = _tiny_dit()
    for block in dit.blocks:
        block.self_attn.self_attention_policy = "holocine_first_frame"
    return dit


def _forward(dit, latents, context, checkpoint=False):
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    return model_fn_wan_video(
        dit=dit, latents=latents, timestep=torch.tensor([500.]),
        context=context, latent_shot_groups=[(0, 2), (2, 4)],
        temporal_position_mode="global_time_native_rope_v1",
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=checkpoint,
    ).noise


def test_one_layer_routes_each_text_row_to_its_own_shot():
    dit = _holocine_dit()
    # 一层内先视频自注意力、再逐镜文字；第二层起文字可经首帧合法跨镜传播。
    dit.blocks = torch.nn.ModuleList([dit.blocks[0]])
    latents = torch.randn(1, 48, 4, 8, 8)
    context = torch.randn(2, 16, 4096, requires_grad=True)
    original = _forward(dit, latents, context)
    changed_context = context.detach().clone()
    changed_context[1] += torch.randn_like(changed_context[1])
    changed = _forward(dit, latents, changed_context)
    torch.testing.assert_close(original[:, :, :2], changed[:, :, :2], atol=0, rtol=0)
    assert not torch.allclose(original[:, :, 2:], changed[:, :, 2:])
    text_gradient, = torch.autograd.grad(original[:, :, :2].square().mean(), context)
    assert text_gradient[0].abs().sum() > 0
    assert torch.count_nonzero(text_gradient[1]) == 0


def test_state_dict_roundtrip_and_checkpoint_recompute_keep_outputs_and_gradients():
    dit = _holocine_dit()
    restored = _holocine_dit()
    restored.load_state_dict(copy.deepcopy(dit.state_dict()), strict=True)
    latents = torch.randn(1, 48, 4, 8, 8, requires_grad=True)
    copied_latents = latents.detach().clone().requires_grad_(True)
    context = torch.randn(2, 16, 4096)
    expected = _forward(dit, latents, context)
    actual = _forward(restored, copied_latents, context, checkpoint=True)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(copied_latents.grad, latents.grad)
    for (name, parameter), (restored_name, restored_parameter) in zip(
        dit.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert name == restored_name
        if parameter.grad is None:
            assert restored_parameter.grad is None
        else:
            torch.testing.assert_close(restored_parameter.grad, parameter.grad)
