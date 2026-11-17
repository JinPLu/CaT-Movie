"""目标加权误差的数学约束；这些张量不代表模型训练结果。"""
import pytest
import torch
import torch.nn.functional as F

from mspc.training.target_loss import target_flow_mse


def test_explicit_uniform_path_preserves_existing_mse():
    generator = torch.Generator().manual_seed(42)
    prediction = torch.randn(1, 48, 3, 30, 52, generator=generator, dtype=torch.bfloat16)
    target = torch.randn(prediction.shape, generator=generator, dtype=torch.bfloat16)
    loss, observed = target_flow_mse(prediction, target, None)
    assert torch.equal(loss, F.mse_loss(prediction.float(), target.float()))
    assert loss is observed


def test_weight_sum_normalization_preserves_scale_and_subject_relative_gradient():
    prediction = torch.ones(1, 48, 1, 1, 4, requires_grad=True)
    weights = torch.tensor([2., 1., 1., 1.]).view(1, 1, 1, 1, 4)
    loss, uniform = target_flow_mse(prediction, torch.zeros_like(prediction), weights)
    assert loss.item() == uniform.item() == 1
    loss.backward()
    position_gradient = prediction.grad.sum(dim=1).flatten()
    torch.testing.assert_close(position_gradient, torch.tensor([.8, .4, .4, .4]))
    torch.testing.assert_close(position_gradient[0] / position_gradient.sum(), torch.tensor(.4))


def test_different_errors_use_channel_mean_and_spatial_weight_sum():
    prediction = torch.tensor([1., 3., 2., 4.]).reshape(1, 2, 1, 1, 2)
    weights = torch.tensor([2., 1.]).reshape(1, 1, 1, 1, 2)
    loss, uniform = target_flow_mse(prediction, torch.zeros_like(prediction), weights)
    # 两个位置的跨通道均方为 2.5 和 12.5。
    torch.testing.assert_close(loss, torch.tensor((2 * 2.5 + 12.5) / 3))
    torch.testing.assert_close(uniform, torch.tensor(7.5))


@pytest.mark.parametrize("shape", [(1, 1, 3, 15, 26), (1, 48, 3, 30, 52), (1, 1, 4, 30, 52)])
def test_rejects_patch_grid_channels_and_reference_prefix_in_weights(shape):
    prediction = torch.zeros(1, 48, 3, 30, 52)
    with pytest.raises(ValueError, match="target latent grid"):
        target_flow_mse(prediction, prediction, torch.ones(shape))


@pytest.mark.parametrize("value", [0., 2.1, float("nan"), float("inf")])
def test_rejects_invalid_region_weights(value):
    prediction = torch.zeros(1, 48, 1, 1, 1)
    with pytest.raises(ValueError, match="finite and in"):
        target_flow_mse(prediction, prediction, torch.full((1, 1, 1, 1, 1), value))


def test_region_weights_cannot_train_a_mask_generator():
    prediction = torch.zeros(1, 48, 1, 1, 1)
    with pytest.raises(ValueError, match="without gradients"):
        target_flow_mse(prediction, prediction, torch.ones(1, 1, 1, 1, 1, requires_grad=True))
