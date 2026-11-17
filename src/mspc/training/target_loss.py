"""目标 latent 的 FlowMatch 空间误差；参考前缀由调用方在进入这里前切除。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def target_flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_loss_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回（按空间权重和归一化的 MSE，原均匀 MSE）。

    prediction/target 是相同形状的 [B,C,T,H,W] 目标流。
    权重是 [B,1,T,H,W]，在通道上共享；SAM 主体并集的覆盖率使它处于 [1,2]。
    None 是配置明确选择的原均匀目标，不表示缺失标注的自动回退。
    时间步权重仍归 scheduler，由调用方在本函数返回后施加。
    """
    if prediction.ndim != 5 or prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty matching [B,C,T,H,W] tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    # 保留既有均匀路径的逐值行为，独立交付原 MSE 供观测。
    uniform_mse = F.mse_loss(prediction.float(), target.float())
    if target_loss_weights is None:
        return uniform_mse, uniform_mse
    expected = (prediction.shape[0], 1, *prediction.shape[2:])
    if tuple(target_loss_weights.shape) != expected:
        raise ValueError(f"target_loss_weights must match target latent grid {expected}")
    if not target_loss_weights.is_floating_point() or target_loss_weights.requires_grad:
        raise ValueError("target_loss_weights must be a floating tensor without gradients")
    if not torch.isfinite(target_loss_weights).all() or not torch.all(
        (target_loss_weights >= 1) & (target_loss_weights <= 2)
    ):
        raise ValueError("target_loss_weights must be finite and in [1,2]")
    weights = target_loss_weights.to(device=prediction.device, dtype=torch.float32)
    spatial_error = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
    weighted_mse = (spatial_error * weights).sum() / weights.sum()
    return weighted_mse, uniform_mse
