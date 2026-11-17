"""初始噪声按 A100 排布：排布算法与非 CUDA 设备的拒绝（GPU 上的逐位核对：
A100 上与原生 torch.randn 逐位相同，网格更小的卡报错，网格更大的卡能复现 A100 的噪声）。"""
import sys

import pytest
import torch

from tests.support.paths import REPO_ROOT

_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from diffsynth.diffusion.a100_noise_layout import A100_SMS, A100_THREADS_PER_SM, _threads, randn_in_a100_layout


def test_the_a100_launch_uses_its_full_grid_for_renderer_sized_draws():
    # 704p 目标一窗 (1, 48, 30, 44, 80)：A100 网格 108 × 8 × 256 = 221,184 个线程全用上。
    assert _threads(A100_SMS, A100_THREADS_PER_SM, 1 * 48 * 30 * 44 * 80) == 221_184
    assert _threads(A100_SMS, A100_THREADS_PER_SM, 1000) == 1024


def test_noise_is_drawn_only_on_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        randn_in_a100_layout((1, 48, 2, 44, 80), 42, torch.device("cpu"))
