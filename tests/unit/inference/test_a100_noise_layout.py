"""初始噪声统一按 A100 逻辑网格启动；物理 SM 数不参与抽样合同。"""
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.support.paths import REPO_ROOT

_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from diffsynth.diffusion.a100_noise_layout import A100_SMS, A100_THREADS_PER_SM, _threads, randn_in_a100_layout
from diffsynth.diffusion import a100_noise_layout as noise_layout


def test_the_a100_launch_uses_its_full_grid_for_renderer_sized_draws():
    # 704p 目标一窗 (1, 48, 30, 44, 80)：A100 网格 108 × 8 × 256 = 221,184 个线程全用上。
    assert _threads(A100_SMS, A100_THREADS_PER_SM, 1 * 48 * 30 * 44 * 80) == 221_184
    assert _threads(A100_SMS, A100_THREADS_PER_SM, 1000) == 1024


def test_noise_is_drawn_only_on_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        randn_in_a100_layout((1, 48, 2, 44, 80), 42, torch.device("cpu"))


def test_public_noise_uses_fixed_logical_grid_without_querying_physical_sms(monkeypatch):
    shape = (1, 16, 30, 60, 104)
    output = torch.empty(shape)
    calls = []
    monkeypatch.setattr(noise_layout.torch, "empty", lambda *a, **k: output)
    class Generator:
        def __init__(self, device):
            assert device == torch.device("cuda:0")
        def manual_seed(self, seed):
            assert seed == 42
            return self
        def initial_seed(self):
            return 42
    monkeypatch.setattr(noise_layout.torch, "Generator", Generator)
    monkeypatch.setattr(noise_layout, "_noise_extension", lambda: SimpleNamespace(
        fill_normal=lambda tensor, seed, threads: calls.append((tensor, seed, threads))))
    assert randn_in_a100_layout(shape, 42, torch.device("cuda:0")) is output
    assert len(calls) == 1
    assert calls[0][0] is output and calls[0][1:] == (42, 221_184)


def test_extension_is_built_once_per_process_with_one_source_owner(monkeypatch):
    from torch.utils import cpp_extension
    calls = []
    extension = object()
    monkeypatch.setattr(cpp_extension, "load_inline", lambda **kwargs: calls.append(kwargs) or extension)
    noise_layout._noise_extension.cache_clear()
    try:
        assert noise_layout._noise_extension() is extension
        assert noise_layout._noise_extension() is extension
        assert len(calls) == 1
        assert calls[0]["functions"] == ["fill_normal"]
        assert calls[0]["extra_cuda_cflags"] == ["-O2"]
    finally:
        noise_layout._noise_extension.cache_clear()
