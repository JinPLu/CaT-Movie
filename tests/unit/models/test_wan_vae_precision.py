"""共同VAE编码入口隔离训练engine的外层AMP，保留调用方既有dtype。"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/wan22_runtime"))
from diffsynth.models.wan_video_vae import WanVideoVAE, WanVideoVAE38


@pytest.mark.parametrize("vae_type,stride", [(WanVideoVAE, 8), (WanVideoVAE38, 16)])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_common_encode_isolates_outer_amp_and_preserves_input_dtype(
    monkeypatch, vae_type, stride, tiled, input_dtype
):
    # 不分配完整VAE：使用真实共同encode，只在昂贵的实际编码边界观察上下文。
    vae = object.__new__(vae_type)
    nn.Module.__init__(vae)
    vae.upsampling_factor = stride
    calls = []

    def encode_backend(video, device, *tile_options):
        calls.append({"autocast": torch.is_autocast_enabled("cpu"),
                      "input_dtype": video.dtype, "tile_options": tile_options})
        # AMP-sensitive运算：外层CPU AMP泄漏时这个FP32 linear会变成BF16。
        # 不复刻VAE算术，只验证进入昂贵编码前的精度边界。
        values = video.movedim(1, -1).float()
        return F.linear(values, torch.eye(3)).movedim(-1, 1)

    method = "tiled_encode" if tiled else "single_encode"
    monkeypatch.setattr(vae, method, encode_backend)
    video = torch.linspace(-1, 1, 3 * 5 * 4 * 4).reshape(3, 5, 4, 4).to(input_dtype)
    options = {"device": torch.device("cpu"), "tiled": tiled,
               "tile_size": (4, 6), "tile_stride": (2, 3)}
    with torch.autocast("cpu", enabled=False):
        expected = vae.encode([video], **options)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = vae.encode([video], **options)
        assert torch.is_autocast_enabled("cpu")  # 只隔离encode，不更改调用方环境。
    assert torch.equal(actual, expected)
    assert actual.dtype == expected.dtype == torch.float32
    assert calls == [{"autocast": False, "input_dtype": input_dtype,
                      "tile_options": ((4 * stride, 6 * stride), (2 * stride, 3 * stride)) if tiled else ()}] * 2


@pytest.mark.parametrize("vae_type", [WanVideoVAE, WanVideoVAE38])
@pytest.mark.parametrize("tiled", [False, True])
def test_common_encode_keeps_the_input_device_until_the_encoding_boundary(monkeypatch, vae_type, tiled):
    # A meta input cannot be copied to CPU. This checks the real shared wrapper
    # without a CUDA device or allocating the full VAE's parameters.
    vae = object.__new__(vae_type)
    nn.Module.__init__(vae)
    vae.upsampling_factor = 8 if vae_type is WanVideoVAE else 16
    video = torch.empty(3, 5, 4, 4, device="meta", dtype=torch.bfloat16)
    seen = []

    def encode_backend(batch, device, *tile_options):
        seen.append((batch.device, batch.dtype, device))
        return batch

    monkeypatch.setattr(vae, "tiled_encode" if tiled else "single_encode", encode_backend)
    result = vae.encode([video], device=torch.device("cpu"), tiled=tiled)
    assert seen == [(torch.device("meta"), torch.bfloat16, torch.device("cpu"))]
    assert result.device.type == "meta" and result.shape == (1, 3, 5, 4, 4)
