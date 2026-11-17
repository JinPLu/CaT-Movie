"""Draw the sampler's initial CUDA noise in the element layout an A100 produces, on any GPU whose grid can hold it.

PyTorch's CUDA normal kernel (``ATen/native/cuda/DistributionTemplates.h``, ``calc_execution_policy``) launches
``grid = min(SMs * maxThreadsPerSM / 256, ceil(N / 256))`` blocks of 256 threads; thread ``idx`` is Philox subsequence
``idx`` and writes element ``li = idx + S * (4k + ii)`` from its k-th ``curand_normal4`` call, component ``ii``, with
``S = grid * 256``. The same seed therefore gives different noise on GPUs with different SM counts (108 against
148 SMs): measured on 11-14, only 4-7% of the elements agree, and the rendered stories differ like a seed change.

``randn_in_a100_layout`` launches the device's own full grid once, then keeps the first ``S_A100`` threads of every
draw, which is exactly what an A100 writes. On an A100 the two layouts are the same, so the result equals the native
``torch.randn`` bit for bit. A device whose grid is smaller than the A100's cannot hold the A100 layout and raises. (Measured: on a 148-SM GPU
the reproduction is bit-equal to its own native draw when told its own SM count, and rendering with the A100 layout
there matches the A100 stories' first window at 31-35 dB PSNR.)
"""
from __future__ import annotations

import math

import torch

BLOCK = 256
A100_SMS = 108
A100_THREADS_PER_SM = 2048


def _threads(sms: int, threads_per_sm: int, numel: int) -> int:
    return min(sms * (threads_per_sm // BLOCK), math.ceil(numel / BLOCK)) * BLOCK


def randn_in_a100_layout(shape, seed: int, device: torch.device) -> torch.Tensor:
    """fp32 standard normal noise of ``shape`` on CUDA ``device``, laid out as an A100 draws ``torch.randn`` with ``seed``."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"A100-layout noise is drawn on a CUDA device, got {device}")
    numel = math.prod(shape)
    target = _threads(A100_SMS, A100_THREADS_PER_SM, numel)
    props = torch.cuda.get_device_properties(device)
    own = props.multi_processor_count * (props.max_threads_per_multi_processor // BLOCK) * BLOCK
    if own < target:
        raise RuntimeError(f"{props.name} grid of {own} threads cannot hold the A100 noise layout of {target} threads")
    draws = math.ceil(numel / (4 * target))
    generator = torch.Generator(device).manual_seed(seed)
    raw = torch.randn(4 * own * draws, generator=generator, device=device, dtype=torch.float32)
    return raw.view(draws, 4, own)[:, :, :target].reshape(-1)[:numel].view(shape)
