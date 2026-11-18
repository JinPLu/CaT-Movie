"""Draw the sampler's initial CUDA noise in the element layout an A100 produces.

PyTorch's CUDA normal kernel (``ATen/native/cuda/DistributionTemplates.h``, ``calc_execution_policy``) launches
``grid = min(SMs * maxThreadsPerSM / 256, ceil(N / 256))`` blocks of 256 threads; thread ``idx`` is Philox subsequence
``idx`` and writes element ``li = idx + S * (4k + ii)`` from its k-th ``curand_normal4`` call, component ``ii``, with
``S = grid * 256``. The same seed therefore gives different noise on GPUs with different SM counts (108 against
148 SMs): measured on 11-14, only 4-7% of the elements agree, and the rendered stories differ like a seed change.

The kernel launches this same logical grid on every CUDA GPU. Its blocks need not reside on all SMs at once,
so devices with fewer physical SMs do not change the Philox subsequences or the output layout. The extension is built
once in PyTorch's source/version-aware cache; prewarm it before starting several renderer workers.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch

BLOCK = 256
A100_SMS = 108
A100_THREADS_PER_SM = 2048

_CPP_SOURCE = """
void fill_normal(torch::Tensor output, uint64_t seed, int64_t threads);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <cstdint>

__global__ void a100_normal(float* output, int64_t count, uint64_t seed) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, index, 0, &state);
    for (int64_t start = index; start < count; start += 4 * stride) {
        const float4 values = curand_normal4(&state);
        output[start] = values.x;
        if (start + stride < count) output[start + stride] = values.y;
        if (start + 2 * stride < count) output[start + 2 * stride] = values.z;
        if (start + 3 * stride < count) output[start + 3 * stride] = values.w;
    }
}

void fill_normal(torch::Tensor output, uint64_t seed, int64_t threads) {
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kFloat && output.is_contiguous(),
                "A100-layout noise requires a contiguous CUDA fp32 output");
    TORCH_CHECK(threads > 0 && threads % 256 == 0, "logical grid must contain whole 256-thread blocks");
    const c10::cuda::CUDAGuard guard(output.device());
    a100_normal<<<threads / 256, 256, 0, c10::cuda::getCurrentCUDAStream(output.get_device()).stream()>>>(
        output.data_ptr<float>(), output.numel(), seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def _noise_extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="storyboard_a100_noise", cpp_sources=_CPP_SOURCE, cuda_sources=_CUDA_SOURCE,
        functions=["fill_normal"], extra_cflags=["-O2"], extra_cuda_cflags=["-O2"], with_cuda=True,
        verbose=False)


def _threads(sms: int, threads_per_sm: int, numel: int) -> int:
    return min(sms * (threads_per_sm // BLOCK), math.ceil(numel / BLOCK)) * BLOCK


def randn_in_a100_layout(shape, seed: int, device: torch.device) -> torch.Tensor:
    """fp32 standard normal noise of ``shape`` on CUDA ``device``, laid out as an A100 draws ``torch.randn`` with ``seed``."""
    return _randn_in_grid(shape, seed, device, _threads(A100_SMS, A100_THREADS_PER_SM, math.prod(shape)))


def _randn_in_grid(shape, seed: int, device: torch.device, threads: int) -> torch.Tensor:
    """Draw one explicit logical grid; the private argument also permits native-grid equivalence checks."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"A100-layout noise is drawn on a CUDA device, got {device}")
    output = torch.empty(shape, device=device, dtype=torch.float32)
    if output.numel() == 0:
        return output
    # Use PyTorch's seed validation/remapping, including its signed-seed semantics.
    # This local generator draws nothing and leaves the global CUDA RNG untouched.
    generator = torch.Generator(device).manual_seed(seed)
    _noise_extension().fill_normal(output, generator.initial_seed(), threads)
    return output
