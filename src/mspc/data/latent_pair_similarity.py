"""两张潜帧是不是「同一机位接着拍」：位置对齐余弦减打乱余弦。

训练配对核查（``tools/audit/training_pair_similarity.py``）与推理侧记忆去重
（``continuity_dev_32`` 的 ``generated_prev_dedupe`` 臂）共用这一把尺子。

**指标**：两张潜帧 ``z1, z2``（``[C, H, W]``）在每个空间位置取 C 维向量的余弦，空间平均得
「位置对齐余弦」；把 ``z2`` 的空间位置按固定置换打乱后同样算得「打乱余弦」；指标是两者之差。
打乱余弦吸收了「颜色／整体内容相近」，差值只剩「同一位置上是同样的东西」，也就是构图与
机位的延续。置换由 ``PERMUTATION_SEED`` 按网格大小确定，所有配对共用。

**阈值**：:data:`TAU_FIRST_FIRST` 是 ``real_holdout_96`` 真片「两边都被 TransNet 干净检出的
硬切」上「前镜第 0 潜帧 → 本镜第 0 潜帧」的 95 分位（``real-tau``）。
"""

from __future__ import annotations

import numpy as np


PERMUTATION_SEED = 0
TAU_FIRST_FIRST = 0.2456


def _permutation(positions: int) -> np.ndarray:
    return np.random.default_rng(PERMUTATION_SEED).permutation(positions)


def aligned_minus_shuffled_cosine(z1: np.ndarray, z2: np.ndarray) -> float:
    """``[C, H, W]`` 两帧：逐位置通道余弦的空间均值，减去 ``z2`` 空间打乱后的同一量。"""

    if z1.shape != z2.shape or z1.ndim != 3:
        raise ValueError(f"need two [C, H, W] latents of equal shape, got {z1.shape} and {z2.shape}")
    channels = z1.shape[0]
    a = z1.reshape(channels, -1).astype(np.float64)
    b = z2.reshape(channels, -1).astype(np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)
    aligned = float(np.mean(np.sum(a * b, axis=0)))
    shuffled = float(np.mean(np.sum(a * b[:, _permutation(a.shape[1])], axis=0)))
    return aligned - shuffled
