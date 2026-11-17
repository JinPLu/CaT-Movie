"""历史评测与分析共用的配对bootstrap数学函数。

按story_id配对、有限值交集与旧DINO功效常量保留供既有分析消费者使用；
这些常量不定义整片1–4评分的分母、样本量或显著性门槛。
重采样次数由调用者显式传入，cluster_bootstrap沿用其固定轮数与种子。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 42
PREREGISTERED_MDE = 0.05


def load_rows(paths: Iterable[Path]) -> dict[str, dict[str, float | None]]:
    """把一批打分结果文件合并成 ``{story_id: {metric_id: value}}``。

    同一 story 的同一列在两份文件里取值不同即抛：这是刻意的，``scene_paired`` 那份会对
    ``dinov2_scene_consistency`` 重算，与 ``torch4`` 里的同名列不是一回事，不能静默选一个。
    """

    merged: dict[str, dict[str, float | None]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"{path}: status is {payload.get('status')!r}, not 'completed'")
        for row in payload["results"]:
            story_id = row.get("story_id") or row["case_id"]
            bucket = merged.setdefault(story_id, {})
            for metric_id, value in row["metrics"].items():
                if metric_id in bucket and bucket[metric_id] != value:
                    raise ValueError(
                        f"{path}: {story_id}.{metric_id} disagrees with an earlier file "
                        f"({bucket[metric_id]!r} vs {value!r})"
                    )
                bucket[metric_id] = value
    return merged


def paired(
    left: dict[str, dict[str, float | None]],
    right: dict[str, dict[str, float | None]],
    metric_id: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """两侧都对这一列给出有限值的 story：交集分母。"""

    stories, a_values, b_values = [], [], []
    for story_id in sorted(set(left) & set(right)):
        a = left[story_id].get(metric_id)
        b = right[story_id].get(metric_id)
        if a is None or b is None:
            continue
        if not (np.isfinite(float(a)) and np.isfinite(float(b))):
            continue
        stories.append(story_id)
        a_values.append(float(a))
        b_values.append(float(b))
    return stories, np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float)


def bootstrap_means(
    diffs: np.ndarray, rng: np.random.Generator, *, resamples: int
) -> np.ndarray:
    """``resamples`` 次有放回重采样的配对差均值。"""

    if type(resamples) is not int or resamples < 1:
        raise ValueError(f"resamples must be a positive int, got {resamples!r}")
    index = rng.integers(0, diffs.size, size=(resamples, diffs.size))
    return diffs[index].mean(axis=1)


def one_sided_p(diffs: np.ndarray, rng: np.random.Generator, *, resamples: int) -> float:
    """P(bootstrap mean <= 0)：H₀: Δ≤0 对 H₁: Δ>0。"""

    means = bootstrap_means(diffs, rng, resamples=resamples)
    return float((means <= 0.0).mean())


def paired_mean_interval(
    diffs: np.ndarray,
    rng: np.random.Generator,
    *,
    resamples: int,
) -> tuple[float, float, float]:
    """配对差的均值，以及重采样均值的 2.5% 与 97.5% 分位。

    这是区间，不是固定门槛。旧 A/A 的 0.022 不在这里当噪声上界。
    """

    if diffs.ndim != 1 or diffs.size < 2:
        raise ValueError(f"diffs must be a 1-d array of at least 2 values, got shape {diffs.shape}")
    if not np.isfinite(diffs).all():
        raise ValueError("diffs must be finite")
    means = bootstrap_means(diffs, rng, resamples=resamples)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lower), float(upper)


def required_n(sd_of_diff: float, effect: float = PREREGISTERED_MDE) -> float | None:
    """单侧 α=0.05、80% 功效下的配对 n（正态近似）。"""

    if sd_of_diff <= 0 or effect <= 0:
        return None
    return float(((1.6449 + 0.8416) * sd_of_diff / effect) ** 2)


def cluster_bootstrap(values: dict[str, list[float]]) -> dict[str, Any]:
    """按故事聚类的均值与 95% 区间：每轮有放回抽故事，取被抽故事全部镜的均值。"""

    stories = sorted(values)
    sums = np.array([sum(values[s]) for s in stories])
    counts = np.array([len(values[s]) for s in stories])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(stories), size=(BOOTSTRAP_ROUNDS, len(stories)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "stories": len(stories),
        "shots": int(counts.sum()),
    }
