"""逐样本账与验证账 → 给人看的 loss 曲线。纯函数，只用标准库：训练之外（转发到 SwanLab 的那一端）算。

训练循环每个 rank 每条样本写一行 ``samples.rank{r}.jsonl``（step、这一步抽到的时间步、目标段、loss），主进程每次更新
写一行 ``metrics.jsonl``（全rank loss、lr_used/lr_next、梯度范数、耗时）；验证每次前向写一行 ``validation.rank{r}.jsonl``。汇成：

- ``train/loss``：这一次更新全部 rank 全部样本的均值（8 卡 × 累积 1 即 8 条），不是主进程那一条；
- ``train/uniform_loss``：同次预测与scheduler权重，目标空间均匀计权；``target_weighted_mse``与
  ``unweighted_mse``是施加scheduler时间步权重之前的两种误差；
- ``train/loss_avg{W}``：最近 W 次更新全部样本的均值；
- ``train_by_timestep/…``、``train_by_shots/…``：同一窗口里按时间步四档、按目标段镜数分开的均值；
- ``val/loss``、``val_by_timestep/…``、``val_by_shots/…``：一遍验证全部前向的均值与同样的分法。
- ``val_intermediate/…``：固定小面板的独立曲线，不与完整面板均值相接。

流匹配 loss 随时间步差一个量级，混在一起的均值主要反映这一批抽到了哪些时间步；分档以后每条曲线才看得出涨落。
四档的边界是训练抽签的四分位：下标 u 在 0–1 上均匀，调度器（shift=5）把它映到时间步 1000·5u／(1＋4u)，
u＝0.25、0.5、0.75 处是 625、833.3、937.5。名字里第一个 ``/`` 之前是 SwanLab 的图组。
"""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping

TIMESTEP_EDGES = (625.0, 2500.0 / 3.0, 937.5)
TIMESTEP_BUCKETS = ("t000_625", "t625_833", "t833_938", "t938_1000")  # 噪声由少到多
UPDATE_SCALARS = ("loss", "uniform_loss", "target_weighted_mse", "unweighted_mse",
                  "lr_used", "lr_next", "grad_norm", "step_seconds", "samples_seen")


def update_scalar_curves(metrics: Mapping[str, Any]) -> dict[str, float]:
    """同一份更新账用于离线SwanLab和实时转发；耗时是主进程视角。

    父阶段与子阶段重叠，例如forward包含VAE/T5/DiT，不可相加当总用时。
    """
    point = {f"train/{name}": float(metrics[name]) for name in UPDATE_SCALARS if name in metrics}
    point.update({f"timing/{name}": float(value) for name, value in metrics.items() if name.endswith("_seconds")})
    point.update({f"workload/{name}": float(metrics[name]) for name in
                  ("reference_encoded_frames", "train_rgb_frames", "train_latent_frames", "shot_count") if name in metrics})
    return point


def planner_curves(metric_rows: Iterable[Mapping[str, Any]], validation_rows: Iterable[Mapping[str, Any]],
                   *, window: int) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """规划器更新账已是全 rank 监督 token 均值；滑动窗口仍按监督 token 加权。"""
    if window < 1:
        raise ValueError("window must be positive")
    metrics = list(metric_rows)
    if [row["step"] for row in metrics] != list(range(1, len(metrics) + 1)):
        raise ValueError("planner steps must be complete, unique and consecutive")
    train = {}
    for index, row in enumerate(metrics):
        recent = metrics[max(0, index - window + 1):index + 1]
        if any(item["supervised_tokens"] <= 0 for item in recent):
            raise ValueError("planner update needs a positive supervised-token denominator")
        point = {f"train/{key}": float(row[key]) for key in ("loss", "lr", "grad_norm", "step_seconds")}
        point[f"train/loss_avg{window}"] = (
            sum(item["loss"] * item["supervised_tokens"] for item in recent)
            / sum(item["supervised_tokens"] for item in recent)
        )
        point["workload/tokens_per_second_per_gpu"] = float(row["tokens_per_second_per_gpu"])
        point["workload/sequence_tokens"] = float(row["sequence_tokens"])
        point["workload/supervised_tokens"] = float(row["supervised_tokens"])
        point.update({str(key): float(value) for key, value in row["resource_metrics"].items()})
        train[row["step"]] = point
    validation = {row["step"]: {"val/loss": float(row["val_loss"])} for row in validation_rows}
    return train, validation


def timestep_bucket(timestep: float) -> str:
    return TIMESTEP_BUCKETS[sum(timestep >= edge for edge in TIMESTEP_EDGES)]


def _split(rows: Iterable[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    """一组前向按时间步档与镜数分开的均值；``shots`` 取行里的镜数，训练行从目标段 ``span`` 算。"""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        shots = row["shots"] if "shots" in row else row["span"][1] - row["span"][0]
        groups[f"{prefix}_by_timestep/{timestep_bucket(row['timestep'])}"].append(row["loss"])
        groups[f"{prefix}_by_shots/shots_{shots}"].append(row["loss"])
    return {name: fmean(values) for name, values in groups.items()}


def train_curves(sample_rows: Iterable[Mapping[str, Any]], metric_rows: Iterable[Mapping[str, Any]], *,
                 samples_per_update: int, window: int) -> dict[int, dict[str, float]]:
    """每个**记全了的**更新一组标量：主进程那一行到了、``samples_per_update`` 条样本行也都到了才算。

    记不全的更新（文件还没写到）之后的更新也不出，曲线只往前延伸、不留洞。
    """
    by_step: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_step[row["step"]].append(row)
    metrics = {row["step"]: row for row in metric_rows}
    curves: dict[int, dict[str, float]] = {}
    step = 1
    while step in metrics and len(by_step[step]) == samples_per_update:
        recent = [row for past in range(max(1, step - window + 1), step + 1) for row in by_step[past]]
        point = {f"train/loss_avg{window}": fmean(row["loss"] for row in recent), **_split(recent, "train")}
        point.update(update_scalar_curves(metrics[step]))
        curves[step] = point
        step += 1
    if any(len(rows) > samples_per_update for rows in by_step.values()):
        raise ValueError("an update has more sample rows than samples_per_update; check the world size and accumulation")
    return curves


def validation_curves(rows: Iterable[Mapping[str, Any]], *, forwards_per_pass: int) -> dict[int, dict[str, float]]:
    """按每遍的完整分母汇总；新日志自述面板，历史日志使用显式 ``forwards_per_pass``。"""
    by_step: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    curves = {}
    for step, passes in sorted(by_step.items()):
        if any("validation_panel" in row or "validation_forwards" in row for row in passes):
            panels = {(row["validation_panel"], row["validation_forwards"]) for row in passes}
            if len(panels) != 1:
                raise ValueError("a validation pass has inconsistent panel metadata")
            panel, expected = panels.pop()
            if panel not in {"full", "intermediate"} or type(expected) is not int or expected < 1:
                raise ValueError("a validation pass has invalid panel metadata")
            prefix = "val" if panel == "full" else "val_intermediate"
        else:
            expected, prefix = forwards_per_pass, "val"
        if len(passes) > expected:
            raise ValueError("a validation pass has more rows than forwards_per_pass")
        if len(passes) != expected:
            continue
        point = {f"{prefix}/loss": fmean(row["loss"] for row in passes), **_split(passes, prefix)}
        if all("forward_seconds" in row and "data_seconds" in row and "rank" in row for row in passes):
            ranks = sorted({row["rank"] for row in passes})
            for stage in ("data_seconds", "forward_seconds"):
                totals = [sum(row[stage] for row in passes if row["rank"] == rank) for rank in ranks]
                point[f"{prefix}_timing/{stage}_rank_mean"] = fmean(totals)
                point[f"{prefix}_timing/{stage}_rank_max"] = max(totals)
        curves[step] = point
    return curves
