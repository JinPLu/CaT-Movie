"""训练标量的外部展示（SwanLab）。本地逐步日志是事实来源，这里只把同一批数再记一份，不另算。

用法照 SwanLab 官方文档（docs.swanlab.cn，2026-10-11 读）：

- ``swanlab.init(project=…, workspace=…, experiment_name=…, config=…, logdir=…, mode=…)``，``mode`` 取 ``online``（上传云端）或
  ``offline``（只落本地盘）；``swanlab.log({名: 值}, step=…)``，同一指标同一 step 只留第一次；名字里的 ``/`` 用来分组；
  ``swanlab.finish()`` 收尾。
- 密钥走环境变量 ``SWANLAB_API_KEY``（登录时最先读它）。这里从配置写明的文件读进环境变量，不打印、不进 argv。
- 离线跑出的日志在 ``<logdir>/run-*``，事后在能联网的机器上 ``swanlab sync <logdir>/run-*`` 上传（SDK 不低于 0.8.0）。
- 多卡训练只在主进程建这个对象并记数；其余进程不碰 SwanLab。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

MODES = ("online", "offline")
KEYS = ("project", "workspace", "experiment_name", "mode", "api_key_file")
SWANLOG_DIRNAME = "swanlog"


@dataclass(frozen=True)
class LossTracking:
    project: str
    workspace: str
    experiment_name: str
    mode: str
    api_key_file: str | None

    @classmethod
    def from_mapping(cls, raw: Any) -> "LossTracking":
        if not isinstance(raw, Mapping) or set(raw) != set(KEYS):
            raise ValueError(f"loss_tracking must contain exactly {list(KEYS)}")
        for key in ("project", "workspace", "experiment_name"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ValueError(f"loss_tracking.{key} must be a nonempty string")
        if raw["mode"] not in MODES:
            raise ValueError(f"loss_tracking.mode must be one of {MODES}")
        key_file = raw["api_key_file"]
        if raw["mode"] == "online":
            if not isinstance(key_file, str) or not key_file.strip():
                raise ValueError("loss_tracking.api_key_file names the SwanLab API key file when mode is online")
        elif key_file is not None:
            raise ValueError("loss_tracking.api_key_file must be null when mode is offline (no key is read)")
        return cls(raw["project"], raw["workspace"], raw["experiment_name"], raw["mode"], key_file)


class SwanLabRun:
    """一次训练对应一个 SwanLab 实验；只在主进程构造。"""

    def __init__(self, tracking: LossTracking, *, hyperparameters: Mapping[str, Any], log_dir: Path) -> None:
        import swanlab

        if tracking.mode == "online":
            key_path = Path(tracking.api_key_file)
            if not key_path.is_file():
                raise FileNotFoundError(
                    f"SwanLab API key file is missing: {key_path}. Put the key from the SwanLab account settings page "
                    "into this file (one line), or set loss_tracking.mode to offline and upload later with `swanlab sync`.")
            os.environ["SWANLAB_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
        self.logdir = Path(log_dir) / SWANLOG_DIRNAME
        # 看板里的实验名＝配置里的名字（写训练配置名）加这次启动的 UTC 时刻，同一配置的多次运行分得开。
        self.experiment_name = f"{tracking.experiment_name}-{datetime.now(timezone.utc):%Y%m%dT%H%MZ}"
        self._swanlab = swanlab
        self._run = swanlab.init(project=tracking.project, workspace=tracking.workspace,
                                 experiment_name=self.experiment_name, config=dict(hyperparameters),
                                 logdir=str(self.logdir), mode=tracking.mode)

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        self._swanlab.log(dict(metrics), step=step)

    def finish(self) -> None:
        self._swanlab.finish()


def open_loss_tracking(raw: Any, *, hyperparameters: Mapping[str, Any], log_dir: Path) -> SwanLabRun | None:
    """配置里 ``loss_tracking`` 为 null 就不接外部展示；否则按配置建实验。"""
    if raw is None:
        return None
    return SwanLabRun(LossTracking.from_mapping(raw), hyperparameters=hyperparameters, log_dir=log_dir)


#: 每次优化器更新写进 ``logs/metrics.jsonl`` 的耗时项：键名 -> ``timings`` 里的来源。
UPDATE_METRIC_TIMINGS = (("step_seconds", "step_total"), ("data_seconds", "data_wait"),
                         ("forward_seconds", "forward"), ("backward_seconds", "backward"))

CPU_STAGE_TIMINGS = ("dataset_selection_seconds", "target_read_seconds", "reference_open_seconds",
                     "reference_index_seconds", "reference_decode_seconds", "reference_crop_seconds",
                     "reference_read_seconds")
GPU_STAGE_TIMINGS = ("input_prepare_seconds", "reference_vae_seconds", "t5_seconds", "dit_forward_seconds")
# Engine的forward hook也包住raw_model验证，fwd累计不等于训练前向；该项由loop直接计时。
DEEPSPEED_TIMERS = {"deepspeed_backward_seconds": "bwd",
                   "deepspeed_backward_inner_seconds": "bwd_inner",
                   "deepspeed_backward_reduce_seconds": "bwd_allreduce",
                   "deepspeed_update_seconds": "step"}


def reported_deepspeed_timings(model: Any) -> dict[str, float]:
    """DeepSpeed原生更新边界计时，毫秒转秒；step包含裁剪、更新及参数同步。

    Accelerate在backward内调用engine.step；外层optimizer.step是空操作。
    非DeepSpeed模型没有这组仪表；未启用wall_clock_breakdown的引擎不报数。
    """
    getter = getattr(model, "get_wall_clock_timers", None)
    if getter is None:
        return {}
    measured = getter()
    if not measured:
        return {}
    return {name: float(measured[source]) / 1000 for name, source in DEEPSPEED_TIMERS.items()}


def reported_grad_norm(model: Any) -> float | None:
    """DeepSpeed 引擎在它自己的优化步里算出的全局梯度范数；不是 DeepSpeed 引擎（或还没走过一步）就没有这个读数。"""
    getter = getattr(model, "get_global_grad_norm", None)
    value = None if getter is None else getter()
    return None if value is None else float(value)


def update_metrics(*, losses: Mapping[str, float], learning_rate: float, next_learning_rate: float,
                   timings: Mapping[str, Any], samples_seen: int, elapsed_seconds: float,
                   grad_norm: float | None) -> dict[str, float]:
    """一次更新的全rank损失均值；耗时与工作量仍是主rank视角。

    training_step_seconds排除checkpoint hook、取数和验证；既有step_seconds包含checkpoint hook。
    elapsed_seconds从起点验证前到本条记录前，包含已经发生的验证、保存、等待与日志；
    本条记录之后的验证和最终补存不在该点内，不是完整运行的最终耗时。
    """
    metrics: dict[str, float] = {**{name: float(value) for name, value in losses.items()},
                               "lr_used": float(learning_rate), "lr_next": float(next_learning_rate),
                               "samples_seen": int(samples_seen), "elapsed_seconds": float(elapsed_seconds)}
    for name, source in UPDATE_METRIC_TIMINGS:
        metrics[name] = float(timings[source])
    for name in (*CPU_STAGE_TIMINGS, *DEEPSPEED_TIMERS, "reference_encoded_frames",
                 "train_rgb_frames", "train_latent_frames", "shot_count"):
        if name in timings:
            metrics[name] = float(timings[name])
    # 未同步的perf_counter只量到GPU提交，不能作为GPU阶段耗时上传。
    if timings.get("cuda_synchronized"):
        metrics.update({name: float(timings[name]) for name in GPU_STAGE_TIMINGS if name in timings})
    if "logger_step" in timings:
        metrics["checkpoint_hook_seconds"] = float(timings["logger_step"])
    metrics["training_step_seconds"] = metrics["step_seconds"] - float(timings.get("logger_step", 0.0))
    metrics["update_wall_seconds"] = metrics["step_seconds"] + metrics["data_seconds"]
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm
    return metrics


def global_update_statistics(accelerator: Any, loss_sums: Mapping[str, float], sample_count: int,
                             *, stop_requested: bool, invalid_stop: bool) -> tuple[dict[str, float], int, bool, bool]:
    """损失、样本分母及停止决定共用每更新一次的原有归约。"""
    import torch

    names = sorted(loss_sums)
    totals = accelerator.reduce(
        torch.tensor([*(loss_sums[name] for name in names), int(stop_requested), int(invalid_stop), sample_count],
                     dtype=torch.float64, device=accelerator.device), reduction="sum")
    *sums, stop, invalid, count = totals.tolist()
    return {name: value / count for name, value in zip(names, sums, strict=True)}, int(count), bool(stop), bool(invalid)
