"""Shared MSPC SFT training loop."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
if TYPE_CHECKING:
    from accelerate import Accelerator
from tqdm import tqdm

from mspc.training.loss_tracking import (
    CPU_STAGE_TIMINGS, GPU_STAGE_TIMINGS, DEEPSPEED_TIMERS, open_loss_tracking,
    reported_deepspeed_timings, reported_grad_norm, update_metrics, global_update_statistics,
)
from mspc.core.loss_curves import update_scalar_curves


def disable_hf_datasets_probe_for_torch_dataset() -> None:
    # The main-server runtime has datasets 2.14.4 with pyarrow 24.0.0. Accelerate only
    # probes HF datasets to special-case IterableDataset sharding; this trainer
    # uses a torch Dataset, so disable that unrelated import path.
    import accelerate.data_loader as accelerate_data_loader

    accelerate_data_loader.is_datasets_available = lambda: False


def dataloader_kwargs(args: Any) -> dict[str, Any]:
    per_device_batch_size = args.per_device_batch_size
    if type(per_device_batch_size) is not int or per_device_batch_size != 1:
        raise ValueError(
            "per_device_batch_size must be 1 because the current trainers "
            "process one complete variable-length sample per device"
        )
    num_workers = args.dataset_num_workers
    kwargs: dict[str, Any] = {
        "batch_size": per_device_batch_size,
        "drop_last": args.dataloader_drop_last,
        "num_workers": num_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
        kwargs["persistent_workers"] = args.dataloader_persistent_workers
    elif args.dataloader_persistent_workers:
        raise ValueError("dataloader.persistent_workers must be False when dataset_num_workers is 0")
    return kwargs


def collate_single_complete_sample(batch: list[Any]) -> Any:
    if len(batch) != 1:
        raise ValueError(
            "single-sample collator received "
            f"{len(batch)} samples; per_device_batch_size must remain 1"
        )
    return batch[0]


def data_order_generator(args: Any) -> torch.Generator:
    return torch.Generator().manual_seed(args.training_seed)


#: 一次一起按成本排序的更新步数。只在相邻 ``bucket_size`` 个位置里排序（2026-10-05 之前的做法）
#: 在 accum=1 时什么也不改——同一步的几个样本还是同一批随机样本；要让同一步各 rank 拿到长度接近的
#: 样本，必须先在更大的窗口里排序再切。窗口内各步的次序再打乱一次，所以相邻几步的长度不按大小递增。
COST_BUCKET_WINDOW_STEPS = 64


class CostBucketedSampler(torch.utils.data.Sampler):
    """确定性打乱，再在每 ``COST_BUCKET_WINDOW_STEPS × bucket_size`` 个连续位置内按成本分桶。

    ``per_device_batch_size=1`` 且 ``split_batches=false`` 时，accelerate 把打乱序列的
    第 i 个位置发给第 ``i % world_size`` 个 rank，所以「连续 ``bucket_size`` 个位置的样本
    成本接近」就等于「同一次更新里各 rank、各 microstep 的预测负载接近」。``bucket_size`` 取
    ``world_size * gradient_accumulation_steps``。

    做法：同一条 generator 先抽一个排列，**先去掉不准入的样本**，再在每个窗口内按成本稳定排序、切成
    ``bucket_size`` 一组，用同一条 generator 打乱组的次序。这只改**一个 epoch 内的次序**，不改哪些样本
    落在哪个 epoch，也不改 ``len``，所以 ``training.seed`` 仍然完全决定数据顺序。

    不准入的样本必须在分组之前去掉：11-16 定稿续训发现，分好组后再丢样本，丢一条后面所有位置错一位，
    此后每次更新都横跨两个成本不同的组，各 rank 等最慢的那个，每步慢约 2 秒。
    """

    def __init__(
        self,
        costs: Sequence[float | int],
        *,
        generator: torch.Generator,
        bucket_size: int,
        eligible_indices: Sequence[int] | None = None,
    ) -> None:
        if not costs:
            raise ValueError("CostBucketedSampler needs at least one sample cost")
        if type(bucket_size) is not int or bucket_size < 1:
            raise ValueError(f"bucket_size must be a positive int; got {bucket_size!r}")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in costs):
            raise ValueError("sample costs must be finite positive numbers")
        self.costs = tuple(float(value) for value in costs)
        self.generator = generator
        self.bucket_size = bucket_size
        eligible = tuple(range(len(costs))) if eligible_indices is None else tuple(eligible_indices)
        if (not eligible or any(type(index) is not int or not 0 <= index < len(costs) for index in eligible)
                or list(eligible) != sorted(set(eligible))):
            raise ValueError("eligible sample indices must be a nonempty sorted subset of the original cost indices")
        self.eligible_indices = frozenset(eligible)

    def __len__(self) -> int:
        return len(self.eligible_indices)

    def __iter__(self):
        # 排列仍对完整原始索引抽取（data_id 不重新编号），去掉不准入项后再分窗口、分组。
        order = [index for index in torch.randperm(len(self.costs), generator=self.generator).tolist()
                 if index in self.eligible_indices]
        window = COST_BUCKET_WINDOW_STEPS * self.bucket_size
        for start in range(0, len(order), window):
            chunk = sorted(order[start : start + window], key=lambda index: self.costs[index])
            full = len(chunk) - len(chunk) % self.bucket_size
            groups = [chunk[offset : offset + self.bucket_size] for offset in range(0, full, self.bucket_size)]
            for group_index in torch.randperm(len(groups), generator=self.generator).tolist():
                yield from groups[group_index]
            # 只有 epoch 最后一个窗口会剩不满一组的余数；放在最后，不让它把窗口里其余各组错位。
            yield from chunk[full:]


def training_sampler(
    dataset: Any, *, generator: torch.Generator, bucket_size: int,
    experiment_config: Mapping[str, Any] | None = None,
) -> torch.utils.data.Sampler:
    """唯一训练组批入口：显式标定负载、原有潜帧长度或无长度数据集的随机顺序。

    这是按**数据集能力**分派，不是给一条路径加兜底：联合段数据集
    （``mspc.training.block_dataset``：整段 2–77 潜帧、单镜行、单帧组长短差一个量级，
    长度差会让同一步里最慢的 rank 决定步时、其余 rank 在集合通信上空等）报出
    ``latent_frame_upper_bounds``。
    """

    config = {} if experiment_config is None else experiment_config
    eligible = getattr(dataset, "eligible_sample_indices", None)
    coefficients = config.get("training", {}).get("workload_cost")
    if coefficients is not None:
        fields = {"linear_seconds_per_latent", "attention_seconds_per_frame_pair"}
        if (not isinstance(coefficients, Mapping) or set(coefficients) != fields
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       for value in coefficients.values()) or sum(coefficients.values()) <= 0):
            raise ValueError("workload_cost requires finite nonnegative calibrated linear and attention coefficients")
        policy = config.get("model", {}).get("self_attention_policy")
        if policy not in ("holocine_first_frame", "first_frame_communication"):
            raise ValueError("calibrated workload costs require holocine_first_frame or "
                             "first_frame_communication self attention")
        workloads = getattr(dataset, "training_workloads", None)
        if not callable(workloads):
            raise ValueError("calibrated workload costs require dataset.training_workloads metadata")
        records = workloads(self_attention_policy=policy)
        if len(records) != len(dataset):
            raise ValueError("workload metadata must cover every dataset sample exactly once")
        costs = tuple(
            coefficients["linear_seconds_per_latent"] * row["latent_frames"]
            + coefficients["attention_seconds_per_frame_pair"] * row["attention_frame_pairs"]
            for row in records)
        return CostBucketedSampler(costs, generator=generator, bucket_size=bucket_size, eligible_indices=eligible)
    counts = getattr(dataset, "latent_frame_upper_bounds", None)
    base = (
        torch.utils.data.RandomSampler(dataset, generator=generator)
        if counts is None
        else CostBucketedSampler(counts(), generator=generator, bucket_size=bucket_size, eligible_indices=eligible)
    )
    return base


def training_epoch_count(args: Any, dataset: Any, *, updates_per_epoch: int) -> int:
    """固定排除后的尾部复用原sampler下一epoch，完整optimizer调度步数不缩短。"""
    epochs = args.num_epochs
    maximum = args.max_train_steps
    if (getattr(dataset, "excluded_sample_indices", ()) and maximum is not None
            and epochs * updates_per_epoch < maximum):
        if updates_per_epoch < 1:
            raise ValueError("text admission leaves no complete optimizer update")
        epochs = math.ceil(maximum / updates_per_epoch)
    return epochs


def build_lr_scheduler(
    optimizer: Any,
    args: Any,
    *,
    num_processes: int | None = None,
) -> Any:
    scheduler_type = getattr(args, "lr_scheduler_type", None)
    if scheduler_type is None:
        raise ValueError("lr_scheduler_type must be declared explicitly")
    if scheduler_type == "constant":
        factor = getattr(args, "lr_scheduler_factor", None)
        total_iters = getattr(args, "lr_scheduler_total_iters", None)
        if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
            raise ValueError("lr_scheduler_factor must be in (0, 1]")
        if type(total_iters) is not int or total_iters <= 0:
            raise ValueError("lr_scheduler_total_iters must be a positive integer")
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=float(factor),
            total_iters=total_iters,
        )
    if scheduler_type == "constant_with_warmup":
        return build_constant_with_warmup_scheduler(optimizer, args, num_processes)
    raise ValueError(f"unsupported lr_scheduler_type={scheduler_type!r}")


def build_constant_with_warmup_scheduler(
    optimizer: Any,
    args: Any,
    num_processes: int | None,
) -> Any:
    """Linear warmup to the configured learning rate, then constant.

    ``warmup_steps`` counts optimizer updates. Accelerate wraps a plain torch
    scheduler in ``AcceleratedScheduler``: with ``dataloader.split_batches``
    false (the only value this schema accepts) it advances the wrapped
    scheduler once per process on every synchronized update, and not at all on
    accumulation micro-steps. One optimizer update therefore moves
    ``last_epoch`` by ``num_processes``, so the ramp is scaled by the same
    factor to keep ``warmup_steps`` independent of world size.
    """

    warmup_steps = getattr(args, "warmup_steps", None)
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if type(num_processes) is not int or num_processes < 1:
        raise ValueError(
            "constant_with_warmup needs the prepared process count so "
            "warmup_steps stays denominated in optimizer updates"
        )
    ramp = warmup_steps * num_processes

    def warmup_factor(scheduler_step: int) -> float:
        if scheduler_step >= ramp:
            return 1.0
        return float(scheduler_step) / float(ramp)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)


def sync_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def should_log_timing(step: int, args: Any) -> bool:
    timing_log_steps = args.timing_log_steps
    timing_log_every = args.timing_log_every
    return (timing_log_steps > 0 and step <= timing_log_steps) or (
        timing_log_every > 0 and step % timing_log_every == 0
    )


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    return accelerator.unwrap_model(model)


def finite_scalar_loss_value(loss: Any) -> float:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("model loss must be a scalar torch.Tensor")
    value = float(loss.detach().float().item())
    if not math.isfinite(value):
        raise ValueError(f"model loss must be finite, got {value}")
    return value


class CudaStepPeaks:
    """逐步显存峰值：每个 microstep 开始时清零分配器峰值、结束时读回，写进本 rank 的 timing 行。

    11-11 孪生＋SAM 接续时，外部 nvidia-smi 记录脚本在第 3107、3192 步没采到样本，却把两个 7 镜样本记成
    峰值 0、「通过」（运行根 ``control/heavy-step-memory-skip-resume.json``；上一次接续同两条样本实测
    80,797／80,435 MiB）。训练进程自己逐步记下峰值，就不需要外部采样。清零前先把当前峰值并入全程峰值，
    所以 :func:`cuda_peak_memory_line` 报的仍是全程（含验证与存盘）峰值。读分配器计数不需要同步 GPU。
    """

    def __init__(self) -> None:
        self.allocated = 0
        self.reserved = 0

    def _fold(self) -> tuple[int, int]:
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        self.allocated, self.reserved = max(self.allocated, allocated), max(self.reserved, reserved)
        return allocated, reserved

    def start(self) -> None:
        if torch.cuda.is_available():
            self._fold()
            torch.cuda.reset_peak_memory_stats()

    def finish(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {}
        allocated, reserved = self._fold()
        return {"cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved}

    def run_peaks(self) -> tuple[int, int]:
        self._fold()
        return self.allocated, self.reserved


def cuda_peak_memory_line(prefix: str, accelerator: Accelerator, step_peaks: CudaStepPeaks) -> str:
    """本 rank 这张卡的全程显存峰值，一行。

    ``max_memory_allocated`` 是张量分配的峰值，``max_memory_reserved`` 是缓存分配器向驱动要的峰值——
    ``nvidia-smi`` 看到的是后者加上上下文开销，所以两个都报，读门时读 reserved。逐步峰值见
    :class:`CudaStepPeaks` 写进 ``timing.rank*.jsonl`` 的两列。
    """

    if not torch.cuda.is_available():
        return f"{prefix} rank={accelerator.process_index} cuda=unavailable"
    device = torch.cuda.current_device()
    scale = 1024.0**3
    allocated, reserved = step_peaks.run_peaks()
    return (
        f"{prefix} rank={accelerator.process_index} device=cuda:{device} "
        f"peak_allocated_gib={allocated / scale:.2f} "
        f"peak_reserved_gib={reserved / scale:.2f} "
        f"total_gib={torch.cuda.get_device_properties(device).total_memory / scale:.2f}"
    )


def forward_backward(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    data: Any,
    timings: dict[str, Any],
    timing_sync_cuda: bool,
) -> float:
    """一次前向、一次反向，返回这一步的 loss 值。

    **每 rank 每步恰好一次集合通信边界**，与这一条样本的内容无关。此前逐镜臂在这里把
    一条故事切成 K 个镜、逐镜前向再一次反向；该路径已被替换，一步现在就是一次前向、一次反向。

    **为什么这条不变式是硬要求**（实测由来）：一步一条故事而 K 逐样本不同时，逐镜反传会
    让每个 rank 每步发出的集合通信次数不一样——rank 0 拿到 K=2 的故事、rank 3 拿到 K=5
    的，两者就此永久错位。训练集群作业 1158279 正是这样死的：NCCL 在同一个 SeqNum=884 上超
    时，rank 0/6 停在 ``NumelIn=491569410`` 的梯度 allreduce，rank 1-5 停在 ``NumelIn=1``
    的标量 allreduce，600 秒看门狗到点后整组下线。这条是历史取证：出事的代码路径已被替
    换、作业也已消失，只能作为改口的由来引用，不能当成可复跑的断言。

    逐镜前向的可证伪性不受影响：自注意力只在镜内，跨镜身份因此只能经视觉计划这条支路
    到达——那由样本的切法保证，与反向做几次无关。
    """

    sync_cuda(timing_sync_cuda)
    timings["forward_started_unix_ns"] = time.time_ns()
    start = time.perf_counter()
    loss = model(data)
    sync_cuda(timing_sync_cuda)
    timings["forward"] = time.perf_counter() - start
    timings.update(raw_model.last_timing)
    loss_value = finite_scalar_loss_value(loss)
    timings["backward_started_unix_ns"] = time.time_ns()
    start = time.perf_counter()
    accelerator.backward(loss)
    sync_cuda(timing_sync_cuda)
    timings["backward"] = time.perf_counter() - start
    return loss_value


SUM_TIMING_FIELDS = frozenset((
    "data_wait", "forward", "backward", "optimizer_step", "scheduler_step", "zero_grad", "step_total",
    "dataset_video_decode", "dataset_caption", "dataset_total", "reference_encoded_frames",
    "reference_output_frames", "train_rgb_frames", "train_latent_frames", "shot_count",
    *CPU_STAGE_TIMINGS, *GPU_STAGE_TIMINGS, *DEEPSPEED_TIMERS,
))


#: 逐步显存峰值（字节）：一次更新内跨 microstep 取最大，不累加。
MAX_TIMING_FIELDS = frozenset(("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"))


def accumulate_update_timings(update: dict[str, Any], microstep: Mapping[str, Any]) -> None:
    """耗时和计数累加整个更新的microstep；显存峰值取最大；身份和逐样本诊断保留末条。"""
    for key, value in microstep.items():
        if key in SUM_TIMING_FIELDS:
            update[key] = update.get(key, 0.0) + value
        elif key in MAX_TIMING_FIELDS:
            update[key] = max(update.get(key, 0), value)
        else:
            update[key] = value


INT32_MAX = 2**31 - 1


def optimizer_groups_by_dtype(param_groups: Any) -> list[dict[str, Any]]:
    """每个学习率组内部按真实存储dtype分组，避免ZeRO flatten把BF16参数升回FP32。

    原组的超参数、dtype首次出现的顺序与各dtype内的参数顺序保持；一个参数只能出现一次。
    """
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, group in enumerate(param_groups):
        by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for parameter in group["params"]:
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"optimizer group {index} contains a non-tensor parameter")
            if id(parameter) in seen:
                raise ValueError(f"optimizer group {index} repeats a parameter")
            seen.add(id(parameter))
            by_dtype.setdefault(parameter.dtype, []).append(parameter)
        if not by_dtype:
            raise ValueError(f"optimizer group {index} is empty")
        for parameters in by_dtype.values():
            result.append({**group, "params": parameters})
    if not result:
        raise ValueError("optimizer has no parameter groups")
    return result


def require_zero_partitions_within_int32(param_groups: Any, world: int) -> None:
    """DeepSpeed ZeRO 把每个参数组的可训练参数按 world 平分，每个 rank 更新自己那一份；一份超过 2^31−1 个元素就拒绝起训。

    实测（训练集群 2026-10-12，5B 全参、ZeRO-2）：2 卡时每份约 25 亿个元素，第 1 次更新后各 rank 那一份开头的参数变成
    非有限值（作业 1248398、1248490、1248491），累积 2 时直接报 CUDA 非法内存访问（1248523）；4 卡每份约 12.5 亿，
    同一配置 4 步正常（1248524）。超过 32 位下标上限是推断出的原因，分界落在这两者之间是实测。
    """
    for index, group in enumerate(param_groups):
        elements = sum(parameter.numel() for parameter in group["params"] if parameter.requires_grad)
        if math.ceil(elements / world) > INT32_MAX:
            raise RuntimeError(
                f"optimizer parameter group {index} has {elements} trainable elements; split over {world} process(es) "
                f"each ZeRO partition exceeds {INT32_MAX} elements, which corrupts the first update. "
                f"Use at least {math.ceil(elements / INT32_MAX)} processes.")


def report_validation(accelerator: Accelerator, seconds: float, step: int) -> None:
    if accelerator.is_main_process:
        print(f"[validation] step={step} seconds={seconds:.1f}", flush=True)


def format_timing_line(prefix: str, timings: dict[str, Any]) -> str:
    parts = []
    for key, value in timings.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return f"{prefix} " + " ".join(parts)


PROGRESS_TIMING_FIELDS = (
    ("step", "step_total"),
    ("data", "data_wait"),
    ("decode", "dataset_video_decode"),
    ("forward", "forward"),
    ("backward", "backward"),
)
PROGRESS_SAMPLE_FIELDS = (
    ("rgb", "train_rgb_frames"),
    ("latent", "train_latent_frames"),
    ("shots", "shot_count"),
)
class ProgressHistory:
    """Rolling timing history; non-positive log_steps explicitly means all steps."""

    def __init__(self, timing_log_steps: int):
        if type(timing_log_steps) is not int:
            raise TypeError("timing_log_steps must be an integer")
        self.window_size = timing_log_steps if timing_log_steps > 0 else None
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.window_size)

    def metrics(
        self,
        *,
        loss_value: float,
        learning_rate: float,
        timings: dict[str, Any],
        latest_checkpoint_step: int | None,
    ) -> dict[str, str]:
        self._rows.append(timings)
        metrics = {
            "loss": f"{loss_value:.4g}",
            "lr": f"{learning_rate:.3e}",
        }
        for key in ("microstep", "microsteps_in_update"):
            if key in timings:
                metrics[key] = str(timings[key])
        for label, key in PROGRESS_TIMING_FIELDS:
            current = float(timings.get(key, 0.0))
            values = [float(row[key]) for row in self._rows if key in row]
            average = sum(values) / len(values) if values else current
            metrics[label] = f"{average:.2f}s ({current:.2f}s)"
        for label, key in PROGRESS_SAMPLE_FIELDS:
            if key in timings:
                metrics[label] = str(timings[key])
        if latest_checkpoint_step is not None:
            metrics["ckpt"] = str(latest_checkpoint_step)
        return metrics


def format_progress_line(step: int, metrics: Mapping[str, str]) -> str:
    return "[train_progress] " + " ".join(
        [f"step_id={step}", *(f"{key}={value}" for key, value in metrics.items())]
    )


class TrainableStateDictLogger:
    """本次运行自己的步数与 checkpoint 计数。

    ``CHECKPOINT`` 是 **weights-only warm restart**（v5.3 §12）：只载权重，
    optimizer / scheduler / RNG / data order 全部重置。所以步数也从 0 起——
    热启动 checkpoint 文件名里的步号是**它自己的来源**，不是本次运行的起点，
    它记在启动行的 ``checkpoint=<path>:step=<n>`` 里。

    训练集群训练不保存续跑用的优化器/调度器/RNG 状态，只保存
    训练时激活的可训练权重，避免磁盘占用过高。每次运行都是全新的 step 0 起点，
    中断后只能拿最近一份 ``step-N.safetensors`` 热启动，不能接着原来的轨迹走。
    """

    def __init__(
        self,
        output_path: str,
        remove_prefix_in_ckpt: str,
    ):
        if not output_path:
            raise ValueError("output_path is required")
        if not isinstance(remove_prefix_in_ckpt, str) or not remove_prefix_in_ckpt.strip():
            raise ValueError("remove_prefix_in_ckpt must be a non-empty string")
        self.output_path = Path(output_path)
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.num_steps = 0
        self.latest_checkpoint_step: int | None = None

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps: int) -> None:
        if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
            raise RuntimeError("checkpoint step requires a completed synchronized optimizer update")
        self.num_steps += 1
        if self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.latest_checkpoint_step = self.num_steps

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps: int) -> None:
        if self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.latest_checkpoint_step = self.num_steps

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name: str) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self._export_trainable_state_dict(accelerator, model, file_name)
            self.output_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.output_path / file_name
            try:
                accelerator.save(state_dict, checkpoint_path, safe_serialization=True)
            except BaseException:
                checkpoint_path.unlink(missing_ok=True)
                raise
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                checkpoint_path.unlink(missing_ok=True)
                raise RuntimeError(f"checkpoint was not written: {checkpoint_path}")

    def _export_trainable_state_dict(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        file_name: str,
    ) -> dict[str, Any]:
        unwrapped_model = accelerator.unwrap_model(model)
        raw_state_dict = unwrapped_model.state_dict()
        state_dict = unwrapped_model.export_trainable_state_dict(
            raw_state_dict,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        if len(state_dict) == 0:
            trainable_param_names = unwrapped_model.trainable_param_names()
            raw_sample = list(raw_state_dict.keys())[:5]
            trainable_sample = sorted(trainable_param_names)[:5]
            raise RuntimeError(
                "Refusing to save an empty trainable checkpoint "
                f"{file_name!r}: raw_state_dict_keys={len(raw_state_dict)}, "
                f"trainable_param_names={len(trainable_param_names)}, "
                f"raw_sample={raw_sample}, trainable_sample={trainable_sample}"
            )
        print(f"Saving trainable checkpoint {file_name}: tensors={len(state_dict)}", flush=True)
        return state_dict


def train_sft(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    model_logger: Any,
    args: Any,
    validation: Any = None,
) -> None:
    """``validation`` 是 :class:`mspc.training.validation_loss.ValidationLoss`（配置了 ``training.validation`` 时由 runner
    建好传进来）：训练开始前与每隔 ``every_steps`` 次更新，各 rank 在自己那份留出窗上各量一遍。
    可恢复停止点与训练终点强制量完整面板，再保存同一步的完整状态。"""
    # 模型决定各模块的学习率；同一学习率组再按真实dtype分开，ZeRO逐组flatten保持混合存储。
    optimizer_groups = optimizer_groups_by_dtype(model.optimizer_param_groups(args))
    from mspc.training.resume_migration import require_model_migration_layout, resume_source_config
    require_model_migration_layout(model, optimizer_groups, args)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        args,
        num_processes=accelerator.num_processes,
    )
    sampler_generator = data_order_generator(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=training_sampler(
            dataset,
            generator=sampler_generator,
            bucket_size=accelerator.num_processes * args.gradient_accumulation_steps,
            experiment_config=getattr(args, "experiment_config", None),
        ),
        collate_fn=collate_single_complete_sample,
        **dataloader_kwargs(args),
    )

    disable_hf_datasets_probe_for_torch_dataset()
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    raw_model = unwrap_model(accelerator, model)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    epochs = training_epoch_count(args, dataset, updates_per_epoch=updates_per_epoch)
    available_updates = epochs * updates_per_epoch
    if args.max_train_steps is not None and available_updates < args.max_train_steps:
        raise ValueError(f"dataloader provides {available_updates} updates, below max_train_steps={args.max_train_steps}")

    state_policy = getattr(args, "state_checkpoint", None)
    resume_path = getattr(args, "resume_state_path", None)
    training_progress = None
    if state_policy is not None:
        from mspc.training.training_state import TrainingProgress, inspect_training_state
        if getattr(dataset, "stateless_index_samples", False) is not True:
            raise ValueError("complete training state requires stateless per-index sample draws")
        training_progress = TrainingProgress()
        accelerator.register_for_checkpointing(training_progress)
        if resume_path is not None:
            recorded = inspect_training_state(resume_path, config=resume_source_config(args),
                                              world_size=accelerator.num_processes)
            from mspc.training.training_state import partial_module_state, require_partial_module_state
            partial = partial_module_state(args)
            if partial:
                require_partial_module_state(resume_path, raw_model)
            accelerator.load_state(resume_path, **({"load_module_strict": False} if partial else {}))
            if training_progress.completed_steps != recorded["completed_steps"]:
                raise ValueError("training-state progress and completion receipt disagree")
            model_logger.num_steps = training_progress.completed_steps
            exports = [int(path.stem.removeprefix("step-"))
                       for path in Path(args.output_path).glob("step-*.safetensors")]
            model_logger.latest_checkpoint_step = max(
                (step for step in exports if step <= training_progress.completed_steps), default=None)
    elif resume_path is not None:
        raise ValueError("resume-state requires an explicit training state checkpoint policy")
    completed_steps = model_logger.num_steps
    steps_per_epoch = len(dataloader)
    progress_history = ProgressHistory(args.timing_log_steps)
    progress_total = (
        args.max_train_steps
        if args.max_train_steps is not None
        # 总数是整条臂的同步优化器更新数，不是「已走的 + 还要走的」：``completed_steps``
        # 已经落在这几个 epoch 里面了，再加一次会把进度条的分母撑大。
        else args.num_epochs * math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    )
    progress = tqdm(
        total=progress_total,
        initial=completed_steps,
        disable=not accelerator.is_main_process,
        desc="Training",
        dynamic_ncols=True,
        file=args.progress_stream,
    )
    microsteps = 0 if training_progress is None else training_progress.microsteps
    update_microsteps = 0
    update_losses: dict[str, float] = {}
    samples_seen = 0 if training_progress is None else training_progress.samples_seen
    update_timings: dict[str, Any] = {}
    step_peaks = CudaStepPeaks()
    # 逐样本账：数据集给出 ``_sample_metadata`` 时，每 rank 每个 microstep 一行
    # ``<log_dir>/samples.rank{r}.jsonl``（step、样本元数据如 case_id/目标段/参考、这一步抽到的时间步、loss）。
    sample_log = None
    timing_log = None
    if args.log_dir is not None:
        timing_log = (Path(args.log_dir) / f"timing.rank{accelerator.process_index}.jsonl").open("a", encoding="utf-8")
    # 逐更新账：主进程每次优化器更新往 ``<log_dir>/metrics.jsonl`` 写一行（事实来源）；配置了 ``training.loss_tracking``
    # 时同一行按 train/timing/workload 分组再记进 SwanLab（只在主进程）。
    metric_log = tracker = None
    if accelerator.is_main_process and args.log_dir is not None:
        metric_log = (Path(args.log_dir) / "metrics.jsonl").open("a", encoding="utf-8")
        tracker = open_loss_tracking(args.loss_tracking, hyperparameters=args.experiment_config or {},
                                     log_dir=Path(args.log_dir))
    try:
        training_started = time.perf_counter()
        if validation is not None and resume_path is None:
            report_validation(accelerator, validation.run(raw_model, completed_steps), completed_steps)
        first_epoch = 0 if training_progress is None else training_progress.epoch
        for epoch_id in range(first_epoch, epochs):
            next_batch = 0
            if resume_path is not None and epoch_id == first_epoch:
                from mspc.training.training_state import resumed_epoch_batches
                sampler_generator.set_state(training_progress.epoch_sampler_state)
                next_batch = training_progress.next_batch
                if not 0 <= next_batch <= steps_per_epoch:
                    raise ValueError("resume-state batch position exceeds this dataloader")
                batches = resumed_epoch_batches(accelerator, dataloader, next_batch)
            else:
                if training_progress is not None:
                    training_progress.epoch_sampler_state = sampler_generator.get_state().clone()
                batches = dataloader
            last_step_end = time.perf_counter()
            for batch_id, data in enumerate(batches, start=next_batch):
                microsteps += 1
                update_microsteps += 1
                step = completed_steps + 1
                timing_enabled = should_log_timing(step, args)
                timing_sync_cuda = args.timing_sync_cuda and timing_enabled
                raw_model.profile_sync_cuda = timing_sync_cuda
                dataset_timing = data.pop("_dataset_timing", {}) if isinstance(data, dict) else {}
                sample_metadata = data.pop("_sample_metadata", None) if isinstance(data, dict) else None
                if getattr(args, "model_cfg", None) and args.model_cfg.get("text_context_cache") is not None:
                    if sample_metadata is None:
                        raise ValueError("cached text training requires the original sample metadata")
                    data["_text_cache_sample"] = {"split": "train", "metadata": sample_metadata}
                timings: dict[str, Any] = {
                    "rank": accelerator.process_index,
                    "step": step,
                    "microstep": microsteps,
                    "epoch": epoch_id,
                    "data_wait": time.perf_counter() - last_step_end,
                    "cuda_synchronized": timing_sync_cuda,
                }
                timings.update(dataset_timing)
                step_peaks.start()
                step_start = time.perf_counter()
                with accelerator.accumulate(model):
                    # DeepSpeed更新在backward内；必须在它之前读取本次使用的学习率。
                    learning_rate_used = float(optimizer.param_groups[0]["lr"])
                    loss_value = forward_backward(
                        accelerator=accelerator,
                        model=model,
                        raw_model=raw_model,
                        data=data,
                        timings=timings,
                        timing_sync_cuda=timing_sync_cuda,
                    )
                    if accelerator.sync_gradients:
                        timings.update(reported_deepspeed_timings(model))
                    if sample_metadata is not None:
                        if sample_log is None:
                            sample_log = (Path(args.log_dir) / f"samples.rank{accelerator.process_index}.jsonl").open(
                                "a", encoding="utf-8")
                        sample_log.write(json.dumps({"step": step, "microstep": microsteps, **sample_metadata,
                                                     **{k: timings[k] for k in ("timestep", "flow_draw_seed", "target_noise_probe", "target_frame_mse", "unweighted_mse", "target_weighted_mse", "uniform_loss", "loss_weight", "source_alignment_loss", "weighted_source_alignment_loss") if k in timings},
                                                     "timestep": timings.get("timestep"), "loss": loss_value},
                                                    ensure_ascii=False) + "\n")
                        sample_log.flush()
                    start = time.perf_counter()
                    optimizer.step()
                    sync_cuda(timing_sync_cuda)
                    timings["optimizer_step"] = time.perf_counter() - start
                    start = time.perf_counter()
                    scheduler.step()
                    timings["scheduler_step"] = time.perf_counter() - start
                    # Accelerate owns the microstep/no-sync behavior of both wrappers.
                    start = time.perf_counter()
                    optimizer.zero_grad()
                    sync_cuda(timing_sync_cuda)
                    timings["zero_grad"] = time.perf_counter() - start
                    timings["step_total"] = time.perf_counter() - step_start
                    timings.update(step_peaks.finish())
                    if timing_log is not None:
                        timing_log.write(json.dumps({**timings, "sample": sample_metadata,
                            "optimizer_updated": accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped}) + "\n")
                        timing_log.flush()
                    micro_losses = {"loss": loss_value}
                    if "uniform_loss" in timings:
                        micro_losses.update({name: float(timings[name]) for name in
                                             ("uniform_loss", "target_weighted_mse", "unweighted_mse")})
                    for name, value in micro_losses.items():
                        update_losses[name] = update_losses.get(name, 0.0) + value
                    accumulate_update_timings(update_timings, timings)
                    last_step_end = time.perf_counter()
                    if not accelerator.sync_gradients:
                        continue
                    if accelerator.optimizer_step_was_skipped:
                        update_losses = {}
                        update_microsteps = 0
                        update_timings = {}
                        continue
                    logger_start = time.perf_counter()
                    model_logger.on_step_end(accelerator, model, args.save_steps)
                    sync_cuda(timing_sync_cuda)
                    update_timings["logger_step"] = time.perf_counter() - logger_start
                    update_timings["step_total"] += update_timings["logger_step"]
                    update_timings["microsteps_in_update"] = update_microsteps
                    timings = update_timings
                    stop_request, invalid_stop = None, False
                    if accelerator.is_main_process and getattr(args, "run_root", None) is not None:
                        from mspc.training.training_state import read_training_stop_request
                        try:
                            stop_request = read_training_stop_request(
                                Path(args.run_root), run_id=args.run_id, state_enabled=state_policy is not None,
                                completed_steps=step)
                        except (ValueError, OSError) as exc:
                            print(f"[train] invalid stop request: {exc}", flush=True)
                            invalid_stop = True
                    losses, global_samples, stopping, invalid_stop = global_update_statistics(
                        accelerator, update_losses, update_microsteps,
                        stop_requested=stop_request is not None, invalid_stop=invalid_stop)
                    if invalid_stop:
                        raise ValueError("invalid recoverable stop request; no stop state acknowledged")
                    loss_value = losses["loss"]
                    samples_seen += global_samples
                    update_losses = {}
                    update_microsteps = 0
                    update_timings = {}
                    progress_metrics = progress_history.metrics(
                        loss_value=loss_value,
                        learning_rate=learning_rate_used,
                        timings=timings,
                        latest_checkpoint_step=getattr(
                            model_logger, "latest_checkpoint_step", None
                        ),
                    )
                    if accelerator.is_main_process:
                        progress.set_postfix(progress_metrics, refresh=False)
                        progress.write(
                            format_progress_line(step, progress_metrics),
                            file=args.progress_stream,
                        )
                        if metric_log is not None:
                            scalars = update_metrics(
                                losses=losses, learning_rate=learning_rate_used,
                                next_learning_rate=float(optimizer.param_groups[0]["lr"]), timings=timings,
                                samples_seen=samples_seen, elapsed_seconds=time.perf_counter() - training_started,
                                grad_norm=reported_grad_norm(model))
                            metric_log.write(json.dumps({"step": step, **scalars}) + "\n")
                            metric_log.flush()
                            if tracker is not None:
                                tracker.log(step, update_scalar_curves(scalars))
                    if timing_enabled and accelerator.is_local_main_process:
                        print(format_timing_line(args.timing_prefix, timings), flush=True)
                    completed_steps += 1
                    progress.update(1)
                    final_validation = stopping or completed_steps == args.max_train_steps
                    if validation is not None and (final_validation or validation.due(completed_steps)):
                        report_validation(accelerator, validation.run(
                            raw_model, completed_steps, full=final_validation), completed_steps)
                    if training_progress is not None:
                        training_progress.completed_steps = completed_steps
                        training_progress.microsteps = microsteps
                        training_progress.samples_seen = samples_seen
                        training_progress.epoch = epoch_id
                        training_progress.next_batch = batch_id + 1
                        if completed_steps % args.save_steps == 0 or stopping:
                            from mspc.training.training_state import save_training_state
                            saved_state = save_training_state(accelerator, training_progress, args)
                    if stopping:
                        # 停点同时导出推理权重（不在周期存盘点上才另存一份）：11-17 两组停在第 6,661／6,688 步，
                        # 只有完整状态、没有可直接出片的权重。
                        model_logger.on_training_end(accelerator, model, args.save_steps)
                        from mspc.training.training_state import complete_training_stop, TrainingStopped
                        receipt = complete_training_stop(
                            accelerator, state_path=saved_state, progress=training_progress,
                            args=args, request=stop_request)
                        raise TrainingStopped(receipt)
                    last_step_end = time.perf_counter()
                    # 第一步之后打一次：那一步已经走过完整的前向、反向与优化器状态
                    # 分配，是「这条配方在这张卡上要多少」的第一份真读数，不必等
                    # 3000 步跑完才知道是不是超门。
                    if step == 1:
                        print(
                            cuda_peak_memory_line(args.timing_prefix, accelerator, step_peaks),
                            flush=True,
                        )
                    if args.max_train_steps is not None and completed_steps >= args.max_train_steps:
                        print(
                            cuda_peak_memory_line(args.timing_prefix, accelerator, step_peaks),
                            flush=True,
                        )
                        model_logger.on_training_end(accelerator, model, args.save_steps)
                        if training_progress is not None and completed_steps % args.save_steps:
                            from mspc.training.training_state import save_training_state
                            save_training_state(accelerator, training_progress, args)
                        return
    finally:
        progress.close()
        if sample_log is not None:
            sample_log.close()
        if timing_log is not None:
            timing_log.close()
        if metric_log is not None:
            metric_log.close()
        if tracker is not None:
            tracker.finish()
    if args.max_train_steps is not None and completed_steps != args.max_train_steps:
        raise RuntimeError(f"training exhausted at {completed_steps} updates, expected {args.max_train_steps}")
    print(cuda_peak_memory_line(args.timing_prefix, accelerator, step_peaks), flush=True)
    model_logger.on_training_end(accelerator, model, args.save_steps)
    if training_progress is not None and completed_steps % args.save_steps:
        from mspc.training.training_state import save_training_state
        save_training_state(accelerator, training_progress, args)
