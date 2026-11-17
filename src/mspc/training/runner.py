"""Canonical, backend-lazy MSPC SFT training runner."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Sequence

from mspc.config import ExperimentConfig, load_experiment_config
from mspc.config.experiment import WAN22_BLOCK_DATASET_TYPES
from mspc.runs.launch import require_train_launch_dataset
from mspc.training.checkpoint import TrainingCheckpoint, inspect_training_checkpoint
from mspc.core.component_registry import resolve_training_components
from mspc.training.training_args import TrainingArgs, build_common_training_args, replace_training_args
from mspc.training.resume_migration import resume_source_config


class _TeeTextStream:
    """Minimal text stream used by the human log and future loop progress."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self._streams)


class _NullTextStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repository_sha() -> str:
    value = os.environ.get("MSPC_REPOSITORY_SHA")
    if value is not None and (len(value) != 40 or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("MSPC_REPOSITORY_SHA must be a lowercase 40-character Git SHA")
    from mspc.runs.repository import require_clean_worktree

    return require_clean_worktree(Path(__file__).resolve().parents[3], expected_sha=value)


def _runtime_run_id(config: ExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{config.resolved['identity']['run_name']}__{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rebind_logging_stream(
    source: Any, target: Any
) -> tuple[logging.Handler, ...]:
    rebound: list[logging.Handler] = []
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if getattr(handler, "stream", None) is source:
                _set_logging_stream(handler, target)
                rebound.append(handler)
    return tuple(rebound)


def _set_logging_stream(handler: logging.Handler, target: Any) -> None:
    """Replace a handler stream without flushing a possibly invalid temporary."""

    source = getattr(handler, "stream", None)
    if source is target:
        return

    handler.acquire()
    try:
        handler.stream = target
    finally:
        handler.release()


def _reserve_run_root(
    accelerator: Any,
    accelerate_utils: Any,
    config: ExperimentConfig,
    args: TrainingArgs,
    checkpoint: TrainingCheckpoint | None,
) -> tuple[Path, TrainingArgs]:
    """新运行独占新目录；显式resume复用原目录并核整份配置及拓扑。"""
    result: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            batch_contract = _training_batch_contract(
                config,
                accelerator.num_processes,
                gpus_per_node=args.gpus_per_node,
                nnodes=args.nnodes,
            )
            resumed = args.resume_state_path is not None
            if resumed:
                from mspc.training.training_state import inspect_training_state
                state_config = config.to_dict() if args.resume_migration is None else resume_source_config(args)
                state = inspect_training_state(args.resume_state_path, config=state_config,
                                               world_size=accelerator.num_processes)
                run_id, run_root = state["run_id"], Path(state["run_root"])
                snapshot = json.loads((run_root / "run-snapshot.json").read_text())
                source_snapshot = snapshot
                if args.resume_migration is not None:
                    from mspc.training.resume_migration import (
                        archive_migration_source_snapshot, migration_source_snapshot,
                    )
                    source_snapshot = migration_source_snapshot(args.resume_migration, snapshot)
                    archive_migration_source_snapshot(args.resume_migration, source_snapshot)
                if source_snapshot["config"] != state_config or snapshot["batch_contract"] != batch_contract:
                    raise ValueError("resume run snapshot config or batch topology differs")
                if snapshot["effective_topology"] != {
                        "gpus_per_node": args.gpus_per_node, "nnodes": args.nnodes,
                        "world_size": accelerator.num_processes}:
                    raise ValueError("resume run node/GPU topology differs")
                source_sha = (_repository_sha() if args.resume_migration is None
                              else args.resume_migration["from_repository_sha"])
                if source_snapshot["repository_sha"] != source_sha:
                    raise ValueError("resume run code revision differs")
                if args.max_train_steps is not None and state["completed_steps"] >= args.max_train_steps:
                    raise ValueError("resume-state already reached the configured training target")
                from mspc.training.training_state import archive_future_training_artifacts
                artifact_archive = archive_future_training_artifacts(
                    run_root, completed_steps=state["completed_steps"], run_id=run_id)
            else:
                run_id = _runtime_run_id(config)
                run_root = Path(args.output_root) / run_id
            output_path = run_root / "checkpoints"
            log_dir = run_root / "logs"
            fresh_snapshot = {
                "artifact_type": "mspc_training_run_snapshot",
                "schema_version": 1,
                "status": "running",
                "started_at": _now_iso(),
                "repository_sha": _repository_sha(),
                "source_config": str(config.source_path),
                "training_seed": args.training_seed,
                "rank_seed_rule": "training_seed + process_index",
                "world_size": accelerator.num_processes,
                "effective_topology": {
                    "gpus_per_node": args.gpus_per_node,
                    "nnodes": args.nnodes,
                    "world_size": args.gpus_per_node * args.nnodes,
                },
                "batch_contract": batch_contract,
                "config": config.to_dict(),
                "cache_stage": args.cache_stage_plan,
                "train_text_density": (
                    None
                    if config.resolved["task"] is None
                    else config.resolved["task"].get("train_text_density")
                ),
                "runtime": {
                    "run_id": run_id,
                    "run_root": str(run_root),
                    "output_path": str(output_path),
                    "log_dir": str(log_dir),
                    "checkpoint": (
                        None if checkpoint is None else checkpoint.to_dict()
                    ),
                },
            }
            if resumed:
                snapshot.update(status="running", resumed_at=_now_iso(), resume_state=args.resume_state_path,
                                cache_stage=args.cache_stage_plan)
                snapshot["resumed_artifact_archive"] = artifact_archive
                if args.resume_migration is not None:
                    snapshot.update(repository_sha=_repository_sha(), source_config=str(config.source_path),
                                    config=config.to_dict(), resume_migration=args.resume_migration)
                snapshot.pop("stop", None)
                snapshot.pop("stopped_at", None)
            else:
                snapshot = fresh_snapshot
                run_root.mkdir(parents=True, exist_ok=False)
                output_path.mkdir(parents=True, exist_ok=False)
                log_dir.mkdir(parents=True, exist_ok=False)
                (log_dir / ".spool").mkdir()
                (run_root / "rank-status").mkdir()
            _write_json(
                run_root / "run-snapshot.json",
                snapshot,
            )
            result[0] = {
                "ok": True,
                "run_id": run_id,
                "run_root": str(run_root),
                "output_path": str(output_path),
                "log_dir": str(log_dir),
                "repository_sha": snapshot["repository_sha"],
                "resume_migration_provenance": snapshot.get("resume_migration"),
            }
        except Exception as exc:
            result[0] = {
                "ok": False,
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
    accelerate_utils.broadcast_object_list(result, from_process=0)
    if not result[0] or not result[0].get("ok"):
        failure = result[0] or {"exception_type": "RuntimeError", "detail": "missing reservation result"}
        raise RuntimeError(
            f"training run root reservation failed: {failure['exception_type']}: {failure['detail']}"
        )
    accelerator.wait_for_everyone()
    updated_args = replace_training_args(
        args,
        run_id=result[0]["run_id"],
        run_root=result[0]["run_root"],
        output_path=result[0]["output_path"],
        log_dir=result[0]["log_dir"],
        repository_sha=result[0]["repository_sha"],
        resume_migration_provenance=result[0]["resume_migration_provenance"],
    )
    return Path(updated_args.run_root), updated_args


def _update_run_snapshot(run_root: Path, **updates: Any) -> None:
    path = run_root / "run-snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    _write_json(path, payload)


def _open_rank_spool_log(spool_dir: Path, process_index: int) -> tuple[Path, Any]:
    """给本 rank 开一份独占的原始日志，同一个 run root 可以开好几次。

    独占创建（``"x"``）是有意的：两个进程写同一个 rank 的日志会把 traceback 搅在
    一起，那正是这份 spool 唯一要保住的东西。但它当初是按「一个 run root 只跑一次」
    写的，续跑违反了这个前提——2026-08-31 的第二段验证就是在这里以
    ``FileExistsError`` 退出的，训练一步都没跑。

    改成按尝试次数取名：全新一跑仍是 ``rank-00000.log``（读取侧
    ``tools/run/run_status.py`` 用 ``rank-*.log`` 通配，后续尝试照样被收进去），
    续跑落到 ``rank-00000.attempt-01.log``。独占语义一点没松：谁先创建成功谁拥有
    这个名字，另一个进程会继续往下找。
    """

    attempt = 0
    while True:
        name = (
            f"rank-{process_index:05d}.log"
            if attempt == 0
            else f"rank-{process_index:05d}.attempt-{attempt:02d}.log"
        )
        path = spool_dir / name
        try:
            return path, path.open("x", encoding="utf-8", buffering=1)
        except FileExistsError:
            attempt += 1


def _open_human_log(log_dir: Path) -> Any:
    """开这条臂那份人读的日志；续跑接着往下写。

    只有主进程写它，而且它是**整条臂的叙事**——续跑接上去才读得连贯。这里不做独占
    创建：全新一跑时 ``run_root.mkdir(exist_ok=False)`` 已经保证目录是新的、文件不
    存在，``"x"`` 没有多保住什么；而续跑回到同一个 run root 时它会直接
    ``FileExistsError``，2026-08-31 的第三段验证就死在这里，一步没跑。

    与 :func:`_open_rank_spool_log` 处理得不同是有意的：那一份是**每个 rank 的原始
    stderr**，八个进程各写各的，按尝试分文件才不会把两次尝试的 traceback 读串。
    """

    return (log_dir / "train.log").open("a", encoding="utf-8", buffering=1)


def _write_rank_status(run_root: Path, rank: int, payload: dict[str, Any]) -> None:
    _write_json(
        run_root / "rank-status" / f"rank-{rank:05d}.json",
        {"rank": rank, **payload},
    )


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _training_batch_contract(
    config: ExperimentConfig,
    world_size: int,
    *,
    gpus_per_node: int,
    nnodes: int,
) -> dict[str, int | bool]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("accelerator world_size must be a positive integer")
    data = config.resolved
    gpus_per_node, nnodes = _validate_runtime_topology(
        gpus_per_node=gpus_per_node,
        nnodes=nnodes,
    )
    expected_world_size = nnodes * gpus_per_node
    if world_size != expected_world_size:
        raise ValueError(
            f"launched world_size={world_size} does not match effective "
            f"nnodes*gpus_per_node={nnodes}*{gpus_per_node}={expected_world_size}"
        )
    per_device_batch_size = data["dataloader"]["per_device_batch_size"]
    gradient_accumulation_steps = data["training"]["gradient_accumulation_steps"]
    global_micro_batch_size = per_device_batch_size * world_size
    return {
        "per_device_batch_size": per_device_batch_size,
        "split_batches_across_processes": data["dataloader"]["split_batches"],
        "drop_last_incomplete_global_batch": data["dataloader"]["drop_last"],
        "data_parallel_world_size": world_size,
        "global_micro_batch_size": global_micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_optimizer_batch_size": (
            global_micro_batch_size * gradient_accumulation_steps
        ),
    }


def _validate_runtime_topology(
    *,
    gpus_per_node: int,
    nnodes: int,
) -> tuple[int, int]:
    """Validate the explicit topology propagated by the launch owner."""

    if type(gpus_per_node) is not int or gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be a positive integer")
    if type(nnodes) is not int or nnodes <= 0:
        raise ValueError("nnodes must be a positive integer")
    return gpus_per_node, nnodes


def _human_start_lines(
    config: ExperimentConfig,
    args: Any,
    accelerator: Any,
    raw_log_path: Path,
    checkpoint: TrainingCheckpoint | None,
) -> tuple[str, ...]:
    data = config.to_dict()
    identity = data["identity"]
    model = data["model"]
    dataset = data["dataset"]
    task = data["task"]
    training = data["training"]
    optimization = data["optimization"]
    dataloader = data["dataloader"]
    batch = _training_batch_contract(
        config,
        accelerator.num_processes,
        gpus_per_node=args.gpus_per_node,
        nnodes=args.nnodes,
    )
    model_type = model["type"]
    model_name = model_type
    # Readable optimization keys follow lr_scheduler_type, so the banner has to
    # branch on the same type instead of assuming the ConstantLR keys exist.
    scheduler_type = optimization.get("lr_scheduler_type")
    if scheduler_type == "constant":
        scheduler_text = (
            f"{optimization['lr_scheduler_type']}:"
            f"factor={optimization['lr_scheduler_factor']}:"
            f"iters={optimization['lr_scheduler_total_iters']}"
        )
    elif scheduler_type == "constant_with_warmup":
        scheduler_text = f"{scheduler_type}:warmup_steps={optimization['warmup_steps']}"
    elif scheduler_type is None:
        scheduler_text = "legacy_constant_default"
    else:
        raise ValueError(f"unsupported lr_scheduler_type={scheduler_type!r}")

    persistent = _compact_json(dataloader["persistent_workers"])
    pin_memory = _compact_json(dataloader["pin_memory"])
    checkpoint_value = "none"
    if checkpoint is not None:
        checkpoint_value = f"{checkpoint.path}:step={checkpoint.step}"

    if dataset["type"] in WAN22_BLOCK_DATASET_TYPES:
        target_source = (f"targets={dataset['target_index_path']}" if dataset["type"] == "block_latent_cache"
                         else "targets=online_frozen_vae")
        dataset_line = (
            f"members={dataset['members_path']} {target_source} "
            f"sample_budget={args.dataset_cfg['sample_budget']} capacity={dataset['segment_capacity']} "
            f"sampling={dataset['sampling_manifest_path']}"
        )
    else:
        dataset_line = (
            f"metadata={dataset['metadata_path']} repeat={dataset['repeat']} "
            f"video={task['frame_policy']},{task['width']}x{task['height']},"
            f"max_rgb={task['max_rgb_frames']},fps={task['target_fps']},"
            f"vae_stride={task['vae_temporal_stride']}"
        )

    return (
        f"[train] START run={args.run_id} config={config.source_path}",
        f"[train] COMPONENTS model={model_type}:{model_name} "
        f"dataset={dataset['type']} trainer={data['trainer']} checkpoint={checkpoint_value} "
        f"precision={training['mixed_precision']} "
        f"{dataset_line}",
        f"[train] TRAIN_DISTRIBUTED lr={optimization['learning_rate']} "
        f"scheduler={scheduler_text} "
        f"epochs={training['num_epochs']} "
        f"target_steps={training['max_train_steps']} "
        f"grad_accum={training['gradient_accumulation_steps']} "
        f"per_device_batch={batch['per_device_batch_size']} "
        f"global_micro_batch={batch['global_micro_batch_size']} "
        f"effective_batch={batch['effective_optimizer_batch_size']} "
        f"split_batches={_compact_json(dataloader['split_batches'])} "
        f"drop_last={_compact_json(dataloader['drop_last'])} "
        f"save_steps={training['save_steps']} "
        f"world={accelerator.num_processes} "
        f"topology={args.nnodes}x{args.gpus_per_node} "
        f"workers={dataloader['dataset_num_workers']} "
        f"prefetch={dataloader['prefetch_factor']} persistent={persistent} "
        f"pin_memory={pin_memory}",
        f"[train] PATHS run={args.run_root} output={args.output_path} "
        f"log={args.log_dir} raw={raw_log_path}",
    )


def _record_training_failure(
    *,
    exc: BaseException,
    accelerator: Any,
    args: Any,
    run_root: Path,
    rank_seed: int,
    raw_log_path: Path,
    log_stream: Any,
    human_stderr: Any,
) -> None:
    failed_at = _now_iso()
    traceback.print_exc(file=log_stream)
    _write_rank_status(
        run_root,
        accelerator.process_index,
        {
            "status": "failed",
            "failed_at": failed_at,
            "training_seed": args.training_seed,
            "rank_seed": rank_seed,
            "exception_type": type(exc).__name__,
            "detail": str(exc),
        },
    )
    if accelerator.is_main_process:
        _update_run_snapshot(
            run_root,
            status="failed",
            failed_at=failed_at,
            failure={
                "rank": accelerator.process_index,
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            },
        )
        print(
            f"[train] FAILED type={type(exc).__name__} detail={exc} raw={raw_log_path}",
            file=human_stderr,
            flush=True,
        )


def _load_block_vae(dataset_config: Any, model: Any, dataset: Any, validation: Any) -> None:
    """同一冻结VAE负责在线target和reference；离线target仍严格核来源与几何。"""
    online_targets = dataset_config["type"] == "block_video"
    if not online_targets and not dataset.references_enabled:
        return
    provenance = None if online_targets else Path(dataset_config["cache_provenance_path"])
    model.load_frozen_vae(provenance)
    vae = model.reference_runtime.pipe.vae
    channels, stride = vae.z_dim, vae.upsampling_factor
    if type(channels) is not int or channels < 1 or type(stride) is not int or stride < 1:
        raise ValueError("frozen VAE has invalid latent geometry")
    selected_datasets = (dataset,) if validation is None else (dataset, validation.dataset)
    for selected in selected_datasets:
        height, width = selected.height, selected.width
        if (type(height) is not int or height < 1 or type(width) is not int or width < 1
                or height % stride or width % stride):
            raise ValueError("target data has invalid geometry for frozen VAE")
        if online_targets:
            selected.bind_online_vae_geometry(channels, stride)
        elif selected.target_encoding != model.reference_encoding:
            raise ValueError("frozen reference VAE/encoding differs from target cache")
        if (selected.latent_channels != channels or selected.vae_spatial_stride != stride
                or selected.latent_spatial_shape != (height // stride, width // stride)):
            raise ValueError("frozen VAE latent geometry differs from target data")


def create_accelerator(config: ExperimentConfig, args: TrainingArgs):
    """训练配置直接构造原生插件；启动器不再传第二份优化参数。"""
    from datetime import timedelta
    import accelerate
    from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, GradientAccumulationPlugin

    engine_config = config.to_dict()["runtime"]["deepspeed"]
    engine_config.update(
        train_batch_size=args.gpus_per_node * args.nnodes * args.per_device_batch_size * args.gradient_accumulation_steps,
        train_micro_batch_size_per_gpu=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    return accelerate.Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=engine_config, enable_msamp=False),
        mixed_precision=args.mixed_precision,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps),
        dataloader_config=accelerate.DataLoaderConfiguration(
            split_batches=args.dataloader_split_batches,
            even_batches=True,
        ),
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=config.resolved["runtime"]["distributed_timeout_seconds"])),
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )


def _record_training_stop(*, accelerator: Any, run_root: Path, receipt: dict[str, Any],
                          human_stdout: Any) -> None:
    stopped_at = _now_iso()
    _write_rank_status(run_root, accelerator.process_index,
                       {"status": "stopped", "stopped_at": stopped_at, "stop": receipt})
    if accelerator.is_main_process:
        _update_run_snapshot(run_root, status="stopped", stopped_at=stopped_at, stop=receipt)
        print(f"[train] STOPPED step={receipt['completed_steps']} state={receipt['state_path']}",
              file=human_stdout, flush=True)


def _archive_resumed_training_logs(*, accelerator: Any, accelerate_utils: Any,
                                   args: TrainingArgs, config: ExperimentConfig) -> None:
    from mspc.training.training_state import archive_training_log_tail, inspect_training_state

    failure = None
    try:
        state_config = config.to_dict() if args.resume_migration is None else resume_source_config(args)
        state = inspect_training_state(args.resume_state_path, config=state_config,
                                       world_size=accelerator.num_processes)
        archive_training_log_tail(Path(args.log_dir), completed_steps=state["completed_steps"],
                                  rank=accelerator.process_index,
                                  is_main_process=accelerator.is_main_process)
    except BaseException as exc:
        failure = {"rank": accelerator.process_index, "exception_type": type(exc).__name__,
                   "detail": str(exc)}
    # Each rank owns separate files. A rank-local read/rename failure must reach
    # every rank before any process enters model initialization or a barrier.
    failures = [item for item in accelerate_utils.gather_object([failure]) if item is not None]
    if failures:
        details = "; ".join(f"rank={item['rank']} {item['exception_type']}: {item['detail']}"
                            for item in sorted(failures, key=lambda item: item["rank"]))
        raise RuntimeError(f"resume training log archive failed: {details}")


def run_training(
    config: ExperimentConfig,
    checkpoint_override: str | Path | None = None,
    *,
    resume_state: str | Path | None = None,
    resume_migration: str | Path | None = None,
    cache_stage: str | Path | None = None,
    gpus_per_node: int,
    nnodes: int,
) -> int:
    """Execute one validated training configuration.

    ``checkpoint_override`` 仍是步数0的权重热启动。``resume_state`` 只接受声明了
    完整状态保存策略的同一配置、同一拓扑运行，不将旧权重文件当作续训状态。
    """

    require_train_launch_dataset(config)
    terminal_stdout = sys.stdout
    terminal_stderr = sys.stderr
    with tempfile.SpooledTemporaryFile(
        mode="w+", encoding="utf-8", max_size=1024 * 1024
    ) as bootstrap_log:
        with redirect_stdout(bootstrap_log), redirect_stderr(bootstrap_log):
            checkpoint = inspect_training_checkpoint(
                config.resolved["checkpoint"], checkpoint_override
            )
            components = resolve_training_components(config)
            args = build_common_training_args(config)
            if cache_stage is not None:
                from mspc.training.cache_readiness import read_cache_stage
                if config.resolved["dataset"]["type"] != "block_latent_cache":
                    raise ValueError("--cache-stage requires a production block_latent_cache dataset")
                args = replace_training_args(args, cache_stage_plan=read_cache_stage(cache_stage))
            if resume_state is not None:
                if checkpoint is not None:
                    raise ValueError("resume-state and weights-only checkpoint are mutually exclusive")
                if args.state_checkpoint is None:
                    raise ValueError("resume-state requires explicit training.state_checkpoint policy")
                args = replace_training_args(args, resume_state_path=str(Path(resume_state).resolve(strict=True)))
            gpus_per_node, nnodes = _validate_runtime_topology(
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            from mspc.training.training_args import configure_training_topology
            args = configure_training_topology(args, gpus_per_node=gpus_per_node, nnodes=nnodes)
            if resume_migration is not None:
                if args.resume_state_path is None:
                    raise ValueError("resume-migration requires an explicit complete resume-state")
                # 续训到新停点：两种缓存都覆盖新阶段（停点须晚于已训步数，由 require_cache_stage 核）。
                if args.cache_stage_plan is None:
                    raise ValueError("resume-migration requires an explicit cache stage for the new stop point")
                from mspc.training.resume_migration import read_resume_migration
                contract = read_resume_migration(resume_migration, state_path=args.resume_state_path,
                    target_config=config.to_dict(), repository_sha=_repository_sha(),
                    world_size=gpus_per_node * nnodes, target_config_path=config.source_path)
                args = replace_training_args(args, resume_migration=contract)
            args = components.model.configure_args(args, config, checkpoint)

            import accelerate
            from accelerate.utils import DistributedType
            from mspc.training.loop import (
                TrainableStateDictLogger, optimizer_groups_by_dtype, require_zero_partitions_within_int32,
            )

            accelerator = create_accelerator(config, args)
            if accelerator.num_processes != gpus_per_node * nnodes:
                accelerator.end_training()
                raise ValueError("actual process count differs from the requested training topology")
            accelerate.utils.set_seed(args.training_seed, device_specific=True)
            try:
                run_root, args = _reserve_run_root(
                    accelerator,
                    accelerate.utils,
                    config,
                    args,
                    checkpoint,
                )
            except BaseException:
                accelerator.end_training()
                raise

        bootstrap_handlers = _rebind_logging_stream(
            bootstrap_log, terminal_stderr
        )

        bootstrap_log.seek(0)
        bootstrap_text = bootstrap_log.read()

    log_dir = Path(args.log_dir)
    rank_seed = args.training_seed + accelerator.process_index

    try:
        if args.resume_state_path is not None:
            _archive_resumed_training_logs(accelerator=accelerator, accelerate_utils=accelerate.utils,
                                           args=args, config=config)
        raw_log_path, raw_log_stream = _open_rank_spool_log(
            log_dir / ".spool", accelerator.process_index
        )
    except BaseException as exc:
        failed_at = _now_iso()
        _write_rank_status(
            run_root,
            accelerator.process_index,
            {
                "status": "failed",
                "failed_at": failed_at,
                "training_seed": args.training_seed,
                "rank_seed": rank_seed,
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            },
        )
        if accelerator.is_main_process:
            _update_run_snapshot(
                run_root,
                status="failed",
                failed_at=failed_at,
                failure={
                    "rank": accelerator.process_index,
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            print(
                f"[train] FAILED type={type(exc).__name__} detail={exc} "
                f"raw_unavailable={log_dir / '.spool'}",
                file=terminal_stderr,
                flush=True,
            )
        accelerator.end_training()
        return 1

    with ExitStack() as stream_stack:
        log_stream = stream_stack.enter_context(raw_log_stream)
        for handler in bootstrap_handlers:
            _set_logging_stream(handler, log_stream)
            stream_stack.callback(_set_logging_stream, handler, terminal_stderr)
        log_stream.write(bootstrap_text)
        human_stdout: Any = _NullTextStream()
        human_stderr: Any = terminal_stderr if accelerator.is_main_process else human_stdout
        try:
            _write_rank_status(
                run_root,
                accelerator.process_index,
                {
                    "status": "running",
                    "started_at": _now_iso(),
                    "training_seed": args.training_seed,
                    "rank_seed": rank_seed,
                },
            )
            if accelerator.is_main_process:
                human_log_stream = stream_stack.enter_context(_open_human_log(log_dir))
                human_stdout = _TeeTextStream(terminal_stdout, human_log_stream)
                human_stderr = _TeeTextStream(terminal_stderr, human_log_stream)
                snapshot = json.loads(
                    (run_root / "run-snapshot.json").read_text(encoding="utf-8")
                )
                for line in _human_start_lines(
                    config,
                    args,
                    accelerator,
                    raw_log_path,
                    checkpoint,
                ):
                    print(line, file=human_stdout, flush=True)

            # Loop ownership routes selected progress here. Raw stdout/stderr
            # remain redirected below for unclassified backend diagnostics.
            args = replace_training_args(args, progress_stream=human_stdout)
            if accelerator.is_main_process:
                print("[train] INIT dataset+model", file=human_stdout, flush=True)
        except BaseException as exc:
            _record_training_failure(
                exc=exc,
                accelerator=accelerator,
                args=args,
                run_root=run_root,
                rank_seed=rank_seed,
                raw_log_path=raw_log_path,
                log_stream=log_stream,
                human_stderr=human_stderr,
            )
            accelerator.end_training()
            return 1
        from mspc.training.training_state import TrainingStopped
        try:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                dataset = components.dataset.build_training_dataset(args)
                if args.cache_stage_plan is not None:
                    from mspc.training.cache_readiness import require_cache_stage
                    from mspc.training.training_state import request_training_stop
                    require_cache_stage(dataset, args, world_size=accelerator.num_processes)
                    # Only this operational request changes between stages. The
                    # full configuration, sampler and saved state stay identical.
                    failure = [None]
                    if accelerator.is_main_process and args.cache_stage_plan["target_step"] < args.max_train_steps:
                        try:
                            request_training_stop(run_root, reason="cache stage review",
                                target_step=args.cache_stage_plan["target_step"])
                        except Exception as exc:
                            failure[0] = f"{type(exc).__name__}: {exc}"
                    accelerate.utils.broadcast_object_list(failure, from_process=0)
                    if failure[0] is not None:
                        raise RuntimeError(f"cache stage stop request was not published: {failure[0]}")
                model = components.model.build_model(args, accelerator.device)
                model_logger = TrainableStateDictLogger(
                    output_path=args.output_path,
                    remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
                )
                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    require_zero_partitions_within_int32(
                        optimizer_groups_by_dtype(model.optimizer_param_groups(args)), accelerator.num_processes,
                    )
                validation = None
                if args.validation is not None:
                    from mspc.training.validation_loss import ValidationLoss
                    from mspc.training.loop import dataloader_kwargs

                    validation = ValidationLoss(
                        components.dataset.build_validation_dataset(args),
                        every_steps=args.validation["every_steps"], draws=args.validation["draws"],
                        seed=args.training_seed, rank=accelerator.process_index,
                        world=accelerator.num_processes, log_dir=args.log_dir,
                        loader_kwargs=dataloader_kwargs(args),
                        intermediate_panel=args.validation.get("intermediate_panel"))
                if config.resolved["dataset"]["type"] in WAN22_BLOCK_DATASET_TYPES:
                    _load_block_vae(config.resolved["dataset"], model, dataset, validation)
                if accelerator.is_main_process:
                    _update_run_snapshot(run_root, dataset_summary=dataset.metadata_summary)
                    print("[train] READY", file=human_stdout, flush=True)
                if accelerator.is_local_main_process:
                    print("[dataset] " + json.dumps(dataset.metadata_summary, ensure_ascii=False,
                                                    sort_keys=True), flush=True)
                components.trainer.train_sft(
                    accelerator, dataset, model, model_logger, args, validation=validation
                )
                accelerator.wait_for_everyone()
        except TrainingStopped as stopped:
            _record_training_stop(accelerator=accelerator, run_root=run_root,
                                  receipt=stopped.receipt, human_stdout=human_stdout)
            accelerator.end_training()
            return 0
        except BaseException as exc:
            _record_training_failure(
                exc=exc,
                accelerator=accelerator,
                args=args,
                run_root=run_root,
                rank_seed=rank_seed,
                raw_log_path=raw_log_path,
                log_stream=log_stream,
                human_stderr=human_stderr,
            )
            accelerator.end_training()
            return 1
        _write_rank_status(
            run_root,
            accelerator.process_index,
            {
                "status": "completed",
                "completed_at": _now_iso(),
                "training_seed": args.training_seed,
                "rank_seed": rank_seed,
            },
        )
        if accelerator.is_main_process:
            _update_run_snapshot(
                run_root,
                status="completed",
                completed_at=_now_iso(),
            )
            print(
                f"[train] COMPLETED log={log_dir / 'train.log'}",
                file=human_stdout,
                flush=True,
            )
    accelerator.end_training()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    parser.add_argument("--checkpoint", type=Path, metavar="STEP_N_SAFETENSORS")
    parser.add_argument("--resume-state", type=Path, metavar="TRAINING_STATE_DIRECTORY")
    parser.add_argument("--resume-migration", type=Path, metavar="MIGRATION_JSON")
    parser.add_argument("--cache-stage", type=Path, metavar="OPERATIONAL_CACHE_STAGE_JSON")
    parser.add_argument("--gpus-per-node", required=True, type=int)
    parser.add_argument("--nnodes", required=True, type=int)
    args = parser.parse_args(argv)
    return run_training(
        load_experiment_config(args.config),
        checkpoint_override=args.checkpoint,
        resume_state=args.resume_state,
        resume_migration=args.resume_migration,
        cache_stage=args.cache_stage,
        gpus_per_node=args.gpus_per_node,
        nnodes=args.nnodes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
