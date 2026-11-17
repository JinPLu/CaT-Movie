"""完整训练状态：仅显式声明保存策略的运行使用，权重热启动另走 checkpoint。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import os
import tempfile
from typing import Any

import numpy as np
import torch


class TrainingStopped(Exception):
    """一次完整更新及状态保存已完成，runner应记录stopped而非completed。"""

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__(f"training stopped at step {receipt['completed_steps']}: {receipt['state_path']}")


def read_training_stop_request(run_root: Path, *, run_id: str, state_enabled: bool,
                               completed_steps: int | None = None) -> dict[str, Any] | None:
    path = run_root / "control/stop-request.json"
    if not path.exists():
        return None
    request = json.loads(path.read_text())
    if (not isinstance(request, dict) or set(request) not in (
            {"run_id", "reason"}, {"run_id", "reason", "target_step"})
            or request["run_id"] != run_id or not isinstance(request["reason"], str)
            or not request["reason"].strip()):
        raise ValueError(f"invalid stop request for this run: {path}")
    if not state_enabled:
        raise ValueError("recoverable stop requires training.state_checkpoint")
    if "target_step" in request:
        target = request["target_step"]
        if type(target) is not int or target < 1:
            raise ValueError("scheduled stop target_step must be a positive optimizer update")
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("scheduled stop requires the current completed optimizer update")
        if completed_steps > target:
            raise ValueError(f"scheduled stop target_step={target} was already passed at step {completed_steps}")
        if completed_steps < target:
            return None
    return request


def request_training_stop(run_root: str | Path, *, reason: str, target_step: int | None = None) -> Path:
    """原子发布本run的停止请求；可指定精确完整更新，不更改训练配方。"""
    run_root = Path(run_root).resolve(strict=True)
    snapshot = json.loads((run_root / "run-snapshot.json").read_text())
    if snapshot.get("status") != "running":
        raise ValueError("stop request requires a running training snapshot")
    if snapshot["config"]["training"].get("state_checkpoint") is None:
        raise ValueError("recoverable stop requires training.state_checkpoint")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("stop reason must be non-empty text")
    request = {"run_id": snapshot["runtime"]["run_id"], "reason": reason}
    if target_step is not None:
        if type(target_step) is not int or target_step < 1:
            raise ValueError("scheduled stop target_step must be a positive optimizer update")
        planned = snapshot["config"]["training"].get("max_train_steps")
        if type(planned) is not int or target_step > planned:
            raise ValueError("scheduled stop target_step must not exceed the configured training target")
        request["target_step"] = target_step
    control = run_root / "control"
    path = control / "stop-request.json"
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise FileExistsError(f"another stop request is pending: {path}")
        return path
    if target_step is not None:
        metrics = run_root / "logs/metrics.jsonl"
        if metrics.exists():
            with metrics.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    step = json.loads(line)["step"]
                    if type(step) is not int or step < 0:
                        raise ValueError("invalid completed optimizer update in training metrics")
                    if step >= target_step:
                        raise ValueError(f"scheduled stop target_step={target_step} was already reached at step {step}")
    control.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=".stop-request-", dir=control, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(request, stream, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)  # Exclusive publication of a complete JSON record.
        except FileExistsError:
            if json.loads(path.read_text()) != request:
                raise FileExistsError(f"another stop request is pending: {path}")
    finally:
        temporary.unlink()
    return path


def complete_training_stop(accelerator: Any, *, state_path: Path, progress: Any,
                           args: Any, request: dict[str, Any] | None) -> dict[str, Any]:
    """完整状态公布后才消费请求并ACK；仅停止时作一次控制结果广播。"""
    from accelerate.utils import broadcast_object_list

    result: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            state = inspect_training_state(state_path, config=args.experiment_config,
                                           world_size=accelerator.num_processes)
            if state["completed_steps"] != progress.completed_steps or state["run_id"] != args.run_id:
                raise ValueError("saved stop state belongs to a different update or run")
            root = Path(args.run_root)
            active = root / "control/stop-request.json"
            if json.loads(active.read_text()) != request:
                raise ValueError("stop request changed before the complete state was saved")
            receipt_path = root / f"control/stopped-step-{progress.completed_steps}.json"
            if receipt_path.exists():
                raise FileExistsError(receipt_path)
            # Consumed request without status=stopped is not an ACK if a crash
            # occurs before the enriched receipt is atomically published.
            active.rename(receipt_path)
            receipt = {"status": "stopped", "run_id": args.run_id,
                       "completed_steps": progress.completed_steps,
                       "state_path": str(state_path.resolve()), "request": request}
            temporary = receipt_path.with_name(receipt_path.name + ".tmp")
            with temporary.open("x") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(receipt_path)
            result[0] = {"ok": True, "receipt": receipt}
        except Exception as exc:
            result[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    broadcast_object_list(result, from_process=0)
    if result[0]["ok"] is not True:
        raise RuntimeError(f"stop was not acknowledged: {result[0]['error']}")
    return result[0]["receipt"]


@dataclass
class TrainingProgress:
    """在已完成的优化器更新边界保存数据位置；不保存预取队列。"""

    completed_steps: int = 0
    microsteps: int = 0
    samples_seen: int = 0
    epoch: int = 0
    next_batch: int = 0
    epoch_sampler_state: torch.Tensor | None = None

    def state_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != set(vars(self)):
            raise ValueError("training progress fields differ")
        for name in ("completed_steps", "microsteps", "samples_seen", "epoch", "next_batch"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"invalid training progress {name}")
        value = state["epoch_sampler_state"]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("missing epoch sampler generator state")
        self.__dict__.update(state)


def inspect_training_state(path: str | Path, *, config: dict[str, Any], world_size: int) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("resume-state must name a complete training-state directory")
    state = json.loads((path / "state.json").read_text())
    if state.get("artifact_type") != "mspc_complete_training_state" or state.get("schema_version") != 1:
        raise ValueError("resume-state is not a complete training-state checkpoint")
    if state["config"] != config:
        raise ValueError("resume-state config differs; model, data and training settings must match")
    if state["world_size"] != world_size:
        raise ValueError("resume-state world size differs")
    if type(state["completed_steps"]) is not int or state["completed_steps"] < 1:
        raise ValueError("invalid completed step in training-state checkpoint")
    return state


def archive_future_training_artifacts(run_root: Path, *, completed_steps: int, run_id: str) -> list[str]:
    """恢复前搬走未来轨迹的完整产物，保留原件并让重放步号可重新发布。"""
    run_root = run_root.resolve()
    selected: list[Path] = []
    for state in (run_root / "training-state").glob("step-*"):
        step = int(state.name.removeprefix("step-"))
        if step <= completed_steps:
            continue
        metadata = json.loads((state / "state.json").read_text())
        if (metadata.get("artifact_type") != "mspc_complete_training_state"
                or metadata.get("schema_version") != 1 or metadata.get("run_id") != run_id
                or metadata.get("run_root") != str(run_root) or metadata.get("completed_steps") != step):
            raise ValueError(f"future training state belongs to another run or update: {state}")
        selected.append(state)
    for export in (run_root / "checkpoints").glob("step-*.safetensors"):
        if int(export.stem.removeprefix("step-")) > completed_steps:
            if not export.is_file():
                raise ValueError(f"future checkpoint is not a file: {export}")
            selected.append(export)
    for receipt in (run_root / "control").glob("stopped-step-*.json"):
        if int(receipt.stem.removeprefix("stopped-step-")) > completed_steps:
            selected.append(receipt)
    if not selected:
        return []
    archive = run_root / "resume-tail" / (
        f"step-{completed_steps}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}")
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in selected:
        destination = archive / path.relative_to(run_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        moved.append(str(destination))
    return moved


def archive_training_log_tail(log_dir: Path, *, completed_steps: int, rank: int,
                              is_main_process: bool) -> list[str]:
    """保留原JSONL，主指标流回到恢复点；每rank只修改自己写的已知文件。"""
    # All streams append after restore. Timing rows beyond the saved update
    # would otherwise duplicate replayed steps; validation rows at the saved
    # update already completed before save_training_state and must be kept.
    names = [f"samples.rank{rank}.jsonl", f"timing.rank{rank}.jsonl", f"validation.rank{rank}.jsonl"]
    if is_main_process:
        names.append("metrics.jsonl")
    archive = log_dir / "resume-tail" / (
        f"step-{completed_steps}-rank-{rank}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}")
    saved = []
    for name in names:
        path = log_dir / name
        if not path.exists():
            continue
        original = path.read_bytes()
        lines, kept = original.splitlines(keepends=True), []
        discarded = False
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                # A parseable JSON object without its record delimiter is not
                # committed either; keeping it would join the next append.
                discarded = True
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError(f"malformed complete training log row: {path}:{index+1}")
            if type(row["step"]) is not int or row["step"] < 0:
                raise ValueError(f"invalid training log step: {path}:{index+1}")
            if row["step"] <= completed_steps:
                kept.append(line)
            else:
                discarded = True
        if discarded:
            archive.mkdir(parents=True, exist_ok=True)
            backup = archive / name
            backup.write_bytes(original)
            stage = path.with_name(path.name + ".resume.tmp")
            stage.write_bytes(b"".join(kept))
            stage.replace(path)
            saved.append(str(backup))
    return saved


def _capture_rng() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def resumed_epoch_batches(accelerator: Any, dataloader: Any, next_batch: int):
    """跳过已消费批而不解码；新 iterator 的 base-seed 抽签不进入训练随机流。"""
    state = _capture_rng()
    loader = accelerator.skip_first_batches(dataloader, next_batch)
    iterator = iter(loader)
    try:
        first = next(iterator)
    except StopIteration:
        _restore_rng(state)
        return
    _restore_rng(state)
    yield first
    yield from iterator


def _validate_training_state_files(path: Path, *, world_size: int, deepspeed: bool) -> None:
    """删旧状态前核对现役原生保存器的完整文件集；不重读大张量。"""
    required = [path / "scheduler.bin", path / "custom_checkpoint_0.pkl"]
    required.extend(path / f"random_states_{rank}.pkl" for rank in range(world_size))
    if deepspeed:
        latest = path / "latest"
        if not latest.is_file() or latest.read_text().strip() != "pytorch_model":
            raise ValueError("incomplete DeepSpeed training state: missing native checkpoint tag")
        checkpoint = path / "pytorch_model"
        required.append(checkpoint / "mp_rank_00_model_states.pt")
        required.extend(checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                        for rank in range(world_size))
    else:
        required.extend([path / "model.safetensors", path / "optimizer.bin"])
    invalid = [str(file.relative_to(path)) for file in required
               if not file.is_file() or file.stat().st_size <= 0]
    if invalid:
        raise ValueError(f"incomplete training-state files: {invalid}")


def partial_module_state(args: Any) -> bool:
    """LoRA 训练的完整状态只存可训练参数与缓冲：冻结原权重每次从原生文件重建，不进状态。"""
    freeze = (getattr(args, "training_cfg", None) or {}).get("dit_parameter_freeze")
    return freeze is not None and freeze["recipe"] == "lora_v1"


def require_partial_module_state(path: str | Path, model: Any) -> None:
    """续训前核对：状态里的模块键恰好是本模型的可训练参数与缓冲，缺一个都拒读（加载本身按非严格模式）。"""
    import torch
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    expected = set(model.state_dict()) - frozen
    saved = torch.load(Path(path) / "pytorch_model" / "mp_rank_00_model_states.pt",
                       map_location="cpu", mmap=True, weights_only=False)["module"]
    if set(saved) != expected:
        raise ValueError("partial training state does not hold exactly the trainable module keys; "
                         f"missing={sorted(expected - set(saved))[:5]} extra={sorted(set(saved) - expected)[:5]}")


def save_training_state(accelerator: Any, progress: TrainingProgress, args: Any) -> Path:
    """所有rank写好状态后才原子公布；保留策略只删除本运行的完整旧状态。"""
    if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
        raise RuntimeError("training state requires a completed optimizer update")
    repository_sha = args.repository_sha
    if (not isinstance(repository_sha, str) or len(repository_sha) != 40
            or any(character not in "0123456789abcdef" for character in repository_sha)):
        raise ValueError("training state requires the reserved lowercase 40-character Git SHA")
    keep = args.state_checkpoint["keep"]
    root = Path(args.run_root) / "training-state"
    name = f"step-{progress.completed_steps}"
    from accelerate.utils import DistributedType, broadcast_object_list
    attempt = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") if accelerator.is_main_process else None]
    broadcast_object_list(attempt, from_process=0)
    stage, final = root / f".{name}-{attempt[0]}.incomplete", root / name
    prepared: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise FileExistsError(f"training state already exists: {final}")
            stage.mkdir(exist_ok=False)
            prepared[0] = {"ok": True}
        except Exception as exc:
            prepared[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    broadcast_object_list(prepared, from_process=0)
    if prepared[0]["ok"] is not True:
        raise RuntimeError(f"training state preparation failed: {prepared[0]['error']}")
    # Wan's registered training module contains only DiT; frozen T5/VAE live
    # outside its module registry and are reconstructed from the same config.
    accelerator.save_state(str(stage), safe_serialization=True,
                           **({"exclude_frozen_parameters": True} if partial_module_state(args) else {}))
    accelerator.wait_for_everyone()
    published: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            _validate_training_state_files(stage, world_size=accelerator.num_processes,
                deepspeed=accelerator.distributed_type == DistributedType.DEEPSPEED)
            metadata = {
                "artifact_type": "mspc_complete_training_state", "schema_version": 1,
                "run_root": str(Path(args.run_root).resolve()), "run_id": args.run_id,
                "completed_steps": progress.completed_steps, "world_size": accelerator.num_processes,
                "config": args.experiment_config,
                "repository_sha": repository_sha,
            }
            migration = getattr(args, "resume_migration_provenance", None)
            if migration is not None:
                metadata["resume_migration"] = migration
            (stage / "state.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            stage.rename(final)
            states = sorted((p for p in root.glob("step-*") if (p / "state.json").is_file()),
                            key=lambda p: int(p.name.removeprefix("step-")))
            for old in states[:-keep]:
                recorded = json.loads((old / "state.json").read_text())
                if recorded["run_root"] != metadata["run_root"] or recorded["run_id"] != args.run_id:
                    raise ValueError(f"refusing to prune state belonging to another run: {old}")
                shutil.rmtree(old)
            published[0] = {"ok": True}
        except Exception as exc:
            published[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    broadcast_object_list(published, from_process=0)
    if published[0]["ok"] is not True:
        raise RuntimeError(f"training state publication failed: {published[0]['error']}")
    return final


def install_training_state_readback_probe(
    resume_path: str | Path, readback_dir: str | Path, deadline_epoch: float,
) -> None:
    """显式工程入口：恢复后、下一次forward前原生回存并等待CPU逐项确认。

    正式训练不安装此探针。原状态与回读目录须由本次工程作业独占；探针不
    删除、覆盖或重试任何产物，CPU确认方负责有界归档和ACK。
    """
    import math
    import time
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list

    original = Path(resume_path).resolve(strict=True)
    readback = Path(readback_dir).resolve()
    if not original.is_dir() or not (original / "state.json").is_file():
        raise ValueError("readback probe requires a complete original training-state directory")
    if original == readback or original.is_relative_to(readback) or readback.is_relative_to(original):
        raise ValueError("original and readback training states must have separate directories")
    if type(deadline_epoch) not in (int, float) or not math.isfinite(deadline_epoch) or deadline_epoch <= time.time():
        raise ValueError("readback probe requires a future absolute deadline")
    if getattr(Accelerator, "_mspc_training_state_readback_probe", False):
        raise RuntimeError("training-state readback probe is already installed")
    original_load = Accelerator.load_state

    def write_receipt(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)

    def load_and_readback(accelerator: Any, input_dir=None, *args, **kwargs):
        if input_dir is None or Path(input_dir).resolve(strict=True) != original:
            raise ValueError("readback probe received a different resume-state directory")
        result = original_load(accelerator, input_dir, *args, **kwargs)
        restored_rng = _capture_rng()
        iteration = accelerator.project_configuration.iteration
        error: list[str | None] = [None]
        if accelerator.is_main_process:
            if readback.exists():
                error[0] = f"readback probe requires a fresh directory: {readback}"
            elif time.time() >= deadline_epoch:
                error[0] = "readback deadline expired before the native save"
            elif accelerator.project_configuration.automatic_checkpoint_naming:
                error[0] = "readback probe requires explicit native checkpoint naming"
        broadcast_object_list(error, from_process=0)
        if error[0] is not None:
            raise RuntimeError(error[0])
        try:
            accelerator.wait_for_everyone()
            accelerator.save_state(str(readback), safe_serialization=True)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                write_receipt(readback / "readback-ready.json", {
                    "original": str(original), "readback": str(readback),
                    "world_size": accelerator.num_processes,
                })
            accelerator.wait_for_everyone()
            accepted = readback / "readback-accepted.json"
            failed = readback / "readback-failed.json"
            while True:
                if failed.exists():
                    raise RuntimeError(f"CPU training-state readback failed: {failed.read_text()}")
                if accepted.exists():
                    receipt = json.loads(accepted.read_text())
                    if (receipt.get("passed") is not True or receipt.get("original") != str(original)
                            or receipt.get("readback") != str(readback)):
                        raise ValueError("CPU training-state readback ACK differs from this probe")
                    break
                if time.time() >= deadline_epoch:
                    raise TimeoutError("CPU training-state readback was not accepted before the deadline")
                time.sleep(0.2)
            accelerator.wait_for_everyone()
        finally:
            # Native save increments its project counter; the engineering readback
            # must not advance training RNG or alter subsequent checkpoint naming.
            accelerator.project_configuration.iteration = iteration
            _restore_rng(restored_rng)
        return result

    Accelerator.load_state = load_and_readback
    Accelerator._mspc_training_state_readback_probe = True


def compare_training_state_readback(original: str | Path, readback: str | Path) -> dict[str, Any]:
    """CPU逐项N0比较原生状态；只排除MSPC收据与本探针的控制收据。"""
    from safetensors import safe_open

    original, readback = Path(original).resolve(strict=True), Path(readback).resolve(strict=True)
    ignored = {"state.json", "readback-ready.json", "readback-accepted.json", "readback-failed.json"}

    def files(root: Path) -> dict[str, Path]:
        return {str(path.relative_to(root)): path for path in root.rglob("*")
                if path.is_file() and str(path.relative_to(root)) not in ignored}

    def equal(first: Any, second: Any, location: str) -> None:
        if type(first) is not type(second):
            raise AssertionError(f"training-state type differs: {location}")
        if isinstance(first, torch.Tensor):
            if first.dtype != second.dtype or first.shape != second.shape or not torch.equal(first, second):
                raise AssertionError(f"training-state tensor differs: {location}")
        elif isinstance(first, np.ndarray):
            if first.dtype != second.dtype or first.shape != second.shape or not np.array_equal(first, second):
                raise AssertionError(f"training-state array differs: {location}")
        elif isinstance(first, dict):
            if first.keys() != second.keys():
                raise AssertionError(f"training-state keys differ: {location}")
            for key in first:
                equal(first[key], second[key], f"{location}/{key}")
        elif isinstance(first, (tuple, list)):
            if len(first) != len(second):
                raise AssertionError(f"training-state sequence length differs: {location}")
            for index, (left, right) in enumerate(zip(first, second, strict=True)):
                equal(left, right, f"{location}/{index}")
        elif (type(first).__module__, type(first).__name__) == (
                "deepspeed.runtime.fp16.loss_scaler", "LossScaler"):
            # ZeRO2 stores this native object even with FP16 disabled. It has
            # identity equality, so compare its two actual static-scaler fields.
            from deepspeed.runtime.fp16.loss_scaler import LossScaler
            if type(first) is not LossScaler or set(vars(first)) != {"cur_scale", "dynamic"} or set(
                    vars(second)) != {"cur_scale", "dynamic"}:
                raise AssertionError(f"native static LossScaler fields differ: {location}")
            equal(first.cur_scale, second.cur_scale, f"{location}/cur_scale")
            equal(first.dynamic, second.dynamic, f"{location}/dynamic")
        elif first != second:
            raise AssertionError(f"training-state value differs: {location}")

    first_files, second_files = files(original), files(readback)
    if not first_files or first_files.keys() != second_files.keys():
        raise AssertionError("native training-state file sets differ or are empty")
    for name, first_path in first_files.items():
        second_path = second_files[name]
        if first_path.suffix in {".pt", ".bin", ".pkl"}:
            first = torch.load(first_path, map_location="cpu", weights_only=False)
            second = torch.load(second_path, map_location="cpu", weights_only=False)
            equal(first, second, name)
            del first, second
        elif first_path.suffix == ".safetensors":
            with safe_open(first_path, framework="pt", device="cpu") as first, safe_open(
                    second_path, framework="pt", device="cpu") as second:
                if set(first.keys()) != set(second.keys()):
                    raise AssertionError(f"training-state tensor names differ: {name}")
                for key in first.keys():
                    equal(first.get_tensor(key), second.get_tensor(key), f"{name}/{key}")
        else:
            with first_path.open("rb") as first, second_path.open("rb") as second:
                while True:
                    block = first.read(8 * 1024 * 1024)
                    if block != second.read(8 * 1024 * 1024):
                        raise AssertionError(f"training-state auxiliary file differs: {name}")
                    if not block:
                        break
    return {"original": str(original), "readback": str(readback),
            "files": len(first_files),
            "total_bytes": sum(path.stat().st_size for path in first_files.values()),
            "readback_total_bytes": sum(path.stat().st_size for path in second_files.values())}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="在完整更新边界保存完整状态并请求停止；不发送训练信号。")
    commands = parser.add_subparsers(dest="command", required=True)
    stop = commands.add_parser("request-stop")
    stop.add_argument("--run-root", type=Path, required=True)
    stop.add_argument("--reason", required=True)
    stop.add_argument("--target-step", type=int,
                      help="精确完成此optimizer更新后停止；省略则在下一完整更新停止")
    args = parser.parse_args()
    path = request_training_stop(args.run_root, reason=args.reason, target_step=args.target_step)
    print(json.dumps({"status": "requested", "request_path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
