"""Resolve an experiment config to an inspectable launch plan."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from mspc.config.experiment import (
    WAN22_BLOCK_DATASET_TYPES,
    ExperimentConfig,
)
from mspc.core.wan22_contract import WAN22_CACHE_LAYOUTS


class LaunchMode(str, Enum):
    TRAIN = "train"
    INFER = "infer"


@dataclass(frozen=True)
class TrainLaunchOverrides:
    """Explicit cluster operands supplied by the Wan-compatible shell entry."""

    gpus_per_node: int
    nnodes: int
    node_rank: int
    master_addr: str
    master_port: str

    def __post_init__(self) -> None:
        if type(self.gpus_per_node) is not int or self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be a positive int")
        if type(self.nnodes) is not int or self.nnodes <= 0:
            raise ValueError("nnodes must be a positive int")
        if type(self.node_rank) is not int or not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank must be an int in [0, nnodes)")
        if not isinstance(self.master_addr, str) or not self.master_addr.strip():
            raise ValueError("master_addr must be a non-empty string")
        if not isinstance(self.master_port, str) or not self.master_port.isdigit():
            raise ValueError("master_port must be a decimal port string")
        port = int(self.master_port)
        if not 1 <= port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")


@dataclass(frozen=True)
class ResolvedLaunch:
    mode: LaunchMode
    executable: Path
    entrypoint: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    output_path: Path
    checkpoint: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "executable": str(self.executable),
            "entrypoint": str(self.entrypoint),
            "argv": list(self.argv),
            "environment": dict(sorted(self.environment.items())),
            "output_path": str(self.output_path),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def require_renderer_input_launch_artifacts(config: ExperimentConfig) -> None:
    """Check common block inputs and only the cache manifests read by latent storage."""

    data = config.resolved
    dataset = data["dataset"]
    if dataset["type"] not in WAN22_BLOCK_DATASET_TYPES:
        return
    paths = [Path(dataset[name]) for name in ("members_path", "sampling_manifest_path")]
    cached_targets = dataset["type"] == "block_latent_cache"
    if cached_targets:
        paths.append(Path(dataset["target_index_path"]))
    validation = data["training"]["validation"]
    if validation is not None:
        paths.extend(Path(validation[name]) for name in ("members_path", "sampling_manifest_path"))
    for path in paths:
        if not path.is_file():
            raise ValueError(f"block training requires input file: {path}")
    if cached_targets:
        # 索引实际出处决定读哪个根；在线视频不读取潜变量index或manifest。
        origins = {json.loads(line)["origin"] for line in Path(dataset["target_index_path"]).read_text().splitlines()}
        for origin in origins:
            path = Path(dataset["target_roots"][origin]) / "manifest.json"
            if not path.is_file():
                raise ValueError(f"block training requires input file: {path}")


def require_train_launch_dataset(config: ExperimentConfig) -> None:
    """Validate training-only dataset admission before backend resolution."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    data = config.resolved
    # 同一个block训练入口显式选择video或latents；source/legacy仍仅用于预处理或审计。
    # 可训练类型从config.experiment的唯一名单读取。
    if data["dataset"]["type"] not in WAN22_BLOCK_DATASET_TYPES:
        raise ValueError(
            "Wan renderer training requires an explicit block dataset config "
            f"({', '.join(sorted(WAN22_BLOCK_DATASET_TYPES))}); "
            "source and legacy configs are precompute/audit-only: "
            f"{config.source_path}"
        )
    require_renderer_input_launch_artifacts(config)


def require_precompute_launch_dataset(config: ExperimentConfig) -> None:
    """Validate the raw cache-source contract before starting Wan2.2 encoding."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    data = config.resolved
    if (
        data["model"]["type"] == "wan22_diffsynth"
        and data["dataset"]["type"] != "multishot_csv"
    ):
        raise ValueError(
            "Wan2.2 latent precompute requires a multishot_csv cache-source "
            "config; cached configs are training-only: "
            f"{config.source_path}"
        )
    layout = data["dataset"].get("cache_layout")
    if layout not in WAN22_CACHE_LAYOUTS:
        raise ValueError(
            "Wan2.2 latent precompute requires dataset.cache_layout in "
            f"{sorted(WAN22_CACHE_LAYOUTS)}: {config.source_path}"
        )
    precompute = data.get("precompute")
    if not isinstance(precompute, Mapping):
        raise ValueError(
            "Wan2.2 latent precompute requires config.precompute "
            f"(encode_batch_size and decode DataLoader knobs): {config.source_path}"
        )


def record_post_training_smoke_failure(
    payload: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Record a post-training visual smoke failure without failing the launch chain.

    Training has already completed when this runs. A smoke reject or exception
    does not revoke that completion and must not change the training exit code.
    """

    recorded = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "interrupts_launch": False,
    }
    payload["smoke_gate"] = recorded
    print(
        "post-training smoke failed after training completed; "
        f"launch chain continues: {recorded['error_type']}: {recorded['error']}",
        file=sys.stderr,
    )
    return recorded


def resolve_launch(
    config: ExperimentConfig,
    *,
    mode: LaunchMode,
    train_overrides: TrainLaunchOverrides | None = None,
    checkpoint_override: Path | None = None,
    resume_state_override: Path | None = None,
    resume_migration_override: Path | None = None,
    cache_stage_override: Path | None = None,
) -> ResolvedLaunch:
    """Resolve command/environment only; this function never starts a process."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(mode, LaunchMode):
        raise TypeError("mode must be a LaunchMode")
    if train_overrides is not None and not isinstance(train_overrides, TrainLaunchOverrides):
        raise TypeError("train_overrides must be TrainLaunchOverrides or None")
    if mode is LaunchMode.INFER and train_overrides is not None:
        raise ValueError("train_overrides are valid only for training launches")
    if mode is LaunchMode.INFER and (checkpoint_override is not None or resume_state_override is not None
                                    or cache_stage_override is not None or resume_migration_override is not None):
        raise ValueError("checkpoint, resume-state and cache-stage overrides are valid only for training launches")
    data = config.resolved
    runtime = data["runtime"]
    launch = data["launch"]
    common_env = {
        "HF_HOME": runtime["hf_cache"],
        "HF_HUB_OFFLINE": "1" if runtime["offline_hf"] else "0",
        "TRANSFORMERS_OFFLINE": "1" if runtime["offline_hf"] else "0",
        "HF_DATASETS_OFFLINE": "1" if runtime["offline_hf"] else "0",
        "TOKENIZERS_PARALLELISM": data["environment"]["tokenizers_parallelism"],
        "PYTHONUNBUFFERED": data["environment"]["python_unbuffered"],
        "WANDB_MODE": data["environment"]["wandb_mode"],
    }
    if launch["cuda_visible_devices"] is not None:
        common_env["CUDA_VISIBLE_DEVICES"] = launch["cuda_visible_devices"]

    if mode is LaunchMode.TRAIN:
        require_train_launch_dataset(config)
        if train_overrides is None:
            raise ValueError("train_overrides are required for training launches")
        # torchrun与训练使用同一解释器；不再读取宿主的Accelerate YAML或另一份DeepSpeed JSON。
        executable = Path(sys.executable)
        entrypoint = Path(launch["train_entrypoint"])
        cluster = train_overrides
        common_env.update({
            "PYTHONPATH": ":".join(launch["train_pythonpath"]),
            "NCCL_DEBUG": data["nccl"]["debug"],
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": data["nccl"]["async_error_handling"],
            "CUDA_DEVICE_MAX_CONNECTIONS": data["nccl"]["device_max_connections"],
            "TORCH_NCCL_BLOCKING_WAIT": data["nccl"]["blocking_wait"],
            "NCCL_BUFFSIZE": data["nccl"]["buffsize"],
            "NCCL_NTHREADS": data["nccl"]["nthreads"],
        })
        argv_parts = [
            str(executable), "-m", "torch.distributed.run",
            "--nproc_per_node", str(cluster.gpus_per_node),
            "--nnodes", str(cluster.nnodes),
            "--node_rank", str(cluster.node_rank),
            "--master_addr", cluster.master_addr,
            "--master_port", cluster.master_port,
            str(entrypoint), "--config", str(config.source_path),
            "--gpus-per-node", str(cluster.gpus_per_node),
            "--nnodes", str(cluster.nnodes),
        ]
        selected_checkpoint = checkpoint_override or (
            None if data["checkpoint"] is None else Path(data["checkpoint"])
        )
        if selected_checkpoint is not None:
            argv_parts.extend(("--checkpoint", str(selected_checkpoint)))
        if resume_state_override is not None:
            if selected_checkpoint is not None:
                raise ValueError("resume-state and weights-only checkpoint are mutually exclusive")
            if "state_checkpoint" not in data["training"]:
                raise ValueError("resume-state requires training.state_checkpoint policy")
            argv_parts.extend(("--resume-state", str(resume_state_override)))
        if resume_migration_override is not None:
            if resume_state_override is None:
                raise ValueError("resume-migration requires resume-state")
            argv_parts.extend(("--resume-migration", str(resume_migration_override)))
        if cache_stage_override is not None:
            if data["dataset"]["type"] != "block_latent_cache":
                raise ValueError("cache-stage requires production block_latent_cache targets")
            argv_parts.extend(("--cache-stage", str(cache_stage_override)))
        argv = tuple(argv_parts)
        checkpoint = selected_checkpoint
        output_path = Path(data["output"]["root"])
    else:
        inference = data["inference"]
        if not inference["enabled"]:
            raise ValueError("config inference.enabled=false cannot resolve an inference launch")
        if inference.get("weight_source") == "training_output_checkpoint":
            raise ValueError(
                "training_output_checkpoint is resolved only by the post-training smoke gate"
            )
        # Same reasoning as the training branch: this inference runs the same
        # model stack as whoever launched it, so it belongs in the same env.
        # Reading runtime.python here meant the post-training smoke gate ignored
        # the architecture routing that had just placed training correctly — on
        # 2026-08-26 the A100 gate trained all 16 steps and then died loading the
        # DiT for the smoke, on "driver too old", inside the cu130 env.
        executable = Path(sys.executable)
        entrypoint = Path(launch["infer_entrypoint"])
        common_env["PYTHONPATH"] = ":".join(launch["infer_pythonpath"])
        argv = (
            str(executable), str(entrypoint), "--config", str(config.source_path),
        )
        checkpoint = inference["dit_checkpoint"]
        output_path = Path(inference["output_dir"])

    return ResolvedLaunch(
        mode=mode, executable=executable, entrypoint=entrypoint, argv=argv,
        environment=MappingProxyType(common_env), output_path=output_path,
        checkpoint=None if checkpoint is None else Path(checkpoint),
    )
