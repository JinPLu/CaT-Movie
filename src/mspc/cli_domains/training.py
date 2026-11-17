"""Lazy training command registration for the root :mod:`mspc` CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from mspc.cli_contract import CliOutcome
from mspc.config import load_experiment_config
from mspc.runs import (
    LaunchMode,
    TrainLaunchOverrides,
    record_post_training_smoke_failure,
    resolve_launch,
)


TRAINING_COMMANDS = frozenset({"train"})


def register_training_commands(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser("train", help="plan or run one strict training config")
    command.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    command.add_argument("--checkpoint", type=Path, metavar="STEP_N_SAFETENSORS")
    command.add_argument("--resume-state", type=Path, metavar="TRAINING_STATE_DIRECTORY")
    command.add_argument("--resume-migration", type=Path, metavar="MIGRATION_JSON")
    command.add_argument("--cache-stage", type=Path, metavar="STAGE_JSON")
    command.add_argument(
        "--execution-mode",
        required=True,
        choices=("plan", "run"),
    )
    add_cluster_arguments(command)


def add_cluster_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--gpus-per-node", type=int)
    command.add_argument("--nnodes", type=int)
    command.add_argument("--node-rank", type=int)
    command.add_argument("--master-addr")
    command.add_argument("--master-port")


def _cluster_overrides(args: argparse.Namespace) -> TrainLaunchOverrides:
    names = ("gpus_per_node", "nnodes", "node_rank", "master_addr", "master_port")
    values = {name: getattr(args, name, None) for name in names}
    supplied = [name for name, value in values.items() if value is not None]
    if not supplied:
        raise ValueError("cluster launch operands are required for training")
    if len(supplied) != len(names):
        missing = sorted(set(names) - set(supplied))
        raise ValueError(f"cluster launch overrides must be supplied together; missing={missing}")
    return TrainLaunchOverrides(**values)


def run_training_command(args: argparse.Namespace) -> CliOutcome:
    if args.command not in TRAINING_COMMANDS:
        raise ValueError(f"unsupported training command: {args.command}")
    config = load_experiment_config(args.config)
    launch = resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=_cluster_overrides(args),
        checkpoint_override=args.checkpoint,
        resume_state_override=args.resume_state,
        resume_migration_override=getattr(args, "resume_migration", None),
        cache_stage_override=args.cache_stage,
    )
    payload = launch.to_dict()
    payload["execution_mode"] = args.execution_mode
    if args.execution_mode == "plan":
        return CliOutcome(payload=payload, exit_code=0)
    output_root = Path(config.resolved["output"]["root"])
    existing_run_roots = (
        {path.resolve() for path in output_root.iterdir() if path.is_dir()}
        if output_root.is_dir()
        else set()
    )
    completed = subprocess.run(
        launch.argv,
        env={**os.environ, **launch.environment},
        check=False,
    )
    payload["returncode"] = completed.returncode
    inference = config.resolved["inference"]
    if (
        completed.returncode == 0
        and inference is not None
        and inference["enabled"]
        and inference["weight_source"] == "training_output_checkpoint"
        and args.node_rank == 0
    ):
        from mspc.training.wan22_smoke_gate import (
            resolve_new_run_root,
            run_training_output_smoke_gate,
        )

        try:
            run_root = resolve_new_run_root(output_root, existing_run_roots)
            payload["smoke_gate"] = run_training_output_smoke_gate(config, run_root)
        except Exception as exc:
            record_post_training_smoke_failure(payload, exc)
    return CliOutcome(
        payload=payload,
        exit_code=completed.returncode,
        emit_payload=False,
    )
