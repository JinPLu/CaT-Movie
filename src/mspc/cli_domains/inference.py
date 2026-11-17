"""Inference command registration for the root :mod:`mspc` CLI.

Local DiT execution goes through the strict launch contract.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from mspc.cli_contract import CliOutcome
from mspc.config import load_experiment_config
from mspc.runs import LaunchMode, resolve_launch


INFERENCE_COMMANDS = frozenset({"infer"})


def register_inference_commands(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser("infer", help="plan or run one strict inference config")
    command.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    command.add_argument(
        "--execution-mode",
        required=True,
        choices=("plan", "run"),
    )


def run_inference_command(args: argparse.Namespace) -> CliOutcome:
    if args.command not in INFERENCE_COMMANDS:
        raise ValueError(f"unsupported inference command: {args.command}")
    config = load_experiment_config(args.config)
    launch = resolve_launch(config, mode=LaunchMode.INFER)
    payload = launch.to_dict()
    payload["inference_kind"] = "experiment_launch"
    payload["execution_mode"] = args.execution_mode
    if args.execution_mode == "plan":
        return CliOutcome(payload=payload, exit_code=0)
    completed = subprocess.run(
        launch.argv,
        env={**os.environ, **launch.environment},
        check=False,
    )
    payload["returncode"] = completed.returncode
    return CliOutcome(payload=payload, exit_code=completed.returncode)
