"""Root MSPC CLI: owns the parser and routes each command to its domain."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

from .cli_contract import CliOutcome
from .cli_domains.data import DATA_COMMANDS, register_data_commands, run_data_command
from .cli_domains.inference import (
    INFERENCE_COMMANDS,
    register_inference_commands,
    run_inference_command,
)
from .cli_domains.training import (
    TRAINING_COMMANDS,
    _cluster_overrides,
    add_cluster_arguments,
    register_training_commands,
    run_training_command,
)
from .config import ExperimentConfigError, load_experiment_config
from .runs import LaunchMode, resolve_launch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mspc")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("doctor", "validate a config and report local contract prerequisites"),
        ("inspect", "print the strictly resolved config without executing it"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    launch = commands.add_parser(
        "launch-inspect",
        help="resolve a strict experiment launch without starting a process",
    )
    launch.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    launch.add_argument("--mode", required=True, choices=[mode.value for mode in LaunchMode])
    add_cluster_arguments(launch)

    register_data_commands(commands)
    register_training_commands(commands)
    register_inference_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"inspect", "doctor"}:
            loaded = load_experiment_config(args.config)
        if args.command == "inspect":
            result = {
                "source_path": str(loaded.source_path),
                "config": loaded.to_dict(),
            }
        elif args.command == "doctor":
            result = {
                "valid": True,
                "source_path": str(loaded.source_path),
                "python": platform.python_version(),
            }
        elif args.command == "launch-inspect":
            experiment = load_experiment_config(args.config)
            mode = LaunchMode(args.mode)
            result = resolve_launch(
                experiment,
                mode=mode,
                train_overrides=(
                    _cluster_overrides(args) if mode is LaunchMode.TRAIN else None
                ),
            ).to_dict()
        elif args.command in DATA_COMMANDS:
            result = run_data_command(args)
        elif args.command in TRAINING_COMMANDS:
            result = run_training_command(args)
        elif args.command in INFERENCE_COMMANDS:
            result = run_inference_command(args)
        else:  # argparse prevents this; keep execution closed by construction.
            raise RuntimeError(f"unsupported command: {args.command}")
    except (
        ExperimentConfigError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"mspc: error: {exc}", file=sys.stderr)
        return 2
    outcome = result if isinstance(result, CliOutcome) else CliOutcome(payload=result, exit_code=0)
    if outcome.emit_payload:
        print(json.dumps(outcome.payload, indent=2, sort_keys=True, ensure_ascii=False))
    return outcome.exit_code
