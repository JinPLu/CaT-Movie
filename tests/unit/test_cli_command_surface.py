from __future__ import annotations

import argparse

import pytest

from mspc.cli import build_parser, main
from mspc.cli_domains import data, inference, training


CURRENT_COMMANDS = {
    "doctor",
    "inspect",
    "launch-inspect",
    "train",
    "data-precompute-latents",
    "data-merge-latent-cache",
    "infer",
}


def _parser(register) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mspc")
    commands = parser.add_subparsers(dest="command", required=True)
    register(commands)
    return parser


def test_root_parser_lists_exactly_the_current_commands() -> None:
    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == CURRENT_COMMANDS


@pytest.mark.parametrize(
    ("register", "declared"),
    [
        (data.register_data_commands, data.DATA_COMMANDS),
        (training.register_training_commands, training.TRAINING_COMMANDS),
        (inference.register_inference_commands, inference.INFERENCE_COMMANDS),
    ],
)
def test_registered_commands_match_the_declared_set(register, declared) -> None:
    commands = _parser(register)._subparsers._group_actions[0].choices
    assert set(commands) == set(declared)
    assert len(commands) == len(declared)


def test_removed_command_names_are_not_silently_accepted() -> None:
    """Argparse rejects an unknown subcommand, so a stale script cannot quietly
    resolve to a different implementation than the one it was written against."""

    for removed in (
        "benchmark-compact21",
        "data-ci-vid-training-admission",
        "evaluation-metrics",
        "infer-dashscope-run",
        "infer-wan22-m0",
    ):
        with pytest.raises(SystemExit):
            main([removed])
