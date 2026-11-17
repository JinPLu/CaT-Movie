"""Strict run-manifest schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = 1


class RunKind(str, Enum):
    DATA = "data"
    GENERATE = "generate"
    TRAIN = "train"
    INFER = "infer"
    EVALUATE = "evaluate"
    REVIEW = "review"
    DOCTOR = "doctor"


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    kind: RunKind
    status: RunStatus
    output_path: Path
    cli_argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != RUN_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {RUN_MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.kind, RunKind):
            raise TypeError("kind must be a RunKind")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if not isinstance(self.output_path, Path) or not self.output_path.is_absolute():
            raise ValueError("output_path must be resolved and absolute")
        if (
            not isinstance(self.cli_argv, tuple)
            or not self.cli_argv
            or any(not isinstance(arg, str) for arg in self.cli_argv)
        ):
            raise ValueError("cli_argv must contain the explicit invocation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "status": self.status.value,
            "output_path": str(self.output_path),
            "cli_argv": list(self.cli_argv),
        }
