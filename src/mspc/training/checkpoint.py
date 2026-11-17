"""Resolve one explicit training checkpoint input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


_STEP_FILE = re.compile(r"^(?:latest-)?step-(\d+)\.safetensors$")


@dataclass(frozen=True)
class TrainingCheckpoint:
    path: Path
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "step": self.step,
        }


def _checkpoint_step(path: Path) -> int:
    match = _STEP_FILE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "checkpoint filename must be step-N.safetensors or "
            "latest-step-N.safetensors"
        )
    return int(match.group(1))


def inspect_training_checkpoint(
    configured: str | None,
    override: str | Path | None = None,
) -> TrainingCheckpoint | None:
    selected = override if override is not None else configured
    if selected is None:
        return None
    path = Path(selected).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"checkpoint must be a file: {path}")
    return TrainingCheckpoint(
        path=path,
        step=_checkpoint_step(path),
    )
