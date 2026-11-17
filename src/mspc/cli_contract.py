"""Small result contract shared by root CLI domain handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CliOutcome:
    payload: Any
    exit_code: int
    emit_payload: bool = True

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int or not 0 <= self.exit_code <= 255:
            raise ValueError("CLI exit_code must be an integer from 0 through 255")
        if type(self.emit_payload) is not bool:
            raise TypeError("CLI emit_payload must be a bool")
