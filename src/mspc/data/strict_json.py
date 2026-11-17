"""One strict JSON decoder for all MSPC data contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {token}")


def loads_strict_json(text: str, *, label: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"{label} must be JSON text")
    if not text:
        raise ValueError(f"{label} must not be empty")
    if text != text.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {exc.msg}") from exc


def load_strict_json_file(path: str | Path, *, label: str) -> Any:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return loads_strict_json(text, label=label)
