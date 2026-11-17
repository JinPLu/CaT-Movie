"""Single Python 3.10+ compatibility owner for TOML parsing."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

__all__ = ["tomllib"]
