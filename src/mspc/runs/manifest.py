"""Atomic run-manifest persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .schema import RunManifest


def write_run_manifest(
    path: str | Path, manifest: RunManifest, *, replace: bool
) -> Path:
    """Write a manifest atomically with an explicit replacement decision."""

    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("run manifest path must be absolute")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(destination)
    payload = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    temp_path: Path | None = None
    try:
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temp_path, destination)
        else:
            # Hard-link creation is atomic and fails if another writer won the
            # destination name between the earlier check and this operation.
            os.link(temp_path, destination)
            temp_path.unlink()
        temp_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return destination
