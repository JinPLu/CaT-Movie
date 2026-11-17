from __future__ import annotations

import json
from pathlib import Path

import pytest

from mspc.runs import RunKind, RunManifest, RunStatus, write_run_manifest


def manifest(output_path: Path) -> RunManifest:
    return RunManifest(
        schema_version=1,
        kind=RunKind.DATA,
        status=RunStatus.SUCCESS,
        output_path=output_path,
        cli_argv=("mspc", "data", "--config", "/tmp/data.yaml"),
    )


def test_write_run_manifest_requires_explicit_replace(tmp_path: Path) -> None:
    destination = (tmp_path / "run_manifest.json").resolve()
    first = manifest((tmp_path / "output-a").resolve())
    second = manifest((tmp_path / "output-b").resolve())

    write_run_manifest(destination, first, replace=False)
    with pytest.raises(FileExistsError):
        write_run_manifest(destination, second, replace=False)
    assert json.loads(destination.read_text(encoding="utf-8"))["output_path"].endswith("output-a")

    write_run_manifest(destination, second, replace=True)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["output_path"].endswith("output-b")
    assert payload["cli_argv"] == ["mspc", "data", "--config", "/tmp/data.yaml"]


def test_run_manifest_rejects_relative_output_path() -> None:
    with pytest.raises(ValueError, match="output_path"):
        manifest(Path("relative-output"))
