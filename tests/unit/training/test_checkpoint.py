from __future__ import annotations

from pathlib import Path

import pytest

from mspc.training.checkpoint import inspect_training_checkpoint


def test_no_checkpoint_is_valid() -> None:
    assert inspect_training_checkpoint(None) is None


def test_trailing_checkpoint_overrides_config_and_infers_step(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "step-100.safetensors"
    override = tmp_path / "step-500.safetensors"
    configured.touch()
    override.touch()

    checkpoint = inspect_training_checkpoint(str(configured), override)

    assert checkpoint is not None
    assert checkpoint.path == override
    assert checkpoint.step == 500


def test_checkpoint_sidecar_is_not_inspected(tmp_path: Path) -> None:
    path = tmp_path / "step-800.safetensors"
    path.touch()
    path.with_suffix(".receipt.json").write_text("not-json", encoding="utf-8")

    checkpoint = inspect_training_checkpoint(str(path))

    assert checkpoint is not None
    assert checkpoint.step == 800
    assert checkpoint.to_dict() == {"path": str(path), "step": 800}


@pytest.mark.parametrize("name", ["checkpoint.safetensors", "step-x.safetensors"])
def test_checkpoint_requires_canonical_step_filename(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.touch()
    with pytest.raises(ValueError, match="checkpoint filename"):
        inspect_training_checkpoint(str(path))


def test_missing_checkpoint_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_training_checkpoint(str(tmp_path / "step-500.safetensors"))
