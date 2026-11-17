from __future__ import annotations

from pathlib import Path

import pytest

from mspc.data.atomic_publish import publish_directory, publish_file


def test_publish_directory_claim_rejects_existing(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "ok.txt").write_text("x", encoding="utf-8")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(staged, target, claim=True)


def test_publish_directory_claim_swaps_empty_reservation(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "ok.txt").write_text("x", encoding="utf-8")
    target = tmp_path / "out"
    publish_directory(staged, target, claim=True)
    assert (target / "ok.txt").read_text(encoding="utf-8") == "x"
    assert not staged.exists()


def test_publish_directory_overwrite_without_backup(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new.txt").write_text("n", encoding="utf-8")
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_text("o", encoding="utf-8")
    publish_directory(staged, target, overwrite=True)
    assert (target / "new.txt").is_file()
    assert not (target / "old.txt").exists()


def test_publish_directory_backup_restores_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new.txt").write_text("n", encoding="utf-8")
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_text("o", encoding="utf-8")

    original_replace = __import__("os").replace
    calls = {"n": 0}

    def flaky_replace(src: Path | str, dst: Path | str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected swap failure")
        original_replace(src, dst)

    monkeypatch.setattr("mspc.data.atomic_publish.os.replace", flaky_replace)
    with pytest.raises(OSError, match="injected swap failure"):
        publish_directory(staged, target, overwrite=True, backup_on_overwrite=True)
    assert (target / "old.txt").read_text(encoding="utf-8") == "o"


def test_publish_file(tmp_path: Path) -> None:
    staged = tmp_path / "staged.txt"
    staged.write_text("hello", encoding="utf-8")
    target = tmp_path / "final.txt"
    publish_file(staged, target)
    assert target.read_text(encoding="utf-8") == "hello"
