"""Single owner for staged-directory / staged-file atomic publish.

Data producers used to copy 16 near-identical ``_publish`` helpers. Semantics
drifted (rmtree vs backup vs empty-name claim). Call sites must go through this
module.

``src/mspc/data/multishot_media_ready.py`` still has a local ``_publish_dir``
because an HQ ProcessPool agent owns that file this window.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from mspc.data.io import fsync_directory


def publish_file(staged: Path, target: Path) -> None:
    """Atomically replace ``target`` with a staged file."""

    os.replace(staged, target)


def publish_directory(
    staged: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    backup_on_overwrite: bool = False,
    claim: bool = False,
) -> None:
    """Publish a fully-built staged directory to ``output_dir``.

    ``claim=True``: mkdir an empty final name first so a concurrent writer
    cannot steal the path, then replace only that empty claim.
    ``backup_on_overwrite=True``: move the existing directory aside and restore
    it if the swap fails.
    Otherwise: rmtree when ``overwrite`` else raise ``FileExistsError``.
    """

    if claim and (overwrite or backup_on_overwrite):
        raise ValueError(
            "publish_directory cannot combine claim=True with overwrite"
        )
    if claim:
        _publish_claimed(staged, output_dir)
        return
    if backup_on_overwrite:
        _publish_with_backup(staged, output_dir, overwrite=overwrite)
        return
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.replace(staged, output_dir)


def _publish_claimed(staged: Path, output_dir: Path) -> None:
    try:
        output_dir.mkdir()
    except FileExistsError:
        raise FileExistsError(output_dir) from None
    fsync_directory(output_dir.parent)
    try:
        os.replace(staged, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        try:
            output_dir.rmdir()
            fsync_directory(output_dir.parent)
        except OSError:
            pass
        raise


def _publish_with_backup(
    staged: Path, output_dir: Path, *, overwrite: bool
) -> None:
    backup: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        backup = output_dir.with_name(f".{output_dir.name}.backup.{os.getpid()}")
        if backup.exists():
            raise FileExistsError(backup)
        os.replace(output_dir, backup)
    try:
        os.replace(staged, output_dir)
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)
