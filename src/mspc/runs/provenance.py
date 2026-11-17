"""Deterministic Git provenance without recording diff contents."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    dirty: bool
    diff_hash: str | None

    def __post_init__(self) -> None:
        if not self.commit:
            raise ValueError("git commit must be explicit")
        if self.dirty and self.diff_hash is None:
            raise ValueError("dirty Git provenance requires diff_hash")
        if not self.dirty and self.diff_hash is not None:
            raise ValueError("clean Git provenance must not include diff_hash")
        if self.diff_hash is not None and (
            len(self.diff_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.diff_hash)
        ):
            raise ValueError("diff_hash must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"commit": self.commit, "dirty": self.dirty}
        if self.diff_hash is not None:
            result["diff_hash"] = self.diff_hash
        return result


def _git(repo_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail or exc}") from exc


def _hash_untracked(hasher: "hashlib._Hash", repo_root: Path) -> bool:
    raw = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted(part for part in raw.split(b"\0") if part)
    for encoded in paths:
        relative = os.fsdecode(encoded)
        path = repo_root / relative
        hasher.update(b"untracked\0" + encoded + b"\0")
        if path.is_symlink():
            hasher.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        else:
            hasher.update(b"non-file")
    return bool(paths)


def capture_git_provenance(repo_root: str | Path) -> GitProvenance:
    root = Path(repo_root).resolve(strict=True)
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = _git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    hasher = hashlib.sha256()
    hasher.update(b"tracked\0")
    hasher.update(tracked_diff)
    has_untracked = _hash_untracked(hasher, root)
    dirty = bool(tracked_diff) or has_untracked
    return GitProvenance(
        commit=commit,
        dirty=dirty,
        diff_hash=hasher.hexdigest() if dirty else None,
    )
