"""Stable repository paths for tests moved under categorized directories."""

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "tests").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate Storyboard repository root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
