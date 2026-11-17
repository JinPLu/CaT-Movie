"""冻结训练源码的真实Git身份；环境声明仅作断言。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def require_clean_worktree(root: str | Path, *, expected_sha: str | None = None) -> str:
    """核独立checkout/worktree的实际HEAD与tracked clean，不信archive声明。

    Git 2.34.1只从system/global读取safe.directory。每次核验用独占临时
    global文件信任一个物理根，不改用户配置；禁止继承重定向仓库/index的环境。
    """
    if expected_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("expected repository identity must be a lowercase 40-character Git SHA")
    physical = Path(root).resolve(strict=True)
    if not physical.is_dir() or not (physical / ".git").exists():
        raise ValueError(f"frozen source requires a real Git checkout or worktree: {physical}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="storyboard-git-safe-") as safe:
        safe.write("[safe]\n\tdirectory = " + json.dumps(str(physical), ensure_ascii=False) + "\n")
        safe.flush()
        environment = dict(os.environ)
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
            environment.pop(name, None)
        environment.update(GIT_CONFIG_GLOBAL=safe.name, GIT_OPTIONAL_LOCKS="0")

        def git(*args: str) -> str:
            try:
                return subprocess.run(["git", *args], cwd=physical, env=environment,
                    check=True, capture_output=True, text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f"cannot verify frozen Git source: {physical}") from error

        toplevel = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
        if toplevel != physical or git("rev-parse", "--is-inside-work-tree") != "true":
            raise ValueError(f"frozen source must be the actual Git worktree root: {physical}")
        actual = git("rev-parse", "--verify", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40}", actual):
            raise ValueError(f"frozen source has no full Git commit identity: {physical}")
        if expected_sha is not None and actual != expected_sha:
            raise ValueError(f"frozen source Git HEAD differs: {physical}: expected {expected_sha}, actual {actual}")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"{physical} has modified tracked files")
        return actual
