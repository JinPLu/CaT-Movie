#!/usr/bin/env python3
"""Verify checked-in upstream provenance and local fork fingerprints offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from mspc.config.toml_compat import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse


SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = {"schema_version", "upstreams"}
UPSTREAM_KEYS = {
    "kind",
    "name",
    "url",
    "base_commit",
    "fork_commit",
    "license",
    "license_upstream_path",
    "license_path",
    "license_sha256",
    "checkout_path",
    "imports",
}
IMPORT_KEYS = {
    "mode",
    "upstream_path",
    "local_path",
    "expected_sha256",
    "patch_path",
    "patch_sha256",
    "materialize_path",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class ContractError(ValueError):
    """The manifest or its required local inputs violate the contract."""


class ParityError(RuntimeError):
    """A checked artifact does not match its declared fingerprint."""


class ContractArgumentParser(argparse.ArgumentParser):
    """Convert CLI usage failures into the same machine-readable contract path."""

    def error(self, message: str) -> NoReturn:
        raise ContractError(f"CLI usage error: {message}")


@dataclass(frozen=True)
class ImportSpec:
    mode: str
    upstream_path: str
    local_path: str | None
    expected_sha256: str
    patch_path: str | None = None
    patch_sha256: str | None = None
    materialize_path: str | None = None


@dataclass(frozen=True)
class UpstreamSpec:
    kind: str
    name: str
    url: str
    base_commit: str
    fork_commit: str
    license: str
    license_upstream_path: str
    license_path: str
    license_sha256: str
    checkout_path: str
    imports: tuple[ImportSpec, ...]


def _contract(message: str) -> NoReturn:
    raise ContractError(message)


def _require_table(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _contract(f"{context} must be a TOML table")
    return value


def _require_string(table: dict[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        _contract(f"{context}.{key} must be a non-empty string")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        _contract(f"{context} has unknown keys: {', '.join(unknown)}")


def _require_sha256(value: str, context: str) -> str:
    if not HEX_SHA256.fullmatch(value):
        _contract(f"{context} must be a lowercase 64-character SHA256")
    return value


def _parse_import(raw: Any, context: str) -> ImportSpec:
    table = _require_table(raw, context)
    _reject_unknown(table, IMPORT_KEYS, context)
    mode = _require_string(table, "mode", context)
    if mode not in {"exact-copy", "normalized-diff"}:
        _contract(f"{context}.mode has unsupported value: {mode}")
    raw_local_path = table.get("local_path")
    if raw_local_path is None:
        local_path = None
    elif isinstance(raw_local_path, str) and raw_local_path:
        local_path = raw_local_path
    else:
        _contract(f"{context}.local_path must be a non-empty string when provided")
    spec = ImportSpec(
        mode=mode,
        upstream_path=_require_string(table, "upstream_path", context),
        local_path=local_path,
        expected_sha256=_require_sha256(
            _require_string(table, "expected_sha256", context),
            f"{context}.expected_sha256",
        ),
        patch_path=table.get("patch_path"),
        patch_sha256=table.get("patch_sha256"),
        materialize_path=table.get("materialize_path"),
    )
    if spec.mode == "exact-copy":
        if spec.patch_path is not None or spec.patch_sha256 is not None:
            _contract(f"{context} exact-copy must not declare patch fields")
    else:
        if not isinstance(spec.patch_path, str) or not spec.patch_path:
            _contract(f"{context}.patch_path must be set for normalized-diff")
        if not isinstance(spec.patch_sha256, str):
            _contract(f"{context}.patch_sha256 must be set for normalized-diff")
        _require_sha256(spec.patch_sha256, f"{context}.patch_sha256")
    if spec.materialize_path is not None:
        if not isinstance(spec.materialize_path, str) or not spec.materialize_path:
            _contract(f"{context}.materialize_path must be a non-empty relative path")
        relative = Path(spec.materialize_path)
        if relative.is_absolute() or ".." in relative.parts:
            _contract(f"{context}.materialize_path must remain inside --materialize-root")
    if spec.local_path is None and spec.materialize_path is None:
        _contract(
            f"{context}.local_path is required unless materialize_path provides the runtime boundary"
        )
    return spec


def _parse_upstream(raw: Any, index: int) -> UpstreamSpec:
    context = f"upstreams[{index}]"
    table = _require_table(raw, context)
    _reject_unknown(table, UPSTREAM_KEYS, context)
    url = _require_string(table, "url", context)
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
        _contract(f"{context}.url must be an absolute HTTP(S) URL")
    base_commit = _require_string(table, "base_commit", context)
    fork_commit = _require_string(table, "fork_commit", context)
    if not FULL_COMMIT.fullmatch(base_commit):
        _contract(f"{context}.base_commit must be a lowercase full 40-character commit")
    if not FULL_COMMIT.fullmatch(fork_commit):
        _contract(f"{context}.fork_commit must be a lowercase full 40-character commit")
    kind = _require_string(table, "kind", context)
    if kind not in {"dependency", "fork"}:
        _contract(f"{context}.kind must be dependency or fork")
    raw_imports = table.get("imports", [])
    if not isinstance(raw_imports, list):
        _contract(f"{context}.imports must be an array of tables")
    if kind == "fork" and not raw_imports:
        _contract(f"{context}.imports must be non-empty for a fork")
    if kind == "dependency" and raw_imports:
        _contract(f"{context}.imports must be absent for a dependency")
    return UpstreamSpec(
        kind=kind,
        name=_require_string(table, "name", context),
        url=url,
        base_commit=base_commit,
        fork_commit=fork_commit,
        license=_require_string(table, "license", context),
        license_upstream_path=_require_string(table, "license_upstream_path", context),
        license_path=_require_string(table, "license_path", context),
        license_sha256=_require_sha256(
            _require_string(table, "license_sha256", context),
            f"{context}.license_sha256",
        ),
        checkout_path=_require_string(table, "checkout_path", context),
        imports=tuple(
            _parse_import(item, f"{context}.imports[{item_index}]")
            for item_index, item in enumerate(raw_imports)
        ),
    )


def load_manifest(path: Path) -> tuple[UpstreamSpec, ...]:
    if not path.is_file():
        _contract(f"manifest does not exist or is not a file: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        _contract(f"cannot read TOML manifest {path}: {exc}")
    table = _require_table(raw, "manifest")
    _reject_unknown(table, TOP_LEVEL_KEYS, "manifest")
    if table.get("schema_version") != SCHEMA_VERSION:
        _contract(f"manifest.schema_version must equal {SCHEMA_VERSION}")
    upstreams = table.get("upstreams")
    if not isinstance(upstreams, list) or not upstreams:
        _contract("manifest.upstreams must be a non-empty array of tables")
    specs = tuple(_parse_upstream(item, index) for index, item in enumerate(upstreams))
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        _contract("manifest upstream names must be unique")
    return specs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_python_cache_artifact(path: Path, boundary: Path) -> bool:
    """Exclude only interpreter-generated cache files from source boundaries."""
    relative = path.relative_to(boundary)
    return "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}


def _checked_path(root: Path, declared: str, context: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        _contract(f"{context} must be a relative path contained by --root")
    unresolved = root / relative
    if not unresolved.exists() and not unresolved.is_symlink():
        _contract(f"{context} does not exist: {unresolved}")
    root_resolved = root.resolve(strict=True)
    try:
        path = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _contract(f"{context} cannot be resolved: {unresolved}: {exc}")
    try:
        path.relative_to(root_resolved)
    except ValueError:
        _contract(f"{context} escapes --root through a symbolic link: {unresolved}")
    return path


def _tree_records(path: Path) -> list[tuple[str, str, str]]:
    if path.is_symlink():
        _contract(f"symbolic links are not accepted in upstream boundaries: {path}")
    if path.is_file():
        return [("file", ".", file_sha256(path))]
    if not path.is_dir():
        _contract(f"upstream boundary is not a regular file or directory: {path}")
    records: list[tuple[str, str, str]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if _is_python_cache_artifact(child, path):
            continue
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            _contract(f"symbolic links are not accepted in upstream boundaries: {child}")
        if child.is_file():
            records.append(("file", relative, file_sha256(child)))
        elif not child.is_dir():
            _contract(f"special filesystem entry is not accepted: {child}")
    return records


def stable_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for kind, relative, content_hash in _tree_records(path):
        digest.update(f"{kind}\0{relative}\0{content_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_apply_environment(cwd: Path) -> dict[str, str]:
    """Prevent `git apply` from binding to an unrelated parent worktree.

    Materialization staging directories are often created under ignored runtime
    folders inside this repository.  Without a ceiling, `git apply` discovers
    the parent Storyboard worktree and interprets patch paths relative to that
    root instead of the staging directory.
    """

    env = os.environ.copy()
    ceiling = str(cwd.resolve().parent)
    existing = env.get("GIT_CEILING_DIRECTORIES")
    env["GIT_CEILING_DIRECTORIES"] = (
        ceiling if not existing else f"{ceiling}{os.pathsep}{existing}"
    )
    return env


def _normalized_content_hash(path: Path) -> tuple[str, str]:
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return "binary", hashlib.sha256(content).hexdigest()
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return "text", hashlib.sha256(normalized).hexdigest()


def _file_map(path: Path) -> dict[str, tuple[str, str]]:
    if path.is_symlink():
        _contract(f"symbolic links are not accepted in upstream boundaries: {path}")
    if path.is_file():
        return {".": _normalized_content_hash(path)}
    if not path.is_dir():
        _contract(f"upstream boundary is not a regular file or directory: {path}")
    result: dict[str, tuple[str, str]] = {}
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if _is_python_cache_artifact(child, path):
            continue
        if child.is_symlink():
            _contract(f"symbolic links are not accepted in upstream boundaries: {child}")
        if child.is_file():
            result[child.relative_to(path).as_posix()] = _normalized_content_hash(child)
        elif not child.is_dir():
            _contract(f"special filesystem entry is not accepted: {child}")
    return result


def normalized_diff_sha256(upstream: Path, local: Path) -> str:
    upstream_files = _file_map(upstream)
    local_files = _file_map(local)
    digest = hashlib.sha256()
    for relative in sorted(set(upstream_files) | set(local_files)):
        left = upstream_files.get(relative)
        right = local_files.get(relative)
        if left == right:
            continue
        left_kind, left_hash = left or ("missing", "-")
        right_kind, right_hash = right or ("missing", "-")
        record = f"{relative}\0{left_kind}\0{left_hash}\0{right_kind}\0{right_hash}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _rebuild_with_patch(
    upstream: Path,
    local: Path,
    patch: Path,
    git_bin: Path,
    context: str,
) -> str:
    if upstream.is_dir() != local.is_dir() or upstream.is_file() != local.is_file():
        _contract(f"{context} normalized-diff boundaries must have matching file types")
    with tempfile.TemporaryDirectory(prefix="mspc-upstream-rebuild-") as temporary:
        rebuilt_root = Path(temporary) / "rebuilt"
        if upstream.is_dir():
            shutil.copytree(upstream, rebuilt_root)
            rebuilt_target = rebuilt_root
        elif upstream.is_file():
            rebuilt_root.mkdir()
            rebuilt_target = rebuilt_root / upstream.name
            shutil.copy2(upstream, rebuilt_target)
        else:
            _contract(f"{context} normalized-diff boundary is not a regular file or directory")
        command = [str(git_bin), "apply", "--whitespace=nowarn", str(patch)]
        apply_env = _git_apply_environment(rebuilt_root)
        try:
            check = subprocess.run(
                [*command[:2], "--check", *command[2:]],
                cwd=rebuilt_root,
                env=apply_env,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            _contract(f"cannot execute --git-bin {git_bin}: {exc}")
        if check.returncode != 0:
            raise ParityError(
                f"{context} patch does not apply to declared upstream: "
                f"{(check.stderr or check.stdout).strip()}"
            )
        try:
            applied = subprocess.run(
                command,
                cwd=rebuilt_root,
                env=apply_env,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            _contract(f"cannot execute --git-bin {git_bin}: {exc}")
        if applied.returncode != 0:
            raise ParityError(
                f"{context} patch application failed after successful check: "
                f"{(applied.stderr or applied.stdout).strip()}"
            )
        rebuilt_hash = stable_tree_sha256(rebuilt_target)
        local_hash = stable_tree_sha256(local)
        if rebuilt_hash != local_hash:
            raise ParityError(
                f"{context} patch rebuild mismatch: rebuilt {rebuilt_hash}, local {local_hash}"
            )
        return rebuilt_hash


def _materialization_target(root: Path, declared: str, context: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        _contract(f"{context} must be a relative path inside --materialize-root")
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve(strict=True)
    target = root_resolved / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.parent.resolve(strict=True).relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        _contract(f"{context} escapes --materialize-root: {target}: {exc}")
    return target


def _copy_boundary(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _materialize_import(
    *,
    upstream: Path,
    spec: ImportSpec,
    patch: Path | None,
    git_bin: Path,
    materialize_root: Path,
    collision: str,
    context: str,
) -> Path:
    if spec.materialize_path is None:
        _contract(f"{context}.materialize_path is required for materialization")
    target = _materialization_target(
        materialize_root,
        spec.materialize_path,
        f"{context}.materialize_path",
    )
    if target.exists() or target.is_symlink():
        if collision == "error":
            _contract(f"materialization target already exists: {target}")
        if target.is_symlink():
            _contract(f"materialization target must not be a symbolic link: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    with tempfile.TemporaryDirectory(
        prefix="mspc-upstream-materialize-",
        dir=materialize_root,
    ) as temporary:
        staging_root = Path(temporary) / "boundary"
        if upstream.is_dir():
            shutil.copytree(upstream, staging_root)
            rebuilt = staging_root
            patch_cwd = staging_root
        elif upstream.is_file():
            staging_root.mkdir()
            rebuilt = staging_root / upstream.name
            shutil.copy2(upstream, rebuilt)
            patch_cwd = staging_root
        else:
            _contract(f"{context}.upstream_path is not a regular file or directory")

        if spec.mode == "normalized-diff":
            if patch is None:
                _contract(f"{context}.patch_path is required for normalized-diff")
            command = [str(git_bin), "apply", "--whitespace=nowarn", str(patch)]
            apply_env = _git_apply_environment(patch_cwd)
            try:
                check = subprocess.run(
                    [*command[:2], "--check", *command[2:]],
                    cwd=patch_cwd,
                    env=apply_env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                applied = (
                    subprocess.run(
                        command,
                        cwd=patch_cwd,
                        env=apply_env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if check.returncode == 0
                    else None
                )
            except OSError as exc:
                _contract(f"cannot execute --git-bin {git_bin}: {exc}")
            if check.returncode != 0:
                raise ParityError(
                    f"{context} patch does not apply during materialization: "
                    f"{(check.stderr or check.stdout).strip()}"
                )
            if applied is None or applied.returncode != 0:
                raise ParityError(
                    f"{context} patch application failed during materialization: "
                    f"{((applied.stderr or applied.stdout) if applied else '').strip()}"
                )
        _copy_boundary(rebuilt, target)
    return target


def _verified_checkout_head(checkout: Path, git_bin: Path, expected: str, context: str) -> str:
    try:
        completed = subprocess.run(
            [str(git_bin), "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        _contract(f"cannot execute --git-bin {git_bin}: {exc}")
    if completed.returncode != 0:
        _contract(
            f"{context} is not a readable git checkout: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    actual = completed.stdout.strip()
    if actual != expected:
        raise ParityError(f"{context} HEAD mismatch: expected {expected}, got {actual}")
    return actual


def verify_manifest(
    manifest: Path,
    root: Path,
    source_root: Path,
    git_bin: Path,
    materialize_root: Path | None = None,
    materialize_collision: str | None = None,
    upstream_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not root.is_dir():
        _contract(f"--root does not exist or is not a directory: {root}")
    if not source_root.is_dir():
        _contract(f"--source-root does not exist or is not a directory: {source_root}")
    if not git_bin.is_file():
        _contract(f"--git-bin does not exist or is not a regular file: {git_bin}")
    specs = load_manifest(manifest)
    if upstream_names is not None:
        requested = set(upstream_names)
        available = {spec.name for spec in specs}
        unknown = sorted(requested - available)
        if unknown:
            _contract(f"--upstream-name does not identify a declared upstream: {', '.join(unknown)}")
        specs = tuple(spec for spec in specs if spec.name in requested)
    if (materialize_root is None) != (materialize_collision is None):
        _contract("--materialize-root and --materialize-collision must be provided together")
    if materialize_collision not in {None, "error", "replace"}:
        _contract("--materialize-collision must be error or replace")
    results: list[dict[str, Any]] = []
    for upstream_index, spec in enumerate(specs):
        context = f"upstreams[{upstream_index}]"
        checkout = _checked_path(source_root, spec.checkout_path, f"{context}.checkout_path")
        if not checkout.is_dir():
            _contract(f"{context}.checkout_path must identify a git checkout directory: {checkout}")
        checkout_head = _verified_checkout_head(
            checkout, git_bin, spec.base_commit, f"{context}.checkout_path"
        )
        license_path = _checked_path(root, spec.license_path, f"{context}.license_path")
        if not license_path.is_file():
            _contract(f"{context}.license_path must identify a regular file: {license_path}")
        actual_license_sha256 = file_sha256(license_path)
        if actual_license_sha256 != spec.license_sha256:
            raise ParityError(
                f"{context}.license_sha256 mismatch: expected {spec.license_sha256}, "
                f"got {actual_license_sha256}"
            )
        upstream_license_path = _checked_path(
            checkout,
            spec.license_upstream_path,
            f"{context}.license_upstream_path",
        )
        if not upstream_license_path.is_file():
            _contract(
                f"{context}.license_upstream_path must identify a regular file: "
                f"{upstream_license_path}"
            )
        upstream_license_sha256 = file_sha256(upstream_license_path)
        if upstream_license_sha256 != actual_license_sha256:
            raise ParityError(
                f"{context} license copy differs from pinned checkout: upstream "
                f"{upstream_license_sha256}, local {actual_license_sha256}"
            )
        import_results: list[dict[str, str]] = []
        for import_index, import_spec in enumerate(spec.imports):
            import_context = f"{context}.imports[{import_index}]"
            upstream_path = _checked_path(
                checkout, import_spec.upstream_path, f"{import_context}.upstream_path"
            )
            patch_path = None
            patch_hash = None
            if import_spec.mode == "normalized-diff":
                patch_path = _checked_path(
                    root, import_spec.patch_path or "", f"{import_context}.patch_path"
                )
                if not patch_path.is_file():
                    _contract(f"{import_context}.patch_path must identify a regular file: {patch_path}")
                patch_hash = file_sha256(patch_path)
                if patch_hash != import_spec.patch_sha256:
                    raise ParityError(
                        f"{import_context}.patch_sha256 mismatch: expected "
                        f"{import_spec.patch_sha256}, got {patch_hash}"
                    )
            if materialize_root is not None and import_spec.materialize_path is not None:
                local_path = _materialize_import(
                    upstream=upstream_path,
                    spec=import_spec,
                    patch=patch_path,
                    git_bin=git_bin,
                    materialize_root=materialize_root,
                    collision=materialize_collision or "error",
                    context=import_context,
                )
                # Relative to --materialize-root: the root is a caller-supplied
                # absolute path, so recording it would make an otherwise
                # reproducible report differ per machine and per checkout.
                materialized_path = str(local_path.relative_to(materialize_root))
            else:
                if import_spec.local_path is None:
                    _contract(
                        f"{import_context} has no local boundary; run with --materialize-root"
                    )
                local_path = _checked_path(root, import_spec.local_path, f"{import_context}.local_path")
                materialized_path = None
            if import_spec.mode == "exact-copy":
                upstream_hash = stable_tree_sha256(upstream_path)
                local_hash = stable_tree_sha256(local_path)
                if upstream_hash != local_hash:
                    raise ParityError(
                        f"{import_context} exact-copy mismatch: upstream {upstream_hash}, "
                        f"local {local_hash}"
                    )
                actual = local_hash
            else:
                actual = normalized_diff_sha256(upstream_path, local_path)
                rebuilt_hash = stable_tree_sha256(local_path)
                if materialized_path is None:
                    rebuilt_hash = _rebuild_with_patch(
                        upstream_path, local_path, patch_path, git_bin, import_context
                    )
            if actual != import_spec.expected_sha256:
                raise ParityError(
                    f"{import_context}.expected_sha256 mismatch: expected "
                    f"{import_spec.expected_sha256}, got {actual}"
                )
            result = {"mode": import_spec.mode, "actual_sha256": actual}
            if patch_hash is not None:
                result["patch_sha256"] = patch_hash
                result["rebuilt_sha256"] = rebuilt_hash
            if materialized_path is not None:
                result["materialized_path"] = materialized_path
                result["materialized_sha256"] = stable_tree_sha256(local_path)
            import_results.append(result)
        results.append(
            {
                "kind": spec.kind,
                "name": spec.name,
                "url": spec.url,
                "base_commit": spec.base_commit,
                "verified_checkout_head": checkout_head,
                "fork_commit": spec.fork_commit,
                "license": spec.license,
                "license_sha256": actual_license_sha256,
                "upstream_license_sha256": upstream_license_sha256,
                "imports": import_results,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "status": "pass", "upstreams": results}


def build_parser() -> argparse.ArgumentParser:
    parser = ContractArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit TOML provenance manifest")
    parser.add_argument(
        "--root", type=Path, required=True, help="Clean repository root for local/license/patch paths"
    )
    parser.add_argument(
        "--materialize-root",
        type=Path,
        help="Explicit external root where declared patched runtime boundaries are rebuilt",
    )
    parser.add_argument(
        "--materialize-collision",
        choices=("error", "replace"),
        help="Required collision policy when --materialize-root is set",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Explicit external checkout root at the manifest's immutable upstream commit",
    )
    parser.add_argument(
        "--git-bin",
        type=Path,
        required=True,
        help="Explicit git executable used to verify checkout commits and rebuild patches",
    )
    parser.add_argument(
        "--upstream-name",
        action="append",
        help="Limit verification/materialization to a declared upstream name; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        report = verify_manifest(
            args.manifest,
            args.root,
            args.source_root,
            args.git_bin,
            args.materialize_root,
            args.materialize_collision,
            tuple(args.upstream_name) if args.upstream_name else None,
        )
    except ContractError as exc:
        report = {"schema_version": SCHEMA_VERSION, "status": "contract-error", "error": str(exc)}
        print(json.dumps(report, sort_keys=True))
        return 2
    except ParityError as exc:
        report = {"schema_version": SCHEMA_VERSION, "status": "parity-fail", "error": str(exc)}
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
