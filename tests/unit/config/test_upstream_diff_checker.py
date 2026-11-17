from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "audit" / "check_upstream_diff.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_upstream_diff", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, *, mode: str = "exact-copy") -> tuple[Path, Path, Path, object]:
    tool = _load_tool()
    root = tmp_path / "clean-repository"
    source_root = tmp_path / "external-upstream-checkout"
    checkout = source_root / "fixture-repo"
    upstream = checkout / "upstream"
    local = root / "local"
    upstream.mkdir(parents=True)
    local.mkdir(parents=True)
    (upstream / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (local / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    license_path = root / "licenses" / "LICENSE"
    license_path.parent.mkdir()
    license_path.write_text("Apache License fixture\n", encoding="utf-8")
    (checkout / "LICENSE").write_text("Apache License fixture\n", encoding="utf-8")
    git_bin = shutil.which("git")
    assert git_bin is not None
    subprocess.run([git_bin, "init", "-q", str(checkout)], check=True)
    subprocess.run([git_bin, "-C", str(checkout), "add", "upstream", "LICENSE"], check=True)
    subprocess.run(
        [
            git_bin,
            "-C",
            str(checkout),
            "-c",
            "user.name=MSPC Test",
            "-c",
            "user.email=mspc-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    base_commit = subprocess.run(
        [git_bin, "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    imports = {
        "mode": mode,
        "upstream_path": "upstream",
        "local_path": "local",
    }
    if mode == "exact-copy":
        imports["expected_sha256"] = tool.stable_tree_sha256(local)
    else:
        (local / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
        patch = root / "patches" / "local.patch"
        patch.parent.mkdir()
        patch.write_text(
            "diff --git a/model.py b/model.py\n"
            "--- a/model.py\n"
            "+++ b/model.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n",
            encoding="utf-8",
        )
        imports.update(
            expected_sha256=tool.normalized_diff_sha256(upstream, local),
            patch_path="patches/local.patch",
            patch_sha256=_sha256(patch),
        )
    manifest = root / "upstreams.toml"
    lines = [
        "schema_version = 1",
        "",
        "[[upstreams]]",
        'kind = "fork"',
        'name = "fixture"',
        'url = "https://example.com/upstream.git"',
        f'base_commit = "{base_commit}"',
        f'fork_commit = "{"b" * 40}"',
        'license = "Apache-2.0"',
        'license_upstream_path = "LICENSE"',
        'license_path = "licenses/LICENSE"',
        f'license_sha256 = "{_sha256(license_path)}"',
        'checkout_path = "fixture-repo"',
        "",
        "[[upstreams.imports]]",
    ]
    for key, value in imports.items():
        lines.append(f'{key} = "{value}"')
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, source_root, manifest, tool


def _run(
    root: Path,
    source_root: Path,
    manifest: Path,
    *,
    git_bin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    selected_git = git_bin or shutil.which("git")
    assert selected_git is not None
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--source-root",
            str(source_root),
            "--git-bin",
            selected_git,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ["exact-copy", "normalized-diff"])
def test_checker_passes_and_emits_machine_readable_json(tmp_path: Path, mode: str) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path, mode=mode)
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["upstreams"][0]["imports"][0]["mode"] == mode


def test_dependency_pin_requires_no_import_boundary(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    dependency = text.replace('kind = "fork"', 'kind = "dependency"').split(
        "\n[[upstreams.imports]]", 1
    )[0]
    manifest.write_text(dependency + "\n", encoding="utf-8")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["upstreams"][0]["kind"] == "dependency"
    assert report["upstreams"][0]["imports"] == []


def test_checker_rejects_unknown_upstream_filter(tmp_path: Path) -> None:
    root, source_root, manifest, tool = _fixture(tmp_path)
    git_bin = shutil.which("git")
    assert git_bin is not None

    with pytest.raises(tool.ContractError, match="--upstream-name"):
        tool.verify_manifest(
            manifest,
            root,
            source_root,
            Path(git_bin),
            upstream_names=("not-declared",),
        )


def test_hash_mismatch_is_parity_failure(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    (root / "local" / "model.py").write_text("VALUE = 3\n", encoding="utf-8")
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "parity-fail"


def test_python_cache_artifacts_do_not_change_source_boundary(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    for boundary in (root / "local", source_root / "fixture-repo" / "upstream"):
        cache = boundary / "__pycache__"
        cache.mkdir()
        (cache / "model.cpython-312.pyc").write_bytes(b"generated cache")
        (boundary / "legacy.pyo").write_bytes(b"generated cache")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unregistered_source_file_still_changes_boundary(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    (root / "local" / "unexpected.py").write_text("VALUE = 2\n", encoding="utf-8")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "parity-fail"


def test_patch_that_cannot_rebuild_local_tree_is_parity_failure(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path, mode="normalized-diff")
    patch = root / "patches" / "local.patch"
    patch.write_text(patch.read_text().replace("VALUE = 2", "VALUE = 9"), encoding="utf-8")
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace(
            next(line for line in text.splitlines() if line.startswith("patch_sha256 = ")),
            f'patch_sha256 = "{_sha256(patch)}"',
        ),
        encoding="utf-8",
    )
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "parity-fail"
    assert "patch rebuild mismatch" in report["error"]


def test_normalized_diff_rebuilds_a_single_file(tmp_path: Path) -> None:
    root, source_root, manifest, tool = _fixture(tmp_path, mode="normalized-diff")
    checkout_file = source_root / "fixture-repo" / "upstream" / "model.py"
    local_file = root / "local" / "model.py"
    text = manifest.read_text(encoding="utf-8")
    text = text.replace('upstream_path = "upstream"', 'upstream_path = "upstream/model.py"')
    text = text.replace('local_path = "local"', 'local_path = "local/model.py"')
    text = text.replace(
        next(line for line in text.splitlines() if line.startswith("expected_sha256 = ")),
        f'expected_sha256 = "{tool.normalized_diff_sha256(checkout_file, local_file)}"',
    )
    manifest.write_text(text, encoding="utf-8")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_materializes_patched_runtime_without_local_vendor_tree(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path, mode="normalized-diff")
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + 'materialize_path = "runtime/upstream"\n',
        encoding="utf-8",
    )
    shutil.rmtree(root / "local")
    materialize_root = tmp_path / "external-runtime"
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--source-root",
            str(source_root),
            "--git-bin",
            shutil.which("git") or "git",
            "--materialize-root",
            str(materialize_root),
            "--materialize-collision",
            "error",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (materialize_root / "runtime/upstream/model.py").read_text() == "VALUE = 2\n"
    result = json.loads(completed.stdout)["upstreams"][0]["imports"][0]
    assert result["materialized_sha256"]


def test_materialized_only_import_requires_explicit_materialization(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path, mode="normalized-diff")
    text = manifest.read_text(encoding="utf-8")
    text = text.replace('local_path = "local"\n', "")
    text += 'materialize_path = "runtime/upstream"\n'
    manifest.write_text(text, encoding="utf-8")
    shutil.rmtree(root / "local")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 2
    assert "--materialize-root" in json.loads(completed.stdout)["error"]


def test_materializes_inside_parent_git_worktree_uses_staging_directory(
    tmp_path: Path,
) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path, mode="normalized-diff")
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + 'materialize_path = "runtime/upstream"\n',
        encoding="utf-8",
    )
    shutil.rmtree(root / "local")
    git_bin = shutil.which("git") or "git"
    subprocess.run([git_bin, "init", "-q", str(root)], check=True)
    materialize_root = root / ".runtime"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--source-root",
            str(source_root),
            "--git-bin",
            git_bin,
            "--materialize-root",
            str(materialize_root),
            "--materialize-collision",
            "error",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (materialize_root / "runtime/upstream/model.py").read_text() == "VALUE = 2\n"
    assert not (root / "model.py").exists()


def test_materialization_still_verifies_non_runtime_local_boundaries(
    tmp_path: Path,
) -> None:
    root, source_root, manifest, tool = _fixture(tmp_path, mode="normalized-diff")
    local_copy = root / "local-copy"
    shutil.copytree(source_root / "fixture-repo" / "upstream", local_copy)
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + 'materialize_path = "runtime/upstream"\n'
        + "\n[[upstreams.imports]]\n"
        + 'mode = "exact-copy"\n'
        + 'upstream_path = "upstream"\n'
        + 'local_path = "local-copy"\n'
        + f'expected_sha256 = "{tool.stable_tree_sha256(local_copy)}"\n',
        encoding="utf-8",
    )
    (local_copy / "model.py").write_text("VALUE = 3\n", encoding="utf-8")
    materialize_root = tmp_path / "external-runtime"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--source-root",
            str(source_root),
            "--git-bin",
            shutil.which("git") or "git",
            "--materialize-root",
            str(materialize_root),
            "--materialize-collision",
            "error",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "parity-fail"


def test_missing_path_is_contract_error(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    (root / "local" / "model.py").unlink()
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 1  # the directory exists but its stable tree no longer matches
    manifest.write_text(manifest.read_text().replace('local_path = "local"', 'local_path = "absent"'))
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "contract-error"


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ('name = "fixture"', 'name = "fixture"\nunexpected = true'),
        ('mode = "exact-copy"', 'mode = "mystery"'),
    ],
)
def test_unknown_key_or_mode_is_contract_error(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    manifest.write_text(manifest.read_text().replace(needle, replacement), encoding="utf-8")
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "contract-error"


def test_license_hash_mismatch_is_parity_failure(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    (root / "licenses" / "LICENSE").write_text("changed\n", encoding="utf-8")
    completed = _run(root, source_root, manifest)
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "parity-fail"
    assert "license_sha256 mismatch" in report["error"]


def test_license_copy_must_match_pinned_checkout(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    local_license = root / "licenses" / "LICENSE"
    local_license.write_text("different but explicitly hashed\n", encoding="utf-8")
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        next(line for line in text.splitlines() if line.startswith("license_sha256 = ")),
        f'license_sha256 = "{_sha256(local_license)}"',
    )
    manifest.write_text(text, encoding="utf-8")

    completed = _run(root, source_root, manifest)

    assert completed.returncode == 1
    assert "license copy differs from pinned checkout" in json.loads(completed.stdout)["error"]


def test_non_executable_git_is_machine_readable_contract_error(tmp_path: Path) -> None:
    root, source_root, manifest, _ = _fixture(tmp_path)
    fake_git = tmp_path / "not-executable-git"
    fake_git.write_text("not executable\n", encoding="utf-8")
    fake_git.chmod(0o600)
    completed = _run(root, source_root, manifest, git_bin=str(fake_git))
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "contract-error"
    assert "cannot execute --git-bin" in report["error"]


def test_cli_usage_error_returns_two() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL)], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "contract-error"
    assert "CLI usage error" in report["error"]
