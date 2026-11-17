"""The TransNetV2 LD_LIBRARY_PATH hazard must be refused before any work starts.

TensorFlow finds CUDA through the scoring env's own ``nvidia/*/lib`` directories.
Without them on ``LD_LIBRARY_PATH`` it reports 0 GPUs and ``_bind_tensorflow_device``
raises -- but the TransNet engine is built lazily, inside the per-case loop, so
that raise lands only after every video has been decoded (and, under the
sidecar, generated). These tests pin the preflight that turns that into a
refusal measured in seconds, and pin the metric-id list it keys off to the real
dispatch so the two cannot drift apart.

The production engine now runs that check in a TF-only subprocess; the parent
must never import tensorflow.  The in-process helper keeps the same strength
and is what the worker calls.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mspc.evaluation import renderer_qualification_official_backend as backend
from mspc.evaluation.renderer_qualification_public_metrics import (
    DIAGNOSTIC_METRIC_IDS,
    PUBLIC_METRIC_IDS,
    MetricMetadata,
    PublicMetricCase,
)


ROOT = Path(__file__).resolve().parents[3]
RUN_LIB = ROOT / "run/lib/tensorflow_libs.sh"


class _StubTensorflowConfig:
    def __init__(self, gpus: list[object]) -> None:
        self._gpus = gpus

    def list_physical_devices(self, kind: str) -> list[object]:
        assert kind == "GPU"
        return list(self._gpus)


class _StubTensorflow:
    def __init__(self, gpus: list[object]) -> None:
        self.config = _StubTensorflowConfig(gpus)


@pytest.fixture
def stub_tensorflow(monkeypatch: pytest.MonkeyPatch):
    """Install an importable fake ``tensorflow`` with a chosen GPU list."""

    def install(gpus: list[object]) -> None:
        monkeypatch.setitem(sys.modules, "tensorflow", _StubTensorflow(gpus))

    return install


def test_preflight_refuses_a_cuda_transnet_run_when_tensorflow_sees_no_gpu(
    stub_tensorflow,
) -> None:
    stub_tensorflow([])
    with pytest.raises(RuntimeError) as excinfo:
        backend.preflight_tensorflow_gpu("cuda")
    message = str(excinfo.value)
    # The remediation has to be actionable on its own: what went wrong, the exact
    # command that seeds the libs for this very interpreter, and why setting it
    # from here would not work. It points at run/lib/tensorflow_libs.sh rather than
    # restating how to find the libs, so there is only one such description to keep.
    assert "0 GPU(s)" in message
    assert "LD_LIBRARY_PATH" in message
    assert "run/lib/tensorflow_libs.sh" in message
    assert "mspc_seed_tensorflow_libs" in message
    assert sys.executable in message


def test_preflight_passes_when_tensorflow_reports_a_gpu(stub_tensorflow) -> None:
    stub_tensorflow([object()])
    backend.preflight_tensorflow_gpu("cuda")


def test_production_factory_is_the_subprocess_engine() -> None:
    assert backend.OfficialMetricBackend.transnet_factory is backend._SubprocessTransNetEngine
    assert backend._DEFAULT_BACKEND.transnet_factory is backend._SubprocessTransNetEngine


def test_preflight_is_a_noop_without_a_tensorflow_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No tensorflow installed at all: a diagnostic-only selection must not even
    # try to import it, so gated rescoring keeps working on a host where the
    # TensorFlow side is broken.
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    backend.OfficialMetricBackend(device="cuda").preflight(
        ["dinov2_character_consistency_gated", "subject_presence", "n_pairs_detected"]
    )


def test_preflight_is_a_noop_on_cpu(stub_tensorflow) -> None:
    stub_tensorflow([])
    backend.OfficialMetricBackend(device="cpu").preflight(
        ["transnet_transition_deviation"]
    )


def test_preflight_is_a_noop_for_a_substituted_transnet_factory(
    stub_tensorflow,
) -> None:
    # The CPU contract tests swap the factory precisely so TensorFlow is never
    # imported; the preflight must not reintroduce that import behind them.
    stub_tensorflow([])
    engine = backend.OfficialMetricBackend(
        device="cuda", transnet_factory=lambda root, device: object()
    )
    engine.preflight(["transnet_transition_deviation"])


def test_preflight_reports_a_missing_tensorflow_specifically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorflow":
            raise ImportError("No module named 'tensorflow'")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match="not importable"):
        backend.preflight_tensorflow_gpu("cuda")


def test_score_metric_carries_the_preflight_for_module_callable_callers() -> None:
    # The CLI resolves a backend by "module:callable" and looks the hook up on
    # whatever it got; without this attribute the CLI silently skips the check.
    assert callable(getattr(backend.score_metric, "preflight", None))


def test_tensorflow_metric_ids_matches_what_actually_builds_the_transnet_engine(
    tmp_path: Path,
) -> None:
    """Hold the constant to the dispatch, not to a comment.

    Every supported metric id is scored with a backend whose engines are all
    recording stubs; only the ids in ``TENSORFLOW_METRIC_IDS`` may reach the
    TransNet factory. If someone routes a new metric through TransNetV2 without
    listing it, this fails rather than shipping a preflight with a hole.
    """

    supported = (
        tuple(PUBLIC_METRIC_IDS)
        + tuple(DIAGNOSTIC_METRIC_IDS)
    )
    case = _case(tmp_path)
    reached: set[str] = set()
    chose_some_engine: set[str] = set()

    for metric_id in supported:
        built: list[str] = []

        def _record(name: str):
            def factory(root: Path, device: str):
                built.append(name)
                raise _StopScoring()

            return factory

        engine = backend.OfficialMetricBackend(
            decoder=lambda path: ([object()] * 9, 15.0),
            videoscore_factory=_record("videoscore"),
            viclip_factory=_record("viclip"),
            dino_factory=_record("dino"),
            transnet_factory=_record("transnet"),
        )
        metric = MetricMetadata(metric_id=metric_id, direction="higher", unit="ratio")
        try:
            engine(metric, case)
        except _StopScoring:
            pass
        except Exception:
            # Some diagnostic columns need a pair manifest this bare case does
            # not carry and fail before choosing an engine. That is still
            # evidence they do not reach TransNetV2.
            pass
        if built:
            chose_some_engine.add(metric_id)
        if "transnet" in built:
            reached.add(metric_id)

    # Guard the guard: if the stubs stopped being reached at all, the assertion
    # below would pass vacuously and the constant would drift unnoticed.
    assert chose_some_engine, "no metric reached any engine; the stub setup is wrong"

    assert reached == set(backend.TENSORFLOW_METRIC_IDS), (
        "TENSORFLOW_METRIC_IDS is out of step with the dispatch: "
        f"reached={sorted(reached)} declared={sorted(backend.TENSORFLOW_METRIC_IDS)}"
    )


class _StopScoring(Exception):
    """Raised by a stub factory to stop scoring once the engine is chosen."""


def _case(tmp_path: Path) -> PublicMetricCase:
    """A case whose resource root and video file exist, so dispatch is reached.

    ``_engine`` checks the resource root and ``_frames`` stats the video before
    any factory runs, so a case pointing at nonexistent paths would fail before
    choosing an engine and make the dispatch survey vacuous.
    """

    resource_root = tmp_path / "resources"
    resource_root.mkdir(exist_ok=True)
    video = tmp_path / "case.mp4"
    if not video.exists():
        video.write_bytes(b"not a real video; the decoder is stubbed")
    return PublicMetricCase(
        case_id="c1",
        arm_id="a1",
        story_prompt="p",
        shots=({"shot_id": "s01", "prompt": "x", "frame_count": 9},),
        video_path=video,
        resource_root=resource_root,
    )


def test_parent_preflight_never_imports_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setattr(
        backend._SubprocessTransNetEngine,
        "preflight_device",
        staticmethod(lambda device: None),
    )
    backend.OfficialMetricBackend(device="cuda").preflight(
        ["transnet_transition_deviation"]
    )


def test_parent_preflight_surfaces_a_child_zero_gpu_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(device: str) -> None:
        raise RuntimeError(
            "TransNetV2 requested cuda but TensorFlow sees 0 GPU(s). "
            "TensorFlow finds CUDA through this environment's own nvidia/*/lib "
            "directories, which are not on LD_LIBRARY_PATH."
        )

    monkeypatch.setattr(
        backend._SubprocessTransNetEngine,
        "preflight_device",
        staticmethod(_boom),
    )
    with pytest.raises(RuntimeError, match="0 GPU") as excinfo:
        backend.OfficialMetricBackend(device="cuda").preflight(
            ["transnet_transition_deviation"]
        )
    assert "LD_LIBRARY_PATH" in str(excinfo.value)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_subprocess_engine_sends_decoded_frames_as_npy_and_returns_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "fake_worker.py"
    seen = tmp_path / "seen.json"
    worker.write_text(
        "\n".join(
            [
                "import argparse, json, sys",
                "import numpy as np",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--mode', required=True)",
                "parser.add_argument('--device', required=True)",
                "parser.add_argument('--resource-root')",
                "parser.add_argument('--frames-npy')",
                "args = parser.parse_args()",
                "if args.mode == 'preflight':",
                "    json.dump({'status': 'ok', 'device': args.device}, sys.stdout)",
                "    raise SystemExit(0)",
                "frames = np.load(args.frames_npy, allow_pickle=False)",
                f"json.dump({{'mode': args.mode, 'device': args.device, 'root': args.resource_root, 'shape': list(frames.shape)}}, open({str(seen)!r}, 'w'))",
                "json.dump({'status': 'ok', 'probabilities': [0.25] * len(frames)}, sys.stdout)",
                "print()",
            ]
        ),
        encoding="utf-8",
    )
    fake_python = tmp_path / "tf-only" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    site = tmp_path / "tf-only" / "lib" / "python3.11" / "site-packages"
    (site / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    _write_executable(
        fake_python,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        f"  printf '%s\\n' {str(site)}\n"
        "  exit 0\n"
        "fi\n"
        f"exec {sys.executable} \"$@\"\n",
    )
    monkeypatch.setenv("MSPC_TRANSNET_PYTHON", str(fake_python))
    monkeypatch.setattr(backend, "TRANSNET_WORKER", worker)
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(5)]
    engine = backend._SubprocessTransNetEngine(tmp_path, "cuda")
    values = engine.predict(frames)
    assert values == [0.25] * 5
    recorded = json.loads(seen.read_text(encoding="utf-8"))
    assert recorded["shape"] == [5, 4, 6, 3]
    assert recorded["device"] == "cuda"
    assert recorded["root"] == str(tmp_path.resolve())


def test_subprocess_engine_fails_loud_when_the_worker_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "failing_worker.py"
    worker.write_text(
        "import json, sys\n"
        "json.dump({'status': 'error', 'error': 'RuntimeError: TensorFlow sees 0 GPU(s)'}, sys.stdout)\n"
        "print()\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "tf-only" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    site = tmp_path / "tf-only" / "lib" / "python3.11" / "site-packages"
    (site / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    _write_executable(
        fake_python,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        f"  printf '%s\\n' {str(site)}\n"
        "  exit 0\n"
        "fi\n"
        f"exec {sys.executable} \"$@\"\n",
    )
    monkeypatch.setenv("MSPC_TRANSNET_PYTHON", str(fake_python))
    monkeypatch.setattr(backend, "TRANSNET_WORKER", worker)
    with pytest.raises(RuntimeError, match="0 GPU") as excinfo:
        backend._SubprocessTransNetEngine(tmp_path, "cuda").predict(
            [np.zeros((2, 2, 3), dtype=np.uint8)]
        )
    assert "TF-only worker failed" in str(excinfo.value)


def test_resolve_transnet_python_requires_an_explicit_or_registered_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MSPC_TRANSNET_PYTHON", raising=False)
    monkeypatch.delenv("MSPC_GPU_ARCH", raising=False)
    monkeypatch.delenv("MSPC_GPU_ARCH_OVERRIDE", raising=False)
    monkeypatch.delenv("MSPC_SHARED_ROOT", raising=False)
    monkeypatch.setenv("MSPC_ENV_REGISTRY", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="not readable"):
        backend.resolve_transnet_python()


def test_resolve_transnet_python_keeps_the_venv_entry_and_does_not_follow_the_binary_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    uv_base = shared / "env" / "uv_python_bernini_probe" / "cpython-3.11.15" / "bin"
    uv_base.mkdir(parents=True)
    real_python = uv_base / "python3.11"
    _write_executable(real_python, "#!/bin/sh\nexit 0\n")
    venv_bin = shared / "env" / "renderer_transnet" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(real_python)
    monkeypatch.setenv("MSPC_TRANSNET_PYTHON", str(venv_python))
    resolved = backend.resolve_transnet_python()
    assert resolved == venv_bin.resolve() / "python"
    assert resolved.name == "python"
    assert "renderer_transnet" in resolved.parts
    assert "uv_python_bernini_probe" not in resolved.parts
    assert Path(os.path.realpath(resolved)) == real_python.resolve()
    assert resolved != Path(os.path.realpath(resolved))


def test_resolve_transnet_python_registry_path_does_not_collapse_into_uv_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    uv_base = shared / "env" / "uv_python_bernini_probe" / "cpython-3.11.15" / "bin"
    uv_base.mkdir(parents=True)
    real_python = uv_base / "python3.11"
    _write_executable(real_python, "#!/bin/sh\nexit 0\n")
    venv_bin = shared / "env" / "renderer_transnet" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(real_python)
    registry = {
        "schema_version": "mspc_env_by_arch_v1",
        "shared_root_env": "MSPC_SHARED_ROOT",
        "roles": {
            "renderer_transnet": {
                "envs": {
                    "sm80": {"python": "env/renderer_transnet/bin/python"},
                }
            }
        },
    }
    registry_path = tmp_path / "env_by_arch.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.delenv("MSPC_TRANSNET_PYTHON", raising=False)
    monkeypatch.setenv("MSPC_ENV_REGISTRY", str(registry_path))
    monkeypatch.setenv("MSPC_SHARED_ROOT", str(shared))
    monkeypatch.setenv("MSPC_GPU_ARCH", "sm80")
    resolved = backend.resolve_transnet_python()
    assert "renderer_transnet" in resolved.parts
    assert "uv_python_bernini_probe" not in resolved.parts


def test_nvidia_lib_dirs_names_the_invoked_interpreter_and_its_realpath(
    tmp_path: Path,
) -> None:
    uv_base = tmp_path / "uv_python_bernini_probe" / "bin"
    uv_base.mkdir(parents=True)
    real_python = uv_base / "python3.11"
    empty_site = tmp_path / "empty-site"
    empty_site.mkdir()
    _write_executable(real_python, f"#!/bin/sh\nprintf '%s\\n' {empty_site}\n")
    venv_python = tmp_path / "renderer_transnet" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    with pytest.raises(RuntimeError, match="no nvidia") as excinfo:
        backend.nvidia_lib_dirs_from_interpreter(venv_python)
    message = str(excinfo.value)
    assert str(venv_python) in message
    assert "realpath=" in message
    assert "uv_python_bernini_probe" in message


def test_worker_predict_without_frames_fails_loud_with_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run/transnet_worker.py"),
            "--mode",
            "predict",
            "--device",
            "cuda",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert "frames-npy" in payload["error"]


def test_worker_import_does_not_load_torch() -> None:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'tools/run')!r})\n"
        "import transnet_worker\n"
        "assert 'torch' not in sys.modules\n"
        "official = transnet_worker.load_backend_without_torch()\n"
        "assert 'torch' not in sys.modules\n"
        "assert official._SubprocessTransNetEngine.__name__ == '_SubprocessTransNetEngine'\n"
        "assert callable(official.preflight_tensorflow_gpu)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


# --------------------------------------------------------------------------
# The shared shell derivation
# --------------------------------------------------------------------------


def test_run_lib_tensorflow_libs_exists_and_is_the_single_owner() -> None:
    assert RUN_LIB.is_file(), f"{RUN_LIB} is the single owner of the derivation"
    body = RUN_LIB.read_text(encoding="utf-8")
    assert "mspc_seed_tensorflow_libs()" in body


def test_no_caller_keeps_a_private_copy_of_the_derivation() -> None:
    """The interpreter-reported site-packages derivation must exist once."""

    needle = "getsitepackages"
    owners = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "run").rglob("*.sh"))
        if needle in path.read_text(encoding="utf-8")
    ]
    assert owners == ["run/lib/tensorflow_libs.sh"], (
        "the derivation was duplicated again; keep it in run/lib/tensorflow_libs.sh"
    )
    body = RUN_LIB.read_text(encoding="utf-8")
    assert "lib/python*/site-packages/nvidia/*/lib" not in body
    assert "nvidia/*/lib" in body


def test_tensorflow_libs_follows_chained_site_packages(tmp_path: Path) -> None:
    env_dir = tmp_path / "renderer_public_metrics"
    python = env_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    chained = tmp_path / "bernini_r1" / "lib" / "python3.11" / "site-packages"
    nvidia_lib = chained / "nvidia" / "cublas" / "lib"
    nvidia_lib.mkdir(parents=True)
    _write_executable(
        python,
        "#!/bin/sh\n"
        f"printf '%s\\n' {str(chained)}\n",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source {RUN_LIB}; '
            f'mspc_seed_tensorflow_libs "{python}" test; '
            'printf "%s\\n" "$LD_LIBRARY_PATH"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert str(nvidia_lib) in completed.stdout
    assert "seeded" in completed.stderr


def test_tensorflow_libs_fails_loud_when_interpreter_reports_no_nvidia_lib(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "empty_env"
    python = env_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    empty_site = tmp_path / "empty-site-packages"
    empty_site.mkdir()
    _write_executable(python, f"#!/bin/sh\nprintf '%s\\n' {str(empty_site)}\n")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source {RUN_LIB}; '
            f'mspc_seed_tensorflow_libs "{python}" test',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "no nvidia" in completed.stderr
    assert str(python) in completed.stderr
