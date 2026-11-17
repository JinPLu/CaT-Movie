#!/usr/bin/env python3
"""pyiqa-only worker for the renderer qualification quality columns.

Started with an interpreter that carries pyiqa 0.1.16 and its own
``transformers>=5.0`` stack, decoupled from the main scoring env's
``transformers>=4.45,<5``.  Unlike ``transnet_worker.py`` this process is
*expected* to import torch -- the isolation here is a version conflict, not a
torch/TensorFlow library-exclusivity conflict, so there is no "parent/child
must never import X" contract to enforce.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _stub_package(name: str, path: Path) -> None:
    import types

    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


def load_backend() -> Any:
    """Load the official backend by file, mirroring ``transnet_worker.py``.

    ``mspc/__init__.py`` imports config, which reaches modules this
    interpreter's own, deliberately narrow dependency set does not carry
    (deepspeed, flash-attn, bernini, decord, ...).  Loading the two backend
    modules directly by file path, with ``mspc``/``mspc.evaluation`` stubbed
    as empty packages, keeps this worker's only real dependency on the
    project to those two files -- the same trick ``transnet_worker.py`` uses,
    for the same reason, even though this worker has no torch to avoid.
    """

    import importlib.util

    src = REPO_ROOT / "src"
    _stub_package("mspc", src / "mspc")
    _stub_package("mspc.evaluation", src / "mspc" / "evaluation")

    def _load(name: str, file_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load(
        "mspc.evaluation.renderer_qualification_public_metrics",
        src / "mspc" / "evaluation" / "renderer_qualification_public_metrics.py",
    )
    return _load(
        "mspc.evaluation.renderer_qualification_official_backend",
        src / "mspc" / "evaluation" / "renderer_qualification_official_backend.py",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pyiqa-only renderer quality worker")
    parser.add_argument("--mode", required=True, choices=("preflight", "predict"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--frames-npy", type=Path)
    parser.add_argument("--shot-counts")
    args = parser.parse_args(argv)
    try:
        official = load_backend()
        if args.mode == "preflight":
            official.preflight_quality_gpu(args.device)
            _emit({
                "status": "ok",
                "device": args.device,
                "executable": sys.executable,
                "prefix": sys.prefix,
            })
            return 0
        if args.resource_root is None or args.frames_npy is None or not args.shot_counts:
            raise ValueError(
                "predict requires --resource-root, --frames-npy and --shot-counts"
            )
        import numpy as np

        frames = np.load(args.frames_npy, allow_pickle=False)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames npy must be NHWC RGB, got shape {frames.shape}")
        counts = [int(part) for part in args.shot_counts.split(",") if part]
        if not counts or sum(counts) != len(frames):
            raise ValueError(
                f"--shot-counts sums to {sum(counts)}, decoded {len(frames)} frames"
            )
        shots: list[list[Any]] = []
        start = 0
        for count in counts:
            shots.append(list(frames[start : start + count]))
            start += count
        # pyiqa prints its own "Loading pretrained model ..." lines on stdout,
        # and stdout is the protocol this worker speaks to its parent.  Send
        # that chatter to stderr for the duration rather than teaching the
        # parent to skip over noise: the parent then parses exactly what this
        # process meant to say, and the diagnostics are still there to read.
        with contextlib.redirect_stdout(sys.stderr):
            engine = official._PyIQAEngine(args.resource_root, args.device)
            scores = engine.score(shots)
        _emit({
            "status": "ok",
            "scores": {name: float(value) for name, value in scores.items()},
            "executable": sys.executable,
            "prefix": sys.prefix,
        })
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        _emit({
            "status": "error",
            "error": message,
            "executable": sys.executable,
            "prefix": sys.prefix,
        })
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
