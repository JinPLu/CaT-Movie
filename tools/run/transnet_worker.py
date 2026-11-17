#!/usr/bin/env python3
"""TensorFlow-only TransNetV2 worker. The parent scoring process must never import tensorflow.

Started with an interpreter that does not install torch. Finding torch already
imported, or pulling it in through the backend import, is a contract failure
rather than a CPU fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _refuse_torch() -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "transnet worker found torch already imported; the TF-only isolation "
            "contract is broken"
        )


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


def load_backend_without_torch() -> Any:
    """Load the official backend by file so ``import mspc`` cannot pull torch.

    ``mspc/__init__.py`` imports config, which reaches ``models.wan22_story_token``
    and ``import torch``.  The worker must not take that path.
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
    official = _load(
        "mspc.evaluation.renderer_qualification_official_backend",
        src / "mspc" / "evaluation" / "renderer_qualification_official_backend.py",
    )
    if "torch" in sys.modules:
        raise RuntimeError(
            "importing the TransNet backend pulled torch into the TF-only worker"
        )
    return official


def _load_backend() -> tuple[Any, Any]:
    official = load_backend_without_torch()
    return official._TransNetEngine, official.preflight_tensorflow_gpu


def main(argv: list[str] | None = None) -> int:
    _refuse_torch()
    parser = argparse.ArgumentParser(description="TF-only TransNetV2 worker")
    parser.add_argument("--mode", required=True, choices=("preflight", "predict", "serve"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--frames-npy", type=Path)
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            engine_cls, preflight_tensorflow_gpu = _load_backend()
        if args.mode == 'serve':
            if args.resource_root is None:
                raise ValueError('serve requires --resource-root')
            import numpy as np
            with contextlib.redirect_stdout(sys.stderr):
                engine = engine_cls(args.resource_root, args.device)
            for line in sys.stdin:
                request = json.loads(line)
                frames = np.load(request['frames_npy'], allow_pickle=False)
                if frames.ndim != 4 or frames.shape[-1] != 3:
                    raise ValueError(f'frames npy must be NHWC RGB, got shape {frames.shape}')
                with contextlib.redirect_stdout(sys.stderr):
                    probabilities = engine.predict(frames)
                del frames
                _emit({'status': 'ok', 'result': [float(value) for value in probabilities]})
            return 0
        if args.mode == "preflight":
            preflight_tensorflow_gpu(args.device)
            _emit({
                "status": "ok",
                "device": args.device,
                "executable": sys.executable,
                "prefix": sys.prefix,
            })
            return 0
        if args.resource_root is None or args.frames_npy is None:
            raise ValueError("predict requires --resource-root and --frames-npy")
        import numpy as np

        frames = np.load(args.frames_npy, allow_pickle=False)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames npy must be NHWC RGB, got shape {frames.shape}")
        probabilities = engine_cls(args.resource_root, args.device).predict(frames)
        _emit({
            "status": "ok",
            "probabilities": [float(value) for value in probabilities],
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
