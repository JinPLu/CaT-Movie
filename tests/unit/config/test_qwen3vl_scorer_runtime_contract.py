from __future__ import annotations

from pathlib import Path

import yaml

from mspc.config.toml_compat import tomllib


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = (
    "${MSPC_SHARED_ROOT}/Storyboard/"
    ".runtime/venvs/qwen3vl_scorer_cu128"
)


def test_qwen3vl_scorer_runtime_is_locked_and_project_local() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["optional-dependencies"]["gpu-vllm-scorer"])
    sources = project["tool"]["uv"]["sources"]

    assert {
        "torch==2.10.0",
        "torchaudio==2.10.0",
        "torchvision==0.25.0",
        "transformers==4.57.6",
        "vllm==0.18.0; sys_platform == 'linux'",
    } <= dependencies
    for package in ("torch", "torchaudio", "torchvision"):
        source = next(
            row for row in sources[package] if row.get("extra") == "gpu-vllm-scorer"
        )
        assert source["index"] == "pytorch-cu128"


def test_qwen3vl_scorer_service_uses_the_project_runtime() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/serving/vllm_qwen3vl8b_scorer_serve.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["vllm_executable"] == f"{RUNTIME}/bin/vllm"
    assert "--enforce-eager" in config["extra_cli_args"]
