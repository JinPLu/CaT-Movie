from __future__ import annotations

import json
from pathlib import Path

import pytest

from mspc.config.experiment import ExperimentConfigError, load_experiment_config
from mspc.config.shared_root import (
    SHARED_ROOT_ENV,
    SHARED_ROOT_TOKEN,
    SharedRootError,
    SharedRootSettings,
    expand_shared_root_token,
)
from tests.unit.config.test_experiment_config import experiment_mapping


def test_shared_root_settings_require_absolute_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SHARED_ROOT_ENV, raising=False)
    with pytest.raises(SharedRootError, match="must be set"):
        SharedRootSettings.from_env()
    monkeypatch.setenv(SHARED_ROOT_ENV, "relative/root")
    with pytest.raises(SharedRootError, match="absolute"):
        SharedRootSettings.from_env()
    monkeypatch.setenv(SHARED_ROOT_ENV, "/tmp/shared-root")
    assert SharedRootSettings.from_env().root == Path("/tmp/shared-root")


def test_expand_shared_root_token_is_a_no_op_without_token() -> None:
    assert expand_shared_root_token("/already/absolute", "runtime.python") == (
        "/already/absolute"
    )


def test_p4_injected_path_fails_when_shared_root_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SHARED_ROOT_ENV, raising=False)
    mapping = experiment_mapping()
    mapping["runtime"]["python"] = f"{SHARED_ROOT_TOKEN}/env/bin/python"
    path = tmp_path / "injected.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match=SHARED_ROOT_ENV):
        load_experiment_config(path)


def test_p4_injected_path_expands_required_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SHARED_ROOT_ENV, "/tmp/shared-workspace")
    mapping = experiment_mapping()
    mapping["runtime"]["python"] = f"{SHARED_ROOT_TOKEN}/env/bin/python"
    path = tmp_path / "injected.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    resolved = load_experiment_config(path).resolved
    assert resolved["runtime"]["python"] == "/tmp/shared-workspace/env/bin/python"


def test_code_token_is_the_running_checkout_and_data_paths_use_the_data_root(tmp_path):
    from mspc.config.shared_root import RUNNING_CODE_ROOT, expand_shared_root_token
    env = {"MSPC_SHARED_ROOT": str(tmp_path)}
    assert expand_shared_root_token("${MSPC_CODE_ROOT}/third_party/wan22_runtime", "x") == \
        str(RUNNING_CODE_ROOT / "third_party/wan22_runtime")
    other = tmp_path / "frozen/Storyboard"
    assert expand_shared_root_token("${MSPC_CODE_ROOT}/src", "x", code_root=other) == str(other.resolve() / "src")
    assert expand_shared_root_token("${MSPC_SHARED_ROOT}/Storyboard/outputs/run", "x", environ=env) == \
        str(tmp_path / "Storyboard/outputs/run")
    assert expand_shared_root_token("${MSPC_SHARED_ROOT}/Storyboard-runs/a", "x", environ=env) == \
        str(tmp_path / "Storyboard-runs/a")


@pytest.mark.parametrize("raw", ["${MSPC_SHARED_ROOT}/Storyboard/src", "${MSPC_SHARED_ROOT}/Storyboard",
                                 '{"upstream": "${MSPC_SHARED_ROOT}/Storyboard/third_party/wan22_runtime"}'])
def test_code_through_a_data_root_that_holds_other_code_is_refused(tmp_path, raw):
    """数据根下的检出不是正在运行的这份代码时，经数据根指代码会悄悄换代码，直接报错。"""
    from mspc.config.shared_root import SharedRootError, expand_shared_root_token
    with pytest.raises(SharedRootError, match="MSPC_CODE_ROOT"):
        expand_shared_root_token(raw, "x", environ={"MSPC_SHARED_ROOT": str(tmp_path)})


def test_old_per_commit_roots_holding_the_running_code_still_expand(tmp_path):
    """旧的按提交根（代码与软链在同一根下）照常：数据根下的检出就是正在运行的代码。"""
    from mspc.config.shared_root import expand_shared_root_token
    root = tmp_path / "sha-root"
    (root / "Storyboard").mkdir(parents=True)
    assert expand_shared_root_token("${MSPC_SHARED_ROOT}/Storyboard/src", "x",
                                    environ={"MSPC_SHARED_ROOT": str(root)},
                                    code_root=root / "Storyboard") == str(root / "Storyboard/src")
