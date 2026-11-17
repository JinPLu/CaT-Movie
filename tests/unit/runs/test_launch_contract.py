from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mspc.config.experiment import parse_experiment_mapping
from mspc.runs.launch import LaunchMode, TrainLaunchOverrides, resolve_launch
from tests.unit.config.test_experiment_config import DIFFSYNTH_ROOT, experiment_mapping


def _experiment_mapping():
    mapping = experiment_mapping()
    for key in ("gpus_per_node", "nnodes", "node_rank", "master_addr", "master_port"):
        mapping["launch"].pop(key, None)
    return mapping


@pytest.fixture
def sealed_cache_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """这几条测的是命令与环境的拼装，不是数据集准入；准入另有测试。"""

    monkeypatch.setattr("mspc.runs.launch.require_train_launch_dataset", lambda config: None)


def _cluster_overrides() -> TrainLaunchOverrides:
    return TrainLaunchOverrides(
        gpus_per_node=4,
        nnodes=1,
        node_rank=0,
        master_addr="localhost",
        master_port="6200",
    )


def test_resolve_train_launch_is_complete_and_side_effect_free(
    tmp_path: Path, sealed_cache_admitted: None
) -> None:
    config = parse_experiment_mapping(_experiment_mapping(), source_path=tmp_path / "experiment.yaml")
    launch = resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=_cluster_overrides(),
    )
    assert launch.executable == Path(sys.executable)
    assert launch.argv[0] == str(launch.executable)
    assert launch.argv[-6:] == (
        "--config",
        str(tmp_path / "experiment.yaml"),
        "--gpus-per-node",
        "4",
        "--nnodes",
        "1",
    )
    assert launch.environment["PYTHONPATH"] == f"{tmp_path / 'src'}:{DIFFSYNTH_ROOT}"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert not (tmp_path / "outputs").exists()


def test_resolve_infer_launch_runs_under_the_launching_interpreter(tmp_path: Path) -> None:
    """推理跑的是与启动方相同的模型栈，所以用同一个解释器，不再读 ``runtime.python``。

    读 config 那一版的后果实测过：2026-08-26 的 A100 16 步门训练完整跑完 16 步，
    紧接着的 smoke 推理在载 DiT 时死于 `driver too old`——因为它绕开了刚刚把训练
    放对位置的架构路由，回到了 config 里写死的 cu130 环境。
    """

    config = parse_experiment_mapping(_experiment_mapping(), source_path=tmp_path / "experiment.json")
    launch = resolve_launch(config, mode=LaunchMode.INFER)
    assert launch.executable == Path(sys.executable)
    assert launch.entrypoint == tmp_path / "infer.py"
    assert launch.environment["PYTHONPATH"] == f"{tmp_path / 'src'}:{DIFFSYNTH_ROOT}"
    assert launch.argv == (
        sys.executable,
        str(tmp_path / "infer.py"),
        "--config",
        str(tmp_path / "experiment.json"),
    )
    assert launch.output_path == tmp_path / "outputs/inference"
    assert launch.checkpoint == tmp_path / "checkpoints/step-500.safetensors"


def test_cuda_environment_is_absent_only_when_schema_value_is_null(
    tmp_path: Path, sealed_cache_admitted: None
) -> None:
    mapping = _experiment_mapping()
    mapping["launch"]["cuda_visible_devices"] = "0,1,2,3"
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")
    assert resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=_cluster_overrides(),
    ).environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_train_checkpoint_override_is_forwarded_to_runner(
    tmp_path: Path, sealed_cache_admitted: None
) -> None:
    config = parse_experiment_mapping(
        _experiment_mapping(), source_path=tmp_path / "experiment.yaml"
    )
    checkpoint = tmp_path / "step-500.safetensors"
    launch = resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=_cluster_overrides(),
        checkpoint_override=checkpoint,
    )
    assert launch.argv[-2:] == ("--checkpoint", str(checkpoint))
    assert launch.checkpoint == checkpoint


def test_launch_requires_enum_not_string(tmp_path: Path) -> None:
    config = parse_experiment_mapping(_experiment_mapping(), source_path=tmp_path / "experiment.yaml")
    with pytest.raises(TypeError, match="LaunchMode"):
        resolve_launch(config, mode="train")  # type: ignore[arg-type]


def test_train_launch_requires_explicit_cluster_operands(
    tmp_path: Path, sealed_cache_admitted: None
) -> None:
    config = parse_experiment_mapping(
        _experiment_mapping(),
        source_path=tmp_path / "experiment.yaml",
    )
    with pytest.raises(ValueError, match="train_overrides are required"):
        resolve_launch(config, mode=LaunchMode.TRAIN)


def test_train_launch_accepts_explicit_wan_cluster_operands(
    tmp_path: Path, sealed_cache_admitted: None
) -> None:
    config = parse_experiment_mapping(_experiment_mapping(), source_path=tmp_path / "experiment.json")
    launch = resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=TrainLaunchOverrides(
            gpus_per_node=8,
            nnodes=2,
            node_rank=1,
            master_addr="10.0.0.8",
            master_port="6200",
        ),
    )
    assert launch.argv[launch.argv.index("--nproc_per_node") + 1] == "8"
    assert launch.argv[launch.argv.index("--nnodes") + 1] == "2"
    assert launch.argv[launch.argv.index("--node_rank") + 1] == "1"
    assert launch.argv[launch.argv.index("--master_addr") + 1] == "10.0.0.8"
    assert launch.argv[launch.argv.index("--gpus-per-node") + 1] == "8"
    assert launch.argv[launch.argv.index("--nnodes") + 1] == "2"


def test_train_launch_rejects_invalid_cluster_operands() -> None:
    with pytest.raises(ValueError, match="node_rank"):
        TrainLaunchOverrides(
            gpus_per_node=8,
            nnodes=1,
            node_rank=1,
            master_addr="localhost",
            master_port="6200",
        )
