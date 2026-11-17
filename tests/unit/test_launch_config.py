from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mspc.cli_domains import inference as inference_cli
from mspc.cli_domains import training as training_cli
from mspc.cli_domains.inference import register_inference_commands
from mspc.cli_domains.training import register_training_commands
from mspc.config import load_experiment_config, parse_experiment_mapping
from mspc.runs import LaunchMode, TrainLaunchOverrides, resolve_launch
from tests.unit.config.test_experiment_config import experiment_mapping


def test_train_and_infer_cli_registration_is_lazy() -> None:
    before = set(sys.modules)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_training_commands(commands)
    register_inference_commands(commands)
    train = parser.parse_args(["train", "--config", "/tmp/a.yaml", "--execution-mode", "plan"])
    infer = parser.parse_args(["infer", "--config", "/tmp/a.yaml", "--execution-mode", "run"])
    assert (train.command, train.execution_mode) == ("train", "plan")
    assert (infer.command, infer.execution_mode) == ("infer", "run")
    assert "torch" not in set(sys.modules) - before
    assert "accelerate" not in set(sys.modules) - before


def test_train_cli_accepts_complete_cluster_overrides() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_training_commands(commands)
    args = parser.parse_args([
        "train", "--config", "/tmp/a.json", "--execution-mode", "plan",
        "--gpus-per-node", "8", "--nnodes", "2", "--node-rank", "1",
        "--master-addr", "10.0.0.8", "--master-port", "6200",
    ])
    overrides = training_cli._cluster_overrides(args)
    assert overrides is not None
    assert overrides.gpus_per_node == 8
    assert overrides.nnodes == 2
    assert overrides.node_rank == 1


@pytest.mark.parametrize('shell_entry', [False, True])
def test_cache_stage_crosses_real_shell_root_cli_and_launch_plan(tmp_path, shell_entry) -> None:
    """真实子进程只做plan，核shell→根CLI→torchrun/runner参数，不申请设备。"""
    from tests.support.paths import REPO_ROOT

    raw = json.loads((REPO_ROOT/'tests/fixtures/block_renderer_config.json').read_text())
    raw['runtime']['python'] = sys.executable
    raw['training']['state_checkpoint'] = {'keep': 1}
    for name in ('members_path', 'sampling_manifest_path', 'target_index_path'):
        artifact = tmp_path/name
        artifact.write_text('')
        raw['dataset'][name] = str(artifact)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(raw))
    stage = tmp_path/'cache-stage.json'
    stage.write_text(json.dumps({'artifact_type': 'mspc_training_cache_stage', 'schema_version': 1,
                                 'target_step': 3, 'sample_indices': [0]}))
    topology = {'NPROC_PER_NODE': '1', 'WORLD_SIZE': '1', 'RANK': '0',
                'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '6200'}
    env = {**os.environ, **topology, 'MSPC_SHARED_ROOT': str(tmp_path),
           'PYTHONPATH': str(REPO_ROOT/'src'), 'MSPC_BOOTSTRAP_PYTHON': sys.executable}
    state = tmp_path/'state'
    state.mkdir()
    migration = tmp_path/'migration.json'
    migration.write_text('{}')
    common = [str(config), '--resume-state', str(state), '--resume-migration', str(migration),
              '--cache-stage', str(stage), '--execution-mode', 'plan']
    command = (['bash', str(REPO_ROOT/'run/train.sh'), *common] if shell_entry else
        [sys.executable, '-m', 'mspc', 'train', '--config', *common,
         '--gpus-per-node', '1', '--nnodes', '1', '--node-rank', '0',
         '--master-addr', '127.0.0.1', '--master-port', '6200'])
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # shell输出启动说明，根CLI的完整JSON对象仍是最后一条输出。
    start = completed.stdout.index('{')
    plan = json.loads(completed.stdout[start:])
    assert plan['argv'][-6:] == ['--resume-state', str(state), '--resume-migration', str(migration),
                                '--cache-stage', str(stage)]
    assert plan['execution_mode'] == 'plan'


def test_train_cli_rejects_partial_cluster_overrides() -> None:
    args = argparse.Namespace(gpus_per_node=8)
    try:
        training_cli._cluster_overrides(args)
    except ValueError as exc:
        assert "must be supplied together" in str(exc)
    else:
        raise AssertionError("partial cluster overrides unexpectedly accepted")


def test_train_cli_requires_cluster_operands() -> None:
    with pytest.raises(ValueError, match="cluster launch operands are required"):
        training_cli._cluster_overrides(argparse.Namespace())


def test_record_post_training_smoke_failure_does_not_mark_launch_failed() -> None:
    from mspc.runs.launch import record_post_training_smoke_failure

    payload: dict[str, object] = {"returncode": 0}
    recorded = record_post_training_smoke_failure(
        payload, RuntimeError("visual smoke rejected")
    )
    assert payload["returncode"] == 0
    assert recorded["status"] == "failed"
    assert recorded["interrupts_launch"] is False
    assert recorded["error_type"] == "RuntimeError"
    assert "visual smoke rejected" in recorded["error"]
    assert payload["smoke_gate"] == recorded


def test_train_run_records_smoke_failure_without_failing_the_chain(
    monkeypatch, tmp_path
) -> None:
    import mspc.training.wan22_smoke_gate as smoke_mod

    launch = SimpleNamespace(
        argv=("/runtime/accelerate", "launch"),
        environment={"PYTHONUNBUFFERED": "1"},
        to_dict=lambda: {"mode": "train"},
    )
    output_root = tmp_path / "runs"
    output_root.mkdir()
    config = SimpleNamespace(
        resolved={
            "output": {"root": str(output_root)},
            "inference": {
                "enabled": True,
                "weight_source": "training_output_checkpoint",
            },
        }
    )
    monkeypatch.setattr(training_cli, "load_experiment_config", lambda path: config)
    monkeypatch.setattr(training_cli, "resolve_launch", lambda *args, **kwargs: launch)
    monkeypatch.setattr(
        training_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        smoke_mod,
        "resolve_new_run_root",
        lambda output_root, existing: output_root / "new-run",
    )
    monkeypatch.setattr(
        smoke_mod,
        "run_training_output_smoke_gate",
        lambda config, run_root: (_ for _ in ()).throw(
            RuntimeError("visual smoke rejected")
        ),
    )
    outcome = training_cli.run_training_command(
        argparse.Namespace(
            command="train",
            config=tmp_path / "config.json",
            execution_mode="run",
            gpus_per_node=1,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port="6200",
            checkpoint=None,
            resume_state=None,
            cache_stage=None,
        )
    )
    assert outcome.exit_code == 0
    assert outcome.payload["returncode"] == 0
    assert outcome.payload["smoke_gate"]["status"] == "failed"
    assert outcome.payload["smoke_gate"]["interrupts_launch"] is False
    assert "visual smoke rejected" in outcome.payload["smoke_gate"]["error"]


def test_train_run_skips_smoke_gate_when_inference_is_null(monkeypatch, tmp_path) -> None:
    launch = SimpleNamespace(
        argv=("/runtime/accelerate", "launch"),
        environment={"PYTHONUNBUFFERED": "1"},
        to_dict=lambda: {"mode": "train"},
    )
    config = SimpleNamespace(
        resolved={
            "output": {"root": str(tmp_path / "runs")},
            "inference": None,
        }
    )
    monkeypatch.setattr(training_cli, "load_experiment_config", lambda path: config)
    monkeypatch.setattr(training_cli, "resolve_launch", lambda *args, **kwargs: launch)
    monkeypatch.setattr(
        training_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    outcome = training_cli.run_training_command(
        argparse.Namespace(
            command="train",
            config=tmp_path / "config.json",
            execution_mode="run",
            gpus_per_node=1,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port="6200",
            checkpoint=None,
            resume_state=None,
            cache_stage=None,
        )
    )
    assert outcome.exit_code == 0
    assert "smoke_gate" not in outcome.payload


def test_train_run_suppresses_only_the_root_json_payload(monkeypatch, tmp_path) -> None:
    launch = SimpleNamespace(
        argv=("/runtime/accelerate", "launch"),
        environment={"PYTHONUNBUFFERED": "1"},
        to_dict=lambda: {"mode": "train"},
    )
    config = SimpleNamespace(
        resolved={
            "output": {"root": str(tmp_path / "runs")},
            "inference": {"enabled": False, "weight_source": None},
        }
    )
    monkeypatch.setattr(training_cli, "load_experiment_config", lambda path: config)
    monkeypatch.setattr(training_cli, "resolve_launch", lambda *args, **kwargs: launch)
    monkeypatch.setattr(
        training_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    outcome = training_cli.run_training_command(
        argparse.Namespace(
            command="train",
            config=tmp_path / "config.py",
            execution_mode="run",
            gpus_per_node=8,
            nnodes=2,
            node_rank=0,
            master_addr="10.0.0.8",
            master_port="6200",
            checkpoint=None,
            resume_state=None,
            cache_stage=None,
        )
    )
    assert outcome.exit_code == 0
    assert outcome.emit_payload is False
    assert outcome.payload["returncode"] == 0


def test_launch_plans_use_canonical_config_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mspc.runs.launch.require_train_launch_dataset", lambda config: None)
    config = parse_experiment_mapping(
        experiment_mapping(), source_path=tmp_path / "experiment.yaml"
    )
    train = resolve_launch(
        config,
        mode=LaunchMode.TRAIN,
        train_overrides=TrainLaunchOverrides(
            gpus_per_node=8,
            nnodes=2,
            node_rank=0,
            master_addr="10.0.0.8",
            master_port="6200",
        ),
    )
    infer = resolve_launch(config, mode=LaunchMode.INFER)
    assert "--config" in train.argv and "--config_path" not in train.argv
    assert "--config" in infer.argv and "--config_path" not in infer.argv


def test_disabled_inference_cannot_be_planned(tmp_path) -> None:
    mapping = experiment_mapping()
    mapping["inference"].update(
        enabled=False,
        test_csv_path=None,
        output_dir=None,
        dit_checkpoint=None,
    )
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "train-only.yaml")
    try:
        resolve_launch(config, mode=LaunchMode.INFER)
    except ValueError as exc:
        assert "inference.enabled=false" in str(exc)
    else:
        raise AssertionError("disabled inference unexpectedly resolved")


def test_infer_command_plans_the_launch_without_starting_it(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "experiment.yaml"
    config = parse_experiment_mapping(experiment_mapping(), source_path=config_path)
    monkeypatch.setattr(inference_cli, "load_experiment_config", lambda path: config)

    outcome = inference_cli.run_inference_command(
        argparse.Namespace(command="infer", config=config_path, execution_mode="plan")
    )

    assert outcome.exit_code == 0
    assert outcome.payload["inference_kind"] == "experiment_launch"
    assert outcome.payload["execution_mode"] == "plan"
    assert outcome.payload["argv"][-2:] == ["--config", str(config_path)]
