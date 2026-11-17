"""SwanLab 展示的接线：配置四个键缺一不可；密钥只从文件进环境变量；离线不读密钥；记的数就是本地账那一行。"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mspc.training.loss_tracking import LossTracking, open_loss_tracking
from tests.support.paths import REPO_ROOT

OFFLINE = {"project": "Storyboard", "workspace": "JinPLu", "experiment_name": "smoke", "mode": "offline", "api_key_file": None}


@pytest.fixture
def fake_swanlab(monkeypatch):
    calls = {"init": [], "log": [], "finish": 0}
    module = SimpleNamespace(
        init=lambda **kwargs: calls["init"].append(kwargs) or object(),
        log=lambda metrics, step: calls["log"].append((step, metrics)),
        finish=lambda: calls.__setitem__("finish", calls["finish"] + 1))
    monkeypatch.setitem(sys.modules, "swanlab", module)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    return calls


def test_every_key_is_required_and_modes_are_the_official_two():
    assert LossTracking.from_mapping(OFFLINE).mode == "offline"
    for broken in ({k: v for k, v in OFFLINE.items() if k != "project"}, dict(OFFLINE, mode="cloud"),
                   dict(OFFLINE, experiment_name=" "), dict(OFFLINE, api_key_file="/somewhere/key.txt"),
                   dict(OFFLINE, mode="online"), dict(OFFLINE, extra=1)):
        with pytest.raises(ValueError):
            LossTracking.from_mapping(broken)


def test_offline_run_reads_no_key_and_logs_under_the_run_log_dir(fake_swanlab, tmp_path):
    run = open_loss_tracking(OFFLINE, hyperparameters={"training": {"seed": 42}}, log_dir=tmp_path)
    (call,) = fake_swanlab["init"]
    assert call["experiment_name"].startswith("smoke-20") and call["experiment_name"].endswith("Z")
    assert {k: v for k, v in call.items() if k != "experiment_name"} == {"project": "Storyboard", "workspace": "JinPLu",
        "config": {"training": {"seed": 42}}, "logdir": str(tmp_path / "swanlog"), "mode": "offline"}
    assert "SWANLAB_API_KEY" not in os.environ
    run.log(3, {"train/loss": 0.5})
    run.finish()
    assert fake_swanlab["log"] == [(3, {"train/loss": 0.5})] and fake_swanlab["finish"] == 1
    assert open_loss_tracking(None, hyperparameters={}, log_dir=tmp_path) is None


def test_online_run_needs_the_key_file_and_passes_it_only_through_the_environment(fake_swanlab, tmp_path):
    online = dict(OFFLINE, mode="online", api_key_file=str(tmp_path / "swanlab_api_key.txt"))
    with pytest.raises(FileNotFoundError, match="swanlab_api_key.txt"):
        open_loss_tracking(online, hyperparameters={}, log_dir=tmp_path)
    assert fake_swanlab["init"] == []
    (tmp_path / "swanlab_api_key.txt").write_text("placeholder-not-a-real-key\n")
    open_loss_tracking(online, hyperparameters={}, log_dir=tmp_path)
    assert os.environ["SWANLAB_API_KEY"] == "placeholder-not-a-real-key"
    assert "placeholder" not in json.dumps(fake_swanlab["init"])


def test_update_metrics_are_the_raw_numbers_of_the_progress_line():
    from mspc.training.loss_tracking import reported_grad_norm, update_metrics
    timings = {"step_total": 12.5, "data_wait": 0.2, "forward": 4.0, "backward": 7.0, "rank": 0}
    assert update_metrics(losses={"loss": 0.25}, learning_rate=1e-5, next_learning_rate=2e-5,
                          timings=timings, samples_seen=16, elapsed_seconds=40., grad_norm=None) == {
        "loss": 0.25, "lr_used": 1e-5, "lr_next": 2e-5, "samples_seen": 16, "step_seconds": 12.5, "data_seconds": 0.2,
        "forward_seconds": 4.0, "backward_seconds": 7.0, "update_wall_seconds": 12.7,
        "training_step_seconds": 12.5, "elapsed_seconds": 40.}
    assert update_metrics(losses={"loss": 0.25}, learning_rate=1e-5, next_learning_rate=2e-5,
                          timings=timings, samples_seen=16, elapsed_seconds=40., grad_norm=1.5)["grad_norm"] == 1.5
    assert reported_grad_norm(SimpleNamespace()) is None
    assert reported_grad_norm(SimpleNamespace(get_global_grad_norm=lambda: 2)) == 2.0


def test_native_deepspeed_update_timers_are_not_the_outer_optimizer_noop():
    from mspc.training.loss_tracking import reported_deepspeed_timings
    engine = SimpleNamespace(get_wall_clock_timers=lambda: {
        "fwd": 1000., "bwd": 4000., "bwd_inner": 3500., "bwd_allreduce": 500., "step": 200.})
    result = reported_deepspeed_timings(engine)
    assert result["deepspeed_update_seconds"] == .2
    assert result["deepspeed_backward_seconds"] == 4.


def test_accumulated_stage_metrics_reach_the_cloud_without_gpu_dispatch_times():
    from mspc.core.loss_curves import train_curves
    from mspc.training.loop import accumulate_update_timings
    from mspc.training.loss_tracking import update_metrics
    update = {}
    for seconds, frames in ((.125, 1), (.25, 3)):
        accumulate_update_timings(update, {"step_total": 5., "data_wait": .1, "forward": 2., "backward": 3.,
            "reference_vae_seconds": seconds, "reference_encoded_frames": frames,
            "reference_read_seconds": seconds * 2, "cuda_synchronized": True})
    metrics = update_metrics(losses={"loss": .5, "uniform_loss": .75, "target_weighted_mse": 1., "unweighted_mse": 1.5},
                             learning_rate=1e-5, next_learning_rate=2e-5,
                             timings=update, samples_seen=2, elapsed_seconds=20., grad_norm=1.)
    rows = [{"step": 1, "loss": .5, "timestep": 900., "span": [0, 1]}] * 2
    point = train_curves(rows, [{"step": 1, **metrics}], samples_per_update=2, window=1)[1]
    assert point["timing/reference_vae_seconds"] == .375
    assert point["timing/reference_read_seconds"] == .75
    assert point["workload/reference_encoded_frames"] == 4
    assert point["timing/update_wall_seconds"] == 10.2
    assert point["train/uniform_loss"] == .75 and point["train/target_weighted_mse"] == 1.
    assert point["train/unweighted_mse"] == 1.5 and point["timing/elapsed_seconds"] == 20.
    update["cuda_synchronized"] = False
    unsynchronized = update_metrics(losses={"loss": .5}, learning_rate=1e-5, next_learning_rate=2e-5,
                                    timings=update, samples_seen=2, elapsed_seconds=20., grad_norm=1.)
    assert "reference_vae_seconds" not in unsynchronized


def test_first_train_config_carries_the_tracking_block():
    from mspc.config import load_experiment_config
    from mspc.config.experiment import ExperimentConfigError, parse_experiment_mapping
    from mspc.training.training_args import build_common_training_args
    path = REPO_ROOT / "tests/fixtures/block_renderer_config.json"
    args = build_common_training_args(load_experiment_config(path))
    assert args.loss_tracking == {"project": "Storyboard", "workspace": "JinPLu",
                                  "experiment_name": "block_renderer_contract", "mode": "offline", "api_key_file": None}
    assert args.experiment_config["training"]["max_train_steps"] == 2000
    raw = json.loads(path.read_text())
    raw["training"]["loss_tracking"]["mode"] = "cloud"
    with pytest.raises(ExperimentConfigError, match="loss_tracking"):
        parse_experiment_mapping(raw, source_path=path)
