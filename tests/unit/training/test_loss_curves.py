"""loss 曲线汇总：全 rank 均值、滑动窗口、时间步分档、只出记全了的更新与量完了的验证。纯函数，合成行。"""
from __future__ import annotations

import pytest

from mspc.core.loss_curves import TIMESTEP_BUCKETS, timestep_bucket, train_curves, validation_curves


def sample(step, loss, timestep=900.0, span=(0, 2)):
    return {"step": step, "loss": loss, "timestep": timestep, "span": list(span)}


def metric(step):
    return {"step": step, "loss": 4.0 * step - 2.0, "lr_used": 1e-5 * (step - 1), "lr_next": 1e-5 * step, "grad_norm": 2.0, "step_seconds": 5.0, "samples_seen": 2 * step}


def test_timestep_buckets_are_the_quartiles_of_the_shifted_schedule():
    # shift=5：下标 u 映到 1000·5u／(1＋4u)；四分位处是 625、833.3、937.5。
    assert [timestep_bucket(1000 * 5 * u / (1 + 4 * u)) for u in (0.1, 0.3, 0.6, 0.9)] == list(TIMESTEP_BUCKETS)
    assert timestep_bucket(625.0) == "t625_833" and timestep_bucket(1000.0) == "t938_1000"


def test_update_loss_is_the_mean_over_every_rank_and_the_window_averages_recent_updates():
    samples = [sample(1, 1.0), sample(1, 3.0), sample(2, 5.0, 100.0, (0, 1)), sample(2, 7.0), sample(3, 9.0), sample(3, 11.0)]
    curves = train_curves(samples, [metric(s) for s in (1, 2, 3)], samples_per_update=2, window=2)
    assert [curves[s]["train/loss"] for s in (1, 2, 3)] == [2.0, 6.0, 10.0]  # 直接消费训练时归约的全rank均值
    assert [curves[s]["train/loss_avg2"] for s in (1, 2, 3)] == [2.0, 4.0, 8.0]
    assert curves[2]["train_by_timestep/t000_625"] == 5.0 and curves[2]["train_by_timestep/t833_938"] == pytest.approx(11 / 3)
    assert curves[2]["train_by_shots/shots_1"] == 5.0 and curves[3]["train_by_shots/shots_2"] == pytest.approx(27 / 3)
    assert curves[3]["train/lr_used"] == pytest.approx(2e-5) and curves[3]["train/lr_next"] == pytest.approx(3e-5) and curves[3]["train/grad_norm"] == 2.0


def test_only_complete_updates_come_out_and_nothing_after_a_gap():
    samples = [sample(1, 1.0), sample(1, 1.0), sample(2, 1.0), sample(3, 1.0), sample(3, 1.0)]
    assert list(train_curves(samples, [metric(s) for s in (1, 2, 3)], samples_per_update=2, window=5)) == [1]
    assert list(train_curves(samples[:2], [], samples_per_update=2, window=5)) == []  # 主进程那一行还没到
    with pytest.raises(ValueError, match="samples_per_update"):
        train_curves(samples + [sample(1, 1.0)], [metric(1)], samples_per_update=2, window=5)


def test_a_validation_pass_comes_out_once_every_forward_is_in():
    rows = [{"step": 0, "loss": 4.0, "timestep": 990.0, "shots": 1}, {"step": 0, "loss": 2.0, "timestep": 300.0, "shots": 3},
            {"step": 250, "loss": 1.0, "timestep": 990.0, "shots": 1}]
    curves = validation_curves(rows, forwards_per_pass=2)
    assert list(curves) == [0]  # 第 250 步那一遍只到了一半
    assert curves[0] == {"val/loss": 3.0, "val_by_timestep/t938_1000": 4.0, "val_by_shots/shots_1": 4.0,
                         "val_by_timestep/t000_625": 2.0, "val_by_shots/shots_3": 2.0}


def test_validation_timing_keeps_slowest_rank_separate_from_mean():
    rows = [{"step": 0, "loss": 1., "timestep": 900., "shots": 1,
             "rank": rank, "data_seconds": data, "forward_seconds": forward}
            for rank, data, forward in [(0, 1., 2.), (0, 0., 2.), (1, 3., 2.), (1, 0., 2.)]]
    point = validation_curves(rows, forwards_per_pass=4)[0]
    assert point["val_timing/data_seconds_rank_mean"] == 2.
    assert point["val_timing/data_seconds_rank_max"] == 3.
    assert point["val_timing/forward_seconds_rank_max"] == 4.


def test_full_and_intermediate_panels_have_their_own_completion_counts_and_curves():
    rows = [{"step": step, "loss": float(index + 1), "timestep": 900., "shots": 1,
             "validation_panel": panel, "validation_forwards": expected}
            for step, panel, expected, count in [(0, "full", 4, 4), (500, "intermediate", 2, 2),
                                                 (1000, "intermediate", 2, 1), (3000, "full", 4, 3)]
            for index in range(count)]
    curves = validation_curves(rows, forwards_per_pass=4)
    assert list(curves) == [0, 500]
    assert curves[0]["val/loss"] == 2.5
    assert curves[500]["val_intermediate/loss"] == 1.5
    assert "val/loss" not in curves[500] and "val_intermediate/loss" not in curves[0]
