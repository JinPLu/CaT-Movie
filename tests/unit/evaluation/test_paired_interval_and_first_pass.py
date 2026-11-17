"""Paired interval and conjunctive first-pass, without a fixed noise cutoff."""

from __future__ import annotations

import numpy as np

from mspc.evaluation.paired_bootstrap import paired_mean_interval


def test_a_constant_difference_interval_stays_on_that_difference() -> None:
    diffs = np.full(20, 0.1)
    mean, lower, upper = paired_mean_interval(
        diffs, np.random.default_rng(0), resamples=200
    )
    assert np.allclose((mean, lower, upper), 0.1)


def test_a_one_value_difference_is_rejected() -> None:
    try:
        paired_mean_interval(np.array([0.1]), np.random.default_rng(0), resamples=20)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("expected a one-value difference to fail")
