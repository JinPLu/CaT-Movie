"""``mspc.evaluation.paired_bootstrap``：重采样次数是显式参数，两个消费端各自传入。"""

from __future__ import annotations

import numpy as np
import pytest

from mspc.evaluation.paired_bootstrap import (
    BOOTSTRAP_RESAMPLES,
    bootstrap_means,
    one_sided_p,
    required_n,
)


def test_bootstrap_means_draws_exactly_the_requested_number_of_resamples() -> None:
    diffs = np.asarray([0.1, -0.2, 0.3, 0.05], dtype=float)
    means = bootstrap_means(diffs, np.random.default_rng(0), resamples=37)
    assert means.shape == (37,)
    assert bootstrap_means(diffs, np.random.default_rng(0), resamples=BOOTSTRAP_RESAMPLES).shape == (
        BOOTSTRAP_RESAMPLES,
    )


@pytest.mark.parametrize("resamples", [0, -1, 10.0, True])
def test_resamples_must_be_a_positive_int(resamples) -> None:
    diffs = np.asarray([0.1, -0.2], dtype=float)
    with pytest.raises(ValueError, match="resamples must be a positive int"):
        one_sided_p(diffs, np.random.default_rng(0), resamples=resamples)


def test_one_sided_p_is_exact_at_the_two_ends() -> None:
    rng = np.random.default_rng(1)
    assert one_sided_p(np.asarray([0.2, 0.4, 0.1]), rng, resamples=500) == 0.0
    assert one_sided_p(np.asarray([-0.2, -0.4, -0.1]), rng, resamples=500) == 1.0


def test_required_n_matches_the_normal_approximation() -> None:
    assert required_n(0.0) is None
    assert required_n(0.1, effect=0.0) is None
    assert required_n(0.1) == pytest.approx(((1.6449 + 0.8416) * 0.1 / 0.05) ** 2)
