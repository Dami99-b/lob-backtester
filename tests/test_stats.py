import numpy as np
import pytest

from lobtest.stats import bootstrap_sharpe_ci, deflated_sharpe, sharpe


def test_sharpe_of_known_series():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, 252) == pytest.approx(expected)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2_000)
    lo, hi = bootstrap_sharpe_ci(r, 252)
    assert lo < sharpe(r, 252) < hi


def test_deflated_sharpe_drops_with_more_trials():
    one = deflated_sharpe(sr=0.1, n_trials=1, n_obs=1_000, skew=0.0, kurt=3.0)
    many = deflated_sharpe(sr=0.1, n_trials=100, n_obs=1_000, skew=0.0, kurt=3.0)
    assert 0.0 <= many < one <= 1.0


def test_single_trial_deflated_sharpe_is_probabilistic_sharpe():
    # PSR against 0 for a clearly positive Sharpe over many observations is ~1
    assert deflated_sharpe(sr=0.2, n_trials=1, n_obs=2_000, skew=0.0, kurt=3.0) > 0.999


def test_block_bootstrap_is_wider_for_autocorrelated_returns():
    rng = np.random.default_rng(1)
    e = rng.normal(0.0002, 0.01, 4_000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, e.size):  # strongly autocorrelated AR(1) returns
        ar[i] = 0.8 * ar[i - 1] + e[i]
    lo_b, hi_b = bootstrap_sharpe_ci(ar, 252, block=50)
    lo_1, hi_1 = bootstrap_sharpe_ci(ar, 252, block=1)
    assert (hi_b - lo_b) > (hi_1 - lo_1)
