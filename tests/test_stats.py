import numpy as np
import pytest
from conftest import todo

from lobtest.stats import bootstrap_sharpe_ci, deflated_sharpe, sharpe


@todo
def test_sharpe_of_known_series():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, 252) == pytest.approx(expected)


@todo
def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2_000)
    lo, hi = bootstrap_sharpe_ci(r, 252)
    assert lo < sharpe(r, 252) < hi


@todo
def test_deflated_sharpe_drops_with_more_trials():
    one = deflated_sharpe(sr=0.1, n_trials=1, n_obs=1_000, skew=0.0, kurt=3.0)
    many = deflated_sharpe(sr=0.1, n_trials=100, n_obs=1_000, skew=0.0, kurt=3.0)
    assert 0.0 <= many < one <= 1.0
