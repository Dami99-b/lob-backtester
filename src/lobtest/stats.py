"""Performance statistics: how good is a strategy, and how sure are we?

A single Sharpe ratio is a point estimate from one noisy sample. These tools
answer the questions a quant researcher actually asks:

- ``sharpe``: the point estimate.
- ``bootstrap_sharpe_ci``: how uncertain it is. Uses a *moving-block* bootstrap,
  because strategy returns are autocorrelated (inventory carries over from one
  period to the next); resampling single returns would destroy that structure
  and make the interval too narrow.
- ``deflated_sharpe``: was it luck? If you try many parameter settings and keep
  the best, the winner's Sharpe is inflated. The Deflated Sharpe Ratio (Bailey &
  Lopez de Prado, 2014) is the probability the true Sharpe exceeds what the best
  of ``n_trials`` pure-noise strategies would have scored, also correcting for
  skewed and fat-tailed returns.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (risk-free rate 0, sample std with ddof=1)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("need at least two returns")
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: float,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    block: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """(1 - alpha) percentile interval for the Sharpe ratio, moving-block bootstrap.

    Each resample stitches together random contiguous blocks of ``block`` returns
    until it has the original length, so short-range dependence is preserved.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 2 * block:
        raise ValueError("series too short for the chosen block length")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    srs = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0)
    srs *= math.sqrt(periods_per_year)
    lo, hi = np.quantile(srs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _sharpe_std(sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """Std error of a per-period Sharpe estimate under non-normal returns
    (Mertens 2002). ``kurt`` is raw kurtosis (3 for a normal distribution)."""
    return math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n_obs - 1))


def deflated_sharpe(sr: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    """Probability that the true (per-period) Sharpe beats the best of ``n_trials``
    noise strategies. ``sr`` must be per-period, not annualised.

    With one trial this reduces to the Probabilistic Sharpe Ratio against 0.
    """
    if n_trials < 1 or n_obs < 2:
        raise ValueError("need n_trials >= 1 and n_obs >= 2")
    se = _sharpe_std(sr, n_obs, skew, kurt)
    if n_trials == 1:
        benchmark = 0.0
    else:
        expected_max_z = (1 - _EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + _EULER_GAMMA * (
            _N.inv_cdf(1 - 1 / (n_trials * math.e))
        )
        benchmark = se * expected_max_z
    return _N.cdf((sr - benchmark) / se)


def describe(returns: np.ndarray, periods_per_year: float, n_trials: int = 1) -> dict[str, float]:
    """Everything above in one dict, for reports."""
    r = np.asarray(returns, dtype=float)
    if r.size < 4:
        raise ValueError(f"need more return samples to describe a strategy (got {r.size})")
    centered = r - r.mean()
    sd = r.std(ddof=0)
    skew = float((centered**3).mean() / sd**3) if sd > 0 else 0.0
    kurt = float((centered**4).mean() / sd**4) if sd > 0 else 3.0
    per_period = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    lo, hi = bootstrap_sharpe_ci(r, periods_per_year, block=max(2, min(20, r.size // 10)))
    return {
        "sharpe": sharpe(r, periods_per_year),
        "sharpe_ci_low": lo,
        "sharpe_ci_high": hi,
        "deflated_sharpe": deflated_sharpe(per_period, n_trials, r.size, skew, kurt),
        "skew": skew,
        "kurtosis": kurt,
    }
