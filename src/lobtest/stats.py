"""Performance statistics. YOUR CODE: this is where your probability and stats show.

QRT-style interviewers will ask how you know a Sharpe ratio isn't luck.
"""

from __future__ import annotations

import numpy as np


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio of per-period returns (risk-free rate = 0)."""
    raise NotImplementedError("TODO")


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: float,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    block: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for Sharpe via a moving-block bootstrap (keeps autocorrelation)."""
    raise NotImplementedError("TODO")


def deflated_sharpe(sr: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float:
    """Probability the true Sharpe > 0 after correcting for multiple testing
    (Bailey & Lopez de Prado, 2014)."""
    raise NotImplementedError("TODO")
