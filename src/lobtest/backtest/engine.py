"""Event-driven backtest engine. YOUR CODE (scaffolded).

Design rules that make a backtest credible (interviewers check these):
1. Latency: a strategy order sent at t reaches the book at t + latency_ns.
2. Queue position: a passive order only fills after the volume ahead of it at
   that price has traded or been cancelled.
3. Fees: maker/taker fees in basis points, charged on every fill.
4. No look-ahead: the strategy only sees book state up to the current message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    latency_ns: int = 1_000_000  # 1 ms
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.0


@dataclass
class Result:
    pnl: np.ndarray = field(default_factory=lambda: np.zeros(0))  # mark-to-market PnL series
    positions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fills: int = 0
    fees_paid: float = 0.0


def run(messages: np.ndarray, strategy, config: Config | None = None) -> Result:
    """TODO: replay messages through OrderBook, feed the strategy, simulate fills."""
    raise NotImplementedError("TODO: implement the backtest loop")
