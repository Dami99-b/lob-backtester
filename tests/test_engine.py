"""Hand-built scenarios for each modelling rule in backtest/engine.py."""

import numpy as np
import pytest

from lobtest.backtest.engine import Config, run
from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE, MESSAGE_DTYPE
from lobtest.strategies.base import Quote
from lobtest.strategies.market_maker import MarketMaker
from lobtest.synthetic import generate


class FixedBid:
    """Always wants one bid at a fixed price."""

    def __init__(self, price, qty=100):
        self.quote = Quote(BID, price, qty)

    def on_book(self, ts_ns, best_bid, best_ask, position):
        return [self.quote]


def stream(rows):
    return np.array(rows, dtype=MESSAGE_DTYPE)


BASE = [
    (1, ADD, BID, 1, 100, 300),
    (2, ADD, ASK, 2, 101, 300),
]
CFG = Config(latency_ns=10, maker_fee_bps=0.0, tick_size=1.0, sample_every_ns=1)


def test_queue_position_executions_must_clear_volume_ahead_first():
    msgs = stream(BASE + [(20, EXECUTE, BID, 1, 100, 200), (30, EXECUTE, BID, 1, 100, 100)])
    res = run(msgs, FixedBid(100), CFG)
    assert res.fills == 0
    msgs = stream(BASE + [(20, ADD, BID, 3, 100, 100), (30, EXECUTE, BID, 1, 100, 300)])
    res = run(msgs, FixedBid(100), CFG)
    assert res.fills == 0


def test_excess_execution_fills_us():
    msgs = stream(
        BASE
        + [
            (20, ADD, BID, 3, 100, 100),
            (30, EXECUTE, BID, 1, 100, 300),
            (40, EXECUTE, BID, 3, 100, 60),
        ]
    )
    res = run(msgs, FixedBid(100), CFG)
    assert res.volume == 60
    assert res.position[-1] == 60


def test_latency_order_cannot_fill_before_it_arrives():
    msgs = stream(
        BASE
        + [(5, EXECUTE, BID, 1, 100, 300), (6, ADD, BID, 3, 99, 50), (7, EXECUTE, BID, 3, 99, 50)]
    )
    res = run(msgs, FixedBid(100), CFG)
    assert res.fills == 0


def test_trade_through_fills_resting_order():
    msgs = stream(
        [(1, ADD, BID, 1, 99, 300), (2, ADD, ASK, 2, 101, 300), (30, EXECUTE, BID, 1, 99, 80)]
    )
    res = run(msgs, FixedBid(100), CFG)
    assert res.volume == 80


def test_cancel_shrinks_queue_ahead_only_when_level_drops_below_it():
    msgs = stream(
        BASE
        + [
            (20, CANCEL, BID, 1, 100, 250),
            (30, ADD, BID, 3, 100, 100),
            (40, EXECUTE, BID, 1, 100, 50),
            (50, EXECUTE, BID, 3, 100, 30),
        ]
    )
    res = run(msgs, FixedBid(100), CFG)
    assert res.volume == 30


def test_fees_are_charged_in_basis_points_of_notional():
    msgs = stream(
        BASE
        + [
            (20, EXECUTE, BID, 1, 100, 300),
            (30, ADD, BID, 3, 100, 10),
            (40, EXECUTE, BID, 3, 100, 10),
        ]
    )
    res = run(msgs, FixedBid(100), Config(latency_ns=10, maker_fee_bps=10, tick_size=1.0))
    assert res.volume == 10
    assert res.fees_paid == pytest.approx(100 * 10 * 10 / 10_000)


def test_market_maker_respects_position_cap_on_synthetic_day():
    msgs = generate(200_000, seed=7)
    mm = MarketMaker(size=100, max_position=500)
    res = run(msgs, mm, Config(latency_ns=500_000))
    assert res.fills > 0
    assert np.abs(res.position).max() <= 500
