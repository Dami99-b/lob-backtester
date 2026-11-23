"""Event-driven backtest engine for passive (resting) strategies on L3 data.

The engine replays market messages one at a time and simulates how *our*
orders would have fared. The four modelling rules below are what separate a
credible market-making backtest from a fantasy one:

1. **Latency.** A quote the strategy decides on at time t only reaches the
   exchange at t + ``latency_ns``. Cancels are delayed the same way, so an order
   we have already decided to pull can still be filled in the meantime, as in
   real life.

2. **Queue position (price-time priority).** When our order arrives at a price
   level, everyone already resting there is ahead of us: ``ahead = level qty at
   arrival``. Executions at our price consume ``ahead`` first; only the excess
   fills us. Cancels at our level are assumed to come from *behind* us unless
   the level shrinks below ``ahead`` (then ``ahead`` is clamped to the level).
   This is deliberately conservative: real queues lose some volume in front of
   you to cancels, so true fills are, if anything, a little better.

3. **Trade-through fills.** If an execution happens at a price *worse* than
   ours on our side (e.g. a sell at 99 while we bid 100), the aggressor would
   have hit us first, so we fill (up to that execution's size).

4. **No look-ahead.** The strategy only sees the book after the message has
   been applied, and its new quotes only act on later messages.

Simplifications, stated plainly: our orders do not change the book other
traders see (no market impact; fine for small size), and only passive
strategies are supported (no crossing the spread, so no taker fees).
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field

import numpy as np

from lobtest.book import OrderBook
from lobtest.events import ASK, BID, EXECUTE
from lobtest.strategies.base import Quote


@dataclass
class Config:
    latency_ns: int = 1_000_000  # 1 ms each way between decision and exchange
    maker_fee_bps: float = 1.0  # fee on each passive fill, in basis points of notional
    tick_size: float = 0.01  # currency per price tick
    sample_every_ns: int = 1_000_000_000  # mark-to-market sampling interval (1 s)


@dataclass
class Result:
    ts_ns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    pnl: np.ndarray = field(default_factory=lambda: np.zeros(0))  # mark-to-market, after fees
    position: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fills: int = 0
    volume: int = 0
    fees_paid: float = 0.0


@dataclass
class _Resting:
    price: int
    qty: int
    ahead: int  # resting volume in front of us at this price


def run(
    messages: np.ndarray, strategy, config: Config | None = None, book: OrderBook | None = None
) -> Result:
    """Backtest ``strategy`` over ``messages``.

    ``book`` is the market state before the first message (e.g. a snapshot when
    evaluating a slice of the day); it is copied, never modified.
    """
    cfg = config or Config()
    book = copy.deepcopy(book) if book is not None else OrderBook()
    resting: dict[int, _Resting] = {}  # side -> our live order at the exchange
    intended: dict[int, tuple[int, int] | None] = {BID: None, ASK: None}  # side -> (price, qty)
    # Pending exchange actions: (arrival_ts, seq, side, price, qty); qty 0 = cancel.
    pending: list[tuple[int, int, int, int, int]] = []
    seq = 0

    position = 0
    cash = 0.0
    fees = 0.0
    fills = volume = 0
    samples_ts: list[int] = []
    samples_pnl: list[float] = []
    samples_pos: list[int] = []
    next_sample = int(messages["ts_ns"][0]) if len(messages) else 0

    def fill(side: int, price: int, qty: int) -> None:
        nonlocal position, cash, fees, fills, volume
        notional = price * cfg.tick_size * qty
        fee = notional * cfg.maker_fee_bps / 10_000
        if side == BID:  # our bid was hit: we bought
            position += qty
            cash -= notional
        else:
            position -= qty
            cash += notional
        cash -= fee
        fees += fee
        fills += 1
        volume += qty

    last_top: tuple[int, int] | None = None
    for msg in messages:
        t = int(msg["ts_ns"])

        # 1. Deliver our orders/cancels whose latency has elapsed.
        while pending and pending[0][0] <= t:
            _, _, side, price, qty = heapq.heappop(pending)
            if qty == 0:
                resting.pop(side, None)
            else:
                resting[side] = _Resting(price, qty, book.level_qty(side, price))

        # 2. Match the market message against our resting orders, then apply it.
        kind, side, price, qty = (
            int(msg["type"]),
            int(msg["side"]),
            int(msg["price"]),
            int(msg["qty"]),
        )
        ours = resting.get(side)
        if kind == EXECUTE and ours is not None:
            at_our_price = price == ours.price
            through_us = (price < ours.price) if side == BID else (price > ours.price)
            filled = 0
            if at_our_price:
                ours.ahead -= qty
                if ours.ahead < 0:
                    filled = min(-ours.ahead, ours.qty)
                    ours.ahead = 0
            elif through_us:
                filled = min(qty, ours.qty)
            if filled:
                fill(side, ours.price, filled)
                ours.qty -= filled
                if ours.qty == 0:
                    del resting[side]
                    intended[side] = None  # strategy may re-quote
        book.apply(msg)
        for s, order in resting.items():
            order.ahead = min(order.ahead, book.level_qty(s, order.price))

        # 3. Let the strategy react to the new top of book.
        bb, ba = book.best_bid(), book.best_ask()
        best_bid = bb[0] if bb else -1
        best_ask = ba[0] if ba else -1
        top = (best_bid, best_ask)
        if top != last_top or not resting:
            last_top = top
            wanted = {q.side: q for q in strategy.on_book(t, best_bid, best_ask, position)}
            for s in (BID, ASK):
                q: Quote | None = wanted.get(s)
                target = (q.price, q.qty) if q else None
                if target == intended[s]:
                    continue  # unchanged quote keeps its queue position
                intended[s] = target
                seq += 1
                if target is None:
                    heapq.heappush(pending, (t + cfg.latency_ns, seq, s, 0, 0))
                else:
                    heapq.heappush(pending, (t + cfg.latency_ns, seq, s, target[0], target[1]))

        # 4. Mark to market on a fixed clock.
        if t >= next_sample and best_bid >= 0 and best_ask >= 0:
            mid = (best_bid + best_ask) / 2 * cfg.tick_size
            samples_ts.append(t)
            samples_pnl.append(cash + position * mid)
            samples_pos.append(position)
            next_sample = t + cfg.sample_every_ns

    return Result(
        ts_ns=np.array(samples_ts, dtype=np.int64),
        pnl=np.array(samples_pnl),
        position=np.array(samples_pos, dtype=np.int64),
        fills=fills,
        volume=volume,
        fees_paid=fees,
    )
