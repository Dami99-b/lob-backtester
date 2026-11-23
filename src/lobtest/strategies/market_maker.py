"""Inventory-aware market maker.

Earns the spread by resting a bid and an ask at the touch. Its risk is
inventory: if it keeps getting filled on one side (adverse selection), its
position grows. So it

1. **skews** quotes away from its inventory: when long it lowers both prices
   (less keen to buy, more keen to sell), when short it raises them; and
2. **caps** position: at ``max_position`` it stops quoting the side that would
   grow the position further.

The skew is linear in inventory: ``skew_ticks = round(skew_per_unit * position)``.
Real desks use richer models (e.g. Avellaneda-Stoikov, which derives the skew
from volatility and risk aversion); linear skew is the simplest version that
shows the idea.
"""

from __future__ import annotations

from lobtest.events import ASK, BID
from lobtest.strategies.base import Quote


class MarketMaker:
    def __init__(self, size: int = 100, max_position: int = 1_000, skew_ticks_at_max: int = 2):
        self.size = size
        self.max_position = max_position
        self.skew_per_unit = skew_ticks_at_max / max_position

    def on_book(self, ts_ns: int, best_bid: int, best_ask: int, position: int) -> list[Quote]:
        if best_bid < 0 or best_ask < 0:
            return []
        skew = round(self.skew_per_unit * position)
        bid = best_bid - max(skew, 0)
        ask = best_ask - min(skew, 0)
        if ask <= bid:
            return []
        quotes = []
        if position + self.size <= self.max_position:
            quotes.append(Quote(BID, bid, self.size))
        if position - self.size >= -self.max_position:
            quotes.append(Quote(ASK, ask, self.size))
        return quotes
