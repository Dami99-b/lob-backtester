"""Simple inventory-aware market maker. YOUR STRATEGY CODE.

Idea: quote one tick inside/at the touch on both sides; skew quotes away from
your inventory so position mean-reverts toward zero; cap position.
"""

from __future__ import annotations

from lobtest.strategies.base import Order


class MarketMaker:
    def __init__(self, size: int = 100, max_position: int = 1_000) -> None:
        self.size = size
        self.max_position = max_position

    def on_book(self, ts_ns: int, best_bid: int, best_ask: int, position: int) -> list[Order]:
        raise NotImplementedError("TODO: implement your quoting logic")

    def on_fill(self, ts_ns: int, order: Order, fill_qty: int, fill_price: int) -> None:
        pass
