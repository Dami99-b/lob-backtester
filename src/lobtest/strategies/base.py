"""Strategy interface. The engine calls on_book after every book update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Order:
    side: int  # events.BID / events.ASK
    price: int  # ticks
    qty: int
    client_id: int


class Strategy(Protocol):
    def on_book(self, ts_ns: int, best_bid: int, best_ask: int, position: int) -> list[Order]:
        """Return new orders to send (they reach the market after the latency delay)."""
        ...

    def on_fill(self, ts_ns: int, order: Order, fill_qty: int, fill_price: int) -> None: ...
