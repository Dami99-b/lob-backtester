"""Strategy interface.

A strategy is a pure decision function: given what it can see (top of book and
its own position), it returns the quotes it *wants* resting right now. The
engine owns everything about how markets actually behave: latency, queue
position, fills and fees. Keeping that split means a strategy can't cheat by
accident, and the same strategy runs unchanged on synthetic or real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Quote:
    side: int  # events.BID / events.ASK
    price: int  # ticks
    qty: int


class Strategy(Protocol):
    def on_book(self, ts_ns: int, best_bid: int, best_ask: int, position: int) -> list[Quote]:
        """Return the quotes to keep resting (at most one per side).

        Called after every book update. Returning the same price as last time
        keeps the existing order and its queue position; a new price cancels and
        replaces it, which sends it to the back of the new level's queue.
        ``best_bid`` / ``best_ask`` are -1 when that side is empty.
        """
        ...
