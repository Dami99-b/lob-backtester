"""Limit order book rebuilt from L3 messages.

YOUR CORE CODE. Interviewers will ask about every choice here, so write it
yourself. Spec (the tests in tests/test_book.py check all of it):

- apply(msg) handles ADD, CANCEL (partial or full) and EXECUTE (partial or full).
- best_bid() / best_ask() return (price, total_qty) or None when that side is empty.
- depth(side, levels) returns the top `levels` price levels as [(price, qty), ...],
  best first.
- Unknown order ids or over-sized cancels raise ValueError.

Start with dicts (order_id -> order, price -> level qty) plus sorted price lists,
measure it with bench/bench_replay.py, then optimise. Record each version's
numbers in docs/PERFORMANCE.md: the before/after story is what makes the CV line.
"""

from __future__ import annotations

import numpy as np


class OrderBook:
    def __init__(self) -> None:
        # TODO: choose your data structures.
        pass

    def apply(self, msg: np.void) -> None:
        raise NotImplementedError("TODO: implement OrderBook.apply")

    def best_bid(self) -> tuple[int, int] | None:
        raise NotImplementedError("TODO: implement OrderBook.best_bid")

    def best_ask(self) -> tuple[int, int] | None:
        raise NotImplementedError("TODO: implement OrderBook.best_ask")

    def depth(self, side: int, levels: int) -> list[tuple[int, int]]:
        raise NotImplementedError("TODO: implement OrderBook.depth")


def replay(messages: np.ndarray, book: OrderBook | None = None) -> OrderBook:
    """Pure-Python replay. This is the baseline your Numba version must beat."""
    book = book or OrderBook()
    for msg in messages:
        book.apply(msg)
    return book
