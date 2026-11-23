"""Limit order book rebuilt from L3 (market-by-order) messages.

Design:

- ``_orders``: order_id -> [side, price, remaining_qty]. Every CANCEL/EXECUTE
  names an order id, so an O(1) hash lookup is the natural index.
- ``_levels[side]``: price -> total resting quantity at that price. The book is
  aggregated per level because that is what best bid/ask and depth report.
- ``_heaps[side]``: a heap of prices for O(log n) best-price access. Bids are
  stored negated so both sides use Python's min-heap. Prices whose level has
  emptied are left in the heap and skipped lazily when they reach the top
  ("lazy deletion"); this keeps every removal O(1) instead of O(n).

This pure-Python version is the readable reference. ``replay_fast.py`` is the
Numba version, and a test checks the two agree after every message.
"""

from __future__ import annotations

import heapq

import numpy as np

from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE


class OrderBook:
    def __init__(self) -> None:
        self._orders: dict[int, list[int]] = {}
        self._levels: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self._heaps: tuple[list[int], list[int]] = ([], [])

    def apply(self, msg: np.void) -> None:
        kind = int(msg["type"])
        order_id = int(msg["order_id"])
        qty = int(msg["qty"])
        if qty <= 0:
            raise ValueError(f"non-positive quantity {qty} for order {order_id}")
        if kind == ADD:
            self._add(order_id, int(msg["side"]), int(msg["price"]), qty)
        elif kind in (CANCEL, EXECUTE):
            self._reduce(order_id, qty)
        else:
            raise ValueError(f"unknown message type {kind}")

    def _add(self, order_id: int, side: int, price: int, qty: int) -> None:
        if order_id in self._orders:
            raise ValueError(f"duplicate order id {order_id}")
        if side not in (BID, ASK):
            raise ValueError(f"unknown side {side}")
        self._orders[order_id] = [side, price, qty]
        level = self._levels[side]
        if price not in level:
            heapq.heappush(self._heaps[side], -price if side == BID else price)
            level[price] = 0
        level[price] += qty

    def _reduce(self, order_id: int, qty: int) -> None:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"unknown order id {order_id}")
        side, price, remaining = order
        if qty > remaining:
            raise ValueError(f"cannot remove {qty} from order {order_id} with {remaining} left")
        level = self._levels[side]
        level[price] -= qty
        if level[price] == 0:
            del level[price]
        if qty == remaining:
            del self._orders[order_id]
        else:
            order[2] = remaining - qty

    def _best(self, side: int) -> tuple[int, int] | None:
        heap, level = self._heaps[side], self._levels[side]
        while heap:
            price = -heap[0] if side == BID else heap[0]
            if price in level:
                return price, level[price]
            heapq.heappop(heap)
        return None

    def best_bid(self) -> tuple[int, int] | None:
        return self._best(BID)

    def best_ask(self) -> tuple[int, int] | None:
        return self._best(ASK)

    def depth(self, side: int, levels: int) -> list[tuple[int, int]]:
        prices = sorted(self._levels[side], reverse=(side == BID))[:levels]
        return [(p, self._levels[side][p]) for p in prices]

    def level_qty(self, side: int, price: int) -> int:
        """Total resting quantity at one price (0 if the level is empty)."""
        return self._levels[side].get(price, 0)

    def __len__(self) -> int:
        return len(self._orders)


def replay(messages: np.ndarray, book: OrderBook | None = None) -> OrderBook:
    """Pure-Python replay: the reference the Numba core is checked against."""
    book = book or OrderBook()
    for msg in messages:
        book.apply(msg)
    return book
