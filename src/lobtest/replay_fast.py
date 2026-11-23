"""Numba replay core: the same book as ``book.py``, built for speed.

Why it is fast (each point is a likely interview question):

1. **No Python objects in the loop.** Messages arrive as a NumPy structured
   array; the loop is compiled by ``numba.njit`` to machine code.
2. **Dense arrays instead of hash maps.** Order ids are remapped once, up front,
   to 0..n_orders-1 with ``np.unique(return_inverse=True)``, so per-order state
   lives in plain arrays indexed by that slot. Prices are mapped to
   ``price - min_price``, so each side's levels are one ``int64`` array indexed by
   price. Lookups become array indexing: no hashing, contiguous memory.
3. **Best price tracked incrementally.** Adding at a better price moves the best
   pointer in O(1). Emptying the best level scans outward to the next non-empty
   level; in real books liquidity sits near the touch, so that scan is short.

The trade-off: the price arrays are sized to the day's price range. That is
cheap for equities and crypto in integer ticks (thousands to millions of
levels), but a pathological range would need a sparse structure instead.
"""

from __future__ import annotations

import numba
import numpy as np

from lobtest.events import ADD, BID

EMPTY = -1


@numba.njit(cache=True)
def _replay(kinds, sides, slots, prices, qtys, n_slots, n_prices):  # pragma: no cover - jitted
    n = kinds.shape[0]
    out = np.empty((n, 4), dtype=np.int64)
    bid_qty = np.zeros(n_prices, dtype=np.int64)
    ask_qty = np.zeros(n_prices, dtype=np.int64)
    order_side = np.zeros(n_slots, dtype=np.int8)
    order_price = np.zeros(n_slots, dtype=np.int64)
    order_qty = np.zeros(n_slots, dtype=np.int64)
    best_bid = -1
    best_ask = -1

    for i in range(n):
        slot = slots[i]
        q = qtys[i]
        if kinds[i] == ADD:
            p = prices[i]
            order_side[slot] = sides[i]
            order_price[slot] = p
            order_qty[slot] = q
            if sides[i] == BID:
                bid_qty[p] += q
                if best_bid == -1 or p > best_bid:
                    best_bid = p
            else:
                ask_qty[p] += q
                if best_ask == -1 or p < best_ask:
                    best_ask = p
        else:
            p = order_price[slot]
            order_qty[slot] -= q
            if order_side[slot] == BID:
                bid_qty[p] -= q
                if p == best_bid and bid_qty[p] == 0:
                    j = p - 1
                    while j >= 0 and bid_qty[j] == 0:
                        j -= 1
                    best_bid = j
            else:
                ask_qty[p] -= q
                if p == best_ask and ask_qty[p] == 0:
                    j = p + 1
                    while j < n_prices and ask_qty[j] == 0:
                        j += 1
                    best_ask = j if j < n_prices else -1

        if best_bid >= 0:
            out[i, 0] = best_bid
            out[i, 1] = bid_qty[best_bid]
        else:
            out[i, 0] = EMPTY
            out[i, 1] = EMPTY
        if best_ask >= 0:
            out[i, 2] = best_ask
            out[i, 3] = ask_qty[best_ask]
        else:
            out[i, 2] = EMPTY
            out[i, 3] = EMPTY
    return out


def replay_top_of_book(messages: np.ndarray) -> np.ndarray:
    """Return (best_bid, bid_qty, best_ask, ask_qty) after each message; -1 = empty side.

    Assumes a valid stream (cancels reference live orders). Validate untrusted
    data with the reference ``OrderBook`` first; the fast path skips checks.
    """
    if len(messages) == 0:
        return np.empty((0, 4), dtype=np.int64)
    _, slots = np.unique(messages["order_id"], return_inverse=True)
    min_price = int(messages["price"].min())
    rel_prices = (messages["price"] - min_price).astype(np.int64)
    n_prices = int(rel_prices.max()) + 1
    out = _replay(
        messages["type"].astype(np.int8),
        messages["side"].astype(np.int8),
        slots.astype(np.int64),
        rel_prices,
        messages["qty"].astype(np.int64),
        int(slots.max()) + 1,
        n_prices,
    )
    for col in (0, 2):
        mask = out[:, col] != EMPTY
        out[mask, col] += min_price
    return out
