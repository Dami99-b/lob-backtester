"""Deterministic synthetic L3 message stream for tests and benchmarks.

Generates a valid stream: every CANCEL/EXECUTE references a live order and never
removes more than its remaining quantity. Good enough to exercise the book and
measure throughput before you download real data.
"""

from __future__ import annotations

import numpy as np

from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE, MESSAGE_DTYPE


def generate(n: int, seed: int = 0, mid: int = 10_000, spread: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=MESSAGE_DTYPE)
    live: dict[int, list[int]] = {}  # order_id -> [side, price, qty]
    live_ids: list[int] = []
    next_id = 1
    ts = 0
    for i in range(n):
        ts += int(rng.integers(50, 5_000))
        roll = rng.random()
        if not live_ids or roll < 0.55:
            side = BID if rng.random() < 0.5 else ASK
            offset = int(rng.geometric(0.3))
            price = (
                mid - spread // 2 - offset + 1 if side == BID else mid + spread // 2 + offset - 1
            )
            qty = int(rng.integers(1, 20)) * 100
            oid = next_id
            next_id += 1
            live[oid] = [side, price, qty]
            live_ids.append(oid)
            out[i] = (ts, ADD, side, oid, price, qty)
            continue
        j = int(rng.integers(len(live_ids)))
        oid = live_ids[j]
        side, price, qty = live[oid]
        kind = CANCEL if roll < 0.85 else EXECUTE
        take = qty if rng.random() < 0.7 else max(1, qty // 2)
        out[i] = (ts, kind, side, oid, price, take)
        remaining = qty - take
        if remaining == 0:
            del live[oid]
            live_ids[j] = live_ids[-1]
            live_ids.pop()
        else:
            live[oid][2] = remaining
    return out
