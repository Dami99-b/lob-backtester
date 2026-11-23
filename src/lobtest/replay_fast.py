"""Numba replay core (phase 2).

Once OrderBook works and is tested, port the hot loop here with @numba.njit and
array-based levels (e.g. a price-indexed array of quantities around the mid).
The test `test_fast_replay_matches_reference` checks your fast version produces
the same top of book as the reference after every message batch.
"""

from __future__ import annotations

import numpy as np


def replay_top_of_book(messages: np.ndarray) -> np.ndarray:
    """Return an array of (best_bid, bid_qty, best_ask, ask_qty) after each message.

    TODO: implement with numba.njit. Use -1 for an empty side.
    """
    raise NotImplementedError("TODO: implement the Numba replay core")
