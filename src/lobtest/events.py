"""Market-by-order (L3) message schema shared by every module.

Messages are stored as a NumPy structured array so the replay core can run in
Numba without Python objects. Prices are integer ticks, never floats.
"""

from __future__ import annotations

import numpy as np

ADD = 0
CANCEL = 1
EXECUTE = 2

BID = 0
ASK = 1

MESSAGE_DTYPE = np.dtype(
    [
        ("ts_ns", np.int64),
        ("type", np.int8),
        ("side", np.int8),
        ("order_id", np.int64),
        ("price", np.int64),
        ("qty", np.int64),
    ]
)


def empty(n: int) -> np.ndarray:
    """Allocate an uninitialised message array of length n."""
    return np.empty(n, dtype=MESSAGE_DTYPE)
