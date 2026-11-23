"""Market-by-order (L3) message schema shared by every module.

Messages are stored as a NumPy structured array so the replay core can run in
Numba without Python objects. Prices are integer ticks, never floats.
"""

from __future__ import annotations

import numpy as np

# Message types
ADD = 0  # new resting order
CANCEL = 1  # remove (part of) a resting order
EXECUTE = 2  # resting order traded against an incoming aggressor

# Sides
BID = 0
ASK = 1

MESSAGE_DTYPE = np.dtype(
    [
        ("ts_ns", np.int64),  # exchange timestamp, nanoseconds
        ("type", np.int8),  # ADD / CANCEL / EXECUTE
        ("side", np.int8),  # BID / ASK
        ("order_id", np.int64),
        ("price", np.int64),  # integer ticks
        ("qty", np.int64),  # shares / lots
    ]
)


def empty(n: int) -> np.ndarray:
    """Allocate an uninitialised message array of length n."""
    return np.empty(n, dtype=MESSAGE_DTYPE)
