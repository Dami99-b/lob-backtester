import numpy as np

from lobtest.events import ADD, ASK, BID
from lobtest.synthetic import generate


def test_stream_is_valid_and_deterministic():
    a = generate(20_000, seed=1)
    b = generate(20_000, seed=1)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a["ts_ns"]) > 0)
    live: dict[int, int] = {}
    for m in a:
        if m["type"] == ADD:
            assert m["order_id"] not in live
            live[int(m["order_id"])] = int(m["qty"])
        else:
            remaining = live[int(m["order_id"])] - int(m["qty"])
            assert remaining >= 0
            if remaining == 0:
                del live[int(m["order_id"])]
            else:
                live[int(m["order_id"])] = remaining


def test_book_never_crosses_by_construction():
    msgs = generate(5_000, seed=2)
    adds = msgs[msgs["type"] == ADD]
    assert adds[adds["side"] == BID]["price"].max() < adds[adds["side"] == ASK]["price"].min()
