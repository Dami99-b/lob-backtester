"""Spec for OrderBook. Remove @todo from each test as you make it pass."""

import numpy as np
import pytest
from conftest import todo
from hypothesis import given, settings
from hypothesis import strategies as st

from lobtest.book import OrderBook, replay
from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE, MESSAGE_DTYPE
from lobtest.synthetic import generate


def msg(t, kind, side, oid, price, qty):
    return np.array([(t, kind, side, oid, price, qty)], dtype=MESSAGE_DTYPE)[0]


@todo
def test_add_sets_top_of_book():
    b = OrderBook()
    b.apply(msg(1, ADD, BID, 1, 99, 100))
    b.apply(msg(2, ADD, BID, 2, 99, 50))
    b.apply(msg(3, ADD, ASK, 3, 101, 70))
    assert b.best_bid() == (99, 150)
    assert b.best_ask() == (101, 70)


@todo
def test_partial_and_full_cancel():
    b = OrderBook()
    b.apply(msg(1, ADD, BID, 1, 99, 100))
    b.apply(msg(2, ADD, BID, 2, 98, 100))
    b.apply(msg(3, CANCEL, BID, 1, 99, 40))
    assert b.best_bid() == (99, 60)
    b.apply(msg(4, CANCEL, BID, 1, 99, 60))
    assert b.best_bid() == (98, 100)


@todo
def test_execute_removes_liquidity_and_empty_side_is_none():
    b = OrderBook()
    b.apply(msg(1, ADD, ASK, 1, 101, 100))
    b.apply(msg(2, EXECUTE, ASK, 1, 101, 100))
    assert b.best_ask() is None


@todo
def test_depth_is_sorted_best_first():
    b = OrderBook()
    for i, p in enumerate([97, 99, 98]):
        b.apply(msg(i, ADD, BID, i + 1, p, 10))
    assert b.depth(BID, 2) == [(99, 10), (98, 10)]


@todo
def test_bad_messages_raise():
    b = OrderBook()
    with pytest.raises(ValueError):
        b.apply(msg(1, CANCEL, BID, 42, 99, 1))
    b.apply(msg(2, ADD, BID, 1, 99, 10))
    with pytest.raises(ValueError):
        b.apply(msg(3, CANCEL, BID, 1, 99, 11))


@todo
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(1, 3_000))
def test_total_quantity_matches_live_orders(seed, n):
    msgs = generate(n, seed=seed)
    book = replay(msgs)
    live: dict[int, tuple[int, int]] = {}
    for m in msgs:
        oid, side, qty = int(m["order_id"]), int(m["side"]), int(m["qty"])
        if m["type"] == ADD:
            live[oid] = (side, qty)
        else:
            s, q = live[oid]
            live[oid] = (s, q - qty)
    for side in (BID, ASK):
        expected = sum(q for s, q in live.values() if s == side)
        got = sum(q for _, q in book.depth(side, 10**9))
        assert got == expected
