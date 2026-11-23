"""ITCH 5.0 parser, checked against hand-encoded messages (spec field layouts)."""

import io
import struct

import numpy as np

from lobtest.book import replay
from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE
from lobtest.loaders.itch import parse


def frame(body: bytes) -> bytes:
    return struct.pack(">H", len(body)) + body


def header(kind: bytes, ts: int) -> bytes:
    return kind + struct.pack(">HH", 1, 0) + ts.to_bytes(6, "big")


def add(ts, ref, side, shares, stock, price, mpid=False):
    body = header(b"F" if mpid else b"A", ts) + struct.pack(
        ">QcI8sI", ref, side, shares, stock.ljust(8), price
    )
    return frame(body + (b"MPID" if mpid else b""))


def executed(ts, ref, shares):
    return frame(header(b"E", ts) + struct.pack(">QIQ", ref, shares, 1))


def executed_px(ts, ref, shares, price):
    return frame(header(b"C", ts) + struct.pack(">QIQcI", ref, shares, 1, b"Y", price))


def cancel(ts, ref, shares):
    return frame(header(b"X", ts) + struct.pack(">QI", ref, shares))


def delete(ts, ref):
    return frame(header(b"D", ts) + struct.pack(">Q", ref))


def replace(ts, ref, new_ref, shares, price):
    return frame(header(b"U", ts) + struct.pack(">QQII", ref, new_ref, shares, price))


def system_event(ts):
    return frame(header(b"S", ts) + b"O")


def test_message_lengths_match_the_spec():
    assert len(add(0, 1, b"B", 1, b"AAPL", 1)) - 2 == 36
    assert len(add(0, 1, b"B", 1, b"AAPL", 1, mpid=True)) - 2 == 40
    assert len(executed(0, 1, 1)) - 2 == 31
    assert len(executed_px(0, 1, 1, 1)) - 2 == 36
    assert len(cancel(0, 1, 1)) - 2 == 23
    assert len(delete(0, 1)) - 2 == 19
    assert len(replace(0, 1, 2, 1, 1)) - 2 == 35


def test_parse_translates_every_message_type_for_one_stock():
    data = b"".join(
        [
            system_event(1),
            add(10, 1, b"B", 500, b"AAPL", 1_500_000),
            add(11, 2, b"S", 300, b"AAPL", 1_500_100, mpid=True),
            add(12, 3, b"B", 999, b"MSFT", 4_000_000),
            executed(20, 1, 100),
            executed_px(21, 2, 50, 1_500_200),
            cancel(22, 1, 150),
            executed(23, 3, 10),
            replace(30, 2, 4, 400, 1_500_000 + 300),
            delete(40, 1),
        ]
    )
    m = parse(io.BytesIO(data), "AAPL")
    expected = [
        (10, ADD, BID, 1, 15000, 500),
        (11, ADD, ASK, 2, 15001, 300),
        (20, EXECUTE, BID, 1, 15000, 100),
        (21, EXECUTE, ASK, 2, 15001, 50),
        (22, CANCEL, BID, 1, 15000, 150),
        (30, CANCEL, ASK, 2, 15001, 250),
        (30, ADD, ASK, 4, 15003, 400),
        (40, CANCEL, BID, 1, 15000, 250),
    ]
    assert [tuple(int(x) for x in row) for row in m] == expected
    book = replay(m)
    assert book.best_bid() is None
    assert book.best_ask() == (15003, 400)


def test_truncated_file_raises():
    data = add(10, 1, b"B", 500, b"AAPL", 1_500_000)[:-3]
    try:
        parse(io.BytesIO(data), "AAPL")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_empty_stream_gives_empty_array():
    assert parse(io.BytesIO(b""), "AAPL").shape == (0,)
    assert np.dtype(parse(io.BytesIO(b""), "AAPL").dtype) is not None
