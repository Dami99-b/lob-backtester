"""NASDAQ TotalView-ITCH 5.0 parser -> lobtest L3 messages for one stock.

ITCH is NASDAQ's binary, market-by-order feed: every order added, executed,
cancelled, deleted or replaced on the exchange, timestamped to the nanosecond.
Sample days are published at https://emi.nasdaq.com/ITCH/ (gzip files, several GB).

File framing: each message is a 2-byte big-endian length followed by the
message. All integers are big-endian; prices have 4 implied decimals; the
timestamp is a 6-byte count of nanoseconds since midnight.

Message types used (others are skipped by their length prefix):

=====  ==================  ==============================================
Type   Meaning             Translated to
=====  ==================  ==============================================
A, F   Add order           ADD
E      Executed            EXECUTE at the order's resting price
C      Executed w/ price   EXECUTE at the order's resting price
X      Partial cancel      CANCEL of the cancelled shares
D      Delete              CANCEL of all remaining shares
U      Replace             CANCEL of the old order + ADD of the new one
=====  ==================  ==============================================

Delete and Replace don't carry the remaining quantity, so the parser keeps a
small table of live orders for the selected stock. Prices are converted to
integer ticks of ``tick`` (default one cent = 100 in ITCH's 4-decimal units).

Usage: python -m lobtest.loaders.itch <file.itch[.gz]> <SYMBOL> <out.npy>
"""

from __future__ import annotations

import gzip
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from lobtest.events import ADD, ASK, BID, CANCEL, EXECUTE, MESSAGE_DTYPE

_HEADER = struct.Struct(">cHH6s")
_ADD = struct.Struct(">QcI8sI")
_EXEC = struct.Struct(">QIQ")
_CANCEL = struct.Struct(">QI")
_DELETE = struct.Struct(">Q")
_REPLACE = struct.Struct(">QQII")


def _timestamp(raw: bytes) -> int:
    return int.from_bytes(raw, "big")


def _open(path: str | Path) -> BinaryIO:
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def iter_messages(stream: BinaryIO) -> Iterator[bytes]:
    """Yield raw ITCH messages (without the length prefix)."""
    read = stream.read
    while True:
        prefix = read(2)
        if len(prefix) < 2:
            return
        (length,) = struct.unpack(">H", prefix)
        body = read(length)
        if len(body) < length:
            raise ValueError("truncated ITCH message at end of file")
        yield body


def parse(stream: BinaryIO, symbol: str, tick: int = 100) -> np.ndarray:
    """Translate the ITCH messages for ``symbol`` into a lobtest message array."""
    stock = symbol.encode().ljust(8)
    live: dict[int, list[int]] = {}
    rows: list[tuple[int, int, int, int, int, int]] = []
    for msg in iter_messages(stream):
        kind = msg[:1]
        if kind not in b"AFEXDCU":
            continue
        _, _, _, raw_ts = _HEADER.unpack_from(msg)
        ts = _timestamp(raw_ts)
        if kind in (b"A", b"F"):
            ref, side_c, shares, msg_stock, price = _ADD.unpack_from(msg, 11)
            if msg_stock != stock:
                continue
            side = BID if side_c == b"B" else ASK
            p = price // tick
            live[ref] = [side, p, shares]
            rows.append((ts, ADD, side, ref, p, shares))
            continue
        ref = struct.unpack_from(">Q", msg, 11)[0]
        order = live.get(ref)
        if order is None:
            continue
        side, p, remaining = order
        if kind in (b"E", b"C"):
            _, shares, _ = _EXEC.unpack_from(msg, 11)
            rows.append((ts, EXECUTE, side, ref, p, shares))
            remaining -= shares
        elif kind == b"X":
            _, shares = _CANCEL.unpack_from(msg, 11)
            rows.append((ts, CANCEL, side, ref, p, shares))
            remaining -= shares
        elif kind == b"D":
            rows.append((ts, CANCEL, side, ref, p, remaining))
            remaining = 0
        else:
            _, new_ref, shares, price = _REPLACE.unpack_from(msg, 11)
            rows.append((ts, CANCEL, side, ref, p, remaining))
            remaining = 0
            new_p = price // tick
            live[new_ref] = [side, new_p, shares]
            rows.append((ts, ADD, side, new_ref, new_p, shares))
        if remaining <= 0:
            del live[ref]
        else:
            order[2] = remaining
    return np.array(rows, dtype=MESSAGE_DTYPE)


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        sys.exit("usage: python -m lobtest.loaders.itch <file.itch[.gz]> <SYMBOL> <out.npy>")
    with _open(argv[1]) as f:
        messages = parse(f, argv[2])
    np.save(argv[3], messages)
    print(f"wrote {len(messages):,} {argv[2]} messages to {argv[3]}")


if __name__ == "__main__":
    main(sys.argv)
