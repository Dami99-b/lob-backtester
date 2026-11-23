"""ITCH parser throughput: encodes the synthetic stream as ITCH 5.0, then parses it.

Run: python bench/bench_itch.py [n_messages]
"""

from __future__ import annotations

import io
import struct
import sys
import time

from lobtest.events import ADD, BID, CANCEL
from lobtest.loaders.itch import parse
from lobtest.synthetic import generate


def encode(messages) -> bytes:
    out = bytearray()
    for m in messages:
        hdr = struct.pack(">HH", 1, 0) + int(m["ts_ns"]).to_bytes(6, "big")
        if m["type"] == ADD:
            side = b"B" if m["side"] == BID else b"S"
            body = (
                b"A"
                + hdr
                + struct.pack(
                    ">QcI8sI",
                    int(m["order_id"]),
                    side,
                    int(m["qty"]),
                    b"TEST    ",
                    int(m["price"]) * 100,
                )
            )
        elif m["type"] == CANCEL:
            body = b"X" + hdr + struct.pack(">QI", int(m["order_id"]), int(m["qty"]))
        else:
            body = b"E" + hdr + struct.pack(">QIQ", int(m["order_id"]), int(m["qty"]), 0)
        out += struct.pack(">H", len(body)) + body
    return bytes(out)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    raw = encode(generate(n, seed=42))
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        parsed = parse(io.BytesIO(raw), "TEST")
        samples.append(time.perf_counter() - t0)
    best = sorted(samples)[1]
    print(f"parsed {len(parsed):,} messages ({len(raw) / 1e6:.0f} MB) at {n / best:,.0f} msgs/s")


if __name__ == "__main__":
    main()
