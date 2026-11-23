"""Replay throughput benchmark. Run: python bench/bench_replay.py [n_messages]

Prints messages/second for the pure-Python book and the Numba core (when they
exist) and saves a JSON record under bench/results/ with machine info, so every
number you put on your CV is reproducible.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from lobtest.backtest.engine import Config, run
from lobtest.book import replay
from lobtest.replay_fast import replay_top_of_book
from lobtest.strategies.market_maker import MarketMaker
from lobtest.synthetic import generate


def timed(fn, msgs, repeats=5):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(msgs)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    msgs = generate(n, seed=42)
    results = {
        "n_messages": n,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "when": datetime.now(UTC).isoformat(),
    }

    def backtest(m):
        return run(m, MarketMaker(), Config(sample_every_ns=1_000_000))

    for name, fn in [
        ("python_book", replay),
        ("numba_core", replay_top_of_book),
        ("backtest", backtest),
    ]:
        try:
            if name == "numba_core":
                fn(msgs[:1_000])  # JIT warm-up, not timed
            secs = timed(fn, msgs, repeats=3 if name == "backtest" else 5)
            results[name] = {"seconds": secs, "msgs_per_s": n / secs}
            print(f"{name:12s} {n / secs:>14,.0f} msgs/s")
        except NotImplementedError:
            print(f"{name:12s} not implemented yet")
    out = Path("bench/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"replay-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
