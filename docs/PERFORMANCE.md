# Performance log

All numbers from `bench/bench_replay.py 1000000` and `bench/bench_itch.py 1000000`
on the synthetic stream (seed 42), Linux, Python 3.13, 2 vCPUs. Median of runs.
Raw JSON results are written to `bench/results/`.

| Date | Component | Implementation | msgs/s |
|---|---|---|---:|
| 2026-11-23 | Book replay | Pure Python: dicts + heaps with lazy deletion | ~410,000 |
| 2026-11-23 | Book replay | Numba: remapped ids, price-indexed arrays | ~9,000,000 (8.0-10.7M) |
| 2026-11-23 | Backtest | Engine + market maker (Python book) | ~150,000 |
| 2026-11-23 | ITCH parse | Pure Python `struct` streaming | ~460,000 |

Next targets: run the backtest on the Numba core, and a bulk ITCH parser.
