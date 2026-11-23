# lob-backtester

Event-driven backtesting on real limit-order-book data, with a fast replay core.

**Status:** skeleton. The data tooling, tests, benchmarks and CI are in place;
the book, replay core, engine, strategy and statistics are being implemented.

## Layout

| Path | What | Status |
|---|---|---|
| `src/lobtest/events.py` | L3 message schema (NumPy structured dtype, integer ticks) | done |
| `src/lobtest/synthetic.py` | Deterministic, valid synthetic message stream | done |
| `src/lobtest/loaders/binance.py` | Binance trade CSV to Parquet | done |
| `src/lobtest/book.py` | Order book rebuilt from L3 messages | TODO |
| `src/lobtest/replay_fast.py` | Numba replay core | TODO |
| `src/lobtest/backtest/engine.py` | Latency, queue position, fees, no look-ahead | TODO |
| `src/lobtest/strategies/market_maker.py` | Inventory-aware market maker | TODO |
| `src/lobtest/stats.py` | Sharpe, block-bootstrap CI, deflated Sharpe | TODO |
| `bench/bench_replay.py` | Throughput benchmark, saves JSON results | done |

## Quick start

```bash
pip install -e ".[dev]"
pytest                          # TODO tests show as xfail until implemented
python bench/bench_replay.py    # msgs/s for each replay implementation
```

## Roadmap

1. **Book:** make `tests/test_book.py` pass; record the baseline in `docs/PERFORMANCE.md`.
2. **Fast core:** Numba replay matching the reference message-for-message; report the speed-up.
3. **Real data:** load Binance / NASDAQ ITCH samples (see `data/README.md`).
4. **Engine:** latency, queue position, maker/taker fees; market-maker strategy.
5. **Statistics:** Sharpe with block-bootstrap CI, deflated Sharpe, walk-forward evaluation.
