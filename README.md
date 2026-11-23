# lob-backtester

Event-driven backtesting on real limit-order-book data, with a fast replay core.

Replays market-by-order data (NASDAQ TotalView-ITCH 5.0 or a synthetic stream),
rebuilds the order book, and backtests a market-making strategy with latency,
queue position and fees, then judges it with walk-forward evaluation, block-bootstrap
confidence intervals and the Deflated Sharpe Ratio.

| Component | Throughput (1M messages, 2 vCPUs) |
|---|---:|
| Reference book (pure Python) | ~410K msgs/s |
| Numba replay core | ~9M msgs/s (about 20x) |
| Backtest engine | ~150K msgs/s |
| ITCH 5.0 parser | ~460K msgs/s |

Design decisions and their trade-offs: [docs/DESIGN.md](docs/DESIGN.md).

## Layout

| Path | What | Status |
|---|---|---|
| `src/lobtest/events.py` | L3 message schema (NumPy structured dtype, integer ticks) | done |
| `src/lobtest/synthetic.py` | Deterministic, valid synthetic message stream | done |
| `src/lobtest/loaders/binance.py` | Binance trade CSV to Parquet | done |
| `src/lobtest/book.py` | Order book rebuilt from L3 messages (dicts + lazy-deletion heaps) | done |
| `src/lobtest/replay_fast.py` | Numba replay core, checked message-for-message against the book | done |
| `src/lobtest/backtest/engine.py` | Latency, queue position, trade-throughs, fees, no look-ahead | done |
| `src/lobtest/strategies/market_maker.py` | Inventory-skewed market maker with a position cap | done |
| `src/lobtest/stats.py` | Sharpe, moving-block bootstrap CI, Deflated Sharpe Ratio | done |
| `src/lobtest/walkforward.py` | Tune on fold k-1, report fold k only | done |
| `src/lobtest/loaders/itch.py` | NASDAQ ITCH 5.0 parser (A, F, E, C, X, D, U) | done |
| `bench/bench_replay.py` | Throughput benchmark, saves JSON results | done |

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # 25 tests
python bench/bench_replay.py             # book, Numba core and backtest throughput
python -m lobtest.walkforward 1000000 5  # walk-forward report on synthetic data

# Real data: download a NASDAQ ITCH 5.0 sample day (see data/README.md), then
python -m lobtest.loaders.itch data/01302019.NASDAQ_ITCH50.gz AAPL data/aapl.npy
python -m lobtest.walkforward 0 5 data/aapl.npy
```

## Results so far

On the synthetic stream, walk-forward finds **no edge** (out-of-sample PnL about 0,
Deflated Sharpe 0.07). That is the correct answer: the synthetic market has no
signal, and at a price of 100 the one-tick spread earned per share exactly equals
the 1 bp fee. Real-data results go here once an ITCH day has been replayed.

## Next

1. Replay a full NASDAQ ITCH day and report results per stock.
2. Run the backtest on the Numba core instead of the Python book.
3. Bulk ITCH parsing (scan length prefixes with NumPy/Numba).
4. Fill mark-outs (PnL 1s / 10s / 60s after each fill) to measure adverse selection.
