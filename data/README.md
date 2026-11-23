# Data (not committed)

Real data, downloaded yourself:

- Binance public data: https://data.binance.vision (spot -> daily -> trades / aggTrades, e.g. BTCUSDT)
- NASDAQ TotalView-ITCH 5.0 samples: https://emi.nasdaq.com/ITCH/ (large .gz binary files)

Convert to Parquet with `python -m lobtest.loaders.binance <csv> data/<name>.parquet`.

Until then, tests and benchmarks use `lobtest.synthetic`, which needs no downloads.
