"""Convert Binance public trade CSVs (data.binance.vision) to Parquet.

Binance spot `trades` daily files have no header and these columns:
trade_id, price, qty, quote_qty, time, is_buyer_maker, is_best_match

Usage: python -m lobtest.loaders.binance BTCUSDT-trades-2026-11-01.csv data/btc.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


def load_trades_csv(path: str | Path, tick_size: float = 0.01) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if not df.empty and not str(df.iloc[0]["trade_id"]).isdigit():  # some files include a header
        df = df.iloc[1:]
    df = df.astype({"trade_id": "int64", "price": "float64", "qty": "float64", "time": "int64"})
    # Binance switched from milliseconds to microseconds in 2025; normalise to ns.
    unit = 1_000 if df["time"].max() > 10**15 else 1_000_000
    return pd.DataFrame(
        {
            "ts_ns": df["time"] * unit,
            "price_ticks": (df["price"] / tick_size).round().astype("int64"),
            "qty": df["qty"],
            "aggressor_buy": ~df["is_buyer_maker"].astype(str).str.lower().eq("true"),
        }
    )


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: python -m lobtest.loaders.binance <trades.csv> <out.parquet>")
    df = load_trades_csv(argv[1])
    df.to_parquet(argv[2], index=False)
    print(f"wrote {len(df):,} trades to {argv[2]}")


if __name__ == "__main__":
    main(sys.argv)
