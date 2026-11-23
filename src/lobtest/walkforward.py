"""Walk-forward evaluation: tune on the past, judge on the future.

Picking parameters on the same data you report results on is the classic way
to fool yourself. Walk-forward splits the day into consecutive folds; for each
fold k it picks the parameter set with the best Sharpe on fold k-1 and records
that choice's performance on fold k, which it has never seen. Only those
out-of-sample results are reported, and the Deflated Sharpe Ratio is computed
with ``n_trials`` = the size of the grid that was searched.

Run: python -m lobtest.walkforward [n_messages] [n_folds] [data.npy]
(without a data file it uses the synthetic stream)
"""

from __future__ import annotations

import copy
import itertools
import json
import sys

import numpy as np

from lobtest import stats
from lobtest.backtest.engine import Config, run
from lobtest.book import OrderBook, replay
from lobtest.strategies.market_maker import MarketMaker
from lobtest.synthetic import generate

GRID = {"size": [100, 200], "skew_ticks_at_max": [0, 1, 2, 4]}


def _returns(messages: np.ndarray, params: dict, cfg: Config, book: OrderBook) -> np.ndarray:
    res = run(messages, MarketMaker(max_position=1_000, **params), cfg, book)
    return np.diff(res.pnl)


def walk_forward(messages: np.ndarray, n_folds: int = 5, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    folds = np.array_split(messages, n_folds)
    snapshots = [OrderBook()]
    for fold in folds[:-1]:
        snapshots.append(replay(fold, copy.deepcopy(snapshots[-1])))
    grid = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
    periods_per_year = 252 * 6.5 * 3600 * 1e9 / cfg.sample_every_ns
    oos: list[np.ndarray] = []
    chosen = []
    for k in range(1, n_folds):
        scores = []
        for params in grid:
            r = _returns(folds[k - 1], params, cfg, snapshots[k - 1])
            scores.append(stats.sharpe(r, periods_per_year) if r.size > 1 else -np.inf)
        best = grid[int(np.argmax(scores))]
        chosen.append(best)
        oos.append(_returns(folds[k], best, cfg, snapshots[k]))
    oos_returns = np.concatenate(oos)
    report = stats.describe(oos_returns, periods_per_year, n_trials=len(grid))
    report.update(
        {
            "folds": n_folds,
            "grid_size": len(grid),
            "chosen_params": chosen,
            "oos_samples": int(oos_returns.size),
            "oos_pnl": float(oos_returns.sum()),
        }
    )
    return report


def main(argv: list[str]) -> None:
    n = int(argv[1]) if len(argv) > 1 else 1_000_000
    n_folds = int(argv[2]) if len(argv) > 2 else 5
    if len(argv) > 3:
        messages, cfg = np.load(argv[3]), Config()
    else:
        messages, cfg = generate(n, seed=11), Config(sample_every_ns=1_000_000)
    print(json.dumps(walk_forward(messages, n_folds, cfg), indent=2))


if __name__ == "__main__":
    main(sys.argv)
