# Design notes and interview prep

This file explains every major decision in lob-backtester and the questions an
interviewer is likely to ask about it. Read it with the code open.

## 1. Data model (`events.py`)

**Decision:** one flat message type (ADD / CANCEL / EXECUTE) in a NumPy
structured array; prices as integer ticks.

- *Why integer ticks?* Floats can't represent most decimal prices exactly
  (0.1 + 0.2 != 0.3). Equality checks on prices ("is this execution at my
  price?") must be exact, so every price is an integer count of ticks.
- *Why a structured array, not objects?* It is one contiguous block of memory,
  can be saved with `np.save`, sliced without copying, and passed straight to
  Numba. A list of Python objects would be ~10x larger and can't be compiled.
- *Why only three message types?* For the book, an execution and a cancel both
  just remove quantity; ITCH's Delete and Replace are expressed as CANCEL (+ ADD).
  Fewer types means fewer code paths to test.

**Likely questions:** What breaks if prices are floats? How would you support a
stock whose tick size changes with price (e.g. sub-penny under $1)?

## 2. Reference order book (`book.py`)

**Decision:** hash map of orders, hash map of price levels per side, and a heap
per side with lazy deletion for the best price.

| Operation | Cost | Why |
|---|---|---|
| add | O(log L) only when a new level appears, else O(1) | heap push per new level |
| cancel / execute | O(1) | dict lookups; emptied levels stay in the heap |
| best bid / ask | amortised O(1) | stale heap tops popped once each |
| depth(k) | O(L log L) | sort on demand; snapshots are rare |

- *Why lazy deletion?* Removing an arbitrary element from a heap is O(L). Instead
  we leave the stale price in the heap and skip it when it reaches the top. Each
  stale entry is popped at most once, so the total cost is the same as if we had
  removed it eagerly.
- *Why push only when a level is created?* So the heap never holds two live
  entries for one price; it stays at most (live levels + stale levels) in size.
- *Validation:* unknown ids, duplicate ids and over-sized cancels raise
  `ValueError`. Real feeds do have gaps, and silent corruption of the book is far
  worse than a loud failure.

**Likely questions:** Why not a sorted list or a balanced tree? (Sorted list
insertion is O(L); a tree gives O(log L) everywhere but Python has none built in,
and a heap is enough because we only ever need the *best* price on the hot path.)
How would you make `depth` fast? (Keep levels in a sorted container, or in an
array indexed by price, as the Numba core does.)

## 3. Fast replay core (`replay_fast.py`)

**Decision:** Numba-compiled loop over dense arrays; ~9M messages/s against
~0.4M/s for the reference (about 20x).

- *Order id remapping.* `np.unique(..., return_inverse=True)` turns arbitrary
  64-bit order ids into slots 0..n-1 once, up front, so per-order state is three
  plain arrays. No hashing inside the loop.
- *Price-indexed levels.* Each side is an `int64` array indexed by
  `price - min_price`. A level lookup is one memory access.
- *Best price.* Improving the best is O(1). When the best level empties we scan
  outward to the next non-empty level. That scan is short in practice because
  liquidity clusters near the touch; worst case it is O(price range).
- *No validation in the fast path.* Validate untrusted data with the reference
  book first; `test_fast_replay_matches_reference` proves the two agree after
  every message.

**Likely questions:** What is the memory cost of the price arrays? (8 bytes x
price range x 2 sides; 1M levels = 16 MB.) What would you do for a huge price
range? (Sparse levels, or a window around the mid that re-centres.) Why is the
first call slow? (JIT compilation; the benchmark warms up first and `cache=True`
stores the compiled code.)

## 4. Backtest engine (`backtest/engine.py`)

**Decision:** event-driven loop that owns all market realism; strategies only
return desired quotes.

1. **Latency.** Decisions take `latency_ns` to reach the exchange; cancels too,
   so an order can be filled after you decided to pull it.
2. **Queue position.** On arrival, `ahead = resting quantity at our price`.
   Executions at our price consume `ahead` before us.
3. **Cancels.** Assumed to come from behind us unless the level shrinks below
   `ahead` (then `ahead` is clamped). This is conservative: real cancels sometimes
   come from in front, which would fill us sooner.
4. **Trade-throughs.** An execution at a price worse than ours on our side means
   the aggressor would have hit us first, so we fill.
5. **No look-ahead.** The strategy sees the book only after each message is
   applied, and its quotes act only on later messages.

Stated limits: our orders don't change the book others see (no market impact,
fine for small size); passive strategies only (no taker orders or taker fees).

**Likely questions:** Why is the cancel assumption conservative, and how would
you estimate the real fraction of cancels ahead of you? (Fit it from the data:
compare where cancelled orders sat in the queue.) What happens to queue position
when you change your quote? (You go to the back of the new level; unchanged
quotes keep their place, which is why the engine only re-sends changed quotes.)

## 5. Strategy (`strategies/market_maker.py`)

Quotes at the touch, skews both prices away from inventory, and stops quoting
the side that would breach `max_position`. Linear skew is the simplest version
of what Avellaneda-Stoikov derive formally from volatility and risk aversion.

**Likely questions:** What is adverse selection? (You get filled more often
right before the price moves against you; informed traders pick off stale
quotes.) How would you detect it? (Mark-out: PnL of each fill measured 1s, 10s,
60s later.)

## 6. Statistics (`stats.py`) and walk-forward (`walkforward.py`)

- **Moving-block bootstrap** for the Sharpe confidence interval, because PnL
  changes are autocorrelated (inventory carries over). Resampling single points
  would understate uncertainty; `test_block_bootstrap_is_wider_for_autocorrelated_returns`
  demonstrates this.
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014): the probability the
  true Sharpe beats the best of `n_trials` noise strategies, adjusted for skew and
  fat tails. With one trial it is the Probabilistic Sharpe Ratio.
- **Walk-forward:** pick parameters on fold k-1, report fold k only. Each fold
  starts from a book snapshot, because later folds cancel earlier orders.

**Result on synthetic data:** out-of-sample PnL is about 0 and the deflated
Sharpe is 0.07, i.e. no edge. That is the correct answer: the synthetic market
has no signal, and at a price of 100 the 1-tick spread earned per share exactly
equals the 1 bp fee. A framework that showed profit here would be broken.

**Likely questions:** Why not annualise with sqrt(252)? (Returns here are per
sampling interval, so annualise with the number of intervals in a trading
year.) What else inflates backtest Sharpe? (Look-ahead, survivorship bias,
ignoring fees and latency, optimistic fill assumptions, multiple testing.)

## 7. ITCH parser (`loaders/itch.py`)

Streams the 2-byte-length-framed binary file, keeps a table of live orders for
the chosen stock (Delete and Replace don't carry quantities), and maps Replace
to cancel + add, which loses queue priority exactly as it does on NASDAQ.
Executions are recorded at the order's resting price so the book stays
consistent even when a `C` message prints at a different price.

Pure Python parses ~460K messages/s; a full NASDAQ day (hundreds of millions of
messages across all stocks) takes minutes. Next step: a vectorised or Numba
parser that scans the length prefixes in bulk.
