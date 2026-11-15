# Orderflow

Orderflow is a serious Python toolkit for market microstructure research, tick-data reshaping, backtesting, and PostgreSQL storage.


## Package Layout

- `orderflow.core` for shared config, exceptions, and paths
- `orderflow.data` for ingestion and bar compression
- `orderflow.market` for auctions, DOM, profiles, and market utilities
- `orderflow.analysis` for statistics, regimes, and simulation
- `orderflow.backtester` for the backtest engine (`BacktestEngine`), exits and execution models (`orderflow.backtesting` is a legacy re-export)
- `orderflow.storage` for database loaders and CLI entry points
- `orderflow.visualization` for plotting helpers

## Install

From the parent repo: `pip install --no-deps -e vendor/Orderflow` (editable — a plain install copies the
package into site-packages and your edits here stop reaching the runners). Tests: `pytest tests/`.

## Backtest reliability: walk-forward and Monte Carlo

Both read finished blotters (`trades_df`). Neither re-runs the engine.

### Monte Carlo: how bad could it have been?

Reshuffles the trading days of one blotter, 10,000 times.

```python
result = engine.run(data, signals, tp_ticks=12, sl_ticks=8)
mc = result.monte_carlo()                 # 10,000 paths, seed 42

mc.p_loss                                 # share of paths that end at or below zero
mc.paths["max_drawdown"].quantile(0.95)   # drawdown to size the account on
mc.paths["net_pnl"].quantile(0.05)        # a bad outcome
mc.observed_net_pnl                       # the real history
mc.observed_max_drawdown

mc.plot().savefig("monte_carlo.png")      # a notebook shows the figure by itself
```

Any blotter works: `monte_carlo(trades_df, n_resamples=10_000, seed=42)`.

The figure has two charts. Left: 200 simulated equity paths, the band that holds 90% of all paths, their
median, the observed path. Right: max drawdown of all paths, with the observed value and the 95% value.

| Read | Good | Bad |
|---|---|---|
| `p_loss` | near 0 | above about 0.2: the profit may be luck |
| `observed_max_drawdown` against the 95% value | close to it | far below it: the real order of days was kind, expect worse live |
| observed path on the chart | inside the band, near the median | riding the top of the band: a lucky history |

### Walk-forward: do the picked parameters keep earning?

Picks the best parameter set on a train window, keeps its trades of the next test window, steps forward.

```python
from orderflow.backtester import walk_forward

blotters = {   # one full-history run per parameter set
    (tp, sl): engine.run(data, signals, tp_ticks=tp, sl_ticks=sl).trades_df
    for tp, sl in [(8, 8), (12, 8), (20, 10)]
}
wf = walk_forward(blotters, train_days=60, test_days=20, dates=data["Date"].unique())

wf.efficiency   # out-of-sample P&L per day / in-sample P&L per day
wf.folds        # per fold: windows, chosen set, train score and P&L, test trades and P&L
wf.oos_trades   # the chosen sets' test-window trades: the out-of-sample blotter
```

Options: `anchored=True` (the train window grows from the first day), `min_train_trades=30` (fewer trades
in a train window and the set cannot be chosen), `score=trade_sharpe` (any function of a P&L array).

| Read | Good | Bad |
|---|---|---|
| `efficiency` | near 1 | below about 0.5: fitted to noise. Negative: the picks lose. `None`: no fold traded, or the picks lost in sample |
| `folds["chosen"]` | the same set, or neighbours, fold after fold | a new winner every fold. `None`: no set was eligible |
| `folds["test_net_pnl"]` | profit in most folds | all the profit from one fold |

### Both together

Walk-forward first, then Monte Carlo on what it kept. Judge the strategy on this.

```python
from orderflow.backtester import monte_carlo

mc = monte_carlo(wf.oos_trades)
mc.plot().savefig("oos_monte_carlo.png")
```

Monte Carlo on an in-sample blotter flatters an overfit.

### Limits

- 0.2 and 0.5 are rules of thumb, not laws.
- `walk_forward` picks only among the sets it is given and does not correct for how many were tried.
- `monte_carlo` treats days as independent: streaks and regimes are lost.
- `monte_carlo` needs many trading days: three days give only 27 mixes.
- The chart is day-end equity: a fall inside a day counts in `max_drawdown` but is not drawn.
- A blotter with no trades raises `ValueError("no trades to resample")`.

## Behaviour contracts

- `BacktestEngine.run` pairs signal sides with entry ticks by position and raises `ValueError` on
  duplicate signal `Index`, a signal `Index` missing from `data`, signals out of tick order, or duplicate
  `data` `Index` — each used to shift every later trade onto the wrong side without an error.
- The per-tick session functions of `volume_profile` / `vwap`, the bar-id assigners of `compressor` and
  `find_stacked_imbalances` run Numba kernels and keep their original Python loop beside them (`_<name>_py`,
  `_compute_*_bar_ids_py`, `_detect`) as the reference. Kernel and loop must stay bit-identical:
  `tests/test_session_kernels.py`, `tests/test_bar_id_kernels.py` and `tests/test_footprint_kernel.py` compare
  them on random tapes, so a behaviour change goes into both. A kernel only takes the input it is known to
  reproduce (the session functions: a pandas frame with a float64 `Price`, an int64 or float64 `Volume` and
  string `SessionType` labels); any other input runs the loop and returns, raises, prints and warns as it
  always did, in minutes instead of seconds on a full file. A kernel runs on a thread of its own, compiled
  `nogil` (`orderflow.core._kernel_calls.run_kernel`), while the caller's thread moves the loop's `tqdm` bar
  and stays free for signals: Ctrl-C raises `KeyboardInterrupt` within milliseconds and the kernel stops.
- `get_daily_session_moving_POC`: `Prev_POC` is the POC of the session that just ended, whatever the gap
  (Sunday's session carries Friday's); 0 only on the first session of the frame.
- `get_volume_profile_areas` (`VA_Areas`): the first tick of the frame, and the first tick after each
  RTH→ETH reset, seed the POC; value-area expansion steps over zero-volume price levels. Labels are
  `PO` / `VA` / `na` (`PO` is `POC` truncated by the 2-character label array).
- `walk_forward` reads finished blotters (one `trades_df` per parameter set, each from one full-history
  run) and never re-runs the engine. Windows are counted in trading days and test windows tile. A trade
  belongs to the day of its `entry_datetime`, so the stitched out-of-sample blotter is exact only when
  every candidate is flat at each day boundary. Pass `dates` (the tick frame's `Date` values) or days on
  which nothing traded are not counted.
- `monte_carlo` resamples whole trading days of a blotter with replacement (days without trades are not
  in the pool) and keeps each day's trades in entry order. Its totals match
  `stats.bootstrap_total_by_group` for the same seed, to float rounding. Its drawdown counts the zero
  start as a peak, so a loss on the first trade is a drawdown; `stats.max_drawdown_absolute` does not
  count that one.
