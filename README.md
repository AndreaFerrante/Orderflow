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

## Behaviour contracts

- `BacktestEngine.run` pairs signal sides with entry ticks by position and raises `ValueError` on
  duplicate signal `Index`, a signal `Index` missing from `data`, signals out of tick order, or duplicate
  `data` `Index` — each used to shift every later trade onto the wrong side without an error.
- `get_daily_session_moving_POC`: `Prev_POC` is the POC of the session that just ended, whatever the gap
  (Sunday's session carries Friday's); 0 only on the first session of the frame.
- `get_volume_profile_areas` (`VA_Areas`): the first tick of the frame, and the first tick after each
  RTH→ETH reset, seed the POC; value-area expansion steps over zero-volume price levels. Labels are
  `PO` / `VA` / `na` (`PO` is `POC` truncated by the 2-character label array).
