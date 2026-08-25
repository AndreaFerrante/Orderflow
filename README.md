# Orderflow

Orderflow is a serious Python toolkit for market microstructure research, tick-data reshaping, backtesting, and PostgreSQL storage.

## Package Layout

- `orderflow.core` for shared config, exceptions, and paths
- `orderflow.data` for ingestion and bar compression
- `orderflow.market` for auctions, DOM, profiles, and market utilities
- `orderflow.analysis` for statistics, regimes, and simulation
- `orderflow.backtesting` for the backtest engine and execution models
- `orderflow.storage` for database loaders and CLI entry points
- `orderflow.visualization` for plotting helpers

## GC auction-acceptance research strategy

The repository includes a causal, bounded-memory Polars + Numba implementation
of a Gold futures setup combining prior-session volume value, developing volume
outside value, same-price auctions, executed imbalance and completed diagonal
footprints. See [docs/GC_AUCTION_ACCEPTANCE.md](docs/GC_AUCTION_ACCEPTANCE.md)
for the frozen definition, executable-quote backtest and multi-file runner.
