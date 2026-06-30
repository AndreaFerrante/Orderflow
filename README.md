# Orderflow

Orderflow is a Python toolkit for market microstructure research, tick-data reshaping, backtesting, and PostgreSQL storage.

## Package Layout

- `orderflow.core` for shared config, exceptions, and paths
- `orderflow.data` for ingestion and bar compression
- `orderflow.market` for auctions, DOM, profiles, and market utilities
- `orderflow.analysis` for statistics, regimes, and simulation
- `orderflow.backtesting` for the backtest engine and execution models
- `orderflow.storage` for database loaders and CLI entry points
- `orderflow.visualization` for plotting helpers
