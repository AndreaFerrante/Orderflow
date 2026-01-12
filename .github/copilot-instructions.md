<!-- Copilot instructions for the Orderflow project -->
# Orderflow — AI coding agent instructions

Purpose: help a code-writing agent become productive quickly in this repo (tick-level market-data research tools).

1) Big picture
- The package lives in the orderflow folder and focuses on tick-by-tick analytics: auction segmentation, volume profile, VWAP, footprints and micro-backtests.
- Core modules: [orderflow/_volume_factory.py](orderflow/_volume_factory.py) (I/O & time handling), [orderflow/auctions.py](orderflow/auctions.py) (auction segmentation + blocks + forward outcomes), [orderflow/volume_profile.py](orderflow/volume_profile.py) and [orderflow/volume_profile_kde.py](orderflow/volume_profile_kde.py) (KDE & peaks), plus small utilities like [orderflow/vwap.py](orderflow/vwap.py) and [orderflow/backtester.py](orderflow/backtester.py).

2) Developer workflow & quick commands
- Install deps: pip install -r requirements.txt from the project root.
- Run an example script: python -m orderflow.runners.runner (or run orderflow/runners/runner.py directly). Example data lives under data/tbt/ (e.g. data/tbt/2023_06_29.txt).
- There are no formal tests in the repo. Use small slices of data in data/tbt for manual validation and plotting.

3) Project-specific conventions and patterns
- Data files are semicolon-delimited CSVs with columns: Date, Time, Price, Volume, TradeType, AskPrice, BidPrice, AskSize, BidSize, etc. See README.md for full schema.
- Most functions expect Polars DataFrames (`polars.DataFrame`). Prefer Polars for performance and for asof joins (many functions rely on sorted, typed columns).
- Time handling: many modules assume `Datetime` column (polars.Datetime) and that rows are sorted. Use functions in [orderflow/_volume_factory.py](orderflow/_volume_factory.py) to correct nanoseconds and to apply timezone offsets (`apply_offset_given_dataframe`).
- Auction/block functions assume specific column names: `AuctionId`, `StartTime`, `EndTime`, `BuyVolume`, `SellVolume`, `Imbalance`. Blocks use `join_asof` and depend on strictly increasing `Datetime`.

4) Integration & performance notes
- `volume_profile_kde.py` uses numba and parallel kernels. Numba must be available; expect heavy CPU usage for large price series — test on small dataset first.
- Polars `join_asof` is used extensively for time-aligned lookups — ensure the right sort order and matching key names when composing joins.

5) Common pitfalls for contributors
- Forgetting to convert to polars or ensuring `Datetime` dtype causes silent failures or exceptions. Verify `Datetime` exists and is sorted before calling `aggregate_auctions`, `get_valid_blocks`, or `compute_forward_outcomes`.
- Timezone offsets are non-trivial: use `apply_offset_given_dataframe` instead of ad-hoc shifts.
- KDE/numba tuning parameter `h` (bandwidth) appears in example runner calls — keep numeric defaults small when debugging.

6) Useful files to inspect when implementing features
- [orderflow/_volume_factory.py](orderflow/_volume_factory.py) — I/O, parsing, timezone helpers
- [orderflow/auctions.py](orderflow/auctions.py) — auction segmentation, `aggregate_auctions`, `get_valid_blocks`, `compute_forward_outcomes`
- [orderflow/volume_profile_kde.py](orderflow/volume_profile_kde.py) — KDE and peak detection (numba)
- [orderflow/runners/runner.py](orderflow/runners/runner.py) — end-to-end example showing common pipelines and plotting

If any area above is unclear or you want the instructions to include additional workflow commands (CI, packaging, or test examples), tell me which part to expand.
