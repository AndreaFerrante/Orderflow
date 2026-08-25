# GC auction-acceptance strategy

This module is a reproducible implementation of an exploratory Gold futures
order-flow setup. It combines auction, footprint and volume-profile context
without using future session information.

## Frozen short specification

The researched candidate is `gc_acceptance_below_short`:

1. Build a completed 70% volume value area for session N-1.
2. During session N, require at least 15% of cumulative traded volume below the
   prior VAL.
3. Segment a trade-price auction while consecutive records remain at the same
   trade price. The auction must contain at least 20 records.
4. Its first price transition must be downward (seller-won auction).
5. Normalized executed imbalance must be at most -0.80:
   `(buy_volume - sell_volume) / (buy_volume + sell_volume)`.
   A magnitude of 0.80 therefore means 90% dominant-side volume, not 80%.
6. The most recently completed one-second footprint must contain at least one
   3:1 sell-side diagonal imbalance and be no more than two seconds old.
7. Enter short on the record after the auction decision boundary. Exit using
   the first executable ask at least one second after entry.

The symmetric long implementation is included for comparison, but the short
side is the main research candidate.

## Install and run

```bash
pip install -e .
orderflow-gc-acceptance /path/to/extracted/sierra/files \
  --output gc_results \
  --side short \
  --tick-size 0.10 \
  --tick-value 10.0 \
  --commission 4.50
```

The input can be one Sierra text file or a directory searched recursively.
Required columns are `Date`, `Time`, `Sequence`, `Price`, `Volume`,
`TradeType`, `AskPrice`, `BidPrice`, `AskSize`, `BidSize`, `TotalAskDepth` and
`TotalBidDepth`. Extra DOM columns are ignored by this strategy.

The runner deliberately processes one file at a time. It writes:

- `trade_parts/*.parquet`: detailed trade ledgers;
- `counts_by_file.csv`: number of trades per source file;
- `summary.csv`: aggregate net performance;
- `manifest.json`: exact parameters, contract economics and library versions.

Existing Parquet parts are never overwritten or mixed into a new run.

## Python API

```python
from orderflow.analysis.strategies import (
    GCAuctionAcceptanceConfig,
    GCContractSpec,
    evaluate_gc_auction_acceptance,
)

trades = evaluate_gc_auction_acceptance(
    ticks,
    config=GCAuctionAcceptanceConfig(),
    contract=GCContractSpec(
        tick_size=0.10,
        tick_value=10.0,
        round_turn_commission=4.50,
    ),
    sides=("short",),
)
```

## Reference reproduction

On the original 52 weekly GC files used during development, the packaged
runner reproduced:

| Side | Trades | Mean net ticks | Median | Capped +/-5 mean |
|---|---:|---:|---:|---:|
| Short | 111 | +2.613 | +1.550 | +1.051 |
| Long | 187 | +0.796 | -0.450 | +0.042 |

These figures are a historical research result, not a guarantee of future
performance. The original sample was explored during strategy development;
independent data and conservative fill stress are required.
