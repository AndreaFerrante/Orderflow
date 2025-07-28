"""
orderflow_core.py — minimal, fast core (Pure Polars)

Public API:
    load_tick_data
    aggregate_auctions
    identify_valid_blocks
    compute_forward_outcomes

Notes
-----
• Segmentation choices: 'quote_any' | 'quote_both' | 'mid_change'
• Imbalance choices: 'ratio' (piecewise buy/sell) | 'bounded' ((b-s)/(b+s))
• Block detection is O(n) using streak segmentation + prefix sums.
"""
from __future__ import annotations

from typing import Literal, Optional
import polars as pl

# Defaults (override per call as needed)
N_CONSECUTIVE_DEFAULT: int = 3
VOLUME_THRESHOLD_DEFAULT: int = 20
BUY_CODE_DEFAULT: int = 1
SELL_CODE_DEFAULT: int = 2
EPS_DEFAULT: float = 1e-6

__all__ = [
    "load_tick_data",
    "aggregate_auctions",
    "identify_valid_blocks",
    "compute_forward_outcomes",
    "N_CONSECUTIVE_DEFAULT",
    "VOLUME_THRESHOLD_DEFAULT",
    "BUY_CODE_DEFAULT",
    "SELL_CODE_DEFAULT",
]

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_tick_data(path: str, separator: str = ";", ensure_types: bool = True) -> pl.DataFrame:
    """Load L2 ticks and build a 'timestamp' column. Returns a DataFrame sorted by time.

    Required columns: Date, Time, BidPrice, AskPrice, Volume, TradeType
    """
    df = pl.read_csv(path, separator=separator)
    req = ["Date", "Time", "BidPrice", "AskPrice", "Volume", "TradeType"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if ensure_types:
        df = df.with_columns([
            pl.col("BidPrice").cast(pl.Float64),
            pl.col("AskPrice").cast(pl.Float64),
            pl.col("Volume").cast(pl.Float64),
            pl.col("TradeType").cast(pl.Int64),
        ])

    df = df.with_columns(
        (pl.col("Date") + pl.lit(" ") + pl.col("Time"))
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
        .alias("timestamp")
    )

    return df.sort("timestamp")

# -----------------------------------------------------------------------------
# Auction aggregation
# -----------------------------------------------------------------------------

def aggregate_auctions(
    df: pl.DataFrame,
    *,
    buy_code: int = BUY_CODE_DEFAULT,
    sell_code: int = SELL_CODE_DEFAULT,
    eps: float = EPS_DEFAULT,
    imbalance_mode: Literal["ratio", "bounded"] = "ratio",
    segmentation: Literal["quote_any", "quote_both", "mid_change"] = "quote_any",
    time_cap_ms: Optional[int] = None,
) -> pl.DataFrame:
    """Segment rows into auctions (stable quote regime) and aggregate per-auction stats."""
    bid, ask = pl.col("BidPrice"), pl.col("AskPrice")

    if segmentation == "quote_any":
        change = (bid != bid.shift(1)) | (ask != ask.shift(1))
    elif segmentation == "quote_both":
        change = (bid != bid.shift(1)) & (ask != ask.shift(1))
    elif segmentation == "mid_change":
        change = ((bid + ask) / 2.0) != ((bid + ask) / 2.0).shift(1)
    else:
        raise ValueError("Unknown segmentation")

    if time_cap_ms is not None:
        gap_ms = (
            pl.col("timestamp").cast(pl.Int64) - pl.col("timestamp").shift(1).cast(pl.Int64)
        ) // 1_000_000
        change = change | (gap_ms >= pl.lit(time_cap_ms))

    df = df.with_columns(change.fill_null(True).cast(pl.Int64).alias("new_flag"))
    df = df.with_columns(pl.col("new_flag").cum_sum().alias("auction_id"))

    agg = df.group_by("auction_id").agg([
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        bid.first().alias("bid"),
        ask.first().alias("ask"),
        pl.len().alias("n_trades"),
        pl.sum("Volume").alias("total_volume"),
        pl.col("Volume").filter(pl.col("TradeType") == buy_code).sum().fill_null(0).alias("buy_volume"),
        pl.col("Volume").filter(pl.col("TradeType") == sell_code).sum().fill_null(0).alias("sell_volume"),
    ])

    agg = agg.with_columns((pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"))

    # ---------------------------------------------------------------------
    # Imbalance definition
    # ---------------------------------------------------------------------
    # Two options controlled by `imbalance_mode`:
    #
    # 1) "ratio"  (signed dominance ratio; **unbounded**)
    #    • if buy > sell:  + buy / (sell + eps)
    #    • if sell > buy:  - sell / (buy + eps)
    #    • else:           0
    #    Emphasizes dominance; e.g. buy=120,sell=30 → +4.0. If the opposite
    #    side is ~0, magnitude can be very large (eps avoids div/0).
    #
    # 2) "bounded" (signed normalized difference; in ~[-1, +1])
    #       (buy - sell) / (buy + sell + eps)
    #    Interpretable as fraction of net pressure: +1 all buy, -1 all sell.
    #
    # We also provide `delta = buy - sell` as raw, unnormalized imbalance.
    # The block detector below uses only the **sign** of `imbalance`.
    if imbalance_mode == "ratio":
        imbalance = (
            pl.when(pl.col("buy_volume") > pl.col("sell_volume"))
            .then(pl.col("buy_volume") / (pl.col("sell_volume") + pl.lit(eps)))
            .when(pl.col("sell_volume") > pl.col("buy_volume"))
            .then(-pl.col("sell_volume") / (pl.col("buy_volume") + pl.lit(eps)))
            .otherwise(0.0)
        )
    else:  # bounded
        imbalance = (
            (pl.col("buy_volume") - pl.col("sell_volume"))
            / (pl.col("buy_volume") + pl.col("sell_volume") + pl.lit(eps))
        )

    label = (
        pl.when((pl.col("buy_volume") > 0) & (pl.col("sell_volume") == 0)).then(pl.lit("buy_only"))
        .when((pl.col("sell_volume") > 0) & (pl.col("buy_volume") == 0)).then(pl.lit("sell_only"))
        .when((pl.col("sell_volume") > 0) & (pl.col("buy_volume") > 0)).then(pl.lit("both"))
        .otherwise(pl.lit("none"))
    )

    return agg.with_columns([imbalance.alias("imbalance"), label.alias("label")])

# -----------------------------------------------------------------------------
# O(n) block detection (no list slicing/exploding)
# -----------------------------------------------------------------------------

def identify_valid_blocks(
    agg: pl.DataFrame,
    *,
    n_consecutive: int = N_CONSECUTIVE_DEFAULT,
    vol_thresh: int = VOLUME_THRESHOLD_DEFAULT,
    require_nonzero_imbalance: bool = True,
    return_ids_list: bool = False,
) -> pl.DataFrame:
    """Find runs of exactly n_consecutive auctions with same imbalance sign and volume/label filters."""
    base = (
        agg
        .filter((pl.col("label") == "both") & (pl.col("total_volume") > vol_thresh))
        .sort("auction_id")
        .with_row_index("row_idx")
        .with_columns([
            pl.when(pl.col("imbalance") > 0).then(1).when(pl.col("imbalance") < 0).then(-1).otherwise(0).alias("imb_sign")
        ])
    )

    if require_nonzero_imbalance:
        base = base.filter(pl.col("imb_sign") != 0)

    # Use the sign of the chosen `imbalance` to classify side for runs
    base = base.with_columns(
        (((pl.col("auction_id") - pl.col("auction_id").shift(1)) != 1) | (pl.col("imb_sign") != pl.col("imb_sign").shift(1)))
        .fill_null(True).cast(pl.Int32).alias("streak_break")
    )
    base = base.with_columns(pl.col("streak_break").cum_sum().alias("streak_id"))

    base = base.with_columns([
        (pl.col("row_idx") - pl.col("row_idx").first().over("streak_id")).alias("pos"),
        pl.col("total_volume").cum_sum().over("streak_id").alias("cum_vol"),
        pl.col("imbalance").cum_sum().over("streak_id").alias("cum_imb"),
    ])

    n = n_consecutive
    ends = (
        base.filter(pl.col("pos") >= (n - 1))
        .with_columns([
            (pl.col("row_idx") - (n - 1)).alias("start_row_idx"),
            pl.when(pl.col("cum_vol").shift(n).over("streak_id").is_null())
              .then(pl.col("cum_vol"))
              .otherwise(pl.col("cum_vol") - pl.col("cum_vol").shift(n).over("streak_id")).alias("block_volume"),
            pl.when(pl.col("cum_imb").shift(n).over("streak_id").is_null())
              .then(pl.col("cum_imb")/pl.lit(n))
              .otherwise((pl.col("cum_imb") - pl.col("cum_imb").shift(n).over("streak_id"))/pl.lit(n)).alias("block_imbalance"),
        ])
    )

    starts = base.select(["streak_id", "row_idx", pl.col("auction_id").alias("start_id"), pl.col("start").alias("start_ts")])
    spans = (
        ends.join(starts, left_on=["streak_id", "start_row_idx"], right_on=["streak_id", "row_idx"], how="inner")
            .select([
                pl.col("start_id"),
                pl.col("auction_id").alias("end_id"),
                pl.col("start_ts").alias("start"),
                pl.col("end").alias("end"),
                pl.col("block_volume").alias("total_volume"),
                pl.col("block_imbalance").alias("imbalance"),
            ])
    )

    if not return_ids_list:
        return spans.with_row_index("block_id")

    # Optional: materialize full auction id list per block
    id_map = base.select(["streak_id", "row_idx", pl.col("auction_id").alias("aid")])
    ranges = (
        ends.join(starts, left_on=["streak_id", "start_row_idx"], right_on=["streak_id", "row_idx"], how="inner")
            .with_columns(pl.int_ranges(pl.col("start_row_idx"), pl.col("row_idx") + 1).alias("row_range"))
    )
    exploded = ranges.explode("row_range").join(id_map, left_on=["streak_id", "row_range"], right_on=["streak_id", "row_idx"], how="inner")

    return (
        exploded.group_by(["streak_id", "row_idx"]).agg([
            pl.col("aid").alias("auction_ids"),
            pl.first("start").alias("start"),
            pl.first("end").alias("end"),
            pl.first("block_volume").alias("total_volume"),
            pl.first("block_imbalance").alias("imbalance"),
        ]).with_row_index("block_id").select(["block_id", "auction_ids", "start", "end", "total_volume", "imbalance"]) 
    )

# -----------------------------------------------------------------------------
# Forward outcomes (entry/exit prices & returns)
# -----------------------------------------------------------------------------

def compute_forward_outcomes(
    df_ticks: pl.DataFrame,
    blocks: pl.DataFrame,
    *,
    minutes_ahead: int = 5,
    price_source: Literal["mid","trade","bid","ask"] = "mid",
) -> pl.DataFrame:
    """Per block: get price at entry=end, exit=end+minutes, and simple return.

    NOTE: Uses **left=blocks .join_asof(right=price)** so that for each target
    timestamp (entry/exit) we pick the last price with `timestamp <= target`.
    This avoids the nulls you observed when the target lies **after** the last
    tick in the window.
    """
    # Build price series (sorted)
    if price_source == "mid":
        price = df_ticks.select([
            pl.col("timestamp"),
            ((pl.col("BidPrice") + pl.col("AskPrice")) / 2).alias("price"),
        ])
    elif price_source == "trade" and "Price" in df_ticks.columns:
        price = df_ticks.select([pl.col("timestamp"), pl.col("Price").alias("price")])
    elif price_source == "bid":
        price = df_ticks.select([pl.col("timestamp"), pl.col("BidPrice").alias("price")])
    elif price_source == "ask":
        price = df_ticks.select([pl.col("timestamp"), pl.col("AskPrice").alias("price")])
    else:
        price = df_ticks.select([
            pl.col("timestamp"),
            ((pl.col("BidPrice") + pl.col("AskPrice")) / 2).alias("price"),
        ])
    price = price.sort("timestamp")

    # Entry/exit timestamps per block
    b = (
        blocks
        .with_columns([
            pl.col("end").alias("entry_ts"),
            (pl.col("end") + pl.duration(minutes=minutes_ahead)).alias("exit_ts"),
        ])
        .select(["block_id", "start_id", "end_id", "entry_ts", "exit_ts"])  # lean
        .sort("entry_ts")
    )

    # For each entry_ts, take last price with timestamp <= entry_ts
    entry = (
        b.select(["block_id", "entry_ts"]).sort("entry_ts")
         .join_asof(price, left_on="entry_ts", right_on="timestamp", strategy="backward")
         .rename({"price": "entry_price"})
         .select(["block_id", "entry_ts", "entry_price"])
    )

    # For each exit_ts, take last price with timestamp <= exit_ts
    exit_ = (
        b.select(["block_id", "exit_ts"]).sort("exit_ts")
         .join_asof(price, left_on="exit_ts", right_on="timestamp", strategy="backward")
         .rename({"price": "exit_price"})
         .select(["block_id", "exit_ts", "exit_price"])
    )

    # Combine and compute simple return
    res = (
        b.join(entry, on=["block_id", "entry_ts"], how="left")
         .join(exit_, on=["block_id", "exit_ts"], how="left")
         .with_columns(
             ((pl.col("exit_price") - pl.col("entry_price")) / pl.col("entry_price")).alias("simple_return")
         )
         .select([
             "block_id", "start_id", "end_id",
             "entry_ts", "exit_ts",
             "entry_price", "exit_price", "simple_return",
         ])
    )

    return res
