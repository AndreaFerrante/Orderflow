from typing import Literal, Optional
import polars as pl


N_CONSECUTIVE_DEFAULT: int = 3
VOLUME_THRESHOLD_DEFAULT: int = 20
BUY_CODE_DEFAULT: int = 2
SELL_CODE_DEFAULT: int = 1
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


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Use get_tickers_in_folder function
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def load_tick_data(path: str, separator: str = ";", ensure_types: bool = True) -> pl.DataFrame:

    """
    Load L2 ticks and build a 'timestamp' column. Returns a DataFrame sorted by time.
    Required columns: Date, Time, BidPrice, AskPrice, Volume, TradeType
    """

    df       = pl.read_csv(path, separator=separator)
    required = {"Date", "Time", "BidPrice", "AskPrice", "Volume", "TradeType"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

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
        .alias("Datetime")
    )

    return df.sort("Datetime")


def aggregate_auctions(
    df: pl.DataFrame = None,
    buy_code: int = BUY_CODE_DEFAULT,
    sell_code: int = SELL_CODE_DEFAULT,
    eps: float = EPS_DEFAULT,
    imbalance_mode: Literal["ratio", "bounded"] = "ratio",
    segmentation: Literal["quote_any", "quote_both", "mid_change"] = "quote_any",
    time_cap_ms: Optional[int] = None,
) -> pl.DataFrame:
    
    """
    Segment rows into auctions (stable quote regime) and aggregate per-auction stats.
    """
    
    if df is None:
        raise Exception("Pass a dataframe parameter to the function in Polars DataFrame format.")
        return
    
    bid, ask = df["BidPrice"], df["AskPrice"]

    match segmentation:
        case "quote_any":
            change = (bid != bid.shift(1)) | (ask != ask.shift(1)) # Either bid OR ask price changed from previous tick
        case "quote_both":
            change = (bid != bid.shift(1)) & (ask != ask.shift(1)) # Both bid AND ask prices changed from previous tick
        case "mid_change":
            mid_price = (bid + ask) / 2.0
            change = mid_price != mid_price.shift(1) # Mid-point price changed from previous tick
        case _:
            raise ValueError(f"Unknown segmentation type: '{segmentation}'. Valid options are: 'quote_any', 'quote_both', 'mid_change'")

    if time_cap_ms is not None:
        gap_ms = (pl.col("Datetime").cast(pl.Int64) - pl.col("Datetime").shift(1).cast(pl.Int64)) // 1_000_000
        change = change | (gap_ms >= pl.lit(time_cap_ms))

    # Adding auction id . . .
    df = (
            df.
            with_columns(change.fill_null(True).cast(pl.Int64).alias("AskBidSpread")).
            with_columns(pl.col("AskBidSpread").cum_sum().alias("AuctionId"))
        )

    agg = (
            df.
            group_by("AuctionId").
            agg([
                pl.col("Datetime").min().alias("StartTime"),
                pl.col("Datetime").max().alias("EndTime"),
                pl.col("BidPrice").first().alias("FirstBidprice"),
                pl.col("AskPrice").first().alias("FirstAskPrice"),
                pl.col("BidPrice").last().alias("LastBidPrice"),
                pl.col("AskPrice").last().alias("LastAskPrice"),
                pl.len().alias("NumTrades"),
                pl.sum("Volume").alias("TotalVolumeOnSpread"),
                pl.col("Volume").filter(pl.col("TradeType") == buy_code).sum().fill_null(0).alias("BuyVolume"),
                pl.col("Volume").filter(pl.col("TradeType") == sell_code).sum().fill_null(0).alias("SellVolume"),
            ])
        )

    agg = agg.with_columns((pl.col("BuyVolume") - pl.col("SellVolume")).alias("Delta"))


    # Imbalance definition controlled by `imbalance_mode`:
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
            pl.
            when(pl.col("BuyVolume") > pl.col("SellVolume")).
                then(pl.col("BuyVolume") / (pl.col("SellVolume") + pl.lit(eps))).
            when(pl.col("BuyVolume") < pl.col("SellVolume")).
                then(-pl.col("SellVolume") / (pl.col("BuyVolume") + pl.lit(eps))).
            otherwise(0.0)
        )
    else:
        imbalance = (
            (pl.col("BuyVolume") - pl.col("SellVolume")) / (pl.col("BuyVolume") + pl.col("SellVolume") + pl.lit(eps))
        )

    label = (
        pl.
            when((pl.col("BuyVolume") > 0) & (pl.col("SellVolume") == 0)).
                then(pl.lit("buy_only")).
            when((pl.col("SellVolume") > 0) & (pl.col("BuyVolume") == 0)).
                then(pl.lit("sell_only")).
            when((pl.col("SellVolume") > 0) & (pl.col("BuyVolume") > 0)).
                then(pl.lit("both")).
        otherwise(pl.lit("none"))
    )
    
    agg = agg.with_columns([imbalance.alias("Imbalance"), label.alias("Label")])
    agg = agg.sort(['StartTime'], descending=False)

    return agg


def identify_valid_blocks(
    agg: pl.DataFrame = None,
    n_consecutive: int = N_CONSECUTIVE_DEFAULT,
    vol_thresh: int = VOLUME_THRESHOLD_DEFAULT,
    require_nonzero_imbalance: bool = True,
    return_ids_list: bool = False,
) -> pl.DataFrame:
    
    """
    Find runs of exactly n_consecutive auctions with same imbalance sign and volume/label filters.
    """
    
    if agg is None:
        raise Exception(f"Pass as paramter to the function e not None Polars DataFrame.")
        return

    base = (
        agg
        #.filter((pl.col("Label") == "both") & (pl.col("TotalVolumeOnSpread") > vol_thresh))
        .filter(pl.col("TotalVolumeOnSpread") > vol_thresh)
        .sort("AuctionId")
        .with_row_index("RowIdx")
        .with_columns([
            pl.when(pl.col("Imbalance") > 0).
                then(1).
            when(pl.col("Imbalance") < 0).
                then(-1).
            otherwise(0).
            alias("ImbalanceDirection")
        ])
    )

    if require_nonzero_imbalance:
        base = base.filter(pl.col("ImbalanceDirection") != 0)

    # Use the sign of the chosen 'Imbalance' to classify side for runs
    base = base.with_columns(
        (((pl.col("AuctionId") - pl.col("AuctionId").shift(1)) != 1) | (pl.col("ImbalanceDirection") != pl.col("ImbalanceDirection").shift(1))).
            fill_null(True).
            cast(pl.Int32).
            alias("StreakBreak")
    )
    
    base = base.with_columns(pl.col("StreakBreak").cum_sum().alias("StreakId"))
    base = base.with_columns([
        (pl.col("RowIdx") - pl.col("RowIdx").first().over("StreakId")).alias("Position"),
         pl.col("TotalVolumeOnSpread").cum_sum().over("StreakId").alias("StreakCumVol"),
         pl.col("Imbalance").cum_sum().over("StreakId").alias("StreakCumImb"),
    ])

    ends = (
        base
        .filter(pl.col("Position") >= (n_consecutive - 1))
        .with_columns([
            (pl.col("RowIdx") - (n_consecutive - 1)).alias("StartRowIdx"),
            pl.when(pl.col("StreakCumVol").shift(n_consecutive).over("StreakId").is_null())
              .then(pl.col("StreakCumVol"))
              .otherwise(pl.col("StreakCumVol") - pl.col("StreakCumVol").shift(n_consecutive).over("StreakId")).alias("BlockVolume"),
            pl.when(pl.col("StreakCumImb").shift(n_consecutive).over("StreakId").is_null())
              .then(pl.col("StreakCumImb") / pl.lit(n_consecutive))
              .otherwise((pl.col("StreakCumImb") - pl.col("StreakCumImb").shift(n_consecutive).over("StreakId")) / pl.lit(n_consecutive)).alias("BlockImbalance"),
        ])
    )

    starts = base.select(["StreakId", "RowIdx", pl.col("AuctionId").alias("StartTimeStartId"), "StartTime"])
    spans = (
        ends.join(starts, left_on=["StreakId", "StartRowIdx"], right_on=["StreakId", "RowIdx"], how="inner")
            .select([
                pl.col("StartTimeStartId"),
                pl.col("AuctionId").alias("AuctionIdEndId"),
                pl.col("StartTime"),
                pl.col("EndTime"),
                pl.col("BlockVolume").alias("TotalBlockVolume"),
                pl.col("BlockImbalance").alias("TotalBlockImbalance"),
            ])
    )

    if not return_ids_list:
        return spans.with_row_index("block_id")

    # Optional: materialize full auction id list per block
    id_map = base.select(["StreakId", "RowIdx", pl.col("auction_id").alias("aid")])
    ranges = (
        ends.join(starts, left_on=["StreakId", "StartRowIdx"], right_on=["StreakId", "RowIdx"], how="inner")
            .with_columns(pl.int_ranges(pl.col("StartRowIdx"), pl.col("RowIdx") + 1).alias("row_range"))
    )
    exploded = ranges.explode("row_range").join(id_map, left_on=["StreakId", "row_range"], right_on=["StreakId", "RowIdx"], how="inner")

    return (
        exploded.group_by(["StreakId", "RowIdx"]).agg([
            pl.col("aid").alias("auction_ids"),
            pl.first("start").alias("start"),
            pl.first("end").alias("end"),
            pl.first("block_volume").alias("total_volume"),
            pl.first("block_imbalance").alias("imbalance"),
        ]).with_row_index("block_id").select(["block_id", "auction_ids", "start", "end", "total_volume", "imbalance"]) 
    )


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
