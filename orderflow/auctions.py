from typing import Literal, Optional, Sequence
import polars.selectors as cs
import polars as pl


N_CONSECUTIVE_DEFAULT: int = 3
VOLUME_THRESHOLD_DEFAULT: int = 1000  # This is the total volume on a single spread !
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


def get_valid_blocks(
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

    starts = base.select(["StreakId", "RowIdx", pl.col("AuctionId").alias("AuctionStartId"), "StartTime"])
    spans = (
        ends.join(starts, left_on=["StreakId", "StartRowIdx"], right_on=["StreakId", "RowIdx"], how="inner")
            .select([
                pl.col("AuctionStartId"),
                pl.col("AuctionId").alias("AuctionEndId"),
                pl.col("StartTime"),
                pl.col("EndTime"),
                pl.col("BlockVolume").alias("TotalBlockVolume"),
                pl.col("BlockImbalance").alias("TotalBlockImbalance"),
            ])
    )

    if not return_ids_list:
        return spans.with_row_index("BlockId")

    # Optional: materialize full auction id list per block
    id_map = base.select(["StreakId", "RowIdx", "AuctionId"])
    ranges = (
        ends.join(starts, left_on=["StreakId", "StartRowIdx"], right_on=["StreakId", "RowIdx"], how="inner")
            .with_columns(pl.int_ranges(pl.col("StartRowIdx"), pl.col("RowIdx") + 1).alias("RowRange"))
    )
    exploded = ranges.explode("RowRange").join(id_map, left_on=["StreakId", "RowRange"], right_on=["StreakId", "RowIdx"], how="inner")

    return (
        exploded.
        group_by(["StreakId", 
                  "RowIdx"]).
        agg([
            pl.col("AuctionId"),
            pl.first("StartTime"),
            pl.first("EndTime"),
            pl.first("TotalBlockVolume"),
            pl.first("TotalBlockImbalance"),
        ]).
        with_row_index("BlockId").
        select(["BlockId", 
                "AuctionId", 
                "StartTime", 
                "EndTime", 
                "TotalBlockVolume", 
                "TotalBlockImbalance"]) 
    )


def compute_forward_outcomes(
    df_ticks: pl.DataFrame = None,
    blocks: pl.DataFrame = None,
    minutes_ahead: int = 15,
    price_source: Literal["mid","trade","bid","ask"] = "trade",
) -> pl.DataFrame:
    
    """
    Per block: get price at entry=end, exit=end+minutes, and simple return.

    NOTE: Uses **left=blocks .join_asof(right=price)** so that for each target
    timestamp (entry/exit) we pick the last price with `timestamp <= target`.
    This avoids the nulls observed when the target lies **after** the last
    tick in the window.
    """
    
    if df_ticks is None or blocks is None:
        raise Exception("Parameters 'df_ticks' and 'blocks' must not be None.")
        return
    
    if "Price" not in df_ticks.columns:
        raise Exception("Parameter called df_ticks must have a column named 'Price'.")
        return
    
    # Build price series (sorted)
    match price_source:
        case "mid":
            price = df_ticks.select([pl.col("Datetime"), ((pl.col("BidPrice") + pl.col("AskPrice")) / 2).alias("Price")])
        case "trade":
            price = df_ticks.select(["Datetime", "Price"])
        case "bid":
            price = df_ticks.select(["Datetime", pl.col("BidPrice").alias("Price")])
        case "ask":
            price = df_ticks.select(["Datetime", pl.col("AskPrice").alias("Price")])
        case _:
            price = df_ticks.select(["Datetime", ((pl.col("BidPrice") + pl.col("AskPrice")) / 2).alias("Price")])
    
    #-----------------------------
    price = price.sort("Datetime")
    #-----------------------------
    
    # Entry/exit timestamps per block
    b = (
            blocks
            .with_columns([
                pl.col("EndTime").alias("StartTime"),
               (pl.col("EndTime") + pl.duration(minutes=minutes_ahead)).alias("EndTime"),
            ])
            .select(["BlockId", "AuctionStartId", "AuctionEndId", "StartTime", "EndTime"])
            .sort("StartTime")
        )

    # For each StartTime, take last price with timestamp <= StartTime
    entry = (
                b
                .select(["BlockId", "StartTime"])
                .join_asof(price, left_on="StartTime", right_on="Datetime", strategy="backward")
                .rename({"Price": "EntryPrice"})
                .select(["BlockId", "StartTime", "EntryPrice"])
            )

    # For each exit_ts, take last price with timestamp <= exit_ts
    exit_ = (
                b.select(["BlockId", "EndTime"]).sort("EndTime")
                .join_asof(price, left_on="EndTime", right_on="Datetime", strategy="backward")
                .rename({"Price": "ExitPrice"})
                .select(["BlockId", "EndTime", "ExitPrice"])
            )

    # Combine and compute simple returns
    res = (
                b
                .join(entry, on=["BlockId", "StartTime"], how="left")
                .join(exit_, on=["BlockId", "EndTime"], how="left")
                .with_columns(((pl.col("ExitPrice") - pl.col("EntryPrice")) / pl.col("EntryPrice")).alias("SimpleReturn"))
                .select([
                    "BlockId", "AuctionStartId", "AuctionEndId", "StartTime", "EndTime", "EntryPrice", "ExitPrice", "SimpleReturn",
                ])
            )

    return res


def compute_forward_outcomes_from_timestamps(
    df_ticks: pl.DataFrame = None,
    entries: pl.DataFrame = None,
    minutes_ahead: int = 5,
    price_source: Literal["mid", "trade", "bid", "ask"] = "mid",
    by: Sequence[str] | None = None,
) -> pl.DataFrame:
    
    """
    Compute forward simple returns using pure timestamps (no BlockId).

    Inputs
    ------
    df_ticks: Polars DataFrame with at least 'Datetime' and price columns depending on `price_source`.
              Optionally contains partition columns listed in `by` (e.g., 'Symbol').
    entries:  Polars DataFrame providing entry timestamps:
              - must have 'entry_ts' (Datetime) OR 'Datetime' (will be renamed to 'entry_ts').
              - may contain the same partition columns in `by`.

    Parameters
    ----------
    minutes_ahead: horizon in minutes for the forward exit timestamp.
    price_source:  which price to use: 'mid', 'trade', 'bid', or 'ask'.
    by:            optional sequence of column names to partition as-of joins
                   (e.g., by=['Symbol']). If provided, both `df_ticks` and `entries`
                   must contain these columns.

    Returns
    -------
    Polars DataFrame with columns:
      [by..., entry_id, entry_ts, exit_ts, entry_price, exit_price, simple_return]
    """

    if df_ticks is None or entries is None:
        raise ValueError("`df_ticks` and `entries` must not be None.")

    by = list(by) if by else []

    # Build price series (Datetime, Price) depending on source ----
    if price_source == "mid":
        
        required = {"Datetime", "BidPrice", "AskPrice"}
        missing = required - set(df_ticks.columns)
        
        if missing:
            raise ValueError(f"df_ticks missing columns for mid price: {sorted(missing)}")
        price = df_ticks.select([
            *by,
            pl.col("Datetime"),
            ((pl.col("BidPrice") + pl.col("AskPrice")) / 2).alias("Price"),
        ])
        
    elif price_source == "trade":
        required = {"Datetime", "Price"}
        missing = required - set(df_ticks.columns)
        if missing:
            raise ValueError(f"df_ticks missing columns for trade price: {sorted(missing)}")
        price = df_ticks.select([*by, "Datetime", "Price"])
        
    elif price_source == "bid":
        required = {"Datetime", "BidPrice"}
        missing = required - set(df_ticks.columns)
        if missing:
            raise ValueError(f"df_ticks missing columns for bid price: {sorted(missing)}")
        price = df_ticks.select([*by, "Datetime", pl.col("BidPrice").alias("Price")])
        
    elif price_source == "ask":
        required = {"Datetime", "AskPrice"}
        missing = required - set(df_ticks.columns)
        if missing:
            raise ValueError(f"df_ticks missing columns for ask price: {sorted(missing)}")
        price = df_ticks.select([*by, "Datetime", pl.col("AskPrice").alias("Price")])
        
    else:
        raise ValueError(f"Unknown price_source: {price_source}")

    # Sort for asof (must sort by all `by` keys then time)
    sort_keys_price = [*by, "Datetime"]
    price = price.sort(sort_keys_price)

    # Normalize entries to have 'entry_ts' and optional partitions ----
    if "entry_ts" not in entries.columns:
        
        if "Datetime" in entries.columns:
            entries = entries.rename({"Datetime": "entry_ts"})
        else:
            raise ValueError("`entries` must have 'entry_ts' (Datetime) or 'Datetime'.")

    # Check that partition columns exist in entries if used
    missing_by_entries = set(by) - set(entries.columns)
    if missing_by_entries:
        raise ValueError(f"`entries` missing partition columns: {sorted(missing_by_entries)}")

    # Add exit_ts and a unique entry_id to stabilize joins even with duplicate timestamps
    entries_work = (
        entries
        .with_columns([
            (pl.col("entry_ts") + pl.duration(minutes=minutes_ahead)).alias("exit_ts"),
        ])
        .with_row_index(name="entry_id")  # deterministic id in current order
        .select([*by, "entry_id", "entry_ts", "exit_ts"])
        .sort([*by, "entry_ts"])
    )

    # As-of join for entry (last price <= entry_ts) ----
    entry_join = entries_work.join_asof(
        price,
        left_on="entry_ts",
        right_on="Datetime",
        by=by if by else None,
        strategy="backward",
    ).rename({"Price": "entry_price"}).select([*by, "entry_id", "entry_ts", "entry_price"])

    # As-of join for exit (last price <= exit_ts) ----
    exit_join = entries_work.sort([*by, "exit_ts"]).join_asof(
        price,
        left_on="exit_ts",
        right_on="Datetime",
        by=by if by else None,
        strategy="backward",
    ).rename({"Price": "exit_price"}).select([*by, "entry_id", "exit_ts", "exit_price"])

    # Merge & compute simple return ----
    out = (
        entries_work
        .join(entry_join, on=[*by, "entry_id", "entry_ts"], how="left")
        .join(exit_join,  on=[*by, "entry_id", "exit_ts"],  how="left")
        .with_columns(
            pl.when(pl.col("entry_price").is_not_null() & pl.col("exit_price").is_not_null())
              .then((pl.col("exit_price") - pl.col("entry_price")) / pl.col("entry_price"))
              .otherwise(None)
              .alias("simple_return")
        )
    )

    return out.select([*by, "entry_id", "entry_ts", "exit_ts", "entry_price", "exit_price", "simple_return"])





import matplotlib.pyplot as plt

df            = load_tick_data(path = r'C:/tmp/ZN.txt')
df_agg        = aggregate_auctions(df=df); # df_agg.write_excel(workbook=r'C:/tmp/ZN_agg.xlsx')
df_agg        = df_agg.with_columns(TradeSide = pl.when(pl.col("BuyVolume") > pl.col("SellVolume")).then(pl.lit("Long")).otherwise(pl.lit("Short")))
df_agg_blocks = get_valid_blocks(agg=df_agg); # df_agg_blocks.write_excel(workbook=r'C:/tmp/ZN_blocks.xlsx')
df_forward    = compute_forward_outcomes(df_ticks=df, blocks=df_agg_blocks)
df_forward    = df_forward.join(other=df_agg.select(cs.by_name('AuctionId', 'TradeSide')), how='left', left_on='AuctionEndId', right_on='AuctionId')
df_forward    = df_forward.with_columns(SimpleReturn = pl.when(pl.col("TradeSide") == "Short").then(pl.col("SimpleReturn") * -1).otherwise(pl.col("SimpleReturn")))

plt.plot(df_forward['SimpleReturn'].cum_sum())


buy_cond   = (pl.col("BuyVolume")  >= 4000) & (pl.col("SellVolume") <= 500)
sell_cond  = (pl.col("SellVolume") >= 4000) & (pl.col("BuyVolume")  <= 500)
big_trades = (
    df_agg
    .filter(buy_cond | sell_cond)
    .with_columns(
        TradeSide = (
            pl.when(buy_cond)
              .then(pl.lit("LongTrade"))
              .when(sell_cond)
              .then(pl.lit("ShortTrade"))
              .otherwise(pl.lit("Unknown"))
        ),
        # Optional: strength of the imbalance for ranking/significance tests
        Imbalance = (pl.col("BuyVolume") - pl.col("SellVolume"))
    )
)

buy  = big_trades.filter(pl.col("TradeSide") == "LongTrade")
sell = big_trades.filter(pl.col("TradeSide") == "ShortTrade")
# buy  = big_trades.filter(pl.col("Imbalance") >= 5000)
# sell = big_trades.filter(pl.col("Imbalance") <= -5000)

plt.plot(df['Datetime'], df['Price'], zorder=0, lw=0.5)
plt.scatter(buy['EndTime'], buy['LastAskPrice'], zorder=1,   s=1, c='lime') #, edgecolors='black')
plt.scatter(sell['EndTime'], sell['LastBidPrice'], zorder=1, s=1, c='red')#,  edgecolors='black')
plt.savefig("C:/Users/IRONMAN/Desktop/entries.png", dpi=1200) 

