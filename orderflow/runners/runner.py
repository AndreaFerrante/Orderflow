import math
from datetime import time

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

import orderflow.configuration as cf
from orderflow._volume_factory import get_tickers_in_folder, get_market_evening_session
from orderflow.volume_profile import get_volume_profile_peaks_valleys, get_daily_high_and_low_by_session, get_dynamic_cumulative_delta_per_session
from orderflow.volume_profile_kde import gaussian_kde_numba_parallel, get_kde_high_low_price_peaks
from orderflow.volume_profile import get_volume_profile_areas, get_volume_profile_node_volume, get_daily_session_moving_POC
from orderflow.auctions import (
    aggregate_auctions,
    get_valid_blocks,
    compute_forward_outcomes,
)
from orderflow.compressor import compress_to_minute_bars_pl


# See 100 elements in polars tables while printing !
pl.Config.set_tbl_rows(100)


# ----------------- CONFIG -----------------------------------------------------
FOLDER         = r"C:/Users/Tommy/Documents/PycharmProjects/Orderflow/Sources/ES/"
TICKER         = "ES"
MARKET         = "CME"
SEPARATOR      = ";"
EXTENSION      = ".txt"

# Strategy params
N_CONSECUTIVE  = 2
VOL_THRESH     = 100
MINUTES_AHEAD  = 3
MIN_ABS_IMB    = 1.5 # <-- NEW: require |Imbalance| >= 2.0 (ratio mode => ≥2x dominance)


# ----------------- LOAD TICKS FOR THE MONTH (CONCAT WEEKS) -------------------
df_ticks = get_tickers_in_folder(
    path      = FOLDER,
    ticker    = TICKER,
    extension = EXTENSION,
    separator = SEPARATOR,
    market    = MARKET
)

# ------------------------------- ENRICH TICKS --------------------------------
df_ticks = df_ticks.with_columns(Hour = pl.col("Datetime").dt.hour())
df_ticks = df_ticks.with_columns(SessionType = get_market_evening_session(data=df_ticks, ticker="ES"))
df_ticks = df_ticks.with_columns(VA_Areas     = get_volume_profile_areas(df_ticks.to_pandas()))
df_ticks = df_ticks.with_columns(ValleysPeaks = get_volume_profile_peaks_valleys(df_ticks.to_pandas()))
df_ticks = df_ticks.with_columns(POC = get_daily_session_moving_POC(df_ticks.to_pandas()))

df_cd = get_dynamic_cumulative_delta_per_session(df_ticks.to_pandas())
df_cd_reset = df_cd.reset_index(drop=True)
df_cd_polars = pl.from_pandas(df_cd_reset)
df_ticks = pl.concat([df_ticks, df_cd_polars], how="horizontal")

lows, highs  = get_daily_high_and_low_by_session(df_ticks.to_pandas())
df_ticks = df_ticks.with_columns([
    pl.Series("Session_High", highs),
    pl.Series("Session_Low", lows),
])

volume_nodes, total_volumes = get_volume_profile_node_volume(df_ticks.to_pandas())
df_ticks = df_ticks.with_columns([
    pl.Series("Node_Volume", volume_nodes),
    pl.Series("Session_Volume", total_volumes),
])
# Identify LVN by Volume distibution
tick_size = cf.FUTURE_VALUES.loc[cf.FUTURE_VALUES['Ticker'] == TICKER, 'Tick_Size'].values[0]
df_ticks = df_ticks.with_columns(
    (
        pl.col("Node_Volume")
        < 0.25
        * (pl.col("Session_Volume") / ((pl.col("Session_High") - pl.col("Session_Low")) / tick_size))
    )
    .cast(pl.Int8)
    .alias("LVN")
)

#df_ticks = pl.read_csv(r'C:\Users\Tommy\Documents\PycharmProjects\Orderflow\Sources\ES\ESZ25-CME_ENR_20251212_225959.csv', separator=';')
# df_ticks = df_ticks.with_columns(
#     pl.col("Datetime")
#     .str.strptime(
#         pl.Datetime,
#         "%Y-%m-%dT%H:%M:%S%.f",
#         strict=True
#     )
#     .alias("Datetime")
# )

# ------------------------- FILTER TRADE TRIGGERS TICKS -----------------------
# filter by time
filtered_df_ticks = df_ticks.filter(
    (pl.col("Datetime").dt.time() >= time(8, 30, 0)) &
    (pl.col("Datetime").dt.time() <= time(15, 0, 0))
)

# identify big prints
filtered_df_ticks = filtered_df_ticks.filter(
    pl.col('Volume') >= 35
)

# Condition 1: TradeType=2 (buy), outside VA, > POC, is LVN or Low Volume Area
buy_signal = (
    (pl.col('TradeType') == 2) &
    (pl.col('VA_Areas') == 'na') &
    (pl.col('Price') > pl.col('POC')) &
    ((pl.col('LVN') == 1) | (pl.col('ValleysPeaks') <= -1))
)
# Condition 2: TradeType=1 (sell), outside VA, < POC, is LVN or Low Volume Area
sell_signal = (
    (pl.col('TradeType') == 1) &
    (pl.col('VA_Areas') == 'na') &
    (pl.col('Price') < pl.col('POC')) &
    ((pl.col('LVN') == 1) | (pl.col('ValleysPeaks') <= -1))
)

# buy_signal OR sell_signal
filtered_df_ticks = filtered_df_ticks.filter(buy_signal | sell_signal)

# ------------------------- ENRICH TRADE TRIGGERS TICKS -----------------------
# add 1 minute bars info
one_min_df_bars = compress_to_minute_bars_pl(df_ticks)
one_min_df_bars_sorted = one_min_df_bars.sort("Datetime")
one_min_df_bars_sorted = one_min_df_bars_sorted.with_columns([
        pl.col("Datetime").alias("current_bar_datetime")
    ])

rename_map = {
        col: f"current_bar_{col.lower()}" 
        for col in one_min_df_bars_sorted.columns 
        if col not in ["Datetime", "current_bar_datetime"]
    }

one_min_df_bars_sorted_renamed = one_min_df_bars_sorted.rename(rename_map)

one_min_df_bars_extended = one_min_df_bars_sorted_renamed.with_columns([
        # Datetime next bar
        pl.col("Datetime").shift(-1).alias("next_bar_datetime"),
        
        # next bar OHLCV
        pl.col("current_bar_open").shift(-1).alias("next_bar_open"),
        pl.col("current_bar_high").shift(-1).alias("next_bar_high"),
        pl.col("current_bar_low").shift(-1).alias("next_bar_low"),
        pl.col("current_bar_close").shift(-1).alias("next_bar_close"),
        pl.col("current_bar_volume").shift(-1).alias("next_bar_volume"),
        pl.col("current_bar_numberoftrades").shift(-1).alias("next_bar_num_trades"),
        pl.col("current_bar_askvolume").shift(-1).alias("next_bar_ask_volume"),
        pl.col("current_bar_bidvolume").shift(-1).alias("next_bar_bid_volume"),
    ])

filtered_df_ticks_with_bars = filtered_df_ticks.join_asof(
        one_min_df_bars_extended,
        left_on="Datetime",
        right_on="Datetime",
        strategy="backward"
    )

# Plotting the KDE curve overlap here ...
# df_ticks_gb   = (df_ticks.
#                  group_by("Price").
#                  agg(pl.sum("Volume"),
#                      pl.min("Datetime").alias("MinDatetime"),
#                      pl.max("Datetime").alias("MaxDatetime")).
#                  sort(["Price"]))
# bigger        = df_ticks.filter(pl.col('Volume') >= 50)
# bigger        = bigger.group_by(pl.col('Price')).agg(pl.sum('Volume')).sort('Price')
# prices        = np.array(df_ticks_gb['Price'])
# volumes       = np.array(df_ticks_gb['Volume'])
# min_dt        = np.array(df_ticks_gb['MinDatetime'])
# max_dt        = np.array(df_ticks_gb['MaxDatetime'])
# kde_values    = gaussian_kde_numba_parallel(source=prices, weight=volumes, h=.5)
# kde_peaks     = get_kde_high_low_price_peaks(kde_values)
# pv_prices     = prices[kde_peaks]
# kde_df        = pd.DataFrame({'Price':prices, 'Volume':volumes ,'kde':kde_values})

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('Price')
# ax1.set_ylabel('Volume / Counter', color='red')
# ax1.plot(kde_df['Price'], kde_df['kde'], color='red')
# ax2 = ax1.twinx()
# ax2.bar(prices, volumes, color='blue', edgecolor='black', alpha=0.5, width=0.25)
# ax3 = ax2.twinx()
# ax3.scatter(pv_prices, kde_values[kde_peaks], color='lime', zorder=5)
# fig.tight_layout()
# plt.show()


# ----------------- AUCTIONS (ratio mode so 2.0 means ≥2x dominance) ----------
df_agg = aggregate_auctions(
    df             = df_ticks,
    imbalance_mode = "ratio",  # <-- important for MIN_ABS_IMB=2.0
).with_columns(
    pl.when(pl.col("BuyVolume") > pl.col("SellVolume"))
      .then(pl.lit("Long"))
      .otherwise(pl.lit("Short"))
    .alias("TradeSide")
)
print("[Monthly] Auctions:", df_agg.shape)


# ----------------- BLOCKS (EVENTS) with |Imbalance| >= 2.0 -------------------
df_blocks = get_valid_blocks(
    agg               = df_agg,
    n_consecutive     = N_CONSECUTIVE,
    vol_thresh        = VOL_THRESH,
    min_abs_imbalance = MIN_ABS_IMB,   # <-- require strong imbalance for entries
)
print("[Monthly] Blocks:", df_blocks.shape)

# ------------------ Add block info to the TRADE TRIGGERS ---------------------
# A score is calcuated to measure if and when a block of consecutive unbalance
# auctions happened before the trigger.
# score results:
# 1.0: blocks within 1 minute before
# 0.7: blocks between 1 and 2 minutes before
# 0.4: blocks between 2 and 3 minutes before
# 0.0: otherwise

# Step 1: add direction info column
df_blocks_prepared = df_blocks.with_columns([
    # 1 (bullish), -1 (bearish), 0 (otherwise)
    pl.when(pl.col("TotalBlockImbalance") > 0)
    .then(pl.lit(1))
    .when(pl.col("TotalBlockImbalance") < 0)
    .then(pl.lit(-1))
    .otherwise(pl.lit(0))
    .alias("block_direction")
])
# Step 2: add direction info column
tick_signals_prepared = filtered_df_ticks_with_bars.with_columns([
    # 1 (buy) o -1 (sell)
    pl.when(pl.col("TradeType") == 2)
    .then(pl.lit(1))
    .when(pl.col("TradeType") == 1)
    .then(pl.lit(-1))
    .otherwise(pl.lit(0))
    .alias("tick_direction")
])
# Step 3: join dfs
tick_signals = tick_signals_prepared.join_asof(
        df_blocks_prepared,
        left_on="Datetime",
        right_on="EndTime",
        strategy="backward",
        suffix="_block"
    )
# Step 4: Calculate time distance and score
tick_signals = tick_signals.with_columns([
    # Distancd in secs
    (pl.col("Datetime") - pl.col("EndTime"))
    .dt.total_seconds()
    .alias("block_distance_sec"),
])
# Step 5: Filter by lookback window and same direction
tick_signals = tick_signals.with_columns([
    # Block is valid if:
    # 1. Exists (EndTime not null)
    # 2. it is inside the lookback window
    # 3. same tick direction
    pl.when(
        pl.col("EndTime").is_not_null() &
        (pl.col("block_distance_sec") <= 3 * 60) & # 3 minutes max
        (pl.col("block_direction") == pl.col("tick_direction")) &
        (pl.col("block_direction") != 0)
    )
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("has_aligned_block")
])
# Step 6: Calculate proximity score
tick_signals = tick_signals.with_columns([
    pl.when(~pl.col("has_aligned_block"))
    .then(pl.lit(0.0))
    .when(pl.col("block_distance_sec") <= 60)  # 0-1 minute
    .then(pl.lit(1.0))
    .when(pl.col("block_distance_sec") <= 120)  # 1-2 minutes
    .then(pl.lit(0.7))
    .when(pl.col("block_distance_sec") <= 180)  # 2-3 minutes
    .then(pl.lit(0.4))
    .otherwise(pl.lit(0.0))
    .alias("block_score")
])
 # Step 7: clean 1
tick_signals = tick_signals.with_columns([
    pl.when(pl.col("has_aligned_block"))
    .then(pl.col("TotalBlockImbalance"))
    .otherwise(pl.lit(None))
    .alias("block_imbalance"),
    
    
    pl.when(pl.col("has_aligned_block"))
    .then(pl.col("block_distance_sec"))
    .otherwise(pl.lit(None))
    .alias("block_distance_sec_final")
])
# Step 8: clean 2
cols_to_drop = [
    "tick_direction",
    "block_direction", 
    "AuctionStartId_block",
    "AuctionEndId_block",
    "EndTime",
    "StartTime",
    "TotalBlockVolume",
    "TotalBlockImbalance",
    "BlockId_block",
    "block_distance_sec",
    "has_aligned_block",
    "block_imbalance",
    "bloc_distance_sec"
]   

existing_cols_to_drop = [c for c in cols_to_drop if c in tick_signals.columns]
tick_signals = tick_signals.drop(existing_cols_to_drop)

if "block_distance_sec_final" in tick_signals.columns:
    tick_signals = tick_signals.rename({"block_distance_sec_final": "block_distance_sec"})
    

# --------------------------------------------------------------------
df_blocks = df_blocks.join_asof(df_ticks.select(['Datetime', 'Price']),
                                left_on  = "StartTime",
                                right_on = "Datetime",
                                strategy = "backward")
df_blocks = df_blocks.join_asof(df_agg.select(['StartTime', 'TradeSide']),
                                left_on  = "StartTime",
                                right_on = "StartTime",
                                strategy = "backward")

longs  = df_blocks.filter(df_blocks['TradeSide']=='Long')
shorts = df_blocks.filter(df_blocks['TradeSide']=='Short')
plt.plot(df_ticks['Datetime'], df_ticks['Price'])
plt.scatter(shorts['EndTime'], shorts['Price'], color='red', zorder=5)
plt.scatter(longs['EndTime'], longs['Price'], color='lime', zorder=5)
# --------------------------------------------------------------------

if df_blocks.is_empty():
    print(f"No blocks.")
else:
    # ------------- FORWARD OUTCOMES (MICRO-BACKTEST) -------------------------
    df_fwd = compute_forward_outcomes(
        df_ticks      = df_ticks,
        blocks        = df_blocks,
        minutes_ahead = MINUTES_AHEAD
    ).join(
        df_agg.select(["AuctionId","TradeSide"]),
        left_on="AuctionEndId", right_on="AuctionId", how="left"
    ).with_columns(
        pl.when(pl.col("TradeSide")=="Short")
          .then(-pl.col("SimpleReturnInTicks"))
          .otherwise(pl.col("SimpleReturnInTicks"))
          .alias("PnLTicks")
    )

    # ------------- QUICK STATS + CUM PnL PLOT --------------------------------
    pnl = df_fwd.get_column("PnLTicks").to_numpy() if "PnLTicks" in df_fwd.columns else np.array([])
    if pnl.size:
        hits  = (pnl > 0).mean()
        avg   = pnl.mean()
        std   = pnl.std(ddof=1) if pnl.size > 1 else float("nan")
        tstat = avg / (std / math.sqrt(pnl.size)) if (pnl.size > 1 and std > 0) else float("nan")
        cum   = np.cumsum(pnl)

        # max drawdown
        peak, max_dd = -1e18, 0.0
        for v in cum:
            peak   = max(peak, v)
            max_dd = max(max_dd, peak - v)

        print(f"Trades: {pnl.size} | Hit: {hits:.1%} | Avg: {avg:.3f} ticks | "f"T: {tstat:.3f} | MaxDD: {max_dd:.1f}")

        plt.figure()
        plt.plot(cum)
        plt.title("Cumulative PnL (ticks)")
        plt.xlabel("Event index")
        plt.ylabel("Cum PnL (ticks)")
        plt.title(f"Ticker {TICKER}, Consecutive {N_CONSECUTIVE}, VolThres {VOL_THRESH}, MinImb {MIN_ABS_IMB}, Ahead {MINUTES_AHEAD}min")
        plt.tight_layout()
        plt.show()

    else:
        print(f"No PnLTicks computed — check joins/columns.")
