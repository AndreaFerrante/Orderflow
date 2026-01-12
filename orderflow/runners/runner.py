import math

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from orderflow._volume_factory import get_tickers_in_folder, get_market_evening_session
from orderflow.volume_profile import get_volume_profile_peaks_valleys
from orderflow.volume_profile_kde import gaussian_kde_numba_parallel, get_kde_high_low_price_peaks
from orderflow.volume_profile import get_volume_profile_areas
from orderflow.auctions import (
    aggregate_auctions,
    get_valid_blocks,
    compute_forward_outcomes,
)


# See 100 elements in polars tables while printing !
pl.Config.set_tbl_rows(100)


# ----------------- CONFIG -----------------------------------------------------
FOLDER         = r"/mnt/c/__tmp__"
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
df_ticks = df_ticks.with_columns(Hour = pl.col("Datetime").dt.hour())
df_ticks = df_ticks.with_columns(SessionType = get_market_evening_session(data=df_ticks, ticker="ES"))
#df_ticks = df_ticks.with_columns(VA_Areas     = get_volume_profile_areas(df_ticks.to_pandas()))
#df_ticks = df_ticks.with_columns(ValleysPeaks = get_volume_profile_peaks_valleys(df_ticks.to_pandas()))


# Plotting the KDE curve overlap here ...
df_ticks_gb   = (df_ticks.
                 group_by("Price").
                 agg(pl.sum("Volume"),
                     pl.min("Datetime").alias("MinDatetime"),
                     pl.max("Datetime").alias("MaxDatetime")).
                 sort(["Price"]))
bigger        = df_ticks.filter(pl.col('Volume') >= 50)
bigger        = bigger.group_by(pl.col('Price')).agg(pl.sum('Volume')).sort('Price')
prices        = np.array(df_ticks_gb['Price'])
volumes       = np.array(df_ticks_gb['Volume'])
min_dt        = np.array(df_ticks_gb['MinDatetime'])
max_dt        = np.array(df_ticks_gb['MaxDatetime'])
kde_values    = gaussian_kde_numba_parallel(source=prices, weight=volumes, h=.5)
kde_peaks     = get_kde_high_low_price_peaks(kde_values)
pv_prices     = prices[kde_peaks]
kde_df        = pd.DataFrame({'Price':prices, 'Volume':volumes ,'kde':kde_values})

fig, ax1 = plt.subplots()
ax1.set_xlabel('Price')
ax1.set_ylabel('Volume / Counter', color='red')
ax1.plot(kde_df['Price'], kde_df['kde'], color='red')
ax2 = ax1.twinx()
ax2.bar(prices, volumes, color='blue', edgecolor='black', alpha=0.5, width=0.25)
ax3 = ax2.twinx()
ax3.scatter(pv_prices, kde_values[kde_peaks], color='lime', zorder=5)
fig.tight_layout()
plt.show()


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
