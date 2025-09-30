import os, re, glob, math
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from orderflow._volume_factory import get_tickers_in_folder
from orderflow.auctions import (
    aggregate_auctions,
    get_valid_blocks,
    compute_forward_outcomes,
)


# ----------------- CONFIG -----------------------------------------------------
FOLDER         = r"C:\__tmp__"
TICKER         = "ZN"
MARKET         = "CBOT"
SEPARATOR      = ";"
EXTENSION      = ".txt"

# Strategy params
N_CONSECUTIVE  = 3
VOL_THRESH     = 1000
MINUTES_AHEAD  = 2
MIN_ABS_IMB    = 3.0 # <-- NEW: require |Imbalance| >= 2.0 (ratio mode => ≥2x dominance)


# ----------------- LOAD TICKS FOR THE MONTH (CONCAT WEEKS) -------------------
df_ticks = get_tickers_in_folder(
    path      = FOLDER,
    ticker    = TICKER,
    extension = EXTENSION,
    separator = SEPARATOR,
    market    = MARKET,
)


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
            peak = max(peak, v); max_dd = max(max_dd, peak - v)

        print(f"Trades: {pnl.size} | Hit: {hits:.1%} | Avg: {avg:.3f} ticks | "f"T: {tstat:.3f} | MaxDD: {max_dd:.1f}")

        plt.figure()
        plt.plot(cum)
        plt.title("Cumulative PnL (ticks)")
        plt.xlabel("Event index")
        plt.ylabel("Cum PnL (ticks)")
        plt.tight_layout()
        plt.show()

    else:
        print(f"No PnLTicks computed — check joins/columns.")

# (Optional) keep df_fwd for later analysis; it includes PnLTicks and timestamps.
