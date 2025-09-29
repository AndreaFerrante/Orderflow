# === Monthly runner (ONLY June 2025) =========================================
# Folder layout: E:\python\DATA\ZN_ferrante\ZN_2025\ZN*.txt  (one file per week)
# Workflow:
#   - find files for June 2025 (YYYYMM == "202506")
#   - load/concat those weeks (ONLY that month)
#   - auctions (imbalance_mode="ratio") -> consecutive-imbalance blocks (|Imb|>=2) -> forward outcomes
#   - stats + cumulative PnL plot
# =============================================================================

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
FOLDER         = r""
TICKER         = "ZN"
MARKET         = "CBOT"
SEPARATOR      = ";"
EXTENSION      = ".txt"

# Strategy params
N_CONSECUTIVE  = 3
VOL_THRESH     = 1000
MINUTES_AHEAD  = 2
MIN_ABS_IMB    = 3.0   # <-- NEW: require |Imbalance| >= 2.0 (ratio mode => ≥2x dominance)

# Process ONLY this month key (YYYYMM)
MONTH_KEY      = "202505"   # June 2025

# ----------------- DISCOVER & FILTER FILES TO TARGET MONTH -------------------
FOLDER = FOLDER.rstrip("\\/") + "\\"
files = glob.glob(fr"{FOLDER}{TICKER}*{EXTENSION}")
if not files:
    raise FileNotFoundError(f"No '{TICKER}*{EXTENSION}' files found in: {FOLDER}")

rx = re.compile(r"_(\d{8})_")  # captures YYYYMMDD in filename (between underscores)
month_files = []
for f in files:
    name = os.path.basename(f)
    m = rx.search(name)
    if not m:
        # if pattern missing, skip (or use file mtime to infer)
        continue
    yyyymmdd = m.group(1)   # e.g., "20250509"
    ym = yyyymmdd[:6]       # "202505"
    if ym == MONTH_KEY:
        month_files.append(name)

if not month_files:
    raise FileNotFoundError(f"No files matched month {MONTH_KEY} in {FOLDER}")

month_files = sorted(month_files)
print(f"[Monthly] {MONTH_KEY}: {len(month_files)} weekly files -> {month_files}")


# ----------------- LOAD TICKS FOR THE MONTH (CONCAT WEEKS) -------------------
monthly_ticks = []
for one_name in month_files:
    df_ticks = get_tickers_in_folder(
        path=FOLDER,             # MUST end with '\'
        ticker=TICKER,
        single_file=one_name,    # basename only (loader does path + name)
        extension=EXTENSION,
        separator=SEPARATOR,
        market=MARKET,
    )
    monthly_ticks.append(df_ticks)

df_ticks_m = pl.concat(monthly_ticks, how="vertical_relaxed").sort("Datetime")
print("[Monthly] Ticks:", df_ticks_m.shape)


# ----------------- AUCTIONS (ratio mode so 2.0 means ≥2x dominance) ----------
df_agg = aggregate_auctions(
    df=df_ticks_m,
    imbalance_mode="ratio",       # <-- important for MIN_ABS_IMB=2.0
).with_columns(
    pl.when(pl.col("BuyVolume") > pl.col("SellVolume"))
      .then(pl.lit("Long")).otherwise(pl.lit("Short")).alias("TradeSide")
)
print("[Monthly] Auctions:", df_agg.shape)


# ----------------- BLOCKS (EVENTS) with |Imbalance| >= 2.0 -------------------
df_blocks = get_valid_blocks(
    agg=df_agg,
    n_consecutive=N_CONSECUTIVE,
    vol_thresh=VOL_THRESH,
    min_abs_imbalance=MIN_ABS_IMB,   # <-- require strong imbalance for entries
)
print("[Monthly] Blocks:", df_blocks.shape)

if df_blocks.is_empty():
    print(f"[Monthly] No blocks in {MONTH_KEY} with current params "
          f"(N={N_CONSECUTIVE}, VOL>={VOL_THRESH}, |Imb|>={MIN_ABS_IMB}).")
else:
    # ------------- FORWARD OUTCOMES (MICRO-BACKTEST) -------------------------
    df_fwd = compute_forward_outcomes(
        df_ticks=df_ticks_m,
        blocks=df_blocks,
        minutes_ahead=MINUTES_AHEAD
    ).join(
        df_agg.select(["AuctionId","TradeSide"]),
        left_on="AuctionEndId", right_on="AuctionId", how="left"
    ).with_columns(
        pl.when(pl.col("TradeSide")=="Short")
          .then(-pl.col("SimpleReturnInTicks"))
          .otherwise(pl.col("SimpleReturnInTicks"))
          .alias("PnLTicks"),
        pl.lit(MONTH_KEY).alias("Month")
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

        print(f"[{MONTH_KEY}] Trades: {pnl.size} | Hit: {hits:.1%} | Avg: {avg:.3f} ticks | "
              f"T: {tstat:.3f} | MaxDD: {max_dd:.1f}")

        plt.figure()
        plt.plot(cum)
        plt.title(f"Cumulative PnL (ticks) — {MONTH_KEY}")
        plt.xlabel("Event index")
        plt.ylabel("Cum PnL (ticks)")
        plt.tight_layout()
        plt.show()
    else:
        print(f"[Monthly] No PnLTicks computed — check joins/columns.")

# (Optional) keep df_fwd for later analysis; it includes PnLTicks and timestamps.
