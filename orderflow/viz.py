"""
orderflow_viz.py — plotting utilities (matplotlib)

Public API:
    plot_forward_overlaid_relative

Design
------
• Single plotting entry-point that overlays all forward paths on one axis.
• X-axis is **relative time** (seconds from signal), so every path starts at 0.
• Y-axis can be normalized **return** (default) or **delta price**.
• Uses Pandas only for indexing/alignment; main pipeline remains Polars.
"""
from __future__ import annotations

from typing import Literal
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["plot_forward_overlaid_relative"]

# -----------------------------------------------------------------------------
# Local helpers (kept small to avoid importing internals)
# -----------------------------------------------------------------------------

def _ensure_price_series(df: pl.DataFrame, price_source: Literal["mid","trade","bid","ask"] = "mid") -> pl.DataFrame:
    """Return [timestamp, price] according to the selected source."""
    if price_source == "mid":
        out = df.select([pl.col("timestamp"), ((pl.col("BidPrice") + pl.col("AskPrice")) / 2.0).alias("price")])
    elif price_source == "trade" and "Price" in df.columns:
        out = df.select([pl.col("timestamp"), pl.col("Price").alias("price")])
    elif price_source == "bid":
        out = df.select([pl.col("timestamp"), pl.col("BidPrice").alias("price")])
    elif price_source == "ask":
        out = df.select([pl.col("timestamp"), pl.col("AskPrice").alias("price")])
    else:
        out = df.select([pl.col("timestamp"), ((pl.col("BidPrice") + pl.col("AskPrice")) / 2.0).alias("price")])
    return out.sort("timestamp")


def _annotate_side(blocks: pl.DataFrame) -> pl.DataFrame:
    return blocks.with_columns(
        pl.when(pl.col("imbalance") > 0).then(pl.lit("buy"))
         .when(pl.col("imbalance") < 0).then(pl.lit("sell"))
         .otherwise(pl.lit("flat")).alias("side")
    )

# -----------------------------------------------------------------------------
# Plotter: overlay all paths, t=0 alignment
# -----------------------------------------------------------------------------

def plot_forward_overlaid_relative(
    df_ticks: pl.DataFrame,
    blocks: pl.DataFrame,
    *,
    minutes_ahead: int = 5,
    price_source: Literal["mid","trade","bid","ask"] = "mid",
    y_mode: Literal["return", "delta_price"] = "return",
    max_events: int | None = 200,
    color_by_side: bool = True,
    alpha: float = 0.6,
    linewidth: float = 1.0,
    # NEW:
    legend: bool = True,
    legend_max: int = 30,
    annotate_end: bool = False,
) -> None:
    """
    Single-axis overlay: every path starts at t=0 (signal time).
    - y_mode='return':      price/entry - 1
      y_mode='delta_price': price - entry

    Legend:
      - If legend=True and number of plotted paths <= legend_max, show a legend
        with labels "#{block_id} (side)".
      - Otherwise suppress the legend to avoid clutter.
    Optional:
      - annotate_end=True writes the block_id at the end of each path.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, AutoMinorLocator
    import pandas as pd

    # --- polished style (light theme) ---
    rc = {
        "axes.facecolor": "#fbfbfd",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#9aa3ad",
        "axes.labelcolor": "#1f2937",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "grid.color": "#e5e7eb",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
    }

    # --- data prep (unchanged logic) ---
    price = _ensure_price_series(df_ticks, price_source)
    price_pd = price.to_pandas().set_index("timestamp").sort_index()
    if price_pd.empty:
        print("No price data available for plotting.")
        return

    blk = (
        _annotate_side(blocks)
        .with_columns([
            pl.col("end").alias("entry_ts"),
            (pl.col("end") + pl.duration(minutes=minutes_ahead)).alias("exit_ts"),
        ])
        .select(["block_id", "entry_ts", "exit_ts", "side"])
        .sort("block_id")
    ).to_pandas()
    if blk.empty:
        print("No blocks to plot.")
        return

    def slice_window(ts_start, ts_end) -> pd.DataFrame:
        idx = price_pd.index
        if len(idx) == 0:
            return pd.DataFrame(columns=["price"])
        start_pos = idx.searchsorted(ts_start, side="right") - 1
        end_pos   = idx.searchsorted(ts_end,   side="right")
        if start_pos < 0:
            return pd.DataFrame(columns=["price"])
        seg = price_pd.iloc[start_pos:end_pos].copy()
        # Anchor at entry timestamp, using asof entry price when needed
        if not seg.empty and seg.index[0] != ts_start:
            entry_price = float(seg.iloc[0]["price"])
            seg = pd.concat([pd.DataFrame({"price":[entry_price]}, index=[ts_start]),
                             seg.iloc[1:]])
        return seg

    rows: list[pd.DataFrame] = []
    for _, r in blk.iterrows():
        seg = slice_window(r.entry_ts, r.exit_ts)
        if seg.empty:
            continue
        p0 = float(seg.iloc[0]["price"]) if len(seg) else None
        if p0 is None:
            continue
        rel_sec = (seg.index - r.entry_ts).total_seconds().astype(int)
        if y_mode == "return":
            y = seg["price"].values / p0 - 1.0
        else:
            y = seg["price"].values - p0
        rows.append(pd.DataFrame({"block_id": r.block_id,
                                  "side": r.side,
                                  "t": rel_sec,
                                  "y": y}))
    if not rows:
        print("No segments available to plot.")
        return

    long = pd.concat(rows, ignore_index=True)
    long = long[long["t"].between(0, minutes_ahead * 60)]

    # color palette (cleaner hues)
    if color_by_side:
        side_to_color = {"buy": "#1f7a8c", "sell": "#c44536", "flat": "#94a3b8"}
        def line_color(side): return side_to_color.get(side, "#3b82f6")
    else:
        def line_color(_): return None

    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.4)

        # soften spines a bit
        for s in ax.spines.values():
            s.set_alpha(0.5)

        # draw lines
        plotted = 0
        for bid, g in long.groupby("block_id"):
            if max_events is not None and plotted >= max_events:
                break
            side = g["side"].iloc[0] if "side" in g.columns and len(g) else "flat"
            clr  = line_color(side)
            lbl  = f"#{bid} ({side})"
            (h,) = ax.plot(
                g["t"], g["y"],
                alpha=alpha,
                linewidth=linewidth,
                label=lbl if legend else None,
                color=clr,
                solid_capstyle="round",
                zorder=2,
            )
            if annotate_end and len(g):
                xe, ye = g["t"].iloc[-1], g["y"].iloc[-1]
                ax.plot(xe, ye, marker="o", markersize=3.5,
                        color=h.get_color(), zorder=3)
                ax.text(xe, ye, f"{bid}", fontsize=8, color=h.get_color(),
                        va="center", ha="left")
            plotted += 1

        if plotted == 0:
            print("Nothing to plot after filtering.")
            return

        # baseline at 0
        ax.axhline(0, linestyle="-", linewidth=1.0, alpha=0.35, color="#6b7280")

        # labels
        ax.set_title(f"Forward paths (t=0 at signal, {minutes_ahead}m, {y_mode})")
        ax.set_xlabel("Seconds from entry")
        ax.set_ylabel("Return from entry" if y_mode == "return" else "Δ Price from entry")

        # percent format for returns
        if y_mode == "return":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:,.1f}%"))

        # minor ticks for nicer grid
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # legend outside if not too many
        if legend and plotted <= legend_max:
            leg = ax.legend(title="Signals", fontsize=9, title_fontsize=10, ncol=2,
                            loc="upper left", bbox_to_anchor=(1.02, 1.0),
                            frameon=True, fancybox=True, framealpha=0.9)
            leg.get_frame().set_edgecolor("#9aa3ad")
            fig.tight_layout(rect=(0, 0, 0.80, 1))
        else:
            fig.tight_layout()

        plt.xlim(0, minutes_ahead * 60)
        plt.show()
