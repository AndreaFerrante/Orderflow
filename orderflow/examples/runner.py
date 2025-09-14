"""
examples/runner.py  – quick demo: load → aggregate → detect → forward plot
Run:  python examples/runner.py
"""

import polars as pl
from pathlib import Path

# --- local imports (submodules because __init__.py is untouched) -------------
from orderflow.auctions import (
    load_tick_data,
    aggregate_auctions,
    identify_valid_blocks,
    compute_forward_outcomes,
)
from orderflow.viz import plot_forward_overlaid_relative

# --- parameters --------------------------------------------------------------
DATA_PATH        = Path(r"E:/python/DATA/CLF22-NYMEX_20211203_225959/CLF22-NYMEX_20211203_225959.txt")
SEGMENTATION     = "quote_any"
IMBALANCE_MODE   = "ratio"
N_CONSECUTIVE    = 3
VOLUME_THRESHOLD = 20
MINUTES_AHEAD    = 10
PRICE_SOURCE     = "mid"
Y_MODE           = "return"
MAX_EVENTS       = 200

# -----------------------------------------------------------------------------


def main() -> None:
    df     = load_tick_data(str(DATA_PATH), separator=";")
    agg    = aggregate_auctions(df, segmentation=SEGMENTATION, imbalance_mode=IMBALANCE_MODE)
    blocks = identify_valid_blocks(agg, n_consecutive=N_CONSECUTIVE, vol_thresh=VOLUME_THRESHOLD)

    print("\nDetected blocks:")
    print(blocks.select("block_id","start_id","end_id","start","end","total_volume","imbalance"))

    outcomes = compute_forward_outcomes(
        df_ticks=df,
        blocks=blocks,
        minutes_ahead=MINUTES_AHEAD,
        price_source=PRICE_SOURCE,
    )
    print("\nForward outcomes (first rows):")
    print(outcomes.head())

    plot_forward_overlaid_relative(
        df_ticks=df,
        blocks=blocks,
        minutes_ahead=MINUTES_AHEAD,
        price_source=PRICE_SOURCE,
        y_mode=Y_MODE,
        max_events=MAX_EVENTS,
        color_by_side=True,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_max=30,
    )


if __name__ == "__main__":
    main()
