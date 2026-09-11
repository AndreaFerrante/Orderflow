"""Causal session context: prior-session levels, running state and trend state.

Every output row uses only ticks at or before that row. RTH is 08:30-15:59:59.999999 CT;
sessions are keyed by ``Date`` on RTH ticks.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from orderflow.market.large_order_flow import _running_vwap_crosses

__all__ = [
    "prior_rth_levels", "running_session_state", "vwap_slope_at",
    "calibrate_slope_thresholds", "classify_trend_state",
]

_LEVEL_SCHEMA = {
    "Date": pl.String, "prev_high": pl.Float64, "prev_low": pl.Float64,
    "prev_close": pl.Float64, "prev_poc": pl.Float64,
    "prev_vah": pl.Float64, "prev_val": pl.Float64,
}


def prior_rth_levels(ticks: pl.DataFrame, *, tick_size: float,
                     value_area_share: float = 0.70) -> pl.DataFrame:
    rth = ticks.filter(pl.col("SessionType") == "RTH")
    highs = rth.group_by("Date").agg(pl.col("Price").max().alias("high"))
    lows = rth.group_by("Date").agg(pl.col("Price").min().alias("low"))
    closes = (
        rth.sort("Datetime").group_by("Date", maintain_order=True)
        .agg(pl.col("Price").last().alias("close"))
    )
    pocs = rth.group_by("Date").agg(pl.col("Price").alias("_prices"), pl.col("Volume").alias("_volumes"))
    poc_rows = []
    for date, prices, volumes in zip(pocs["Date"], pocs["_prices"], pocs["_volumes"]):
        keys = [round(p / tick_size) for p in prices]
        vol_by_key: dict[int, float] = {}
        for k, v in zip(keys, volumes):
            vol_by_key[k] = vol_by_key.get(k, 0.0) + v
        poc_key = max(vol_by_key, key=lambda k: (vol_by_key[k], -k))
        min_key, max_key = min(vol_by_key), max(vol_by_key)
        total = sum(vol_by_key.values())
        target = value_area_share * total
        val_key = vah_key = poc_key
        accumulated = vol_by_key[poc_key]
        while accumulated < target and (val_key > min_key or vah_key < max_key):
            below = vol_by_key.get(val_key - 1, 0.0) if val_key > min_key else None
            above = vol_by_key.get(vah_key + 1, 0.0) if vah_key < max_key else None
            if below is None:
                take_above = True
            elif above is None:
                take_above = False
            else:
                take_above = above >= below
            if take_above:
                vah_key += 1
                accumulated += above
            else:
                val_key -= 1
                accumulated += below
        poc_rows.append({
            "Date": date,
            "poc": poc_key * tick_size,
            "vah": vah_key * tick_size,
            "val": val_key * tick_size,
        })
    poc_frame = pl.DataFrame(poc_rows, schema={"Date": pl.String, "poc": pl.Float64,
                                                "vah": pl.Float64, "val": pl.Float64})

    daily = (
        highs.join(lows, on="Date")
        .join(closes, on="Date")
        .join(poc_frame, on="Date")
        .sort("Date")
    )

    shifted = daily.select(
        pl.col("Date"),
        pl.col("high").shift(1).alias("prev_high"),
        pl.col("low").shift(1).alias("prev_low"),
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("poc").shift(1).alias("prev_poc"),
        pl.col("vah").shift(1).alias("prev_vah"),
        pl.col("val").shift(1).alias("prev_val"),
    )
    return shifted.select(list(_LEVEL_SCHEMA.keys()))
