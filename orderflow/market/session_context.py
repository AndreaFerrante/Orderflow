"""Causal session context: prior-session levels, running state and trend state.

Every output row uses only ticks at or before that row. RTH is 08:30-15:59:59.999999 CT;
sessions are keyed by ``Date`` on RTH ticks.

Anchor
------
``open_vwap``/``open_poc`` in :func:`running_session_state` are the enriched ``vwap``/``POC``
columns read at the first RTH tick of the day. Those columns reset at the RTH->ETH transition
(previous evening's Globex open), not at 08:30 -- so at RTH open they already carry the
overnight ETH session, not a fresh start. This is causal (still only ticks at or before that
row) and deliberate: ``band_z``, VWAP crosses and slope/drift all measure against this
Globex-anchored VWAP/POC, not an 08:30-anchored one.
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


_STATE_SCHEMA = {
    "Index": pl.Int64, "Date": pl.String, "minutes_since_open": pl.Float64,
    "run_high": pl.Float64, "run_low": pl.Float64, "open_price": pl.Float64,
    "open_vwap": pl.Float64, "open_poc": pl.Float64, "vwap": pl.Float64,
    "vwap_sd1_top": pl.Float64, "POC": pl.Float64, "vwap_crosses": pl.Int64,
    "open_location": pl.String, "returned_to_value": pl.Boolean,
}


def running_session_state(ticks: pl.DataFrame, levels: pl.DataFrame, *, tick_size: float,
                          cross_confirm_ticks: int = 2) -> pl.DataFrame:
    rth = ticks.filter(pl.col("SessionType") == "RTH").sort("Index")
    open_time = pl.col("Date").str.to_datetime("%Y-%m-%d").dt.offset_by("8h30m")
    out = rth.with_columns(
        pl.col("Price").cum_max().over("Date").alias("run_high"),
        pl.col("Price").cum_min().over("Date").alias("run_low"),
        ((pl.col("Datetime") - open_time).dt.total_microseconds() / 60_000_000.0)
        .alias("minutes_since_open"),
        pl.col("Price").first().over("Date").alias("open_price"),
        pl.col("vwap").first().over("Date").alias("open_vwap"),
        pl.col("POC").first().over("Date").alias("open_poc"),
    )
    confirm_distance = cross_confirm_ticks * tick_size
    out = out.sort("Index").with_columns(
        pl.map_batches(
            ["Price", "vwap"],
            lambda cols: pl.Series(
                _running_vwap_crosses(cols[0].to_numpy(), cols[1].to_numpy(),
                                      confirm_distance=confirm_distance),
                dtype=pl.Int64,
            ),
        ).over("Date").alias("vwap_crosses"),
    )
    out = out.join(levels, on="Date", how="left").with_columns(
        pl.when(pl.col("prev_val").is_null() | pl.col("prev_vah").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("open_price") < pl.col("prev_val")).then(pl.lit("below"))
        .when(pl.col("open_price") > pl.col("prev_vah")).then(pl.lit("above"))
        .otherwise(pl.lit("inside"))
        .alias("open_location"),
    )
    out = out.sort("Index").with_columns(
        pl.col("prev_val").is_not_null()
        .and_(pl.col("Price") >= pl.col("prev_val"))
        .and_(pl.col("Price") <= pl.col("prev_vah"))
        .cum_max().over("Date").fill_null(False)
        .alias("returned_to_value"),
    )
    return out.select(list(_STATE_SCHEMA.keys()))


def vwap_slope_at(ticks: pl.DataFrame, *, at_ct: str = "10:00") -> pl.DataFrame:
    """VWAP drift in points/hour from RTH open to a clock time (at_ct).

    Per session (Date), compute the slope of VWAP from the first RTH tick to the
    last RTH tick at or before `at_ct` (microsecond-precise). Drops sessions with
    no tick at/before at_ct or duration == 0.
    """
    rth = ticks.filter(pl.col("SessionType") == "RTH").sort("Datetime")

    if rth.height == 0:
        return pl.DataFrame(schema={"Date": pl.String, "vwap_slope": pl.Float64})

    # Parse at_ct (HH:MM) to hours and minutes
    h_str, m_str = at_ct.split(":")
    hours_offset = int(h_str)
    minutes_offset = int(m_str)

    # Create cutoff datetime per Date: Date at 00:00 + offset (microsecond-aware)
    # This ensures 10:00:00.000000 passes the filter, but 10:00:00.500000 does not
    cutoff_expr = (
        pl.col("Date").str.to_datetime("%Y-%m-%d")
        + pl.duration(hours=hours_offset, minutes=minutes_offset)
    )

    # Get first tick per Date (RTH open reference)
    first_ticks = (
        rth.group_by("Date", maintain_order=True)
        .agg([
            pl.col("Datetime").first().alias("first_dt"),
            pl.col("vwap").first().alias("vwap_open"),
        ])
    )

    # Get last tick at or before cutoff per Date
    at_ticks = (
        rth.filter(pl.col("Datetime") <= cutoff_expr)
        .group_by("Date", maintain_order=True)
        .agg([
            pl.col("Datetime").last().alias("at_dt"),
            pl.col("vwap").last().alias("vwap_at"),
        ])
    )

    # Join, compute slope, drop invalid rows
    result = (
        first_ticks.join(at_ticks, on="Date", how="inner")
        .with_columns([
            (
                (pl.col("at_dt") - pl.col("first_dt"))
                .dt.total_microseconds() / 3.6e9
            ).alias("hours"),
        ])
        .filter(pl.col("hours") > 0)
        .with_columns([
            ((pl.col("vwap_at") - pl.col("vwap_open")) / pl.col("hours"))
            .alias("vwap_slope"),
        ])
        .select(["Date", "vwap_slope"])
        .sort("Date")
    )

    return result if result.height > 0 else pl.DataFrame(schema={"Date": pl.String, "vwap_slope": pl.Float64})


def calibrate_slope_thresholds(slopes, *, directional_q: float = 0.70,
                               rotational_q: float = 0.40) -> tuple[float, float]:
    """Quantiles of |VWAP slope| that freeze trend/range thresholds.

    Parameters
    ----------
    slopes : polars.Series or array-like
        VWAP slope values (points/hour), from vwap_slope_at.
    directional_q : float
        Quantile for directional threshold (default 0.70).
    rotational_q : float
        Quantile for rotational threshold (default 0.40).

    Returns
    -------
    tuple[float, float]
        (directional_threshold, rotational_threshold), computed from quantiles of |slopes|.

    Raises
    ------
    ValueError
        If no finite values remain after dropping NaN/null.
    """
    # Convert to numpy, handling Polars Series
    if isinstance(slopes, pl.Series):
        values = slopes.to_numpy()
    else:
        values = np.asarray(slopes, dtype=float)

    # Drop NaN values
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]

    if len(finite_values) == 0:
        raise ValueError("No finite values in slopes")

    # Absolute values and compute quantiles
    abs_values = np.abs(finite_values)
    directional = float(np.quantile(abs_values, directional_q))
    rotational = float(np.quantile(abs_values, rotational_q))

    return directional, rotational


def classify_trend_state(state: pl.DataFrame, *, directional_slope_min: float,
                         rotational_slope_max: float, min_minutes: float = 30) -> pl.Series:
    """Label each running-state row TREND_UP / TREND_DOWN / RANGE / UNKNOWN, as of that tick.

    slope = (vwap - open_vwap) / (minutes_since_open / 60); drift = POC - open_poc.
    c1: open_location in (above, below) and not returned_to_value, direction +1 above / -1 below.
    c2: |slope| >= directional_slope_min, direction sign(slope).
    c3: vwap_crosses <= 1 (no direction).
    c4: sign(drift) == sign(slope) != 0, direction sign(slope).
    TREND if >=3 of c1..c4 hold and every held directional condition (c1, c2, c4) agrees in sign.
    Else RANGE if vwap_crosses >= 2 and |slope| <= rotational_slope_max and
    (open_location == inside or returned_to_value). Else UNKNOWN.
    """
    slope = (pl.col("vwap") - pl.col("open_vwap")) / (pl.col("minutes_since_open") / 60.0)
    drift = pl.col("POC") - pl.col("open_poc")
    slope_sign = pl.when(slope > 0).then(1).when(slope < 0).then(-1).otherwise(0)
    drift_sign = pl.when(drift > 0).then(1).when(drift < 0).then(-1).otherwise(0)

    c1_holds = pl.col("open_location").is_in(["above", "below"]) & ~pl.col("returned_to_value")
    c1_dir = pl.when(pl.col("open_location") == "above").then(1).otherwise(-1)

    c2_holds = slope.abs() >= directional_slope_min
    c2_dir = slope_sign

    c3_holds = pl.col("vwap_crosses") <= 1

    c4_holds = (drift_sign == slope_sign) & (slope_sign != 0)
    c4_dir = slope_sign

    holds_count = (
        c1_holds.cast(pl.Int32) + c2_holds.cast(pl.Int32)
        + c3_holds.cast(pl.Int32) + c4_holds.cast(pl.Int32)
    )

    # Sum of held directional signs vs. count of held directional conditions: equal in
    # magnitude only when every held directional condition agrees.
    dir_sum = (
        pl.when(c1_holds).then(c1_dir).otherwise(0)
        + pl.when(c2_holds).then(c2_dir).otherwise(0)
        + pl.when(c4_holds).then(c4_dir).otherwise(0)
    )
    dir_count = c1_holds.cast(pl.Int32) + c2_holds.cast(pl.Int32) + c4_holds.cast(pl.Int32)

    is_trend = (holds_count >= 3) & (dir_sum.abs() == dir_count) & (dir_count > 0)
    trend_dir = pl.when(dir_sum > 0).then(1).otherwise(-1)

    is_range = (
        (pl.col("vwap_crosses") >= 2) & (slope.abs() <= rotational_slope_max)
        & ((pl.col("open_location") == "inside") | pl.col("returned_to_value"))
    )

    guard_unknown = (pl.col("minutes_since_open") < min_minutes) | (
        pl.col("open_location").is_null() | (pl.col("open_location") == "unknown")
    )

    result = (
        pl.when(guard_unknown).then(pl.lit("UNKNOWN"))
        .when(is_trend).then(pl.when(trend_dir > 0).then(pl.lit("TREND_UP")).otherwise(pl.lit("TREND_DOWN")))
        .when(is_range).then(pl.lit("RANGE"))
        .otherwise(pl.lit("UNKNOWN"))
    )
    return state.select(result.alias("trend_state")).to_series()
