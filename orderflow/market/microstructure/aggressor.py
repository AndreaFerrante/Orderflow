"""Aggressor order reconstruction from tick runs.

A single aggressive order sweeping several price levels prints as *N* separate
fills on CME MDP 3.0. Sierra Chart writes those fills with strictly increasing
timestamps, breaking ties by incrementing microseconds, so a run of consecutive
ticks each ``gap_us`` apart or less is one match event -- one aggressor order.

Recovering that grouping is what makes order-level microstructure measurable on
trade-conditional data: aggressor size, sweep breadth, and the signed flow that
price impact is measured against.

Measured on MES tick data (2,000,000-tick samples, four files spanning
2024-12-22 to 2025-12-12): 57.5% of consecutive timestamp differences are exactly
1 microsecond, grouping yields ~2.35 fills per order, and 96.4% of raw tie-break
groups are already single-sided. Splitting on ``TradeType`` as well removes the
remainder, which are two aggressors landing in the same match event.

Caveat that must travel with these functions: because ties are broken
*artificially*, the 1-microsecond spacing is a grouping key, not a true
inter-arrival time. Do not derive latency or arrival-rate features from it.

Conventions
-----------
``TradeType`` follows Sierra Chart: ``2`` is an ask trade (aggressive buy),
``1`` is a bid trade (aggressive sell).

All functions are strictly causal and prefix-invariant: computing on the input
truncated at row *k* reproduces every complete order before that point.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["assign_aggressor_order_ids", "build_aggressor_orders", "flag_sweeps"]


_REQUIRED_COLUMNS = ("Index", "Datetime", "Price", "Volume", "TradeType")

_ORDER_SCHEMA = {
    "aggressor_id": pl.Int64,
    "aggressor_side": pl.Int64,
    "aggressor_size": pl.Float64,
    "aggressor_levels": pl.Int64,
    "aggressor_n_fills": pl.Int64,
    "aggressor_price_start": pl.Float64,
    "aggressor_price_end": pl.Float64,
    "aggressor_vwap": pl.Float64,
    "aggressor_first_index": pl.Int64,
    "aggressor_last_index": pl.Int64,
}


def _validate(ticks: pl.DataFrame, gap_us: int) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in ticks.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(gap_us, (int, np.integer)) or gap_us < 0:
        raise ValueError(f"gap_us must be a non-negative integer, got {gap_us!r}")


def assign_aggressor_order_ids(
    ticks: pl.DataFrame,
    *,
    gap_us: int = 1,
) -> pl.DataFrame:
    """Label each tick with the aggressor order it belongs to.

    A new order starts when either the gap to the previous tick exceeds
    ``gap_us`` microseconds, or ``TradeType`` changes. The second condition is
    what guarantees every emitted order is single-sided.

    Parameters
    ----------
    ticks : pl.DataFrame
        Tick data carrying ``Index``, ``Datetime``, ``Price``, ``Volume`` and
        ``TradeType``, in chronological order.
    gap_us : int, default 1
        Maximum microsecond gap that still counts as the same match event. The
        default matches Sierra Chart's tie-break increment.

    Returns
    -------
    pl.DataFrame
        The input with an added ``aggressor_id`` column, contiguous from zero.
        Row order is preserved and the input frame is not mutated.
    """
    _validate(ticks, gap_us)

    if ticks.height == 0:
        return ticks.with_columns(pl.Series("aggressor_id", [], dtype=pl.Int64))

    dt = ticks["Datetime"].to_numpy().astype("datetime64[us]").astype("int64")
    tt = ticks["TradeType"].to_numpy()

    starts = np.empty(len(dt), dtype=bool)
    starts[0] = False
    starts[1:] = (np.diff(dt) > gap_us) | (tt[1:] != tt[:-1])

    return ticks.with_columns(
        pl.Series("aggressor_id", np.cumsum(starts), dtype=pl.Int64)
    )


def build_aggressor_orders(
    ticks: pl.DataFrame,
    *,
    gap_us: int = 1,
    tick_size: float,
) -> pl.DataFrame:
    """Aggregate ticks into one row per reconstructed aggressor order.

    No size filter is applied, deliberately. Stealth trading splits informed
    orders into medium-size clips precisely to avoid a size signature, and the
    largest prints are disproportionately uninformed liquidity demand. Size
    enters the strategy only through a time-of-day z-score and through realized
    price impact, never as a raw threshold.

    Parameters
    ----------
    ticks : pl.DataFrame
        Tick data, see :func:`assign_aggressor_order_ids`.
    gap_us : int, default 1
        Grouping gap in microseconds.
    tick_size : float
        Instrument minimum price increment, used to convert the order's price
        span into a level count. Required -- there is no default, because a
        wrong tick size silently mis-counts swept levels.

    Returns
    -------
    pl.DataFrame
        One row per order, ordered by ``aggressor_id``:

        ``aggressor_side``
            The group's single ``TradeType``: 2 aggressive buy, 1 aggressive sell.
        ``aggressor_size``
            Total volume executed.
        ``aggressor_levels``
            Distinct price levels spanned, ``round(span / tick_size) + 1``.
        ``aggressor_n_fills``
            Number of source ticks.
        ``aggressor_price_start`` / ``aggressor_price_end``
            First and last fill price, in fill order -- **not** min and max, so
            the direction of a sweep survives.
        ``aggressor_vwap``
            Volume-weighted mean fill price.
        ``aggressor_first_index`` / ``aggressor_last_index``
            ``Index`` bounds, for causal joins back to the tick frame.
    """
    _validate(ticks, gap_us)
    if not isinstance(tick_size, (int, float)) or tick_size <= 0:
        raise ValueError(f"tick_size must be a positive number, got {tick_size!r}")

    if ticks.height == 0:
        return pl.DataFrame(schema=_ORDER_SCHEMA)

    labelled = assign_aggressor_order_ids(ticks, gap_us=gap_us)

    return (
        labelled.group_by("aggressor_id", maintain_order=True)
        .agg(
            pl.col("TradeType").first().alias("aggressor_side"),
            pl.col("Volume").sum().cast(pl.Float64).alias("aggressor_size"),
            pl.col("Price").min().alias("_price_min"),
            pl.col("Price").max().alias("_price_max"),
            pl.len().cast(pl.Int64).alias("aggressor_n_fills"),
            pl.col("Price").first().alias("aggressor_price_start"),
            pl.col("Price").last().alias("aggressor_price_end"),
            (pl.col("Price") * pl.col("Volume")).sum().alias("_notional"),
            pl.col("Index").first().alias("aggressor_first_index"),
            pl.col("Index").last().alias("aggressor_last_index"),
        )
        .with_columns(
            (
                ((pl.col("_price_max") - pl.col("_price_min")) / tick_size)
                .round(0)
                .cast(pl.Int64)
                + 1
            ).alias("aggressor_levels"),
            (pl.col("_notional") / pl.col("aggressor_size")).alias("aggressor_vwap"),
        )
        .drop("_price_min", "_price_max", "_notional")
        .select(list(_ORDER_SCHEMA))
    )


def flag_sweeps(orders: pl.DataFrame, *, min_levels: int = 3) -> pl.DataFrame:
    """Mark aggressor orders that consumed several price levels in one direction.

    Aggressive volume printing at three or more distinct levels within one match
    event is almost certainly a single aggressor order clearing the book, which
    is the momentum trigger. The direction check is not decoration: an aggressive
    buy whose last fill sits *below* its first is a data artefact or a mixed
    group, and trading it would take the position backwards.

    Parameters
    ----------
    orders : pl.DataFrame
        Output of :func:`build_aggressor_orders`.
    min_levels : int, default 3
        Minimum distinct price levels for a sweep.

    Returns
    -------
    pl.DataFrame
        The input with ``is_sweep`` (bool) and ``sweep_direction``
        (``+1`` up, ``-1`` down, ``0`` not a sweep) appended.
    """
    if not isinstance(min_levels, (int, np.integer)) or min_levels < 2:
        raise ValueError(f"min_levels must be an integer >= 2, got {min_levels!r}")

    required = ("aggressor_side", "aggressor_levels",
                "aggressor_price_start", "aggressor_price_end")
    missing = [c for c in required if c not in orders.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if orders.height == 0:
        return orders.with_columns(
            pl.Series("is_sweep", [], dtype=pl.Boolean),
            pl.Series("sweep_direction", [], dtype=pl.Int64),
        )

    up = (pl.col("aggressor_side") == 2) & (
        pl.col("aggressor_price_end") > pl.col("aggressor_price_start")
    )
    down = (pl.col("aggressor_side") == 1) & (
        pl.col("aggressor_price_end") < pl.col("aggressor_price_start")
    )
    wide = pl.col("aggressor_levels") >= min_levels

    return orders.with_columns(
        (wide & (up | down)).alias("is_sweep"),
    ).with_columns(
        pl.when(pl.col("is_sweep") & up)
        .then(1)
        .when(pl.col("is_sweep") & down)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("sweep_direction")
    )
