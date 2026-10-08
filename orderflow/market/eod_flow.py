"""End-of-day flow: the day's cumulative delta late in the session, faded into the close.

``TradeType 2`` is ask volume (buy aggression), ``TradeType 1`` is bid volume (sell aggression). The
day's CVD is the buy volume minus the sell volume of the session so far. A day of net buying is
traded short into the close and a day of net selling long: the signal fades the flow.

A signal at ``signal_time`` reads ticks up to and including that time and nothing later.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

EVENT_COLUMNS = ["Date", "signal_index", "signal_datetime", "entry_index", "entry_price", "side", "day_cvd"]


def find_eod_cvd_fade_signals(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    signal_time: str = "14:45:00",
    exit_time: str = "15:00:00",
    min_abs_cvd: float = 0.0,
    session: str = "RTH",
) -> pl.DataFrame:
    """One row per day that has a signal, in date order.

    ``ticks`` needs ``Index``, ``Date``, ``Datetime``, ``SessionType``, ``Price``, ``Volume`` and
    ``TradeType`` (1 or 2). Columns: ``Date``, ``signal_index`` and ``signal_datetime`` (the last
    ``session`` tick at or before ``signal_time``), ``entry_index`` and ``entry_price`` (the first
    tick after ``signal_time`` and before ``exit_time``), ``side`` (+1 long, -1 short: the opposite
    of the sign of the day's CVD) and ``day_cvd`` (buy minus sell volume of the session up to the
    signal tick). A day with a CVD of zero, with ``abs(day_cvd) < min_abs_cvd`` or with no tick to
    enter on has no row.
    """
    signal_at, exit_at = dt.time.fromisoformat(signal_time), dt.time.fromisoformat(exit_time)
    clock = pl.col("Datetime").dt.time()
    before = clock <= signal_at
    after = (clock > signal_at) & (clock < exit_at)
    signed = pl.when(pl.col("TradeType") == 2).then(pl.col("Volume")).otherwise(-pl.col("Volume"))
    days = (
        ticks.lazy()
        .select("Index", "Date", "Datetime", "SessionType", "Price", "Volume", "TradeType")
        .filter(pl.col("SessionType") == session)
        .sort("Index")
        .group_by("Date", maintain_order=True)
        .agg(
            pl.col("Index").filter(before).last().alias("signal_index"),
            pl.col("Datetime").filter(before).last().alias("signal_datetime"),
            pl.col("Index").filter(after).first().alias("entry_index"),
            pl.col("Price").filter(after).first().alias("entry_price"),
            signed.filter(before).sum().alias("day_cvd"),
        )
        .filter(pl.col("signal_index").is_not_null() & pl.col("entry_index").is_not_null()
                & (pl.col("day_cvd") != 0) & (pl.col("day_cvd").abs() >= min_abs_cvd))
        .with_columns(side=pl.when(pl.col("day_cvd") > 0).then(-1).otherwise(1).cast(pl.Int64))
        .sort("Date")
        .collect()
    )
    return days.select(EVENT_COLUMNS)
