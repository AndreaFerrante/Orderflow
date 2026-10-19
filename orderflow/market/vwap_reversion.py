"""VWAP band-touch signals: a fade of the 2 SD band, entered on the next tick.

A price at or below the lower 2 SD band of the VWAP is sold aggression pushed to an extreme
(``TradeType 1``, bid trades) and is traded long; a price at or above the upper band is bought
aggression (``TradeType 2``, ask trades) and is traded short. The signal fades the move.

A signal at tick ``t`` reads ticks up to and including ``t`` and nothing later; the entry is the next
tick of the same day.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

EVENT_COLUMNS = ["Date", "signal_index", "signal_datetime", "entry_index", "entry_price", "side", "stop_loss"]


def _edge(touch: pl.Expr) -> pl.Expr:
    """True where ``touch`` turns on: it is true here and was not true on the previous tick of the day."""
    return touch & ~touch.shift(1).over("Date").fill_null(False)


def find_vwap_band_touch_signals(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    entry_from: str = "09:00:00",
    session: str = "RTH",
) -> pl.DataFrame:
    """One row per signal, in ``signal_index`` order.

    ``ticks`` needs ``Index``, ``Date``, ``Datetime``, ``SessionType``, ``Price``, ``vwap_sd2_top``,
    ``vwap_sd2_bottom``, ``vwap_sd3_top`` and ``vwap_sd3_bottom``. Columns: ``Date``, ``signal_index``
    and ``signal_datetime`` (the touch tick), ``entry_index`` and ``entry_price`` (the next ``session``
    tick of the same ``Date``), ``side`` (+1 long at ``Price <= vwap_sd2_bottom``, -1 short at
    ``Price >= vwap_sd2_top``) and ``stop_loss`` (the 3 SD band on the entry tick: the bottom one for
    a long, the top one for a short).

    A signal is the tick where the touch turns on; the first tick of a day has no previous tick, so it
    counts as turning on. A signal before ``entry_from`` (clock time) is not emitted, and neither is the
    continuation of a touch that began before it. A signal with no later ``session`` tick that day has
    no row.
    """
    entry_at = dt.time.fromisoformat(entry_from)
    clock = pl.col("Datetime").dt.time()
    nxt = ("Index", "Price", "vwap_sd3_bottom", "vwap_sd3_top")
    day = (
        ticks.lazy()
        .filter(pl.col("SessionType") == session)
        .sort("Index")
        .with_columns(pl.col(*nxt).shift(-1).over("Date").name.prefix("next_"))
    )
    sides = [(1, pl.col("Price") <= pl.col("vwap_sd2_bottom"), "next_vwap_sd3_bottom"),
             (-1, pl.col("Price") >= pl.col("vwap_sd2_top"), "next_vwap_sd3_top")]
    events = [
        day.filter(_edge(touch) & (clock >= entry_at)).select(
            "Date",
            pl.col("Index").alias("signal_index"),
            pl.col("Datetime").alias("signal_datetime"),
            pl.col("next_Index").alias("entry_index"),
            pl.col("next_Price").alias("entry_price"),
            pl.lit(side, dtype=pl.Int64).alias("side"),
            pl.col(stop).alias("stop_loss"),
        )
        for side, touch, stop in sides
    ]
    return (
        pl.concat(events)
        .filter(pl.col("entry_index").is_not_null())
        .sort("signal_index")
        .collect()
        .select(EVENT_COLUMNS)
    )
