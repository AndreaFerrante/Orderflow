"""VWAP band-touch signals: a fade of the 2 SD band, entered on the next tick.

A price at or below the lower 2 SD band of the VWAP is sold aggression pushed to an extreme
(``TradeType 1``, bid trades) and is traded long; a price at or above the upper band is bought
aggression (``TradeType 2``, ask trades) and is traded short. The signal fades the move.

A signal at tick ``t`` reads ticks up to and including ``t`` and nothing later; the entry is the next
tick of the same day. A signal whose entry tick is the day's last tick has no row: the backtest engine
never evaluates exits on the entry tick, so that position could not be flattened by the close.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

EVENT_COLUMNS = ["Date", "signal_index", "signal_datetime", "entry_index", "entry_price", "side", "stop_loss"]


def _edge(touch: pl.Expr) -> pl.Expr:
    """True where ``touch`` turns on: it is true here and was not true on the previous tick of the day."""
    return touch & ~touch.shift(1).over("Date").fill_null(False)


def _session_ticks(ticks: pl.DataFrame | pl.LazyFrame, session: str, by: str | list[str]) -> pl.LazyFrame:
    """The ``session`` ticks of ``ticks``, sorted by ``by``."""
    return ticks.lazy().filter(pl.col("SessionType") == session).sort(by)


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
    continuation of a touch that began before it. A signal has no row unless the entry tick is followed
    by another ``session`` tick that day, so a position can always be flattened by the close.
    """
    entry_at = dt.time.fromisoformat(entry_from)
    clock = pl.col("Datetime").dt.time()
    nxt = ("Index", "Price", "vwap_sd3_bottom", "vwap_sd3_top")
    day = (
        _session_ticks(ticks, session, "Index")
        .with_columns(pl.col(*nxt).shift(-1).over("Date").name.prefix("next_"),
                      pl.col("Index").shift(-2).over("Date").alias("after_entry_Index"))
    )
    can_flatten = pl.col("after_entry_Index").is_not_null()  # the entry tick is not the day's last
    sides = [(1, pl.col("Price") <= pl.col("vwap_sd2_bottom"), "next_vwap_sd3_bottom"),
             (-1, pl.col("Price") >= pl.col("vwap_sd2_top"), "next_vwap_sd3_top")]
    events = [
        day.filter(_edge(touch) & (clock >= entry_at) & can_flatten).select(
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
        .sort("signal_index")
        .collect()
        .select(EVENT_COLUMNS)
    )


SESSION_COLUMNS = ["Date", "clean", "reason", "max_gap_minutes"]


def _since_midnight(clock: str) -> dt.timedelta:
    t = dt.time.fromisoformat(clock)
    return dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


def _previous_weekday(date: pl.Expr) -> pl.Expr:
    """The ``YYYY-MM-DD`` string of the weekday (Monday to Friday) before the ``YYYY-MM-DD`` string ``date``."""
    day = date.str.to_date()
    back = pl.when(day.dt.weekday() == 1).then(3).when(day.dt.weekday() == 7).then(2).otherwise(1)
    return (day - pl.duration(days=back)).dt.strftime("%Y-%m-%d")


def find_clean_sessions(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    session: str = "RTH",
    session_open: str = "08:00:00",
    session_close: str = "13:30:00",
    max_gap_minutes: float = 10.0,
    anchor_time: str = "13:30:00",
) -> pl.DataFrame:
    """One row per ``Date`` with ``session`` ticks, sorted by ``Date``: is its VWAP trustworthy?

    ``ticks`` needs ``Date``, ``Datetime``, ``SessionType`` and ``Session_Volume``. Columns: ``Date``,
    ``clean``, ``reason`` and ``max_gap_minutes`` (the largest gap between consecutive ``session`` ticks
    of the day; a day with one tick has 0.0). ``reason`` is the first failing check, in this order:
    ``gap`` (``max_gap_minutes`` above the threshold), ``late_start`` (the first tick is more than
    ``max_gap_minutes`` after ``session_open``), ``early_end`` (the last tick is more than
    ``max_gap_minutes`` before ``session_close``), ``anchor``, else ``ok``. ``session_open`` and
    ``session_close`` are clock times on the calendar day of the day's first tick.

    ``anchor``: the VWAP must restart at ``anchor_time``. A reset is a tick whose ``Session_Volume`` is
    below the previous tick's, over all ticks in tape order (``Datetime``, then ``Index`` when the frame
    has one, so ticks sharing a ``Datetime`` keep their order). Of the resets strictly before the day's first ``session``
    tick, the last one must fall from ``anchor_time`` up to one minute later, on the previous weekday
    (Monday to Friday; a Monday looks back to Friday), whether or not that weekday has ``session``
    ticks. No such reset fails, and so does a day after a missing weekday.
    """
    volume = pl.col("Session_Volume")
    tape = ["Datetime", "Index"] if "Index" in ticks.lazy().collect_schema().names() else ["Datetime"]
    resets = ticks.lazy().sort(tape).filter(volume < volume.shift(1)).select(
        pl.col("Datetime").alias("reset_at"), pl.col("Date").alias("reset_date")
    )
    days = (
        _session_ticks(ticks, session, tape)
        .group_by("Date", maintain_order=True)
        .agg(
            pl.col("Datetime").min().alias("first"),
            pl.col("Datetime").max().alias("last"),
            (pl.col("Datetime").diff().max().dt.total_microseconds() / 60_000_000).fill_null(0.0).alias("max_gap_minutes"),
        )
        .sort("first")
        .with_columns(_previous_weekday(pl.col("Date")).alias("prev_date"))
        .join_asof(resets, left_on="first", right_on="reset_at", allow_exact_matches=False)
        .sort("Date")
    )
    midnight = pl.col("first").dt.truncate("1d")
    tolerance = dt.timedelta(minutes=max_gap_minutes)
    anchor_from = _since_midnight(anchor_time)
    reset_clock = pl.col("reset_at") - pl.col("reset_at").dt.truncate("1d")
    anchored = (
        (reset_clock >= anchor_from)
        & (reset_clock < anchor_from + dt.timedelta(minutes=1))
        & (pl.col("reset_date") == pl.col("prev_date"))
    )
    reason = (
        pl.when(pl.col("max_gap_minutes") > max_gap_minutes).then(pl.lit("gap"))
        .when(pl.col("first") - (midnight + _since_midnight(session_open)) > tolerance).then(pl.lit("late_start"))
        .when((midnight + _since_midnight(session_close)) - pl.col("last") > tolerance).then(pl.lit("early_end"))
        .when(~anchored.fill_null(False)).then(pl.lit("anchor"))
        .otherwise(pl.lit("ok"))
    )
    return days.select("Date", (reason == "ok").alias("clean"), reason.alias("reason"), "max_gap_minutes").collect().select(
        SESSION_COLUMNS
    )


def prior_session_range(ticks: pl.DataFrame | pl.LazyFrame, *, session: str = "RTH") -> pl.DataFrame:
    """One row per ``Date`` with ``session`` ticks, sorted by ``Date``: the previous such ``Date``'s range.

    ``ticks`` needs ``Date``, ``SessionType`` and ``Price``. ``prior_day_range`` is the previous
    ``Date``'s ``max(Price) - min(Price)`` over its ``session`` ticks; the first ``Date`` has null.
    """
    return (
        _session_ticks(ticks, session, "Date")
        .group_by("Date")
        .agg((pl.col("Price").max() - pl.col("Price").min()).alias("range"))
        .sort("Date")
        .select("Date", pl.col("range").shift(1).alias("prior_day_range"))
        .collect()
    )
