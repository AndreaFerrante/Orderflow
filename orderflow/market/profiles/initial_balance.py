"""Initial balance: the range established in the first N minutes of RTH.

The initial balance is the auction's opening agreement on value.  Price
leaving it is an *initial balance extension* -- the session has failed to
contain price and is searching for a new area of acceptance.

Every calculation here is strictly causal: nothing reads a tick later than
the one it labels.
"""

from __future__ import annotations

import polars as pl

_IB_SCHEMA = {
    "Date": pl.Utf8,
    "ib_high": pl.Float64,
    "ib_low": pl.Float64,
    "ib_range_ticks": pl.Float64,
    "ib_end_index": pl.Int64,
}

_REQUIRED_COLUMNS = ("Index", "Datetime", "Date", "Price", "SessionType")


def compute_initial_balance(
    ticks: pl.DataFrame,
    window_minutes: int = 30,
    tick_size: float = 0.25,
) -> pl.DataFrame:
    """One row per RTH session describing its initial balance.

    Parameters
    ----------
    ticks
        Enriched tick frame.  Requires ``Index``, ``Datetime``, ``Date``,
        ``Price`` and ``SessionType``.
    window_minutes
        Length of the initial balance window, measured from each session's
        own first RTH tick.
    tick_size
        Instrument minimum price increment, used only to express the range
        in ticks.

    Returns
    -------
    pl.DataFrame
        ``Date``, ``ib_high``, ``ib_low``, ``ib_range_ticks``,
        ``ib_end_index``, sorted by ``Date``.

    Notes
    -----
    The window is anchored on the session's *own* first RTH tick rather than
    a hardcoded clock time, so a half day or a late open produces a correct
    initial balance instead of a silently truncated one.

    Sessions whose total RTH span is shorter than ``window_minutes`` are
    dropped rather than clipped: a partial initial balance is not an initial
    balance, and silently returning one would understate the range on
    exactly the abnormal days that most need flagging.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in ticks.columns]
    if missing:
        raise ValueError(
            f"ticks is missing {missing}; compute_initial_balance requires "
            f"{list(_REQUIRED_COLUMNS)}"
        )

    rth = ticks.filter(pl.col("SessionType") == "RTH")
    if rth.height == 0:
        return pl.DataFrame(schema=_IB_SCHEMA)

    window = pl.duration(minutes=window_minutes)

    spans = rth.group_by("Date").agg(
        pl.col("Datetime").min().alias("session_open"),
        pl.col("Datetime").max().alias("session_last"),
    )

    return (
        rth.join(spans, on="Date", how="inner")
        .filter((pl.col("session_last") - pl.col("session_open")) >= window)
        .filter(pl.col("Datetime") < pl.col("session_open") + window)
        .group_by("Date")
        .agg(
            pl.col("Price").max().alias("ib_high"),
            pl.col("Price").min().alias("ib_low"),
            pl.col("Index").max().alias("ib_end_index"),
        )
        .with_columns(
            ((pl.col("ib_high") - pl.col("ib_low")) / tick_size).alias("ib_range_ticks")
        )
        .select(list(_IB_SCHEMA))
        .sort("Date")
    )
