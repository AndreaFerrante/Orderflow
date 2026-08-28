"""Initial balance: the range established in the first N minutes of RTH.

The initial balance is the auction's opening agreement on value.  Price
leaving it is an *initial balance extension* -- the session has failed to
contain price and is searching for a new area of acceptance.

Every calculation here is strictly causal: nothing reads a tick later than
the one it labels.
"""

from __future__ import annotations

import warnings

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


_BREAKOUT_SCHEMA = {
    "Date": pl.Utf8,
    "signal_index": pl.Int64,
    "entry_index": pl.Int64,
    "direction": pl.Int64,
    "cvd_delta": pl.Float64,
    "ib_high": pl.Float64,
    "ib_low": pl.Float64,
    "ib_range_ticks": pl.Float64,
}

_BREAKOUT_REQUIRED_COLUMNS = (
    "Index", "Datetime", "Date", "Price", "SessionType", "CD_Ask", "CD_Bid",
)


def find_ib_breakouts(
    ticks: pl.DataFrame,
    ib: pl.DataFrame,
    cvd_lookback_ticks: int = 2000,
    cvd_min_delta: float = 0.0,
    entry_cutoff_hour: int = 14,
) -> pl.DataFrame:
    """First initial-balance break per direction per session, gated on flow.

    Parameters
    ----------
    ticks
        Enriched tick frame.  Requires the columns in
        ``_BREAKOUT_REQUIRED_COLUMNS``.
    ib
        Output of :func:`compute_initial_balance`.
    cvd_lookback_ticks
        How many ticks back the cumulative-delta difference is measured.
    cvd_min_delta
        Magnitude the difference must exceed, in contracts.  ``0.0`` gates on
        sign alone.
    entry_cutoff_hour
        Breaks at or after this hour are dropped.

    Returns
    -------
    pl.DataFrame
        ``Date``, ``signal_index``, ``entry_index``, ``direction``,
        ``cvd_delta``, ``ib_high``, ``ib_low``, ``ib_range_ticks``.
        ``direction`` is ``+1`` above the initial balance, ``-1`` below.

    Notes
    -----
    **The gate is a difference, never a level.**  ``CD_Ask``/``CD_Bid`` in the
    enriched files accumulate across the whole file and do *not* reset at the
    RTH boundary, so their absolute value at a break carries no information
    about that session.  ``cvd_delta`` is
    ``(CD_Ask - CD_Bid)`` now minus its value ``cvd_lookback_ticks`` rows ago,
    computed within the session (``.over("Date")``), and is buy-positive by
    construction.

    The signal is the first tick that *both* breaks the edge and passes the
    gate -- a break that fails the gate does not consume the session's one
    signal for that direction.

    ``entry_index`` is the next tick.  Reading it is the single permitted
    forward look: a fill cannot happen on the signal tick itself.
    """
    missing = [c for c in _BREAKOUT_REQUIRED_COLUMNS if c not in ticks.columns]
    if missing:
        raise ValueError(
            f"ticks is missing {missing}; find_ib_breakouts requires "
            f"{list(_BREAKOUT_REQUIRED_COLUMNS)}"
        )
    if ib.height == 0:
        return pl.DataFrame(schema=_BREAKOUT_SCHEMA)

    rth = (
        ticks.filter(pl.col("SessionType") == "RTH")
        .sort("Index")
        .with_columns((pl.col("CD_Ask") - pl.col("CD_Bid")).alias("_cvd"))
    )
    rth = rth.with_columns(
        (pl.col("_cvd") - pl.col("_cvd").shift(cvd_lookback_ticks).over("Date"))
        .alias("cvd_delta"),
        pl.col("Index").shift(-1).over("Date").alias("entry_index"),
    )

    post_ib = rth.join(ib, on="Date", how="inner").filter(
        pl.col("Index") > pl.col("ib_end_index")
    )

    n_missing = (
        post_ib.group_by("Date")
        .agg(pl.col("cvd_delta").is_not_null().any().alias("has_cvd_delta"))
        .filter(~pl.col("has_cvd_delta"))
        .height
    )
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} session(s) produced no cvd_delta before their "
            f"first initial-balance break: fewer than "
            f"cvd_lookback_ticks={cvd_lookback_ticks} ticks of session "
            f"history were available, so any break in those sessions was "
            f"dropped. Lower cvd_lookback_ticks to include them."
        )

    qualified = (
        post_ib
        .filter(pl.col("Datetime").dt.hour() < entry_cutoff_hour)
        .filter(pl.col("cvd_delta").is_not_null())
        .filter(pl.col("entry_index").is_not_null())
        .with_columns(
            pl.when(pl.col("Price") > pl.col("ib_high"))
            .then(1)
            .when(pl.col("Price") < pl.col("ib_low"))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("direction")
        )
        .filter(pl.col("direction") != 0)
        .filter(
            ((pl.col("direction") == 1) & (pl.col("cvd_delta") > cvd_min_delta))
            | ((pl.col("direction") == -1) & (pl.col("cvd_delta") < -cvd_min_delta))
        )
    )

    if qualified.height == 0:
        return pl.DataFrame(schema=_BREAKOUT_SCHEMA)

    return (
        qualified.sort("Index")
        .group_by(["Date", "direction"], maintain_order=True)
        .first()
        .with_columns(pl.col("Index").alias("signal_index"))
        .select(list(_BREAKOUT_SCHEMA))
        .sort("signal_index")
    )
