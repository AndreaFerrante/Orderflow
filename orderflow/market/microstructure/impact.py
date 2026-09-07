"""Price impact measured from trades and quotes.

Kyle's lambda is the coefficient linking signed order flow to price movement:
how far the market moves per contract of one-sided aggression. Estimating it
usually requires inferring which side lifted, but tick data carrying the quote
at execution gives the aggressor exactly -- ``TradeType`` says which side was
hit -- so lambda can be measured rather than modelled.

What it is for:

* **Low lambda under heavy flow** means volume arrived and price did not move.
  Something is absorbing it, which is the fade setup.
* **High lambda** means a thin book and informed flow. Never fade it; this is
  also the structural exit when a position is already open.

Lambda is only meaningful **conditional on the flow percentile**. A low reading
during trivial volume says nothing at all -- price did not move because nothing
happened. Bucket by a flow z-score first, then take the quantile of lambda
within the bucket.

Window convention
-----------------
Half-open, ``(t - window, t]``. The reference price is the mid at the last tick
at or before ``t - window``; the signed volume covers everything strictly after
that tick through ``t``. A trade landing exactly on the boundary happened before
the interval opened and its effect is already inside the reference mid, so
counting it in the volume as well would attribute one trade's impact twice.

Missing values are ``null``, never ``NaN`` -- see ``book.py``'s note. Polars
orders ``NaN`` above every float, so a ``NaN`` lambda would *pass* the
top-quantile informed-flow filter rather than dropping out of it.

Conventions
-----------
``TradeType`` follows Sierra Chart: ``2`` is an ask trade (aggressive buy,
positive signed volume), ``1`` is a bid trade (aggressive sell, negative). A sign
error here does not raise -- it inverts every reading in the file.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["compute_realized_lambda", "compute_absorption_ratio"]


_REQUIRED_COLUMNS = ("Datetime", "AskPrice", "BidPrice", "Volume", "TradeType")

_MICROSECONDS_PER_SECOND = 1_000_000


def compute_realized_lambda(
    ticks: pl.DataFrame,
    *,
    window_seconds: float,
    min_abs_signed_volume: float = 1.0,
) -> pl.DataFrame:
    """Measure price impact per unit of signed aggressive volume.

    ::

        mid_t                = (AskPrice_t + BidPrice_t) / 2
        reference            = last tick at or before t - window_seconds
        signed_volume_window = sum over (reference, t] of
                               Volume * (+1 if TradeType == 2 else -1)
        realized_lambda      = (mid_t - mid_reference) / signed_volume_window

    Parameters
    ----------
    ticks : pl.DataFrame
        Tick data carrying ``Datetime``, ``AskPrice``, ``BidPrice``, ``Volume``
        and ``TradeType``, in chronological order.
    window_seconds : float
        Lookback for both the price change and the flow that caused it.
        Required -- the value determines what "impact" means here, so there is
        no sensible default.
    min_abs_signed_volume : float, default 1.0
        Floor on ``|signed_volume_window|``. Below it the denominator is too
        close to zero for the quotient to mean anything.

    Returns
    -------
    pl.DataFrame
        The input with ``mid``, ``signed_volume_window`` and ``realized_lambda``
        appended. ``realized_lambda`` is **null** when either:

        * no tick exists at or before ``t - window_seconds``, so the window is
          not yet full; or
        * ``|signed_volume_window| < min_abs_signed_volume``. Buying and selling
          that cancel out leave a near-zero denominator, and dividing by it
          produces an enormous lambda that reads as maximal informed flow --
          the opposite of what balanced two-way trade means.

    Raises
    ------
    ValueError
        Missing columns, non-positive ``window_seconds``, negative
        ``min_abs_signed_volume``, or non-monotonic ``Datetime``.
    """
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        raise ValueError(
            f"window_seconds must be a positive number, got {window_seconds!r}"
        )
    if not isinstance(min_abs_signed_volume, (int, float)) or min_abs_signed_volume < 0:
        raise ValueError(
            f"min_abs_signed_volume must be non-negative, "
            f"got {min_abs_signed_volume!r}"
        )

    missing = [c for c in _REQUIRED_COLUMNS if c not in ticks.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if ticks.height == 0:
        return ticks.with_columns(
            pl.Series("mid", [], dtype=pl.Float64),
            pl.Series("signed_volume_window", [], dtype=pl.Float64),
            pl.Series("realized_lambda", [], dtype=pl.Float64),
        )

    timestamps = (
        ticks["Datetime"].to_numpy().astype("datetime64[us]").astype("int64")
    )
    if timestamps.size > 1 and np.diff(timestamps).min() < 0:
        raise ValueError(
            "Datetime must be non-decreasing; the input is out of chronological "
            "order and every window would be measured against the wrong reference"
        )

    mid = (
        ticks["AskPrice"].to_numpy().astype(np.float64)
        + ticks["BidPrice"].to_numpy().astype(np.float64)
    ) / 2.0

    signed = ticks["Volume"].to_numpy().astype(np.float64) * np.where(
        ticks["TradeType"].to_numpy() == 2, 1.0, -1.0
    )

    # Index of the last tick at or before t - window, per row. -1 means no such
    # tick exists and the window has not filled yet.
    window_us = int(round(window_seconds * _MICROSECONDS_PER_SECOND))
    reference = np.searchsorted(timestamps, timestamps - window_us, side="right") - 1
    warm = reference < 0

    # Signed volume over (reference, t] is the difference of a running total, so
    # the whole column is one cumulative sum rather than a window per row.
    # A leading zero lets reference == -1 index harmlessly during the arithmetic;
    # those rows are discarded by `warm` immediately afterwards.
    cumulative = np.concatenate([[0.0], np.cumsum(signed)])
    signed_window = cumulative[np.arange(len(signed)) + 1] - cumulative[reference + 1]

    reference_mid = mid[np.where(warm, 0, reference)]
    usable = ~warm & (np.abs(signed_window) >= min_abs_signed_volume)

    realized = np.full(len(mid), np.nan, dtype=np.float64)
    np.divide(mid - reference_mid, signed_window, out=realized, where=usable)

    return ticks.with_columns(
        pl.Series("mid", mid, dtype=pl.Float64),
        pl.Series("signed_volume_window", signed_window, dtype=pl.Float64),
        pl.Series("realized_lambda", realized, dtype=pl.Float64).fill_nan(None),
    )


_ABSORPTION_COLUMNS = (
    "aggressor_size",
    "aggressor_price_start",
    "aggressor_price_end",
)


def compute_absorption_ratio(
    orders: pl.DataFrame,
    *,
    sigma: pl.Series,
    session_volume_to_date: pl.Series,
    impact_constant: float,
) -> pl.DataFrame:
    """Compare how far price moved against how far the square-root law says it should.

    Price impact scales roughly with the square root of order size relative to
    the volume already traded -- the most robust empirical regularity in market
    microstructure. That turns absorption from a judgement call into a
    measurement::

        expected_move    = impact_constant * sigma * sqrt(size / session_volume)
        realized_move    = |price_end - price_start|
        absorption_ratio = realized_move / expected_move

    A ratio well below 1 means price moved materially less than the volume
    justifies: something took the other side without repricing. A ratio above 1
    means the move ran past what the size explains, which is the opposite
    situation and must not be faded.

    The advantage over a percentile threshold is that the scale is
    interpretable. "Price moved a quarter of what this much volume normally
    moves it" survives a change of instrument, session length or volatility
    regime; "bottom decile of refresh ratio" does not.

    Parameters
    ----------
    orders : pl.DataFrame
        Aggressor orders from :func:`build_aggressor_orders`, carrying
        ``aggressor_size``, ``aggressor_price_start`` and
        ``aggressor_price_end``.
    sigma : pl.Series
        Causal volatility estimate, one value per order, in price units.
    session_volume_to_date : pl.Series
        Cumulative volume traded this session **up to** each order. Never the
        session total -- that is lookahead, and it biases every early-session
        order toward looking absorbed.
    impact_constant : float
        The law's scale factor. Fit it once on a held-out calibration period and
        then freeze it: it sets where "materially below 1" falls, so refitting
        it per run turns the threshold into a free parameter.

    Returns
    -------
    pl.DataFrame
        The input with ``expected_move``, ``realized_move`` and
        ``absorption_ratio`` appended. ``absorption_ratio`` is **null** wherever
        ``expected_move`` is not strictly positive -- zero or null ``sigma``,
        zero size, or no session volume yet. Those are undecidable rather than
        infinitely absorbed, and per the module note the sentinel is null and
        not NaN so they drop out of comparisons instead of passing them.
    """
    if not isinstance(impact_constant, (int, float)) or impact_constant <= 0:
        raise ValueError(
            f"impact_constant must be a positive number, got {impact_constant!r}"
        )

    missing = [c for c in _ABSORPTION_COLUMNS if c not in orders.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # A misaligned series would pair each order with another order's volatility,
    # which produces plausible numbers and no error at all.
    for name, series in (("sigma", sigma),
                         ("session_volume_to_date", session_volume_to_date)):
        if len(series) != orders.height:
            raise ValueError(
                f"{name} length {len(series)} does not match orders height "
                f"{orders.height}"
            )

    sigma_values = pl.Series(sigma, dtype=pl.Float64).to_numpy(allow_copy=True)
    volume_values = pl.Series(session_volume_to_date, dtype=pl.Float64).to_numpy(
        allow_copy=True
    )
    size_values = orders["aggressor_size"].cast(pl.Float64).to_numpy()

    # Guard the inputs, not the result. Polars divides by zero to `inf` rather
    # than NaN, and `inf > 0` is True -- so a result-side guard lets a
    # zero-volume row through with expected_move = inf, whereupon
    # realized_move / inf is 0.0 and the order reads as *maximally absorbed*.
    # For an absorption measure that is the most dangerous wrong answer there is.
    decidable = (
        np.isfinite(sigma_values)
        & (sigma_values > 0)
        & (size_values > 0)
        & (volume_values > 0)
    )

    expected = np.full(orders.height, np.nan, dtype=np.float64)
    expected[decidable] = (
        impact_constant
        * sigma_values[decidable]
        * np.sqrt(size_values[decidable] / volume_values[decidable])
    )

    realized = np.abs(
        orders["aggressor_price_end"].cast(pl.Float64).to_numpy()
        - orders["aggressor_price_start"].cast(pl.Float64).to_numpy()
    )

    ratio = np.full(orders.height, np.nan, dtype=np.float64)
    np.divide(realized, expected, out=ratio, where=decidable)

    return orders.with_columns(
        pl.Series("expected_move", expected, dtype=pl.Float64).fill_nan(None),
        pl.Series("realized_move", realized, dtype=pl.Float64),
        # null, not NaN -- see the module note
        pl.Series("absorption_ratio", ratio, dtype=pl.Float64).fill_nan(None),
    )
