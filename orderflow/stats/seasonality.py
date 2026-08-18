"""Time-of-day standardisation for intraday features.

Intraday activity is U-shaped: heavy at the open, thin through midday, heavy
again into the close. On MES the 13:00 CT hour carries about 45% of the 09:00
hour's volume. Any absolute threshold calibrated across a whole session
therefore misfires depending on when it is applied -- a flow burst that is
unremarkable at the open is a tail event at 12:45.

The fix is to express every magnitude-valued trigger as a z-score against its
own time-of-day bucket, so "unusually large" means unusually large *for this
time of day*.

Order of operations
-------------------
Normalise by depth first, then z-score. The same aggressive volume moves price
further when the book is thin, and the afternoon book is thinner than the
morning one, so a z-score of raw flow measures the wrong quantity: it would flag
ordinary afternoon volume as extreme purely because the denominator shrank.

Causality
---------
The reference distribution for a session is built from **prior completed
sessions only**. Every row in a session shares one reference, so an expanding
window *within* the day is still leakage. This is the single easiest place in an
intraday pipeline to introduce lookahead, and it does not announce itself: the
result is a plausible, slightly-too-good backtest with no error and no obviously
wrong number.

Missing values are ``null``, never ``NaN`` -- see ``book.py``'s note. Polars
orders ``NaN`` above every float, so a ``NaN`` z-score would *pass* a
``>= threshold`` burst filter rather than dropping out of it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["bucket_of_day", "causal_bucket_zscore", "normalize_by_depth"]


_MICROSECONDS_PER_MINUTE = 60_000_000


def bucket_of_day(
    datetimes: pl.Series,
    *,
    session_open_minutes: int,
    bucket_minutes: int,
) -> pl.Series:
    """Assign each timestamp to a fixed bucket measured from the session open.

    Parameters
    ----------
    datetimes : pl.Series
        Timestamps, in the same timezone convention as ``session_open_minutes``.
    session_open_minutes : int
        Minutes past midnight at which the session opens. MES RTH is 08:30 CT,
        so 510.
    bucket_minutes : int
        Bucket width. Narrow buckets track the intraday shape more closely but
        leave fewer observations per bucket for the reference distribution.

    Returns
    -------
    pl.Series
        Integer bucket index. **Negative before the session open**, deliberately:
        pre-open ticks are out of range rather than members of bucket 0, and a
        silent fold would contaminate the first bucket's reference with
        overnight activity.
    """
    if isinstance(bucket_minutes, bool) or not isinstance(
        bucket_minutes, (int, np.integer)
    ) or bucket_minutes <= 0:
        raise ValueError(
            f"bucket_minutes must be a positive integer, got {bucket_minutes!r}"
        )

    if len(datetimes) == 0:
        return pl.Series("bucket", [], dtype=pl.Int64)

    timestamps = datetimes.to_numpy().astype("datetime64[us]").astype("int64")
    midnight = (timestamps // (1440 * _MICROSECONDS_PER_MINUTE)) * (
        1440 * _MICROSECONDS_PER_MINUTE
    )
    minutes_into_day = (timestamps - midnight) // _MICROSECONDS_PER_MINUTE

    offset = minutes_into_day - session_open_minutes
    # Floor division, so negatives stay negative instead of truncating toward zero.
    return pl.Series("bucket", offset // bucket_minutes, dtype=pl.Int64)


def normalize_by_depth(
    values: pl.Series,
    *,
    ask_depth: pl.Series,
    bid_depth: pl.Series,
) -> pl.Series:
    """Scale a flow measure by the total book depth available to absorb it.

    Order flow imbalance predicts price change close to linearly, but the
    coefficient scales inversely with book depth: the same aggressive volume
    moves price further when there is less resting size to absorb it. What
    matters is flow *relative to what is there*, not flow in contracts.

    Parameters
    ----------
    values : pl.Series
        Flow measure, signed. Selling pressure is negative and stays negative.
    ask_depth, bid_depth : pl.Series
        Total displayed depth on each side at the same instant.

    Returns
    -------
    pl.Series
        ``values / (ask_depth + bid_depth)``, or **null** where the combined
        depth is not strictly positive. An empty book makes the ratio undefined,
        not infinite -- and Polars divides by zero to ``inf``, which passes every
        threshold it is compared against.
    """
    for name, series in (("ask_depth", ask_depth), ("bid_depth", bid_depth)):
        if len(series) != len(values):
            raise ValueError(
                f"{name} length {len(series)} does not match values length "
                f"{len(values)}"
            )

    if len(values) == 0:
        return pl.Series("normalized", [], dtype=pl.Float64)

    flow = pl.Series(values, dtype=pl.Float64).to_numpy(allow_copy=True)
    total = (
        pl.Series(ask_depth, dtype=pl.Float64).to_numpy(allow_copy=True)
        + pl.Series(bid_depth, dtype=pl.Float64).to_numpy(allow_copy=True)
    )

    usable = np.isfinite(total) & (total > 0)
    out = np.full(len(flow), np.nan, dtype=np.float64)
    np.divide(flow, total, out=out, where=usable)

    return pl.Series("normalized", out, dtype=pl.Float64).fill_nan(None)


def causal_bucket_zscore(
    values: pl.Series,
    *,
    buckets: pl.Series,
    session_ids: pl.Series,
    lookback_sessions: int,
    min_observations: int,
) -> pl.Series:
    """Standardise each value against its own time-of-day bucket in prior sessions.

    For every ``(bucket, session)`` pair the reference set is every observation
    in that bucket drawn from the trailing ``lookback_sessions`` **completed**
    sessions -- strictly those with a lower ``session_id``. Mean and sample
    standard deviation come from that set alone.

    Parameters
    ----------
    values : pl.Series
        The quantity to standardise. Nulls are ignored: they carry no
        information, so they neither count toward ``min_observations`` nor shift
        the reference mean.
    buckets : pl.Series
        Time-of-day bucket, from :func:`bucket_of_day`.
    session_ids : pl.Series
        Session index, **non-decreasing**. Rows out of session order would make
        the trailing window meaningless, so that raises rather than producing
        quiet nonsense.
    lookback_sessions : int
        How many completed sessions form the reference. Short windows adapt to
        regime changes; long ones are steadier but stale.
    min_observations : int
        Minimum reference observations before a z-score is emitted. Must be at
        least 2, since a sample standard deviation is undefined below that.

    Returns
    -------
    pl.Series
        Z-scores, **null** where the reference has fewer than
        ``min_observations`` values or zero variance. Null fails every
        comparison, so no signal fires on an unstandardisable value.

    Notes
    -----
    Every row of a session shares one reference distribution, computed before
    the session began. Recomputing it as the session progresses -- an expanding
    window within the day -- is lookahead, and it is the single easiest way to
    make an intraday backtest quietly better than the strategy.
    """
    if isinstance(lookback_sessions, bool) or not isinstance(
        lookback_sessions, (int, np.integer)
    ) or lookback_sessions <= 0:
        raise ValueError(
            f"lookback_sessions must be a positive integer, got {lookback_sessions!r}"
        )
    if isinstance(min_observations, bool) or not isinstance(
        min_observations, (int, np.integer)
    ) or min_observations < 2:
        raise ValueError(
            f"min_observations must be an integer of at least 2 (a sample "
            f"standard deviation is undefined below that), got {min_observations!r}"
        )

    for name, series in (("buckets", buckets), ("session_ids", session_ids)):
        if len(series) != len(values):
            raise ValueError(
                f"{name} length {len(series)} does not match values length "
                f"{len(values)}"
            )

    if len(values) == 0:
        return pl.Series("zscore", [], dtype=pl.Float64)

    value_array = pl.Series(values, dtype=pl.Float64).to_numpy(allow_copy=True)
    bucket_array = pl.Series(buckets, dtype=pl.Int64).to_numpy(allow_copy=True)
    session_array = pl.Series(session_ids, dtype=pl.Int64).to_numpy(allow_copy=True)

    if len(session_array) > 1 and np.diff(session_array).min() < 0:
        raise ValueError(
            "session_ids must be non-decreasing; rows out of session order would "
            "make the trailing reference window meaningless"
        )

    # Each (bucket, session) cell is summarised once -- count, sum, sum of
    # squares -- and a session's reference is the total of the cells in the
    # sessions before it. Summarising first means the per-row cost is a lookup
    # rather than a rescan of the whole history, which is the difference between
    # seconds and hours on tick-scale input.
    known = ~np.isnan(value_array)
    bucket_codes, bucket_labels = _dense_codes(bucket_array)
    n_buckets = len(bucket_labels)
    n_sessions = int(session_array.max()) + 1

    shape = (n_buckets, n_sessions)
    counts = np.zeros(shape, dtype=np.float64)
    sums = np.zeros(shape, dtype=np.float64)
    sums_sq = np.zeros(shape, dtype=np.float64)

    cells = (bucket_codes[known], session_array[known])
    np.add.at(counts, cells, 1.0)
    np.add.at(sums, cells, value_array[known])
    np.add.at(sums_sq, cells, value_array[known] ** 2)

    # Running totals over sessions, with a leading zero column so the window for
    # session s is a plain difference: totals[s] - totals[s - lookback].
    running_counts = np.concatenate([np.zeros((n_buckets, 1)), counts.cumsum(axis=1)], axis=1)
    running_sums = np.concatenate([np.zeros((n_buckets, 1)), sums.cumsum(axis=1)], axis=1)
    running_sums_sq = np.concatenate([np.zeros((n_buckets, 1)), sums_sq.cumsum(axis=1)], axis=1)

    # The trailing window for session s is the sessions [s - lookback_sessions,
    # s - 1]. It stops at s - 1, which is what keeps the current session out of
    # its own reference.
    sessions = np.arange(n_sessions)
    newest = sessions                                    # exclusive of s itself
    oldest = np.maximum(sessions - lookback_sessions, 0)

    window_counts = running_counts[:, newest] - running_counts[:, oldest]
    window_sums = running_sums[:, newest] - running_sums[:, oldest]
    window_sums_sq = running_sums_sq[:, newest] - running_sums_sq[:, oldest]

    with np.errstate(invalid="ignore", divide="ignore"):
        means = window_sums / window_counts
        # Sample variance from the running totals. Clamped at zero because the
        # algebraic form can produce a tiny negative when every observation is
        # identical.
        variances = np.maximum(
            (window_sums_sq - window_counts * means**2) / (window_counts - 1.0), 0.0
        )
        spreads = np.sqrt(variances)

    row_counts = window_counts[bucket_codes, session_array]
    row_means = means[bucket_codes, session_array]
    row_spreads = spreads[bucket_codes, session_array]

    usable = known & (row_counts >= min_observations) & (row_spreads > 0)

    out = np.full(len(value_array), np.nan, dtype=np.float64)
    np.divide(value_array - row_means, row_spreads, out=out, where=usable)

    return pl.Series("zscore", out, dtype=pl.Float64).fill_nan(None)


def _dense_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary integer labels onto contiguous 0..n-1 codes.

    Buckets can be negative (before the session open) and sparse, so they cannot
    index an array directly.
    """
    labels, codes = np.unique(values, return_inverse=True)
    return codes, labels
