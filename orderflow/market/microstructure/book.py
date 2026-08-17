"""Per-price book state accumulated from the depth-of-market ladder.

Sierra Chart tick data carries a depth snapshot alongside every trade: an anchor
price per side plus *N* displayed sizes stepping away from it. Walking that
ladder and remembering, for each price, the most size ever shown there and the
volume actually executed there is what makes hidden liquidity measurable.

Two derived quantities are built on this state and are pure functions of it, so
the walk happens once:

* **Refresh ratio** -- volume traded at a price divided by the most ever
  displayed there. A ratio of 3 means three times more traded than was ever
  visible, so someone was refreshing or icebergging.
* **Vanish ratio** -- how much displayed depth disappeared without being traded,
  which separates a genuine defender from a pulled order.

Ladder geometry
---------------
::

    ask level i:  price = AskDOMPrice + i * tick_size,  size = AskDOM_i
    bid level i:  price = BidDOMPrice - i * tick_size,  size = BidDOM_i

Map through ``AskDOMPrice`` / ``BidDOMPrice``, **never** through ``AskPrice`` /
``BidPrice``. Those agree with the ladder anchors on only about 60% of rows,
because the depth feed and the quote are separate snapshots.

Data facts this module accounts for, all measured across four MES files
(2024-12-22 to 2025-12-12) in ``tests/test_lof_data_baseline.py``:

* The ladder is occasionally **locked**, ``AskDOMPrice == BidDOMPrice``, on
  2.9e-05 to 2.9e-04 of rows -- at file start before the feed initialises, and
  scattered afterwards. Both level-0 prices then resolve to one price, so
  reading both sides would count that depth twice. Locked snapshots are skipped.
* The ladder is **never crossed**. If it ever is, the feed is corrupt and this
  module raises rather than guessing.
* ``DepthSequence`` is monotonic and changes on roughly 55% of rows, so depth
  evolves independently of trades. Only rows where it advanced carry a new
  snapshot; every row's ``Volume`` still counts as traded.

A price outside the ladder window is *unobserved*, not empty. ``n_depth_updates``
is emitted so callers can disqualify levels whose statistics rest on too few
observations -- a level that entered the window moments before a setup has an
artificially low maximum, which inflates any refresh ratio built on it.

All functions are strictly causal: the accumulators only ever grow, so computing
on the input truncated at row *k* reproduces the state as it stood at *k*.

Missing values are ``null``, never ``NaN``
------------------------------------------
Polars orders ``NaN`` **above** every float, so ``pl.col("x") >= 2.5`` returns
``True`` for ``NaN`` -- the opposite of Python and NumPy, where it is ``False``::

    >>> pl.DataFrame({"x": [float("nan")]}).filter(pl.col("x") >= 2.5).height
    1

Any derived ratio here that used ``NaN`` to mean "do not trust this level" would
therefore make untrustworthy levels *pass* every threshold rather than fail it,
firing signals hardest at prices the book never displayed. ``null`` compares to
``null``, which ``filter`` drops. Use ``null``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["accumulate_book_state", "compute_refresh_ratio"]


_REQUIRED_COLUMNS = (
    "DepthSequence",
    "Price",
    "Volume",
    "AskDOMPrice",
    "BidDOMPrice",
)

_STATE_SCHEMA = {
    "session_id": pl.Int64,
    "price": pl.Float64,
    "max_displayed": pl.Float64,
    "last_displayed": pl.Float64,
    "traded": pl.Float64,
    "n_depth_updates": pl.Int64,
}


def _session_ids(session_types: np.ndarray) -> np.ndarray:
    """Increment at each RTH to non-RTH transition.

    The same convention ``build_cvd_vwap_bars`` uses, so volume profile, CVD,
    VWAP and book state all reset on the same boundary.
    """
    if len(session_types) == 0:
        return np.zeros(0, dtype=np.int64)
    is_rth = session_types == "RTH"
    starts = np.zeros(len(session_types), dtype=np.int64)
    starts[1:] = (is_rth[:-1] & ~is_rth[1:]).astype(np.int64)
    return np.cumsum(starts)


def accumulate_book_state(
    ticks: pl.DataFrame,
    *,
    tick_size: float,
    max_levels: int = 30,
    session_col: str = "SessionType",
) -> pl.DataFrame:
    """Accumulate per-price depth and trade statistics, per session.

    Parameters
    ----------
    ticks : pl.DataFrame
        Enriched tick data carrying ``DepthSequence``, ``Price``, ``Volume``,
        ``AskDOMPrice``, ``BidDOMPrice``, ``AskDOM_0..N``, ``BidDOM_0..N`` and a
        session-type column, in chronological order.
    tick_size : float
        Instrument minimum price increment, used to step away from each anchor.
        Required, with no default: a wrong tick size silently places every level
        at the wrong price.
    max_levels : int, default 30
        Ladder levels read per side. Lower it when the frame carries a truncated
        column projection.
    session_col : str, default "SessionType"
        Column holding ``"RTH"`` / ``"ETH"`` labels.

    Returns
    -------
    pl.DataFrame
        One row per ``(session_id, price)``, sorted:

        ``max_displayed``
            Largest size ever shown at this price this session.
        ``last_displayed``
            Size shown in the most recent depth snapshot covering this price.
        ``traded``
            Volume executed at this price this session.
        ``n_depth_updates``
            Depth snapshots that covered this price. Low values mean the price
            entered the ladder window recently and its maximum is unreliable.

    Raises
    ------
    ValueError
        Missing columns, non-positive ``tick_size`` or ``max_levels``, or a
        crossed ladder.
    """
    if not isinstance(tick_size, (int, float)) or tick_size <= 0:
        raise ValueError(f"tick_size must be a positive number, got {tick_size!r}")
    if isinstance(max_levels, bool) or not isinstance(max_levels, (int, np.integer)) \
            or max_levels <= 0:
        raise ValueError(f"max_levels must be a positive integer, got {max_levels!r}")

    missing = [c for c in _REQUIRED_COLUMNS if c not in ticks.columns]
    if session_col not in ticks.columns:
        missing.append(session_col)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    level_cols = [f"AskDOM_{i}" for i in range(max_levels)]
    level_cols += [f"BidDOM_{i}" for i in range(max_levels)]
    missing_levels = [c for c in level_cols if c not in ticks.columns]
    if missing_levels:
        raise ValueError(f"Missing required columns: {missing_levels}")

    if ticks.height == 0:
        return pl.DataFrame(schema=_STATE_SCHEMA)

    session_ids = _session_ids(ticks[session_col].to_numpy())
    depth_seq = ticks["DepthSequence"].to_numpy()
    trade_price = ticks["Price"].to_numpy().astype(np.float64)
    volume = ticks["Volume"].to_numpy().astype(np.float64)
    ask_anchor = ticks["AskDOMPrice"].to_numpy().astype(np.float64)
    bid_anchor = ticks["BidDOMPrice"].to_numpy().astype(np.float64)

    if np.any(ask_anchor < bid_anchor):
        raise ValueError(
            "ladder anchors are crossed (AskDOMPrice < BidDOMPrice); the depth "
            "feed is corrupt"
        )

    ask_sizes = np.column_stack(
        [ticks[f"AskDOM_{i}"].to_numpy() for i in range(max_levels)]
    ).astype(np.float64)
    bid_sizes = np.column_stack(
        [ticks[f"BidDOM_{i}"].to_numpy() for i in range(max_levels)]
    ).astype(np.float64)

    # Prices are stored as integer tick counts. Float keys would make 100.00
    # arrived at by different arithmetic paths compare unequal.
    def to_ticks(prices: np.ndarray) -> np.ndarray:
        return np.rint(prices / tick_size).astype(np.int64)

    ask_anchor_ticks = to_ticks(ask_anchor)
    bid_anchor_ticks = to_ticks(bid_anchor)
    trade_ticks = to_ticks(trade_price)

    # Each snapshot writes a contiguous run of prices: the ask ladder occupies
    # anchor .. anchor + max_levels - 1, the bid ladder anchor - max_levels + 1
    # .. anchor. Contiguity is what lets the whole ladder land in three array
    # slice operations instead of one dict lookup per level -- 60 Python
    # operations per snapshot became 6, which is the difference between 17
    # minutes and 30 seconds on a full file.
    lo_tick = int(min(bid_anchor_ticks.min() - max_levels + 1, trade_ticks.min()))
    hi_tick = int(max(ask_anchor_ticks.max() + max_levels - 1, trade_ticks.max()))
    width = hi_tick - lo_tick + 1

    n_sessions = int(session_ids[-1]) + 1
    shape = (n_sessions, width)
    max_displayed = np.zeros(shape, dtype=np.float64)
    last_displayed = np.zeros(shape, dtype=np.float64)
    traded = np.zeros(shape, dtype=np.float64)
    n_updates = np.zeros(shape, dtype=np.int64)
    seen = np.zeros(shape, dtype=bool)

    # Trades are independent of the depth snapshots, so they vectorise wholesale.
    traded_rows = volume > 0
    if traded_rows.any():
        np.add.at(
            traded,
            (session_ids[traded_rows], trade_ticks[traded_rows] - lo_tick),
            volume[traded_rows],
        )
        seen[session_ids[traded_rows], trade_ticks[traded_rows] - lo_tick] = True

    # Only a changed DepthSequence carries a new snapshot; re-reading the same one
    # would count its depth repeatedly. A locked ladder (ask anchor == bid anchor)
    # maps both level-0 prices to one price, so reading both sides would
    # double-count depth there -- skip the snapshot, the trade volume above still
    # counted.
    is_new_snapshot = np.empty(len(depth_seq), dtype=bool)
    is_new_snapshot[0] = True
    is_new_snapshot[1:] = depth_seq[1:] != depth_seq[:-1]
    usable = is_new_snapshot & (ask_anchor_ticks != bid_anchor_ticks)

    # ponytail: one Python iteration per depth snapshot, roughly 6.5 min on a full
    # 56.8M-tick file (2.6 min for RTH only). The cost is numpy call overhead on
    # 30-element slices, not the arithmetic. If Activity 19's repeated calibration
    # runs make this hurt, njit this loop with a Python fallback -- the house
    # convention for hot paths -- rather than obscuring it further.
    for i in np.flatnonzero(usable):
        session = session_ids[i]

        ask_lo = ask_anchor_ticks[i] - lo_tick
        ask_hi = ask_lo + max_levels
        bid_hi = bid_anchor_ticks[i] - lo_tick + 1
        bid_lo = bid_hi - max_levels

        for lo, hi, sizes in (
            (ask_lo, ask_hi, ask_sizes[i]),
            # The bid ladder steps downward from its anchor, so the slice is
            # reversed to put level 0 at the high end.
            (bid_lo, bid_hi, bid_sizes[i][::-1]),
        ):
            np.maximum(max_displayed[session, lo:hi], sizes,
                       out=max_displayed[session, lo:hi])
            last_displayed[session, lo:hi] = sizes
            n_updates[session, lo:hi] += 1
            seen[session, lo:hi] = True

    # Keep only prices that were actually observed -- the dense grid spans every
    # tick between the session's extremes, most of which the ladder never covered.
    session_idx, price_idx = np.nonzero(seen)

    # Reconstructing price as tick_count * tick_size reintroduces the float error
    # the integer grid was chosen to avoid: 1001 * 0.10 is 100.10000000000001,
    # which will not compare or join equal to 100.10. Exact for binary-fraction
    # tick sizes like 0.25, not for 0.10 (FGBL) or 0.03125 (ZB). Round it away --
    # ten decimals is far below any real instrument's precision.
    prices = np.round((price_idx + lo_tick) * tick_size, 10)

    return (
        pl.DataFrame(
            {
                "session_id": session_idx,
                "price": prices,
                "max_displayed": max_displayed[session_idx, price_idx],
                "last_displayed": last_displayed[session_idx, price_idx],
                "traded": traded[session_idx, price_idx],
                "n_depth_updates": n_updates[session_idx, price_idx],
            }
        )
        .cast(_STATE_SCHEMA)
        .sort("session_id", "price")
    )


def compute_refresh_ratio(
    book_state: pl.DataFrame,
    *,
    min_depth_updates: int = 5,
) -> pl.DataFrame:
    """Volume traded at a price divided by the most ever displayed there.

    ``R = 3`` means three times more volume traded at the price than was ever
    visible on the book there, so someone was refreshing or icebergging. This is
    a better proxy for a large hidden participant than order size, because it
    identifies the trader deliberately concealing size rather than the one who
    failed to.

    **R is ordinal, never causal.** Without market-by-order data, one iceberg
    replenishing twenty times and twenty independent traders arriving at the
    same price are indistinguishable. Rank R within a comparable population --
    the same time-of-day bucket, say -- rather than comparing raw values across
    a session, and record the raw value for later analysis.

    R alone is not a signal. It is only meaningful alongside price failing to
    break through the level.

    Parameters
    ----------
    book_state : pl.DataFrame
        Output of :func:`accumulate_book_state`, carrying ``max_displayed``,
        ``traded`` and ``n_depth_updates``.
    min_depth_updates : int, default 5
        Minimum depth snapshots covering a price before its ratio is trusted.

    Returns
    -------
    pl.DataFrame
        The input with ``refresh_ratio`` appended. **null** where the ratio
        cannot be trusted:

        * ``max_displayed <= 0`` -- the price was never displayed in the ladder
          window, so it is unobserved rather than infinitely refreshed. Dividing
          would give ``inf``, which compares greater than any threshold and
          would make every unobserved price read as maximal absorption.
        * ``n_depth_updates < min_depth_updates`` -- the price entered the
          window too recently for its maximum to mean anything, which inflates
          the ratio for a reason unrelated to hidden liquidity.

    Notes
    -----
    **The sentinel is ``null``, not ``NaN``, and the difference is not cosmetic.**
    Polars orders ``NaN`` *above* every float, so ``pl.col("x") >= 2.5`` returns
    ``True`` for ``NaN`` -- the opposite of Python and NumPy, where the same
    comparison is ``False``. A ``NaN`` sentinel here would have made every
    unobserved and every thin-history level *pass* the refresh threshold instead
    of failing it, firing signals hardest at prices the book never showed.

    ``null`` yields ``null`` under comparison, which ``filter`` drops. No signal
    fires on an untrustworthy level, which is the intended behaviour rather than
    a fallback.
    """
    if isinstance(min_depth_updates, bool) or \
            not isinstance(min_depth_updates, (int, np.integer)) or \
            min_depth_updates < 0:
        raise ValueError(
            f"min_depth_updates must be a non-negative integer, "
            f"got {min_depth_updates!r}"
        )

    required = ("max_displayed", "traded", "n_depth_updates")
    missing = [c for c in required if c not in book_state.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trustworthy = (pl.col("max_displayed") > 0) & (
        pl.col("n_depth_updates") >= min_depth_updates
    )
    return book_state.with_columns(
        pl.when(trustworthy)
        .then(pl.col("traded") / pl.col("max_displayed"))
        .otherwise(None)  # null, not NaN -- see Notes
        .cast(pl.Float64)
        .alias("refresh_ratio")
    )
