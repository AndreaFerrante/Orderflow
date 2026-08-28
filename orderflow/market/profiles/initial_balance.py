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
    "ib_mid": pl.Float64,
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
        ``Date``, ``ib_high``, ``ib_low``, ``ib_mid``,
        ``ib_range_ticks``, ``ib_end_index``, sorted by ``Date``.

        ``ib_mid`` is ``(ib_high + ib_low) / 2`` -- the geometric centre of
        the balance area, which is what a reversion trade targets.  It is
        deliberately not a volume-weighted centre: that is the point of
        control, a different object with its own function.

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
            ((pl.col("ib_high") + pl.col("ib_low")) / 2.0).alias("ib_mid"),
            ((pl.col("ib_high") - pl.col("ib_low")) / tick_size).alias("ib_range_ticks"),
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
    "entry_price": pl.Float64,
    "ib_high": pl.Float64,
    "ib_low": pl.Float64,
    "ib_mid": pl.Float64,
    "ib_range_ticks": pl.Float64,
    "mid_distance_ticks": pl.Float64,
}

#: How ``cvd_delta`` must relate to the break for a signal to survive.
#: ``confirming`` -- flow pushes the way price broke; the continuation
#: thesis.  ``divergent`` -- price broke *against* the flow, so nothing is
#: participating in the extension; the reversion thesis.  ``off`` -- take
#: every break, the control.
_FLOW_GATES = ("confirming", "divergent", "off")

#: What a low-volume-node trigger means.  ``skip`` moves the signal to the
#: next clean tick and keeps the session; ``drop`` disqualifies the whole
#: extension because it *began* through a vacuum.  See ``find_ib_breakouts``.
_LVN_POLICIES = ("off", "skip", "drop")

_BREAKOUT_REQUIRED_COLUMNS = (
    "Index", "Datetime", "Date", "Price", "SessionType", "CD_Ask", "CD_Bid",
)


def find_ib_breakouts(
    ticks: pl.DataFrame,
    ib: pl.DataFrame,
    cvd_lookback_ticks: int = 2000,
    cvd_min_delta: float = 0.0,
    entry_cutoff_hour: int = 14,
    flow_gate: str = "confirming",
    tick_size: float = 0.25,
    lvn_policy: str = "off",
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
    flow_gate
        ``"confirming"`` keeps breaks the flow pushed (the continuation
        thesis); ``"divergent"`` keeps breaks that ran *against* the flow
        (the reversion thesis: an extension nobody is participating in);
        ``"off"`` keeps every break.
    tick_size
        Instrument minimum price increment, used only to express
        ``mid_distance_ticks``.
    lvn_policy
        What to do about breaks triggering on a low volume node.  Requires
        the enriched ``LVN`` column unless ``"off"``.  A break through a low
        volume node is price moving across a vacuum: there is nothing at
        that level to reject it, so it tends to continue.  Those are the
        breaks a *reversion* strategy must not fade.

        ``"off"`` ignores the node flag.  ``"skip"`` takes the first
        qualifying tick that is not on a node, keeping the session.
        ``"drop"`` consults only the FIRST tick beyond the edge: if the
        break began through a vacuum the whole extension is disqualified and
        no later tick rehabilitates it.

        The two are different strategies, not two spellings of one.
        ``"skip"`` changes the price you enter at; ``"drop"`` changes which
        sessions you trade at all.

    Returns
    -------
    pl.DataFrame
        ``Date``, ``signal_index``, ``entry_index``, ``direction``,
        ``cvd_delta``, ``entry_price``, ``ib_high``, ``ib_low``, ``ib_mid``,
        ``ib_range_ticks``, ``mid_distance_ticks``.
        ``direction`` is ``+1`` above the initial balance, ``-1`` below --
        it describes where price went, not which way to trade.  A breakout
        strategy buys ``+1``; a reversion strategy sells it.

        ``mid_distance_ticks`` is ``|entry_price - ib_mid| / tick_size``: how
        far a fill would have to travel to reach the centre of the balance
        area.  It is a magnitude, positive on both sides.

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
    if flow_gate not in _FLOW_GATES:
        raise ValueError(
            f"flow_gate must be one of {list(_FLOW_GATES)} -- got {flow_gate!r}"
        )
    if lvn_policy not in _LVN_POLICIES:
        raise ValueError(
            f"lvn_policy must be one of {list(_LVN_POLICIES)} -- got {lvn_policy!r}"
        )
    required = _BREAKOUT_REQUIRED_COLUMNS + (("LVN",) if lvn_policy != "off" else ())
    missing = [c for c in required if c not in ticks.columns]
    if missing:
        raise ValueError(
            f"ticks is missing {missing}; find_ib_breakouts requires "
            f"{list(required)}"
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
        pl.col("Price").shift(-1).over("Date").alias("entry_price"),
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
    )

    if lvn_policy == "skip":
        # Gates on the trigger tick, not the entry tick: the setup is "the
        # break happened at a level with volume behind it", and the entry
        # tick is merely the next print. Dropping a trigger does not forfeit
        # the session -- the first *qualifying* tick becomes the signal,
        # exactly as with the flow gate.
        qualified = qualified.filter(pl.col("LVN") != 1)
    elif lvn_policy == "drop":
        # Consult only the first tick beyond the edge, node or not. The
        # verdict is on the extension, so it is read where the extension
        # began -- before the flow gate has had a chance to move the signal
        # somewhere cleaner.
        first_touch = (
            post_ib
            .with_columns(
                pl.when(pl.col("Price") > pl.col("ib_high")).then(1)
                .when(pl.col("Price") < pl.col("ib_low")).then(-1)
                .otherwise(0).cast(pl.Int64).alias("direction")
            )
            .filter(pl.col("direction") != 0)
            .sort("Index")
            .group_by(["Date", "direction"], maintain_order=True)
            .first()
            .filter(pl.col("LVN") != 1)
            .select("Date", "direction")
        )
        qualified = qualified.join(first_touch, on=["Date", "direction"], how="semi")

    with_flow = pl.col("direction") * pl.col("cvd_delta") > cvd_min_delta
    if flow_gate == "confirming":
        qualified = qualified.filter(with_flow)
    elif flow_gate == "divergent":
        # Not ``~with_flow``: that would also admit a break whose flow is
        # merely small. The extension has to run against the tape by at
        # least cvd_min_delta to count as unparticipated.
        qualified = qualified.filter(
            pl.col("direction") * pl.col("cvd_delta") < -cvd_min_delta
        )

    qualified = qualified.with_columns(
        ((pl.col("entry_price") - pl.col("ib_mid")).abs() / tick_size)
        .alias("mid_distance_ticks")
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
