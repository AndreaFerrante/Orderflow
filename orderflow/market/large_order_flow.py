"""Session regime classification for large-order-flow strategies.

Which rulebook is live today? Some sessions rotate around a stable value area and
reward fading the edges; others trend, and fading them is how a month of gains
disappears. Telling them apart *before* the trading window opens is what this
module does.

The classifier reads only structural, causal inputs: how wide the first hour was
relative to recent sessions, whether price opened inside the prior session's
value area, how often price has crossed VWAP, which way VWAP is drifting, and
whether the developing point of control is migrating with it.

Freezing
--------
A session's label is fixed at the trading window's open and **does not change**.
A session whose inputs stop supporting the label stops trading; it does not
switch rulebooks. Re-evaluating intraday is how a strategy ends up fading a
breakout and buying the same move an hour later.

``regime_live`` recomputes the same rules over the full session and exists purely
as an edge-decay signal: a frozen label the live inputs no longer support is
information about the classifier, not a trading instruction. Nothing acts on it.

Sessions
--------
A session runs from the overnight ETH open through the following RTH close,
resetting at the RTH-to-ETH transition -- the same boundary volume profile, CVD,
VWAP and the delta-bar compressor all use.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

__all__ = [
    "apply_session_gates",
    "classify_session_regime",
    "find_absorption_fade_signals",
    "find_sweep_momentum_signals",
]

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

_REQUIRED_COLUMNS = ("Datetime", "Price", "vwap", "POC", "Prev_POC")

ROTATIONAL = "ROTATIONAL"
DIRECTIONAL = "DIRECTIONAL"
INDETERMINATE = "INDETERMINATE"


def _minutes(hhmm: str, *, name: str) -> int:
    if not isinstance(hhmm, str) or not _TIME_RE.match(hhmm):
        raise ValueError(f"{name} must be an 'HH:MM' string, got {hhmm!r}")
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def _count_vwap_crosses(
    price: np.ndarray, vwap: np.ndarray, *, confirm_distance: float
) -> int:
    """Count *confirmed* side changes of price against VWAP.

    A raw sign flip is not a cross. At tick resolution the bid-ask bounce flips
    the sign of ``price - vwap`` hundreds of times a session -- measured median
    172, p90 433 on MES -- so counting flips makes "price crossed VWAP twice"
    unreachable and "at most once" true only of sessions that never approached
    it. Spec 6.1 asks for *confirmed* changes, and this is what confirms them.

    A side is established only once price is at least ``confirm_distance`` beyond
    VWAP. Movement inside that band is noise around the line and changes
    nothing. This is ordinary hysteresis: the band must be crossed in full to
    register, so oscillation at the boundary cannot accumulate crosses.
    """
    if confirm_distance <= 0:
        raise ValueError(
            f"confirm_distance must be positive, got {confirm_distance!r}"
        )

    distance = price - vwap
    crosses = 0
    side = 0
    for value in distance:
        if value > confirm_distance:
            new_side = 1
        elif value < -confirm_distance:
            new_side = -1
        else:
            continue
        if side != 0 and new_side != side:
            crosses += 1
        side = new_side
    return crosses


def classify_session_regime(
    ticks: pl.DataFrame,
    *,
    ib_start_ct: str,
    ib_end_ct: str,
    ib_width_lookback_sessions: int,
    min_window_minutes: int,
    window_open_ct: str,
    rotational_slope_max: float,
    directional_slope_min: float,
    cross_confirm_distance: float,
    min_conditions: int = 3,
) -> pl.DataFrame:
    """Label each session ROTATIONAL, DIRECTIONAL or INDETERMINATE.

    Parameters
    ----------
    ticks : pl.DataFrame
        Enriched ticks carrying ``Datetime``, ``Price``, ``vwap``, ``POC`` and
        ``Prev_POC``, in chronological order. ``SessionType`` and ``Date`` are
        used when present. ``prev_va_low`` / ``prev_va_high`` give the prior
        session's value area; without them, opening location is unknown and
        every session is INDETERMINATE.
    ib_start_ct, ib_end_ct : str
        Initial-balance window, ``HH:MM`` Chicago time.
    ib_width_lookback_sessions : int
        Trailing completed sessions forming the IB-width reference.
    min_window_minutes : int
        Minimum elapsed RTH before a session may be classified at all.
    window_open_ct : str
        When the label freezes. Only data at or before this time feeds ``regime``.
    rotational_slope_max : float
        VWAP drift per hour below which a session may be ROTATIONAL.
    directional_slope_min : float
        VWAP drift per hour above which a session may be DIRECTIONAL.
    cross_confirm_distance : float
        How far beyond VWAP price must travel, in price units, before a side
        change counts. Without it the bid-ask bounce registers hundreds of
        crosses per session and both cross conditions become unreachable.
    min_conditions : int, default 3
        How many of the four conditions must hold for a label. Requiring all
        four leaves about 6% of sessions labelled, because each condition passes
        roughly half the population and their conjunction collapses; scoring
        keeps each input's information without that collapse. A session scoring
        at or above the threshold on *both* labels is INDETERMINATE -- arguing
        equally for both is not evidence of either.

    Returns
    -------
    pl.DataFrame
        One row per session: ``session_id``, ``date``, ``regime``,
        ``regime_live``, and every classifier input, so the decay report and any
        recalibration can work from the recorded values rather than recomputing.

    Notes
    -----
    Returns one row per session rather than per bar. Nothing downstream consumes
    a bar-level regime -- the entry rules ask only "what is this session" -- and
    a per-bar axis would be a larger contract than any caller needs.
    """
    for name, value in (("ib_start_ct", ib_start_ct), ("ib_end_ct", ib_end_ct),
                        ("window_open_ct", window_open_ct)):
        _minutes(value, name=name)

    missing = [c for c in _REQUIRED_COLUMNS if c not in ticks.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    schema = {
        "session_id": pl.Int64,
        "date": pl.String,
        "regime": pl.String,
        "regime_live": pl.String,
        "ib_width": pl.Float64,
        "ib_width_z": pl.Float64,
        "open_location": pl.String,
        "vwap_crosses": pl.Int64,
        "vwap_slope": pl.Float64,
        "poc_migration": pl.Float64,
        "elapsed_minutes": pl.Float64,
        "rotational_score": pl.Int64,
        "directional_score": pl.Int64,
    }
    if ticks.height == 0:
        return pl.DataFrame(schema=schema)

    frame = ticks
    if "SessionType" in frame.columns:
        frame = frame.filter(pl.col("SessionType") == "RTH")
    if frame.height == 0:
        return pl.DataFrame(schema=schema)

    if "Date" not in frame.columns:
        frame = frame.with_columns(pl.col("Datetime").dt.date().cast(pl.String).alias("Date"))

    ib_start = _minutes(ib_start_ct, name="ib_start_ct")
    ib_end = _minutes(ib_end_ct, name="ib_end_ct")
    window_open = _minutes(window_open_ct, name="window_open_ct")

    # Cast before multiplying: Polars returns Int8 from dt.hour(), so hour * 60
    # silently overflows for any hour past 02:00 -- 11:00 comes out as -108, and
    # every window comparison downstream is then nonsense without raising.
    frame = frame.with_columns(
        (
            pl.col("Datetime").dt.hour().cast(pl.Int64) * 60
            + pl.col("Datetime").dt.minute().cast(pl.Int64)
        )
        .cast(pl.Float64)
        .alias("_minute_of_day")
    )

    has_value_area = {"prev_va_low", "prev_va_high"}.issubset(frame.columns)

    rows = []
    ib_history: list[float] = []

    for session_id, (date, session) in enumerate(
        frame.group_by("Date", maintain_order=True)
    ):
        date_label = str(date[0]) if isinstance(date, tuple) else str(date)
        minute = session["_minute_of_day"].to_numpy()

        ib_mask = (minute >= ib_start) & (minute < ib_end)
        ib_prices = session["Price"].to_numpy()[ib_mask]
        ib_width = float(ib_prices.max() - ib_prices.min()) if ib_prices.size else float("nan")

        # The reference is prior completed sessions only.
        reference = ib_history[-ib_width_lookback_sessions:]
        if len(reference) >= 2 and np.std(reference, ddof=1) > 0 and not np.isnan(ib_width):
            ib_width_z = float((ib_width - np.mean(reference)) / np.std(reference, ddof=1))
        else:
            ib_width_z = float("nan")

        open_price = float(session["Price"][0])
        if has_value_area:
            va_low = float(session["prev_va_low"][0])
            va_high = float(session["prev_va_high"][0])
            if va_low <= open_price <= va_high:
                open_location = "inside"
            elif open_price > va_high:
                open_location = "above"
            else:
                open_location = "below"
        else:
            open_location = "unknown"

        prev_poc = float(session["Prev_POC"][0])

        def evaluate(cutoff_minute: float) -> tuple[str, dict]:
            """Classify using only data at or before ``cutoff_minute``."""
            visible = minute <= cutoff_minute
            price = session["Price"].to_numpy()[visible]
            vwap = session["vwap"].to_numpy()[visible]
            poc = session["POC"].to_numpy()[visible]
            observed = minute[visible]

            if price.size < 2:
                return INDETERMINATE, {
                    "vwap_crosses": 0, "vwap_slope": float("nan"),
                    "poc_migration": float("nan"), "elapsed_minutes": 0.0,
                    "rotational_score": 0, "directional_score": 0,
                }

            elapsed = float(observed[-1] - observed[0])
            crosses = _count_vwap_crosses(
                price, vwap, confirm_distance=cross_confirm_distance
            )
            hours = elapsed / 60.0
            slope = float((vwap[-1] - vwap[0]) / hours) if hours > 0 else float("nan")
            migration = float(poc[-1] - poc[0])

            inputs = {
                "vwap_crosses": crosses,
                "vwap_slope": slope,
                "poc_migration": migration,
                "elapsed_minutes": elapsed,
                "rotational_score": 0,
                "directional_score": 0,
            }

            # Spec 6.1: a session with no prior POC has no opening-location
            # reference, so it cannot be classified at all.
            if prev_poc <= 0 or np.isnan(ib_width_z) or np.isnan(slope):
                return INDETERMINATE, inputs
            if elapsed < min_window_minutes:
                return INDETERMINATE, inputs

            # Score, do not conjoin. Requiring all four conditions at once left
            # 3 to 4 labelled sessions in 60: each is satisfied by roughly half
            # the population, so their `and` lands near 6%. Counting how many
            # hold keeps every input's information while letting a session
            # qualify on the strength of the rest when one disagrees.
            rotational_score = int(
                (ib_width_z > 0)
                + (open_location == "inside")
                + (crosses >= 2)
                + (abs(slope) < rotational_slope_max)
            )
            directional_score = int(
                (ib_width_z < 0)
                + (crosses <= 1)
                + (abs(slope) > directional_slope_min)
                + (migration != 0 and np.sign(migration) == np.sign(slope))
            )
            inputs["rotational_score"] = rotational_score
            inputs["directional_score"] = directional_score

            rotational = rotational_score >= min_conditions
            directional = directional_score >= min_conditions

            # A session arguing equally for both is not evidence of either.
            if rotational and directional:
                return INDETERMINATE, inputs
            if rotational:
                return ROTATIONAL, inputs
            if directional:
                return DIRECTIONAL, inputs
            return INDETERMINATE, inputs

        regime, frozen_inputs = evaluate(float(window_open))
        regime_live, _ = evaluate(float(minute[-1]))

        rows.append(
            {
                "session_id": session_id,
                "date": date_label,
                "regime": regime,
                "regime_live": regime_live,
                "ib_width": ib_width,
                "ib_width_z": ib_width_z,
                "open_location": open_location,
                **frozen_inputs,
            }
        )

        if not np.isnan(ib_width):
            ib_history.append(ib_width)

    return pl.DataFrame(rows, schema=schema)


def apply_session_gates(
    regime: pl.DataFrame,
    *,
    sigma: pl.DataFrame,
    vol_kill_switch_multiple: float,
    vol_kill_switch_lookback_sessions: int,
) -> pl.DataFrame:
    """Add the trend-day veto and the volatility kill switch to a regime frame.

    Both are session-level overrides, and both are applied **after**
    classification without touching ``regime``. Keeping the classifier's own
    output intact is what lets it be validated separately from the gates that
    consume it.

    **Trend-day veto.** A session that opens outside the prior value area and
    never trades back inside is a trend day. Roughly one session in six runs one
    direction all day, and those are the sessions that destroy a fade strategy.
    The veto blocks the fade; it never promotes a session to the momentum
    rulebook, so a vetoed session with no directional label simply does not
    trade.

    **Volatility kill switch.** Set once the session's volatility exceeds its
    trailing distribution by ``vol_kill_switch_multiple``, and **latched** from
    then on. Absorption strategies are short volatility in disguise: a smooth
    curve, then one outsized loss. A gate that switches back off when volatility
    dips is a soft filter wearing a hard filter's name.

    Parameters
    ----------
    regime : pl.DataFrame
        Output of :func:`classify_session_regime`, carrying ``session_id``,
        ``regime``, ``open_location`` and ``returned_to_value``.
    sigma : pl.DataFrame
        One row per session with ``session_id`` and ``sigma``. A null sigma is
        unknown, not extreme, and never trips the switch.
    vol_kill_switch_multiple : float
        Multiple of the trailing mean sigma that counts as extreme.
    vol_kill_switch_lookback_sessions : int
        Trailing completed sessions forming that reference.

    Returns
    -------
    pl.DataFrame
        The input with ``trend_day_veto`` and ``volatility_halted`` appended.
    """
    if not isinstance(vol_kill_switch_multiple, (int, float)) \
            or vol_kill_switch_multiple <= 0:
        raise ValueError(
            f"vol_kill_switch_multiple must be positive, "
            f"got {vol_kill_switch_multiple!r}"
        )
    if isinstance(vol_kill_switch_lookback_sessions, bool) or not isinstance(
        vol_kill_switch_lookback_sessions, (int, np.integer)
    ) or vol_kill_switch_lookback_sessions <= 0:
        raise ValueError(
            f"vol_kill_switch_lookback_sessions must be a positive integer, "
            f"got {vol_kill_switch_lookback_sessions!r}"
        )

    required = ("session_id", "regime", "open_location", "returned_to_value")
    missing = [c for c in required if c not in regime.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "session_id" not in sigma.columns or "sigma" not in sigma.columns:
        raise ValueError("sigma must carry session_id and sigma columns")

    if regime.height == 0:
        return regime.with_columns(
            pl.Series("trend_day_veto", [], dtype=pl.Boolean),
            pl.Series("volatility_halted", [], dtype=pl.Boolean),
        )

    missing_sessions = set(regime["session_id"].to_list()) - set(
        sigma["session_id"].to_list()
    )
    if missing_sessions:
        raise ValueError(
            f"sigma is missing session_id values: {sorted(missing_sessions)}"
        )

    veto = (pl.col("open_location") != "inside") & ~pl.col("returned_to_value")

    joined = regime.join(sigma, on="session_id", how="left")
    sigma_values = joined["sigma"].to_numpy().astype(np.float64)

    halted = np.zeros(len(sigma_values), dtype=bool)
    latched = False
    for i, value in enumerate(sigma_values):
        if not latched and not np.isnan(value):
            window = sigma_values[max(0, i - vol_kill_switch_lookback_sessions):i]
            window = window[~np.isnan(window)]
            if window.size >= 2:
                reference = float(np.mean(window))
                if reference > 0 and value > reference * vol_kill_switch_multiple:
                    latched = True
        halted[i] = latched

    return regime.with_columns(
        veto.alias("trend_day_veto"),
        pl.Series("volatility_halted", halted, dtype=pl.Boolean),
    )


_FEATURE_COLUMNS = (
    "session_id",
    "Index",
    "signal_index",
    "hour_ct",
    "minute_ct",
    "regime",
    "trend_day_veto",
    "volatility_halted",
    "level_price",
    "distance_to_level_ticks",
    "level_touches",
    "side",
    "flow_z",
    "refresh_rank",
    "vanish_rank",
    "lambda_rank",
    "closed_beyond_level",
    "trigger_confirmed",
    "trigger_bar_delta",
    "next_bar_open",
    "sigma",
)

_SIGNAL_COLUMNS = (
    "signal_index",
    "Index",
    "TradeType",
    "signal_direction",
    "entry_price",
    "stop_loss",
    "target_type",
    "variant",
    "regime",
    "sigma_at_entry",
    "stop_distance_points",
    "level_price",
    "refresh_rank",
    "vanish_rank",
    "lambda_rank",
    "flow_z",
    "trapped_traders",
    "cvd_divergence",
    "stacked_levels",
)


def find_absorption_fade_signals(
    features: pl.DataFrame,
    *,
    band_tolerance_ticks: int,
    min_level_touches: int,
    flow_burst_z: float,
    refresh_rank_min: float,
    vanish_rank_max: float,
    lambda_rank_max: float,
    stop_vol_multiple: float,
    window_hours: tuple,
    late_entry_hour: int,
    late_entry_minute: int,
) -> pl.DataFrame:
    """Select absorption-fade entries from a prepared bar-level feature frame.

    Aggressive flow arrives at a level that was chosen in advance, something
    absorbs it without repricing, and the defender does not simply pull. Fade the
    exhausted push.

    Ten conditions, all required. Each is expressed as a column comparison so
    that a test can break exactly one and watch the signal disappear -- a filter
    nobody can prove is wired in is a filter that will eventually stop being.

    **Ranked, not thresholded.** Refresh and vanish ratios arrive as ranks within
    their session rather than raw values. Both raw measures accumulate from the
    session open, so their scales bear no relation to the thresholds the source
    methodology describes: an absolute refresh floor of 2.5 admits three quarters
    of all prices, and an absolute vanish ceiling of 0.20 admits none at all.
    Ranking restores the intended meaning -- unusually refreshed, unusually
    persistent -- without inventing a scale.

    Parameters
    ----------
    features : pl.DataFrame
        One row per candidate bar, from :func:`build_absorption_features`.
    band_tolerance_ticks : int
        How close to a level counts as "at" it.
    min_level_touches : int
        Prior touches required before a level may be faded. The first test of a
        level fails more often than the second, because the flow driving price
        there is usually a metaorder still working.
    flow_burst_z : float
        Minimum depth-normalised flow z-score into the level.
    refresh_rank_min : float
        Minimum within-session rank of the refresh ratio, 0 to 1.
    vanish_rank_max : float
        Maximum within-session rank of the vanish ratio. High vanish means the
        depth left untraded, which is pulling rather than defending.
    lambda_rank_max : float
        Maximum within-session rank of realised lambda. High lambda is informed
        flow moving a thin book, and must never be faded.
    stop_vol_multiple : float
        Stop offset beyond the level, in units of sigma. Stops cluster one tick
        past obvious levels; this deliberately sits outside that pile.
    window_hours : tuple of int
        Chicago hours in which entries are permitted.
    late_entry_hour, late_entry_minute : int
        After this time no new entry is taken, because a late trade has less room
        to work before the hard flat.

    Returns
    -------
    pl.DataFrame
        One row per accepted signal, sorted ascending by ``Index``. The engine
        consumes signals by array position rather than by timestamp lookup, so
        unsorted rows would attach to the wrong ticks.

        ``TradeType`` is 2 for long and 1 for short -- the engine validates that
        column and rejects anything else, and it is what actually determines
        trade direction.
    """
    missing = [c for c in _FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if features.height == 0:
        return pl.DataFrame(
            schema={
                "signal_index": pl.Int64, "Index": pl.Int64, "TradeType": pl.Int64,
                "signal_direction": pl.String, "entry_price": pl.Float64,
                "stop_loss": pl.Float64, "target_type": pl.String, "variant": pl.String,
                "regime": pl.String, "sigma_at_entry": pl.Float64,
                "stop_distance_points": pl.Float64, "level_price": pl.Float64,
                "refresh_rank": pl.Float64, "vanish_rank": pl.Float64,
                "lambda_rank": pl.Float64, "flow_z": pl.Float64,
                "trapped_traders": pl.Boolean, "cvd_divergence": pl.Boolean,
                "stacked_levels": pl.Boolean,
            }
        )

    is_long = pl.col("side") == "long"
    minutes = pl.col("hour_ct") * 60 + pl.col("minute_ct")
    cutoff = late_entry_hour * 60 + late_entry_minute

    accepted = (
        # 1. Regime and session gates. A vetoed or halted session does not fade.
        (pl.col("regime") == "ROTATIONAL")
        & ~pl.col("trend_day_veto")
        & ~pl.col("volatility_halted")
        # 2. Inside the trading window, before the late-entry cutoff.
        & pl.col("hour_ct").is_in(list(window_hours))
        & (minutes <= cutoff)
        # 3. At a level chosen in advance, not in no-man's land.
        & (pl.col("distance_to_level_ticks").abs() <= band_tolerance_ticks)
        # 4. Not the first test of that level.
        & (pl.col("level_touches") >= min_level_touches)
        # 5. A real burst of flow into the level.
        & (pl.col("flow_z") >= flow_burst_z)
        # 6. Something is absorbing it.
        & (pl.col("refresh_rank") >= refresh_rank_min)
        # 7. Not informed flow through a thin book.
        & (pl.col("lambda_rank") <= lambda_rank_max)
        # 8. The defender is not merely pulling. A cancel, not a preference.
        & (pl.col("vanish_rank") <= vanish_rank_max)
        # 9. The level held: price did not close beyond it.
        & ~pl.col("closed_beyond_level")
        # 10. A trigger bar closed back on the origin side, with agreeing delta.
        & pl.col("trigger_confirmed")
        & pl.when(is_long)
        .then(pl.col("trigger_bar_delta") > 0)
        .otherwise(pl.col("trigger_bar_delta") < 0)
    )

    # A null measure is untrustworthy, never permissive. Polars propagates null
    # through comparisons, and `fill_null(False)` is what turns that into a
    # rejection rather than letting it reach the filter as an unknown.
    stop_offset = pl.col("sigma") * stop_vol_multiple

    return (
        features.filter(accepted.fill_null(False))
        .with_columns(
            pl.when(is_long).then(2).otherwise(1).cast(pl.Int64).alias("TradeType"),
            pl.col("side").alias("signal_direction"),
            pl.col("next_bar_open").alias("entry_price"),
            pl.when(is_long)
            .then(pl.col("level_price") - stop_offset)
            .otherwise(pl.col("level_price") + stop_offset)
            .alias("stop_loss"),
            pl.lit("triple_barrier").alias("target_type"),
            pl.lit("lof_absorption_fade").alias("variant"),
            pl.col("sigma").alias("sigma_at_entry"),
            stop_offset.alias("stop_distance_points"),
        )
        .select(list(_SIGNAL_COLUMNS))
        .sort("Index")
    )


_SWEEP_FEATURE_COLUMNS = (
    "session_id",
    "Index",
    "signal_index",
    "hour_ct",
    "minute_ct",
    "regime",
    "side",
    "sweep_levels",
    "sweep_size_z",
    "sweep_origin_price",
    "destination_thin",
    "thin_area_far_edge",
    "farside_refresh_rank",
    "lambda_rank",
    "pullback_confirmed",
    "pullback_held_origin",
    "trigger_bar_delta",
    "next_bar_open",
    "sigma",
)

_SWEEP_SIGNAL_COLUMNS = (
    "signal_index",
    "Index",
    "TradeType",
    "signal_direction",
    "entry_price",
    "stop_loss",
    "target_type",
    "variant",
    "regime",
    "sigma_at_entry",
    "stop_distance_points",
    "sweep_levels",
    "sweep_size_z",
    "sweep_origin_price",
    "thin_area_far_edge",
    "farside_refresh_rank",
    "lambda_rank",
    "aggressor_n_fills",
)

_SWEEP_SIGNAL_SCHEMA = {
    "signal_index": pl.Int64, "Index": pl.Int64, "TradeType": pl.Int64,
    "signal_direction": pl.String, "entry_price": pl.Float64, "stop_loss": pl.Float64,
    "target_type": pl.String, "variant": pl.String, "regime": pl.String,
    "sigma_at_entry": pl.Float64, "stop_distance_points": pl.Float64,
    "sweep_levels": pl.Int64, "sweep_size_z": pl.Float64,
    "sweep_origin_price": pl.Float64, "thin_area_far_edge": pl.Float64,
    "farside_refresh_rank": pl.Float64, "lambda_rank": pl.Float64,
    "aggressor_n_fills": pl.Int64,
}


def find_sweep_momentum_signals(
    features: pl.DataFrame,
    *,
    sweep_min_levels: int,
    flow_burst_z: float,
    farside_refresh_rank_max: float,
    lambda_rank_min: float,
    stop_vol_multiple: float,
    window_hours: tuple,
    late_entry_hour: int,
    late_entry_minute: int,
) -> pl.DataFrame:
    """Select sweep-momentum entries from a prepared bar-level feature frame.

    One aggressor order clears several price levels into an area where little has
    traded on previous sessions, and nothing is waiting on the far side. Price
    moves through thin areas quickly because there is nobody there to stop it,
    and the same premise is what the stop keys on: re-entry means the thesis
    failed.

    **This is not Rule A with the sign flipped.** Three conditions invert and two
    gates do not apply:

    * Far-side refresh must be **low**. Rule A wants a hidden defender; this rule
      wants nobody home.
    * Lambda must **not** be in the bottom quantile. A sweep into collapsing
      lambda is being absorbed, which is Rule A's setup -- this is the condition
      that stops both rules firing on one event.
    * The **volatility kill switch does not apply**. It exists because absorption
      is short volatility in disguise; momentum is long it, so a violent session
      is where this rule belongs.
    * The **trend-day veto does not apply**. It blocks the fade and explicitly
      does not enable momentum, so it has nothing to say about a session already
      labelled directional.

    Selectivity is the live risk here. Raw sweeps run at roughly 45 per session
    inside the trading window against a source expectation of one or two setups
    an evening, so conditions 4 through 7 supply nearly all the filtering. If the
    trade count comes back high, that is where to look first.

    Parameters
    ----------
    features : pl.DataFrame
        One row per candidate bar, from :func:`build_sweep_features`.
    sweep_min_levels : int
        Price levels an aggressor order must clear to count as a sweep.
    flow_burst_z : float
        Minimum time-bucketed z-score of the sweep's size.
    farside_refresh_rank_max : float
        Maximum within-session refresh rank beyond the sweep's endpoint. High
        refresh there means hidden size is waiting, which is a reason not to go.
    lambda_rank_min : float
        Minimum within-session rank of realised lambda. Below it, the sweep is
        being absorbed rather than clearing.
    stop_vol_multiple : float
        Stop offset beyond the thin area's far edge, in units of sigma.
    window_hours : tuple of int
        Chicago hours in which entries are permitted.
    late_entry_hour, late_entry_minute : int
        After this time no new entry is taken.

    Returns
    -------
    pl.DataFrame
        One row per accepted signal, sorted ascending by ``Index``, carrying
        ``TradeType`` (2 long, 1 short) for the engine.

        ``aggressor_n_fills`` is **recorded, not filtered on**. Only about a
        tenth of sweeps are shaped like a single aggressor; the rest are chained
        match events. Gating on it would cut candidates by roughly 90%, and
        whether that sharpens the signal or starves an already marginal sample is
        an open question rather than a decided one.
    """
    missing = [c for c in _SWEEP_FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if features.height == 0:
        return pl.DataFrame(schema=_SWEEP_SIGNAL_SCHEMA)

    frame = features
    if "aggressor_n_fills" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("aggressor_n_fills"))

    is_long = pl.col("side") == "long"
    minutes = pl.col("hour_ct") * 60 + pl.col("minute_ct")
    cutoff = late_entry_hour * 60 + late_entry_minute

    accepted = (
        # 1. Directional session. Neither session gate applies here -- see above.
        (pl.col("regime") == "DIRECTIONAL")
        # 2. Inside the window, before the late-entry cutoff.
        & pl.col("hour_ct").is_in(list(window_hours))
        & (minutes <= cutoff)
        # 3. A real sweep, unusual for this time of day.
        & (pl.col("sweep_levels") >= sweep_min_levels)
        & (pl.col("sweep_size_z") >= flow_burst_z)
        # 4. Into somewhere little has traded before.
        & pl.col("destination_thin")
        # 5. And on the far side, nothing.
        & (pl.col("farside_refresh_rank") <= farside_refresh_rank_max)
        # 6. Clearing, not being absorbed.
        & (pl.col("lambda_rank") >= lambda_rank_min)
        # 7. A shallow pullback that held, then a bar closing with the sweep.
        & pl.col("pullback_confirmed")
        & pl.col("pullback_held_origin")
        & pl.when(is_long)
        .then(pl.col("trigger_bar_delta") > 0)
        .otherwise(pl.col("trigger_bar_delta") < 0)
    )

    # A null measure is untrustworthy, never permissive. Polars propagates null
    # through comparisons; fill_null(False) turns that into a rejection.
    stop_offset = pl.col("sigma") * stop_vol_multiple

    return (
        frame.filter(accepted.fill_null(False))
        .with_columns(
            pl.when(is_long).then(2).otherwise(1).cast(pl.Int64).alias("TradeType"),
            pl.col("side").alias("signal_direction"),
            pl.col("next_bar_open").alias("entry_price"),
            # The thin area is not supposed to be revisited, so the stop sits
            # just beyond the edge price came in through.
            pl.when(is_long)
            .then(pl.col("thin_area_far_edge") - stop_offset)
            .otherwise(pl.col("thin_area_far_edge") + stop_offset)
            .alias("stop_loss"),
            pl.lit("triple_barrier").alias("target_type"),
            pl.lit("lof_sweep_momentum").alias("variant"),
            pl.col("sigma").alias("sigma_at_entry"),
            stop_offset.alias("stop_distance_points"),
        )
        .select(list(_SWEEP_SIGNAL_COLUMNS))
        .sort("Index")
    )
