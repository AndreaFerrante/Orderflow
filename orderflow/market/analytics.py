"""
CVD / VWAP analytics for delta-bar-based strategy signals.

All functions are reusable and strategy-agnostic.  They operate on
enriched tick DataFrames that already carry session VWAP, SD bands,
CD_Ask / CD_Bid, and SessionType columns.
"""

from __future__ import annotations

import polars as pl

from orderflow.compressor import assign_delta_bar_ids


_REQUIRED_TICK_COLUMNS = {
    "Index", "Sequence", "Datetime", "Date", "Time", "SessionType",
    "Price", "Volume", "TradeType", "BidPrice", "AskPrice",
    "CD_Ask", "CD_Bid", "vwap",
    "vwap_sd1_top", "vwap_sd1_bottom",
    "vwap_sd2_top", "vwap_sd2_bottom",
    "VA_Areas", "LVN", "ValleysPeaks",
}


def build_cvd_vwap_bars(
    ticks: pl.DataFrame,
    *,
    delta_bar_threshold: float,
) -> pl.DataFrame:
    """Build completed 500-contract delta bars with full metadata.

    CVD is computed solely from ``CD_Ask - CD_Bid`` at each bar's last
    source tick.  An enriched ``CVD`` column, if present, is ignored.

    Parameters
    ----------
    ticks : pl.DataFrame
        Enriched tick data with all required columns.
    delta_bar_threshold : float
        Absolute running-delta threshold for bar closure.

    Returns
    -------
    pl.DataFrame
        One row per completed delta bar.
    """
    missing = _REQUIRED_TICK_COLUMNS - set(ticks.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(ticks) == 0:
        return _empty_bars()

    ticks = ticks.sort("Datetime", "Sequence", "Index")

    if ticks["Index"].n_unique() != len(ticks):
        raise ValueError("Duplicate Index values")

    ticks = assign_delta_bar_ids(ticks, delta_bar_threshold)

    # Session ID: increments at every RTH-to-non-RTH transition
    session_id = _compute_session_ids(ticks["SessionType"].to_numpy())
    ticks = ticks.with_columns(pl.Series("_session_id", session_id))

    # VWAP-touch detection (within each bar, consecutive tick analysis)
    ticks = ticks.with_columns(
        (pl.col("Price") >= pl.col("vwap")).alias("_above"),
    )
    ticks = ticks.with_columns(
        pl.col("_above").shift(1).over("delta_bar_id").alias("_prev_above"),
    )
    ticks = ticks.with_columns(
        (
            pl.col("_prev_above").is_not_null()
            & ~pl.col("_prev_above")
            & pl.col("_above")
        ).alias("_touch_below"),
        (
            pl.col("_prev_above").is_not_null()
            & pl.col("_prev_above")
            & ~pl.col("_above")
        ).alias("_touch_above"),
    )

    # Next-bar mapping: first tick of each bar (including incomplete)
    all_bar_first = (
        ticks.group_by("delta_bar_id", maintain_order=True)
        .agg(
            pl.col("Index").first().alias("_nxt_idx"),
            pl.col("Price").first().alias("_nxt_price"),
            pl.col("BidPrice").first().alias("_nxt_bid"),
            pl.col("AskPrice").first().alias("_nxt_ask"),
            pl.col("vwap").first().alias("_nxt_vwap"),
        )
        .with_columns(
            pl.col("_nxt_idx").shift(-1).alias("next_bar_first_index"),
            pl.col("_nxt_price").shift(-1).alias("next_bar_open"),
            pl.col("_nxt_bid").shift(-1).alias("next_bar_bid"),
            pl.col("_nxt_ask").shift(-1).alias("next_bar_ask"),
            pl.col("_nxt_vwap").shift(-1).alias("next_bar_vwap"),
        )
        .select(
            "delta_bar_id",
            "next_bar_first_index", "next_bar_open",
            "next_bar_bid", "next_bar_ask", "next_bar_vwap",
        )
    )

    # Filter to completed ticks
    completed = ticks.filter(pl.col("delta_bar_complete"))

    if len(completed) == 0:
        return _empty_bars()

    # Sort expressions for earliest extreme tick
    _high_sort = ["Price", "Index"]
    _high_desc = [True, False]
    _low_sort = ["Price", "Index"]
    _low_desc = [False, False]

    bars = (
        completed.group_by("delta_bar_id", maintain_order=True)
        .agg(
            # Session
            pl.col("_session_id").first().alias("session_id"),
            # Time
            pl.col("Datetime").first().alias("OpenTime"),
            pl.col("Datetime").last().alias("CloseTime"),
            # OHLC
            pl.col("Price").first().alias("Open"),
            pl.col("Price").max().alias("High"),
            pl.col("Price").min().alias("Low"),
            pl.col("Price").last().alias("Close"),
            # Volume
            pl.col("Volume").sum().alias("Volume"),
            pl.when(pl.col("TradeType") == 2)
                .then(pl.col("Volume")).otherwise(0).sum().alias("AskVolume"),
            pl.when(pl.col("TradeType") == 1)
                .then(pl.col("Volume")).otherwise(0).sum().alias("BidVolume"),
            # CVD at bar close: CD_Ask - CD_Bid (never from enriched CVD column)
            pl.col("CD_Ask").last().alias("_cd_ask_close"),
            pl.col("CD_Bid").last().alias("_cd_bid_close"),
            # VWAP at close
            pl.col("vwap").last().alias("vwap"),
            # Source indices
            pl.col("Index").first().alias("first_index"),
            pl.col("Index").last().alias("last_index"),
            # High-tick metadata (earliest tick with max price)
            pl.col("Index").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_index"),
            pl.col("AskPrice").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_ask"),
            pl.col("vwap_sd1_top").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_vwap_sd1_top"),
            pl.col("vwap_sd2_top").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_vwap_sd2_top"),
            pl.col("VA_Areas").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_VA_Areas"),
            pl.col("LVN").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_LVN"),
            pl.col("ValleysPeaks").sort_by(_high_sort, descending=_high_desc)
                .first().alias("high_ValleysPeaks"),
            # Low-tick metadata (earliest tick with min price)
            pl.col("Index").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_index"),
            pl.col("BidPrice").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_bid"),
            pl.col("vwap_sd1_bottom").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_vwap_sd1_bottom"),
            pl.col("vwap_sd2_bottom").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_vwap_sd2_bottom"),
            pl.col("VA_Areas").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_VA_Areas"),
            pl.col("LVN").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_LVN"),
            pl.col("ValleysPeaks").sort_by(_low_sort, descending=_low_desc)
                .first().alias("low_ValleysPeaks"),
            # VWAP touches
            pl.col("_touch_below").any().alias("vwap_touched_from_below"),
            pl.col("_touch_above").any().alias("vwap_touched_from_above"),
        )
    )

    # Derived columns
    bars = bars.with_columns(
        (pl.col("AskVolume") - pl.col("BidVolume")).alias("bar_delta"),
        (pl.col("_cd_ask_close") - pl.col("_cd_bid_close")).alias("cvd"),
    ).drop("_cd_ask_close", "_cd_bid_close")

    # Join next-bar info
    bars = bars.join(all_bar_first, on="delta_bar_id", how="left")

    # Add sequential bar_index and drop internal id
    bars = bars.with_row_index("bar_index").drop("delta_bar_id")

    return bars


def classify_cvd_vwap_day_state(
    ticks: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    min_day_type_minutes: int,
) -> pl.DataFrame:
    """Label each bar with point-in-time day state.

    Uses only RTH ticks at or before each bar's ``last_index``.
    Precedence after warm-up: BALANCE > ROTATIONAL > TREND > UNCLASSIFIED.
    Before ``min_day_type_minutes`` elapse, state is UNCLASSIFIED.
    Cross count and slope reset at each new session.
    """
    if len(bars) == 0:
        return bars.with_columns(pl.lit("UNCLASSIFIED").alias("day_state"))

    # Add session IDs to ticks for session-scoped filtering
    tick_sessions = _compute_session_ids(ticks["SessionType"].to_numpy())
    ticks_with_sid = ticks.with_columns(pl.Series("_sid", tick_sessions))

    rth = ticks_with_sid.filter(pl.col("SessionType") == "RTH")

    # Precompute running stats per session for efficiency
    # Build a mapping: session_id → sorted RTH ticks
    session_groups: dict[int, pl.DataFrame] = {}
    if len(rth) > 0:
        for sid in rth["_sid"].unique().to_list():
            session_groups[sid] = rth.filter(pl.col("_sid") == sid).sort("Index")

    states: list[str] = []
    prev_session_id = -1
    prev_sign = 0
    cross_count = 0

    for i in range(len(bars)):
        bar = bars.row(i, named=True)
        sid = bar["session_id"]
        last_idx = bar["last_index"]

        # Reset on new session
        if sid != prev_session_id:
            prev_sign = 0
            cross_count = 0
            prev_session_id = sid

        sg = session_groups.get(sid)
        if sg is None or len(sg) == 0:
            states.append("UNCLASSIFIED")
            continue

        relevant = sg.filter(pl.col("Index") <= last_idx)
        if len(relevant) == 0:
            states.append("UNCLASSIFIED")
            continue

        rth_open_time = relevant["Datetime"][0]
        bar_close_time = bar["CloseTime"]
        elapsed_minutes = (bar_close_time - rth_open_time).total_seconds() / 60.0

        # Cross detection at bar close level
        bar_close = bar["Close"]
        bar_vwap = bar["vwap"]
        sign = 1 if bar_close > bar_vwap else (-1 if bar_close < bar_vwap else 0)
        if sign != 0:
            if prev_sign != 0 and sign != prev_sign:
                cross_count += 1
            prev_sign = sign

        if elapsed_minutes < min_day_type_minutes:
            states.append("UNCLASSIFIED")
            continue

        elapsed_hours = elapsed_minutes / 60.0

        rth_open_vwap = float(relevant["vwap"][0])
        current_vwap = float(bar_vwap)
        vwap_slope = (current_vwap - rth_open_vwap) / elapsed_hours if elapsed_hours > 0 else 0.0

        total = len(relevant)
        above = relevant.filter(pl.col("Price") > pl.col("vwap")).height
        below = relevant.filter(pl.col("Price") < pl.col("vwap")).height
        pct_above = above / total
        pct_below = below / total

        # Precedence: BALANCE > ROTATIONAL > TREND > UNCLASSIFIED
        if abs(vwap_slope) < 3.0 or cross_count >= 4:
            states.append("BALANCE")
        elif cross_count >= 1:
            states.append("ROTATIONAL")
        elif vwap_slope > 5.0 and pct_above > 0.80 and bar_close > bar_vwap:
            states.append("TREND_UP")
        elif vwap_slope < -5.0 and pct_below > 0.80 and bar_close < bar_vwap:
            states.append("TREND_DOWN")
        else:
            states.append("UNCLASSIFIED")

    return bars.with_columns(pl.Series("day_state", states))


def _compute_session_ids(session_types) -> list[int]:
    """Compute session IDs from SessionType array.

    A new session starts when SessionType transitions from RTH to a
    non-RTH value (the ETH reset point).
    """
    n = len(session_types)
    ids = [0] * n
    current = 0
    for i in range(1, n):
        if session_types[i - 1] == "RTH" and session_types[i] != "RTH":
            current += 1
        ids[i] = current
    return ids


def find_confirmed_cvd_swings(
    bars: pl.DataFrame,
    *,
    swing_left_bars: int,
    swing_right_bars: int,
    band_tolerance_ticks: int,
    tick_size: float,
) -> pl.DataFrame:
    """Detect confirmed swing highs and lows from completed delta bars.

    A swing high at bar *i* requires ``High[i] > High[j]`` strictly for
    every *j* in the left and right windows.  Same for swing lows with
    ``<``.  Ties disqualify.  Confirmation occurs when the last right-side
    bar closes.  Location is qualified against SD1/SD2 bands within
    *band_tolerance_ticks * tick_size* or beyond SD2.
    """
    n = len(bars)
    swings: list[dict] = []
    tol = band_tolerance_ticks * tick_size

    highs = bars["High"].to_list()
    lows = bars["Low"].to_list()
    sessions = bars["session_id"].to_list()

    for i in range(swing_left_bars, n - swing_right_bars):
        session = sessions[i]

        # Swing high check
        left_ok = all(
            sessions[j] == session and highs[i] > highs[j]
            for j in range(i - swing_left_bars, i)
        )
        right_ok = all(
            sessions[j] == session and highs[i] > highs[j]
            for j in range(i + 1, i + swing_right_bars + 1)
        )
        if left_ok and right_ok:
            bar = bars.row(i, named=True)
            conf = bars.row(i + swing_right_bars, named=True)
            h = bar["High"]
            h_sd1 = bar["high_vwap_sd1_top"]
            h_sd2 = bar["high_vwap_sd2_top"]
            loc = (
                abs(h - h_sd1) <= tol
                or abs(h - h_sd2) <= tol
                or h > h_sd2
            )
            swings.append({
                "swing_type": "high",
                "price": h,
                "cvd": bar["cvd"],
                "extreme_index": bar["high_index"],
                "extreme_bar_index": bar["bar_index"],
                "confirmation_bar_index": conf["bar_index"],
                "session_id": session,
                "high_ask": bar["high_ask"],
                "high_vwap_sd1_top": h_sd1,
                "high_vwap_sd2_top": h_sd2,
                "VA_Areas": bar["high_VA_Areas"],
                "LVN": bar["high_LVN"],
                "ValleysPeaks": bar["high_ValleysPeaks"],
                "location_qualified": loc,
            })

        # Swing low check
        left_ok = all(
            sessions[j] == session and lows[i] < lows[j]
            for j in range(i - swing_left_bars, i)
        )
        right_ok = all(
            sessions[j] == session and lows[i] < lows[j]
            for j in range(i + 1, i + swing_right_bars + 1)
        )
        if left_ok and right_ok:
            bar = bars.row(i, named=True)
            conf = bars.row(i + swing_right_bars, named=True)
            lo = bar["Low"]
            lo_sd1 = bar["low_vwap_sd1_bottom"]
            lo_sd2 = bar["low_vwap_sd2_bottom"]
            loc = (
                abs(lo - lo_sd1) <= tol
                or abs(lo - lo_sd2) <= tol
                or lo < lo_sd2
            )
            swings.append({
                "swing_type": "low",
                "price": lo,
                "cvd": bar["cvd"],
                "extreme_index": bar["low_index"],
                "extreme_bar_index": bar["bar_index"],
                "confirmation_bar_index": conf["bar_index"],
                "session_id": session,
                "low_bid": bar["low_bid"],
                "low_vwap_sd1_bottom": lo_sd1,
                "low_vwap_sd2_bottom": lo_sd2,
                "VA_Areas": bar["low_VA_Areas"],
                "LVN": bar["low_LVN"],
                "ValleysPeaks": bar["low_ValleysPeaks"],
                "location_qualified": loc,
            })

    if not swings:
        return _empty_swings()

    return pl.DataFrame(swings)


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "signal_index": pl.Int64,
        "Index": pl.Int64,
        "signal_direction": pl.Utf8,
        "entry_price": pl.Float64,
        "stop_loss": pl.Float64,
        "target_type": pl.Utf8,
        "variant": pl.Utf8,
        "extreme_index": pl.Int64,
        "extreme_bar_index": pl.UInt32,
        "confirmation_bar_index": pl.UInt32,
        "trigger_bar_index": pl.UInt32,
        "day_state": pl.Utf8,
        "VA_Areas": pl.Utf8,
        "LVN": pl.Int64,
        "ValleysPeaks": pl.Int64,
    })


def _check_worst_fill(
    trigger: dict,
    direction: str,
    slippage_max_ticks: int,
    tick_size: float,
    use_spread: bool,
) -> bool:
    nbo = trigger["next_bar_open"]
    nbv = trigger["next_bar_vwap"]
    if nbo is None or nbv is None:
        return False
    if use_spread:
        anchor = trigger["next_bar_ask"] if direction == "long" else trigger["next_bar_bid"]
        if anchor is None:
            return False
    else:
        anchor = nbo
    slip = slippage_max_ticks * tick_size
    if direction == "long":
        return anchor + slip < nbv
    return anchor - slip > nbv


def find_cvd_vwap_divergence_signals(
    bars: pl.DataFrame,
    swings: pl.DataFrame,
    *,
    min_cvd_divergence: float,
    stop_buffer_ticks: int,
    tick_size: float,
    slippage_max_ticks: int,
    use_spread: bool,
) -> pl.DataFrame:
    """Detect CVD/price divergence signals from confirmed swings.

    Compares each location-qualified swing with the most recent prior
    same-type swing in the same session.  Prior swings need not be
    location-qualified.
    """
    if len(bars) == 0 or len(swings) == 0:
        return _empty_signals()
    if "day_state" not in bars.columns:
        raise ValueError("bars must have day_state column")

    sw = swings.sort("confirmation_bar_index").to_dicts()
    bar_list = bars.sort("bar_index").to_dicts()
    n_bars = len(bar_list)

    next_same_conf: list[int | None] = [None] * len(sw)
    for i in range(len(sw)):
        for j in range(i + 1, len(sw)):
            if (sw[j]["swing_type"] == sw[i]["swing_type"]
                    and sw[j]["session_id"] == sw[i]["session_id"]):
                next_same_conf[i] = sw[j]["confirmation_bar_index"]
                break

    signals: list[dict] = []

    for si, current in enumerate(sw):
        if not current["location_qualified"]:
            continue

        prior = None
        for j in range(si - 1, -1, -1):
            if (sw[j]["swing_type"] == current["swing_type"]
                    and sw[j]["session_id"] == current["session_id"]):
                prior = sw[j]
                break
        if prior is None:
            continue

        stype = current["swing_type"]
        if stype == "low":
            if not (current["price"] < prior["price"]
                    and current["cvd"] >= prior["cvd"] + min_cvd_divergence):
                continue
            direction = "long"
        else:
            if not (current["price"] > prior["price"]
                    and current["cvd"] <= prior["cvd"] - min_cvd_divergence):
                continue
            direction = "short"

        conf_bar_idx = current["confirmation_bar_index"]
        session = current["session_id"]
        expiry = next_same_conf[si]

        trigger = None
        for bi in range(n_bars):
            b = bar_list[bi]
            if b["bar_index"] <= conf_bar_idx:
                continue
            if b["session_id"] != session:
                break
            if expiry is not None and b["bar_index"] >= expiry:
                break
            if direction == "long" and b["vwap_touched_from_below"]:
                break
            if direction == "short" and b["vwap_touched_from_above"]:
                break

            ds = b.get("day_state", "UNCLASSIFIED")
            if direction == "long" and ds == "TREND_DOWN":
                continue
            if direction == "short" and ds == "TREND_UP":
                continue

            if bi == 0:
                continue
            prev_b = bar_list[bi - 1]
            if direction == "long" and b["Close"] > prev_b["Close"] and b["bar_delta"] > 0:
                trigger = b
                break
            if direction == "short" and b["Close"] < prev_b["Close"] and b["bar_delta"] < 0:
                trigger = b
                break

        if trigger is None:
            continue
        if trigger["next_bar_first_index"] is None:
            continue
        if not _check_worst_fill(trigger, direction, slippage_max_ticks, tick_size, use_spread):
            continue

        stop = (current["low_bid"] - stop_buffer_ticks * tick_size if direction == "long"
                else current["high_ask"] + stop_buffer_ticks * tick_size)

        signals.append({
            "signal_index": trigger["last_index"],
            "Index": trigger["next_bar_first_index"],
            "signal_direction": direction,
            "entry_price": trigger["next_bar_open"],
            "stop_loss": stop,
            "target_type": "live_vwap",
            "variant": "cvd_vwap_divergence",
            "extreme_index": current["extreme_index"],
            "extreme_bar_index": current["extreme_bar_index"],
            "confirmation_bar_index": conf_bar_idx,
            "trigger_bar_index": trigger["bar_index"],
            "day_state": trigger["day_state"],
            "VA_Areas": current["VA_Areas"],
            "LVN": current["LVN"],
            "ValleysPeaks": current["ValleysPeaks"],
        })

    if not signals:
        return _empty_signals()
    return pl.DataFrame(signals).cast({
        "extreme_bar_index": pl.UInt32,
        "confirmation_bar_index": pl.UInt32,
        "trigger_bar_index": pl.UInt32,
    })


def find_cvd_vwap_climax_signals(
    bars: pl.DataFrame,
    *,
    climax_lookback_bars: int,
    min_climax_price_move: float,
    min_climax_cvd_move: float,
    stop_buffer_ticks: int,
    tick_size: float,
    slippage_max_ticks: int,
    use_spread: bool,
    band_tolerance_ticks: int,
) -> pl.DataFrame:
    """Detect CVD climax exhaustion signals.

    Price and CVD both make new window extremes after a sustained move,
    then reverse.  Allowed in any day state including trends.
    """
    if len(bars) == 0:
        return _empty_signals()
    if "day_state" not in bars.columns:
        raise ValueError("bars must have day_state column")

    bar_list = bars.sort("bar_index").to_dicts()
    n_bars = len(bar_list)
    tol = band_tolerance_ticks * tick_size
    signals: list[dict] = []

    for i in range(climax_lookback_bars, n_bars):
        bar = bar_list[i]
        session = bar["session_id"]

        window = []
        for j in range(i - climax_lookback_bars, i):
            if bar_list[j]["session_id"] == session:
                window.append(bar_list[j])
        if len(window) < climax_lookback_bars:
            continue

        w_lows = [w["Low"] for w in window]
        w_highs = [w["High"] for w in window]
        w_cvds = [w["cvd"] for w in window]

        # Bullish climax: new price low AND new CVD low
        if bar["Low"] < min(w_lows) and bar["cvd"] < min(w_cvds):
            price_move = max(w_highs + [bar["High"]]) - bar["Low"]
            cvd_move = max(w_cvds + [bar["cvd"]]) - bar["cvd"]
            if price_move >= min_climax_price_move and cvd_move >= min_climax_cvd_move:
                lo = bar["Low"]
                sd1 = bar["low_vwap_sd1_bottom"]
                sd2 = bar["low_vwap_sd2_bottom"]
                if abs(lo - sd1) <= tol or abs(lo - sd2) <= tol or lo < sd2:
                    sig = _find_climax_trigger(
                        bar_list, i, n_bars, session, "long", bar,
                        slippage_max_ticks, tick_size, use_spread, stop_buffer_ticks,
                    )
                    if sig is not None:
                        signals.append(sig)

        # Bearish climax: new price high AND new CVD high
        if bar["High"] > max(w_highs) and bar["cvd"] > max(w_cvds):
            price_move = bar["High"] - min(w_lows + [bar["Low"]])
            cvd_move = bar["cvd"] - min(w_cvds + [bar["cvd"]])
            if price_move >= min_climax_price_move and cvd_move >= min_climax_cvd_move:
                hi = bar["High"]
                sd1 = bar["high_vwap_sd1_top"]
                sd2 = bar["high_vwap_sd2_top"]
                if abs(hi - sd1) <= tol or abs(hi - sd2) <= tol or hi > sd2:
                    sig = _find_climax_trigger(
                        bar_list, i, n_bars, session, "short", bar,
                        slippage_max_ticks, tick_size, use_spread, stop_buffer_ticks,
                    )
                    if sig is not None:
                        signals.append(sig)

    if not signals:
        return _empty_signals()
    return pl.DataFrame(signals).cast({
        "extreme_bar_index": pl.UInt32,
        "confirmation_bar_index": pl.UInt32,
        "trigger_bar_index": pl.UInt32,
    })


def _find_climax_trigger(
    bar_list: list[dict],
    extreme_pos: int,
    n_bars: int,
    session: int,
    direction: str,
    extreme_bar: dict,
    slippage_max_ticks: int,
    tick_size: float,
    use_spread: bool,
    stop_buffer_ticks: int,
) -> dict | None:
    for bi in range(extreme_pos + 1, n_bars):
        b = bar_list[bi]
        if b["session_id"] != session:
            break
        if direction == "long" and b["vwap_touched_from_below"]:
            break
        if direction == "short" and b["vwap_touched_from_above"]:
            break
        if direction == "long" and b["Low"] < extreme_bar["Low"]:
            break
        if direction == "short" and b["High"] > extreme_bar["High"]:
            break

        prev_b = bar_list[bi - 1]
        triggered = False
        if direction == "long" and b["Close"] > prev_b["Close"] and b["bar_delta"] > 0:
            triggered = True
        if direction == "short" and b["Close"] < prev_b["Close"] and b["bar_delta"] < 0:
            triggered = True
        if not triggered:
            continue

        if b["next_bar_first_index"] is None:
            return None
        if not _check_worst_fill(b, direction, slippage_max_ticks, tick_size, use_spread):
            return None

        stop = (extreme_bar["low_bid"] - stop_buffer_ticks * tick_size if direction == "long"
                else extreme_bar["high_ask"] + stop_buffer_ticks * tick_size)

        return {
            "signal_index": b["last_index"],
            "Index": b["next_bar_first_index"],
            "signal_direction": direction,
            "entry_price": b["next_bar_open"],
            "stop_loss": stop,
            "target_type": "live_vwap",
            "variant": "cvd_vwap_climax",
            "extreme_index": extreme_bar["low_index"] if direction == "long" else extreme_bar["high_index"],
            "extreme_bar_index": extreme_bar["bar_index"],
            "confirmation_bar_index": extreme_bar["bar_index"],
            "trigger_bar_index": b["bar_index"],
            "day_state": b["day_state"],
            "VA_Areas": extreme_bar["low_VA_Areas"] if direction == "long" else extreme_bar["high_VA_Areas"],
            "LVN": extreme_bar["low_LVN"] if direction == "long" else extreme_bar["high_LVN"],
            "ValleysPeaks": extreme_bar["low_ValleysPeaks"] if direction == "long" else extreme_bar["high_ValleysPeaks"],
        }
    return None


def _empty_swings() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "swing_type": pl.Utf8,
        "price": pl.Float64,
        "cvd": pl.Float64,
        "extreme_index": pl.Int64,
        "extreme_bar_index": pl.UInt32,
        "confirmation_bar_index": pl.UInt32,
        "session_id": pl.Int64,
        "high_ask": pl.Float64,
        "high_vwap_sd1_top": pl.Float64,
        "high_vwap_sd2_top": pl.Float64,
        "low_bid": pl.Float64,
        "low_vwap_sd1_bottom": pl.Float64,
        "low_vwap_sd2_bottom": pl.Float64,
        "VA_Areas": pl.Utf8,
        "LVN": pl.Int64,
        "ValleysPeaks": pl.Int64,
        "location_qualified": pl.Boolean,
    })


def _empty_bars() -> pl.DataFrame:
    """Return a schema-correct empty bars DataFrame."""
    return pl.DataFrame(schema={
        "bar_index": pl.UInt32,
        "session_id": pl.Int64,
        "OpenTime": pl.Datetime,
        "CloseTime": pl.Datetime,
        "Open": pl.Float64, "High": pl.Float64,
        "Low": pl.Float64, "Close": pl.Float64,
        "Volume": pl.Float64,
        "AskVolume": pl.Float64, "BidVolume": pl.Float64,
        "bar_delta": pl.Float64, "cvd": pl.Float64,
        "vwap": pl.Float64,
        "first_index": pl.Int64, "last_index": pl.Int64,
        "next_bar_first_index": pl.Int64, "next_bar_open": pl.Float64,
        "next_bar_bid": pl.Float64, "next_bar_ask": pl.Float64,
        "next_bar_vwap": pl.Float64,
        "high_index": pl.Int64, "high_ask": pl.Float64,
        "high_vwap_sd1_top": pl.Float64, "high_vwap_sd2_top": pl.Float64,
        "high_VA_Areas": pl.Utf8, "high_LVN": pl.Int64,
        "high_ValleysPeaks": pl.Int64,
        "low_index": pl.Int64, "low_bid": pl.Float64,
        "low_vwap_sd1_bottom": pl.Float64, "low_vwap_sd2_bottom": pl.Float64,
        "low_VA_Areas": pl.Utf8, "low_LVN": pl.Int64,
        "low_ValleysPeaks": pl.Int64,
        "vwap_touched_from_below": pl.Boolean,
        "vwap_touched_from_above": pl.Boolean,
    })
