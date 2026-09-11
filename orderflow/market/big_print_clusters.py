"""Clusters of big aggressive prints, their post-cluster hold, context, and book selection.

TradeType 2 = aggressive buy (+1), TradeType 1 = aggressive sell (-1). RTH only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["find_big_print_clusters", "post_cluster_hold", "cluster_context_asof",
           "select_join_signals", "select_fade_signals", "attach_structural_exits"]

_CLUSTER_SCHEMA = {
    "cluster_id": pl.Int64, "first_index": pl.Int64, "qualify_index": pl.Int64,
    "qualify_datetime": pl.Datetime("us"), "n_prints": pl.Int64, "gross_volume": pl.Float64,
    "net_signed_volume": pl.Float64, "side": pl.Int64, "cluster_vwap": pl.Float64,
    "far_price": pl.Float64, "origin_price": pl.Float64, "Date": pl.String,
}


def find_big_print_clusters(ticks: pl.DataFrame, *, min_print_size: int, window_s: float,
                            min_prints: int) -> pl.DataFrame:
    """Detect clusters of big aggressive prints close in time.

    RTH-only. Ticks are labelled with aggressor run membership on the full RTH
    tape (before the size filter), so a swept order that crosses the size
    threshold on several fills counts once. Consecutive counted prints on the
    same ``Date`` chain together while the gap between them stays within
    ``window_s`` seconds; one row is emitted per chain, at its
    ``min_prints``-th counted print.

    Parameters
    ----------
    ticks : pl.DataFrame
        Tick data carrying ``Index``, ``Datetime``, ``Date``, ``Price``,
        ``Volume``, ``TradeType`` and ``SessionType``, in chronological order.
    min_print_size : int
        Minimum single-fill volume to count as a "big" fill.
    window_s : float
        Maximum gap, in seconds, between consecutive counted prints that still
        joins them into the same chain.
    min_prints : int
        Number of counted prints in a chain required to emit a cluster row.

    Returns
    -------
    pl.DataFrame
        One row per qualifying cluster, matching ``_CLUSTER_SCHEMA``.
    """
    from orderflow.market.microstructure.aggressor import assign_aggressor_order_ids

    rth = ticks.filter(pl.col("SessionType") == "RTH").sort("Index")
    if rth.height == 0:
        return pl.DataFrame(schema=_CLUSTER_SCHEMA)

    rth = assign_aggressor_order_ids(rth, gap_us=1)

    big = rth.filter(pl.col("Volume") >= min_print_size)
    if big.height == 0:
        return pl.DataFrame(schema=_CLUSTER_SCHEMA)

    signed = pl.when(pl.col("TradeType") == 2).then(pl.col("Volume")).otherwise(-pl.col("Volume"))

    # One counted print per (Date, aggressor_id): sum volume/signed volume over
    # the big fills sharing a run, time = first fill, price range = fills' min/max,
    # last index = last big fill's Index.
    counted = (
        big.with_columns(signed.alias("_signed"))
        .group_by(["Date", "aggressor_id"], maintain_order=True)
        .agg(
            pl.col("Index").first().alias("first_index"),
            pl.col("Index").last().alias("last_index"),
            pl.col("Datetime").first().alias("time"),
            pl.col("Datetime").last().alias("last_time"),
            pl.col("Volume").sum().alias("volume"),
            pl.col("_signed").sum().alias("signed_volume"),
            pl.col("Price").min().alias("price_min"),
            pl.col("Price").max().alias("price_max"),
            (pl.col("Price") * pl.col("Volume")).sum().alias("pv_sum"),
        )
        .sort(["Date", "time"])
    )

    rows = []
    cluster_id = 0
    for date, grp in counted.group_by("Date", maintain_order=True):
        grp = grp.sort("time")
        chain_first_index = None
        chain_time = None
        chain_gross = 0.0
        chain_net = 0.0
        chain_pv = 0.0
        chain_vol = 0.0
        chain_price_min = None
        chain_price_max = None
        chain_n = 0

        for r in grp.iter_rows(named=True):
            t = r["time"]
            new_chain = chain_time is None or (t - chain_time).total_seconds() > window_s
            if new_chain:
                chain_first_index = r["first_index"]
                chain_gross = 0.0
                chain_net = 0.0
                chain_pv = 0.0
                chain_vol = 0.0
                chain_price_min = None
                chain_price_max = None
                chain_n = 0

            chain_time = t
            chain_gross += r["volume"]
            chain_net += r["signed_volume"]
            chain_pv += r["pv_sum"]
            chain_vol += r["volume"]
            chain_price_min = r["price_min"] if chain_price_min is None else min(chain_price_min, r["price_min"])
            chain_price_max = r["price_max"] if chain_price_max is None else max(chain_price_max, r["price_max"])
            chain_n += 1

            if chain_n == min_prints:
                if chain_net != 0:
                    side = 1 if chain_net > 0 else -1
                    far_price = chain_price_max if side == 1 else chain_price_min
                    origin_price = chain_price_min if side == 1 else chain_price_max
                    rows.append({
                        "cluster_id": cluster_id,
                        "first_index": chain_first_index,
                        "qualify_index": r["last_index"],
                        "qualify_datetime": r["last_time"],
                        "n_prints": min_prints,
                        "gross_volume": chain_gross,
                        "net_signed_volume": chain_net,
                        "side": side,
                        "cluster_vwap": chain_pv / chain_vol,
                        "far_price": far_price,
                        "origin_price": origin_price,
                        "Date": date if isinstance(date, str) else date[0],
                    })
                    cluster_id += 1
                # chain_n only reaches min_prints once (it strictly increases),
                # so later prints of the same chain never re-qualify.

    if not rows:
        return pl.DataFrame(schema=_CLUSTER_SCHEMA)

    return pl.DataFrame(rows, schema=_CLUSTER_SCHEMA)


def post_cluster_hold(ticks: pl.DataFrame, clusters: pl.DataFrame, *, hold_s: float = 30.0,
                      tick_size: float) -> pl.DataFrame:
    """Test the 30 s post-cluster hold and map each cluster to its trigger and entry tick.

    RTH ticks only, matched to a cluster's own ``Date``. For each cluster, the
    hold window is ``(qualify_datetime, qualify_datetime + hold_s]``:
    ``hold_passed`` if price never traded ``tol`` against the cluster side and
    ended on-side; ``hold_failed`` if it ended more than ``tol`` against it. The
    trigger is the first RTH tick after the window closes; the entry is the
    first tick of the trigger's next minute bar, priced at ``next_bar_open``.
    Clusters with no trigger or no entry tick in the same session are dropped.
    """
    if clusters.height == 0:
        return clusters.with_columns(
            [pl.lit(None, dtype=pl.Boolean).alias(c) for c in ("hold_passed", "hold_failed")]
            + [pl.lit(None, dtype=pl.Int64).alias("trigger_index"),
               pl.lit(None, dtype=pl.Datetime("us")).alias("trigger_datetime"),
               pl.lit(None, dtype=pl.Float64).alias("trigger_price"),
               pl.lit(None, dtype=pl.Int64).alias("entry_index"),
               pl.lit(None, dtype=pl.Float64).alias("entry_price")])

    rth = ticks.filter(pl.col("SessionType") == "RTH").sort("Index")

    by_date: dict[str, dict[str, np.ndarray]] = {}
    for date, grp in rth.group_by("Date", maintain_order=True):
        d = date if isinstance(date, str) else date[0]
        grp = grp.sort("Index")
        by_date[d] = {
            "datetime": grp["Datetime"].to_numpy(),
            "datetime_us": grp["Datetime"].cast(pl.Int64).to_numpy(),
            "price": grp["Price"].to_numpy(),
            "index": grp["Index"].to_numpy(),
            "current_bar_us": grp["current_bar_datetime"].cast(pl.Int64).to_numpy(),
            "next_bar_us": grp["next_bar_datetime"].cast(pl.Int64).to_numpy(),
            "next_bar_open": grp["next_bar_open"].to_numpy(),
        }

    hold_passed, hold_failed = [], []
    trigger_index, trigger_datetime, trigger_price = [], [], []
    entry_index, entry_price = [], []
    for row in clusters.iter_rows(named=True):
        arrs = by_date.get(row["Date"])
        if arrs is None:
            hold_passed.append(False)
            hold_failed.append(False)
            trigger_index.append(None)
            trigger_datetime.append(None)
            trigger_price.append(None)
            entry_index.append(None)
            entry_price.append(None)
            continue

        dt = arrs["datetime_us"]
        qdt_us = np.datetime64(row["qualify_datetime"], "us").astype(np.int64)
        window_end_us = qdt_us + np.int64(hold_s * 1_000_000)
        lo = int(np.searchsorted(dt, qdt_us, side="right"))
        hi = int(np.searchsorted(dt, window_end_us, side="right"))

        if hi > lo:
            s = row["side"]
            cvwap = row["cluster_vwap"]
            tol = tick_size
            diffs = s * (arrs["price"][lo:hi] - cvwap)
            hold_passed.append(bool(diffs.min() >= -tol and diffs[-1] > 0))
            hold_failed.append(bool(diffs[-1] < -tol))
        else:
            hold_passed.append(False)
            hold_failed.append(False)

        if hi >= len(dt):
            trigger_index.append(None)
            trigger_datetime.append(None)
            trigger_price.append(None)
            entry_index.append(None)
            entry_price.append(None)
            continue

        trigger_index.append(int(arrs["index"][hi]))
        trigger_datetime.append(arrs["datetime"][hi].item())
        trigger_price.append(float(arrs["price"][hi]))

        next_bar_us = arrs["next_bar_us"][hi]
        cbd = arrs["current_bar_us"]
        epos = int(np.searchsorted(cbd, next_bar_us, side="left"))
        if epos < len(cbd) and cbd[epos] == next_bar_us:
            entry_index.append(int(arrs["index"][epos]))
            entry_price.append(float(arrs["next_bar_open"][hi]))
        else:
            entry_index.append(None)
            entry_price.append(None)

    out = clusters.with_columns(
        pl.Series("hold_passed", hold_passed, dtype=pl.Boolean),
        pl.Series("hold_failed", hold_failed, dtype=pl.Boolean),
        pl.Series("trigger_index", trigger_index, dtype=pl.Int64),
        pl.Series("trigger_datetime", trigger_datetime, dtype=pl.Datetime("us")),
        pl.Series("trigger_price", trigger_price, dtype=pl.Float64),
        pl.Series("entry_index", entry_index, dtype=pl.Int64),
        pl.Series("entry_price", entry_price, dtype=pl.Float64))
    return out.filter(pl.col("trigger_index").is_not_null() & pl.col("entry_index").is_not_null())


_CONTEXT_COLUMNS = {
    "minutes_since_open": pl.Float64, "dist_to_side_extreme_ticks": pl.Float64,
    "band_z": pl.Float64, "prior_level_touch": pl.Boolean, "trend_state": pl.String,
    "absorption_ratio": pl.Float64, "book_imbalance_l5": pl.Float64, "location_bucket": pl.String,
}


_MINUTE_US = 60_000_000


def _compute_absorption_ratio(clusters: pl.DataFrame, ticks: pl.DataFrame, *,
                              impact_constant: float) -> pl.Series:
    """Per-cluster square-root-law absorption ratio, causal as of the qualifying print.

    Builds per-Date numpy arrays once (RTH ticks' Index/Price/Datetime, cumulative
    volume, and minute-close prices), then loops over clusters using
    ``np.searchsorted`` for the sigma window and the volume-to-date lookup --
    no per-cluster filter of the whole tick frame.
    """
    from orderflow.market.microstructure.impact import compute_absorption_ratio

    rth = ticks.filter(pl.col("SessionType") == "RTH").sort("Index")
    by_date: dict[str, dict[str, np.ndarray]] = {}
    for date, grp in rth.group_by("Date", maintain_order=True):
        d = date if isinstance(date, str) else date[0]
        grp = grp.sort("Index")
        minute = (grp.with_columns(pl.col("Datetime").dt.truncate("1m").alias("_bucket"))
                  .group_by("_bucket", maintain_order=True)
                  .agg(pl.col("Price").last().alias("close"))
                  .sort("_bucket"))
        by_date[d] = {
            "index": grp["Index"].to_numpy(),
            "price": grp["Price"].to_numpy(),
            "datetime_us": grp["Datetime"].cast(pl.Int64).to_numpy(),
            "cum_volume": np.concatenate(
                [[0.0], np.cumsum(grp["Volume"].to_numpy().astype(np.float64))]),
            "minute_bucket_us": minute["_bucket"].cast(pl.Int64).to_numpy(),
            "minute_close": minute["close"].to_numpy(),
        }

    sigma_vals, volume_vals, price_start, price_end = [], [], [], []
    for row in clusters.iter_rows(named=True):
        arrs = by_date.get(row["Date"])
        first_idx, qualify_idx = row["first_index"], row["qualify_index"]
        if arrs is None or first_idx is None or qualify_idx is None:
            sigma_vals.append(None)
            volume_vals.append(None)
            price_start.append(None)
            price_end.append(None)
            continue

        idx_arr = arrs["index"]
        pos_first = int(np.searchsorted(idx_arr, first_idx, side="left"))
        pos_qualify = int(np.searchsorted(idx_arr, qualify_idx, side="left"))
        if (pos_first >= len(idx_arr) or idx_arr[pos_first] != first_idx
                or pos_qualify >= len(idx_arr) or idx_arr[pos_qualify] != qualify_idx):
            sigma_vals.append(None)
            volume_vals.append(None)
            price_start.append(None)
            price_end.append(None)
            continue

        volume_vals.append(float(arrs["cum_volume"][pos_first]))
        price_start.append(float(arrs["price"][pos_first]))
        price_end.append(float(arrs["price"][pos_qualify]))

        first_datetime_us = int(arrs["datetime_us"][pos_first])
        window_start_us = first_datetime_us - 30 * _MINUTE_US
        bucket_us = arrs["minute_bucket_us"]
        lo = int(np.searchsorted(bucket_us, window_start_us, side="left"))
        # Exclude the minute bucket containing the first print itself: its close is the last
        # tick of that minute, which can fall after the first print (and even after the trigger
        # tick) -- including it would be lookahead. Only minutes closed strictly before count.
        hi = int(np.searchsorted(bucket_us, first_datetime_us - _MINUTE_US, side="right"))
        window = arrs["minute_close"][lo:hi]
        sigma_vals.append(float(np.std(np.diff(window), ddof=1)) if window.size >= 3 else None)

    orders = pl.DataFrame({
        "aggressor_size": clusters["gross_volume"].to_list(),
        "aggressor_price_start": pl.Series(price_start, dtype=pl.Float64),
        "aggressor_price_end": pl.Series(price_end, dtype=pl.Float64),
    })
    result = compute_absorption_ratio(
        orders,
        sigma=pl.Series(sigma_vals, dtype=pl.Float64),
        session_volume_to_date=pl.Series(volume_vals, dtype=pl.Float64),
        impact_constant=impact_constant,
    )
    return result["absorption_ratio"].alias("absorption_ratio")


def cluster_context_asof(clusters: pl.DataFrame, state: pl.DataFrame, levels: pl.DataFrame,
                         ticks: pl.DataFrame, *, tick_size: float, directional_slope_min: float,
                         rotational_slope_max: float, impact_constant: float = 1.0) -> pl.DataFrame:
    """Attach as-of context to each cluster's trigger tick. Strictly causal."""
    if clusters.height == 0:
        return clusters.with_columns([pl.lit(None, dtype=t).alias(c)
                                      for c, t in _CONTEXT_COLUMNS.items()])

    cluster_cols = clusters.columns

    state_cols = ["Index", "minutes_since_open", "run_high", "run_low", "vwap", "vwap_sd1_top",
                  "POC", "open_vwap", "open_poc", "vwap_crosses", "open_location", "returned_to_value"]
    joined = clusters.join(state.select(state_cols), left_on="trigger_index", right_on="Index",
                           how="left")

    side = pl.col("side")
    cvwap = pl.col("cluster_vwap")
    dist_to_side_extreme_ticks = (
        pl.when(side == 1).then((pl.col("run_high") - cvwap) / tick_size)
        .otherwise((cvwap - pl.col("run_low")) / tick_size)
    )
    band_denom = pl.col("vwap_sd1_top") - pl.col("vwap")
    band_z = pl.when(band_denom > 0).then((cvwap - pl.col("vwap")) / band_denom).otherwise(None)
    joined = joined.with_columns(
        dist_to_side_extreme_ticks.alias("dist_to_side_extreme_ticks"),
        band_z.alias("band_z"),
    )

    level_cols = ["Date", "prev_high", "prev_low", "prev_vah", "prev_val"]
    joined = joined.join(levels.select(level_cols), on="Date", how="left")

    def _level_touch(level_col: str) -> pl.Expr:
        level = pl.col(level_col)
        buy = (level - cvwap >= 0) & (level - cvwap <= 4 * tick_size)
        sell = (cvwap - level >= 0) & (cvwap - level <= 4 * tick_size)
        return pl.when(level.is_null()).then(False).when(side == 1).then(buy).otherwise(sell)

    prior_level_touch = (
        _level_touch("prev_high") | _level_touch("prev_low")
        | _level_touch("prev_vah") | _level_touch("prev_val")
    )
    joined = joined.with_columns(prior_level_touch.fill_null(False).alias("prior_level_touch"))

    dom_cols = ["Index"] + [f"BidDOM_{k}" for k in range(5)] + [f"AskDOM_{k}" for k in range(5)]
    joined = joined.join(ticks.select(dom_cols), left_on="qualify_index", right_on="Index",
                         how="left")
    bid_sum = sum(pl.col(f"BidDOM_{k}") for k in range(5))
    ask_sum = sum(pl.col(f"AskDOM_{k}") for k in range(5))
    book_total = bid_sum + ask_sum
    book_imbalance_l5 = pl.when(book_total == 0).then(None).otherwise((bid_sum - ask_sum) / book_total)
    joined = joined.with_columns(book_imbalance_l5.alias("book_imbalance_l5"))

    joined = joined.with_columns(_compute_absorption_ratio(joined, ticks, impact_constant=impact_constant))

    from orderflow.market.session_context import classify_trend_state
    trend_state = classify_trend_state(joined, directional_slope_min=directional_slope_min,
                                       rotational_slope_max=rotational_slope_max)
    joined = joined.with_columns(trend_state)

    location_bucket = (
        pl.when(pl.col("prior_level_touch")).then(pl.lit("prior_level"))
        .when(pl.col("dist_to_side_extreme_ticks") <= 4).then(pl.lit("side_extreme"))
        .when(pl.col("band_z").abs() > 1).then(pl.lit("beyond_1sd"))
        .otherwise(pl.lit("inside_1sd"))
    )
    joined = joined.with_columns(location_bucket.alias("location_bucket"))

    out_cols = cluster_cols + list(_CONTEXT_COLUMNS.keys())
    joined = joined.sort("trigger_index")
    return joined.select(out_cols)


def select_join_signals(ctx: pl.DataFrame, *, band_z_max: float = 1.0, min_extreme_ticks: float = 8,
                        window_ct: tuple = ("09:00", "15:00")) -> pl.DataFrame:
    start_h, start_m = (int(x) for x in window_ct[0].split(":"))
    end_h, end_m = (int(x) for x in window_ct[1].split(":"))
    t = pl.col("trigger_datetime").dt.time()
    in_window = (t >= pl.time(start_h, start_m)) & (t <= pl.time(end_h, end_m))
    mask = (
        (((pl.col("trend_state") == "TREND_UP") & (pl.col("side") == 1))
         | ((pl.col("trend_state") == "TREND_DOWN") & (pl.col("side") == -1)))
        & (pl.col("band_z").abs() <= band_z_max)
        & (pl.col("dist_to_side_extreme_ticks") >= min_extreme_ticks)
        & pl.col("hold_passed")
        & in_window
    )
    return (ctx.filter(mask)
            .with_columns(pl.col("side").cast(pl.Int64).alias("direction"),
                         pl.lit("join").alias("book")))


def select_fade_signals(ctx: pl.DataFrame, *, extreme_ticks: float = 4,
                        window_ct: tuple = ("09:00", "15:00")) -> pl.DataFrame:
    start_h, start_m = (int(x) for x in window_ct[0].split(":"))
    end_h, end_m = (int(x) for x in window_ct[1].split(":"))
    t = pl.col("trigger_datetime").dt.time()
    in_window = (t >= pl.time(start_h, start_m)) & (t <= pl.time(end_h, end_m))
    prior_touch = pl.col("prior_level_touch").fill_null(False)
    absorption_low = (pl.col("absorption_ratio") < 1).fill_null(False)
    mask = (
        (pl.col("trend_state") == "RANGE")
        & ((pl.col("dist_to_side_extreme_ticks") <= extreme_ticks) | prior_touch)
        & (pl.col("hold_failed") | absorption_low)
        & in_window
    )
    return (ctx.filter(mask)
            .with_columns((-pl.col("side")).cast(pl.Int64).alias("direction"),
                         pl.lit("fade").alias("book")))


def attach_structural_exits(signals: pl.DataFrame, *, tick_size: float, buffer_ticks: float = 2,
                            min_sl_ticks: float = 4, max_sl_ticks: float = 40,
                            rr: float = 2.0) -> pl.DataFrame:
    # ponytail: abs() sizes the stop distance only; a structural level on the profit
    # side of entry still yields a loss-side stop at that distance -- drop such rows
    # if it shows up in the trades.
    stop = (
        pl.when(pl.col("book") == "join")
        .then(pl.col("origin_price") - pl.col("direction") * buffer_ticks * tick_size)
        .otherwise(pl.col("far_price") + pl.col("side") * buffer_ticks * tick_size)
    )
    sl_ticks = ((pl.col("entry_price") - stop).abs() / tick_size).round(0)
    out = signals.with_columns(sl_ticks.alias("SL_Ticks"))
    out = out.filter(pl.col("SL_Ticks") <= max_sl_ticks)
    out = out.with_columns(pl.col("SL_Ticks").clip(lower_bound=min_sl_ticks).alias("SL_Ticks"))
    out = out.with_columns((rr * pl.col("SL_Ticks")).alias("TP_Ticks"))
    return out
