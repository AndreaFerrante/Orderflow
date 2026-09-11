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
