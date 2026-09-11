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
