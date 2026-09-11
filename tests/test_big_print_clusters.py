"""Tests for big-print clusters: side sign first, then dedup, chaining, hold, context, books."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from orderflow.market.big_print_clusters import find_big_print_clusters

TICK = 0.25


def prints(rows, date="2025-09-15", start="10:00:00"):
    """rows: (seconds, price, volume, trade_type[, session_type])."""
    t0 = datetime.fromisoformat(f"{date}T{start}")
    return pl.DataFrame([
        {"Datetime": t0 + timedelta(seconds=r[0]), "Date": date, "Price": float(r[1]),
         "Volume": int(r[2]), "TradeType": int(r[3]),
         "SessionType": r[4] if len(r) > 4 else "RTH"}
        for r in rows
    ])


def stack(*frames):
    return (pl.concat(frames).sort("Datetime").with_row_index("Index")
            .with_columns(pl.col("Index").cast(pl.Int64)))


def clusters(ticks, size=50, window=10, n=2):
    return find_big_print_clusters(ticks, min_print_size=size, window_s=window, min_prints=n)


def test_cluster_side_buy_positive_sell_negative():
    ticks = stack(prints([(0, 100.0, 60, 2), (5, 100.25, 60, 2)], date="2025-09-15"),
                  prints([(0, 100.0, 60, 1), (5, 99.75, 60, 1)], date="2025-09-16"))
    assert clusters(ticks).sort("Date")["side"].to_list() == [1, -1]


def test_net_zero_cluster_emits_no_row():
    ticks = stack(prints([(0, 100.0, 60, 2), (5, 100.0, 60, 1)]))
    assert clusters(ticks).height == 0


def test_same_sweep_fills_count_once():
    us = 1e-6
    ticks = stack(prints([(0, 100.0, 60, 2), (us, 100.25, 60, 2), (2 * us, 100.5, 60, 2)]))
    assert clusters(ticks, n=2).height == 0


def test_window_chains_and_splits():
    chained = stack(prints([(0, 100, 60, 2), (8, 100, 60, 2), (16, 100, 60, 2)]))
    split = stack(prints([(0, 100, 60, 2), (8, 100, 60, 2), (30, 100, 60, 2)]))
    assert clusters(chained, n=3).height == 1
    assert clusters(split, n=3).height == 0


def test_one_row_per_cluster_at_qualifying_print():
    ticks = stack(prints([(0, 100, 60, 2), (3, 100, 60, 2), (6, 100, 60, 2), (9, 100, 60, 2)]))
    out = clusters(ticks, n=2)
    assert out.height == 1
    assert out["qualify_index"][0] == ticks["Index"][1]


def test_clusters_never_span_sessions():
    ticks = stack(prints([(0, 100, 60, 2), (6, 100, 60, 2, "ETH")], start="15:59:57"))
    assert clusters(ticks, n=2).height == 0


def test_far_and_origin_price_follow_the_push():
    buy = clusters(stack(prints([(0, 100.0, 60, 2), (5, 100.5, 100, 2)])))
    sell = clusters(stack(prints([(0, 100.5, 60, 1), (5, 100.0, 60, 1)])))
    assert (buy["far_price"][0], buy["origin_price"][0]) == (100.5, 100.0)
    assert (sell["far_price"][0], sell["origin_price"][0]) == (100.0, 100.5)
    assert buy["cluster_vwap"][0] == pytest.approx((100.0 * 60 + 100.5 * 100) / 160)
