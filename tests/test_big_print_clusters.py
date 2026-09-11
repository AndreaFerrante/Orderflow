"""Tests for big-print clusters: side sign first, then dedup, chaining, hold, context, books."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from orderflow.market.big_print_clusters import find_big_print_clusters, post_cluster_hold
from orderflow.market.big_print_clusters import cluster_context_asof

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


def test_qualify_datetime_matches_qualify_index_fill():
    us = 1e-6
    ticks = stack(prints([(0, 100.0, 60, 2), (5, 100.0, 60, 2), (5 + us, 100.25, 60, 2)]))
    out = clusters(ticks, n=2)
    assert out["qualify_index"][0] == ticks["Index"][2]
    assert out["qualify_datetime"][0] == ticks["Datetime"][2]


def with_bars(ticks):
    """Add current/next minute-bar columns and next_bar_open, as the enrichment does."""
    bars = ticks.with_columns(
        pl.col("Datetime").dt.truncate("1m").alias("current_bar_datetime"))
    bars = bars.with_columns(
        (pl.col("current_bar_datetime") + pl.duration(minutes=1)).alias("next_bar_datetime"))
    opens = (bars.group_by("current_bar_datetime").agg(pl.col("Price").first().alias("next_bar_open"))
             .rename({"current_bar_datetime": "next_bar_datetime"}))
    return bars.join(opens, on="next_bar_datetime", how="left").sort("Index")


def cluster_then(path_price, seconds=120):
    """Buy cluster at 100.0 (t=0, t=5), then one 1-lot tick per second at ``path_price``."""
    rows = [(0, 100.0, 60, 2), (5, 100.0, 60, 2)]
    rows += [(6 + i, path_price, 1, 2) for i in range(seconds)]
    return with_bars(stack(prints(rows)))


def hold(ticks):
    return post_cluster_hold(ticks, clusters(ticks), hold_s=30.0, tick_size=TICK)


def test_hold_matched_pair():
    held, failed = hold(cluster_then(100.25)), hold(cluster_then(99.5))
    assert (held["hold_passed"][0], held["hold_failed"][0]) == (True, False)
    assert (failed["hold_passed"][0], failed["hold_failed"][0]) == (False, True)


def test_hold_neither_flag_at_vwap():
    at = hold(cluster_then(100.0))
    assert (at["hold_passed"][0], at["hold_failed"][0]) == (False, False)


def test_trigger_after_hold_window():
    out = hold(cluster_then(100.25))
    assert out["trigger_datetime"][0] > out["qualify_datetime"][0] + timedelta(seconds=30)


def test_entry_is_first_tick_of_next_minute_bar():
    ticks = cluster_then(100.25)
    out = hold(ticks)
    trig = ticks.filter(pl.col("Index") == out["trigger_index"][0])
    entry = ticks.filter(pl.col("Index") == out["entry_index"][0])
    assert entry["current_bar_datetime"][0] == trig["next_bar_datetime"][0]
    assert out["entry_index"][0] == ticks.filter(
        pl.col("current_bar_datetime") == trig["next_bar_datetime"][0])["Index"].min()
    assert out["entry_price"][0] == trig["next_bar_open"][0]


def ctx_inputs(n=40, side=1, cvwap=100.0, run_high=101.0, run_low=99.0, vwap=100.0,
               sd1=101.0, levels=None, bid=10.0, ask=30.0):
    """Hand-built state/ticks for session 2025-09-16: one row per Index 0..n-1."""
    idx = list(range(n))
    state = pl.DataFrame({
        "Index": idx, "Date": ["2025-09-16"] * n,
        "minutes_since_open": [60.0 + i for i in idx],
        "run_high": [run_high] * n, "run_low": [run_low] * n,
        "open_price": [100.0] * n, "open_vwap": [100.0] * n, "open_poc": [100.0] * n,
        "vwap": [vwap] * n, "vwap_sd1_top": [sd1] * n, "POC": [100.0] * n,
        "vwap_crosses": [3] * n, "open_location": ["inside"] * n,
        "returned_to_value": [True] * n,
    })
    t0 = datetime(2025, 9, 16, 9, 0)
    ticks = pl.DataFrame({
        "Index": idx, "Datetime": [t0 + timedelta(minutes=i) for i in idx],
        "Date": ["2025-09-16"] * n, "Price": [100.0] * n, "Volume": [10] * n,
        "SessionType": ["RTH"] * n,
        **{f"BidDOM_{k}": [bid] * n for k in range(5)},
        **{f"AskDOM_{k}": [ask] * n for k in range(5)},
    })
    levels = pl.DataFrame({"Date": ["2025-09-16"], "prev_high": [110.0],
                           "prev_low": [90.0], "prev_close": [100.0], "prev_poc": [100.0],
                           "prev_vah": [108.0], "prev_val": [92.0]}) if levels is None else levels
    return state, ticks, levels


def one_cluster(trigger_index, *, side=1, cvwap=100.0, qualify_index=None, first_index=None):
    q = trigger_index - 1 if qualify_index is None else qualify_index
    return {"cluster_id": trigger_index, "first_index": q - 1 if first_index is None else first_index,
            "qualify_index": q, "trigger_index": trigger_index, "side": side,
            "cluster_vwap": cvwap, "gross_volume": 120.0, "Date": "2025-09-16"}


def context(rows, **kw):
    state, ticks, levels = ctx_inputs(**kw)
    return cluster_context_asof(pl.DataFrame(rows), state, levels, ticks, tick_size=TICK,
                                directional_slope_min=1.0, rotational_slope_max=0.5)


def test_context_joins_state_at_trigger_and_keeps_order():
    out = context([one_cluster(35), one_cluster(10)])
    assert out["trigger_index"].to_list() == [10, 35]
    assert out["minutes_since_open"].to_list() == [70.0, 95.0]


def test_side_extreme_distance_and_band_z():
    out = context([one_cluster(10, side=1), one_cluster(20, side=-1)],
                  run_high=101.0, run_low=99.5, vwap=100.0, sd1=100.5)
    assert out["dist_to_side_extreme_ticks"].to_list() == [4.0, 2.0]
    assert out["band_z"].to_list() == [0.0, 0.0]


def test_prior_level_touch_within_4_ticks_on_side():
    near = pl.DataFrame({"Date": ["2025-09-16"], "prev_high": [110.0], "prev_low": [90.0],
                         "prev_close": [100.0], "prev_poc": [100.0],
                         "prev_vah": [100.75], "prev_val": [92.0]})
    touch = context([one_cluster(10, side=1)], levels=near)
    far = context([one_cluster(10, side=1)])
    wrong_side = context([one_cluster(10, side=-1)], levels=near)
    assert (touch["prior_level_touch"][0], far["prior_level_touch"][0],
            wrong_side["prior_level_touch"][0]) == (True, False, False)


def test_book_imbalance_l5():
    out = context([one_cluster(10)], bid=10.0, ask=30.0)
    assert out["book_imbalance_l5"][0] == pytest.approx((50 - 150) / 200)


def test_absorption_ratio_uses_prior_sigma_and_volume():
    state, ticks, levels = ctx_inputs(n=40)
    prices = [100.0 + (0.25 if i % 2 else 0.0) for i in range(40)]   # 1-minute changes +-0.25
    ticks = ticks.with_columns(pl.Series("Price", prices))
    row = one_cluster(36, qualify_index=35, first_index=34)
    out = cluster_context_asof(pl.DataFrame([row]), state, levels, ticks, tick_size=TICK,
                               directional_slope_min=1.0, rotational_slope_max=0.5)
    window = np.array(prices[4:34])                                   # 30 minutes before first print
    sigma = np.std(np.diff(window), ddof=1)
    volume_before = 10.0 * 34
    expected = abs(prices[35] - prices[34]) / (1.0 * sigma * np.sqrt(120.0 / volume_before))
    assert out["absorption_ratio"][0] == pytest.approx(expected)


def test_location_bucket_priority():
    near = pl.DataFrame({"Date": ["2025-09-16"], "prev_high": [110.0], "prev_low": [90.0],
                         "prev_close": [100.0], "prev_poc": [100.0],
                         "prev_vah": [100.5], "prev_val": [92.0]})
    prior = context([one_cluster(10)], levels=near, run_high=100.5)      # touch and extreme
    extreme = context([one_cluster(10)], run_high=100.5)                 # extreme only
    beyond = context([one_cluster(10, cvwap=101.5)], run_high=110.0)     # band_z 1.5
    inside = context([one_cluster(10)], run_high=110.0)
    assert [f["location_bucket"][0] for f in (prior, extreme, beyond, inside)] == [
        "prior_level", "side_extreme", "beyond_1sd", "inside_1sd"]


def test_context_is_causal():
    state, ticks, levels = ctx_inputs(n=40)
    rows = pl.DataFrame([one_cluster(20)])
    kw = dict(tick_size=TICK, directional_slope_min=1.0, rotational_slope_max=0.5)
    before = cluster_context_asof(rows, state, levels, ticks, **kw)
    later_ticks = ticks.with_columns(pl.when(pl.col("Index") > 20).then(999.0)
                                     .otherwise(pl.col("Price")).alias("Price"))
    later_state = state.with_columns(pl.when(pl.col("Index") > 20).then(999.0)
                                     .otherwise(pl.col("run_high")).alias("run_high"))
    after = cluster_context_asof(rows, later_state, levels, later_ticks, **kw)
    assert before.equals(after)
