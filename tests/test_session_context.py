"""Tests for causal session context: VWAP crosses, prior levels, running state, trend state."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from orderflow.market.large_order_flow import _count_vwap_crosses, _running_vwap_crosses
from orderflow.market.session_context import prior_rth_levels
from orderflow.market.session_context import running_session_state

TICK = 0.25


def session(date, prices, volumes=None, *, start="08:30:00", step_s=10, session_type="RTH",
            **extra):
    """One session of ticks, ``step_s`` seconds apart. ``extra`` holds per-tick column lists."""
    volumes = volumes or [1] * len(prices)
    t0 = datetime.fromisoformat(f"{date}T{start}")
    frame = pl.DataFrame({
        "Datetime": [t0 + timedelta(seconds=step_s * i) for i in range(len(prices))],
        "Date": [date] * len(prices),
        "Price": [float(p) for p in prices],
        "Volume": [int(v) for v in volumes],
        "SessionType": [session_type] * len(prices),
        **{k: [float(x) for x in v] for k, v in extra.items()},
    })
    return frame


def stack(*frames):
    return (pl.concat(frames, how="diagonal").sort("Datetime")
            .with_row_index("Index").with_columns(pl.col("Index").cast(pl.Int64)))


def test_prior_levels_first_session_null():
    ticks = stack(session("2025-09-15", [100, 101, 102]), session("2025-09-16", [103, 104]))
    levels = prior_rth_levels(ticks, tick_size=TICK)
    assert levels.height == 2
    first = levels.filter(pl.col("Date") == "2025-09-15")
    assert first.select(pl.all().exclude("Date").is_null().all()).row(0) == (True,) * 6


def test_prior_levels_carry_previous_session_high_low_close():
    ticks = stack(session("2025-09-15", [100, 102, 101]), session("2025-09-16", [110, 111]))
    row = prior_rth_levels(ticks, tick_size=TICK).filter(pl.col("Date") == "2025-09-16")
    assert row.select("prev_high", "prev_low", "prev_close").row(0) == (102.0, 100.0, 101.0)


def test_prior_poc_is_max_volume_price():
    ticks = stack(session("2025-09-15", [100, 100.25, 100.5], [5, 20, 3]),
                  session("2025-09-16", [101]))
    row = prior_rth_levels(ticks, tick_size=TICK).filter(pl.col("Date") == "2025-09-16")
    assert row["prev_poc"][0] == 100.25


def test_prior_value_area_exact_on_hand_profile():
    # volume by price 100.00:10  100.25:25  100.50:40  100.75:15  101.00:10, total 100.
    # POC 100.50 (40) -> below 25 > above 15, add 100.25 (65) -> above 15 > below 10,
    # add 100.75 (80 >= 70) -> VAL 100.25, VAH 100.75.
    prices = [100.0, 100.25, 100.5, 100.75, 101.0]
    ticks = stack(session("2025-09-15", prices, [10, 25, 40, 15, 10]),
                  session("2025-09-16", [101]))
    row = prior_rth_levels(ticks, tick_size=TICK).filter(pl.col("Date") == "2025-09-16")
    assert row.select("prev_vah", "prev_val").row(0) == (100.75, 100.25)


def test_prior_levels_ignore_eth_ticks():
    ticks = stack(session("2025-09-15", [100, 102]),
                  session("2025-09-15", [200], start="17:00:00", session_type="ETH"),
                  session("2025-09-16", [101]))
    row = prior_rth_levels(ticks, tick_size=TICK).filter(pl.col("Date") == "2025-09-16")
    assert row["prev_high"][0] == 102.0


def test_running_crosses_counts_confirmed_flips():
    price = np.array([2.0, -2.0, 0.5, 2.0])
    out = _running_vwap_crosses(price, np.zeros(4), confirm_distance=1.0)
    assert out.tolist() == [0, 1, 1, 2]


def test_running_crosses_prefix_invariant():
    rng = np.random.default_rng(0)
    price = rng.normal(0.0, 3.0, 500)
    vwap = np.zeros(500)
    full = _running_vwap_crosses(price, vwap, confirm_distance=1.0)
    for k in (1, 7, 250, 499):
        part = _running_vwap_crosses(price[:k], vwap[:k], confirm_distance=1.0)
        assert part.tolist() == full[:k].tolist()
    assert _count_vwap_crosses(price, vwap, confirm_distance=1.0) == int(full[-1])


def state_ticks(day2_prices, *, vwap=None, poc=None, prior=(100, 101, 99, 100)):
    """Session 1 sets prior levels; session 2 is the one under test (08:30 open, 1-minute ticks)."""
    n = len(day2_prices)
    vwap = vwap or [100.0] * n
    poc = poc or [100.0] * n
    s1 = session("2025-09-15", list(prior), step_s=60, vwap=[100.0] * len(prior),
                 vwap_sd1_top=[101.0] * len(prior), POC=[100.0] * len(prior))
    s2 = session("2025-09-16", day2_prices, step_s=60, vwap=vwap,
                 vwap_sd1_top=[v + 1.0 for v in vwap], POC=poc)
    return stack(s1, s2)


def day2(ticks, levels):
    return (running_session_state(ticks, levels, tick_size=TICK)
            .filter(pl.col("Date") == "2025-09-16"))


def test_state_running_extremes_and_minutes_since_open():
    ticks = state_ticks([100, 101, 99, 100])
    out = day2(ticks, prior_rth_levels(ticks, tick_size=TICK))
    assert out["run_high"].to_list() == [100, 101, 101, 101]
    assert out["run_low"].to_list() == [100, 100, 99, 99]
    assert out["minutes_since_open"].to_list() == [0.0, 1.0, 2.0, 3.0]


def test_state_is_causal():
    base = state_ticks([100, 101, 99, 100, 102])
    levels = prior_rth_levels(base, tick_size=TICK)
    before = day2(base, levels).head(3)
    altered = base.with_columns(
        pl.when(pl.col("Index") > before["Index"][2]).then(pl.lit(150.0))
        .otherwise(pl.col("Price")).alias("Price"))
    after = day2(altered, levels).head(3)
    assert before.equals(after)


@pytest.mark.parametrize("open_px,expected", [(100.0, "inside"), (102.0, "above"), (97.0, "below")])
def test_state_open_location_inside_above_below(open_px, expected):
    # prior session 100,101,99,100 at volume 1 -> value area spans 100..101
    ticks = state_ticks([open_px, open_px])
    out = day2(ticks, prior_rth_levels(ticks, tick_size=TICK))
    assert out["open_location"].to_list() == [expected, expected]


def test_state_returned_to_value_latches():
    ticks = state_ticks([102, 100.5, 103])
    out = day2(ticks, prior_rth_levels(ticks, tick_size=TICK))
    assert out["returned_to_value"].to_list() == [False, True, True]


def test_state_vwap_crosses_match_running_helper():
    prices = [100, 101, 99, 101, 100.1, 98]
    ticks = state_ticks(prices)
    out = day2(ticks, prior_rth_levels(ticks, tick_size=TICK))
    expected = _running_vwap_crosses(np.array(prices, float), np.full(6, 100.0),
                                     confirm_distance=2 * TICK)
    assert out["vwap_crosses"].to_list() == expected.tolist()
