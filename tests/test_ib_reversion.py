"""Tests for the initial-balance reversion inputs.

The reversion trade fades an initial-balance break back toward the middle of
the range, so it needs three things the breakout trade never did: the
midpoint itself, the price it would actually be filled at, and the distance
between them.  It also needs the *opposite* order-flow gate -- a break that
runs without supporting flow is the one that fails.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from orderflow.market.profiles.initial_balance import (
    compute_initial_balance,
    find_ib_breakouts,
)

# ib_high 5002, ib_low 4998 -> mid 5000.0, range 16 ticks at 0.25.
IB_PRICES = ([4998.0, 5002.0] * 900)


def build_day(
    date_str="2025-09-15",
    ib_prices=None,
    after_prices=None,
    after_cvd_step=1.0,
    after_start_minutes=31,
    session_minutes=390,
):
    """One RTH session: a 30-minute IB window then a post-IB sequence."""
    ib_prices = IB_PRICES if ib_prices is None else ib_prices
    after_prices = [] if after_prices is None else after_prices
    open_dt = datetime.fromisoformat(f"{date_str}T08:30:00")

    rows = []
    ask, bid = 100_000.0, 100_000.0
    for i, p in enumerate(ib_prices):
        rows.append({
            "Index": i, "Datetime": open_dt + timedelta(seconds=i),
            "Date": date_str, "Price": float(p), "SessionType": "RTH",
            "CD_Ask": ask, "CD_Bid": bid,
        })

    base = len(ib_prices)
    after_open = open_dt + timedelta(minutes=after_start_minutes)
    for i, p in enumerate(after_prices):
        if after_cvd_step >= 0:
            ask += after_cvd_step
        else:
            bid += -after_cvd_step
        rows.append({
            "Index": base + i, "Datetime": after_open + timedelta(seconds=i),
            "Date": date_str, "Price": float(p), "SessionType": "RTH",
            "CD_Ask": ask, "CD_Bid": bid,
        })

    rows.append({
        "Index": base + len(after_prices),
        "Datetime": open_dt + timedelta(minutes=session_minutes),
        "Date": date_str, "Price": float(rows[-1]["Price"]),
        "SessionType": "RTH", "CD_Ask": ask, "CD_Bid": bid,
    })
    return pl.DataFrame(rows)


def run(ticks, **kwargs):
    ib = compute_initial_balance(ticks, window_minutes=30, tick_size=0.25)
    kwargs.setdefault("cvd_lookback_ticks", 10)
    return find_ib_breakouts(ticks, ib, **kwargs)


# --------------------------------------------------------------------------
# the midpoint
# --------------------------------------------------------------------------

def test_initial_balance_carries_its_midpoint():
    ib = compute_initial_balance(build_day(), window_minutes=30, tick_size=0.25)
    assert ib["ib_mid"].to_list() == [5000.0]


def test_midpoint_is_not_the_mean_of_the_ticks():
    """Mid is (high + low) / 2 -- the range's centre, not the average trade.

    The session below spends nearly all its time at 4998, so a volume- or
    tick-weighted centre would sit near 4998. Reversion targets the geometric
    centre of the balance area, which is 5000.
    """
    ticks = build_day(ib_prices=[4998.0] * 1799 + [5002.0])
    ib = compute_initial_balance(ticks, window_minutes=30, tick_size=0.25)
    assert ib["ib_mid"].to_list() == [5000.0]


# --------------------------------------------------------------------------
# fill price and the distance the trade has to travel
# --------------------------------------------------------------------------

def test_entry_price_is_the_price_of_the_entry_tick():
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0, 5004.0, 5004.0])
    out = run(ticks, flow_gate="off")
    row = out.filter(pl.col("direction") == 1).row(0, named=True)
    entry = ticks.filter(pl.col("Index") == row["entry_index"]).row(0, named=True)
    assert row["entry_price"] == entry["Price"] == 5004.0


def test_mid_distance_is_entry_to_midpoint_in_ticks():
    """5004 fill, 5000 mid, 0.25 tick -> 16 ticks to travel."""
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0, 5004.0, 5004.0])
    out = run(ticks, flow_gate="off")
    row = out.filter(pl.col("direction") == 1).row(0, named=True)
    assert row["mid_distance_ticks"] == 16.0


def test_mid_distance_is_positive_below_the_range_too():
    """A downside break travels *up* to the mid; distance is a magnitude."""
    ticks = build_day(
        after_prices=[5000.0] * 20 + [4997.0, 4996.0, 4996.0],
        after_cvd_step=-1.0,
    )
    out = run(ticks, flow_gate="off")
    row = out.filter(pl.col("direction") == -1).row(0, named=True)
    assert row["mid_distance_ticks"] == 16.0


def test_mid_distance_scales_with_tick_size():
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0, 5004.0, 5004.0])
    ib = compute_initial_balance(ticks, window_minutes=30, tick_size=0.5)
    out = find_ib_breakouts(
        ticks, ib, cvd_lookback_ticks=10, flow_gate="off", tick_size=0.5
    )
    assert out.filter(pl.col("direction") == 1).row(0, named=True)["mid_distance_ticks"] == 8.0


# --------------------------------------------------------------------------
# the flow gate
# --------------------------------------------------------------------------

def test_divergent_gate_takes_an_upside_break_on_sell_pressure():
    """Price above the IB high while delta falls: the break has no buyers."""
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20, after_cvd_step=-1.0)
    out = run(ticks, flow_gate="divergent")
    assert out["direction"].to_list() == [1]
    assert out["cvd_delta"][0] < 0


def test_divergent_gate_rejects_an_upside_break_on_buy_pressure():
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20, after_cvd_step=+1.0)
    assert run(ticks, flow_gate="divergent").height == 0


def test_divergent_gate_takes_a_downside_break_on_buy_pressure():
    ticks = build_day(after_prices=[5000.0] * 20 + [4997.0] * 20, after_cvd_step=+1.0)
    out = run(ticks, flow_gate="divergent")
    assert out["direction"].to_list() == [-1]
    assert out["cvd_delta"][0] > 0


def test_confirming_gate_is_still_the_default():
    """The breakout strategy's behaviour must not change under it."""
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20, after_cvd_step=+1.0)
    assert run(ticks)["direction"].to_list() == [1]
    assert run(ticks, flow_gate="confirming")["direction"].to_list() == [1]


def test_off_gate_takes_the_break_whatever_the_flow():
    for step in (+1.0, -1.0):
        ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20, after_cvd_step=step)
        assert run(ticks, flow_gate="off")["direction"].to_list() == [1]


def test_divergent_gate_respects_cvd_min_delta():
    """A 3-contract divergence does not clear a 100-contract threshold."""
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20, after_cvd_step=-0.3)
    assert run(ticks, flow_gate="divergent", cvd_min_delta=100.0).height == 0
    assert run(ticks, flow_gate="divergent", cvd_min_delta=0.0).height == 1


def test_unknown_flow_gate_raises():
    ticks = build_day(after_prices=[5000.0] * 20 + [5003.0] * 20)
    with pytest.raises(ValueError, match="flow_gate"):
        run(ticks, flow_gate="confirmed")
