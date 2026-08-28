"""Tests for initial balance breakout detection.

The first test is the direction test, per CLAUDE.md section 10 rule 4.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from orderflow.market.profiles.initial_balance import (
    compute_initial_balance,
    find_ib_breakouts,
)


def build_day(
    date_str="2025-09-15",
    ib_prices=None,
    after_prices=None,
    after_cvd_step=1.0,
    after_start_minutes=31,
    session_minutes=390,
):
    """A single RTH session: a 30-minute IB then a post-IB sequence.

    `after_cvd_step` is the per-tick change in (CD_Ask - CD_Bid) after the
    IB window: positive means buy aggression, negative means sell aggression.
    Inside the IB window CVD is flat.
    """
    ib_prices = [5000.0] * 1800 if ib_prices is None else ib_prices
    after_prices = [] if after_prices is None else after_prices
    open_dt = datetime.fromisoformat(f"{date_str}T08:30:00")

    rows = []
    # Levels start high and equal, so a naive "level" reading of CVD would
    # look bullish; only the *difference* is informative.
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


def test_upside_break_with_buy_aggression_is_direction_plus_one():
    """THE sign test. Break above the IB high with buy pressure is a LONG."""
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 20,   # 5001 > ib_high 5000
        after_cvd_step=+1.0,
    )

    out = run(ticks)

    assert out.height == 1
    assert out["direction"][0] == 1
    assert out["cvd_delta"][0] > 0


def test_downside_break_with_sell_aggression_is_direction_minus_one():
    ticks = build_day(
        after_prices=[5000.0] * 20 + [4999.0] * 20,   # below ib_low 5000
        after_cvd_step=-1.0,
    )

    out = run(ticks)

    assert out.height == 1
    assert out["direction"][0] == -1
    assert out["cvd_delta"][0] < 0


def test_break_against_the_flow_is_rejected():
    """Upside break while sellers are aggressing: a poke, not an extension."""
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 20,
        after_cvd_step=-1.0,
    )

    out = run(ticks)

    assert out.height == 0


def test_cvd_delta_is_a_difference_not_a_level():
    """CD_Ask/CD_Bid never reset; a large inherited level must not qualify."""
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 20,
        after_cvd_step=0.0,   # flat pressure, huge inherited levels
    )

    out = run(ticks, cvd_min_delta=0.5)

    assert out.height == 0


def test_signal_index_is_the_breaking_tick_and_entry_is_the_next_tick():
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 20,
        after_cvd_step=+1.0,
    )

    out = run(ticks)

    signal_index = out["signal_index"][0]
    assert out["entry_index"][0] == signal_index + 1
    price_at_signal = (
        ticks.filter(pl.col("Index") == signal_index)["Price"][0]
    )
    assert price_at_signal == 5001.0


def test_at_most_one_long_and_one_short_per_session():
    """Repeated breaks of the same edge produce exactly one signal."""
    ticks = build_day(
        after_prices=([5001.0] * 10 + [5000.0] * 10) * 5,
        after_cvd_step=+1.0,
    )

    out = run(ticks)

    assert out.height == 1
    assert out["direction"][0] == 1


def test_a_session_can_produce_one_break_in_each_direction():
    ticks = build_day(
        after_prices=[5001.0] * 10 + [4999.0] * 10,
        after_cvd_step=+1.0,
    )
    # Buy pressure throughout gates out the short, so gate on sign only and
    # allow both: use cvd_min_delta below zero magnitude via a flow that
    # reverses is covered elsewhere; here assert the per-direction cap only.
    out = run(ticks, cvd_min_delta=-1e9)

    assert sorted(out["direction"].to_list()) == [-1, 1]
    assert out.height == 2


def test_breaks_at_or_after_the_cutoff_hour_are_dropped():
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 20,
        after_cvd_step=+1.0,
        after_start_minutes=6 * 60,   # 14:30, past the 14:00 cutoff
    )

    out = run(ticks, entry_cutoff_hour=14)

    assert out.height == 0


def test_ticks_inside_the_ib_window_never_produce_a_signal():
    """Only Index > ib_end_index is scanned."""
    ib_prices = [5000.0] * 1800
    ib_prices[900] = 5010.0   # sets the high from inside the window
    ticks = build_day(ib_prices=ib_prices, after_prices=[5005.0] * 20,
                      after_cvd_step=+1.0)

    out = run(ticks)

    # 5005 is below the 5010 IB high, so nothing broke.
    assert out.height == 0


def test_output_is_causal_under_truncation():
    """Signals computed on a truncated frame match those on the full frame.

    This is the mechanical form of "no lookahead": if any operation read a
    future tick, truncating after the signal would change the answer.
    """
    ticks = build_day(
        after_prices=[5000.0] * 20 + [5001.0] * 200,
        after_cvd_step=+1.0,
    )
    full = run(ticks)
    cut_at = int(full["entry_index"][0])
    truncated = run(ticks.filter(pl.col("Index") <= cut_at))

    assert truncated.height == 1
    assert truncated["signal_index"][0] == full["signal_index"][0]
    assert truncated["direction"][0] == full["direction"][0]
    assert truncated["cvd_delta"][0] == pytest.approx(full["cvd_delta"][0])


def test_empty_ib_returns_an_empty_frame():
    ticks = build_day(after_prices=[5001.0] * 10, after_cvd_step=+1.0)
    empty_ib = compute_initial_balance(
        ticks.filter(pl.col("SessionType") == "ETH")
    )

    out = find_ib_breakouts(ticks, empty_ib, cvd_lookback_ticks=10)

    assert out.height == 0
    assert "direction" in out.columns


def test_short_session_warns_and_drops_the_break_silently_otherwise():
    """Fewer than cvd_lookback_ticks of session history -> no cvd_delta.

    The break is genuine (price clears ib_high with buy aggression) but the
    session doesn't have enough ticks for the lookback to produce a value,
    so cvd_delta is null for the whole session and the break is dropped.
    That drop must be observable, not silent.
    """
    ticks = build_day(
        ib_prices=[5000.0] * 3,
        after_prices=[5000.0] * 5 + [5001.0] * 5,
        after_cvd_step=+1.0,
    )
    # Whole session is 3 + 10 + 1 = 14 ticks, well under the lookback.
    assert ticks.height < 50

    with pytest.warns(UserWarning, match="cvd_lookback_ticks"):
        out = run(ticks, cvd_lookback_ticks=50)

    assert out.height == 0
