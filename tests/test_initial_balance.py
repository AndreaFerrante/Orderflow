"""Tests for the initial balance calculation."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from orderflow.market.profiles.initial_balance import compute_initial_balance


def make_session(date_str, prices, start="08:30:00", session_minutes=390,
                 session_type="RTH", start_index=0):
    """One tick per second from `start`, one row per price in `prices`.

    `session_minutes` controls only the session's total span: a final tick is
    appended that far after the open so the session is long enough not to be
    dropped as a partial.
    """
    open_dt = datetime.fromisoformat(f"{date_str}T{start}")
    rows = [
        {
            "Index": start_index + i,
            "Datetime": open_dt + timedelta(seconds=i),
            "Date": date_str,
            "Price": float(p),
            "SessionType": session_type,
        }
        for i, p in enumerate(prices)
    ]
    rows.append({
        "Index": start_index + len(prices),
        "Datetime": open_dt + timedelta(minutes=session_minutes),
        "Date": date_str,
        "Price": float(prices[-1]),
        "SessionType": session_type,
    })
    return pl.DataFrame(rows)


def test_ib_high_low_come_from_the_first_thirty_minutes_only():
    # Ticks 0..1799 are inside the 30-minute window (1 per second).
    inside = [5000.0] * 1800
    inside[10] = 5010.0   # the window's high
    inside[20] = 4990.0   # the window's low
    ticks = make_session("2025-09-15", inside)
    # A far more extreme print 45 minutes after the open must be ignored.
    late = pl.DataFrame([{
        "Index": 99_999,
        "Datetime": datetime(2025, 9, 15, 9, 15, 0),
        "Date": "2025-09-15",
        "Price": 5100.0,
        "SessionType": "RTH",
    }])
    ticks = pl.concat([ticks, late])

    ib = compute_initial_balance(ticks, window_minutes=30, tick_size=0.25)

    assert ib.height == 1
    assert ib["ib_high"][0] == 5010.0
    assert ib["ib_low"][0] == 4990.0
    assert ib["ib_range_ticks"][0] == pytest.approx(80.0)  # 20.0 / 0.25


def test_ib_window_is_anchored_on_the_sessions_own_first_tick():
    """A late open must still get a full 30-minute IB, not a truncated one."""
    prices = [5000.0] * 1800
    prices[5] = 5008.0
    ticks = make_session("2025-09-15", prices, start="09:15:00")

    ib = compute_initial_balance(ticks, window_minutes=30)

    assert ib["ib_high"][0] == 5008.0


def test_eth_ticks_are_excluded():
    rth = make_session("2025-09-15", [5000.0] * 1800)
    eth = make_session("2025-09-15", [5500.0] * 10, start="03:00:00",
                       session_type="ETH", start_index=50_000)

    ib = compute_initial_balance(pl.concat([rth, eth]), window_minutes=30)

    assert ib["ib_high"][0] == 5000.0


def test_session_shorter_than_the_window_is_dropped_not_clipped():
    """A partial IB is not an IB."""
    short = make_session("2025-09-15", [5000.0] * 10, session_minutes=12)

    ib = compute_initial_balance(short, window_minutes=30)

    assert ib.height == 0


def test_ib_end_index_is_the_last_index_inside_the_window():
    ticks = make_session("2025-09-15", [5000.0] * 1800)

    ib = compute_initial_balance(ticks, window_minutes=30)

    # Index 0..1799 are inside; the trailing session-span tick is not.
    assert ib["ib_end_index"][0] == 1799


def test_empty_input_returns_an_empty_frame_with_the_right_schema():
    empty = pl.DataFrame(
        schema={"Index": pl.Int64, "Datetime": pl.Datetime("us"),
                "Date": pl.Utf8, "Price": pl.Float64, "SessionType": pl.Utf8}
    )

    ib = compute_initial_balance(empty)

    assert ib.height == 0
    assert set(ib.columns) == {
        "Date", "ib_high", "ib_low", "ib_mid", "ib_range_ticks",
        "ib_end_index",
    }
