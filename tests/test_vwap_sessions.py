"""Tests of the clean-session filter and the prior session range: synthetic tapes only."""

from datetime import datetime, timedelta

import polars as pl

from orderflow.market.vwap_reversion import find_clean_sessions, prior_session_range

SCHEMA = {"Date": pl.Utf8, "Datetime": pl.Datetime("us"), "SessionType": pl.Utf8,
          "Session_Volume": pl.Float64, "Price": pl.Float64}


def _minutes(date, start, end):
    t, stop = datetime.fromisoformat(f"{date} {start}"), datetime.fromisoformat(f"{date} {end}")
    while t <= stop:
        yield t
        t += timedelta(minutes=1)


def rth_day(date, start="08:00", end="13:29", drop=None, price=lambda n: 100.0):
    """One tick a minute; Session_Volume 10, 20, ... ; `drop=("10:01","10:11")` removes those ticks inclusive."""
    lo, hi = (datetime.fromisoformat(f"{date} {h}") for h in drop) if drop else (None, None)
    return pl.DataFrame([
        {"Date": date, "Datetime": t, "SessionType": "RTH", "Session_Volume": 10.0 * (n + 1), "Price": price(n)}
        for n, t in enumerate(_minutes(date, start, end)) if not (drop and lo <= t <= hi)
    ], schema=SCHEMA)


def reset(date, hhmm):
    """One ETH tick with Session_Volume 1 at that time: a VWAP reset."""
    return pl.DataFrame([{"Date": date, "Datetime": datetime.fromisoformat(f"{date} {hhmm}"),
                          "SessionType": "ETH", "Session_Volume": 1.0, "Price": 100.0}], schema=SCHEMA)


def frame(*parts):
    """Concatenate parts, sort by Datetime, add Index 0..n-1."""
    return pl.concat(parts).sort("Datetime").with_row_index("Index")


def test_full_day_with_anchor_at_previous_close_is_clean():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), rth_day("2026-03-03"))
    row = find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")
    assert row.height == 1
    assert row["clean"][0] is True
    assert row["reason"][0] == "ok"
    assert row["max_gap_minutes"][0] == 1.0


def test_gap_over_ten_minutes_drops_the_session():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), rth_day("2026-03-03", drop=("10:01", "10:11")))
    row = find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")
    assert row["clean"][0] is False
    assert row["reason"][0] == "gap"
    assert row["max_gap_minutes"][0] == 12.0


def test_late_start_drops_the_session():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), rth_day("2026-03-03", start="08:20"))
    row = find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")
    assert row["clean"][0] is False
    assert row["reason"][0] == "late_start"


def test_early_end_drops_the_session():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), rth_day("2026-03-03", end="13:00"))
    row = find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")
    assert row["clean"][0] is False
    assert row["reason"][0] == "early_end"


def test_anchor_at_1700_reopen_drops_the_session():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "17:00"), rth_day("2026-03-03"))
    row = find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")
    assert row["clean"][0] is False
    assert row["reason"][0] == "anchor"


def test_first_day_without_any_reset_drops_the_session():
    row = find_clean_sessions(frame(rth_day("2026-03-03")))
    assert row["reason"].to_list() == ["anchor"]


def test_reset_two_sessions_back_drops_the_session():
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), rth_day("2026-03-03"), rth_day("2026-03-04"))
    out = find_clean_sessions(f)
    assert out["Date"].to_list() == ["2026-03-02", "2026-03-03", "2026-03-04"]
    assert out.filter(pl.col("Date") == "2026-03-04")["reason"][0] == "anchor"


def test_monday_anchored_at_friday_close_is_clean():
    f = frame(rth_day("2026-03-06"), reset("2026-03-06", "13:30"), rth_day("2026-03-09"))
    assert find_clean_sessions(f).filter(pl.col("Date") == "2026-03-09")["reason"].to_list() == ["ok"]


def test_prior_session_range_is_the_previous_dates_high_minus_low():
    f = frame(rth_day("2026-03-02", price=lambda n: {0: 99.0, 1: 101.5}.get(n, 100.0)),
              rth_day("2026-03-03", price=lambda n: {1: 100.2}.get(n, 100.0)))
    out = prior_session_range(f)
    assert out["Date"].to_list() == ["2026-03-02", "2026-03-03"]
    assert out["prior_day_range"].to_list() == [None, 2.5]


def test_a_first_session_tick_that_drops_volume_is_not_its_own_anchor():
    reopen = reset("2026-03-02", "17:00").with_columns(Session_Volume=pl.lit(5000.0))  # not a reset
    f = frame(rth_day("2026-03-02"), reset("2026-03-02", "13:30"), reopen, rth_day("2026-03-03"))
    assert find_clean_sessions(f).filter(pl.col("Date") == "2026-03-03")["reason"].to_list() == ["ok"]
