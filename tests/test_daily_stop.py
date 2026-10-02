"""Tests for the daily stop applied to a finished trade list."""

import pandas as pd

from orderflow.backtester.risk import apply_daily_stop


def trades(rows):
    """rows: (entry "HH:MM", exit "HH:MM", exit_reason[, day])."""
    out = []
    for r in rows:
        day = r[3] if len(r) > 3 else "2025-09-15"
        out.append({"entry_datetime": pd.Timestamp(f"{day} {r[0]}"),
                    "exit_datetime": pd.Timestamp(f"{day} {r[1]}"), "exit_reason": r[2]})
    return pd.DataFrame(out)


def entries(frame):
    return frame["entry_datetime"].dt.strftime("%m-%d %H:%M").tolist()


def test_trades_entered_after_the_second_stop_are_dropped():
    kept = apply_daily_stop(trades([("09:00", "09:05", "stop_loss"), ("09:10", "09:20", "stop_loss"),
                                    ("09:30", "09:45", "time_exit"), ("10:00", "10:15", "time_exit")]))
    assert entries(kept) == ["09-15 09:00", "09-15 09:10"]


def test_time_exits_do_not_count_as_stops():
    kept = apply_daily_stop(trades([("09:00", "09:05", "stop_loss"), ("09:10", "09:25", "time_exit"),
                                    ("09:30", "09:45", "time_exit"), ("10:00", "10:15", "time_exit")]))
    assert len(kept) == 4
