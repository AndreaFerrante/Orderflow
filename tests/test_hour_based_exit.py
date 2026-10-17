"""HourBasedExit reads the minute of the day from whatever the tick's datetime is."""

from datetime import datetime

import numpy as np
import pytest

from orderflow.backtester.exits import HourBasedExit
from orderflow.backtester.models import ExitReason, PositionState, Tick

CLOCK = [  # time of day, and whether a 15:30 close is due
    ("03:45:10", False),
    ("14:59:59", False),
    ("15:29:59", False),
    ("15:30:00", True),
    ("15:31:07", True),
    ("16:00:00", True),
    ("23:59:59", True),
    ("00:00:00", False),
]


def _signal(stamp):
    tick = Tick(index=0, timestamp=np.int64(0), datetime=stamp, price=5000.0)
    return HourBasedExit(close_hour=15, close_minute=30).on_tick(tick, PositionState(), np.empty(0), {})


@pytest.mark.parametrize("clock, due", CLOCK)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_a_numpy_datetime_of_any_unit(clock, due, unit):
    signal = _signal(np.datetime64(f"2025-06-16T{clock}", unit))
    assert signal.should_exit is due
    if due:
        assert signal.reason is ExitReason.TIME_EXIT and signal.metadata == {"close_time": "15:30"}


@pytest.mark.parametrize("clock, due", CLOCK)
def test_a_python_datetime_and_nanoseconds_since_the_epoch(clock, due):
    stamp = datetime.fromisoformat(f"2025-06-16T{clock}")
    assert _signal(stamp).should_exit is due
    assert _signal(int(np.datetime64(f"2025-06-16T{clock}", "ns").astype(np.int64))).should_exit is due


def test_a_date_before_1970_keeps_its_time_of_day():
    """Minutes since the epoch are negative there; the time of day is still the floor of the division."""
    assert _signal(np.datetime64("1969-12-31T15:30:00", "us")).should_exit is True
    assert _signal(np.datetime64("1969-12-31T15:29:59", "us")).should_exit is False
