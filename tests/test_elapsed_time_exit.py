import numpy as np

from orderflow.backtester.exits import CompositeExit, ElapsedTimeExit
from orderflow.backtester.models import ExitReason, PositionState, Side, Tick


def tick_at(stamp):
    return Tick(index=0, timestamp=np.int64(0), datetime=np.datetime64(stamp), price=100.0)


def position():
    return PositionState(side=Side.LONG, entry_price=100.0,
                         entry_datetime=np.datetime64("2025-09-16T10:00:00"))


def test_elapsed_exit_matched_pair():
    exit_ = ElapsedTimeExit(max_minutes=60)
    early = exit_.on_tick(tick_at("2025-09-16T10:59:59"), position(), np.array([]), {})
    due = exit_.on_tick(tick_at("2025-09-16T11:00:00"), position(), np.array([]), {})
    assert early.should_exit is False
    assert due.should_exit is True and due.reason == ExitReason.TIME_EXIT


def test_elapsed_exit_inside_composite():
    combo = CompositeExit([ElapsedTimeExit(max_minutes=60)])
    assert combo.on_tick(tick_at("2025-09-16T11:00:00"), position(), np.array([]), {}).should_exit
