"""Tests for the per-signal break-even exit.

``BreakEvenExit`` already exists but arms at a scalar tick count. The initial
balance fade's targets span 13 to 330 ticks, so one activation distance is
meaningless across them: 10 ticks is most of a small trade and nothing at all
in a large one. This exit reads its activation per signal, exactly the way
``DynamicTPSLExit`` reads TP and SL.
"""

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester.exits import DynamicBreakEvenExit
from orderflow.backtester.models import (
    ExitReason,
    PositionState,
    Side,
    Tick,
)


def signals(be_ticks=10.0, index=100):
    return pd.DataFrame({"Index": [index], "BE_Ticks": [be_ticks]})


def tick(price, timestamp=100):
    return Tick(
        index=int(timestamp), timestamp=timestamp, datetime=None,
        price=price, volume=1.0, extra={},
    )


def position(side=Side.LONG, entry_price=5000.0):
    return PositionState(side=side, entry_price=entry_price, entry_timestamp=100)


def feed(exit_strategy, pos, prices, entry_timestamp=100):
    """Open a position and walk prices through, returning the first exit."""
    exit_strategy.on_entry(tick(pos.entry_price, entry_timestamp), pos)
    for p in prices:
        signal = exit_strategy.on_tick(tick(p), pos, np.array([]), {})
        if signal.should_exit:
            return signal
    return None


# --------------------------------------------------------------------------
# it does nothing until it arms
# --------------------------------------------------------------------------

def test_does_not_exit_before_the_activation_distance_is_reached():
    """9 ticks in favour on a 10-tick activation, then straight back through
    entry -- the stop has not moved, so this exit stays silent and the real
    stop handles it."""
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0), tick_size=0.25)

    assert feed(ex, position(), [5002.25, 5000.0, 4990.0]) is None


def test_arms_and_fires_when_price_returns_to_entry():
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0), tick_size=0.25)

    got = feed(ex, position(), [5002.5, 5005.0, 5000.0])

    assert got.should_exit
    assert got.reason is ExitReason.BREAK_EVEN
    assert got.exit_price == pytest.approx(5000.0)


def test_short_arms_downward_and_fires_on_the_way_back_up():
    """THE sign test. A SHORT is in profit when price FALLS."""
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0), tick_size=0.25)

    got = feed(ex, position(side=Side.SHORT), [4997.5, 4995.0, 5000.0])

    assert got.should_exit
    assert got.reason is ExitReason.BREAK_EVEN
    assert got.exit_price == pytest.approx(5000.0)


def test_a_short_is_not_armed_by_price_rising():
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0), tick_size=0.25)

    assert feed(ex, position(side=Side.SHORT), [5002.5, 5005.0, 5000.0]) is None


# --------------------------------------------------------------------------
# per-signal activation is the whole point
# --------------------------------------------------------------------------

def test_activation_comes_from_the_signal_not_a_constant():
    """Same prices, two signals, two outcomes."""
    frame = pd.DataFrame({"Index": [100, 200], "BE_Ticks": [10.0, 40.0]})

    tight = DynamicBreakEvenExit(signals_df=frame, tick_size=0.25)
    wide = DynamicBreakEvenExit(signals_df=frame, tick_size=0.25)

    assert feed(tight, position(), [5005.0, 5000.0], entry_timestamp=100) is not None
    assert feed(wide, position(), [5005.0, 5000.0], entry_timestamp=200) is None


def test_zero_be_ticks_disables_the_exit_for_that_signal():
    """The runner writes BE_Ticks=0 when the break-even is switched off; it
    must not arm instantly at zero distance and close every trade at entry."""
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=0.0), tick_size=0.25)

    assert feed(ex, position(), [5005.0, 5000.0, 4990.0]) is None


def test_an_unknown_entry_tick_disables_the_exit():
    """Never inherit the previous trade's activation: a signal frame that
    does not cover this entry means no break-even, not the last one's."""
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0, index=100), tick_size=0.25)

    feed(ex, position(), [5005.0, 5000.0], entry_timestamp=100)      # arms, fires
    assert feed(ex, position(), [5005.0, 5000.0], entry_timestamp=999) is None


def test_state_resets_between_trades():
    """A position that armed must not leave the next one pre-armed."""
    ex = DynamicBreakEvenExit(signals_df=signals(be_ticks=10.0), tick_size=0.25)

    feed(ex, position(), [5005.0, 5000.0])
    assert feed(ex, position(), [5000.25, 4999.0]) is None


# --------------------------------------------------------------------------
# offset
# --------------------------------------------------------------------------

def test_offset_moves_the_stop_beyond_entry():
    """A break-even at exactly entry still loses the round trip; the offset
    is what makes a scratch actually flat."""
    ex = DynamicBreakEvenExit(
        signals_df=signals(be_ticks=10.0), tick_size=0.25, offset_ticks=2.0
    )

    got = feed(ex, position(), [5005.0, 5000.5])

    assert got.exit_price == pytest.approx(5000.5)


def test_offset_is_subtracted_for_a_short():
    ex = DynamicBreakEvenExit(
        signals_df=signals(be_ticks=10.0), tick_size=0.25, offset_ticks=2.0
    )

    got = feed(ex, position(side=Side.SHORT), [4995.0, 4999.5])

    assert got.exit_price == pytest.approx(4999.5)
