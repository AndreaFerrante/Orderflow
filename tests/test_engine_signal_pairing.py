import numpy as np
import pandas as pd
import pytest

from orderflow.backtester import BacktestEngine, Side


def _data(index=None):
    index = np.arange(40) if index is None else np.asarray(index)
    return pd.DataFrame(
        {
            "Index": index,
            "Price": 5000.0,
            "Date": "2025-01-06",
            "Time": "09:00:00",
            "SessionType": "RTH",
            "Datetime": pd.date_range("2025-01-06 09:00", periods=len(index), freq="s"),
        }
    )


def _run(signal_index, trade_type, data=None):
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0, n_contracts=1)
    signals = pd.DataFrame({"Index": signal_index, "TradeType": trade_type})
    return engine.run(_data() if data is None else data, signals, tp_ticks=4, sl_ticks=4)


def test_each_signal_trades_its_own_side():
    # Flat prices never hit TP/SL, so the first trade closes at end of data: one trade, LONG.
    # Two separate runs pin the side mapping itself: TradeType 2 = LONG, 1 = SHORT (CLAUDE.md §4).
    assert [t.side for t in _run([5], [2]).trades] == [Side.LONG]
    assert [t.side for t in _run([5], [1]).trades] == [Side.SHORT]


@pytest.mark.parametrize(
    "signal_index, trade_type, match",
    [
        ([5, 5, 31], [2, 2, 1], "duplicate"),       # same entry tick twice
        ([5, 999, 31], [2, 2, 1], "not in data"),  # entry tick absent from the data
        ([31, 5], [1, 2], "order"),                 # signals out of tick order
    ],
)
def test_malformed_signals_raise_instead_of_shifting_sides(signal_index, trade_type, match):
    # The engine pairs sides with entry ticks by position, so any of these would silently
    # hand every later trade the previous signal's side.
    with pytest.raises(ValueError, match=match):
        _run(signal_index, trade_type)


def test_duplicate_data_index_raises():
    data = _data(np.r_[np.arange(20), np.arange(19, 39)])  # Index 19 appears twice
    with pytest.raises(ValueError, match="duplicate"):
        _run([5], [2], data=data)
