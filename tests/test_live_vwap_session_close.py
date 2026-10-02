"""LiveVWAPExit flattens on the last RTH tick before the RTH->ETH reset.

VWAP resets at that boundary, so without a flatten an open trade "hits" the
reset VWAP on the first ETH tick: artifact P&L, filled after a one-hour halt.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from orderflow.backtester import (
    BacktestEngine,
    ExitReason,
    LiveVWAPExit,
    PositionState,
    Side,
    SlippageMode,
    SlippageModel,
    Tick,
    session_close_flag,
)


def test_flag_marks_only_the_last_rth_tick_before_eth():
    s = ["ETH", "ETH", "RTH", "RTH", "ETH", "ETH", "RTH"]
    got = pl.DataFrame({"SessionType": s}).select(session_close_flag())["session_close"].to_list()
    # ETH->RTH (overnight into the open) is not a reset; the file's last tick has no successor.
    assert got == [False, False, False, True, False, False, False]


@pytest.mark.parametrize("side, entry, stop, vwap", [
    (Side.LONG, 100.0, 95.0, 105.0),
    (Side.SHORT, 100.0, 105.0, 95.0),
])
def test_open_trade_exits_as_session_close(side, entry, stop, vwap):
    exit_ = LiveVWAPExit(signals_df=pd.DataFrame({"Index": [1], "stop_loss": [stop]}))
    pos = PositionState()
    pos.side, pos.entry_price = side, entry
    tick = Tick(index=1, timestamp=np.int64(1), datetime=datetime(2025, 1, 2, 15, 59, 59),
                price=entry, bid=0.0, ask=0.0, date=datetime(2025, 1, 2).date(), session_type="RTH")
    exit_.on_entry(tick, pos)

    assert not exit_.on_tick(tick, pos, np.array([entry]), {"vwap": vwap, "session_close": False}).should_exit
    sig = exit_.on_tick(tick, pos, np.array([entry]), {"vwap": vwap, "session_close": True})
    assert sig.should_exit and sig.reason == ExitReason.SESSION_CLOSE


def test_engine_fills_at_last_rth_tick_not_on_reset_vwap():
    # Long from 100, VWAP 105 never reached in RTH; at 17:00 VWAP resets to the reopen price.
    dt = pd.to_datetime(["2025-01-02 15:59:57", "2025-01-02 15:59:58", "2025-01-02 15:59:59",
                         "2025-01-02 17:00:00", "2025-01-02 17:00:01"])
    data = pd.DataFrame({
        "Index": np.arange(1, 6, dtype=np.int64),
        "Datetime": dt,
        "Price": [100.0, 101.0, 102.0, 98.0, 98.0],
        "Date": dt.date,
        "Time": dt.time,
        "SessionType": ["RTH", "RTH", "RTH", "ETH", "ETH"],
        "vwap": [105.0, 105.0, 105.0, 98.0, 98.0],
    })
    data["session_close"] = pl.from_pandas(data[["SessionType"]]).select(session_close_flag())["session_close"].to_numpy()
    signals = pd.DataFrame({"Index": [1], "TradeType": [2], "stop_loss": [90.0]})
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0.0, progress_bar=False,
                            slippage_model=SlippageModel(mode=SlippageMode.ZERO))

    res = engine.run(data, signals, exit_strategy=LiveVWAPExit(signals_df=signals),
                     indicator_columns=["vwap", "session_close"])

    t = res.trades_df.iloc[0]
    assert len(res.trades_df) == 1
    assert t["exit_reason"] == ExitReason.SESSION_CLOSE.value
    assert t["exit_price"] == 102.0
    assert t["exit_timestamp"] == 3
