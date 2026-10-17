"""
The engine's Python loop (the path every custom exit takes): what it hands to an exit strategy,
and that going straight from a flat tick to the next signal tick changes nothing a strategy, a
risk manager or a caller can observe.
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester import (
    BacktestEngine, CompositeExit, DynamicBreakEvenExit, DynamicTPSLExit, ElapsedTimeExit, ExitReason, ExitSignal,
    FixedTPSLExit, HourBasedExit, LiveVWAPExit, Side, TrailingStopExit, VolatilityExit,
)
from orderflow.backtester.exits import BaseExitStrategy, BreakEvenExit, CVDBreakEvenExit, TimeBasedExit
from orderflow.backtester.models import PositionState, Tick
from orderflow.backtester.risk import RiskManager

N = 400
WINDOW = 7


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    price = 5000.0 + np.cumsum(rng.integers(-2, 3, size=N)) * 0.25
    stamps = pd.date_range("2025-01-02 09:30:00", periods=N, freq="s")
    return pd.DataFrame({
        "Index": np.arange(1000, 1000 + N),
        "Datetime": stamps,
        "Date": stamps.strftime("%Y-%m-%d"),
        "Time": stamps.strftime("%H:%M:%S"),
        "Price": price,
        "SessionType": np.where(np.arange(N) < 300, "RTH", "ETH"),
    })


@dataclass
class _Recorder(BaseExitStrategy):
    """Keeps what the engine passes on every tick; leaves a trade after ``hold`` ticks."""
    hold: int = 5
    seen: list = field(default_factory=list)
    entries: list = field(default_factory=list)

    def on_entry(self, tick, position) -> None:
        self.entries.append((tick.index, tick.date, tick.session_type, position.side))

    def on_tick(self, tick, position, price_history, indicators) -> ExitSignal:
        self.seen.append((tick.index, price_history.copy(), tick.date, tick.session_type))
        if position.ticks_in_trade >= self.hold:
            return ExitSignal(should_exit=True, reason=ExitReason.CUSTOM)
        return ExitSignal(should_exit=False)


def _run(signal_rows, sides, hold=5):
    data = _data()
    signals = pd.DataFrame({"Index": data["Index"].to_numpy()[signal_rows], "TradeType": sides})
    recorder = _Recorder(hold=hold)
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0.0, price_history_window=WINDOW, progress_bar=False)
    result = engine.run(data=data, signals=signals, exit_strategy=recorder)
    return data, recorder, result


def test_an_exit_sees_the_last_window_of_prices_up_to_its_tick():
    """Also on the second trade, entered 200 ticks after the first one closed."""
    data, recorder, _ = _run([3, 250], [2, 1])
    price = data["Price"].to_numpy()
    assert [row for row, *_ in recorder.seen] == [4, 5, 6, 7, 8, 251, 252, 253, 254, 255]
    for row, history, _, _ in recorder.seen:
        np.testing.assert_array_equal(history, price[max(0, row + 1 - WINDOW):row + 1])


@dataclass
class _Scribbler(BaseExitStrategy):
    """Writes into the window on every third row and keeps what it was handed."""
    hold: int = 5
    seen: list = field(default_factory=list)
    previous: object = None

    def on_tick(self, tick, position, price_history, indicators) -> ExitSignal:
        self.seen.append((tick.index, price_history.copy(), price_history is self.previous, price_history.flags.writeable))
        self.previous = price_history
        if tick.index % 3 == 0:
            price_history[0] = -1.0
        return ExitSignal(should_exit=position.ticks_in_trade >= self.hold, reason=ExitReason.CUSTOM)


def _windows_of_the_per_tick_loop(price, window, rows):
    """The ring buffer as the engine kept it tick by tick: appended to while it fills, then shifted
    left by one. On the ``rows`` an exit sees, those divisible by 3 overwrite its oldest price."""
    buffer, length, seen = np.empty(window), 0, {}
    for i, value in enumerate(price):
        if length < window:
            buffer[length] = value
            length += 1
        else:
            buffer[:-1] = buffer[1:]
            buffer[-1] = value
        if i in rows:
            seen[i] = buffer[:length].copy()
            if i % 3 == 0:
                buffer[0] = -1.0
    return seen


@pytest.mark.parametrize("window, signal_rows", [
    (7, [3, 250]),      # the first trade starts while the buffer is still filling
    (7, [3, 10, 17]),   # the next entry is fewer ticks away than the window is long
    (7, [3, 9, 15]),    # ...or is the very next tick
    (50, [3, 30, 70]),  # the buffer fills during a flat stretch, partly from ticks nobody saw
    (1, [3, 250]),
    (N + 10, [3, 250]),  # never fills
])
def test_the_price_window_is_the_engines_one_buffer_shifted_in_place(window, signal_rows):
    """What the loop handed over before it learnt to skip flat ticks: one writable array, the same
    object on every tick once it is full, in which a strategy's own writes ride along as it shifts."""
    data = _data()
    signals = pd.DataFrame({"Index": data["Index"].to_numpy()[signal_rows], "TradeType": [2] * len(signal_rows)})
    scribbler = _Scribbler()
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0.0, price_history_window=window, progress_bar=False)
    engine.run(data=data, signals=signals, exit_strategy=scribbler)

    rows = [row for row, *_ in scribbler.seen]
    assert len(rows) == 5 * len(signal_rows)
    expected = _windows_of_the_per_tick_loop(data["Price"].to_numpy(), window, set(rows))
    full_before = False
    for row, history, same_object, writeable in scribbler.seen:
        np.testing.assert_array_equal(history, expected[row])
        assert writeable
        assert same_object == full_before  # a full buffer is handed over as itself, a filling one as a slice
        full_before = len(history) == window


def test_pushing_several_ticks_at_once_leaves_the_buffer_the_per_tick_update_leaves():
    """Every window, every fill level and every number of skipped ticks around the two block copies."""
    from orderflow.backtester.engine import _push_prices

    def one_tick(buffer, length, value):
        if length < len(buffer):
            buffer[length] = value
            return length + 1
        buffer[:-1] = buffer[1:]
        buffer[-1] = value
        return length

    for window in range(1, 8):
        for before in range(0, 12):
            for count in range(0, 14):
                prices = 100.0 + np.arange(before + count)
                expected, length = np.full(window, -7.0), 0
                for value in prices[:before]:
                    length = one_tick(expected, length, value)
                expected[:length] *= -1.0  # as if a strategy had written into it: any content must ride along
                got = expected.copy()
                got_length = _push_prices(got, length, prices, before, before + count)
                for value in prices[before:]:
                    length = one_tick(expected, length, value)
                assert got_length == length, (window, before, count)
                np.testing.assert_array_equal(got, expected, err_msg=f"window={window} before={before} count={count}")


def test_a_window_of_zero_fails_on_the_first_tick_as_it_always_did():
    """Not a feature, but the loop's behaviour: the buffer update itself raises, signals or not."""
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, price_history_window=0, progress_bar=False)
    with pytest.raises(IndexError, match="index -1 is out of bounds for axis 0 with size 0"):
        engine.run(data=_data(), signals=pd.DataFrame({"Index": [], "TradeType": []}, dtype="int64"), exit_strategy=_Never())


def test_ticks_carry_plain_strings_for_date_and_session():
    data, recorder, _ = _run([3, 298], [2, 1], hold=4)
    for row, _, date, session in recorder.seen:
        assert type(date) is str and date == data["Date"].iloc[row]
        assert type(session) is str and session == data["SessionType"].iloc[row]
    assert recorder.entries == [(3, "2025-01-02", "RTH", Side.LONG), (298, "2025-01-02", "RTH", Side.SHORT)]
    assert {session for *_, session in recorder.seen} == {"RTH", "ETH"}


def test_a_pandas_string_column_is_read_a_block_at_a_time_and_its_elements_stay_what_they_were():
    """Fast element reads without a Python string per row held in memory: converting a whole
    30M-row column at once took gigabytes the loop never needed before."""
    from orderflow.backtester.engine import _plain_strings

    n = 5000
    labels = [None if k % 97 == 0 else ("RTH" if k % 3 else "ETH") for k in range(n)]
    order = list(range(n)) + [4000, 12, 4999, 0, 2500, 2499, 1023, 1024] + list(range(n - 1, n - 50, -1))
    for dtype in ("str", "string[pyarrow]", "string[python]"):
        column = pd.Series(labels, dtype=dtype).values
        reader = _plain_strings(column)
        assert not isinstance(reader, np.ndarray)
        for i in order:
            got, expected = reader[i], column[i]
            assert type(got) is type(expected)
            if expected is pd.NA:
                assert got is pd.NA
            elif isinstance(expected, float):
                assert np.isnan(got)
            else:
                assert got == expected
            assert len(reader._block) <= 1024  # never more than a block of strings at once
    for untouched in (np.array(["RTH", "ETH"], dtype=object), np.arange(3.0), pd.Series(["a", "b"], dtype="category").values,
                      pd.Series([1, None], dtype="Int64").values):
        assert _plain_strings(untouched) is untouched


def test_a_signal_during_an_open_trade_is_spent_and_the_next_one_keeps_its_own_side():
    """Rows 10 and 12 carry signals; the trade opened on 10 is still open on 12, so 12 never trades
    and the signal on row 40 must still be the SHORT it was given (the signal_ptr contract)."""
    _, recorder, result = _run([10, 12, 40], [2, 2, 1])
    trades = result.trades_df
    assert trades["side"].tolist() == ["LONG", "SHORT"]
    assert trades["entry_timestamp"].tolist() == [1010, 1040]
    assert trades["ticks_in_trade"].tolist() == [5, 5]


def test_a_trade_still_open_at_the_end_closes_on_the_last_tick():
    _, _, result = _run([N - 3], [2], hold=50)
    trade = result.trades[0]
    assert trade.exit_reason is ExitReason.END_OF_DATA
    assert trade.exit_timestamp == 1000 + N - 1 and trade.ticks_in_trade == 2


def test_excursions_are_tracked_when_no_risk_level_is_set():
    """With TP and SL left to the exit strategy the risk manager only tracks MAE and MFE: it must still."""
    data, _, result = _run([20], [2], hold=30)
    price = data["Price"].to_numpy()
    path = price[21:51]
    trade = result.trades[0]
    assert trade.mfe_ticks == pytest.approx((path.max() - price[20]) / 0.25)
    assert trade.mae_ticks == pytest.approx((price[20] - path.min()) / 0.25)


def test_a_signal_on_the_last_tick_right_after_an_exit_still_opens():
    """The exit lands on the tick before the last, the next signal on the last tick: it is entered
    and closed there by the end of the data."""
    _, _, result = _run([N - 7, N - 1], [2, 1], hold=5)
    assert [int(t.entry_timestamp) for t in result.trades] == [1000 + N - 7, 1000 + N - 1]
    assert [int(t.exit_timestamp) for t in result.trades] == [1000 + N - 2, 1000 + N - 1]
    last = result.trades[-1]
    assert last.side is Side.SHORT and last.exit_reason is ExitReason.END_OF_DATA and last.ticks_in_trade == 0


@dataclass
class _Never(BaseExitStrategy):
    def on_tick(self, tick, position, price_history, indicators) -> ExitSignal:
        return ExitSignal(should_exit=False)


# One risk level at a time while a custom exit is present (so the Python loop runs): what the engine
# did before the risk manager learned to step aside when it has no level at all. Up five ticks from
# 5000, down nine, up ten. Entry on row 5 at 5000.
_PATH = 5000.0 + np.cumsum([0] * 6 + [1] * 5 + [-1] * 9 + [1] * 10 + [0, 0]) * 0.25
RISK_LEVELS = [
    (2, {"tp_ticks": 3}, ExitReason.TAKE_PROFIT, 108, 5000.75, False),
    (2, {"sl_ticks": 2}, ExitReason.STOP_LOSS, 117, 4999.5, False),
    (2, {"trailing_stop_ticks": 2}, ExitReason.STOP_LOSS, 112, 5000.75, False),
    (2, {"break_even_ticks": 3}, ExitReason.BREAK_EVEN, 115, 5000.0, True),
    (2, {"break_even_ticks": 3, "break_even_offset_ticks": 1}, ExitReason.BREAK_EVEN, 114, 5000.25, True),
    (1, {"tp_ticks": 3}, ExitReason.TAKE_PROFIT, 118, 4999.25, False),
    (1, {"sl_ticks": 2}, ExitReason.STOP_LOSS, 107, 5000.5, False),
    (1, {"trailing_stop_ticks": 2}, ExitReason.TRAILING_STOP, 107, 5000.5, False),
    (1, {"break_even_ticks": 3}, ExitReason.END_OF_DATA, 131, 5001.5, True),
]


@pytest.mark.parametrize("side, levels, reason, exit_index, exit_price, armed", RISK_LEVELS)
def test_a_single_risk_level_still_acts_beside_a_custom_exit(side, levels, reason, exit_index, exit_price, armed):
    n = len(_PATH)
    stamps = pd.date_range("2025-01-02 09:30:00", periods=n, freq="s")
    data = pd.DataFrame({
        "Index": np.arange(100, 100 + n), "Datetime": stamps, "Date": stamps.strftime("%Y-%m-%d"),
        "Time": stamps.strftime("%H:%M:%S"), "Price": _PATH, "SessionType": "RTH",
    })
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0.0, progress_bar=False)
    result = engine.run(
        data=data, signals=pd.DataFrame({"Index": [105], "TradeType": [side]}), exit_strategy=_Never(), **levels
    )
    trade = result.trades[0]
    assert (trade.exit_reason, int(trade.exit_timestamp), trade.exit_price, trade.break_even_activated) == (
        reason, exit_index, exit_price, armed
    )


class _CountingRisk(RiskManager):
    """A risk manager with one of its steps overridden."""
    updates = 0

    def _update_trailing_stop(self, price, pos) -> None:
        type(self).updates += 1
        super()._update_trailing_stop(price, pos)


def test_a_risk_manager_subclass_still_has_every_step_called_when_no_level_is_set():
    """The shortcut for a risk manager without levels is for the class itself: a subclass may
    have given a step a meaning of its own."""
    _CountingRisk.updates = 0
    data = _data()
    engine = BacktestEngine(tick_size=0.25, tick_value=1.25, commission=0.0, progress_bar=False)
    result = engine.run(
        data=data, signals=pd.DataFrame({"Index": [1020], "TradeType": [2]}), exit_strategy=_Recorder(hold=30),
        risk_manager=_CountingRisk(tick_size=0.25, tick_value=1.25),
    )
    assert result.trades[0].ticks_in_trade == 30
    assert _CountingRisk.updates == 30


_ENTRY = pd.DataFrame({
    "Index": [1], "TradeType": [2], "TP_Ticks": [50.0], "SL_Ticks": [50.0], "BE_Ticks": [50.0], "stop_loss": [50.0], "CVD": [0.0],
})
STAYING = {
    "risk manager without levels": lambda: RiskManager(tick_size=0.25),
    "risk manager with levels": lambda: RiskManager(tp_ticks=50, sl_ticks=50, trailing_stop_ticks=50, break_even_ticks=50, tick_size=0.25),
    "FixedTPSLExit": lambda: FixedTPSLExit(tp=50, sl=50),
    "TrailingStopExit": lambda: TrailingStopExit(trail_ticks=50),
    "TrailingStopExit not activated": lambda: TrailingStopExit(trail_ticks=50, activation_ticks=50),
    "BreakEvenExit": lambda: BreakEvenExit(activation_ticks=50),
    "TimeBasedExit": lambda: TimeBasedExit(max_ticks_in_trade=50),
    "ElapsedTimeExit": lambda: ElapsedTimeExit(max_minutes=50),
    "VolatilityExit short history": lambda: VolatilityExit(window=5),
    "VolatilityExit": lambda: VolatilityExit(vol_threshold=50, window=1),
    "CompositeExit": lambda: CompositeExit([FixedTPSLExit(tp=50, sl=50), TimeBasedExit(max_ticks_in_trade=50)]),
    "DynamicTPSLExit": lambda: DynamicTPSLExit(signals_df=_ENTRY),
    "DynamicTPSLExit unknown entry": lambda: DynamicTPSLExit(signals_df=_ENTRY.assign(Index=99)),
    "DynamicBreakEvenExit": lambda: DynamicBreakEvenExit(signals_df=_ENTRY),
    "DynamicBreakEvenExit unknown entry": lambda: DynamicBreakEvenExit(signals_df=_ENTRY.assign(Index=99)),
    "CVDBreakEvenExit": lambda: CVDBreakEvenExit(signals_df=_ENTRY),
    "CVDBreakEvenExit without its column": lambda: CVDBreakEvenExit(signals_df=_ENTRY, cvd_col="CVD"),
    "LiveVWAPExit": lambda: LiveVWAPExit(signals_df=_ENTRY),
    "HourBasedExit": lambda: HourBasedExit(close_hour=23, close_minute=59),
}


@pytest.mark.parametrize("name", STAYING)
def test_a_stay_in_the_trade_answer_is_a_signal_of_its_own(name):
    """Every tick gets a new ``ExitSignal`` with a ``dict`` nobody else holds: a strategy that
    annotates the answer it got back, or pickles it, finds it as it always was."""
    position = PositionState()
    position.side, position.entry_price, position.entry_datetime = Side.LONG, 100.0, np.datetime64("2025-01-02T10:00:00")
    position.max_favorable_price = position.max_adverse_price = 100.0
    tick = Tick(index=1, timestamp=np.int64(1), datetime=np.datetime64("2025-01-02T10:00:01"), price=100.0)
    source = STAYING[name]()
    if isinstance(source, RiskManager):
        source.initialize_position(position)
        answers = [source.check(tick, position) for _ in range(2)]
    else:
        source.on_entry(tick, position)
        indicators = {} if name.endswith("without its column") else {"CVD": 0.0}
        answers = [source.on_tick(tick, position, np.array([100.0]), indicators) for _ in range(2)]
    first, second = answers
    assert not first.should_exit and not second.should_exit
    assert first is not second
    assert type(first.metadata) is dict and first.metadata is not second.metadata
    first.metadata["note"] = 1
    assert second.metadata == {}
    assert pickle.loads(pickle.dumps(second)) == second
