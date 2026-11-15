"""Fill prices of the backtest engine, on both of its loops: synthetic tapes only.

One rule for every exit. A market exit (stop, trailing stop, break-even stop, session close, end of
data, a strategy that names no price) fills at the tick that triggered it, less slippage. A limit exit
(a target) fills at its own level. The kernel and the Python loop must book the same trades.
"""

import numpy as np
import pandas as pd
import pytest

import orderflow.backtester.engine as engine_module
from orderflow.backtester import (
    BacktestEngine, BreakEvenExit, CVDBreakEvenExit, DynamicBreakEvenExit, DynamicTPSLExit, FixedTPSLExit,
    SlippageMode, SlippageModel, TrailingStopExit,
)

LONG, SHORT = 2, 1


def tape(prices, *, sessions=None, days=None, spread=None):
    n = len(prices)
    stamps = pd.date_range("2025-01-02 10:00:00", periods=n, freq="s")
    if days is not None:
        stamps = pd.to_datetime([f"{day} 10:00:00" for day in days]) + pd.to_timedelta(np.arange(n), unit="s")
    frame = pd.DataFrame({
        "Index": np.arange(1, n + 1, dtype=np.int64), "Datetime": stamps, "Date": stamps.strftime("%Y-%m-%d"),
        "Time": stamps.strftime("%H:%M:%S"), "Price": np.asarray(prices, dtype=np.float64),
        "SessionType": sessions if sessions is not None else ["RTH"] * n,
    })
    if spread is not None:
        frame["BidPrice"] = frame["Price"] - spread
        frame["AskPrice"] = frame["Price"] + spread
    return frame


def entry(side, index=1, **columns):
    return pd.DataFrame({"Index": [index], "TradeType": [side], **{k: [v] for k, v in columns.items()}})


def engine(**kwargs):
    return BacktestEngine(tick_size=0.25, tick_value=12.5, commission=0.0, progress_bar=False, **kwargs)


def one_tick_of_slippage():
    return SlippageModel(mode=SlippageMode.FIXED, fixed_ticks=1)


@pytest.fixture(params=["kernel", "python loop"])
def either_loop(request, monkeypatch):
    """Run the test on the Numba kernel, then on the Python loop the engine falls back to without Numba."""
    if request.param == "python loop":
        monkeypatch.setattr(engine_module, "HAS_NUMBA", False)
    return request.param


# ---------------------------------------------------------------------------
# One rule for the exit price
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side, prices, reason, exit_price", [
    (LONG, [100.0, 100.0, 95.0, 95.0], "stop_loss", 95.0),       # stop at 99: sold where the market is
    (LONG, [100.0, 100.0, 105.0, 105.0], "take_profit", 101.0),  # limit at 101: no better than the limit
    (SHORT, [100.0, 100.0, 105.0, 105.0], "stop_loss", 105.0),
    (SHORT, [100.0, 100.0, 95.0, 95.0], "take_profit", 99.0),
])
def test_a_gap_fills_a_stop_at_the_tick_price_and_a_target_at_its_level(either_loop, side, prices, reason, exit_price):
    trade = engine().run(tape(prices), entry(side), tp_ticks=4, sl_ticks=4).trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_price"], trade["exit_timestamp"]) == (reason, exit_price, 3)


@pytest.mark.parametrize("side, prices, reason, entry_price, exit_price", [
    (LONG, [100.0, 100.0, 99.0, 99.0], "stop_loss", 100.25, 98.75),       # market exit: one more tick against
    (LONG, [100.0, 100.0, 101.25, 101.25], "take_profit", 100.25, 101.25),  # limit exit: none
    (SHORT, [100.0, 100.0, 101.0, 101.0], "stop_loss", 99.75, 101.25),
    (SHORT, [100.0, 100.0, 98.75, 98.75], "take_profit", 99.75, 98.75),
])
def test_a_market_exit_pays_slippage_and_a_target_does_not(either_loop, side, prices, reason, entry_price, exit_price):
    run = engine(slippage_model=one_tick_of_slippage()).run(tape(prices), entry(side), tp_ticks=4, sl_ticks=4)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["entry_price"], trade["exit_price"]) == (reason, entry_price, exit_price)


def test_a_trade_open_on_the_last_tick_is_closed_at_the_market(either_loop):
    run = engine(slippage_model=one_tick_of_slippage()).run(tape([100.0, 100.5]), entry(LONG), tp_ticks=40, sl_ticks=40)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_price"]) == ("end_of_data", 100.25)


# ---------------------------------------------------------------------------
# Targets: touch or trade-through
# ---------------------------------------------------------------------------

_TOUCH_THEN_THROUGH = [100.0, 100.0, 101.0, 100.5, 101.25, 101.25]  # target 101: touched on tick 3, crossed on tick 5


def test_a_target_fills_on_touch_by_default(either_loop):
    trade = engine().run(tape(_TOUCH_THEN_THROUGH), entry(LONG), tp_ticks=4, sl_ticks=40).trades_df.iloc[0]
    assert (trade["exit_price"], trade["exit_timestamp"]) == (101.0, 3)


@pytest.mark.parametrize("side, prices", [(LONG, _TOUCH_THEN_THROUGH), (SHORT, [200.0 - p for p in _TOUCH_THEN_THROUGH])])
def test_without_fill_on_touch_a_target_waits_for_a_trade_through_and_fills_at_its_level(either_loop, side, prices):
    run = engine().run(tape(prices), entry(side), tp_ticks=4, sl_ticks=40, fill_on_touch=False)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_price"], trade["exit_timestamp"]) == ("take_profit", prices[2], 5)


# ---------------------------------------------------------------------------
# use_spread
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side, prices, reason, entry_price, exit_price", [
    (LONG, [100.0, 100.0, 98.0, 98.0], "stop_loss", 100.25, 97.75),     # buys the ask, sells the bid
    (SHORT, [100.0, 100.0, 102.0, 102.0], "stop_loss", 99.75, 102.25),  # sells the bid, buys the ask
    (LONG, [100.0, 100.0, 103.0, 103.0], "take_profit", 100.25, 101.25),  # the target is a limit: its level
])
def test_use_spread_crosses_the_quote_on_every_market_fill(either_loop, side, prices, reason, entry_price, exit_price):
    run = engine(use_spread=True).run(tape(prices, spread=0.25), entry(side), tp_ticks=4, sl_ticks=4)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["entry_price"], trade["exit_price"]) == (reason, entry_price, exit_price)


def test_use_spread_reads_the_trade_price_on_a_tick_without_a_quote(either_loop):
    data = tape([100.0, 100.0, 98.0, 98.0], spread=0.25)
    data.loc[0, ["BidPrice", "AskPrice"]] = 0.0
    trade = engine(use_spread=True).run(data, entry(LONG), tp_ticks=40, sl_ticks=40).trades_df.iloc[0]
    assert trade["entry_price"] == 100.0


def test_use_spread_without_quote_columns_is_refused(either_loop):
    with pytest.raises(ValueError, match="use_spread"):
        engine(use_spread=True).run(tape([100.0, 100.0]), entry(LONG), tp_ticks=4, sl_ticks=4)


# ---------------------------------------------------------------------------
# trade_in_rth
# ---------------------------------------------------------------------------

_DRIFT = [100.0, 100.25, 100.5, 100.75, 101.0]
_RTH_THEN_ETH = ["RTH", "RTH", "RTH", "ETH", "ETH"]


def test_trade_in_rth_closes_an_open_trade_on_the_last_rth_tick(either_loop):
    run = engine(trade_in_rth=True).run(tape(_DRIFT, sessions=_RTH_THEN_ETH), entry(LONG), tp_ticks=400, sl_ticks=400)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_price"], trade["exit_timestamp"]) == ("session_close", 100.5, 3)


def test_without_trade_in_rth_the_trade_is_held_over_the_session_break(either_loop):
    run = engine().run(tape(_DRIFT, sessions=_RTH_THEN_ETH), entry(LONG), tp_ticks=400, sl_ticks=400)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_timestamp"]) == ("end_of_data", 5)


def test_trade_in_rth_closes_at_the_day_change_of_a_tape_that_holds_rth_only(either_loop):
    days = ["2025-01-02"] * 3 + ["2025-01-03"] * 2
    run = engine(trade_in_rth=True).run(tape(_DRIFT, days=days), entry(LONG), tp_ticks=400, sl_ticks=400)
    trade = run.trades_df.iloc[0]
    assert (trade["exit_reason"], trade["exit_timestamp"]) == ("session_close", 3)


def test_trade_in_rth_takes_no_entry_on_the_last_rth_tick(either_loop):
    run = engine(trade_in_rth=True).run(tape(_DRIFT, sessions=_RTH_THEN_ETH), entry(LONG, index=3), tp_ticks=400, sl_ticks=400)
    assert run.trades_df.empty


def test_a_stop_on_the_last_rth_tick_is_booked_as_a_stop(either_loop):
    prices = [100.0, 100.0, 99.0, 99.0, 99.0]
    run = engine(trade_in_rth=True).run(tape(prices, sessions=_RTH_THEN_ETH), entry(LONG), tp_ticks=400, sl_ticks=2)
    assert run.trades_df.iloc[0]["exit_reason"] == "stop_loss"


# ---------------------------------------------------------------------------
# The two loops book the same trades
# ---------------------------------------------------------------------------

def random_tape(seed=0, n=4000):
    """Three days of RTH then ETH, a walk on the tick grid with jumps, quotes with holes."""
    rng = np.random.default_rng(seed)
    steps = rng.integers(-2, 3, size=n)
    jumps = rng.random(n) < 0.01
    steps[jumps] = rng.integers(-15, 16, size=int(jumps.sum()))
    prices = 5000.0 + np.cumsum(steps) * 0.25
    part = np.arange(n) * 6 // n  # six equal parts: RTH, ETH on each of three days
    data = tape(prices, sessions=np.where(part % 2 == 0, "RTH", "ETH").tolist(),
                days=np.array(["2025-01-02", "2025-01-03", "2025-01-06"])[part // 2].tolist())
    data["BidPrice"] = prices - 0.25 * rng.integers(0, 2, size=n)
    data["AskPrice"] = data["BidPrice"] + 0.25
    holes = rng.random(n) < 0.02
    data.loc[holes, ["BidPrice", "AskPrice"]] = 0.0
    index = np.sort(rng.choice(np.arange(1, n + 1), size=400, replace=False))
    return data, pd.DataFrame({"Index": index, "TradeType": rng.integers(1, 3, size=index.size)})


LOOP_CASES = [
    ({}, {"tp_ticks": 6, "sl_ticks": 5}),
    ({}, {"tp_ticks": 6}),
    ({}, {"sl_ticks": 6}),
    ({}, {"trailing_stop_ticks": 4}),
    ({}, {"tp_ticks": 8, "sl_ticks": 6, "trailing_stop_ticks": 3}),
    ({}, {"break_even_ticks": 3}),
    ({}, {"tp_ticks": 10, "sl_ticks": 8, "break_even_ticks": 4, "break_even_offset_ticks": 1}),
    ({}, {"tp_ticks": 6, "sl_ticks": 5, "fill_on_touch": False}),
    ({"slippage": True}, {"tp_ticks": 6, "sl_ticks": 5}),
    ({"use_spread": True}, {"tp_ticks": 6, "sl_ticks": 5, "trailing_stop_ticks": 4}),
    ({"trade_in_rth": True}, {"tp_ticks": 12, "sl_ticks": 10}),
    ({"trade_in_rth": True, "use_spread": True, "slippage": True},
     {"tp_ticks": 9, "sl_ticks": 7, "break_even_ticks": 5, "trailing_stop_ticks": 6}),
]


@pytest.mark.parametrize("engine_options, levels", LOOP_CASES, ids=[str({**a, **b}) for a, b in LOOP_CASES])
def test_the_kernel_and_the_python_loop_book_the_same_trades(monkeypatch, engine_options, levels):
    def run(data, signals):
        options = dict(engine_options)
        if options.pop("slippage", False):  # a fresh model each run: a run consumes its draws
            options["slippage_model"] = SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=2, seed=7)
        return engine(**options).run(data, signals, **levels).trades_df

    tapes = [random_tape(seed) for seed in range(5)]
    kernel = [run(*tape_and_signals) for tape_and_signals in tapes]
    monkeypatch.setattr(engine_module, "HAS_NUMBA", False)
    loop = [run(*tape_and_signals) for tape_and_signals in tapes]

    assert sum(len(trades) for trades in kernel) >= 10  # a level set without a stop holds its trades long
    for from_kernel, from_loop in zip(kernel, loop):
        pd.testing.assert_frame_equal(from_kernel, from_loop, check_exact=True)


# ---------------------------------------------------------------------------
# Exit strategies follow the same rule
# ---------------------------------------------------------------------------

def run_with(exit_strategy, prices, signals, **run_kwargs):
    return engine().run(tape(prices), signals, exit_strategy=exit_strategy, **run_kwargs).trades_df.iloc[0]


@pytest.mark.parametrize("prices, reason, exit_price", [
    ([100.0, 100.0, 105.0], "take_profit", 101.0),
    ([100.0, 100.0, 95.0], "stop_loss", 95.0),
])
def test_fixed_and_per_signal_targets_fill_at_their_level_and_their_stops_at_the_tick_price(prices, reason, exit_price):
    signals = entry(LONG, TP_Ticks=4.0, SL_Ticks=4.0)
    for strategy in (FixedTPSLExit(tp=4, sl=4), DynamicTPSLExit(signals_df=signals)):
        trade = run_with(strategy, prices, signals)
        assert (trade["exit_reason"], trade["exit_price"]) == (reason, exit_price)


def test_fixed_and_per_signal_targets_can_wait_for_a_trade_through():
    signals = entry(LONG, TP_Ticks=4.0, SL_Ticks=40.0)
    for strategy in (FixedTPSLExit(tp=4, sl=40, fill_on_touch=False), DynamicTPSLExit(signals_df=signals, fill_on_touch=False)):
        trade = run_with(strategy, _TOUCH_THEN_THROUGH, signals)
        assert (trade["exit_price"], trade["exit_timestamp"]) == (101.0, 5)


def test_strategy_stops_fill_at_the_tick_price_when_the_market_gaps_through_them():
    # Up to 102 (a trailing stop of 4 ticks sits at 101, a break-even stop at 100), then a gap to 96.
    prices = [100.0, 102.0, 102.0, 96.0]
    signals = entry(LONG, BE_Ticks=4.0, CVD=0.0)
    data = tape(prices).assign(CVD=[0.0, 10.0, 10.0, 10.0])
    stops = {
        "trailing_stop": TrailingStopExit(trail_ticks=4),
        "break_even": BreakEvenExit(activation_ticks=4),
        "break_even ": DynamicBreakEvenExit(signals_df=signals),
        "break_even  ": CVDBreakEvenExit(signals_df=signals, min_profit_ticks=4),
    }
    for reason, strategy in stops.items():
        trade = engine().run(data, signals, exit_strategy=strategy, indicator_columns=["CVD"]).trades_df.iloc[0]
        assert (trade["exit_reason"], trade["exit_price"]) == (reason.strip(), 96.0)


def test_a_short_break_even_without_a_stop_loss_closes_the_trade():
    # In the money by 4 ticks, then back through the entry: the mirror of the long case.
    prices = [100.0, 99.0, 99.0, 100.5]
    for options in ({}, {"exit_strategy": FixedTPSLExit(tp=400, sl=400)}):
        trade = engine().run(tape(prices), entry(SHORT), break_even_ticks=4, **options).trades_df.iloc[0]
        assert (trade["exit_reason"], trade["exit_price"]) == ("break_even", 100.5)
