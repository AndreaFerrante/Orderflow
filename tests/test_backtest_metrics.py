"""Sharpe and drawdown of the backtest metrics: hand-built trades and one synthetic tape."""

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester import BacktestEngine, compute_metrics
from orderflow.backtester.models import ExitReason, Side, TradeRecord
from orderflow.stats import max_drawdown_absolute


def trades(pnl_by_day):
    return [
        TradeRecord(
            trade_id=0, side=Side.LONG, entry_datetime=np.datetime64(f"{day}T09:{i:02d}:00"),
            exit_datetime=np.datetime64(f"{day}T09:{i:02d}:30"), entry_timestamp=0, exit_timestamp=0,
            entry_price=0.0, entry_price_pure=0.0, exit_price=0.0, pnl_ticks=0.0, pnl_dollars=pnl, commission=0.0,
            net_pnl=pnl, exit_reason=ExitReason.TAKE_PROFIT, mae_ticks=0.0, mfe_ticks=0.0, ticks_in_trade=1,
        )
        for day, pnls in pnl_by_day.items() for i, pnl in enumerate(pnls)
    ]


def annual_sharpe(daily):
    return np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252.0)


THREE_DAYS = {"2026-03-02": [10.0, 20.0], "2026-03-03": [-5.0, -5.0], "2026-03-04": [15.0, 5.0]}


def test_sharpe_is_that_of_daily_pnl_not_of_single_trades():
    assert compute_metrics(trades(THREE_DAYS)).sharpe_ratio == pytest.approx(annual_sharpe([30.0, -10.0, 20.0]))


def test_a_day_without_trades_is_a_zero_day_when_the_calendar_is_given():
    calendar = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]
    got = compute_metrics(trades(THREE_DAYS), dates=calendar).sharpe_ratio
    assert got == pytest.approx(annual_sharpe([30.0, -10.0, 20.0, 0.0]))


def test_a_trade_on_a_day_missing_from_the_calendar_is_still_counted():
    got = compute_metrics(trades(THREE_DAYS), dates=["2026-03-02", "2026-03-03"]).sharpe_ratio
    assert got == pytest.approx(annual_sharpe([30.0, -10.0, 20.0]))


def test_one_day_of_trades_has_no_sharpe():
    assert compute_metrics(trades({"2026-03-02": [10.0, -4.0, 7.0]})).sharpe_ratio == 0.0


def test_a_loss_on_the_first_trade_is_a_drawdown():
    assert compute_metrics(trades({"2026-03-02": [-50.0, 20.0]})).max_drawdown == 50.0
    assert max_drawdown_absolute([-50.0, 20.0]) == 50.0


def test_drawdown_in_percent_is_measured_from_the_starting_capital_when_the_first_trade_loses():
    metrics = compute_metrics(trades({"2026-03-02": [-50.0, 20.0]}), initial_capital=1000.0)
    assert (metrics.max_drawdown, metrics.max_drawdown_pct) == (50.0, 0.05)


def test_the_engine_counts_the_days_of_the_tape_without_trades():
    # Day 1: +2 ticks. Day 2: -4 ticks. Day 3: no trade. One tick is 12.5.
    prices = [100.0, 100.5, 100.5, 100.0, 99.0, 99.0, 100.0, 100.0, 100.0]
    stamps = pd.to_datetime([f"{day} 10:00:0{i}" for day in ("2025-01-02", "2025-01-03", "2025-01-06") for i in range(3)])
    data = pd.DataFrame({
        "Index": np.arange(1, 10, dtype=np.int64), "Datetime": stamps, "Date": stamps.strftime("%Y-%m-%d"),
        "Time": stamps.strftime("%H:%M:%S"), "Price": prices, "SessionType": "RTH",
    })
    signals = pd.DataFrame({"Index": [1, 4], "TradeType": [2, 2]})
    engine = BacktestEngine(tick_size=0.25, tick_value=12.5, commission=0.0, progress_bar=False)
    result = engine.run(data, signals, tp_ticks=2, sl_ticks=4)
    assert result.trades_df["net_pnl"].tolist() == [25.0, -50.0]
    assert result.metrics.sharpe_ratio == pytest.approx(annual_sharpe([25.0, -50.0, 0.0]))
