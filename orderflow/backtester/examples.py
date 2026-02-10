"""
Usage examples for ``backtester``.

Run this file directly to execute all examples on synthetic data::

    python -m orderflow.backtester.examples

Each function is self-contained and demonstrates a different capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from orderflow.backtester.engine import BacktestEngine
from orderflow.backtester.execution import SlippageMode, SlippageModel
from orderflow.backtester.exits import (
    BaseExitStrategy,
    BreakEvenExit,
    CompositeExit,
    FixedTPSLExit,
    TimeBasedExit,
    TrailingStopExit,
    VolatilityExit,
)
from orderflow.backtester.models import (
    ExitReason,
    ExitSignal,
    PositionState,
    Side,
    Tick,
)
from orderflow.backtester.risk import RiskManager


# ======================================================================== #
#  Synthetic data generator                                                #
# ======================================================================== #

def generate_synthetic_tick_data(
    n_ticks: int = 100_000,
    start_price: float = 4500.0,
    tick_size: float = 0.25,
    volatility: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic tick dataset for testing.

    Returns a DataFrame with columns expected by the engine:
    ``Index``, ``Datetime``, ``Price``, ``Date``, ``Time``, ``Bid``, ``Ask``.
    """
    rng = np.random.default_rng(seed)

    # Random walk in tick increments
    steps = rng.choice([-1, 0, 1], size=n_ticks, p=[0.3, 0.2, 0.5])
    prices = start_price + np.cumsum(steps * tick_size)
    prices = np.round(prices / tick_size) * tick_size  # snap to grid

    # Synthetic timestamps (1-second apart)
    base_dt = pd.Timestamp("2024-01-02 09:30:00")
    datetimes = pd.date_range(base_dt, periods=n_ticks, freq="1s")

    # Spread: 1 tick wide
    bids = prices - tick_size
    asks = prices + tick_size

    df = pd.DataFrame({
        "Index": np.arange(n_ticks, dtype=np.int64),
        "Datetime": datetimes,
        "Price": prices,
        "Bid": bids,
        "Ask": asks,
        "Date": datetimes.date,
        "Time": datetimes.time,
        "Volume": rng.integers(1, 50, size=n_ticks),
    })
    return df


def generate_synthetic_signals(
    data: pd.DataFrame,
    n_signals: int = 50,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate random entry signals from the data.

    ``TradeType``: 1 = SHORT, 2 = LONG (matching the original backtester convention).
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(data["Index"].values, size=n_signals, replace=False))
    sides = rng.choice([1, 2], size=n_signals)
    return pd.DataFrame({"Index": idx, "TradeType": sides})


# ======================================================================== #
#  Example 1: Basic TP / SL via RiskManager                               #
# ======================================================================== #

def example_basic_tp_sl() -> None:
    """
    Simplest usage: fixed take-profit and stop-loss via the engine's
    built-in risk manager (no custom exit strategy).

    This path uses the **Numba-accelerated** loop if Numba is installed.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE 1: Basic TP/SL via RiskManager (Numba fast path)")
    print("=" * 70)

    data = generate_synthetic_tick_data(n_ticks=500_000, seed=42)
    signals = generate_synthetic_signals(data, n_signals=300, seed=123)

    engine = BacktestEngine(
        tick_size=0.25,
        tick_value=12.5,
        commission=4.5,
        n_contracts=1,
        slippage_model=SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=2, seed=42),
        progress_bar=False,
    )

    result = engine.run(
        data,
        signals,
        tp_ticks=10,
        sl_ticks=8,
    )

    result.summary()
    print(f"\nTrades DataFrame shape: {result.trades_df.shape}")
    print(result.trades_df.head(5).to_string(index=False))


# ======================================================================== #
#  Example 2: Composite exit strategy                                      #
# ======================================================================== #

def example_composite_exit() -> None:
    """
    Combine multiple exit strategies: TP/SL + trailing stop + time limit.
    The first strategy to fire wins.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE 2: Composite Exit (TP/SL + Trailing + Time)")
    print("=" * 70)

    data = generate_synthetic_tick_data(n_ticks=200_000, seed=99)
    signals = generate_synthetic_signals(data, n_signals=150, seed=77)

    exit_logic = CompositeExit([
        FixedTPSLExit(tp=12, sl=10, tick_size=0.25),
        TrailingStopExit(trail_ticks=6, tick_size=0.25, activation_ticks=4),
        TimeBasedExit(max_ticks_in_trade=3000),
    ])

    engine = BacktestEngine(
        tick_size=0.25,
        tick_value=12.5,
        commission=4.5,
        slippage_model=SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=1, seed=55),
        progress_bar=False,
    )

    result = engine.run(data, signals, exit_strategy=exit_logic)
    result.summary()


# ======================================================================== #
#  Example 3: Trailing stop with break-even via RiskManager                #
# ======================================================================== #

def example_trailing_break_even() -> None:
    """
    Use the built-in RiskManager for trailing stop + break-even mechanics
    **without** any custom exit strategy.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE 3: RiskManager — Trailing Stop + Break-Even")
    print("=" * 70)

    data = generate_synthetic_tick_data(n_ticks=300_000, seed=7)
    signals = generate_synthetic_signals(data, n_signals=200, seed=14)

    rm = RiskManager(
        tp_ticks=15,
        sl_ticks=10,
        trailing_stop_ticks=8,
        break_even_ticks=5,
        break_even_offset_ticks=1,
        tick_size=0.25,
        tick_value=12.5,
    )

    engine = BacktestEngine(
        tick_size=0.25,
        tick_value=12.5,
        commission=4.5,
        slippage_model=SlippageModel(mode=SlippageMode.GAUSSIAN, max_ticks=1, seed=21),
        progress_bar=False,
    )

    result = engine.run(data, signals, risk_manager=rm)
    result.summary()


# ======================================================================== #
#  Example 4: Custom exit strategy — multi-condition                       #
# ======================================================================== #

@dataclass
class MultiConditionExit(BaseExitStrategy):
    """
    A custom exit demonstrating multi-condition logic:

    1. If unrealised PnL < -5 ticks → stop loss
    2. If rolling volatility > threshold → vol exit
    3. If in-trade > N ticks → time exit
    4. If price crosses a moving-average → take profit

    This showcases the flexibility of the strategy-pattern interface.
    """
    sl_ticks: float = 5.0
    vol_threshold: float = 1.5
    max_duration: int = 2000
    ma_window: int = 50
    tick_size: float = 0.25

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        price = tick.price
        entry = position.entry_price

        # 1. Stop loss
        if position.side == Side.LONG and entry - price >= self.sl_ticks * self.tick_size:
            return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)
        if position.side == Side.SHORT and price - entry >= self.sl_ticks * self.tick_size:
            return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)

        # 2. Volatility
        if len(price_history) >= 30:
            vol = float(np.std(price_history[-30:]))
            if vol > self.vol_threshold:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.VOLATILITY_EXIT,
                    metadata={"vol": vol},
                )

        # 3. Time
        if position.ticks_in_trade >= self.max_duration:
            return ExitSignal(should_exit=True, reason=ExitReason.TIME_EXIT)

        # 4. MA crossover take profit
        if len(price_history) >= self.ma_window:
            ma = float(np.mean(price_history[-self.ma_window:]))
            if position.side == Side.LONG and price > ma and price > entry:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)
            if position.side == Side.SHORT and price < ma and price < entry:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)

        return ExitSignal(should_exit=False)


def example_custom_strategy() -> None:
    """
    Fully custom exit strategy passed as a callable object.
    Demonstrates the strategy-pattern extensibility.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE 4: Custom Multi-Condition Exit Strategy")
    print("=" * 70)

    data = generate_synthetic_tick_data(n_ticks=200_000, seed=33)
    signals = generate_synthetic_signals(data, n_signals=100, seed=66)

    custom_exit = MultiConditionExit(
        sl_ticks=6,
        vol_threshold=1.8,
        max_duration=1500,
        ma_window=40,
        tick_size=0.25,
    )

    engine = BacktestEngine(
        tick_size=0.25,
        tick_value=12.5,
        commission=4.5,
        slippage_model=SlippageModel(mode=SlippageMode.ZERO),
        progress_bar=False,
    )

    # No risk_manager TP/SL — the custom strategy handles everything
    result = engine.run(data, signals, exit_strategy=custom_exit)
    result.summary()


# ======================================================================== #
#  Example 5: Risk manager + custom strategy layered                       #
# ======================================================================== #

def example_layered_risk_and_strategy() -> None:
    """
    The risk manager acts as a **hard guard** (TP=20, SL=15) while the
    custom strategy provides a softer trailing-stop overlay.  Risk fires
    first if breached.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE 5: Layered Risk Manager + Custom Strategy")
    print("=" * 70)

    data = generate_synthetic_tick_data(n_ticks=300_000, seed=50)
    signals = generate_synthetic_signals(data, n_signals=250, seed=51)

    rm = RiskManager(
        tp_ticks=20,
        sl_ticks=15,
        tick_size=0.25,
        tick_value=12.5,
    )

    trail_strategy = TrailingStopExit(
        trail_ticks=8,
        tick_size=0.25,
        activation_ticks=5,
    )

    engine = BacktestEngine(
        tick_size=0.25,
        tick_value=12.5,
        commission=4.5,
        slippage_model=SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=1, seed=52),
        progress_bar=False,
    )

    result = engine.run(data, signals, exit_strategy=trail_strategy, risk_manager=rm)
    result.summary()


# ======================================================================== #
#  Main                                                                    #
# ======================================================================== #

def run_all_examples() -> None:
    """Execute all examples sequentially."""
    example_basic_tp_sl()
    example_composite_exit()
    example_trailing_break_even()
    example_custom_strategy()
    example_layered_risk_and_strategy()
    print("\n\n All examples completed successfully.\n")


if __name__ == "__main__":
    run_all_examples()
