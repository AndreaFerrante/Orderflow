"""
backtester — Professional-grade tick-by-tick backtesting engine.

Architecture
~~~~~~~~~~~~
- **models**:       Data classes, enums, and type definitions (Tick, Position, ExitSignal, …)
- **execution**:    Slippage / fill-simulation models
- **risk**:         Position-level risk management (trailing stop, break-even, …)
- **exits**:        Pluggable exit-strategy library (strategy pattern)
- **metrics**:      Post-trade performance analytics (Sharpe, drawdown, …)
- **engine**:       Core backtester loop (Numba-accelerated + pure-Python fallback)
- **examples**:     Ready-to-run usage examples

Quick start
~~~~~~~~~~~
>>> from orderflow.backtester import BacktestEngine, FixedTPSLExit, SlippageModel
>>> engine = BacktestEngine(tick_size=0.25, tick_value=12.5)
>>> results = engine.run(data, signals, exit_strategy=FixedTPSLExit(tp=10, sl=8))
>>> results.summary()
"""

from orderflow.backtester import backtester
from orderflow.backtester.models import (
    Side,
    ExitReason,
    ExitSignal,
    Tick,
    PositionState,
    TradeRecord,
    BacktestConfig,
)
from orderflow.backtester.execution import SlippageModel, FillSimulator
from orderflow.backtester.risk import RiskManager
from orderflow.backtester.exits import (
    BaseExitStrategy,
    FixedTPSLExit,
    TrailingStopExit,
    BreakEvenExit,
    TimeBasedExit,
    VolatilityExit,
    CompositeExit,
)
from orderflow.backtester.metrics import PerformanceMetrics, compute_metrics
from orderflow.backtester.engine import BacktestEngine, BacktestResult

__all__ = [
    # Models
    "Side",
    "ExitReason",
    "ExitSignal",
    "Tick",
    "PositionState",
    "TradeRecord",
    "BacktestConfig",
    # Execution
    "SlippageModel",
    "FillSimulator",
    # Risk
    "RiskManager",
    # Exits
    "BaseExitStrategy",
    "FixedTPSLExit",
    "TrailingStopExit",
    "BreakEvenExit",
    "TimeBasedExit",
    "VolatilityExit",
    "CompositeExit",
    # Metrics
    "PerformanceMetrics",
    "compute_metrics",
    # Engine
    "BacktestEngine",
    "BacktestResult",
]
