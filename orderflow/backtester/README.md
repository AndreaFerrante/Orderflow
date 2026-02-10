# orderflow.backtester

Professional-grade tick-by-tick backtesting utilities for systematic trading.

## Overview
This package provides a fast, modular backtester with:
- Numba-accelerated core loop and Python fallback
- Pluggable exit strategies and a mechanical risk manager
- Tick-level slippage/fill simulation and post-trade metrics

## Quickstart
Create an engine and run a backtest:

```py
from orderflow.backtester import BacktestEngine, FixedTPSLExit
engine = BacktestEngine(tick_size=0.25, tick_value=12.5)
result = engine.run(data, signals, exit_strategy=FixedTPSLExit(tp=10, sl=8))
print(result.metrics.summary())
```

## Key API
- [`orderflow.backtester.BacktestEngine`](engine.py) — core engine and run loop  
- [`orderflow.backtester.BacktestResult`](engine.py) — result container  
- [`orderflow.backtester.RiskManager`](risk.py) — mechanical TP/SL/trailing-stop overlay  
- [`orderflow.backtester.BaseExitStrategy`](exits.py) and implementations:  
  [`orderflow.backtester.FixedTPSLExit`](exits.py), [`orderflow.backtester.TrailingStopExit`](exits.py),  
  [`orderflow.backtester.BreakEvenExit`](exits.py), [`orderflow.backtester.TimeBasedExit`](exits.py),  
  [`orderflow.backtester.VolatilityExit`](exits.py), [`orderflow.backtester.CompositeExit`](exits.py)  
- [`orderflow.backtester.SlippageModel`](execution.py), [`orderflow.backtester.FillSimulator`](execution.py) — execution models  
- [`orderflow.backtester.PerformanceMetrics`](metrics.py), [`orderflow.backtester.compute_metrics`](metrics.py) — post-trade analytics

## Examples
See runnable examples and usage patterns in [`examples.py`](examples.py).

## Package files
- [__init__.py](__init__.py)  
- [backtester.py](backtester.py)  
- [engine.py](engine.py)  
- [examples.py](examples.py)  
- [execution.py](execution.py)  
- [exits.py](exits.py)  
- [metrics.py](metrics.py)  
- [models.py](models.py)  
- [risk.py](risk.py)

## License
See repository LICENSE.

```