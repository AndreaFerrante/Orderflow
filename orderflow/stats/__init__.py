"""
OrderFlow Statistics Module
===========================

Institutional-grade statistical engine for systematic trading research.

Submodules
----------
stats             Core descriptive stats, risk metrics, time-series diagnostics.
returns           Return series construction, equity curves, drawdown analysis.
"""

# ── Core statistical functions ───────────────────────────────────────────────
from .stats import (
    describe,
    is_skewed,
    get_kurtosis,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    information_ratio,
    max_drawdown,
    var_historical,
    cvar_historical,
    rolling_sharpe,
    autocorrelation,
    hurst_exponent,
    omega_ratio,
    tail_ratio,
    profit_factor,
    gain_to_pain_ratio,
    bootstrap_total_by_group,
)

# ── Return series analysis ───────────────────────────────────────────────────
from .returns import (
    to_log_returns,
    to_arithmetic_returns,
    log_to_arithmetic,
    arithmetic_to_log,
    annualise_return,
    annualise_volatility,
    equity_curve,
    drawdown_series,
    rolling_volatility,
    ewma_volatility,
    underwater_duration,
    max_drawdown_absolute,
    trade_sharpe,
)

__all__ = [
    # stats.py
    "describe",
    "is_skewed",
    "get_kurtosis",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio",
    "max_drawdown",
    "var_historical",
    "cvar_historical",
    "rolling_sharpe",
    "autocorrelation",
    "hurst_exponent",
    "omega_ratio",
    "tail_ratio",
    "profit_factor",
    "gain_to_pain_ratio",
    "bootstrap_total_by_group",
    # returns.py
    "to_log_returns",
    "to_arithmetic_returns",
    "log_to_arithmetic",
    "arithmetic_to_log",
    "annualise_return",
    "annualise_volatility",
    "equity_curve",
    "drawdown_series",
    "rolling_volatility",
    "ewma_volatility",
    "underwater_duration",
    "max_drawdown_absolute",
    "trade_sharpe",
]
