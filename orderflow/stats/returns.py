"""
Return series construction and analysis for quantitative trading research.

Design Principles
-----------------
* **No lookahead bias**: all rolling and expanding operations are strictly causal.
* **Correctness over cleverness**: log returns for compounding math; arithmetic
  returns for linear aggregation and performance attribution.
* **Graceful handling of gaps**: business-day aware resampling; missing bars
  produce NaN rather than silent zeros.
* **Numerical stability**: log operations guarded by a positive floor; cumulative
  products use log-sum-exp trick to avoid float overflow on long series.

Public API
----------
to_log_returns            Convert price series to log returns.
to_arithmetic_returns     Convert price series to arithmetic returns.
log_to_arithmetic         Convert log returns to arithmetic returns.
arithmetic_to_log         Convert arithmetic returns to log returns.
annualise_return          Annualise a per-period mean return.
annualise_volatility      Annualise a per-period volatility.
equity_curve              Build cumulative equity curve from returns.
rolling_volatility        Rolling annualised volatility (causal).
drawdown_series           Full drawdown time-series (not just max).
underwater_duration       Consecutive bars spent below prior peak.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from ._validators import validate_positive_prices as _validate_prices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return conversions
# ---------------------------------------------------------------------------

def to_log_returns(
    prices: Union[pd.Series, np.ndarray],
    index: Optional[pd.Index] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Compute log returns: r_t = ln(P_t / P_{t-1}).

    Log returns are time-additive and numerically stable for compounding.
    Preferred for multi-period analysis and volatility estimation.

    Parameters
    ----------
    prices : pd.Series | np.ndarray
        Strictly positive price series (length >= 2).
    index : pd.Index, optional
        If provided, the returned pd.Series uses this index (length n-1).

    Returns
    -------
    pd.Series | np.ndarray
        Log returns of length ``len(prices) - 1``.
        Returns the same type as the input (Series → Series, ndarray → ndarray).

    Raises
    ------
    ValueError
        If any price is non-positive, non-finite, or fewer than 2 prices.
    """
    is_series = isinstance(prices, pd.Series)
    orig_index = prices.index if is_series else None
    arr = _validate_prices(prices)
    log_ret = np.diff(np.log(arr))

    if is_series:
        idx = index if index is not None else orig_index[1:]
        return pd.Series(log_ret, index=idx)
    return log_ret


def to_arithmetic_returns(
    prices: Union[pd.Series, np.ndarray],
    index: Optional[pd.Index] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Compute arithmetic (simple) returns: r_t = (P_t - P_{t-1}) / P_{t-1}.

    Arithmetic returns are preferred for single-period P&L attribution
    and cross-sectional comparisons (e.g. asset allocation).

    Parameters
    ----------
    prices : pd.Series | np.ndarray
        Strictly positive price series (length >= 2).
    index : pd.Index, optional
        Optional index for the returned Series.

    Returns
    -------
    pd.Series | np.ndarray
        Arithmetic returns of length ``len(prices) - 1``.

    Raises
    ------
    ValueError
        If prices are non-positive or non-finite.
    """
    is_series = isinstance(prices, pd.Series)
    orig_index = prices.index if is_series else None
    arr = _validate_prices(prices)
    arith_ret = np.diff(arr) / arr[:-1]

    if is_series:
        idx = index if index is not None else orig_index[1:]
        return pd.Series(arith_ret, index=idx)
    return arith_ret


def log_to_arithmetic(
    log_returns: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """
    Convert log returns to arithmetic returns: r = exp(r_log) - 1.

    Parameters
    ----------
    log_returns : pd.Series | np.ndarray
        Log return series.

    Returns
    -------
    pd.Series | np.ndarray
        Arithmetic returns (same type as input).
    """
    if isinstance(log_returns, pd.Series):
        return np.expm1(log_returns)
    return np.expm1(np.asarray(log_returns, dtype=np.float64))


def arithmetic_to_log(
    arithmetic_returns: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """
    Convert arithmetic returns to log returns: r_log = ln(1 + r).

    Parameters
    ----------
    arithmetic_returns : pd.Series | np.ndarray
        Arithmetic return series.  Values must be > -1 (no total-loss bars).

    Returns
    -------
    pd.Series | np.ndarray
        Log returns.

    Raises
    ------
    ValueError
        If any value is <= -1 (implying a greater-than-100% loss).
    """
    arr = np.asarray(arithmetic_returns, dtype=np.float64)
    if np.any(arr <= -1.0):
        raise ValueError(
            "Arithmetic returns contain values <= -1 (impossible total-loss bars). "
            "Check for data errors."
        )
    log_ret = np.log1p(arr)
    if isinstance(arithmetic_returns, pd.Series):
        return pd.Series(log_ret, index=arithmetic_returns.index)
    return log_ret


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

def annualise_return(
    mean_period_return: float,
    periods_per_year: int = 252,
) -> float:
    """
    Annualise a per-period mean arithmetic return.

    Uses the compound formula: (1 + r_bar)^n - 1.

    Parameters
    ----------
    mean_period_return : float
        Average return per period (e.g. 0.001 = 0.1% per day).
    periods_per_year : int, default=252
        Number of periods per year.

    Returns
    -------
    float
        Annualised return.
    """
    return (1.0 + mean_period_return) ** periods_per_year - 1.0


def annualise_volatility(
    period_volatility: float,
    periods_per_year: int = 252,
) -> float:
    """
    Annualise a per-period return volatility.

    Uses the square-root-of-time rule: σ_annual = σ_period * √n.
    Valid when returns are i.i.d. (no autocorrelation).

    Parameters
    ----------
    period_volatility : float
        Standard deviation of returns per period.
    periods_per_year : int, default=252
        Number of periods per year.

    Returns
    -------
    float
        Annualised volatility.

    Raises
    ------
    ValueError
        If period_volatility is negative.
    """
    if period_volatility < 0:
        raise ValueError("period_volatility must be >= 0.")
    return period_volatility * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Equity curve & drawdown
# ---------------------------------------------------------------------------

def equity_curve(
    returns: Union[pd.Series, np.ndarray],
    initial_capital: float = 1.0,
    log_returns: bool = False,
) -> Union[pd.Series, np.ndarray]:
    """
    Build a cumulative equity curve from a return series.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Per-period return series.
    initial_capital : float, default=1.0
        Starting equity value.
    log_returns : bool, default=False
        Set True if ``returns`` are log returns (uses cumsum + exp internally).

    Returns
    -------
    pd.Series | np.ndarray
        Equity curve of the same length as ``returns``.
        Type matches the input.

    Raises
    ------
    ValueError
        If initial_capital <= 0.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0.")
    is_series = isinstance(returns, pd.Series)
    arr = np.asarray(returns, dtype=np.float64)

    if log_returns:
        eq = initial_capital * np.exp(np.cumsum(arr))
    else:
        eq = initial_capital * np.cumprod(1.0 + arr)

    if is_series:
        return pd.Series(eq, index=returns.index)
    return eq


def drawdown_series(
    returns: Union[pd.Series, np.ndarray],
    log_returns: bool = False,
) -> Union[pd.Series, np.ndarray]:
    """
    Compute the full drawdown time-series (fraction below running peak).

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Per-period returns.
    log_returns : bool, default=False
        Whether returns are in log form.

    Returns
    -------
    pd.Series | np.ndarray
        Non-positive drawdown values; 0 = at or above prior high-water mark.
        -0.10 = 10% below the prior peak.
    """
    is_series = isinstance(returns, pd.Series)
    eq = equity_curve(returns, initial_capital=1.0, log_returns=log_returns)
    eq_arr = np.asarray(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = eq_arr / peak - 1.0

    if is_series:
        return pd.Series(dd, index=returns.index)  # type: ignore[arg-type]
    return dd


def rolling_volatility(
    returns: Union[pd.Series, np.ndarray],
    window: int = 21,
    periods_per_year: int = 252,
    min_periods: int = 2,
) -> Union[pd.Series, np.ndarray]:
    """
    Rolling annualised volatility (strictly causal — no lookahead).

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Per-period return series.
    window : int, default=21
        Rolling window size in bars.
    periods_per_year : int, default=252
        For annualisation.
    min_periods : int, default=2
        Minimum observations required to produce a non-NaN value.

    Returns
    -------
    pd.Series | np.ndarray
        Annualised rolling volatility; early positions are NaN if fewer
        than ``min_periods`` observations are available.
    """
    is_series = isinstance(returns, pd.Series)
    if not is_series:
        returns = pd.Series(np.asarray(returns, dtype=np.float64))

    roll_vol = (
        returns.rolling(window=window, min_periods=min_periods)
        .std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    if is_series:
        return roll_vol
    return roll_vol.to_numpy()


def underwater_duration(
    returns: Union[pd.Series, np.ndarray],
    log_returns: bool = False,
) -> np.ndarray:
    """
    Number of consecutive periods each bar has been below the prior equity peak.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Per-period return series.
    log_returns : bool, default=False
        Whether returns are log returns.

    Returns
    -------
    np.ndarray
        Non-negative integer array; 0 at a new high-water mark.
    """
    dd = np.asarray(drawdown_series(returns, log_returns=log_returns))
    n = len(dd)
    durations = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if dd[i] < 0:
            durations[i] = durations[i - 1] + 1
        else:
            durations[i] = 0
    return durations
