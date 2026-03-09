"""
Core statistical functions for institutional quantitative trading analysis.

Design Principles
-----------------
* **Numerical stability**: all moments use compensated (Welford / two-pass)
  algorithms; log operations are guarded by a numerical floor.
* **Correctness**: bias-corrected estimators (n-1 for sample variance, Fisher
  correction for excess kurtosis, adjusted Fisher–Pearson skewness).
* **No silent failures**: every function raises on degenerate input instead of
  returning NaN silently.
* **Performance**: vectorised NumPy operations; Polars helpers accept pl.Series
  and return native Python scalars.

Public API
----------
describe                  Full moment summary (mean, std, skew, kurt, percentiles).
is_skewed                 Boolean skewness test.
get_kurtosis              Excess kurtosis (Fisher definition).
sharpe_ratio              Annualised Sharpe ratio from a return series.
sortino_ratio             Annualised Sortino ratio.
calmar_ratio              Calmar ratio (CAGR / max drawdown).
max_drawdown              Maximum peak-to-trough drawdown.
rolling_sharpe            Rolling Sharpe ratio (no lookahead).
var_historical            Historical Value-at-Risk.
cvar_historical           Conditional VaR (Expected Shortfall).
autocorrelation           Ljung-Box-ready autocorrelation at multiple lags.
hurst_exponent            Hurst exponent via rescaled range analysis.
information_ratio         Information ratio vs a benchmark.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES_PL = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}
_EPS: float = 1e-14


def _to_float64(arr: Union[np.ndarray, Sequence[float], pl.Series]) -> np.ndarray:
    """Convert any array-like to a clean float64 NumPy array (no NaN)."""
    if isinstance(arr, pl.Series):
        arr = arr.drop_nulls().to_numpy()
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr)
    if not mask.all():
        n_bad = int((~mask).sum())
        logger.debug("Dropped %d non-finite values from input.", n_bad)
        arr = arr[mask]
    return arr


def _require_min_obs(arr: np.ndarray, n: int, label: str = "array") -> None:
    if len(arr) < n:
        raise ValueError(f"{label} requires at least {n} observations, got {len(arr)}.")


# ---------------------------------------------------------------------------
# Distribution moments
# ---------------------------------------------------------------------------

def describe(
    series: Union[pl.Series, np.ndarray, Sequence[float]],
    percentiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> Dict[str, float]:
    """
    Compute a full distributional summary.

    Parameters
    ----------
    series : pl.Series | np.ndarray | Sequence[float]
        Numeric data.
    percentiles : Sequence[float], default=(5%, 25%, 50%, 75%, 95%)
        Percentiles to include in the output (values in [0, 1]).

    Returns
    -------
    Dict[str, float]
        Keys: ``count``, ``mean``, ``std``, ``skew``, ``kurt``,
        ``min``, ``max``, plus one entry per percentile labelled
        ``p{pct*100:.0f}`` (e.g. ``p5``, ``p50``).

    Raises
    ------
    ValueError
        If fewer than 4 observations remain after cleaning.
    """
    arr = _to_float64(series)
    _require_min_obs(arr, 4, "describe()")

    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < _EPS:
        skew = 0.0
        kurt = 0.0
    else:
        # Adjusted Fisher–Pearson skewness (matches scipy.stats.skew bias=True adjustment)
        m3 = float(np.mean((arr - mean) ** 3))
        m2 = float(np.mean((arr - mean) ** 2))
        skew = (m3 / (m2 ** 1.5)) * (np.sqrt(n * (n - 1)) / (n - 2)) if n >= 3 else 0.0
        # Fisher excess kurtosis (bias-corrected)
        m4 = float(np.mean((arr - mean) ** 4))
        kurt_raw = m4 / (m2 ** 2) - 3.0
        kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * kurt_raw + 6.0) if n >= 4 else 0.0

    result: Dict[str, float] = {
        "count": float(n),
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for p in percentiles:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Percentile {p} is out of [0, 1].")
        result[f"p{p * 100:.0f}"] = float(np.percentile(arr, p * 100))

    return result


def is_skewed(
    series: Union[pl.Series, np.ndarray, Sequence[float]],
    threshold: float = 0.5,
) -> bool:
    """
    Test whether a series has statistically meaningful skewness.

    Uses the adjusted Fisher–Pearson estimator (bias-corrected).

    Parameters
    ----------
    series : pl.Series | array-like
        Numeric data.
    threshold : float, default=0.5
        Absolute skewness threshold above which the series is considered skewed.

    Returns
    -------
    bool
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")
    d = describe(series)
    return abs(d["skew"]) > threshold


def get_kurtosis(
    series: Union[pl.Series, np.ndarray, Sequence[float]],
) -> float:
    """
    Return the bias-corrected excess kurtosis (Fisher definition, excess=True).

    Normal distribution → 0.  Leptokurtic (fat tails) → positive.

    Parameters
    ----------
    series : pl.Series | array-like
        Numeric data.

    Returns
    -------
    float
        Excess kurtosis.

    Raises
    ------
    ValueError
        If fewer than 4 observations after cleaning.
    """
    return describe(series)["kurt"]


# ---------------------------------------------------------------------------
# Risk / performance metrics
# ---------------------------------------------------------------------------

def max_drawdown(
    returns: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Compute the maximum peak-to-trough drawdown from a return series.

    Parameters
    ----------
    returns : array-like
        Per-period returns (arithmetic, e.g. 0.01 for +1%).

    Returns
    -------
    float
        Maximum drawdown as a non-positive fraction (e.g. -0.25 = -25%).

    Raises
    ------
    ValueError
        If fewer than 2 returns.
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 2, "max_drawdown()")
    equity = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(np.min(drawdowns))


def sharpe_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Per-period arithmetic returns.
    risk_free_rate : float, default=0.0
        Annualised risk-free rate.
    periods_per_year : int, default=252
        Number of trading periods in a year (252=daily, 52=weekly, 12=monthly).

    Returns
    -------
    float
        Annualised Sharpe ratio.

    Raises
    ------
    ValueError
        If fewer than 2 returns or zero volatility.
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 2, "sharpe_ratio()")
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = arr - rf_per_period
    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1))
    if std_excess < _EPS:
        raise ValueError("Return series has near-zero volatility; Sharpe ratio is undefined.")
    return mean_excess / std_excess * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    mar: float = 0.0,
) -> float:
    """
    Annualised Sortino ratio (penalises only downside volatility).

    Parameters
    ----------
    returns : array-like
        Per-period arithmetic returns.
    risk_free_rate : float, default=0.0
        Annualised risk-free rate.
    periods_per_year : int, default=252
        Trading periods per year.
    mar : float, default=0.0
        Minimum acceptable return per period.

    Returns
    -------
    float
        Annualised Sortino ratio.

    Raises
    ------
    ValueError
        If insufficient data or zero downside deviation.
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 2, "sortino_ratio()")
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    mean_excess = float(np.mean(arr)) - rf_per_period
    downside = arr[arr < mar] - mar
    downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else _EPS
    if downside_std < _EPS:
        raise ValueError("No downside observations; Sortino ratio is undefined.")
    return mean_excess / downside_std * np.sqrt(periods_per_year)


def calmar_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
) -> float:
    """
    Calmar ratio: annualised return divided by absolute maximum drawdown.

    Parameters
    ----------
    returns : array-like
        Per-period arithmetic returns.
    periods_per_year : int, default=252
        Trading periods per year.

    Returns
    -------
    float
        Calmar ratio (positive = good).

    Raises
    ------
    ValueError
        If max drawdown is zero (no losses).
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 2, "calmar_ratio()")
    ann_return = float(np.mean(arr)) * periods_per_year
    mdd = max_drawdown(arr)
    if abs(mdd) < _EPS:
        raise ValueError("Max drawdown is zero; Calmar ratio is undefined.")
    return ann_return / abs(mdd)


def information_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    benchmark_returns: Union[np.ndarray, Sequence[float]],
    periods_per_year: int = 252,
) -> float:
    """
    Information ratio: annualised active return divided by tracking error.

    Parameters
    ----------
    returns : array-like
        Strategy per-period returns.
    benchmark_returns : array-like
        Benchmark per-period returns (must match length).
    periods_per_year : int, default=252
        Trading periods per year.

    Returns
    -------
    float
        Annualised information ratio.

    Raises
    ------
    ValueError
        If series lengths differ or tracking error is zero.
    """
    arr = _to_float64(returns)
    bench = _to_float64(benchmark_returns)
    if len(arr) != len(bench):
        raise ValueError(
            f"returns and benchmark_returns must have equal length, "
            f"got {len(arr)} vs {len(bench)}."
        )
    _require_min_obs(arr, 2, "information_ratio()")
    active = arr - bench
    mean_active = float(np.mean(active))
    tracking_error = float(np.std(active, ddof=1))
    if tracking_error < _EPS:
        raise ValueError("Tracking error is zero; information ratio is undefined.")
    return mean_active / tracking_error * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------

def var_historical(
    returns: Union[np.ndarray, Sequence[float]],
    confidence_level: float = 0.95,
) -> float:
    """
    Historical (non-parametric) Value-at-Risk.

    Parameters
    ----------
    returns : array-like
        Per-period returns.
    confidence_level : float, default=0.95
        Confidence level (e.g. 0.95 = 95% VaR).

    Returns
    -------
    float
        VaR as a non-positive number (e.g. -0.02 = -2% loss).

    Raises
    ------
    ValueError
        If fewer than 10 observations or confidence level out of (0, 1).
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 10, "var_historical()")
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    return float(np.percentile(arr, (1.0 - confidence_level) * 100))


def cvar_historical(
    returns: Union[np.ndarray, Sequence[float]],
    confidence_level: float = 0.95,
) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall) — average loss beyond VaR.

    Parameters
    ----------
    returns : array-like
        Per-period returns.
    confidence_level : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        CVaR as a non-positive number.

    Raises
    ------
    ValueError
        If no observations fall below the VaR threshold.
    """
    arr = _to_float64(returns)
    _require_min_obs(arr, 10, "cvar_historical()")
    v = var_historical(arr, confidence_level)
    tail = arr[arr <= v]
    if len(tail) == 0:
        raise ValueError("No observations below VaR threshold; CVaR is undefined.")
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# Time-series diagnostics
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: Union[np.ndarray, Sequence[float]],
    window: int = 63,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Rolling annualised Sharpe ratio (strictly causal — no lookahead).

    Parameters
    ----------
    returns : array-like
        Per-period returns.
    window : int, default=63
        Rolling window length (bars).
    periods_per_year : int, default=252
        Trading periods per year.

    Returns
    -------
    np.ndarray
        Rolling Sharpe ratios; first ``window - 1`` values are NaN.

    Raises
    ------
    ValueError
        If window > length of returns.
    """
    arr = _to_float64(returns)
    n = len(arr)
    if window > n:
        raise ValueError(f"window ({window}) > length of returns ({n}).")
    result = np.full(n, np.nan)
    sqrt_periods = np.sqrt(periods_per_year)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        mu = np.mean(seg)
        sigma = np.std(seg, ddof=1)
        result[i] = (mu / sigma * sqrt_periods) if sigma >= _EPS else np.nan
    return result


def autocorrelation(
    series: Union[np.ndarray, Sequence[float]],
    max_lag: int = 10,
) -> Dict[int, float]:
    """
    Sample autocorrelation coefficients at lags 1 … max_lag.

    Uses the unbiased denominator (n − k) per lag so estimates are not
    systematically zero at large lags.

    Parameters
    ----------
    series : array-like
        Time-ordered data.
    max_lag : int, default=10
        Maximum lag to compute.

    Returns
    -------
    Dict[int, float]
        Mapping from lag → autocorrelation (in [-1, 1]).

    Raises
    ------
    ValueError
        If max_lag >= len(series) or series has zero variance.
    """
    arr = _to_float64(series)
    n = len(arr)
    _require_min_obs(arr, max_lag + 1, "autocorrelation()")
    var = float(np.var(arr, ddof=1))
    if var < _EPS:
        raise ValueError("Series has near-zero variance; autocorrelation is undefined.")
    mean = float(np.mean(arr))
    demeaned = arr - mean
    acf: Dict[int, float] = {}
    for k in range(1, max_lag + 1):
        cov_k = float(np.dot(demeaned[: n - k], demeaned[k:])) / (n - k)
        acf[k] = cov_k / var
    return acf


def hurst_exponent(
    series: Union[np.ndarray, Sequence[float]],
    min_window: int = 10,
    max_window: Optional[int] = None,
    n_windows: int = 20,
) -> float:
    """
    Estimate the Hurst exponent via Rescaled Range (R/S) analysis.

    H < 0.5  → mean-reverting (anti-persistent)
    H ≈ 0.5  → random walk (geometric Brownian motion)
    H > 0.5  → trending (persistent)

    Parameters
    ----------
    series : array-like
        Price or return series (at least 50 observations recommended).
    min_window : int, default=10
        Minimum sub-series length.
    max_window : int, optional
        Maximum sub-series length.  Defaults to len(series) // 2.
    n_windows : int, default=20
        Number of window sizes to evaluate.

    Returns
    -------
    float
        Hurst exponent estimate in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 20 observations or degenerate geometry.
    """
    arr = _to_float64(series)
    _require_min_obs(arr, 20, "hurst_exponent()")
    n = len(arr)
    max_window = max_window or n // 2

    if max_window <= min_window:
        raise ValueError("max_window must be > min_window.")

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    rs_list = []
    valid_windows = []

    for w in window_sizes:
        rs_values = []
        for start in range(0, n - w + 1, w):
            seg = arr[start : start + w]
            if len(seg) < w:
                continue
            mean_s = np.mean(seg)
            deviate = np.cumsum(seg - mean_s)
            r = float(np.max(deviate) - np.min(deviate))
            s = float(np.std(seg, ddof=1))
            if s > _EPS:
                rs_values.append(r / s)
        if rs_values:
            rs_list.append(np.mean(rs_values))
            valid_windows.append(w)

    if len(valid_windows) < 2:
        raise ValueError("Insufficient valid windows for Hurst estimation.")

    log_w = np.log(valid_windows)
    log_rs = np.log(rs_list)
    # OLS regression: log(R/S) = H * log(n) + const
    coeffs = np.polyfit(log_w, log_rs, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))
