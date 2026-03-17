"""
Statistical hypothesis tests for quantitative trading research.

Every test in this module is designed for *one* purpose: to tell a quant
researcher whether a statistical assumption holds or breaks.  Results are
returned as structured ``TestResult`` objects — never as raw p-values that
invite misinterpretation.

Design Principles
-----------------
* **No silent p-hacking**: all tests return full result objects with test
  statistic, p-value, and a boolean conclusion at a caller-specified α.
* **Multiple-testing correction**: Holm–Bonferroni (uniformly more powerful
  than Bonferroni) is provided as a first-class utility.
* **Stationarity battery**: ADF + KPSS together guard against the known
  weakness of either test in isolation.
* **No lookahead**: structural-break detection uses expanding or causal windows.
* **Reproducibility**: no global state mutation; all randomised tests accept
  explicit seeds.

Public API
----------
TestResult                Dataclass carrying all test outputs.
adf_test                  Augmented Dickey–Fuller unit root test.
kpss_test                 KPSS level/trend stationarity test.
is_stationary             Combined ADF + KPSS stationarity verdict.
jarque_bera_test          Normality test (skewness + kurtosis).
ljung_box_test            Serial correlation (white-noise) test.
holm_bonferroni           Multiple-comparison correction.
cusum_test                CUSUM structural-break detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from ._validators import EPS, to_float64, require_min_obs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestResult:
    """Immutable container for a single hypothesis-test outcome.

    Attributes
    ----------
    test_name : str
        Human-readable test identifier.
    statistic : float
        Computed test statistic.
    p_value : float
        Observed p-value (two-sided where applicable).
    reject_null : bool
        True if the null hypothesis is rejected at the given significance level.
    alpha : float
        Significance level used for the decision.
    detail : dict
        Extra information (critical values, lag selection, etc.).
    """

    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    detail: Dict

    def __repr__(self) -> str:
        verdict = "REJECT H₀" if self.reject_null else "FAIL TO REJECT H₀"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4g}, α={self.alpha} → {verdict}"
        )


# ---------------------------------------------------------------------------
# Unit root / stationarity tests
# ---------------------------------------------------------------------------

def adf_test(
    series: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.05,
    max_lag: Optional[int] = None,
    regression: Literal["c", "ct", "n"] = "c",
) -> TestResult:
    """
    Augmented Dickey–Fuller test for unit root.

    H₀: series has a unit root (non-stationary).
    H₁: series is stationary.

    Parameters
    ----------
    series : array-like
        Time-ordered data (minimum 20 observations recommended).
    alpha : float, default=0.05
        Significance level.
    max_lag : int, optional
        Maximum lag for AIC-based lag selection.  Defaults to
        ``int(12 * (n / 100) ** 0.25)`` (Schwert's rule).
    regression : {"c", "ct", "n"}, default="c"
        "c" = constant only, "ct" = constant + trend, "n" = none.

    Returns
    -------
    TestResult
    """
    from statsmodels.tsa.stattools import adfuller

    arr = to_float64(series, label="adf_test")
    require_min_obs(arr, 20, "adf_test()")

    if max_lag is None:
        max_lag = int(12.0 * (len(arr) / 100.0) ** 0.25)

    result = adfuller(arr, maxlag=max_lag, regression=regression, autolag="AIC")
    stat, pval, used_lag, nobs, crit, icbest = result

    return TestResult(
        test_name="Augmented Dickey-Fuller",
        statistic=float(stat),
        p_value=float(pval),
        reject_null=pval < alpha,
        alpha=alpha,
        detail={
            "used_lag": int(used_lag),
            "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "ic_best": float(icbest),
            "regression": regression,
        },
    )


def kpss_test(
    series: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.05,
    regression: Literal["c", "ct"] = "c",
    n_lags: Optional[int] = None,
) -> TestResult:
    """
    KPSS test for level or trend stationarity.

    H₀: series is stationary (around a constant or trend).
    H₁: series has a unit root.

    Note: KPSS has the *opposite* null of ADF.  Use both together
    via ``is_stationary()`` for a robust verdict.

    Parameters
    ----------
    series : array-like
        Time-ordered data.
    alpha : float, default=0.05
        Significance level.
    regression : {"c", "ct"}, default="c"
        "c" = level stationarity, "ct" = trend stationarity.
    n_lags : int, optional
        Lag truncation.  Defaults to Schwert's rule via statsmodels.

    Returns
    -------
    TestResult
    """
    from statsmodels.tsa.stattools import kpss as _kpss

    arr = to_float64(series, label="kpss_test")
    require_min_obs(arr, 20, "kpss_test()")

    kwargs = {"regression": regression}
    if n_lags is not None:
        kwargs["nlags"] = n_lags

    stat, pval, used_lag, crit = _kpss(arr, **kwargs)

    return TestResult(
        test_name="KPSS",
        statistic=float(stat),
        p_value=float(pval),
        reject_null=pval < alpha,
        alpha=alpha,
        detail={
            "used_lag": int(used_lag),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "regression": regression,
        },
    )


def is_stationary(
    series: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.05,
) -> Tuple[bool, str]:
    """
    Combined ADF + KPSS stationarity verdict.

    Interpretation matrix:
    ┌────────────┬──────────────┬─────────────────────────────┐
    │ ADF reject │ KPSS reject  │ Conclusion                  │
    ├────────────┼──────────────┼─────────────────────────────┤
    │ Yes        │ No           │ Stationary ✓                │
    │ No         │ Yes          │ Non-stationary (unit root)  │
    │ Yes        │ Yes          │ Trend-stationary            │
    │ No         │ No           │ Inconclusive                │
    └────────────┴──────────────┴─────────────────────────────┘

    Parameters
    ----------
    series : array-like
        Time-ordered data.
    alpha : float, default=0.05
        Significance level for both tests.

    Returns
    -------
    Tuple[bool, str]
        ``(is_stationary, explanation)``
    """
    adf = adf_test(series, alpha=alpha)
    kp = kpss_test(series, alpha=alpha)

    adf_reject = adf.reject_null   # reject unit root → evidence of stationarity
    kpss_reject = kp.reject_null   # reject stationarity → evidence of unit root

    if adf_reject and not kpss_reject:
        return True, "Stationary (ADF rejects unit root, KPSS does not reject stationarity)."
    elif not adf_reject and kpss_reject:
        return False, "Non-stationary (ADF fails to reject unit root, KPSS rejects stationarity)."
    elif adf_reject and kpss_reject:
        return True, "Trend-stationary (ADF rejects unit root, KPSS rejects level stationarity). Consider detrending."
    else:
        return False, "Inconclusive (both tests fail to reject). Collect more data or try differencing."


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

def jarque_bera_test(
    series: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.05,
) -> TestResult:
    """
    Jarque–Bera test for normality.

    H₀: data is drawn from a normal distribution.
    H₁: data is not normally distributed.

    This test checks whether the sample skewness and kurtosis match
    a Gaussian.  Financial return series almost always reject.

    Parameters
    ----------
    series : array-like
        Numeric data.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    TestResult
    """
    arr = to_float64(series, label="jarque_bera_test")
    require_min_obs(arr, 8, "jarque_bera_test()")

    stat, pval = sp_stats.jarque_bera(arr)

    return TestResult(
        test_name="Jarque-Bera",
        statistic=float(stat),
        p_value=float(pval),
        reject_null=pval < alpha,
        alpha=alpha,
        detail={
            "skewness": float(sp_stats.skew(arr)),
            "kurtosis": float(sp_stats.kurtosis(arr)),  # excess
        },
    )


# ---------------------------------------------------------------------------
# Serial correlation
# ---------------------------------------------------------------------------

def ljung_box_test(
    series: Union[np.ndarray, Sequence[float]],
    max_lag: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    """
    Ljung–Box test for serial correlation up to *max_lag*.

    H₀: the data are independently distributed (white noise).
    H₁: the data exhibit serial correlation.

    Uses the maximum-lag Q-statistic (portmanteau test).

    Parameters
    ----------
    series : array-like
        Time-ordered returns or residuals.
    max_lag : int, default=10
        Maximum lag to test.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    TestResult
        The ``detail`` dict contains per-lag Q-statistics and p-values.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    arr = to_float64(series, label="ljung_box_test")
    require_min_obs(arr, max_lag + 5, "ljung_box_test()")

    result = acorr_ljungbox(arr, lags=max_lag, return_df=True)
    # Use the final lag's Q-statistic and p-value for the overall decision
    final_stat = float(result["lb_stat"].iloc[-1])
    final_pval = float(result["lb_pvalue"].iloc[-1])

    return TestResult(
        test_name="Ljung-Box",
        statistic=final_stat,
        p_value=final_pval,
        reject_null=final_pval < alpha,
        alpha=alpha,
        detail={
            "max_lag": max_lag,
            "per_lag_stats": result["lb_stat"].to_dict(),
            "per_lag_pvalues": result["lb_pvalue"].to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> List[Tuple[int, float, bool]]:
    """
    Holm–Bonferroni step-down correction for multiple comparisons.

    Uniformly more powerful than standard Bonferroni while controlling
    the family-wise error rate (FWER) at level α.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw (unadjusted) p-values from *m* independent tests.
    alpha : float, default=0.05
        Family-wise significance level.

    Returns
    -------
    List[Tuple[int, float, bool]]
        Sorted list of ``(original_index, adjusted_p_value, reject_null)``.
        Sorted by original p-value ascending.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results: List[Tuple[int, float, bool]] = []
    cumulative_reject = True

    for rank, (orig_idx, raw_p) in enumerate(indexed):
        adjusted_p = min(raw_p * (m - rank), 1.0)
        reject = cumulative_reject and (adjusted_p < alpha)
        if not reject:
            cumulative_reject = False
        results.append((orig_idx, adjusted_p, reject))

    return results


# ---------------------------------------------------------------------------
# Structural break detection
# ---------------------------------------------------------------------------

def cusum_test(
    series: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.05,
) -> TestResult:
    """
    CUSUM (cumulative sum) test for structural breaks in the mean.

    Computes the Brown–Durbin–Evans CUSUM statistic using recursive
    residuals from an OLS model of the series on a constant.

    H₀: the parameters are stable (no structural break).
    H₁: there is at least one structural break in the mean.

    Parameters
    ----------
    series : array-like
        Time-ordered observations.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    TestResult
        ``detail`` contains the CUSUM path and the index of the
        maximum absolute departure (potential break point).
    """
    arr = to_float64(series, label="cusum_test")
    require_min_obs(arr, 20, "cusum_test()")
    n = len(arr)

    # Recursive residuals approach: regress y on a constant using expanding window
    cumsum_vals = np.zeros(n)
    expanding_mean = np.cumsum(arr) / np.arange(1, n + 1)
    residuals = arr - np.concatenate([[arr[0]], expanding_mean[:-1]])
    sigma = float(np.std(residuals[1:], ddof=1))
    if sigma < EPS:
        return TestResult(
            test_name="CUSUM",
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            detail={"cusum_path": cumsum_vals.tolist(), "break_index": None},
        )

    cumsum_vals = np.cumsum(residuals[1:]) / (sigma * np.sqrt(n))
    max_abs = float(np.max(np.abs(cumsum_vals)))
    break_idx = int(np.argmax(np.abs(cumsum_vals))) + 1  # +1 because residuals[1:]

    # Approximate p-value using Brownian bridge supremum distribution
    # P(sup|B(t)| > x) ≈ 2*exp(-2x²) for standard Brownian bridge
    p_value = float(min(2.0 * np.exp(-2.0 * max_abs ** 2), 1.0))

    return TestResult(
        test_name="CUSUM",
        statistic=max_abs,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
        detail={
            "cusum_path": cumsum_vals.tolist(),
            "break_index": break_idx,
            "n_observations": n,
        },
    )
