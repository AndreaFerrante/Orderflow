"""
Correlation analysis for multi-asset quantitative trading research.

Every function in this module is designed to answer a concrete research
question:
- Are two return series co-moving?  (rolling_correlation)
- Is the linear relationship stable over time?  (correlation_stability)
- Is the dependence monotonic rather than linear?  (rank_correlation)
- Are the portfolio risk factors truly independent?  (correlation_eigenvalues)

Design Principles
-----------------
* **No lookahead**: all rolling operations use strictly past data.
* **Rank-based robustness**: Spearman / Kendall available for non-linear monotonic
  dependence and resistance to outliers.
* **Numerical stability**: correlation matrix eigenvalue analysis guards against
  near-singular matrices and uses double precision throughout.
* **Large-dataset performance**: vectorised NumPy / pandas operations.

Public API
----------
rolling_correlation       Rolling pairwise Pearson correlation (causal).
rank_correlation          Spearman and Kendall rank correlation.
correlation_stability     Detect instability in pairwise correlation over time.
correlation_eigenvalues   Eigenvalue decomposition for factor analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ._validators import EPS, to_float64, require_min_obs, validate_window

logger = logging.getLogger(__name__)


def rolling_correlation(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    window: int = 63,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Rolling Pearson correlation (strictly causal — no lookahead).

    Parameters
    ----------
    x, y : pd.Series | np.ndarray
        Two return or price-change series of equal length.
    window : int, default=63
        Rolling window size (bars).
    min_periods : int, default=20
        Minimum observations for a valid correlation estimate.

    Returns
    -------
    np.ndarray
        Rolling correlation; early positions are NaN.

    Raises
    ------
    ValueError
        If series lengths differ or window exceeds length.
    """
    x_arr = to_float64(x, drop_nonfinite=False, label="rolling_correlation(x)")
    y_arr = to_float64(y, drop_nonfinite=False, label="rolling_correlation(y)")
    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"x and y must have equal length, got {len(x_arr)} vs {len(y_arr)}."
        )
    n = len(x_arr)
    validate_window(window, n, "window")

    sx = pd.Series(x_arr)
    sy = pd.Series(y_arr)
    corr = sx.rolling(window=window, min_periods=min_periods).corr(sy)
    return corr.to_numpy()


def rank_correlation(
    x: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
    method: Literal["spearman", "kendall"] = "spearman",
) -> Tuple[float, float]:
    """
    Rank-based correlation coefficient with p-value.

    Rank correlations are robust to outliers and capture monotonic (not just
    linear) dependence.  Preferred for fat-tailed financial return series.

    Parameters
    ----------
    x, y : array-like
        Paired numeric observations (equal length).
    method : {"spearman", "kendall"}, default="spearman"
        Correlation method.

    Returns
    -------
    Tuple[float, float]
        ``(correlation, p_value)``

    Raises
    ------
    ValueError
        If lengths differ or fewer than 3 observations.
    """
    x_arr = to_float64(x, label="rank_correlation(x)")
    y_arr = to_float64(y, label="rank_correlation(y)")
    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"x and y must have equal length, got {len(x_arr)} vs {len(y_arr)}."
        )
    require_min_obs(x_arr, 3, "rank_correlation()")

    if method == "spearman":
        corr, pval = sp_stats.spearmanr(x_arr, y_arr)
    elif method == "kendall":
        corr, pval = sp_stats.kendalltau(x_arr, y_arr)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'spearman' or 'kendall'.")

    return float(corr), float(pval)


def correlation_stability(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    window: int = 63,
    n_splits: int = 4,
) -> Dict[str, float]:
    """
    Measure the stability of pairwise correlation over time.

    Splits the series into *n_splits* non-overlapping chronological segments
    and computes Pearson correlation in each.  Returns the mean, std, and
    range of the segment correlations — a large std or range indicates
    an unstable relationship.

    Parameters
    ----------
    x, y : pd.Series | np.ndarray
        Equal-length return series.
    window : int, default=63
        Not used for splitting but reserved for rolling analysis extensions.
    n_splits : int, default=4
        Number of chronological segments (e.g. 4 = quarterly for daily data).

    Returns
    -------
    Dict[str, float]
        Keys: ``mean_corr``, ``std_corr``, ``min_corr``, ``max_corr``,
        ``range_corr``, ``n_splits``.

    Raises
    ------
    ValueError
        If lengths differ or fewer than ``10 * n_splits`` observations.
    """
    x_arr = to_float64(x, label="correlation_stability(x)")
    y_arr = to_float64(y, label="correlation_stability(y)")
    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"x and y must have equal length, got {len(x_arr)} vs {len(y_arr)}."
        )
    n = len(x_arr)
    if n < 10 * n_splits:
        raise ValueError(
            f"Need at least {10 * n_splits} observations for {n_splits} splits, got {n}."
        )

    split_size = n // n_splits
    corrs = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size
        seg_x = x_arr[start:end]
        seg_y = y_arr[start:end]
        std_x = float(np.std(seg_x, ddof=1))
        std_y = float(np.std(seg_y, ddof=1))
        if std_x < EPS or std_y < EPS:
            continue
        c = float(np.corrcoef(seg_x, seg_y)[0, 1])
        corrs.append(c)

    if len(corrs) < 2:
        raise ValueError("Not enough non-degenerate segments to measure stability.")

    arr_c = np.array(corrs)
    return {
        "mean_corr": float(np.mean(arr_c)),
        "std_corr": float(np.std(arr_c, ddof=1)),
        "min_corr": float(np.min(arr_c)),
        "max_corr": float(np.max(arr_c)),
        "range_corr": float(np.max(arr_c) - np.min(arr_c)),
        "n_splits": len(corrs),
    }


def correlation_eigenvalues(
    returns_matrix: Union[np.ndarray, pd.DataFrame],
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Eigenvalue decomposition of the correlation matrix.

    Useful for detecting:
    - **Dominant risk factors**: eigenvalues >> 1.0 indicate systematic factors.
    - **Multicollinearity**: near-zero eigenvalues indicate redundant series.
    - **Random noise**: eigenvalues clustered at 1.0 indicate no structure.

    The Marchenko–Pastur threshold (for random matrices) is provided so
    researchers can distinguish signal from noise.

    Parameters
    ----------
    returns_matrix : np.ndarray | pd.DataFrame
        Shape ``(n_observations, n_assets)``.  Each column is a return series.

    Returns
    -------
    Dict[str, ...]
        Keys:
        - ``eigenvalues``: np.ndarray sorted descending.
        - ``explained_variance_ratio``: np.ndarray (fraction of total variance).
        - ``n_significant``: int — eigenvalues above the Marchenko–Pastur upper bound.
        - ``mp_upper``: float — Marchenko–Pastur upper edge.
        - ``condition_number``: float — ratio of largest to smallest eigenvalue.

    Raises
    ------
    ValueError
        If fewer than 2 columns or fewer observations than columns.
    """
    if isinstance(returns_matrix, pd.DataFrame):
        mat = returns_matrix.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(returns_matrix, dtype=np.float64)

    if mat.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D, got shape {mat.shape}.")
    n_obs, n_assets = mat.shape
    if n_assets < 2:
        raise ValueError("Need at least 2 assets for correlation analysis.")
    if n_obs < n_assets:
        raise ValueError(
            f"Need at least as many observations ({n_obs}) as assets ({n_assets})."
        )

    # Drop rows with any NaN
    mask = np.all(np.isfinite(mat), axis=1)
    mat = mat[mask]
    n_obs = mat.shape[0]

    corr = np.corrcoef(mat, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]  # descending

    # Marchenko–Pastur upper bound for a random correlation matrix
    q = n_obs / n_assets
    mp_upper = (1.0 + 1.0 / np.sqrt(q)) ** 2

    total_var = float(np.sum(eigenvalues))
    evr = eigenvalues / total_var
    n_significant = int(np.sum(eigenvalues > mp_upper))

    cond = float(eigenvalues[0] / max(eigenvalues[-1], EPS))

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": evr,
        "n_significant": n_significant,
        "mp_upper": float(mp_upper),
        "condition_number": cond,
    }
