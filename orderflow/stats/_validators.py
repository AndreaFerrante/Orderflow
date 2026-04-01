"""
Shared input validation and type-coercion utilities for the stats module.

Centralises all array cleaning, minimum-observation guards, and dtype
enforcement so that every public function shares identical semantics.
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import numpy as np

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False

logger = logging.getLogger(__name__)

# Numerical floor — prevents log(0) and division by zero.
EPS: float = 1e-14


def to_float64(
    arr: Union[np.ndarray, Sequence[float], "pl.Series"],
    *,
    drop_nonfinite: bool = True,
    label: str = "input",
) -> np.ndarray:
    """Convert any numeric array-like to a clean float64 NumPy array.

    Parameters
    ----------
    arr : array-like or pl.Series
        Input data.
    drop_nonfinite : bool, default True
        If True, silently drop NaN / Inf values and log the count.
    label : str
        Name used in log messages for traceability.

    Returns
    -------
    np.ndarray
        1-D float64 array with only finite values (when *drop_nonfinite*).
    """
    if _HAS_POLARS and isinstance(arr, pl.Series):
        arr = arr.drop_nulls().to_numpy()
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if drop_nonfinite:
        mask = np.isfinite(arr)
        if not mask.all():
            n_bad = int((~mask).sum())
            logger.debug("Dropped %d non-finite values from %s.", n_bad, label)
            arr = arr[mask]
    return arr


def require_min_obs(arr: np.ndarray, n: int, label: str = "array") -> None:
    """Raise ``ValueError`` if *arr* has fewer than *n* elements."""
    if len(arr) < n:
        raise ValueError(
            f"{label} requires at least {n} observations, got {len(arr)}."
        )


def validate_positive_prices(
    prices: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """Coerce to float64 and guarantee strictly positive, finite prices."""
    arr = np.asarray(prices, dtype=np.float64).ravel()
    if arr.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {arr.shape}.")
    if len(arr) < 2:
        raise ValueError(f"Need at least 2 prices, got {len(arr)}.")
    if np.any(arr <= 0):
        raise ValueError(
            "All prices must be strictly positive for return calculations."
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"prices contains {n_bad} non-finite value(s).")
    return arr


def validate_confidence_level(cl: float) -> None:
    """Raise if *cl* is not in the open interval (0, 1)."""
    if not (0 < cl < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {cl}.")


def validate_window(window: int, n: int, label: str = "window") -> None:
    """Raise if *window* exceeds array length or is < 2."""
    if window < 2:
        raise ValueError(f"{label} must be >= 2, got {window}.")
    if window > n:
        raise ValueError(f"{label} ({window}) > array length ({n}).")
