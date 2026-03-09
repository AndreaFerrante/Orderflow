"""
Utilities for Markov state generation and HMM feature engineering.

Design principles:
- No lookahead bias: all rolling computations use strictly past observations.
- Numerical stability: guards against zero-variance windows and degenerate inputs.
- Performance: vectorised NumPy operations throughout; pure-Python loops only where
  unavoidable (e.g. online slope estimation for very large arrays).
- Reproducibility: no global random state mutations; seeds are caller-controlled.
"""

import os
import logging
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_STATES: tuple = ("UP", "DOWN", "FLAT")
_EPS: float = 1e-10  # numerical floor — avoids log(0)


# ---------------------------------------------------------------------------
# State generation from price sequences
# ---------------------------------------------------------------------------

def threshold_prices_states(
    prices: List[float],
    threshold: float = 1e-8,
) -> List[str]:
    """
    Convert a price series into UP / DOWN / FLAT states using a fixed threshold.

    Returns ``len(prices) - 1`` states.  The threshold is an *absolute* price
    difference; for instruments with different tick sizes pass the tick size or
    a sensible multiple.

    Parameters
    ----------
    prices : List[float]
        Chronologically ordered price observations.
    threshold : float, default=1e-8
        Minimum absolute price change to classify as UP or DOWN.

    Returns
    -------
    List[str]
        Sequence of "UP" / "DOWN" / "FLAT" labels.

    Raises
    ------
    ValueError
        If ``len(prices) < 2`` or ``threshold < 0``.
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to produce states.")
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")

    arr = np.asarray(prices, dtype=np.float64)
    diffs = np.diff(arr)
    states: List[str] = []
    for d in diffs:
        if d > threshold:
            states.append("UP")
        elif d < -threshold:
            states.append("DOWN")
        else:
            states.append("FLAT")
    return states


def adaptive_threshold_prices_states(
    prices: List[float],
    window: int = 20,
    z_score_threshold: float = 0.5,
) -> List[str]:
    """
    Convert a price series into states using an adaptive, volatility-scaled threshold.

    The threshold at bar *i* equals ``z_score_threshold * σ(returns[i-window:i])``.
    Only *past* returns are used — strictly no lookahead bias.

    Returns ``len(prices) - 1`` states.

    Parameters
    ----------
    prices : List[float]
        Chronologically ordered price observations.
    window : int, default=20
        Rolling window length for volatility estimation.
    z_score_threshold : float, default=0.5
        Number of standard deviations that defines UP / DOWN.

    Returns
    -------
    List[str]
        Sequence of "UP" / "DOWN" / "FLAT" labels.

    Raises
    ------
    ValueError
        If fewer than 2 prices, window < 2, or z_score_threshold <= 0.
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices.")
    if window < 2:
        raise ValueError("window must be >= 2.")
    if z_score_threshold <= 0:
        raise ValueError("z_score_threshold must be > 0.")

    arr = np.asarray(prices, dtype=np.float64)
    diffs = np.diff(arr)           # length = n - 1
    n = len(diffs)
    states: List[str] = []

    for i in range(n):
        diff = diffs[i]
        # Strictly past window: diffs[max(0,i-window) : i]
        start = max(0, i - window)
        local = diffs[start:i]
        vol = float(np.std(local)) if len(local) >= 2 else _EPS
        vol = max(vol, _EPS)
        band = z_score_threshold * vol
        if diff > band:
            states.append("UP")
        elif diff < -band:
            states.append("DOWN")
        else:
            states.append("FLAT")

    return states


# ---------------------------------------------------------------------------
# Feature engineering for HMM inputs
# ---------------------------------------------------------------------------

def compute_df_features(
    df: pd.DataFrame,
    window_volatility: int = 20,
    window_slope: int = 5,
) -> pd.DataFrame:
    """
    Engineer standardised features from a bar DataFrame for HMM input.

    Computes four causal (no lookahead) features:

    * ``log_return``   — log price return: ``ln(price_t / price_{t-1})``
    * ``volatility``   — rolling std of log returns over *window_volatility* bars
    * ``slope``        — OLS slope of price over last *window_slope* bars
    * ``log_volume``   — ``log1p(volume)`` for variance stabilisation

    All rolling operations use only past data (``min_periods=2``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``'price'`` (float) and ``'volume'`` (float >= 0).
    window_volatility : int, default=20
        Rolling window for volatility.
    window_slope : int, default=5
        Rolling window for price slope.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``log_return``, ``volatility``, ``slope``,
        ``log_volume``.  NaNs in early rows are forward-filled then
        back-filled so the matrix is complete.

    Raises
    ------
    ValueError
        If required columns are missing or windows are out of range.
    """
    required = {"price", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if window_volatility < 2:
        raise ValueError("window_volatility must be >= 2.")
    if window_slope < 2:
        raise ValueError("window_slope must be >= 2.")

    out = df.copy()

    # Log returns (causal: shift(1) refers to previous bar)
    price = out["price"].astype(np.float64)
    log_ret = np.log(price / price.shift(1))   # NaN at index 0 — expected
    out["log_return"] = log_ret

    # Rolling volatility — strictly past window
    out["volatility"] = (
        log_ret
        .rolling(window=window_volatility, min_periods=2)
        .std()
    )

    # Rolling OLS slope — uses pandas rolling apply with a causal window.
    # The x-coordinates are 0..w-1 regardless of window position; the formula
    # (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²) is applied inside a vectorised helper.
    _w = window_slope
    _x = np.arange(_w, dtype=np.float64)
    _sx = _x.sum()
    _sx2 = (_x * _x).sum()
    _denom = _w * _sx2 - _sx * _sx

    def _ols_slope(seg: np.ndarray) -> float:
        m = len(seg)
        if m < 2:
            return 0.0
        if m < _w:
            # Partial window at start — recompute for actual length
            xp = np.arange(m, dtype=np.float64)
            sx_p = xp.sum(); sx2_p = (xp * xp).sum()
            d = m * sx2_p - sx_p * sx_p
            return (m * (xp * seg).sum() - sx_p * seg.sum()) / d if d != 0.0 else 0.0
        sy = seg.sum()
        sxy = (_x * seg).sum()
        return ((_w * sxy - _sx * sy) / _denom) if _denom != 0.0 else 0.0

    out["slope"] = (
        price
        .rolling(window=_w, min_periods=2)
        .apply(_ols_slope, raw=True)
        .fillna(0.0)
    )

    # Log-volume
    volume = out["volume"].astype(np.float64).clip(lower=0.0)
    out["log_volume"] = np.log1p(volume)

    # Fill NaN only for derived features where warm-up NaNs are expected.
    # log_return at index 0 is structurally undefined (no prior price); fill with 0.
    out["log_return"] = out["log_return"].fillna(0.0)
    # Volatility requires min_periods=2 so early rows are NaN; back-fill with
    # the first valid estimate so downstream models receive a complete matrix.
    out["volatility"] = out["volatility"].bfill().fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# HMM model selection
# ---------------------------------------------------------------------------

def select_best_hmm_model(
    data: np.ndarray,
    n_states_range: List[int],
    covariance_type: Literal["full", "diag", "tied", "spherical"] = "full",
    criterion: Literal["bic", "aic"] = "bic",
    random_state: int = 42,
    n_iter: int = 200,
) -> hmm.GaussianHMM:
    """
    Select the Gaussian HMM with the best information criterion.

    Evaluates each candidate number of hidden states by fitting on the full
    dataset and computing BIC or AIC.  BIC is recommended for trading
    applications (more conservative — penalises complexity harder).

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape ``(n_samples, n_features)``, standardised.
    n_states_range : List[int]
        Candidate numbers of hidden states, e.g. ``[2, 3, 4, 5]``.
    covariance_type : {"full", "diag", "tied", "spherical"}, default="full"
        HMM covariance structure.
    criterion : {"bic", "aic"}, default="bic"
        Model selection criterion.
    random_state : int, default=42
        Seed for reproducibility.
    n_iter : int, default=200
        Maximum EM iterations.

    Returns
    -------
    hmm.GaussianHMM
        Best fitted model.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, or ``n_states_range`` is empty,
        or no model converges.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}.")
    if not n_states_range:
        raise ValueError("n_states_range must not be empty.")
    if data.shape[0] < max(n_states_range) * 10:
        logger.warning(
            "Fewer than 10 samples per state candidate. "
            "HMM estimates may be unreliable."
        )

    n_samples, n_features = data.shape
    best_model: Optional[hmm.GaussianHMM] = None
    best_score = np.inf

    for n_states in sorted(n_states_range):
        try:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
            )
            model.fit(data)
            log_likelihood = model.score(data)
        except Exception as exc:
            logger.warning(f"HMM(n_states={n_states}) failed: {exc}")
            continue

        # Parameter count (correct formula per covariance type)
        n_trans = n_states * (n_states - 1)          # transition matrix (rows sum to 1)
        n_means = n_states * n_features              # Gaussian means
        if covariance_type == "full":
            n_cov = n_states * n_features * (n_features + 1) // 2
        elif covariance_type == "diag":
            n_cov = n_states * n_features
        elif covariance_type == "tied":
            n_cov = n_features * (n_features + 1) // 2
        else:  # spherical
            n_cov = n_states
        n_params = n_trans + n_means + n_cov

        ic = (
            n_params * np.log(n_samples) - 2.0 * log_likelihood
            if criterion.lower() == "bic"
            else 2.0 * n_params - 2.0 * log_likelihood
        )

        logger.debug(
            f"HMM n_states={n_states}: {criterion.upper()}={ic:.2f}, "
            f"logL={log_likelihood:.2f}"
        )

        if ic < best_score:
            best_score = ic
            best_model = model

    if best_model is None:
        raise ValueError(
            "No HMM model converged. Check data quality and n_states_range."
        )

    logger.info(
        f"Selected HMM: n_states={best_model.n_components}, "
        f"{criterion.upper()}={best_score:.2f}"
    )
    return best_model


# ---------------------------------------------------------------------------
# SierraChart data loading
# ---------------------------------------------------------------------------

def concat_sc_bar_data(
    data_path: str,
    file_extension: str = "txt",
) -> pd.DataFrame:
    """
    Load and concatenate SierraChart bar export files from a directory.

    The instrument name is extracted from the file stem
    (e.g. ``ESH24-CME.scid_BarData.txt`` → ``ESH24-CME``).

    Parameters
    ----------
    data_path : str
        Directory containing the export files.
    file_extension : str, default="txt"
        File extension to match (without leading dot).

    Returns
    -------
    pd.DataFrame
        Concatenated and time-sorted bar data with an ``Instrument`` column.

    Raises
    ------
    ValueError
        If ``data_path`` is empty or no matching files are found.
    FileNotFoundError
        If ``data_path`` does not exist.
    """
    if not data_path:
        raise ValueError("data_path must not be empty.")

    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    ext = file_extension.lstrip(".")
    files = sorted(root.glob(f"*.{ext}"))
    if not files:
        raise ValueError(
            f"No *.{ext} files found in {data_path}."
        )

    frames: List[pd.DataFrame] = []
    for fp in files:
        single = pd.read_csv(fp, sep=",")
        single.columns = [str(c).strip() for c in single.columns]
        single = single.map(lambda x: x.strip() if isinstance(x, str) else x)
        single.insert(0, "Instrument", fp.stem.split(".")[0])
        frames.append(single)

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined.sort_values(["Date", "Time"], ascending=True, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def simulate_market_data(
    num_steps: int = 10_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV-style market data for testing and examples.

    Uses a geometric Brownian motion with mild drift.

    Parameters
    ----------
    num_steps : int, default=10_000
        Number of bars to simulate.
    seed : int, default=123
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``price``, ``volume``.
    """
    rng = np.random.default_rng(seed)       # local Generator — no global state mutation
    log_returns = rng.normal(0.0002, 0.001, size=num_steps)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    prices = np.concatenate([[100.0], prices])
    volume = np.maximum(rng.normal(1e5, 1e4, size=len(prices)), 0.0)
    return pd.DataFrame({"price": prices, "volume": volume})


def plot_distribution_of_float_series(
    series: pd.Series,
    bins: int = 75,
    title: str = "Series Distribution",
) -> None:
    """
    Plot a histogram of a float Series using matplotlib.

    Parameters
    ----------
    series : pd.Series
        Must be float dtype.
    bins : int, default=75
        Number of histogram bins.
    title : str, default="Series Distribution"
        Chart title.

    Raises
    ------
    ValueError
        If series is not float or is empty.
    """
    if not pd.api.types.is_float_dtype(series):
        raise ValueError("series must be float dtype.")
    clean = series.dropna()
    if len(clean) == 0:
        raise ValueError("series is empty after dropping NaNs.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clean.values, bins=bins, edgecolor="black", alpha=0.65, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
