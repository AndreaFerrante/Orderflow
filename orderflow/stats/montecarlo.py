"""
Monte Carlo simulation engine for quantitative trading strategy robustness analysis.

Design Principles
-----------------
* **Bootstrap correctness**: sampling is with replacement (non-parametric bootstrap)
  over trade P&L; temporal ordering is *not* assumed within individual trade samples
  since trades are treated as i.i.d. observations after strategy execution.
* **No lookahead bias**: the equity curve is always reconstructed forward-in-time
  from the sampled trade sequence.
* **Reproducibility**: every simulation uses a seeded Generator (numpy >=1.17 API)
  so results are deterministic given ``random_state``.
* **Numerical stability**: all aggregates use double precision; confidence intervals
  use exact order statistics (percentile), not Gaussian approximation.
* **Silent-failure prevention**: explicit validation with clear error messages before
  any computation begins.

Public API
----------
MonteCarloResult          Dataclass carrying all simulation outputs.
get_montecarlo_analysis   Run full bootstrap simulation → MonteCarloResult.
plot_montecarlo_paths     Plot all equity curve paths.
plot_montecarlo_distribution  Plot final-equity histogram with CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Immutable container for Monte Carlo simulation outputs.

    Attributes
    ----------
    equity_curves : List[np.ndarray]
        Cumulative P&L path for each simulation (shape: n_simulations × n_trades).
    final_equities : np.ndarray
        Terminal equity value of each simulation (length n_simulations).
    mean_equity : float
        Mean final equity across all simulations.
    std_equity : float
        Standard deviation of final equity.
    min_equity : float
        Worst-case final equity.
    max_equity : float
        Best-case final equity.
    ci_lower : float
        Lower bound of the confidence interval for final equity.
    ci_upper : float
        Upper bound of the confidence interval for final equity.
    win_rate : float
        Fraction of simulations ending with positive equity.
    confidence_level : float
        Confidence level used for the CI.
    n_simulations : int
        Number of simulations run.
    sample_size : int
        Number of trades per simulation.
    """
    equity_curves: List[np.ndarray]
    final_equities: np.ndarray
    mean_equity: float
    std_equity: float
    min_equity: float
    max_equity: float
    ci_lower: float
    ci_upper: float
    win_rate: float
    confidence_level: float
    n_simulations: int
    sample_size: int

    def summary(self) -> dict:
        """Return a flat dict of scalar summary statistics."""
        return {
            "mean_equity": self.mean_equity,
            "std_equity": self.std_equity,
            "min_equity": self.min_equity,
            "max_equity": self.max_equity,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "win_rate": self.win_rate,
            "confidence_level": self.confidence_level,
            "n_simulations": self.n_simulations,
            "sample_size": self.sample_size,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_trades(trades: pd.DataFrame, pnl_col: str, min_trades: int) -> None:
    """Raise informative errors on malformed input before computation begins."""
    if not isinstance(trades, pd.DataFrame):
        raise TypeError(f"trades must be a DataFrame, got {type(trades).__name__}.")
    missing = {pnl_col} - set(trades.columns)
    if missing:
        raise ValueError(f"trades DataFrame is missing required column(s): {missing}.")
    if trades[pnl_col].isna().all():
        raise ValueError(f"Column '{pnl_col}' contains only NaN values.")
    if len(trades) < min_trades:
        raise ValueError(
            f"Need at least {min_trades} trades for a meaningful simulation, "
            f"got {len(trades)}. Collect more historical data."
        )


def get_montecarlo_analysis(
    trades: pd.DataFrame,
    n_rows_sample: int,
    n_simulations: int = 1000,
    pnl_col: str = "Entry_Gains",
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    # Legacy alias kept for backward compatibility
    entry_col_name: Optional[str] = None,
    show_progress: bool = True,
) -> MonteCarloResult:
    """
    Non-parametric bootstrap Monte Carlo simulation for strategy robustness analysis.

    Each simulation draws *n_rows_sample* trades with replacement from the
    historical trade log and computes a cumulative P&L path.  Aggregate
    statistics describe the distribution of terminal equity across all paths.

    Parameters
    ----------
    trades : pd.DataFrame
        Historical trade log.  Must contain the P&L column specified by
        ``pnl_col``.  Any other columns are ignored.
    n_rows_sample : int
        Number of trades to draw per simulation.  Must satisfy
        ``5 <= n_rows_sample <= len(trades)``.
    n_simulations : int, default=1000
        Number of bootstrap replications.  Use >= 1000 for publication-quality CIs.
    pnl_col : str, default="Entry_Gains"
        Column containing per-trade P&L (gains/losses as scalars).
    confidence_level : float, default=0.95
        Confidence level for the terminal equity interval (exact percentile method).
    random_state : int, optional
        Seed for the NumPy Generator.  Guarantees reproducibility without
        mutating the global random state.
    entry_col_name : str, optional
        Deprecated alias for ``pnl_col``; kept for backward compatibility.
    show_progress : bool, default=True
        Whether to display a tqdm progress bar.

    Returns
    -------
    MonteCarloResult
        Dataclass containing equity curves, final equities, and all
        summary statistics.  Call ``.summary()`` for a flat dict.

    Raises
    ------
    TypeError
        If trades is not a DataFrame.
    ValueError
        If required column is missing, fewer than 30 trades, or invalid parameters.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from orderflow.stats import get_montecarlo_analysis
    >>> rng = np.random.default_rng(0)
    >>> trades = pd.DataFrame({"Entry_Gains": rng.normal(10, 50, 200)})
    >>> result = get_montecarlo_analysis(trades, n_rows_sample=100, n_simulations=1000)
    >>> print(result.summary())
    """
    # Backward-compatibility: support old keyword
    if entry_col_name is not None:
        pnl_col = entry_col_name

    _validate_trades(trades, pnl_col, min_trades=30)

    if n_simulations < 10:
        raise ValueError(f"n_simulations must be >= 10, got {n_simulations}.")
    if n_rows_sample < 5:
        raise ValueError(f"n_rows_sample must be >= 5, got {n_rows_sample}.")
    if n_rows_sample > len(trades):
        raise ValueError(
            f"n_rows_sample ({n_rows_sample}) exceeds number of trades ({len(trades)}). "
            "Use a smaller sample or collect more data."
        )
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    # Deterministic seeding without polluting global state
    rng = np.random.default_rng(random_state)

    pnl_values = trades[pnl_col].dropna().to_numpy(dtype=np.float64)
    n_trades = len(pnl_values)

    equity_curves: List[np.ndarray] = []
    final_equities = np.empty(n_simulations, dtype=np.float64)

    logger.info(
        "MC simulation: n_simulations=%d, sample_size=%d, n_trades=%d",
        n_simulations, n_rows_sample, n_trades,
    )

    for i in tqdm(range(n_simulations), desc="Monte Carlo", disable=not show_progress):
        indices = rng.integers(0, n_trades, size=n_rows_sample)
        sample_pnl = pnl_values[indices]
        equity_curves.append(np.cumsum(sample_pnl))
        final_equities[i] = sample_pnl.sum()

    mean_eq = float(np.mean(final_equities))
    std_eq = float(np.std(final_equities, ddof=1))
    min_eq = float(np.min(final_equities))
    max_eq = float(np.max(final_equities))

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(final_equities, alpha / 2 * 100))
    ci_upper = float(np.percentile(final_equities, (1.0 - alpha / 2) * 100))
    win_rate = float(np.mean(final_equities > 0))

    logger.info(
        "MC complete: μ=%.2f σ=%.2f [%.2f, %.2f] win_rate=%.1f%%",
        mean_eq, std_eq, ci_lower, ci_upper, win_rate * 100,
    )

    return MonteCarloResult(
        equity_curves=equity_curves,
        final_equities=final_equities,
        mean_equity=mean_eq,
        std_equity=std_eq,
        min_equity=min_eq,
        max_equity=max_eq,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        win_rate=win_rate,
        confidence_level=confidence_level,
        n_simulations=n_simulations,
        sample_size=n_rows_sample,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_montecarlo_paths(
    result: MonteCarloResult,
    title: str = "Monte Carlo Equity Curves",
    figsize: Tuple[int, int] = (12, 6),
    alpha: float = 0.2,
    show: bool = True,
    max_paths: int = 500,
) -> Optional[plt.Figure]:
    """
    Plot simulated equity curve paths.

    Parameters
    ----------
    result : MonteCarloResult
        Output from ``get_montecarlo_analysis()``.
    title : str
        Plot title.
    figsize : Tuple[int, int], default=(12, 6)
        Figure dimensions.
    alpha : float, default=0.2
        Path transparency.
    show : bool, default=True
        Call ``plt.show()`` if True; return figure if False.
    max_paths : int, default=500
        Cap the number of paths drawn to avoid plot saturation.

    Returns
    -------
    plt.Figure | None
    """
    curves = result.equity_curves
    if not curves:
        raise ValueError("MonteCarloResult contains no equity curves.")

    fig, ax = plt.subplots(figsize=figsize)
    step = max(1, len(curves) // max_paths)

    for curve in curves[::step]:
        ax.plot(curve, alpha=alpha, lw=0.8, color="steelblue")

    # Mean path
    min_len = min(len(c) for c in curves)
    mean_curve = np.mean([c[:min_len] for c in curves], axis=0)
    ax.plot(mean_curve, color="crimson", lw=2.0, label="Mean path", zorder=10)

    # CI band
    p_low = np.percentile([c[:min_len] for c in curves],
                          (1 - result.confidence_level) / 2 * 100, axis=0)
    p_high = np.percentile([c[:min_len] for c in curves],
                           (1 - (1 - result.confidence_level) / 2) * 100, axis=0)
    ax.fill_between(range(min_len), p_low, p_high, alpha=0.15, color="orange",
                    label=f"{result.confidence_level:.0%} CI band")

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_montecarlo_distribution(
    result: MonteCarloResult,
    title: str = "Final Equity Distribution",
    figsize: Tuple[int, int] = (10, 6),
    n_bins: int = 50,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Histogram of terminal equity values with confidence interval markers.

    Parameters
    ----------
    result : MonteCarloResult
        Output from ``get_montecarlo_analysis()``.
    title : str
        Plot title.
    figsize : Tuple[int, int], default=(10, 6)
        Figure dimensions.
    n_bins : int, default=50
        Number of histogram bins.
    show : bool, default=True
        Call ``plt.show()`` if True; return figure if False.

    Returns
    -------
    plt.Figure | None
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(result.final_equities, bins=n_bins, color="steelblue",
            alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(result.mean_equity, color="crimson", lw=2.0, label="Mean")
    ax.axvline(result.ci_lower, color="orange", lw=1.8, linestyle="--",
               label=f"{result.confidence_level:.0%} CI lower")
    ax.axvline(result.ci_upper, color="orange", lw=1.8, linestyle="--",
               label=f"{result.confidence_level:.0%} CI upper")
    ax.axvline(0, color="black", lw=0.8, linestyle=":")

    summary = result.summary()
    info = (
        f"Win rate: {summary['win_rate']:.1%}\n"
        f"μ: {summary['mean_equity']:.2f}\n"
        f"σ: {summary['std_equity']:.2f}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Terminal equity")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig
