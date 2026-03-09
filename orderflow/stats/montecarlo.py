"""
Monte Carlo simulation and analysis for trading equity curves.

Provides tools to assess strategy robustness through statistical resampling
of historical trades and equity curve visualization.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def validate_trades_df(trades: pd.DataFrame, entry_col_name: str) -> None:
    """
    Validate that trades DataFrame has required structure.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades dataframe.
    entry_col_name : str
        Column name for trade gains/losses.

    Raises
    ------
    TypeError
        If trades is not a DataFrame.
    ValueError
        If required columns are missing.
    """
    if not isinstance(trades, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(trades).__name__}.")

    required_cols = {"Datetime", entry_col_name}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(trades) < 10:
        logger.warning(f"Only {len(trades)} trades; consider >= 30 for robust MC.")


def get_montecarlo_analysis(
    trades: pd.DataFrame,
    n_rows_sample: int,
    n_simulations: int = 100,
    entry_col_name: str = "Entry_Gains",
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[List[pd.Series], pd.DataFrame, dict]:
    """
    Run Monte Carlo simulation on historical trades with equity curves.

    Performs random sampling with replacement (bootstrap) to generate multiple
    equity curve scenarios. Useful for assessing strategy robustness and
    computing worst/best case statistics.

    Parameters
    ----------
    trades : pd.DataFrame
        Historical trades with columns: 'Datetime', entry_col_name.
        Must have at least 10 trades.
    n_rows_sample : int
        Number of trades to sample in each iteration (sample size per sim).
    n_simulations : int, default=100
        Number of Monte Carlo simulations to run.
    entry_col_name : str, default="Entry_Gains"
        Column name containing trade P&L/gains.
    confidence_level : float, default=0.95
        For computing confidence intervals on final equity (e.g., 0.95 → 95% CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    equity_patterns : List[pd.Series]
        Cumulative equity curve for each simulation.
    ec_summary : pd.DataFrame
        Summary statistics with columns: EcPattern (final equity),
        bucket (binned ranges), CumPatterns (cumulative by bucket).
    stats_dict : dict
        Summary stats: mean_equity, min_equity, max_equity, std_equity,
        ci_lower, ci_upper, win_rate.

    Raises
    ------
    TypeError
        If trades not a DataFrame.
    ValueError
        If required columns missing or insufficient data.

    Examples
    --------
    >>> trades_df = pd.DataFrame({
    ...     'Datetime': pd.date_range('2023-01-01', periods=50),
    ...     'Entry_Gains': np.random.randn(50) * 10,
    ... })
    >>> equity_patterns, summary, stats = get_montecarlo_analysis(
    ...     trades_df, n_rows_sample=30, n_simulations=500
    ... )
    >>> print(f"Mean final equity: ${stats['mean_equity']:.2f}")
    """
    # Validation
    validate_trades_df(trades, entry_col_name)

    if n_simulations < 10:
        raise ValueError(f"n_simulations must be >= 10, got {n_simulations}.")
    if n_rows_sample < 5:
        raise ValueError(f"n_rows_sample must be >= 5, got {n_rows_sample}.")
    if n_rows_sample > len(trades):
        raise ValueError(
            f"n_rows_sample ({n_rows_sample}) > trades length ({len(trades)})."
        )

    if random_state is not None:
        np.random.seed(random_state)

    equity_patterns: List[pd.Series] = []
    ec_results: List[float] = []

    logger.info(
        f"Running MC simulation: {n_simulations} sims, "
        f"sample_size={n_rows_sample}, n_trades={len(trades)}"
    )

    # Run simulations
    for _ in tqdm(range(n_simulations), desc="Monte Carlo"):
        # Bootstrap sample with replacement
        sample = trades.sample(n=n_rows_sample, replace=True).sort_values(
            "Datetime", ascending=True
        )

        # Cumulative gains
        cumsum = sample[entry_col_name].cumsum()
        equity_patterns.append(cumsum)

        # Final equity for this simulation
        final_equity = sample[entry_col_name].sum()
        ec_results.append(final_equity)

    # Compute summary statistics
    ec_array = np.array(ec_results)
    mean_eq = float(np.mean(ec_array))
    std_eq = float(np.std(ec_array))
    min_eq = float(np.min(ec_array))
    max_eq = float(np.max(ec_array))

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(ec_array, alpha / 2 * 100))
    ci_upper = float(np.percentile(ec_array, (1 - alpha / 2) * 100))

    # Win rate (fraction of sims with positive equity)
    win_rate = float(np.sum(ec_array > 0) / len(ec_array))

    # Summary by bucket
    ec_df = pd.DataFrame({"EcPattern": ec_results})

    # Create bins for distribution
    n_bins = max(10, int(np.sqrt(len(ec_results))))
    bins = np.linspace(min_eq - 1, max_eq + 1, n_bins)
    labels = [f"[{bins[i]:.0f}, {bins[i+1]:.0f})" for i in range(len(bins) - 1)]

    ec_df["bucket"] = pd.cut(ec_df["EcPattern"], bins=bins, labels=labels)
    ec_summary = (
        ec_df.groupby("bucket", observed=True)["EcPattern"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n_sims", "sum": "total_equity"})
    )
    ec_summary["CumPatterns"] = ec_summary["total_equity"].cumsum()
    ec_summary = ec_summary.reset_index()

    stats_dict = {
        "mean_equity": mean_eq,
        "std_equity": std_eq,
        "min_equity": min_eq,
        "max_equity": max_eq,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "win_rate": win_rate,
        "n_simulations": n_simulations,
        "sample_size": n_rows_sample,
    }

    logger.info(
        f"MC complete: μ={mean_eq:.2f}, σ={std_eq:.2f}, "
        f"[{ci_lower:.2f}, {ci_upper:.2f}], win_rate={win_rate:.1%}"
    )

    return equity_patterns, ec_summary, stats_dict


def plot_montecarlo_paths(
    equity_patterns: List[pd.Series],
    title: str = "Monte Carlo Equity Curves",
    figsize: Tuple[int, int] = (12, 6),
    alpha: float = 0.3,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot all Monte Carlo equity curve paths.

    Parameters
    ----------
    equity_patterns : List[pd.Series]
        Cumulative equity curves from get_montecarlo_analysis().
    title : str, default="Monte Carlo Equity Curves"
        Plot title.
    figsize : Tuple[int, int], default=(12, 6)
        Figure size.
    alpha : float, default=0.3
        Line transparency (0-1).
    show : bool, default=True
        Whether to call plt.show().

    Returns
    -------
    plt.Figure or None
        Matplotlib figure object if show=False, else None.
    """
    if not equity_patterns:
        raise ValueError("Empty equity_patterns list.")

    fig, ax = plt.subplots(figsize=figsize)

    for pattern in equity_patterns:
        ax.plot(pattern.values, alpha=alpha, lw=1, color="steelblue")

    # Add mean line
    mean_curve = np.mean([p.values for p in equity_patterns], axis=0)
    ax.plot(mean_curve, color="red", lw=2.5, label="Mean", zorder=10)

    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Cumulative Gains")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()
        return None
    return fig


def plot_montecarlo_distribution(
    stats_dict: dict,
    ec_summary: pd.DataFrame,
    title: str = "Final Equity Distribution",
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot histogram of final equities with confidence intervals.

    Parameters
    ----------
    stats_dict : dict
        Statistics from get_montecarlo_analysis().
    ec_summary : pd.DataFrame
        Summary from get_montecarlo_analysis().
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    show : bool, default=True
        Whether to call plt.show().

    Returns
    -------
    plt.Figure or None
        Matplotlib figure object if show=False, else None.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot cumulative
    ax.bar(
        range(len(ec_summary)),
        ec_summary["CumPatterns"].values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )

    # Mark CI
    ax.axvline(stats_dict["mean_equity"], color="red", lw=2, label="Mean")
    ax.axvline(stats_dict["ci_lower"], color="orange", lw=2, linestyle="--", label="95% CI")
    ax.axvline(stats_dict["ci_upper"], color="orange", lw=2, linestyle="--")

    ax.set_xlabel("Final Equity Bucket")
    ax.set_ylabel("Cumulative Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()
        return None
    return fig


def compute_montecarlo_metrics(
    stats_dict: dict, initial_capital: float = 1.0
) -> dict:
    """
    Compute additional metrics from MC simulation results.

    Parameters
    ----------
    stats_dict : dict
        From get_montecarlo_analysis().
    initial_capital : float, default=1.0
        Starting capital for return calculations.

    Returns
    -------
    dict
        Extended metrics: total_return, sharpe_ratio, sortino_ratio, etc.
    """
    mean_eq = stats_dict["mean_equity"]
    std_eq = stats_dict["std_equity"]

    total_return = mean_eq / initial_capital if initial_capital > 0 else 0
    sharpe = (mean_eq / std_eq) if std_eq > 0 else 0
    sortino = mean_eq / std_eq if std_eq > 0 else 0  # Simplified

    return {
        "total_return": total_return,
        "return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": (
            1 + stats_dict["mean_equity"] / abs(stats_dict["min_equity"])
            if stats_dict["min_equity"] < 0
            else float("inf")
        ),
    }

