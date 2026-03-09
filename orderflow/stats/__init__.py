"""
OrderFlow Statistics Module
===========================

Institutional-grade statistical engine for systematic trading research.

Submodules
----------
stats           Core descriptive stats, risk metrics, time-series diagnostics.
returns         Return series construction, equity curves, drawdown analysis.
montecarlo      Non-parametric bootstrap Monte Carlo for strategy robustness.
markov          Markov chain & HMM regime predictors (no lookahead).
markov_utilities  Feature engineering, HMM model selection, data loading.
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
    underwater_duration,
)

# ── Monte Carlo simulation ───────────────────────────────────────────────────
from .montecarlo import (
    MonteCarloResult,
    get_montecarlo_analysis,
    plot_montecarlo_paths,
    plot_montecarlo_distribution,
)

# ── Markov & HMM regime detection ───────────────────────────────────────────
from .markov import (
    MarkovChainPredictor,
    AdaptiveMarkovChainPredictor,
    MultiFeatureHMM,
    get_states_from_ohlc,
    predict_bar_state,
)

# ── Feature engineering & utilities ─────────────────────────────────────────
from .markov_utilities import (
    threshold_prices_states,
    adaptive_threshold_prices_states,
    simulate_market_data,
    compute_df_features,
    select_best_hmm_model,
    concat_sc_bar_data,
    plot_distribution_of_float_series,
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
    "underwater_duration",
    # montecarlo.py
    "MonteCarloResult",
    "get_montecarlo_analysis",
    "plot_montecarlo_paths",
    "plot_montecarlo_distribution",
    # markov.py
    "MarkovChainPredictor",
    "AdaptiveMarkovChainPredictor",
    "MultiFeatureHMM",
    "get_states_from_ohlc",
    "predict_bar_state",
    # markov_utilities.py
    "threshold_prices_states",
    "adaptive_threshold_prices_states",
    "simulate_market_data",
    "compute_df_features",
    "select_best_hmm_model",
    "concat_sc_bar_data",
    "plot_distribution_of_float_series",
]
