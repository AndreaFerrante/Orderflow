"""
OrderFlow Statistics Module

Provides production-ready statistical analysis tools for market data:
- Markov Chain prediction for state transitions (UP/DOWN/FLAT)
- Hidden Markov Models for multi-feature regime analysis
- Monte Carlo simulation for equity curve analysis
- Technical utilities: HMM selection, state thresholding, feature engineering
"""

# Markov Prediction Classes
from .markov import (
    MarkovChainPredictor,
    AdaptiveMarkovChainPredictor,
    MultiFeatureHMM,
)

# Markov Utilities: State generation, HMM tools, feature engineering
from .markov_utilities import (
    threshold_prices_states,
    adaptive_threshold_prices_states,
    simulate_market_data,
    compute_df_features,
    select_best_hmm_model,
    concat_sc_bar_data,
    plot_distribution_of_float_series,
)

# Monte Carlo Analysis
from .montecarlo import (
    get_montecarlo_analysis,
)

# Statistical Functions (Polars-based)
from .stats import (
    is_skewed,
    get_kurtosis,
)

__all__ = [
    # Markov classes
    "MarkovChainPredictor",
    "AdaptiveMarkovChainPredictor",
    "MultiFeatureHMM",
    # Utilities
    "threshold_prices_states",
    "adaptive_threshold_prices_states",
    "simulate_market_data",
    "compute_df_features",
    "select_best_hmm_model",
    "concat_sc_bar_data",
    "plot_distribution_of_float_series",
    # Monte Carlo
    "get_montecarlo_analysis",
    # Statistics
    "is_skewed",
    "get_kurtosis",
]

__version__ = "0.1.0"
