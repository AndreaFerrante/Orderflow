# Stats Module — Practical Guide

Institutional-grade statistical engine for systematic trading research.
Import everything from the top-level namespace:

```python
from orderflow.stats import sharpe_ratio, adf_test, get_montecarlo_analysis  # etc.
```

---

## File Map

| File | Purpose |
|---|---|
| `stats.py` | Core descriptive stats, risk metrics, time-series diagnostics |
| `returns.py` | Return series construction, equity curves, drawdown analysis |
| `hypothesis.py` | Statistical hypothesis tests (stationarity, normality, breaks) |
| `correlation.py` | Correlation analysis (rolling, rank, stability, eigenvalues) |
| `montecarlo.py` | Bootstrap Monte Carlo for strategy robustness |
| `markov.py` | Markov chain & HMM regime predictors |
| `markov_utilities.py` | Feature engineering, HMM model selection, data loading |
| `_validators.py` | Internal input validation (not user-facing) |

---

## 1. Core Stats (`stats.py`)

### Descriptive Summary

```python
from orderflow.stats import describe

summary = describe(returns_array)
# Returns dict: count, mean, std, skew, kurt, min, max, p5, p25, p50, p75, p95
```

### Skewness & Kurtosis

```python
from orderflow.stats import is_skewed, get_kurtosis

is_skewed(returns, threshold=0.5)   # True if |skew| > 0.5
get_kurtosis(returns)               # Excess kurtosis (0 = normal, >0 = fat tails)
```

### Risk-Adjusted Performance

```python
from orderflow.stats import sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio

# All accept per-period arithmetic returns (e.g. daily)
sharpe  = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
sortino = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
calmar  = calmar_ratio(returns, periods_per_year=252)

# Information ratio requires a benchmark
ir = information_ratio(strategy_returns, benchmark_returns, periods_per_year=252)
```

### Risk Measures

```python
from orderflow.stats import max_drawdown, var_historical, cvar_historical

mdd  = max_drawdown(returns)                         # e.g. -0.25 = -25%
var  = var_historical(returns, confidence_level=0.95) # 95% VaR (non-positive)
cvar = cvar_historical(returns, confidence_level=0.95)# Expected Shortfall
```

### Additional Ratios

```python
from orderflow.stats import omega_ratio, tail_ratio, profit_factor, gain_to_pain_ratio

omega_ratio(returns, threshold=0.0)  # >1 is good; captures full distribution
tail_ratio(returns, percentile=0.95) # >1 = fatter right tail (desirable)
profit_factor(returns)               # gross_profit / gross_loss; >2 is strong
gain_to_pain_ratio(returns)          # Schwager's alternative to Sharpe
```

### Time-Series Diagnostics

```python
from orderflow.stats import rolling_sharpe, autocorrelation, hurst_exponent

# Rolling Sharpe (no lookahead). First window-1 values are NaN.
r_sharpe = rolling_sharpe(returns, window=63, periods_per_year=252)

# Autocorrelation at lags 1..10
acf = autocorrelation(returns, max_lag=10)  # {1: 0.03, 2: -0.01, ...}

# Hurst exponent: <0.5 mean-reverting, ~0.5 random walk, >0.5 trending
H = hurst_exponent(price_series, min_window=10, n_windows=20)
```

---

## 2. Returns (`returns.py`)

### Build Return Series from Prices

```python
from orderflow.stats import to_log_returns, to_arithmetic_returns

log_ret   = to_log_returns(prices)        # ln(P_t / P_{t-1}), length n-1
arith_ret = to_arithmetic_returns(prices)  # (P_t - P_{t-1}) / P_{t-1}
```

**When to use which:**
- **Log returns** for compounding, volatility estimation, multi-period analysis.
- **Arithmetic returns** for single-period P&L attribution and cross-sectional comparisons.

### Convert Between Return Types

```python
from orderflow.stats import log_to_arithmetic, arithmetic_to_log

arith = log_to_arithmetic(log_returns)   # exp(r) - 1
log_r = arithmetic_to_log(arith_returns) # ln(1 + r)
```

### Annualisation

```python
from orderflow.stats import annualise_return, annualise_volatility

ann_ret = annualise_return(mean_daily_return, periods_per_year=252)
ann_vol = annualise_volatility(daily_std, periods_per_year=252)
```

### Equity Curve & Drawdowns

```python
from orderflow.stats import equity_curve, drawdown_series, rolling_volatility, underwater_duration

eq = equity_curve(returns, initial_capital=10_000)
dd = drawdown_series(returns)            # 0 = at peak, -0.10 = 10% below peak
rv = rolling_volatility(returns, window=21, periods_per_year=252)
uw = underwater_duration(returns)        # consecutive bars below prior peak
```

---

## 3. Hypothesis Testing (`hypothesis.py`)

All tests return a `TestResult` dataclass with `.test_name`, `.statistic`, `.p_value`, `.reject_null`, `.alpha`, `.detail`.

### Stationarity (use both together)

```python
from orderflow.stats import adf_test, kpss_test, is_stationary

# Individual tests
adf = adf_test(series, alpha=0.05)    # H0: unit root (non-stationary)
kps = kpss_test(series, alpha=0.05)   # H0: stationary (opposite of ADF)

# Combined verdict (recommended)
stationary, explanation = is_stationary(series, alpha=0.05)
# Returns (True/False, "Stationary (ADF rejects unit root, KPSS does not reject stationarity).")
```

**Interpretation matrix:**

| ADF rejects | KPSS rejects | Verdict |
|---|---|---|
| Yes | No | Stationary |
| No | Yes | Non-stationary |
| Yes | Yes | Trend-stationary (detrend it) |
| No | No | Inconclusive (need more data) |

### Normality

```python
from orderflow.stats import jarque_bera_test

jb = jarque_bera_test(returns, alpha=0.05)
# jb.reject_null = True → returns are NOT normal (expected for financial data)
# jb.detail contains skewness and kurtosis values
```

### Serial Correlation

```python
from orderflow.stats import ljung_box_test

lb = ljung_box_test(returns, max_lag=10, alpha=0.05)
# lb.reject_null = True → returns have significant autocorrelation (exploitable)
# lb.detail['per_lag_pvalues'] gives p-values at each lag
```

### Structural Breaks

```python
from orderflow.stats import cusum_test

cs = cusum_test(series, alpha=0.05)
# cs.reject_null = True → structural break detected
# cs.detail['break_index'] → approximate location of the break
```

### Multiple Testing Correction

```python
from orderflow.stats import holm_bonferroni

# When running multiple tests, correct for false discovery
p_values = [adf.p_value, jb.p_value, lb.p_value]
corrected = holm_bonferroni(p_values, alpha=0.05)
# Returns: [(original_idx, adjusted_p, reject_null), ...]
```

---

## 4. Correlation (`correlation.py`)

### Rolling Correlation

```python
from orderflow.stats import rolling_correlation

# Causal rolling Pearson correlation (no lookahead)
rc = rolling_correlation(returns_x, returns_y, window=63, min_periods=20)
# Returns np.ndarray; early positions are NaN
```

### Rank Correlation

```python
from orderflow.stats import rank_correlation

# Robust to outliers; captures monotonic (not just linear) dependence
corr, pval = rank_correlation(x, y, method="spearman")  # or "kendall"
```

### Stability Over Time

```python
from orderflow.stats import correlation_stability

result = correlation_stability(returns_x, returns_y, n_splits=4)
# result = {mean_corr, std_corr, min_corr, max_corr, range_corr, n_splits}
# Large std_corr or range_corr → unstable relationship (don't trust it)
```

### Factor Analysis (Eigenvalues)

```python
from orderflow.stats import correlation_eigenvalues

# Pass a (n_obs, n_assets) matrix
result = correlation_eigenvalues(returns_matrix)
# result['n_significant']          → number of real factors (above Marchenko-Pastur bound)
# result['eigenvalues']            → sorted descending
# result['explained_variance_ratio'] → fraction of variance per factor
# result['condition_number']       → large = multicollinearity
```

---

## 5. Monte Carlo (`montecarlo.py`)

### Run Simulation

```python
import pandas as pd
from orderflow.stats import get_montecarlo_analysis

# trades DataFrame must have a P&L column
trades = pd.DataFrame({"Entry_Gains": [50, -30, 20, -10, 80, ...]})

result = get_montecarlo_analysis(
    trades,
    n_rows_sample=100,        # trades per simulation
    n_simulations=1000,       # bootstrap replications (>=1000 for publication)
    pnl_col="Entry_Gains",    # column name with per-trade P&L
    confidence_level=0.95,
    random_state=42,           # reproducible
)

print(result.summary())
# {mean_equity, std_equity, min_equity, max_equity, ci_lower, ci_upper, win_rate, ...}
```

### Visualise

```python
from orderflow.stats import plot_montecarlo_paths, plot_montecarlo_distribution

plot_montecarlo_paths(result)         # equity curve spaghetti plot with mean + CI band
plot_montecarlo_distribution(result)  # final equity histogram with CI markers
```

Both accept `show=False` to return the `plt.Figure` instead of displaying.

---

## 6. Markov & HMM Regime Detection (`markov.py`)

### Fixed-Order Markov Chain

```python
from orderflow.stats import MarkovChainPredictor, threshold_prices_states

# Step 1: convert prices to UP/DOWN/FLAT states
states = threshold_prices_states(prices, threshold=0.01)

# Step 2: fit a Markov chain
mc = MarkovChainPredictor(order=2, smoothing_alpha=0.1)
mc.fit(states)

# Step 3: predict
dist = mc.predict_distribution(["UP", "DOWN"])  # {"UP": 0.4, "DOWN": 0.35, "FLAT": 0.25}
next_state = mc.predict_next_state(["UP", "DOWN"])  # "UP"
```

### Adaptive Markov Chain (auto order selection)

```python
from orderflow.stats import AdaptiveMarkovChainPredictor

amc = AdaptiveMarkovChainPredictor(max_order=5, smoothing_alpha=0.1)
amc.fit(states, validation_ratio=0.2)
# amc.best_order → automatically selected (e.g. 3)

dist = amc.predict_distribution(["UP", "DOWN", "FLAT"])
```

### OHLC Bar States + Prediction

```python
from orderflow.stats import get_states_from_ohlc, predict_bar_state

# DataFrame must have Open, High, Low, Close columns
states = get_states_from_ohlc(df, method="close")     # adaptive threshold on close-to-close
states = get_states_from_ohlc(df, method="hl_range")   # bar expansion/contraction
states = get_states_from_ohlc(df, method="oc_range")   # sign of (Close - Open)

# One-call prediction on a fitted predictor
predicted, prob_dist = predict_bar_state(df, predictor=mc, method="close")
```

### Hidden Markov Model (Multi-Feature)

```python
from orderflow.stats import MultiFeatureHMM, select_best_hmm_model, compute_df_features

# Step 1: engineer features from bar data (needs 'price' and 'volume' columns)
df_feat = compute_df_features(df, window_volatility=20, window_slope=5)
# Adds: log_return, volatility, slope, log_volume

# Step 2: select best number of hidden states via BIC
features = df_feat[["log_return", "volatility", "slope", "log_volume"]].values
best_model = select_best_hmm_model(
    features,
    n_states_range=[2, 3, 4, 5],
    criterion="bic",
    random_state=42,
)

# Step 3: wrap and use
hmm_predictor = MultiFeatureHMM(model=best_model)
hidden_states = hmm_predictor.predict_states(features)      # Viterbi decoding
state_probs   = hmm_predictor.predict_proba_states(features) # posterior probabilities
log_lik       = hmm_predictor.score(features)                # model fit quality
```

---

## 7. Markov Utilities (`markov_utilities.py`)

### State Generation

```python
from orderflow.stats import threshold_prices_states, adaptive_threshold_prices_states

# Fixed threshold (absolute price difference)
states = threshold_prices_states(prices, threshold=0.5)

# Adaptive threshold (volatility-scaled, no lookahead)
states = adaptive_threshold_prices_states(prices, window=20, z_score_threshold=0.5)
```

### SierraChart Data Loading

```python
from orderflow.stats import concat_sc_bar_data

# Load all .txt bar export files from a directory
df = concat_sc_bar_data("path/to/sc_exports/", file_extension="txt")
# Adds 'Instrument' column, sorts by Date+Time
```

### Synthetic Data & Plotting

```python
from orderflow.stats import simulate_market_data, plot_distribution_of_float_series

# Generate test data (GBM with drift)
df = simulate_market_data(num_steps=10_000, seed=123)
# Columns: price, volume

# Quick histogram of any float Series
plot_distribution_of_float_series(df["price"], bins=75, title="Price Distribution")
```

---

## Typical Workflow

```python
import numpy as np
import pandas as pd
from orderflow.stats import (
    to_log_returns, describe, sharpe_ratio, max_drawdown,
    is_stationary, jarque_bera_test, ljung_box_test,
    get_montecarlo_analysis, plot_montecarlo_paths,
    hurst_exponent, equity_curve,
)

# 1. Build returns
prices = pd.read_csv("prices.csv")["Close"].values
returns = to_log_returns(prices)

# 2. Descriptive stats
print(describe(returns))
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max DD: {max_drawdown(returns):.2%}")

# 3. Check assumptions
stationary, msg = is_stationary(returns)
print(f"Stationary: {stationary} — {msg}")

jb = jarque_bera_test(returns)
print(f"Normal: {not jb.reject_null}")

lb = ljung_box_test(returns)
print(f"White noise: {not lb.reject_null}")

H = hurst_exponent(returns)
print(f"Hurst: {H:.3f} ({'trending' if H > 0.5 else 'mean-reverting'})")

# 4. Monte Carlo robustness
trades = pd.DataFrame({"Entry_Gains": returns})
mc = get_montecarlo_analysis(trades, n_rows_sample=len(returns)//2, n_simulations=2000)
print(mc.summary())
plot_montecarlo_paths(mc)
```

---

## Input Conventions

- **Returns**: always 1-D arrays or `pd.Series` of per-period arithmetic returns unless noted.
- **Prices**: strictly positive, finite, 1-D. At least 2 values.
- **Minimum observations**: most functions require 2-20+ observations and raise `ValueError` if not met.
- **Polars support**: `stats.py` functions accept `pl.Series` directly (auto-converted to NumPy).
- **No lookahead**: every rolling/expanding operation is strictly causal.
