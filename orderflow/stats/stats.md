# OrderFlow Statistics Module

> Institutional-grade statistical engine for systematic trading research.

---

## Design Philosophy

This module was designed with the mindset of a quantitative trading operation
where **every statistical statement must be defensible** and **every number
must be reproducible**.

### Core Principles

| Principle | How it is enforced |
|---|---|
| **No lookahead bias** | All rolling/expanding operations are strictly causal. Window `[t-w, t]` never includes future bars. |
| **Numerical stability** | Log operations guarded by `EPS = 1e-14`; compensated algorithms for moments; double precision throughout. |
| **No silent failures** | Every function raises `ValueError` on degenerate input (zero variance, insufficient data) instead of returning NaN. |
| **Reproducibility** | All stochastic functions accept `random_state` seeds via NumPy's `default_rng` — no global state mutation. |
| **Correctness over cleverness** | Bias-corrected estimators (n−1 for variance, Fisher for kurtosis). Standard formulas from the literature. |
| **Production readiness** | Structured logging, type hints on all signatures, explicit validation at entry points. |

### Anti-Overfitting Safeguards

- **Multiple-testing correction** (`holm_bonferroni`) to prevent p-hacking.
- **Combined stationarity tests** (ADF + KPSS together) to avoid false confidence from a single test.
- **Marchenko–Pastur threshold** in eigenvalue analysis to distinguish signal from noise.
- **Bootstrap Monte Carlo** for strategy robustness — tests whether performance survives resampling.
- **Structural break detection** (CUSUM) to flag regime changes that invalidate backtest assumptions.

---

## Module Architecture

```
stats/
├── __init__.py            # Public API — all exports
├── _validators.py         # Shared input validation & type coercion
├── stats.py               # Core metrics: moments, Sharpe, Sortino, VaR, Hurst
├── returns.py             # Return conversions, equity curves, drawdowns
├── hypothesis.py          # Statistical tests: ADF, KPSS, Jarque-Bera, Ljung-Box, CUSUM
├── correlation.py         # Correlation: rolling, rank, stability, eigenvalues
├── montecarlo.py          # Bootstrap Monte Carlo simulation
├── markov.py              # Markov chain & HMM regime predictors
├── markov_utilities.py    # Feature engineering, HMM model selection, data I/O
└── stats.md               # This documentation
```

### Module Dependency Graph

```
_validators.py  ← used by all modules (shared validation)
      ↑
stats.py        ← pure NumPy, no external statistical deps
returns.py      ← pure NumPy + pandas
hypothesis.py   ← scipy.stats + statsmodels
correlation.py  ← scipy.stats + pandas
montecarlo.py   ← pure NumPy + tqdm
markov.py       ← hmmlearn + markov_utilities
markov_utilities.py ← hmmlearn + pandas + matplotlib
```

---

## Installation

The stats module requires these dependencies (already in `pyproject.toml`):

```bash
pip install numpy pandas scipy statsmodels hmmlearn tqdm matplotlib
```

Optional for enhanced performance:
```bash
pip install polars  # Alternative to pandas for stats.py input
```

---

## Quick Start

```python
import numpy as np
from orderflow.stats import (
    sharpe_ratio, sortino_ratio, max_drawdown, describe,
    to_log_returns, equity_curve,
    is_stationary, jarque_bera_test,
    rolling_correlation,
    get_montecarlo_analysis,
)

# Generate sample returns
rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, 500)

# ── Risk metrics ──
print(f"Sharpe:   {sharpe_ratio(returns):.2f}")
print(f"Sortino:  {sortino_ratio(returns):.2f}")
print(f"Max DD:   {max_drawdown(returns):.2%}")
print(f"Summary:  {describe(returns)}")

# ── Stationarity check ──
stationary, explanation = is_stationary(returns)
print(f"Stationary: {stationary} — {explanation}")

# ── Normality check ──
jb = jarque_bera_test(returns)
print(jb)  # TestResult: Jarque-Bera: stat=..., p=..., α=0.05 → ...
```

---

## Module Reference

### `stats.py` — Core Metrics

#### Distribution Analysis

| Function | Description |
|---|---|
| `describe(series, percentiles)` | Full moment summary (mean, std, skew, kurt, percentiles) |
| `is_skewed(series, threshold)` | Boolean skewness test (adjusted Fisher–Pearson) |
| `get_kurtosis(series)` | Bias-corrected excess kurtosis (Fisher definition) |

#### Performance Ratios

| Function | Description |
|---|---|
| `sharpe_ratio(returns, risk_free_rate, periods_per_year)` | Annualised Sharpe ratio |
| `sortino_ratio(returns, risk_free_rate, periods_per_year, mar)` | Annualised Sortino (downside-only risk) |
| `calmar_ratio(returns, periods_per_year)` | CAGR / max drawdown |
| `information_ratio(returns, benchmark_returns, periods_per_year)` | Active return / tracking error |
| `omega_ratio(returns, threshold)` | Probability-weighted gain/loss (captures full distribution) |
| `tail_ratio(returns, percentile)` | Right-tail vs left-tail magnitude |
| `profit_factor(returns)` | Gross profit / gross loss |
| `gain_to_pain_ratio(returns)` | Net return / sum of absolute losses (Schwager) |

#### Risk Measures

| Function | Description |
|---|---|
| `max_drawdown(returns)` | Maximum peak-to-trough drawdown |
| `var_historical(returns, confidence_level)` | Historical Value-at-Risk |
| `cvar_historical(returns, confidence_level)` | Conditional VaR (Expected Shortfall) |

#### Time-Series Diagnostics

| Function | Description |
|---|---|
| `rolling_sharpe(returns, window, periods_per_year)` | Rolling Sharpe (causal, no lookahead) |
| `autocorrelation(series, max_lag)` | Sample ACF at lags 1…max_lag |
| `hurst_exponent(series)` | R/S Hurst exponent (mean-reversion vs trend) |

---

### `returns.py` — Return Series

| Function | Description |
|---|---|
| `to_log_returns(prices)` | Price → log returns `ln(P_t / P_{t-1})` |
| `to_arithmetic_returns(prices)` | Price → simple returns `(P_t - P_{t-1}) / P_{t-1}` |
| `log_to_arithmetic(log_returns)` | `exp(r) - 1` |
| `arithmetic_to_log(arith_returns)` | `ln(1 + r)` |
| `annualise_return(mean_return, periods)` | Compound annualisation |
| `annualise_volatility(vol, periods)` | Square-root-of-time annualisation |
| `equity_curve(returns, initial_capital)` | Cumulative equity from returns |
| `drawdown_series(returns)` | Full drawdown time-series |
| `rolling_volatility(returns, window)` | Rolling annualised vol (causal) |
| `underwater_duration(returns)` | Consecutive bars below prior peak |

---

### `hypothesis.py` — Statistical Tests

All tests return a `TestResult` dataclass with `test_name`, `statistic`,
`p_value`, `reject_null`, `alpha`, and `detail`.

| Function | H₀ | Use Case |
|---|---|---|
| `adf_test(series)` | Series has unit root | Check if returns are stationary |
| `kpss_test(series)` | Series is stationary | Complement to ADF |
| `is_stationary(series)` | — | Combined ADF + KPSS verdict |
| `jarque_bera_test(series)` | Data is normal | Check normality assumption |
| `ljung_box_test(series)` | Data is white noise | Detect serial correlation |
| `cusum_test(series)` | No structural break | Detect regime changes |
| `holm_bonferroni(p_values)` | — | Multiple-testing correction (FWER) |

#### Stationarity Decision Matrix

```
┌────────────┬──────────────┬─────────────────────────────┐
│ ADF reject │ KPSS reject  │ Conclusion                  │
├────────────┼──────────────┼─────────────────────────────┤
│ Yes        │ No           │ Stationary ✓                │
│ No         │ Yes          │ Non-stationary (unit root)  │
│ Yes        │ Yes          │ Trend-stationary            │
│ No         │ No           │ Inconclusive                │
└────────────┴──────────────┴─────────────────────────────┘
```

---

### `correlation.py` — Correlation Analysis

| Function | Description |
|---|---|
| `rolling_correlation(x, y, window)` | Rolling Pearson (causal) |
| `rank_correlation(x, y, method)` | Spearman or Kendall with p-value |
| `correlation_stability(x, y, n_splits)` | Segment-wise correlation variability |
| `correlation_eigenvalues(returns_matrix)` | Eigenvalue decomposition + Marchenko–Pastur |

---

### `montecarlo.py` — Bootstrap Simulation

```python
from orderflow.stats import get_montecarlo_analysis
import pandas as pd, numpy as np

trades = pd.DataFrame({"Entry_Gains": np.random.default_rng(0).normal(10, 50, 200)})
result = get_montecarlo_analysis(
    trades, n_rows_sample=100, n_simulations=5000, random_state=42
)
print(result.summary())
# Plot equity paths: plot_montecarlo_paths(result)
# Plot distribution: plot_montecarlo_distribution(result)
```

---

### `markov.py` — Regime Detection

| Class / Function | Description |
|---|---|
| `MarkovChainPredictor(order)` | Fixed-order MC with Laplace smoothing |
| `AdaptiveMarkovChainPredictor(max_order)` | Auto-selects best order via validation log-likelihood |
| `MultiFeatureHMM(model)` | Wrapper for hmmlearn Gaussian HMM |
| `get_states_from_ohlc(df, method)` | OHLC → UP/DOWN/FLAT states |
| `predict_bar_state(df, predictor)` | One-call next-bar prediction |

### `markov_utilities.py` — Feature Engineering

| Function | Description |
|---|---|
| `threshold_prices_states(prices, threshold)` | Fixed-threshold state classification |
| `adaptive_threshold_prices_states(prices, window)` | Volatility-scaled state classification |
| `compute_df_features(df)` | Engineer log_return, volatility, slope, log_volume |
| `select_best_hmm_model(data, n_states_range)` | BIC/AIC model selection |
| `simulate_market_data(num_steps, seed)` | Generate synthetic GBM data |

---

## Example Workflows

### 1. Pre-Trade Statistical Due Diligence

Before running any backtest, validate your data:

```python
from orderflow.stats import (
    is_stationary, jarque_bera_test, ljung_box_test, hurst_exponent,
    to_log_returns,
)

prices = ...  # your price series
log_ret = to_log_returns(prices)

# 1. Check stationarity of returns
stat, msg = is_stationary(log_ret)
print(msg)

# 2. Test for normality (spoiler: it will reject)
jb = jarque_bera_test(log_ret)
print(jb)

# 3. Check for serial correlation (alpha signal or data issue?)
lb = ljung_box_test(log_ret, max_lag=20)
print(lb)

# 4. Characterise trending vs mean-reverting behaviour
h = hurst_exponent(log_ret)
print(f"Hurst exponent: {h:.3f}")
```

### 2. Strategy Performance Report

```python
from orderflow.stats import (
    sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio,
    max_drawdown, profit_factor, gain_to_pain_ratio,
    var_historical, cvar_historical,
    equity_curve, drawdown_series, underwater_duration,
)

returns = ...  # strategy daily returns

report = {
    "sharpe":       sharpe_ratio(returns),
    "sortino":      sortino_ratio(returns),
    "calmar":       calmar_ratio(returns),
    "omega":        omega_ratio(returns),
    "profit_factor": profit_factor(returns),
    "gain_to_pain": gain_to_pain_ratio(returns),
    "max_dd":       max_drawdown(returns),
    "VaR_95":       var_historical(returns, 0.95),
    "CVaR_95":      cvar_historical(returns, 0.95),
}
print(report)
```

### 3. Multi-Strategy Comparison (with Multiple-Testing Correction)

```python
from orderflow.stats import sharpe_ratio, holm_bonferroni
from scipy import stats as sp_stats
import numpy as np

# Simulate 20 strategy return streams
rng = np.random.default_rng(42)
strategies = [rng.normal(0.0001 * i, 0.01, 500) for i in range(20)]

# Test each: H₀ = mean return ≤ 0
p_values = []
for ret in strategies:
    t_stat, p = sp_stats.ttest_1samp(ret, 0.0)
    p_values.append(p / 2 if t_stat > 0 else 1.0)  # one-sided

# Correct for multiple comparisons
results = holm_bonferroni(p_values, alpha=0.05)
for idx, adj_p, reject in results:
    if reject:
        print(f"Strategy {idx}: adj_p={adj_p:.4f} — SIGNIFICANT")
```

### 4. Correlation Regime Analysis

```python
from orderflow.stats import (
    rolling_correlation, rank_correlation,
    correlation_stability, correlation_eigenvalues,
)
import numpy as np

rng = np.random.default_rng(42)
asset_a = rng.normal(0, 0.01, 1000)
asset_b = 0.5 * asset_a + rng.normal(0, 0.008, 1000)

# Rolling correlation
rho = rolling_correlation(asset_a, asset_b, window=63)

# Rank correlation (robust to outliers)
spearman_r, p_val = rank_correlation(asset_a, asset_b, method="spearman")

# Stability over time
stability = correlation_stability(asset_a, asset_b, n_splits=4)
print(f"Correlation range: {stability['range_corr']:.3f}")

# Multi-asset eigenvalue analysis
returns_matrix = np.column_stack([
    rng.normal(0, 0.01, 1000) for _ in range(10)
])
eig = correlation_eigenvalues(returns_matrix)
print(f"Significant factors: {eig['n_significant']} (above MP threshold)")
```

---

## Best Practices

### Do

- ✅ Always check stationarity before fitting time-series models.
- ✅ Use `is_stationary()` (combined ADF + KPSS) rather than a single test.
- ✅ Apply `holm_bonferroni()` when testing multiple strategies simultaneously.
- ✅ Use `random_state` parameter for reproducible Monte Carlo runs.
- ✅ Use log returns for multi-period compounding; arithmetic returns for single-period P&L.
- ✅ Use `rank_correlation` for fat-tailed data instead of Pearson.
- ✅ Check `correlation_eigenvalues` to verify your factors are truly independent.

### Don't

- ❌ Never use a single stationarity test in isolation.
- ❌ Never report raw p-values from multiple tests without correction.
- ❌ Never assume financial returns are normally distributed.
- ❌ Never use Sharpe ratio alone — always complement with Sortino, Calmar, and Omega.
- ❌ Never backtest on the same data used for parameter selection.
- ❌ Never ignore the Hurst exponent — it tells you whether mean-reversion or trend-following is appropriate.
- ❌ Never use rolling statistics with `min_periods=1` — this creates unreliable early estimates.

---

## Common Pitfalls & Warnings

### 1. Survivorship Bias in Sharpe Ratio
The Sharpe ratio is upward-biased when computed on strategies that were selected
*because* they performed well.  Always use `holm_bonferroni` when comparing
multiple strategies.

### 2. Autocorrelated Returns Inflate Sharpe
If `ljung_box_test` rejects white noise, your Sharpe ratio may be overstated.
Consider using the Newey–West adjusted standard error or reducing
`periods_per_year` to account for effective degrees of freedom.

### 3. Non-Stationary Data Produces Spurious Correlations
Two independent random walks will appear correlated purely by accident.
Always check stationarity before computing correlations.

### 4. Drawdown Underestimation with Low Frequency Data
`max_drawdown` on weekly data will miss intra-week drawdowns.
Use the highest-frequency data available for accurate drawdown measurement.

### 5. Monte Carlo Assumes i.i.d. Trades
The bootstrap resamples trades with replacement, destroying temporal
structure.  This is correct for estimating the *distribution of outcomes*
but does not test regime-dependent strategies.

### 6. HMM State Labels Are Arbitrary
Hidden Markov Model states are identified by index (0, 1, 2…), not by
economic meaning.  States can permute between fits.  Always interpret
states by their emission distributions, not their labels.
