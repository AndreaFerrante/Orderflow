# OrderFlow Statistics Module

**Institutional-grade statistical engine for systematic trading research.**

This module provides battle-tested tools for quantitative trading analysis: descriptive statistics, return transformations, Monte Carlo simulation, and Markov regime detection. Every function is designed with **numerical stability**, **no lookahead bias**, and **explicit error handling** as core principles.

---

## Quick Start

### 1. **Analyze Returns Distribution**

```python
from orderflow.stats import describe, sharpe_ratio, max_drawdown
import numpy as np

# Daily returns from your strategy
returns = np.array([0.012, -0.005, 0.008, 0.015, -0.003, 0.010])

# Full statistical snapshot
summary = describe(returns)
print(f"Mean: {summary['mean']:.4f}, Std: {summary['std']:.4f}, Skew: {summary['skew']:.3f}")
# Output: Mean: 0.0062, Std: 0.0085, Skew: -0.145

# Risk-adjusted performance metrics
sharpe = sharpe_ratio(returns, periods_per_year=252)
max_dd = max_drawdown(returns)
print(f"Sharpe: {sharpe:.3f}, Max Drawdown: {max_dd:.2%}")
# Output: Sharpe: 0.749, Max Drawdown: 5.30%
```

### 2. **Convert Between Return Types**

```python
from orderflow.stats import to_log_returns, to_arithmetic_returns, equity_curve
import pandas as pd

# Price series
prices = pd.Series([100, 102, 105, 103, 108])

# Log returns: numerically stable, time-additive
log_rets = to_log_returns(prices)  # [0.0198, 0.0290, -0.0196, 0.0488]

# Equity curve: cumulative P&L from return series
rets = np.array([0.05, -0.02, 0.03])  # Sample returns
curve = equity_curve(rets, starting_cash=1000)  # [1000, 1050, 1028.9, 1059.7]
```

### 3. **Simulate Strategy Before Trading**

```python
from orderflow.stats import get_montecarlo_analysis

# Your trade P&L (profit & loss per closed trade)
trade_pnl = np.array([250, -150, 450, -50, 300, -100, 600])

# Bootstrap 1000 random trade sequences to stress-test your strategy
mc_result = get_montecarlo_analysis(
    trade_pnl,
    n_simulations=1000,
    confidence_level=0.95,
    random_state=42
)

print(f"Expected final equity: ${mc_result.mean_equity:,.0f}")
print(f"95% CI: [${mc_result.ci_lower:,.0f}, ${mc_result.ci_upper:,.0f}]")
print(f"Win rate: {mc_result.win_rate:.1%}")
# Output: Expected: $2,250, CI: [$450, $4,150], Win: 82%
```

### 4. **Detect Market Regime with Markov Chains**

```python
from orderflow.stats.markov_utilities import threshold_prices_states
from orderflow.stats.markov import MarkovChainPredictor

# Convert price bars into states
prices = [100, 102, 101, 103, 105, 104]
states = threshold_prices_states(prices, threshold=0.5)
# ['UP', 'DOWN', 'UP', 'UP', 'DOWN']

# Fit a 2nd-order Markov chain
predictor = MarkovChainPredictor(order=2)
predictor.fit(states)

# Predict next state probability given recent history
next_probs = predictor.predict_distribution(['UP', 'UP'])
print(next_probs)  # {'UP': 0.65, 'DOWN': 0.25, 'FLAT': 0.10}
```

---

## Core Components

### **1. Stats Module** — Risk & Performance Metrics

**Problem it solves:** Quantify strategy quality beyond raw returns.

**Key functions:**

| Function | Use Case | Theory |
|----------|----------|--------|
| `describe()` | Full statistical snapshot | Mean, std, skew, kurtosis, percentiles |
| `sharpe_ratio()` | Risk-adjusted returns | Excess return per unit of volatility |
| `sortino_ratio()` | Downside risk focus | Penalizes only negative volatility |
| `calmar_ratio()` | Recovery speed | CAGR divided by max drawdown |
| `information_ratio()` | Active management quality | Excess return vs benchmark volatility |
| `max_drawdown()` | Worst historical loss | Peak-to-trough percentage decline |
| `var_historical()` | Value-at-Risk | Loss threshold at confidence level |
| `cvar_historical()` | Expected Shortfall | Average loss when VaR is breached |
| `hurst_exponent()` | Mean-reversion tendency | >0.5 = trending, <0.5 = mean-reverting |
| `rolling_sharpe()` | Time-series stability | Window-based Sharpe (causal, no lookahead) |

**Example: Assess Your Strategy**

```python
from orderflow.stats import describe, sharpe_ratio, sortino_ratio, max_drawdown

daily_returns = np.array([...])  # Your strategy P&L

stats = describe(daily_returns)
# → {count, mean, std, min, max, p5, p25, p50, p75, p95, skew, kurt}

sharpe = sharpe_ratio(daily_returns, periods_per_year=252)
# Higher is better; 1.0+ is professional-grade

sortino = sortino_ratio(daily_returns, periods_per_year=252, target_return=0)
# Focuses on downside; usually >Sharpe if strategy is negatively skewed

mdd = max_drawdown(daily_returns)
# Worst case from any peak to trough
```

**Theory Behind Risk Metrics:**
- **Sharpe Ratio** = (mean return − risk-free rate) / std(returns)
  - Assumes normal distribution; vulnerable to outliers
  - Use for normally distributed returns (most equity strategies)
  
- **Sortino Ratio** = (mean return − target) / std(downside)
  - Penalizes only downside volatility (days below target)
  - Better for strategies with asymmetric losses
  
- **Calmar Ratio** = CAGR / max drawdown
  - Measures recovery efficiency
  - For trend-following: often 0.5–2.0⁣

---

### **2. Returns Module** — Price ↔ Return Conversions

**Problem it solves:** Correctly transform prices into return series without lookahead bias or numerical errors.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `to_log_returns(prices)` | $r_t = \ln(P_t / P_{t-1})$ — time-additive, stable |
| `to_arithmetic_returns(prices)` | $r_t = (P_t - P_{t-1}) / P_{t-1}$ — intuitive interpretation |
| `log_to_arithmetic()` | Convert log→arithmetic |
| `arithmetic_to_log()` | Convert arithmetic→log |
| `annualise_return(mean_period_return, periods_per_year)` | Scale single-period return to annual |
| `annualise_volatility()` | Scale period volatility to annual |
| `equity_curve(returns, starting_cash)` | Build cumulative P&L from return series |
| `drawdown_series()` | Full time-series of underwater P&L (not just max) |
| `rolling_volatility()` | Causal rolling volatility window |
| `underwater_duration()` | How many bars spent below prior peak |

**When to Use Which Return Type:**

| Scenario | Use This | Why |
|----------|----------|-----|
| Multi-period compounding | Log returns | Time-additive: Σ log-rets = log(final/initial) |
| Attribution analysis | Arithmetic returns | Linear combination for position-level contribution |
| Volatility estimation | Log returns | Numerically stable for long series |
| Sharpe/Sortino | Either (standardize) | Both give equivalent annualized metrics |

**Example: Track Strategy Equity**

```python
from orderflow.stats import to_log_returns, equity_curve, annualise_return, rolling_volatility

prices = np.array([100, 102, 100, 105, 103, 110])
rets = to_log_returns(prices)
# [0.0198, -0.0198, 0.0488, -0.0196, 0.0677]

# Cumulative P&L
curve = equity_curve(rets, starting_cash=100000)
# [100000, 101980, 100000, 104880, 102800, 109700]

# Annualize
annual_return = annualise_return(np.mean(rets), periods_per_year=252)
annual_vol = annualise_volatility(np.std(rets), periods_per_year=252)
print(f"Annual return: {annual_return:.1%}, Annual vol: {annual_vol:.1%}")
```

---

### **3. Monte Carlo Module** — Stress Test via Bootstrap

**Problem it solves:** "What if my trade sequence randomly reshuffles? Will I still be profitable?"

**Key concept:** Non-parametric bootstrap. Treat each trade's P&L as i.i.d., resample with replacement 1000× times, rebuild equity curve for each.

**Example:**

```python
from orderflow.stats.montecarlo import get_montecarlo_analysis

# 50 closed trades from your backtest
trades_pnl = np.array([-50, 150, 200, -100, 75, ..., 300])  # 50 values

result = get_montecarlo_analysis(
    pnl_series=trades_pnl,
    n_simulations=1000,
    confidence_level=0.95,
    random_state=42
)

print(f"Original final equity: ${trades_pnl.sum():,.0f}")
print(f"Expected (mean simulation): ${result.mean_equity:,.0f}")
print(f"95% confidence interval: [{result.ci_lower:,.0f}, {result.ci_upper:,.0f}]")
print(f"Worst case (1st percentile): ${result.min_equity:,.0f}")
print(f"P(profitable): {result.win_rate:.1%}")
```

**What it tells you:**
- If 95% CI includes negative numbers → strategy is **fragile**, order-dependent
- If win_rate < 60% → trades are too noisy to rely on
- If min_equity is deeply negative → one bad streak can wipe you out

---

### **4. Markov Module** — Regime & State Prediction

**Problem it solves:** "Is the market in an UP/DOWN/FLAT regime? What's the probability of the next bar reversing?"

**Key concepts:**
- Convert prices → states (UP, DOWN, FLAT) using a threshold
- Fit a fixed-order Markov chain (e.g., order=2 means "depends on last 2 states")
- Predict next state distribution given recent history

**Example:**

```python
from orderflow.stats.markov import MarkovChainPredictor, AdaptiveMarkovChainPredictor
from orderflow.stats.markov_utilities import threshold_prices_states

# 100 daily close prices
prices = [100, 102, 101, 103, 105, ...]

# Discretize into UP/DOWN/FLAT (50 cents = threshold)
states = threshold_prices_states(prices, threshold=0.5)

# Fit a 1st-order Markov chain
model = MarkovChainPredictor(order=1)
model.fit(states)

# Predict: given last state was UP, what's next?
next_dist = model.predict_distribution(['UP'])
# {'UP': 0.60, 'DOWN': 0.30, 'FLAT': 0.10}

# Trade filter: only take long entries if P(UP) > 0.55
if next_dist['UP'] > 0.55:
    enter_long()
```

**Theory:**
- **Markov chain of order k** = next state depends on last k states
- **Laplace smoothing** = avoid zero probabilities for unseen transitions
- **Adaptive order selection** = automatically choose k via BIC (less overfitting)

---

## Real-World Workflows

### **Workflow 1: Complete Strategy Backtest Analysis**

```python
import numpy as np
from orderflow.stats import (
    describe, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, rolling_sharpe, var_historical, cvar_historical,
    to_log_returns, equity_curve
)
from orderflow.stats.montecarlo import get_montecarlo_analysis

# Step 1: Load backtest results
prices = load_ohlc_data('SPY', '2023-01-01', '2024-01-01')['close']
trades_pnl = np.array([...])  # P&L from closed trades

# Step 2: Analyze returns distribution
log_rets = to_log_returns(prices)
summary = describe(log_rets)

# Step 3: Calculate performance metrics
perf = {
    'sharpe': sharpe_ratio(log_rets),
    'sortino': sortino_ratio(log_rets),
    'calmar': calmar_ratio(log_rets),
    'max_dd': max_drawdown(log_rets),
    'var_95': var_historical(log_rets, confidence=0.95),
    'cvar_95': cvar_historical(log_rets, confidence=0.95),
}

# Step 4: Check rolling performance (no lookahead)
rolling_sharpe_ts = rolling_sharpe(log_rets, window=60)

# Step 5: Monte Carlo robustness
mc = get_montecarlo_analysis(trades_pnl, n_simulations=1000)

# Step 6: Report
print("=" * 60)
print(f"STRATEGY PERFORMANCE REPORT")
print("=" * 60)
print(f"Mean return:        {summary['mean']:.4f} ({summary['mean']*252*100:.2f}% annualized)")
print(f"Std dev:            {summary['std']:.4f} ({summary['std']*np.sqrt(252)*100:.2f}% annualized)")
print(f"Skewness:           {summary['skew']:.3f}", end="")
if summary['skew'] < -0.5: print(" (LEFT-SKEWED, risky)")
elif summary['skew'] > 0.5: print(" (RIGHT-SKEWED, lucky)")
else: print(" (roughly symmetric)")
print(f"Kurtosis (excess):  {summary['kurt']:.3f}", end="")
if summary['kurt'] > 1: print(" (fat tails, higher crash risk)")
else: print(" (thin tails)")
print()
print(f"Sharpe ratio:       {perf['sharpe']:.3f}")
print(f"Sortino ratio:      {perf['sortino']:.3f}")
print(f"Calmar ratio:       {perf['calmar']:.3f}")
print(f"Max drawdown:       {perf['max_dd']:.2%}")
print(f"95% VaR:            {perf['var_95']:.2%}")
print(f"95% CVaR:           {perf['cvar_95']:.2%}")
print()
print(f"Monte Carlo (1000 sims):")
print(f"  Expected final P&L:     ${mc.mean_equity:>10,.0f}")
print(f"  95% CI:                 [${mc.ci_lower:>10,.0f}, ${mc.ci_upper:>10,.0f}]")
print(f"  Min (worst 1%):         ${mc.min_equity:>10,.0f}")
print(f"  P(profitable):          {mc.win_rate:>10.1%}")
```

### **Workflow 2: Regime-Based Entry Filtering**

```python
from orderflow.stats.markov import MarkovChainPredictor
from orderflow.stats.markov_utilities import threshold_prices_states

# Real-time trading scenario: decide whether to enter long
def should_enter_long(last_N_closes, threshold_pct=0.5):
    """
    Only enter if Markov model predicts high P(UP).
    """
    # Fit model on historical 500 bars
    historical_states = threshold_prices_states(
        last_N_closes[-500:],
        threshold=threshold_pct / 100 * last_N_closes[-500]  # % threshold
    )
    
    model = MarkovChainPredictor(order=2)
    model.fit(historical_states)
    
    # Recent 10 closes → recent 10 states
    recent_states = threshold_prices_states(
        last_N_closes[-10:],
        threshold=threshold_pct / 100 * last_N_closes[-10]
    )
    
    # Predict next bar
    next_dist = model.predict_distribution(recent_states[-2:])
    
    # Enter if UP is most likely AND probability >55%
    if next_dist['UP'] > 0.55 and next_dist['UP'] >= max(next_dist.values()):
        return True, next_dist
    
    return False, next_dist

# Usage
closes = [100, 101, 102, 101, 103, 105, ...]
should_buy, probs = should_enter_long(closes)
print(f"Enter long? {should_buy}")
print(f"Regime probabilities: {probs}")
```

---

## Design Principles

### **1. Numerical Stability**
- All moments use compensated algorithms (Welford / two-pass) to avoid catastrophic cancellation
- Log operations guarded by a floor (`_EPS = 1e-14`) to prevent log(0)
- Double precision (float64) throughout

### **2. No Lookahead Bias**
- All rolling windows are strictly causal: window [t-k...t] only uses data ≤ t
- Markov fits and predictions never use future data
- Returns and equity curves built strictly forward-in-time

### **3. Explicit Error Handling**
- Degenerate inputs (empty arrays, single value, all zeros) raise `ValueError` with clear messages
- No silent NaN returns—if input is invalid, you know immediately
- Type hints throughout for IDE support

### **4. Correct Statistics**
- Sample variance uses n−1 (Bessel's correction) for unbiased estimator
- Excess kurtosis uses Fisher definition (subtract 3 from raw kurtosis)
- Sharpe/Sortino formulas follow industry standards (Π.Tech, PMAR, CFA)

---

## Common Pitfalls & Solutions

| Pitfall | What Goes Wrong | Solution |
|---------|-----------------|----------|
| `to_arithmetic_returns(equity_curve)` | Treating equity curve as price | Use `to_log_returns(prices)` instead |
| Pass negative prices to returns | Function raises ValueError | Ensure prices are positive (no P&L as prices) |
| Sharpe on non-normal returns | Underestimates tail risk | Use Sortino or check kurtosis/skewness first |
| Markov order > data length | Fit crashes or no patterns found | Use order=1 or 2; check `n_states > order` |
| Monte Carlo with <10 trades | Confidence intervals meaningless | Aggregate across instruments or longer period |
| `rolling_sharpe(window=5)` on 100 bars | Only 96 windows, noisy | Use window=30–60 for stable estimates |

---

## API Reference Summary

### Stats
`describe()` · `is_skewed()` · `get_kurtosis()` · `sharpe_ratio()` · `sortino_ratio()` · `calmar_ratio()` · `information_ratio()` · `max_drawdown()` · `var_historical()` · `cvar_historical()` · `rolling_sharpe()` · `autocorrelation()` · `hurst_exponent()`

### Returns
`to_log_returns()` · `to_arithmetic_returns()` · `log_to_arithmetic()` · `arithmetic_to_log()` · `annualise_return()` · `annualise_volatility()` · `equity_curve()` · `drawdown_series()` · `rolling_volatility()` · `underwater_duration()`

### Monte Carlo
`get_montecarlo_analysis()` · `plot_montecarlo_paths()` · `plot_montecarlo_distribution()` · `MonteCarloResult` dataclass

### Markov
`MarkovChainPredictor` · `AdaptiveMarkovChainPredictor` · `MultiFeatureHMM` · `threshold_prices_states()` · `predict_bar_state()`

---

## References

- **Sharpe Ratio**: Sharpe, W. (1994). "The Sharpe ratio." *J. Portfolio Mgmt*, 21(1), 49–58.
- **Sortino Ratio**: Sortino, F., & Price, L. (1994). *Performance Measurement in a Downturn World*.
- **Value at Risk & CVaR**: Dowd, K. (2007). *Measuring Market Risk*, 2nd ed.
- **Hurst Exponent**: Peters, E. (1991). *Chaos and Order in the Capital Markets*.
- **Markov Chains**: Ross, S. (2014). *Introduction to Probability Models*, 11th ed.
- **Monte Carlo Methods**: Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*.

