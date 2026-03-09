# OrderFlow Statistics Module

Professional-grade statistical analysis for tick-by-tick and compressed bar market data. Production-ready tools for regime detection, state prediction, and strategy robustness assessment.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Markov Chain Predictions](#markov-chain-predictions)
3. [Hidden Markov Models](#hidden-markov-models)
4. [Monte Carlo Analysis](#monte-carlo-analysis)
5. [Working with Compressed Bars](#working-with-compressed-bars)
6. [Best Practices](#best-practices)
7. [Complete Examples](#complete-examples)

---

## Module Overview

The `orderflow.stats` module provides cutting-edge statistical tools:

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **MarkovChainPredictor** | Fixed-order state prediction | Price sequences | UP/DOWN/FLAT probabilities |
| **AdaptiveMarkovChainPredictor** | Auto-order selection + validation | Price sequences | Best-order predictor |
| **MultiFeatureHMM** | Multi-dimensional regime detection | Features (return, vol, slope) | Hidden states + probabilities |
| **get_montecarlo_analysis** | Strategy robustness via bootstrap | Trade P&L | Equity curves + CI + metrics |

### Key Features

✓ **Production-Ready**: Full error handling, logging, type hints
✓ **Compressed Bar Support**: Volume/Range/Time bars, OHLC data
✓ **Tick-by-Tick**: Works directly on raw market data
✓ **Validated**: Automatic parameter validation, sensible defaults
✓ **Extensible**: Base classes for custom implementations

---

## Markov Chain Predictions

Predict the next market movement (UP/DOWN/FLAT) based on recent price history.

### Fixed-Order Chain (Simple)

Use when you know the optimal history depth:

```python
from orderflow.stats import MarkovChainPredictor, threshold_prices_states

# Historical prices
prices = [100.0, 101.2, 101.5, 100.8, 101.1, 102.0]

# Convert to states
states = threshold_prices_states(prices, threshold=1e-8)
# → ['UP', 'UP', 'DOWN', 'UP', 'UP']

# Create and fit predictor (order=2 means use last 2 states)
predictor = MarkovChainPredictor(order=2)
predictor.fit(states)

# Predict next state
recent = ['UP', 'UP']
next_state = predictor.predict_next_state(recent)
# → 'UP' or 'DOWN' or 'FLAT'

# Get probability distribution
prob_dist = predictor.predict_distribution(recent)
# → {'UP': 0.65, 'DOWN': 0.20, 'FLAT': 0.15}
```

### Adaptive Chain (Recommended)

Automatically selects the best order using validation:

```python
from orderflow.stats import AdaptiveMarkovChainPredictor, adaptive_threshold_prices_states

# Use adaptive thresholding (accounts for volatility)
states = adaptive_threshold_prices_states(prices, window=20)

# Auto-select order from 1 to 10
predictor = AdaptiveMarkovChainPredictor(max_order=10, smoothing_alpha=0.1)
predictor.fit(states, validation_ratio=0.2)  # 80/20 split

print(f"Selected order: {predictor.best_order}")
# → Selected order: 3

# Make predictions
prediction = predictor.predict_next_state(['UP', 'DOWN', 'UP'])
distribution = predictor.predict_distribution(['UP', 'DOWN', 'UP'])
```

### Production Considerations

| Setting | Recommendation | Rationale |
|---------|---|---|
| **order** | 2-5 | Too low = overcounting; too high = overfitting on sparse patterns |
| **smoothing_alpha** | 0.1-1.0 | Prevents zero probabilities; higher = more conservative |
| **validation_ratio** | 0.15-0.25 | Trade-off between fitting and validation; 0.2 is standard |
| **min_states** | 50+ | Need enough history for reliable estimation |

---

## Hidden Markov Models

Detect hidden market regimes using multi-dimensional features (returns, volatility, order flow).

### Basic HMM Workflow

```python
from orderflow.stats import (
    MultiFeatureHMM,
    select_best_hmm_model,
    compute_df_features,
)
import pandas as pd
import numpy as np

# Prepare market data (tick data or compressed bars)
df = pd.DataFrame({
    'price': [100.0 + i*0.05 + np.random.randn()*0.1 for i in range(1000)],
    'volume': np.random.uniform(1e4, 5e4, 1000),
})

# Engineer features: return, volatility, slope, log-volume
df_features = compute_df_features(
    df,
    window_volatility=20,
    window_slope=5
)

# Extract feature matrix
X = df_features[['return', 'volatility', 'slope', 'log_volume']].values

# Auto-select best HMM (test 2-5 hidden states)
best_model = select_best_hmm_model(
    data=X,
    n_states_range=[2, 3, 4, 5],
    covariance_type='full',
    criterion='bic',
    random_state=42
)

# Wrap in class
hmm = MultiFeatureHMM(model=best_model)
hmm.fit(X)  # Re-fit on full data

# Decode hidden states
states = hmm.predict_states(X)
# → [0, 1, 0, 2, 1, ...]

# Get posterior probabilities for regime confirmation
probs = hmm.predict_proba_states(X)
# → [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], ...]
```

### Interpreting Hidden States

HMM discovers 2-5 market regimes automatically:

```
State 0: Low volatility, positive drift    (calm uptrend)
State 1: High volatility, mean-reversion   (choppy)
State 2: High volatility, negative drift   (panic selloff)
```

Use posterior probabilities for **regime confirmation**:

```python
# Current regime strength
current_probs = probs[-1]  # Last observation
regime_strength = max(current_probs)

if regime_strength > 0.7:
    current_regime = np.argmax(current_probs)
    print(f"Strong regime {current_regime} ({regime_strength:.1%} confidence)")
else:
    print("Weak signal - regime change likely")
```

### HMM Production Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **n_components** | BIC-selected | 2-5 | More states = overfitting; fewer = oversimplification |
| **covariance_type** | 'full' | 'full', 'diag', 'tied' | 'full' is most flexible (uses more data) |
| **criterion** | 'bic' | 'bic', 'aic' | BIC is more conservative; AIC for smaller data |
| **window_volatility** | 20 | 10-40 | Rolling window for volatility; match your bar period |

---

## Monte Carlo Analysis

Assess trading strategy robustness through bootstrap resampling of historical trades.

### Basic Monte Carlo

```python
from orderflow.stats import get_montecarlo_analysis, plot_montecarlo_paths
import pandas as pd

# Historical trades from backtester
trades = pd.DataFrame({
    'Datetime': pd.date_range('2023-01-01', periods=100),
    'Entry_Gains': [5.2, -3.1, 8.5, -1.2, 10.0, ...],  # P&L per trade
})

# Run 500 simulations, sample 50 trades per iteration
equity_patterns, summary, stats = get_montecarlo_analysis(
    trades,
    n_rows_sample=50,
    n_simulations=500,
    entry_col_name='Entry_Gains',
    confidence_level=0.95,
    random_state=42
)

# Results
print(f"Mean final equity: ${stats['mean_equity']:.2f}")
print(f"95% CI: [${stats['ci_lower']:.2f}, ${stats['ci_upper']:.2f}]")
print(f"Win rate: {stats['win_rate']:.1%}")
print(f"Best case: ${stats['max_equity']:.2f}")
print(f"Worst case: ${stats['min_equity']:.2f}")

# Visualize equity paths
plot_montecarlo_paths(equity_patterns, show=True)
```

### Interpreting Results

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| **Mean equity** | Positive | Near-zero | Negative |
| **Confidence interval** | Above zero | Touches zero | Below zero |
| **Win rate** | >60% | 50-60% | <50% |
| **Max/Min ratio** | <-2 | -2 to -1 | >-1 (worst case) |
| **Distribution shape** | Unimodal, tight | Bimodal | Fat tails, highly skewed |

### Production Checklist

```python
# ✓ Minimum trades
if len(trades) < 30:
    print("WARNING: <30 trades. Results may be unstable.")

# ✓ Sample size vs. trades
if n_rows_sample > len(trades) * 0.5:
    print("WARNING: Sample size too large. Use <50% of total trades.")

# ✓ Diversification
if n_simulations < 100:
    print("WARNING: <100 sims. Use >= 500 for production.")

# ✓ Statistical significance
ci_width = stats['ci_upper'] - stats['ci_lower']
if ci_width > stats['mean_equity'] * 2:
    print("WARNING: Large CI relative to mean. Low signal clarity.")
```

### Advanced: Profit Factor & Sharpe Ratio

```python
from orderflow.stats import compute_montecarlo_metrics

metrics = compute_montecarlo_metrics(stats, initial_capital=10000)

print(f"Total return: {metrics['return_pct']:.2f}%")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")  # >1.0 is good
print(f"Profit factor: {metrics['profit_factor']:.2f}")  # >1.5 is solid
```

---

## Working with Compressed Bars

### Volume / Range / Time Bars

```python
from orderflow.compressor import compress_to_bar_once_range_met
from orderflow.stats import get_states_from_ohlc, predict_bar_state
import pandas as pd

# Load tick data
ticks = pd.read_csv('tbt/2023_06_29.txt', sep=';')

# Compress to range bars (4-point = 16 ticks × 0.25 tick size)
bars = compress_to_bar_once_range_met(
    ticks,
    price_range=16,
    tick_size=0.25
)

# Extract states from bar closes
states = get_states_from_ohlc(bars, method='close')
# → ['UP', 'DOWN', 'UP', ...]

# Fit predictor
from orderflow.stats import MarkovChainPredictor
predictor = MarkovChainPredictor(order=3)
predictor.fit(states)

# Predict next bar
next_pred, next_probs = predict_bar_state(bars, predictor)
print(f"Next bar prediction: {next_pred} {next_probs}")
```

### OHLC State Methods

| Method | Use Case | Interpretation |
|--------|----------|---|
| **"close"** | General trend | Close > prev close = UP |
| **"hl_range"** | Volatility & balance | Large range = choppy |
| **"oc_range"** | Intrabar direction | Open to close = bar color |

---

## Best Practices

### 1. Data Preparation

```python
# ✓ Validate inputs
if df.isnull().any().any():
    df = df.dropna()
    logging.warning(f"Dropped {len(df) - len(df_clean)} NaN rows")

# ✓ Use enough history
if len(df) < 100:
    raise ValueError("Minimum 100 bars/ticks required")

# ✓ Check for sufficient variation
state_counts = pd.Series(states).value_counts()
if state_counts.min() < 5:
    logging.warning("Very few samples in some states. Increase data or lower threshold.")
```

### 2. Model Selection

```python
# ✓ Use adaptive models (auto-selection)
predictor = AdaptiveMarkovChainPredictor(max_order=10)
predictor.fit(states)
logging.info(f"Auto-selected order: {predictor.best_order}")

# ✓ Monitor fit quality
patterns = len(predictor.transition_probs_by_order)
coverage = patterns / (3 ** predictor.best_order)
if coverage < 0.1:
    logging.warning(f"Low pattern coverage: {coverage:.1%}")
```

### 3. Backtesting Integration

```python
# Before live trading:
# 1. Train on historical data
predictor.fit(historical_states)

# 2. Validate on out-of-sample period
oos_states = get_states_from_ohlc(oos_data)
accuracy = sum(
    predictor.predict_next_state(oos_states[i-order:i]) == oos_states[i]
    for i in range(order, len(oos_states))
) / (len(oos_states) - order)
print(f"Out-of-sample accuracy: {accuracy:.1%}")

# 3. Trade only if confidence > threshold
dist = predictor.predict_distribution(recent_states)
confidence = max(dist.values())
if confidence > 0.55:  # Only trade with >55% confidence
    signal = max(dist, key=dist.get)
```

### 4. Monitoring & Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)

logger = logging.getLogger('orderflow.stats')

# Models log automatically:
# → "Markov(order=2) fitted on 1000 states. Found 45 unique patterns."
# → "AdaptiveMarkov fitted: best_order=3, val_ll=-0.1234, patterns=120"
# → "HMM fitted: 3 states, log-likelihood=1234.56"
```

---

## Complete Examples

### Example 1: Tick-by-Tick State Prediction

```python
import pandas as pd
from orderflow.stats import (
    threshold_prices_states,
    MarkovChainPredictor,
)

# Load tick data
ticks = pd.read_csv('tbt/2023_06_29.txt', sep=';')

# Extract prices
prices = ticks['Price'].tolist()

# Generate states
states = threshold_prices_states(prices, threshold=0.1)

# Train
predictor = MarkovChainPredictor(order=2)
predictor.fit(states)

# Predict next 5 moves
recent = states[-2:]
for _ in range(5):
    next_state = predictor.predict_next_state(recent)
    recent.append(next_state)
    recent = recent[-2:]
    print(f"Predicted: {next_state}")
```

### Example 2: Multi-Regime HMM + Trading

```python
from orderflow.stats import (
    select_best_hmm_model,
    compute_df_features,
    MultiFeatureHMM,
)
import pandas as pd
import numpy as np

# Compressed bars with features
df_bars = pd.read_csv('bars.csv')
df_feat = compute_df_features(df_bars, window_volatility=20)

# Train HMM
X = df_feat[['return', 'volatility', 'slope', 'log_volume']].values
best_hmm = select_best_hmm_model(
    X, n_states_range=[2, 3, 4], criterion='bic'
)
hmm = MultiFeatureHMM(model=best_hmm)
hmm.fit(X)

# Trading rules by regime
states = hmm.predict_states(X)
probs = hmm.predict_proba_states(X)

for i in range(len(states)):
    regime = states[i]
    confidence = max(probs[i])
    
    if regime == 0 and confidence > 0.7:
        print(f"Bar {i}: Buy (calm uptrend)")
    elif regime == 2 and confidence > 0.7:
        print(f"Bar {i}: Sell (panic)")
    else:
        print(f"Bar {i}: Neutral (low confidence)")
```

### Example 3: Strategy Monte Carlo Validation

```python
from orderflow.stats import (
    get_montecarlo_analysis,
    plot_montecarlo_paths,
    plot_montecarlo_distribution,
)
import pandas as pd

# Load backtest results
trades = pd.read_csv('backtest_trades.csv')
trades['Datetime'] = pd.to_datetime(trades['Datetime'])

# Run MC
equity_patterns, summary, stats = get_montecarlo_analysis(
    trades,
    n_rows_sample=40,
    n_simulations=1000,
    entry_col_name='PnL',
    confidence_level=0.99,  # 99% CI
    random_state=123
)

# Check if robust
if stats['ci_lower'] > 0:
    print("✓ Strategy is ROBUST (99% CI above zero)")
elif stats['mean_equity'] > 0 and stats['win_rate'] > 0.55:
    print("⚠ Strategy is MARGINAL (needs refinement)")
else:
    print("✗ Strategy is NOT ROBUST (reject)")

# Visualize
plot_montecarlo_paths(equity_patterns, show=False)
plot_montecarlo_distribution(stats, summary, show=False)
```

---

## Troubleshooting

### "Model not fitted" RuntimeError

```python
# ✗ Wrong
predictor.predict_next_state(recent)  # No fit() called yet

# ✓ Right
predictor.fit(states)
predictor.predict_next_state(recent)
```

### "Need >= X states for fitting"

```python
# ✓ Ensure sufficient data
if len(states) < 10:
    raise ValueError("Not enough historical states")

predictor.fit(states)
```

### Low pattern coverage / high variance predictions

```python
# Problem: Too many unique patterns, sparse data
if len(predictor.transition_probs_by_order) > len(states) / 5:
    # Solution: Use lower order or more data
    predictor = MarkovChainPredictor(order=1)  # Simpler model
    predictor.fit(states)
```

### Monte Carlo results noisy / unstable

```python
# Problem: Too few simulations
if n_simulations < 100:
    n_simulations = 500  # Use at least 500

# Problem: Sample size wrong
if n_rows_sample > len(trades) * 0.5:
    n_rows_sample = int(len(trades) * 0.3)  # Use ~30% of trades
```

---

## Summary Table

| Use Case | Tool | Why |
|----------|------|-----|
| **Predict next UP/DOWN** | `MarkovChainPredictor` | Simple, fast, good for trend |
| **Auto-select order** | `AdaptiveMarkovChainPredictor` | Avoids manual tuning |
| **Multi-feature regimes** | `MultiFeatureHMM` | Captures complex dynamics |
| **Test strategy robustness** | `get_montecarlo_analysis` | Bootstrap validates walk-forward |
| **Tick-by-tick analysis** | Raw prices → states | High granularity |
| **Compressed bars** | OHLC → states | Practical for live trading |

---

## References

- **Markov Chains**: Ross, S.M. (2014). Introduction to Probability Models
- **HMM**: Rabiner, L. (1989). A Tutorial on Hidden Markov Models
- **Bootstrap**: Efron & Tibshirani (1993). An Introduction to the Bootstrap
- **Regime Detection**: Hamilton, J.D. (1989). Probabilistic Approach to Regime Changes
