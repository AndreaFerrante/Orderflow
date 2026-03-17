# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Orderflow** is a Python module for institutional-grade tick-by-tick trading data analysis and orderflow research. The project is built around four main pillars:

1. **Data Compression** — Transform tick-by-tick data into aggregated bars (volume, range, time-based)
2. **Backtesting** — Tick-level execution simulation with pluggable exit strategies and risk management
3. **Statistical Analysis** — Hypothesis testing, Markov chains, HMM regime detection, Monte Carlo analysis
4. **Market Microstructure** — Volume profiles, VWAP, delta analysis, auction theory, DOM analysis

The package uses **Pandas** and **Polars** for data manipulation (with Polars favored for large datasets) and **Numba** for accelerated computations.

## Installation & Development Setup

```bash
# Install in editable mode (for development)
pip install -e .

# Install with test dependencies
pip install -e ".[dev]"  # if dev extra exists, otherwise use:
pip install pytest pandas numpy polars

# Install from pyproject.toml directly
pip install -r requirements.txt
```

## Commands

### Build & Install
```bash
# Build distribution
python -m build

# Install package locally
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run tests in a specific file
pytest orderflow/test/test_volume_bars.py

# Run a specific test
pytest orderflow/test/test_volume_bars.py::test_name

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=orderflow

# Run a quick check (first test file)
python scripts/test_volume_bars.py
```

### Linting & Code Quality
```bash
# Check for issues (using built-in tools if pre-commit is configured)
# Note: This project uses commitizen for conventional commits (see .pre-commit-config.yaml)
```

### Version & Release
```bash
# Automatic version management via setuptools_scm
# Versions are derived from git tags; stored in orderflow/_version.py

# Create a commit with conventional commit message (required by pre-commit hook)
# Format: type(scope): description
# Examples:
#   feat(backtester): add volatility exit strategy
#   fix(compressor): handle missing datetime columns
#   refactor(stats): improve HMM feature engineering
```

## Architecture

### Module Structure
```
orderflow/
├── backtester/         # Professional-grade tick-by-tick backtester
│   ├── engine.py       # Core BacktestEngine and BacktestResult
│   ├── exits.py        # Exit strategies (FixedTPSL, TrailingStop, Volatility, etc.)
│   ├── execution.py    # SlippageModel, FillSimulator
│   ├── risk.py         # RiskManager for mechanical TP/SL
│   ├── metrics.py      # PerformanceMetrics and post-trade analytics
│   └── examples.py     # Runnable usage patterns
│
├── compressor/         # Tick aggregation (volume, range, time bars)
│   └── compressor.py   # Main compression functions
│
├── stats/              # Statistical analysis & regime detection
│   ├── stats.py        # Descriptive stats, risk metrics, Sharpe, Sortino, etc.
│   ├── returns.py      # Return series, equity curves, drawdowns
│   ├── hypothesis.py   # ADF, KPSS, Jarque-Bera, Ljung-Box, CUSUM
│   ├── correlation.py  # Rolling correlation, rank correlation, factor analysis
│   ├── montecarlo.py   # Bootstrap analysis with confidence intervals
│   ├── markov.py       # Markov chains, fixed-order & adaptive
│   └── markov_utilities.py  # HMM, feature engineering, SierraChart I/O
│
├── volume_profile.py   # Volume profile, POC, VAH/VAL, dynamic delta
├── vwap.py             # VWAP computation
├── auctions.py         # Auction theory, block aggregation
├── dom.py              # Depth of market analysis
├── footprint.py        # Large print filtering
├── viz.py              # Visualization utilities
├── _volume_factory.py   # Volume distribution analysis
└── test/               # Test suite (pytest)
```

### Data Conventions

**Tick-by-tick data format:**
- **Date**: YYYY-MM-DD format
- **Time**: HH:MM:SS.ffffff format (microsecond precision)
- **Price**: float64, actual trade price
- **Volume**: float64, number of contracts/shares
- **TradeType**: int, 1 = BID (buyer-initiated), 2 = ASK (seller-initiated)
- **Optional**: AskPrice, BidPrice, AskSize, BidSize, DOM columns (level 1–N)

**Aggregated bar format (post-compression):**
- OpenTime, CloseTime (datetime)
- Open, High, Low, Close (float64)
- Volume, BidVolume, AskVolume (sum of tick-level sides)
- NumberOfTrades (tick count)

### Key Design Patterns

1. **Dual DataFrame Support**: Most functions accept both Pandas and Polars. Functions ending in `_pl()` are Polars-only; others auto-detect via `isinstance()`.

2. **Automatic Datetime Handling**: Compression functions automatically convert separate Date + Time columns into a Datetime column if missing. This ensures robustness across data sources.

3. **Numba Acceleration**: Hot-path computations (backtester loop, volume bar grouping) use Numba JIT; pure Python fallbacks exist for compatibility.

4. **Lazy Polars Evaluation**: Polars functions use method chaining and lazy evaluation for performance; `.collect()` triggers computation.

5. **No Lookahead Bias**: All rolling/expanding operations are strictly causal (no future data leakage). This is enforced in backtester tick ordering and stats windows.

### Module Responsibilities

**backtester**:
- Simulates tick-by-tick order execution with fill models and slippage.
- Pluggable exit strategies (fixed TP/SL, trailing stop, time-based, volatility, composite).
- Computes trade-level and strategy-level metrics (Sharpe, Sortino, max drawdown, win rate, profit factor).

**compressor**:
- Groups ticks into volume bars (standard), range bars, or time bars.
- Preserves bid/ask volume breakdown and trade counts per bar.
- Handles datetime conversion and missing columns gracefully.

**stats**:
- Descriptive statistics (mean, std, skew, kurtosis, percentiles).
- Risk metrics (Sharpe, Sortino, Calmar, omega, tail ratio, profit factor).
- Hypothesis testing (ADF, KPSS, Jarque-Bera, Ljung-Box, CUSUM, Holm-Bonferroni correction).
- Regime detection via fixed-order Markov chains, adaptive Markov chains, and HMM.
- Monte Carlo bootstrap for strategy robustness with confidence intervals.
- Correlation analysis (rolling, rank, stability, eigenvalue decomposition).

**volume_profile**:
- Computes point-of-control (POC), value area high/low (VAH/VAL) per price level.
- Supports bid/ask delta by level and session aggregation.
- Dynamic cumulative delta across time windows.

**auctions**:
- Implements auction theory concepts (blocks, imbalances, forward-looking outcomes).
- Identifies high-activity auctions by aggregating consecutive ticks.

### Type Hints & Data Validation

- All public functions have full type hints (no `Any` without justification).
- Use `pydantic` for config (see `configuration.py`).
- Vectorized operations preferred over Python loops.
- Timestamps always UTC (or explicitly noted as market time).

### Testing

- Tests live in `orderflow/test/` and use pytest.
- Focus on trading logic (backtester, exits, risk): aim for high coverage.
- Integration tests preferred over mocks for data transformations.
- Example: `test_volume_bars.py` tests compression against known tick data.

### Version & Releases

- **Version source**: `setuptools_scm` derives version from git tags.
- **Fallback**: `pyproject.toml` contains `fallback_version = "0.4.0"`.
- **Conventional Commits**: Pre-commit hook enforces `type(scope): description` (commitizen).
- **Supported types**: feat, fix, refactor, perf, test, chore, docs.

## Common Workflows

### Adding a New Exit Strategy
1. Subclass `BaseExitStrategy` in `orderflow/backtester/exits.py`.
2. Implement `should_exit(tick, position_state, order_book_state) -> ExitSignal`.
3. Add to `examples.py` with a runnable usage pattern.
4. Test with `BacktestEngine.run()` on sample data.

### Adding a Statistical Test
1. Add function to appropriate file in `orderflow/stats/` (e.g., `hypothesis.py` for tests).
2. Return a `TestResult` dataclass with `.test_name`, `.statistic`, `.p_value`, `.reject_null`, `.alpha`, `.detail`.
3. Export from `stats/__init__.py`.
4. Document assumptions and p-value interpretation in docstring.

### Analyzing New Tick Data
1. Load as Pandas/Polars DataFrame with required columns (Date, Time, Price, Volume, TradeType).
2. Compress to bars: `compress_to_volume_bars(tick_data, volume_amount=1000)`.
3. Analyze with volume profile, VWAP, or delta.
4. Run backtest with custom exit strategy.
5. Evaluate with `PerformanceMetrics` or Monte Carlo.

## Notes for Contributors

1. **Preserve causal order**: Backtester processes ticks strictly chronologically; stats functions window data with no future peeking.

2. **DataFrame library choice**: If adding a function, support both Pandas and Polars where possible. Use `isinstance()` checks or separate `_pd()` and `_pl()` variants.

3. **Numba compatibility**: NumPy-friendly loops in hot paths; avoid object dtypes and dynamic allocations in JIT-compiled functions.

4. **Documentation**: Each module has a detailed `.md` guide (e.g., `backtester/README.md`, `compressor/compressor.md`, `stats/stats.md`). Update these when adding features.

5. **Commit discipline**: Use conventional commits. Pre-commit hook will reject commits with malformed messages.

6. **Data immutability**: Tick data is read-only; transformations produce new DataFrames (don't mutate in-place).
