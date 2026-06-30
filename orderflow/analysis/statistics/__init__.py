"""Descriptive statistics and inferential testing."""

from orderflow.stats.correlation import (
    correlation_eigenvalues,
    correlation_stability,
    rank_correlation,
    rolling_correlation,
)
from orderflow.stats.hypothesis import (
    TestResult,
    adf_test,
    cusum_test,
    holm_bonferroni,
    is_stationary,
    jarque_bera_test,
    kpss_test,
    ljung_box_test,
)
from orderflow.stats.returns import (
    annualise_return,
    annualise_volatility,
    arithmetic_to_log,
    drawdown_series,
    equity_curve,
    log_to_arithmetic,
    rolling_volatility,
    to_arithmetic_returns,
    to_log_returns,
    underwater_duration,
)
from orderflow.stats.stats import (
    autocorrelation,
    calmar_ratio,
    cvar_historical,
    describe,
    gain_to_pain_ratio,
    get_kurtosis,
    information_ratio,
    hurst_exponent,
    is_skewed,
    max_drawdown,
    omega_ratio,
    profit_factor,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    var_historical,
)
