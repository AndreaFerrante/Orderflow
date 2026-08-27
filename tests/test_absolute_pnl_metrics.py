"""Currency-denominated P&L metrics for trade blotters.

The sibling ``max_drawdown`` / ``sharpe_ratio`` compound *fractional* returns
and need an account equity base.  These two sum currency amounts instead.
"""

import numpy as np
import pandas as pd
import pytest

from orderflow.stats import max_drawdown_absolute, trade_sharpe


def test_max_drawdown_absolute_measures_peak_to_trough_in_currency():
    # cum: 10, 5, 15, 10 -> peak 10, 10, 15, 15 -> dd 0, 5, 0, 5
    assert max_drawdown_absolute([10.0, -5.0, 10.0, -5.0]) == pytest.approx(5.0)


def test_max_drawdown_absolute_accumulates_across_a_losing_streak():
    """The deepest drawdown spans several trades -- it is not the worst single loss."""
    # cum: 10, 4, -2, 3 -> peak 10 -> trough -2 -> dd 12 > worst single loss 6
    assert max_drawdown_absolute([10.0, -6.0, -6.0, 5.0]) == pytest.approx(12.0)


def test_max_drawdown_absolute_is_zero_on_a_monotonically_rising_curve():
    assert max_drawdown_absolute([1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_max_drawdown_absolute_does_not_compound():
    """Compounding these as fractional returns would give a different answer --
    this is what separates it from max_drawdown()."""
    # sum path: cum 100, 50 -> dd 50.  compound path would be ~ -0.5 fraction.
    assert max_drawdown_absolute([100.0, -50.0]) == pytest.approx(50.0)


def test_max_drawdown_absolute_is_zero_on_empty_input():
    assert max_drawdown_absolute([]) == 0.0


def test_max_drawdown_absolute_accepts_series_and_arrays():
    data = [10.0, -6.0, -6.0, 5.0]
    assert max_drawdown_absolute(pd.Series(data)) == pytest.approx(12.0)
    assert max_drawdown_absolute(np.asarray(data)) == pytest.approx(12.0)


def test_trade_sharpe_is_mean_over_stdev_scaled_by_sqrt_n():
    # mean 2.5, stdev(ddof=1) 8.660254, n 4 -> 0.57735
    assert trade_sharpe([10.0, -5.0, 10.0, -5.0]) == pytest.approx(0.57735, rel=1e-4)


def test_trade_sharpe_sign_follows_the_mean():
    """A losing blotter must never report a positive Sharpe."""
    assert trade_sharpe([-10.0, 5.0, -10.0, 5.0]) < 0
    assert trade_sharpe([10.0, -5.0, 10.0, -5.0]) > 0


def test_trade_sharpe_uses_the_sample_stdev():
    """ddof=1, not ddof=0 -- population stdev would inflate the ratio."""
    data = [10.0, -5.0, 10.0, -5.0]
    population = np.mean(data) / np.std(data, ddof=0) * np.sqrt(len(data))
    assert trade_sharpe(data) != pytest.approx(population)


def test_trade_sharpe_is_zero_when_every_trade_is_identical():
    """Zero variance must return 0.0, not inf or nan."""
    result = trade_sharpe([2.0, 2.0, 2.0])
    assert result == 0.0
    assert np.isfinite(result)


def test_trade_sharpe_is_zero_below_two_trades():
    assert trade_sharpe([5.0]) == 0.0
    assert trade_sharpe([]) == 0.0


def test_trade_sharpe_scales_with_sample_size():
    """Same distribution, more trades -> higher confidence -> higher Sharpe."""
    short = trade_sharpe([10.0, -5.0] * 4)
    long = trade_sharpe([10.0, -5.0] * 16)
    assert long > short
