"""Tests of the Monte Carlo check of a trade blotter: synthetic blotters only."""

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester.monte_carlo import monte_carlo
from orderflow.stats import bootstrap_total_by_group


def blotter(pnl_by_day):
    return pd.DataFrame([
        {"entry_datetime": pd.Timestamp(f"{day} 09:{i:02d}"), "net_pnl": float(p)}
        for day, pnls in pnl_by_day.items() for i, p in enumerate(pnls)
    ])


def random_blotter(n_days=40, seed=3):
    rng = np.random.default_rng(seed)
    return blotter({str(day.date()): rng.normal(1.0, 20.0, size=rng.integers(1, 6)).tolist()
                    for day in pd.bdate_range("2026-01-05", periods=n_days)})


def drawdown_from_zero(pnl):
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return float((np.maximum.accumulate(equity) - equity).max())


def test_a_path_is_whole_days_drawn_with_replacement():
    # Day sums are 2 and -5, so two drawn days can only total 4, -3 or -10.
    paths = monte_carlo(blotter({"2026-03-02": [1, 1], "2026-03-03": [-5]}), n_resamples=2_000).paths
    assert set(paths["net_pnl"].unique()) == {4.0, -3.0, -10.0}


def test_totals_are_those_of_the_group_bootstrap_of_the_stats_module():
    trades = random_blotter()
    day = trades["entry_datetime"].dt.strftime("%Y-%m-%d")
    totals = monte_carlo(trades, n_resamples=500, seed=7).paths["net_pnl"].to_numpy()
    # the same days are drawn; the sums are added in another order, so equal to rounding only
    assert np.allclose(totals, bootstrap_total_by_group(trades["net_pnl"], day, n_resamples=500, seed=7))


def test_the_drawdown_of_a_path_is_the_one_of_its_days_laid_end_to_end_trade_by_trade():
    trades = random_blotter()
    by_day = [g["net_pnl"].to_numpy() for _, g in trades.groupby(trades["entry_datetime"].dt.strftime("%Y-%m-%d"))]
    picks = np.random.default_rng(11).integers(0, len(by_day), size=(200, len(by_day)))
    expected = [drawdown_from_zero(np.concatenate([by_day[d] for d in row])) for row in picks]
    got = monte_carlo(trades, n_resamples=200, seed=11).paths["max_drawdown"].to_numpy()
    assert np.allclose(got, expected)


def test_a_day_keeps_its_trades_in_entry_order_whatever_the_row_order():
    # In time order 5, 5, -3, -3 the curve peaks at 10 and falls to 4. Row order would give 3.
    one_day = blotter({"2026-03-02": [5, 5, -3, -3]}).iloc[[2, 0, 3, 1]]
    result = monte_carlo(one_day, n_resamples=10)
    assert result.observed_max_drawdown == 6.0
    assert (result.paths["max_drawdown"] == 6.0).all()


def test_a_loss_on_the_first_trade_is_a_drawdown_from_the_zero_start():
    result = monte_carlo(blotter({"2026-03-02": [-5, -5, -5]}), n_resamples=10)
    assert result.observed_max_drawdown == 15.0
    assert (result.paths["max_drawdown"] == 15.0).all()


def test_observed_values_are_those_of_the_blotter_as_it_was_traded():
    result = monte_carlo(blotter({"2026-03-02": [10], "2026-03-03": [-4, -2], "2026-03-04": [5]}), n_resamples=10)
    assert result.observed_net_pnl == 9.0
    assert result.observed_max_drawdown == 6.0


def test_p_loss_is_the_share_of_paths_that_do_not_make_money():
    mixed = monte_carlo(blotter({"2026-03-02": [1, 1], "2026-03-03": [-5]}), n_resamples=2_000)
    assert mixed.p_loss == pytest.approx((mixed.paths["net_pnl"] <= 0).mean())
    assert 0.0 < mixed.p_loss < 1.0
    assert monte_carlo(blotter({"2026-03-02": [1], "2026-03-03": [2]}), n_resamples=50).p_loss == 0.0
    assert monte_carlo(blotter({"2026-03-02": [-1], "2026-03-03": [-2]}), n_resamples=50).p_loss == 1.0


def test_the_same_seed_gives_the_same_paths():
    trades = random_blotter()
    first = monte_carlo(trades, n_resamples=300, seed=5).paths
    pd.testing.assert_frame_equal(first, monte_carlo(trades, n_resamples=300, seed=5).paths)
    assert not first.equals(monte_carlo(trades, n_resamples=300, seed=6).paths)


def test_one_row_per_resample():
    paths = monte_carlo(random_blotter(), n_resamples=123).paths
    assert len(paths) == 123
    assert paths.columns.tolist() == ["net_pnl", "max_drawdown"]


@pytest.mark.parametrize("empty", [pd.DataFrame(), pd.DataFrame({"entry_datetime": [], "net_pnl": []})])
def test_a_blotter_with_no_trades_is_refused(empty):
    with pytest.raises(ValueError, match="no trades"):
        monte_carlo(empty)


def test_the_out_of_sample_blotter_of_a_walk_forward_can_be_resampled():
    from orderflow.backtester.walk_forward import walk_forward

    days = [f"2026-03-{d:02d}" for d in range(1, 7)]
    oos = walk_forward({"a": blotter({day: [1.0, -0.5] for day in days})},
                       train_days=2, test_days=2, min_train_trades=1).oos_trades
    result = monte_carlo(oos, n_resamples=20)
    assert len(result.paths) == 20
    assert result.observed_net_pnl == 2.0  # four test days, +0.5 each


def test_the_reliability_tools_are_exported_by_the_backtester_package():
    import orderflow.backtester as backtester

    for name in ("walk_forward_folds", "walk_forward", "WalkForwardResult", "monte_carlo", "MonteCarloResult"):
        assert name in backtester.__all__
        assert hasattr(backtester, name)
