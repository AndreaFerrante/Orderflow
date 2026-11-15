"""Tests of the walk-forward check: synthetic calendars and blotters only."""

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester.walk_forward import walk_forward, walk_forward_folds


def days(n):
    return [f"2026-03-{d:02d}" for d in range(1, n + 1)]


def test_windows_roll_forward_by_the_test_length():
    folds = walk_forward_folds(days(10), train_days=4, test_days=2)
    assert folds["fold"].tolist() == [0, 1, 2]
    assert folds["train_start"].tolist() == ["2026-03-01", "2026-03-03", "2026-03-05"]
    assert folds["train_end"].tolist() == ["2026-03-04", "2026-03-06", "2026-03-08"]
    assert folds["test_start"].tolist() == ["2026-03-05", "2026-03-07", "2026-03-09"]
    assert folds["test_end"].tolist() == ["2026-03-06", "2026-03-08", "2026-03-10"]
    assert folds["n_train_days"].tolist() == [4, 4, 4]
    assert folds["n_test_days"].tolist() == [2, 2, 2]


def test_an_anchored_train_window_always_starts_on_the_first_day():
    folds = walk_forward_folds(days(10), train_days=4, test_days=2, anchored=True)
    assert folds["train_start"].tolist() == ["2026-03-01"] * 3
    assert folds["n_train_days"].tolist() == [4, 6, 8]
    assert folds["test_start"].tolist() == ["2026-03-05", "2026-03-07", "2026-03-09"]


def test_the_last_test_window_may_be_short():
    folds = walk_forward_folds(days(9), train_days=4, test_days=2)
    assert folds["test_end"].tolist() == ["2026-03-06", "2026-03-08", "2026-03-09"]
    assert folds["n_test_days"].tolist() == [2, 2, 1]


def test_days_are_counted_as_given_not_on_the_calendar():
    # Thursday, Friday, Monday, Tuesday: the weekend is not a day.
    folds = walk_forward_folds(["2026-03-05", "2026-03-06", "2026-03-09", "2026-03-10"],
                               train_days=2, test_days=1)
    assert folds["train_end"].tolist() == ["2026-03-06", "2026-03-09"]
    assert folds["test_start"].tolist() == ["2026-03-09", "2026-03-10"]


def test_unsorted_repeated_dates_of_any_type_are_one_sorted_calendar():
    messy = [pd.Timestamp("2026-03-03 14:00"), "2026-03-01", "2026-03-02", "2026-03-01",
             pd.Timestamp("2026-03-04").date()]
    pd.testing.assert_frame_equal(walk_forward_folds(messy, train_days=2, test_days=1),
                                  walk_forward_folds(days(4), train_days=2, test_days=1))


def test_too_few_days_for_one_fold_is_refused():
    with pytest.raises(ValueError, match="fold"):
        walk_forward_folds(days(4), train_days=4, test_days=2)


@pytest.mark.parametrize("train_days, test_days", [(0, 2), (2, 0), (-1, 2), (2.5, 2)])
def test_a_window_that_is_not_a_positive_whole_number_of_days_is_refused(train_days, test_days):
    with pytest.raises(ValueError, match="days"):
        walk_forward_folds(days(10), train_days=train_days, test_days=test_days)


def blotter(pnl_by_day):
    """{"2026-03-01": [1.0, -2.0], ...} -> the engine's trades_df columns the check reads."""
    return pd.DataFrame([
        {"entry_datetime": pd.Timestamp(f"{day} 09:{i:02d}"), "net_pnl": float(p)}
        for day, pnls in pnl_by_day.items() for i, p in enumerate(pnls)
    ])


D = days(6)
A = blotter({D[0]: [5], D[1]: [5], D[2]: [-1], D[3]: [-1], D[4]: [4], D[5]: [4]})
B = blotter({D[0]: [1], D[1]: [1], D[2]: [3], D[3]: [3], D[4]: [-2], D[5]: [-2]})
BY_TOTAL = dict(train_days=2, test_days=2, score=np.sum, min_train_trades=1)


def test_each_fold_trades_the_candidate_that_scored_best_in_its_train_window():
    folds = walk_forward({"a": A, "b": B}, **BY_TOTAL).folds
    assert folds["chosen"].tolist() == ["a", "b"]
    assert folds["train_score"].tolist() == [10.0, 6.0]
    assert folds["train_net_pnl"].tolist() == [10.0, 6.0]
    assert folds["train_trades"].tolist() == [2, 2]
    assert folds["test_net_pnl"].tolist() == [-2.0, -4.0]
    assert folds["test_trades"].tolist() == [2, 2]


def test_the_out_of_sample_blotter_is_the_chosen_trades_of_each_test_window_in_time_order():
    oos = walk_forward({"a": A, "b": B}, **BY_TOTAL).oos_trades
    assert oos["net_pnl"].tolist() == [-1.0, -1.0, -2.0, -2.0]
    assert oos["candidate"].tolist() == ["a", "a", "b", "b"]
    assert oos["fold"].tolist() == [0, 0, 1, 1]
    assert oos["entry_datetime"].is_monotonic_increasing
    assert oos.index.tolist() == [0, 1, 2, 3]


def test_by_default_the_steadier_candidate_wins_not_the_one_with_more_pnl():
    steady = blotter({D[0]: [1, 1, 1, 1.1], D[1]: [0]})
    lumpy = blotter({D[0]: [10, -9, 10, -6], D[1]: [0]})
    result = walk_forward({"lumpy": lumpy, "steady": steady}, train_days=1, test_days=1, min_train_trades=1)
    assert result.folds["chosen"].tolist() == ["steady"]


def test_a_candidate_with_too_few_train_trades_is_not_eligible():
    one_big = blotter({D[0]: [100], D[2]: [0]})
    two_small = blotter({D[0]: [1], D[1]: [1], D[2]: [0]})
    result = walk_forward({"one_big": one_big, "two_small": two_small},
                          train_days=2, test_days=1, score=np.sum, min_train_trades=2)
    assert result.folds["chosen"].tolist() == ["two_small"]


def test_a_fold_with_no_eligible_candidate_sits_out():
    result = walk_forward({"a": A, "b": B}, train_days=2, test_days=2, score=np.sum, min_train_trades=5)
    assert result.folds["chosen"].isna().all()
    assert result.folds["train_score"].isna().all()
    assert result.folds["train_trades"].tolist() == [0, 0]
    assert result.folds["train_net_pnl"].tolist() == [0.0, 0.0]
    assert result.folds["test_trades"].tolist() == [0, 0]
    assert result.folds["test_net_pnl"].tolist() == [0.0, 0.0]
    assert result.oos_trades.empty


def test_a_score_that_is_not_a_number_is_not_eligible():
    result = walk_forward({"a": A}, train_days=2, test_days=2, score=lambda pnl: float("nan"), min_train_trades=1)
    assert result.folds["chosen"].isna().all()


def test_a_candidate_with_no_trades_at_all_is_never_chosen_and_never_fails():
    result = walk_forward({"empty": pd.DataFrame(), "a": A}, **BY_TOTAL)
    assert result.folds["chosen"].tolist() == ["a", "a"]


def test_the_first_candidate_wins_a_tie():
    assert walk_forward({"x": A, "y": A.copy()}, **BY_TOTAL).folds["chosen"].tolist() == ["x", "x"]


def test_a_candidate_key_can_be_a_tuple_of_parameters():
    result = walk_forward({(10, 8): A, (12, 6): B}, **BY_TOTAL)
    assert result.folds["chosen"].tolist() == [(10, 8), (12, 6)]
    assert result.oos_trades["candidate"].tolist() == [(10, 8), (10, 8), (12, 6), (12, 6)]


def test_days_without_trades_count_when_the_calendar_is_given():
    sparse = blotter({D[0]: [1], D[1]: [1], D[4]: [1]})
    with_calendar = walk_forward({"a": sparse}, dates=D, **BY_TOTAL).folds
    assert with_calendar["test_start"].tolist() == [D[2], D[4]]
    # without it the calendar is the three days that traded: one fold, tested on the third
    assert walk_forward({"a": sparse}, **BY_TOTAL).folds["test_start"].tolist() == [D[4]]


def test_trades_are_read_in_time_order_whatever_the_row_order():
    shuffled = {"a": A.sample(frac=1, random_state=0), "b": B.sample(frac=1, random_state=1)}
    expected, got = walk_forward({"a": A, "b": B}, **BY_TOTAL), walk_forward(shuffled, **BY_TOTAL)
    pd.testing.assert_frame_equal(got.folds, expected.folds)
    assert got.oos_trades["net_pnl"].tolist() == expected.oos_trades["net_pnl"].tolist()


def test_no_candidates_is_refused():
    with pytest.raises(ValueError, match="candidate"):
        walk_forward({}, **BY_TOTAL)


def test_efficiency_is_the_out_of_sample_daily_pnl_over_the_in_sample_one():
    # in sample 16 over 4 days = 4 a day; out of sample -6 over 4 days = -1.5 a day
    assert walk_forward({"a": A, "b": B}, **BY_TOTAL).efficiency == pytest.approx(-0.375)


def test_efficiency_is_none_when_the_in_sample_pnl_is_not_positive():
    loser = blotter({D[0]: [-1], D[1]: [-1], D[2]: [1], D[3]: [1]})
    assert walk_forward({"a": loser}, **BY_TOTAL).efficiency is None


def test_efficiency_is_none_when_no_fold_traded():
    result = walk_forward({"a": A, "b": B}, train_days=2, test_days=2, score=np.sum, min_train_trades=5)
    assert result.efficiency is None
