"""Tests of the walk-forward check: synthetic calendars and blotters only."""

import numpy as np
import pandas as pd
import pytest

from orderflow.backtester.walk_forward import walk_forward_folds


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
