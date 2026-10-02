"""Tests for the group bootstrap of a total."""

import numpy as np

from orderflow.stats import bootstrap_total_by_group


def test_whole_groups_are_resampled_not_single_values():
    # Group sums are a = 2 and b = -5, so a resample of two groups can only total 4, -3 or -10.
    # Resampling single values would also produce totals such as 3 or -9.
    totals = bootstrap_total_by_group([1.0, 1.0, -5.0], ["a", "a", "b"], n_resamples=2_000)
    assert set(np.unique(totals)) == {4.0, -3.0, -10.0}


def test_the_same_seed_gives_the_same_distribution():
    values, groups = [3.0, -1.0, 2.0, -4.0], [1, 1, 2, 3]
    first = bootstrap_total_by_group(values, groups, n_resamples=500, seed=7)
    assert np.array_equal(first, bootstrap_total_by_group(values, groups, n_resamples=500, seed=7))
    assert not np.array_equal(first, bootstrap_total_by_group(values, groups, n_resamples=500, seed=8))
