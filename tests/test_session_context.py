"""Tests for causal session context: VWAP crosses, prior levels, running state, trend state."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from orderflow.market.large_order_flow import _count_vwap_crosses, _running_vwap_crosses


def test_running_crosses_counts_confirmed_flips():
    price = np.array([2.0, -2.0, 0.5, 2.0])
    out = _running_vwap_crosses(price, np.zeros(4), confirm_distance=1.0)
    assert out.tolist() == [0, 1, 1, 2]


def test_running_crosses_prefix_invariant():
    rng = np.random.default_rng(0)
    price = rng.normal(0.0, 3.0, 500)
    vwap = np.zeros(500)
    full = _running_vwap_crosses(price, vwap, confirm_distance=1.0)
    for k in (1, 7, 250, 499):
        part = _running_vwap_crosses(price[:k], vwap[:k], confirm_distance=1.0)
        assert part.tolist() == full[:k].tolist()
    assert _count_vwap_crosses(price, vwap, confirm_distance=1.0) == int(full[-1])
