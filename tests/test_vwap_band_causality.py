"""Causality gate: VWAP bands at a row must not depend on later ticks."""
import numpy as np
import pandas as pd
import pytest

from orderflow.market.profiles.vwap import get_vwap


def assert_prefix_stable(fn, ticks, cut):
    full = fn(ticks).iloc[:cut]
    part = fn(ticks.iloc[:cut])
    pd.testing.assert_frame_equal(
        full.reset_index(drop=True), part.reset_index(drop=True), check_exact=True
    )


def make_ticks():
    i = np.arange(400)
    return pd.DataFrame(
        {
            "SessionType": np.where(i < 200, "ETH", "RTH"),
            "Price": 100 + 2 * np.sin(i / 10),
            "Volume": (1 + i % 5).astype("int64"),
        }
    )


def test_the_check_catches_a_lookahead_band():
    def forward_looking(t):  # whole-frame VWAP on every row
        v = (t.Price * t.Volume).sum() / t.Volume.sum()
        return pd.DataFrame({"vwap": [v] * len(t)})

    with pytest.raises(AssertionError):
        assert_prefix_stable(forward_looking, make_ticks(), cut=250)


def test_get_vwap_bands_do_not_change_when_later_ticks_are_removed():
    assert_prefix_stable(get_vwap, make_ticks(), cut=250)
