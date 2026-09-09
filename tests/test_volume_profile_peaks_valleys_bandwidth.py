"""Per-call KDE bandwidth / tick-size wiring for get_volume_profile_peaks_valleys.

The function historically read the module-level ``KDE_VARIANCE_VALUE`` constant
directly, so every instrument was smoothed with the same price-unit bandwidth.
These tests pin the new ``kde_bandwidth`` keyword (per-symbol control) and the
already-existing-but-unpassed ``tick_size`` keyword.
"""

import numpy as np
import pandas as pd
import pytest

from orderflow.core.configuration import (
    FUTURE_VALUES,
    KDE_VARIANCE_BY_TICKER,
    KDE_VARIANCE_VALUE,
)
from orderflow.market.profiles.volume_profile import get_volume_profile_peaks_valleys


TICK = 0.05
BASE = 100.00
N_LEVELS = 21          # BASE .. BASE + 20*TICK
MODE_LOW, MODE_HIGH = 4, 16
VALLEY = 10


def _bimodal_ticks() -> pd.DataFrame:
    """Deterministic tick stream whose session volume profile is bimodal.

    Every price level is seeded once so the profile is fully populated from the
    first iteration, then extra rows add weight around the two modes, leaving a
    clear valley at ``VALLEY``.
    """
    levels = [round(BASE + k * TICK, 2) for k in range(N_LEVELS)]

    def weight(k: int) -> int:
        lo = np.exp(-(((k - MODE_LOW) / 2.0) ** 2))
        hi = np.exp(-(((k - MODE_HIGH) / 2.0) ** 2))
        return 1 + int(round(30 * (lo + hi)))

    prices: list[float] = list(levels)                    # seed pass
    for k, price in enumerate(levels):
        prices.extend([price] * weight(k))                # weight pass

    return pd.DataFrame(
        {
            "Price": prices,
            "Volume": np.ones(len(prices), dtype=float),
            "SessionType": ["RTH"] * len(prices),
        }
    )


@pytest.fixture(scope="module")
def ticks() -> pd.DataFrame:
    return _bimodal_ticks()


def test_bandwidth_is_not_ignored(ticks: pd.DataFrame) -> None:
    """A sub-tick bandwidth and a range-spanning one must smooth differently.

    Guards against the old behaviour where the KDE always used the module
    constant and the argument had no effect.
    """
    narrow = get_volume_profile_peaks_valleys(ticks, kde_bandwidth=TICK)
    wide = get_volume_profile_peaks_valleys(ticks, kde_bandwidth=0.8)

    assert not np.array_equal(narrow, wide)


def test_tick_size_changes_peak_area_classification(ticks: pd.DataFrame) -> None:
    """``tick_size`` feeds the peak-distance math, so it must alter the labels."""
    fine = get_volume_profile_peaks_valleys(ticks, tick_size=0.05, kde_bandwidth=TICK)
    coarse = get_volume_profile_peaks_valleys(ticks, tick_size=0.25, kde_bandwidth=TICK)

    assert not np.array_equal(fine, coarse)


def test_every_future_ticker_has_a_bandwidth() -> None:
    """The lookup must cover every configured instrument so switching symbol
    can never silently fall back to the ES-tuned constant."""
    missing = set(FUTURE_VALUES["Ticker"]) - set(KDE_VARIANCE_BY_TICKER)
    assert not missing, f"no KDE bandwidth for {sorted(missing)}"


def test_es_and_mes_bandwidth_is_the_legacy_constant() -> None:
    """ES/MES must keep the historical value so their results are unchanged."""
    assert KDE_VARIANCE_BY_TICKER["ES"] == KDE_VARIANCE_VALUE
    assert KDE_VARIANCE_BY_TICKER["MES"] == KDE_VARIANCE_VALUE


def test_defaults_match_legacy_bare_call(ticks: pd.DataFrame) -> None:
    """Bare call must stay byte-identical to the old constant-driven behaviour."""
    bare = get_volume_profile_peaks_valleys(ticks)
    explicit = get_volume_profile_peaks_valleys(
        ticks, tick_size=0.25, kde_bandwidth=KDE_VARIANCE_VALUE
    )

    np.testing.assert_array_equal(bare, explicit)
