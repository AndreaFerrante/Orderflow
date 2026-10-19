"""Tests of the VWAP 2 SD band-touch signal: synthetic tapes only."""

from datetime import datetime, timedelta

import polars as pl

from orderflow.market.vwap_reversion import find_vwap_band_touch_signals


def ticks(prices, start="2026-03-02 09:00:00", step_s=60, session="RTH", index0=0):
    t0 = datetime.fromisoformat(start)
    return pl.DataFrame([{
        "Index": index0 + i, "Date": (t0 + timedelta(seconds=step_s * i)).date().isoformat(),
        "Datetime": t0 + timedelta(seconds=step_s * i), "SessionType": session, "Price": float(p),
        "vwap_sd2_top": 102.0, "vwap_sd2_bottom": 98.0, "vwap_sd3_top": 103.0, "vwap_sd3_bottom": 97.0,
    } for i, p in enumerate(prices)])


def test_long_at_lower_band_short_at_upper_band_with_stop_prices():
    long_ = find_vwap_band_touch_signals(ticks([100, 98, 100]))
    assert long_.height == 1
    assert long_["side"].to_list() == [1]
    assert long_.row(0, named=True)["signal_index"] == 1
    assert long_.row(0, named=True)["entry_index"] == 2
    assert long_.row(0, named=True)["entry_price"] == 100.0
    assert long_.row(0, named=True)["stop_loss"] == 97.0
    short = find_vwap_band_touch_signals(ticks([100, 102, 100]))
    assert short["side"].to_list() == [-1]
    assert short.row(0, named=True)["stop_loss"] == 103.0


def test_three_ticks_beyond_the_band_are_one_signal():
    assert find_vwap_band_touch_signals(ticks([100, 98, 98, 98, 100]))["signal_index"].to_list() == [1]


def test_each_day_is_independent():
    both = pl.concat([ticks([100, 98, 98]), ticks([98, 100, 100], start="2026-03-03 09:00:00", index0=3)])
    assert find_vwap_band_touch_signals(both)["signal_index"].to_list() == [1, 3]


def test_signal_rearms_after_price_returns_inside():
    assert find_vwap_band_touch_signals(ticks([100, 98, 100, 98, 100]))["signal_index"].to_list() == [1, 3]


def test_touch_that_started_before_entry_time_is_not_a_signal():
    tape = ticks([98, 98, 100, 98, 100], start="2026-03-02 08:50:00", step_s=600)  # 08:50 ... 09:30
    assert find_vwap_band_touch_signals(tape)["signal_index"].to_list() == [3]


def test_eth_ticks_are_ignored():
    assert find_vwap_band_touch_signals(ticks([100, 98, 100], session="ETH")).height == 0


def test_signal_on_the_last_session_tick_has_no_row():
    assert find_vwap_band_touch_signals(ticks([100, 100, 98])).height == 0


def test_null_bands_give_no_signal():
    tape = ticks([100, 98, 100]).with_columns(vwap_sd2_bottom=pl.lit(None, dtype=pl.Float64))
    assert find_vwap_band_touch_signals(tape).height == 0


def test_long_and_short_are_independent_edges():
    found = find_vwap_band_touch_signals(ticks([100, 98, 102, 100]))
    assert found["side"].to_list() == [1, -1]
    assert found["signal_index"].to_list() == [1, 2]


def test_signal_does_not_read_later_ticks():
    full = ticks([100, 98, 100, 102, 100, 98, 100])
    part = find_vwap_band_touch_signals(full.head(5))
    expected = find_vwap_band_touch_signals(full).filter(pl.col("entry_index") < 5)
    assert part.equals(expected)
    assert part.height == 2
    # a signal reads its own tick, not the next one: changing the entry tick's price leaves it in place
    changed = full.with_columns(Price=pl.when(pl.col("Index") == 2).then(102.0).otherwise(pl.col("Price")))

    def early(tape):
        return find_vwap_band_touch_signals(tape).filter(pl.col("signal_index") <= 1)["signal_index"].to_list()

    assert early(changed) == early(full) == [1]
