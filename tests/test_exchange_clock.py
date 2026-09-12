"""The clock: source-time ticks converted to exchange local time, daylight saving included."""

from datetime import datetime

import polars as pl
import pytest

from orderflow.market.utilities._volume_factory import apply_offset_given_dataframe


def frame(*naive_utc):
    return pl.DataFrame({
        "Datetime": list(naive_utc),
        "Price": [1.0] * len(naive_utc),
    })


def test_winter_utc_is_minus_six():
    out = apply_offset_given_dataframe(frame(datetime(2025, 1, 15, 22, 0)), market="CME")
    assert out["Datetime"].to_list() == [datetime(2025, 1, 15, 16, 0)]


def test_summer_utc_is_minus_five():
    out = apply_offset_given_dataframe(frame(datetime(2025, 7, 15, 22, 0)), market="CME")
    assert out["Datetime"].to_list() == [datetime(2025, 7, 15, 17, 0)]


def test_spring_switch_inside_one_frame():
    # Daylight saving starts at 02:00 CST on 2025-03-09, so 08:00 UTC is the first instant on CDT.
    out = apply_offset_given_dataframe(
        frame(datetime(2025, 3, 9, 6, 0), datetime(2025, 3, 9, 8, 0)), market="CME"
    )
    assert out["Datetime"].to_list() == [datetime(2025, 3, 9, 0, 0), datetime(2025, 3, 9, 3, 0)]


def test_autumn_switch_inside_one_frame():
    # 2025-11-02 07:00 UTC is the first instant back on CST.
    out = apply_offset_given_dataframe(
        frame(datetime(2025, 11, 2, 6, 0), datetime(2025, 11, 2, 8, 0)), market="CME"
    )
    assert out["Datetime"].to_list() == [datetime(2025, 11, 2, 1, 0), datetime(2025, 11, 2, 2, 0)]


def test_autumn_switch_keeps_source_order():
    """Local time repeats 01:00-02:00; the rows must not be reordered into it."""
    out = apply_offset_given_dataframe(
        frame(
            datetime(2025, 11, 2, 5, 30),   # 00:30 CDT
            datetime(2025, 11, 2, 6, 30),   # 01:30 CDT, first pass through the repeated hour
            datetime(2025, 11, 2, 7, 30),   # 01:30 CST, second pass through the same local hour
        ),
        market="CME",
    )
    got = out["Datetime"].to_list()
    assert got == [datetime(2025, 11, 2, 0, 30), datetime(2025, 11, 2, 1, 30), datetime(2025, 11, 2, 1, 30)]


def test_eurex_maps_to_berlin():
    out = apply_offset_given_dataframe(frame(datetime(2025, 7, 15, 12, 0)), market="EUREX")
    assert out["Datetime"].to_list() == [datetime(2025, 7, 15, 14, 0)]


def test_unknown_market_raises():
    with pytest.raises(Exception, match="Unknown market"):
        apply_offset_given_dataframe(frame(datetime(2025, 7, 15, 12, 0)), market="NYSE")


def test_missing_datetime_column_raises():
    with pytest.raises(Exception, match="Datetime"):
        apply_offset_given_dataframe(pl.DataFrame({"Price": [1.0]}), market="CME")
