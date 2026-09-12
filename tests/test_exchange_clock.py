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


def test_autumn_switch_sorts_on_the_source_instant_not_local_time():
    """Rows fed out of source order are the only way to tell the two sorts apart.

    06:30 and 07:30 UTC both land on 01:30 local, so the Datetime column alone cannot
    discriminate. The payload column is what shows which row came first.
    """
    df = pl.DataFrame({
        "Datetime": [
            datetime(2025, 11, 2, 7, 30),   # 01:30 CST, the later instant, fed first
            datetime(2025, 11, 2, 5, 30),   # 00:30 CDT
            datetime(2025, 11, 2, 6, 30),   # 01:30 CDT, the earlier of the two 01:30 rows
        ],
        "Price": [3.0, 1.0, 2.0],
    })
    out = apply_offset_given_dataframe(df, market="CME")
    assert out["Datetime"].to_list() == [
        datetime(2025, 11, 2, 0, 30),
        datetime(2025, 11, 2, 1, 30),
        datetime(2025, 11, 2, 1, 30),
    ]
    # Sorting on the converted column would leave Price 3.0 ahead of 2.0: both rows are
    # 01:30 local, and a stable sort keeps whatever order they were fed in.
    assert out["Price"].to_list() == [1.0, 2.0, 3.0]


def test_eurex_maps_to_berlin():
    out = apply_offset_given_dataframe(frame(datetime(2025, 7, 15, 12, 0)), market="EUREX")
    assert out["Datetime"].to_list() == [datetime(2025, 7, 15, 14, 0)]


def test_unknown_market_raises():
    with pytest.raises(Exception, match="Unknown market"):
        apply_offset_given_dataframe(frame(datetime(2025, 7, 15, 12, 0)), market="NYSE")


def test_missing_datetime_column_raises():
    with pytest.raises(Exception, match="Datetime"):
        apply_offset_given_dataframe(pl.DataFrame({"Price": [1.0]}), market="CME")
