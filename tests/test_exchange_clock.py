"""The clock: source-time ticks converted to exchange local time, daylight saving included."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from orderflow.market.utilities._volume_factory import (
    apply_offset_given_dataframe,
    daily_maintenance_halt_report,
    verify_daily_maintenance_halt_anchor,
    verify_weekly_reopen_anchor,
)


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


def test_weekly_reopen_anchor_passes_across_a_dst_switch():
    """CME/CBOT Globex reopens Sunday 17:00 CT. In UTC that's 22:00 (CDT) or 23:00 (CST) -
    two Sundays either side of a switch must both check out."""
    df = frame(
        datetime(2024, 12, 22, 23, 0),  # Sunday, CST reopen
        datetime(2025, 3, 16, 22, 0),   # Sunday, CDT reopen
    )
    verify_weekly_reopen_anchor(df, market="CME")  # must not raise


def test_weekly_reopen_anchor_fails_on_the_wrong_source_timezone():
    """Data actually recorded in America/Chicago, mislabeled as UTC: the Sunday reopen row
    reads 17:00 raw, not 22:00/23:00, and the first anchor alone must catch it."""
    df = frame(datetime(2024, 12, 22, 17, 0))
    with pytest.raises(Exception, match="anchor failed"):
        verify_weekly_reopen_anchor(df, market="CME", source_timezone="UTC")


def test_weekly_reopen_anchor_catches_a_bad_last_anchor_even_when_first_is_correct():
    """The shape of the original bug: correct at one end of a file, wrong at the other."""
    df = frame(
        datetime(2024, 12, 22, 23, 0),   # correct CST reopen
        datetime(2025, 3, 16, 23, 0),    # should be 22:00 (CDT) - off by one hour
    )
    with pytest.raises(Exception, match="last Sunday"):
        verify_weekly_reopen_anchor(df, market="CME")


def test_weekly_reopen_anchor_unknown_market_raises():
    with pytest.raises(Exception, match="No validated weekly-reopen anchor"):
        verify_weekly_reopen_anchor(frame(datetime(2025, 7, 13, 12, 0)), market="EUREX")


def test_weekly_reopen_anchor_no_sunday_rows_raises():
    with pytest.raises(Exception, match="Sunday"):
        verify_weekly_reopen_anchor(frame(datetime(2025, 7, 14, 12, 0)), market="CME")  # a Monday


def test_weekly_reopen_anchor_missing_datetime_column_raises():
    with pytest.raises(Exception, match="Datetime"):
        verify_weekly_reopen_anchor(pl.DataFrame({"Price": [1.0]}), market="CME")


def gapped_day(day, resume_hour, gap_minutes=61.0, filler_hour=19):
    """Ticks for one Monday-Thursday date, 5-minute-spaced on both sides of one gap ending exactly
    on `resume_hour`:00 - so the gap under test is unambiguously the largest (as in real tick data),
    and the resumption lands sharp on the hour (as the real daily reopen does)."""
    resume = datetime(day.year, day.month, day.day, resume_hour, 0)
    last_before = resume - timedelta(minutes=gap_minutes)
    end = datetime(day.year, day.month, day.day, 23, 30)

    ticks, t = [], datetime(day.year, day.month, day.day, filler_hour, 0)
    while t <= last_before:
        ticks.append(t)
        t += timedelta(minutes=5)
    t = resume
    while t <= end:
        ticks.append(t)
        t += timedelta(minutes=5)
    return ticks


def test_daily_halt_anchor_passes_on_both_sides_of_a_dst_switch():
    # 2025-01-13 Monday, CST (17:00 CT = 23:00 UTC). 2025-07-14 Monday, CDT (17:00 CT = 22:00 UTC).
    ticks = gapped_day(datetime(2025, 1, 13), resume_hour=23) + gapped_day(
        datetime(2025, 7, 14), resume_hour=22
    )
    verify_daily_maintenance_halt_anchor(frame(*ticks), market="CME")  # must not raise


def test_daily_halt_anchor_catches_one_bad_day_among_good_ones():
    good = gapped_day(datetime(2025, 1, 13), resume_hour=23)
    bad = gapped_day(datetime(2025, 1, 20), resume_hour=22)  # off by one hour for a CST Monday
    with pytest.raises(Exception, match="2025-01-20"):
        verify_daily_maintenance_halt_anchor(frame(*(good + bad)), market="CME")


def test_daily_halt_anchor_rejects_a_gap_too_short_to_be_the_halt():
    ticks = gapped_day(datetime(2025, 1, 13), resume_hour=23, gap_minutes=10.0)
    with pytest.raises(Exception, match="2025-01-13"):
        verify_daily_maintenance_halt_anchor(frame(*ticks), market="CME")


def test_daily_halt_anchor_skips_friday_and_sunday():
    """Friday's next gap is the weekend close; Sunday has no same-day halt. Neither is scanned,
    so a frame containing only those two weekdays has no Monday-Thursday rows to check."""
    with pytest.raises(Exception, match="No Monday-Thursday rows"):
        verify_daily_maintenance_halt_anchor(
            frame(datetime(2025, 1, 12, 20, 0), datetime(2025, 1, 17, 20, 0)), market="CME"
        )


def test_daily_halt_anchor_unknown_market_raises():
    with pytest.raises(Exception, match="No validated daily-halt anchor"):
        verify_daily_maintenance_halt_anchor(frame(datetime(2025, 1, 13, 22, 0)), market="EUREX")


def test_daily_halt_anchor_missing_datetime_column_raises():
    with pytest.raises(Exception, match="Datetime"):
        verify_daily_maintenance_halt_anchor(pl.DataFrame({"Price": [1.0]}), market="CME")


def test_daily_halt_anchor_checks_an_already_local_frame_by_identity():
    """source_timezone can equal the exchange zone itself - the case of checking a Datetime column
    that already claims to be exchange-local. No DST shift applies, so the resumption is always at
    local hour 17, winter or summer."""
    ticks = gapped_day(datetime(2025, 1, 13), resume_hour=17, filler_hour=14) + gapped_day(
        datetime(2025, 7, 14), resume_hour=17, filler_hour=14
    )
    verify_daily_maintenance_halt_anchor(
        frame(*ticks), market="CME", source_timezone="America/Chicago"
    )  # must not raise


def test_daily_halt_anchor_skips_a_known_holiday_exception():
    """2025-06-19 (Juneteenth) is Thursday - normally checked - but is a seeded exception, so a
    completely absent halt there must not raise."""
    ticks = [datetime(2025, 6, 19, h, 0) for h in range(19, 24)]  # flat, no gap at all
    verify_daily_maintenance_halt_anchor(frame(*ticks), market="CME")  # must not raise
    report = daily_maintenance_halt_report(frame(*ticks), market="CME")
    assert report.height == 0  # skipped, not force-passed


def test_daily_halt_anchor_exception_does_not_mask_a_different_bad_day():
    """The exception is exact-date. A genuinely wrong day right next to it must still fail."""
    holiday = [datetime(2025, 6, 19, h, 0) for h in range(19, 24)]  # no gap, but excepted
    bad = gapped_day(datetime(2025, 6, 16), resume_hour=21)  # off by one hour for a CDT Monday
    with pytest.raises(Exception, match="2025-06-16"):
        verify_daily_maintenance_halt_anchor(frame(*(holiday + bad)), market="CME")


def test_daily_halt_report_returns_one_row_per_good_day():
    ticks = gapped_day(datetime(2025, 1, 13), resume_hour=23) + gapped_day(
        datetime(2025, 7, 14), resume_hour=22
    )
    report = daily_maintenance_halt_report(frame(*ticks), market="CME")
    assert report.height == 2
    assert report["ok"].to_list() == [True, True]


from orderflow.market.utilities._volume_factory import get_tickers_in_folder_mem_optim


def test_fresh_ingestion_gate_passes_on_correct_clock(tmp_path):
    fixture = tmp_path / "MES_fixture.txt"
    fixture.write_text(
        "Date;Time;Price;Volume\n"
        "2025-01-12;23:00:00.000000;5002.0;2\n"   # Sunday reopen, CST
        "2025-01-15;14:30:00.000000;5000.0;3\n"   # RTH
        "2025-01-15;22:00:00.000000;5001.0;4\n"   # pre-halt, local 16:00
        "2025-01-15;23:00:00.000000;5003.0;1\n"   # halt resumption, local 17:00
    )
    out = get_tickers_in_folder_mem_optim(
        path=str(tmp_path), single_file=fixture.name, market="CME",
        cols=["Date", "Time", "Price", "Volume"], ticker="MES",
    )
    assert out.height == 4


def test_fresh_ingestion_gate_fails_on_wrong_clock(tmp_path):
    fixture = tmp_path / "MES_fixture.txt"
    fixture.write_text(
        "Date;Time;Price;Volume\n"
        "2025-01-12;17:00:00.000000;5002.0;2\n"   # wrong: raw 17:00, UTC predicts 23:00
        "2025-01-15;14:30:00.000000;5000.0;3\n"
        "2025-01-15;22:00:00.000000;5001.0;4\n"
        "2025-01-15;23:00:00.000000;5003.0;1\n"
    )
    with pytest.raises(Exception, match="anchor failed"):
        get_tickers_in_folder_mem_optim(
            path=str(tmp_path), single_file=fixture.name, market="CME",
            cols=["Date", "Time", "Price", "Volume"], ticker="MES",
        )


def test_fresh_ingestion_gate_fails_on_wrong_daily_halt_alone(tmp_path):
    """Correct Sunday reopen (weekday anchor passes), but wrong halt-resumption on Wednesday.
    This proves the daily-halt anchor is independently load-bearing: a fixture where the
    Sunday reopen is correctly at 23:00 UTC, but the Monday-Thursday halt-resumption is
    off by one hour (22:00 instead of 23:00 UTC). The weekly check won't catch this because
    it only verifies Sunday; only the daily check catches the Wednesday error."""
    fixture = tmp_path / "MES_fixture.txt"
    fixture.write_text(
        "Date;Time;Price;Volume\n"
        "2025-01-12;23:00:00.000000;5002.0;2\n"   # Sunday reopen, CORRECT CST (17:00 CT)
        "2025-01-15;14:30:00.000000;5000.0;3\n"   # RTH
        "2025-01-15;21:00:00.000000;5001.0;4\n"   # pre-halt, 15:00 CT (in daily window)
        "2025-01-15;22:00:00.000000;5003.0;1\n"   # halt resumption, WRONG: should be 23:00 for CST
    )
    with pytest.raises(Exception, match="anchor failed"):
        get_tickers_in_folder_mem_optim(
            path=str(tmp_path), single_file=fixture.name, market="CME",
            cols=["Date", "Time", "Price", "Volume"], ticker="MES",
        )
