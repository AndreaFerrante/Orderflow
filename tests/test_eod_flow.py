"""Tests of the end-of-day CVD fade signal: synthetic tapes only."""

from datetime import datetime

import polars as pl

from orderflow.market.eod_flow import EVENT_COLUMNS, find_eod_cvd_fade_signals

ASK, BID = 2, 1  # TradeType: 2 = ask trade = buy aggression, 1 = bid trade = sell aggression


def tape(rows, date="2025-01-02", first_index=100):
    """A row is ``("HH:MM:SS", price, contracts, trade_type)``, optionally ``+ (session,)``."""
    return pl.DataFrame({
        "Index": [first_index + i for i in range(len(rows))],
        "Date": [date] * len(rows),
        "Datetime": [datetime.fromisoformat(f"{date} {row[0]}") for row in rows],
        "SessionType": [row[4] if len(row) > 4 else "RTH" for row in rows],
        "Price": [float(row[1]) for row in rows],
        "Volume": [row[2] for row in rows],
        "TradeType": [row[3] for row in rows],
    })


BUY_DAY = [("09:00:00", 5000.0, 50, ASK), ("12:00:00", 5001.0, 10, BID), ("14:45:00", 5002.0, 5, ASK),
           ("14:45:01", 5002.25, 1, BID), ("14:59:59", 5001.0, 1, ASK)]
SELL_DAY = [("09:00:00", 5000.0, 50, BID), ("12:00:00", 4999.0, 10, ASK), ("14:45:00", 4998.0, 5, BID),
            ("14:45:01", 4997.75, 1, ASK), ("14:59:59", 4999.0, 1, BID)]


def test_a_day_of_net_buying_is_traded_short():
    assert find_eod_cvd_fade_signals(tape(BUY_DAY))["side"].to_list() == [-1]


def test_a_day_of_net_selling_is_traded_long():
    assert find_eod_cvd_fade_signals(tape(SELL_DAY))["side"].to_list() == [1]


def test_the_day_cvd_is_buy_minus_sell_volume_up_to_the_signal_tick():
    assert find_eod_cvd_fade_signals(tape(BUY_DAY))["day_cvd"].to_list() == [50 - 10 + 5]


def test_the_signal_is_the_last_tick_at_the_signal_time_and_the_entry_is_the_next_tick():
    row = find_eod_cvd_fade_signals(tape(BUY_DAY)).row(0, named=True)
    assert (row["signal_index"], row["entry_index"], row["entry_price"]) == (102, 103, 5002.25)
    assert row["signal_datetime"] == datetime(2025, 1, 2, 14, 45, 0)


def test_volume_after_the_signal_time_does_not_change_the_signal():
    late_selling = BUY_DAY[:3] + [("14:45:01", 5002.25, 5000, BID), ("14:59:59", 5001.0, 1, ASK)]
    events = find_eod_cvd_fade_signals(tape(late_selling))
    assert (events["side"].to_list(), events["day_cvd"].to_list()) == ([-1], [45])


def test_evening_ticks_are_not_part_of_the_day():
    rows = [("08:00:00", 5000.0, 900, BID, "ETH")] + BUY_DAY
    events = find_eod_cvd_fade_signals(tape(rows))
    assert (events["side"].to_list(), events["day_cvd"].to_list()) == ([-1], [45])


def test_a_day_with_no_tick_between_the_signal_and_the_exit_has_no_signal():
    rows = BUY_DAY[:3] + [("15:00:00", 5001.0, 1, ASK)]
    assert find_eod_cvd_fade_signals(tape(rows)).is_empty()


def test_a_flat_day_has_no_signal():
    rows = [("09:00:00", 5000.0, 10, ASK), ("14:00:00", 5000.0, 10, BID), ("14:45:01", 5000.0, 1, ASK)]
    assert find_eod_cvd_fade_signals(tape(rows)).is_empty()


def test_a_small_day_is_dropped_by_the_floor_on_the_cvd():
    assert find_eod_cvd_fade_signals(tape(BUY_DAY), min_abs_cvd=46).is_empty()
    assert find_eod_cvd_fade_signals(tape(BUY_DAY), min_abs_cvd=45).height == 1


def test_each_day_gets_its_own_signal_and_an_empty_tape_keeps_the_columns():
    two = pl.concat([tape(BUY_DAY), tape(SELL_DAY, date="2025-01-03", first_index=200)])
    events = find_eod_cvd_fade_signals(two)
    assert (events["Date"].to_list(), events["side"].to_list()) == (["2025-01-02", "2025-01-03"], [-1, 1])
    assert find_eod_cvd_fade_signals(tape(BUY_DAY[:1])).columns == EVENT_COLUMNS
