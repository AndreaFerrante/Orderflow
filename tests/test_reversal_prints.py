"""Tests for reversal prints: side sign first, then each trigger condition, entry, causality."""

from datetime import datetime

import polars as pl

from orderflow.market.reversal_prints import find_reversal_prints

TICK = 0.25


def frame(rows, levels=20):
    """rows: dicts with ``t`` ("HH:MM:SS[.fff]") and ``mid``; everything else has a quiet default.

    Defaults describe a tick that can never trigger: 1 lot, VWAP 100 with a standard deviation of
    1, 10 lots at the quote, 500 lots on every ladder level.
    """
    out = []
    for r in rows:
        day = r.get("day", "2025-09-15")
        mid = float(r["mid"])
        vwap = float(r.get("vwap", 100.0))
        row = {
            "Datetime": datetime.fromisoformat(f"{day}T{r['t']}"), "Date": day,
            "SessionType": r.get("session", "RTH"), "Price": mid,
            "AskPrice": mid + 0.125, "BidPrice": mid - 0.125,
            "Volume": int(r.get("vol", 1)), "TradeType": int(r.get("tt", 2)),
            "AskSize": int(r.get("ask_size", 10)), "BidSize": int(r.get("bid_size", 10)),
            "vwap": vwap, "vwap_sd1_top": vwap + float(r.get("sd", 1.0)),
        }
        for i in range(levels):
            row[f"AskDOM_{i}"] = int(r.get(f"ask_dom_{i}", r.get("book", 500)))
            row[f"BidDOM_{i}"] = int(r.get(f"bid_dom_{i}", r.get("book", 500)))
        out.append(row)
    return (pl.DataFrame(out).sort("Datetime").with_row_index("Index")
            .with_columns(pl.col("Index").cast(pl.Int64)))


def find(ticks, size=100, **kwargs):
    return find_reversal_prints(ticks, tick_size=TICK, min_print_size=size, **kwargs)


def buy_print(**overrides):
    """A 100-lot buy 3 SD below VWAP after price fell one point in 70 s, then two later ticks."""
    print_row = {"t": "10:01:10", "mid": 97.0, "vol": 100, "tt": 2}
    print_row.update(overrides)
    return [{"t": "10:00:00", "mid": 98.0}, print_row,
            {"t": "10:01:11", "mid": 97.25}, {"t": "10:01:20", "mid": 97.5}]


def sell_print(day):
    """The mirror: a 100-lot sell 3 SD above VWAP after price rose one point."""
    return [{"day": day, "t": "10:00:00", "mid": 102.0},
            {"day": day, "t": "10:01:10", "mid": 103.0, "vol": 100, "tt": 1},
            {"day": day, "t": "10:01:11", "mid": 102.75}, {"day": day, "t": "10:01:20", "mid": 102.5}]


def test_buy_print_is_side_plus_one_and_sell_print_is_side_minus_one():
    out = find(frame(buy_print() + sell_print("2025-09-16")))
    assert out["side"].to_list() == [1, -1]


def test_print_below_the_minimum_size_is_ignored():
    assert find(frame(buy_print(vol=99))).height == 0
    assert find(frame(buy_print(vol=100))).height == 1


def test_trade_types_other_than_1_and_2_are_ignored():
    # A sell-shaped setup: if 3 were read as "not a buy", it would trigger as a sell.
    rows = sell_print("2025-09-15")
    rows[1]["tt"] = 3
    assert find(frame(rows)).height == 0


def test_eth_print_never_triggers():
    assert find(frame(buy_print(session="ETH"))).height == 0


def test_print_with_the_prior_move_is_rejected():
    rows = buy_print()
    rows[0]["mid"] = 96.0  # price ROSE into the buy print
    assert find(frame(rows)).height == 0


def test_reference_tick_is_strictly_older_than_the_lookback():
    # The print is at 10:01:04. A tick exactly 60 s earlier (10:00:04, mid 99) must NOT be the
    # reference; the one before it (10:00:00, mid 96) is, and against it price has risen.
    rows = [{"t": "10:00:00", "mid": 96.0}, {"t": "10:00:04", "mid": 99.0},
            {"t": "10:01:04", "mid": 97.0, "vol": 100, "tt": 2},
            {"t": "10:01:05", "mid": 97.0}, {"t": "10:01:10", "mid": 97.0}]
    assert find(frame(rows)).height == 0


def test_first_minute_print_uses_the_first_tick_of_its_own_day():
    # Yesterday closed at 90. Against that tick price has risen and the print would be rejected;
    # against today's first tick (98) it has fallen.
    rows = [{"day": "2025-09-12", "t": "15:30:00", "mid": 90.0},
            {"t": "08:30:00", "mid": 98.0}, {"t": "08:30:30", "mid": 97.0, "vol": 100, "tt": 2},
            {"t": "08:30:31", "mid": 97.0}, {"t": "08:30:40", "mid": 97.0}]
    out = find(frame(rows))
    assert out.height == 1
    assert out["pre_move_ticks"].to_list() == [-4.0]


def test_vwap_variant_needs_two_standard_deviations():
    near = buy_print(mid=98.1)   # 1.9 SD below VWAP
    near[0]["mid"] = 99.0
    at = buy_print(mid=98.0)     # exactly 2 SD below
    at[0]["mid"] = 99.0
    assert find(frame(near)).height == 0
    assert find(frame(at))["variant_vwap"].to_list() == [True]


def test_vwap_variant_rejects_a_print_that_pushes_away_from_vwap():
    # A sell 3 SD BELOW VWAP after price rose: against the move, stretched, but pushing away.
    rows = [{"t": "10:00:00", "mid": 96.0}, {"t": "10:01:10", "mid": 97.0, "vol": 100, "tt": 1},
            {"t": "10:01:11", "mid": 97.0}, {"t": "10:01:20", "mid": 97.0}]
    assert find(frame(rows)).height == 0


def test_vwap_variant_is_off_when_the_band_has_no_width():
    # vwap_sd1_top == vwap on the first ticks of a session: the distance is undefined.
    assert find(frame(buy_print(sd=0.0))).height == 0


def wall_print(**overrides):
    """A 150-lot buy AT VWAP (so never the VWAP variant) into a 120-lot ask, book maximum 119."""
    spec = {"mid": 100.0, "vol": 150, "ask_size": 120, "book": 119}
    spec.update(overrides)
    rows = buy_print(**spec)
    rows[0]["mid"] = 101.0
    return rows


def test_book_variant_needs_the_execution_level_to_be_the_largest_size():
    out = find(frame(wall_print()))
    assert out.height == 1
    assert out["variant_book"].to_list() == [True]
    assert out["variant_vwap"].to_list() == [False]
    assert find(frame(wall_print(book=121))).height == 0


def test_book_variant_needs_the_print_to_take_the_whole_level():
    assert find(frame(wall_print(vol=100))).height == 0


def test_book_variant_reads_the_bid_size_for_a_sell_print():
    rows = [{"t": "10:00:00", "mid": 99.0},
            {"t": "10:01:10", "mid": 100.0, "vol": 150, "tt": 1, "bid_size": 120, "ask_size": 5, "book": 119},
            {"t": "10:01:11", "mid": 100.0}, {"t": "10:01:20", "mid": 100.0}]
    out = find(frame(rows))
    assert out["side"].to_list() == [-1]
    assert out["variant_book"].to_list() == [True]


def test_ladder_levels_beyond_book_levels_are_not_read():
    rows = wall_print(ask_dom_25=10_000)
    assert find(frame(rows, levels=30)).height == 1
