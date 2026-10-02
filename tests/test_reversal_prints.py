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
