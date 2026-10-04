"""Tests for absorption stalls: direction first, then the arrival, the absorbed volume, the three
endings, the mirror, and the event-study tools."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from orderflow.market import absorption_inversion as ai
from orderflow.market.absorption_inversion import STALL_COLUMNS, find_absorption_stalls

TICK = 0.25


def frame(rows):
    """rows: dicts with ``t`` ("HH:MM:SS[.ffffff]"), ``price`` and ``tt``; the rest is quiet.

    A buy (``tt`` 2) trades at the ask and a sell (``tt`` 1) at the bid, the other quote one tick
    away. 1 lot, 10 lots shown on each side, RTH, 2025-09-15.
    """
    out = []
    for r in rows:
        day = r.get("day", "2025-09-15")
        price, tt = float(r["price"]), int(r["tt"])
        out.append({
            "Datetime": datetime.fromisoformat(f"{day}T{r['t']}"), "Date": day,
            "SessionType": r.get("session", "RTH"), "Price": price,
            "Volume": int(r.get("vol", 1)), "TradeType": tt,
            "AskPrice": float(r.get("ask", price if tt == 2 else price + TICK)),
            "BidPrice": float(r.get("bid", price if tt == 1 else price - TICK)),
            "AskSize": int(r.get("ask_size", 10)), "BidSize": int(r.get("bid_size", 10)),
        })
    return (pl.DataFrame(out).sort("Datetime", maintain_order=True).with_row_index("Index")
            .with_columns(pl.col("Index").cast(pl.Int64)))


def mirror(rows):
    """The same tape seen from the other side: prices reflected around 100, sides swapped."""
    out = []
    for r in rows:
        m = {k: v for k, v in r.items() if k not in ("price", "tt", "ask", "bid", "ask_size", "bid_size")}
        m.update(price=200.0 - r["price"], tt=3 - r["tt"])
        if "ask" in r:
            m["bid"] = 200.0 - r["ask"]
        if "bid" in r:
            m["ask"] = 200.0 - r["bid"]
        if "ask_size" in r:
            m["bid_size"] = r["ask_size"]
        if "bid_size" in r:
            m["ask_size"] = r["bid_size"]
        out.append(m)
    return out


def find(ticks, break_ticks=2, max_wait_s=60.0, push_window_s=60.0):
    return find_absorption_stalls(ticks, tick_size=TICK, push_window_s=push_window_s,
                                  break_ticks=break_ticks, max_wait_s=max_wait_s)


def buyers(out):
    return out.filter(pl.col("side") == 1)


def buy_stall(*then, day="2025-09-15"):
    """Sellers trade 100.00, then buyers lift 100.50 three times: 5 + 30 + 15 lots, 20 shown.

    The sell at 100.25 in between is one tick back, not a break-back. ``then`` is what follows.
    """
    rows = [
        {"t": "10:00:00", "price": 100.00, "tt": 1},
        {"t": "10:00:30", "price": 100.00, "tt": 1},
        {"t": "10:01:10", "price": 100.50, "tt": 2, "vol": 5, "ask_size": 20},  # the arrival
        {"t": "10:01:11", "price": 100.50, "tt": 2, "vol": 30},
        {"t": "10:01:12", "price": 100.25, "tt": 1, "vol": 3},
        {"t": "10:01:13", "price": 100.50, "tt": 2, "vol": 15},
        *then,
    ]
    return [dict(row, day=day) for row in rows]


#: An aggressive sell two ticks below the stall, and a quiet tick after it.
BREAK_BACK = {"t": "10:01:14", "price": 100.00, "tt": 1, "vol": 4}


LATER = {"t": "10:01:15", "price": 100.00, "tt": 1}


def test_buyers_trapped_point_short_and_sellers_trapped_point_long():
    sellers = mirror(buy_stall(BREAK_BACK, LATER, day="2025-09-16"))
    out = find(frame(buy_stall(BREAK_BACK, LATER) + sellers))
    assert out["side"].to_list() == [1, -1]
    assert out["ending"].to_list() == ["break_back", "break_back"]
    assert out["TradeType"].to_list() == [1, 2]


def test_a_trade_at_the_high_of_the_window_is_not_an_arrival():
    rows = [{"t": "10:00:00", "price": 100.50, "tt": 1},
            {"t": "10:00:30", "price": 100.50, "tt": 1},
            {"t": "10:01:10", "price": 100.50, "tt": 2, "vol": 50},
            {"t": "10:01:14", "price": 100.00, "tt": 1},
            {"t": "10:01:15", "price": 100.00, "tt": 1}]
    assert buyers(find(frame(rows))).height == 0


def test_a_buy_trade_below_the_stall_is_not_a_break_back():
    low_buy = {"t": "10:01:14", "price": 100.00, "tt": 2}  # the offer dropped; nobody sold
    out = buyers(find(frame(buy_stall(low_buy))))
    assert "break_back" not in out["ending"].to_list()


def test_break_back_needs_break_ticks_ticks():
    deep = {"t": "10:01:16", "price": 99.50, "tt": 1}
    shallow = buyers(find(frame(buy_stall(BREAK_BACK, LATER)), break_ticks=4))
    reached = buyers(find(frame(buy_stall(BREAK_BACK, LATER, deep)), break_ticks=4))
    assert "break_back" not in shallow["ending"].to_list()
    assert reached["ending"].to_list() == ["break_back"]
    assert reached["end_index"].to_list() == [8]


def test_no_arrival_in_the_first_window_of_the_day():
    rows = [{"t": "10:00:00", "price": 100.00, "tt": 1},
            {"t": "10:00:20", "price": 100.50, "tt": 2, "vol": 50},  # a new high, 20 s into the day
            {"t": "10:00:21", "price": 100.00, "tt": 1},
            {"t": "10:00:22", "price": 100.00, "tt": 1}]
    assert find(frame(rows)).height == 0


def test_the_first_trade_after_a_hole_in_the_data_is_not_an_arrival():
    rows = [{"t": "10:00:00", "price": 100.00, "tt": 1},
            {"t": "10:05:00", "price": 100.50, "tt": 2, "vol": 50},  # nothing for five minutes
            {"t": "10:05:01", "price": 100.00, "tt": 1},
            {"t": "10:05:02", "price": 100.00, "tt": 1}]
    assert buyers(find(frame(rows))).height == 0


def test_absorbed_volume_counts_only_aggressive_buys_at_the_stall_price():
    low_buy = {"t": "10:01:13.500", "price": 100.25, "tt": 2, "vol": 7}  # a buy, below the stall
    out = buyers(find(frame(buy_stall(low_buy, BREAK_BACK, LATER))))
    # 5 + 30 + 15: not the 3-lot sell, not the 7-lot buy below, not the 4-lot break-back
    assert out["absorbed_volume"].to_list() == [50]
    assert out["n_trades"].to_list() == [3]
    assert out["price"].to_list() == [100.5]


def test_refill_ratio_is_absorbed_volume_over_the_size_shown_on_arrival():
    out = buyers(find(frame(buy_stall(BREAK_BACK, LATER))))
    assert out["displayed_on_arrival"].to_list() == [20]
    assert out["refill_ratio"].to_list() == [2.5]


def test_refill_ratio_is_null_when_no_size_was_shown_at_the_stall_price():
    stale = buy_stall(BREAK_BACK, LATER)
    stale[2]["ask"] = 100.75  # the quote sat one tick above the trade
    empty = buy_stall(BREAK_BACK, LATER)
    empty[2]["ask_size"] = 0
    stale_out, empty_out = buyers(find(frame(stale))), buyers(find(frame(empty)))
    assert stale_out["displayed_on_arrival"].to_list() == [None]
    assert stale_out["refill_ratio"].to_list() == [None]
    assert stale_out["absorbed_volume"].to_list() == [50]
    assert empty_out["displayed_on_arrival"].to_list() == [0]
    assert empty_out["refill_ratio"].to_list() == [None]


def test_a_trade_above_the_stall_eats_it_and_starts_the_next_one():
    higher = {"t": "10:01:14", "price": 100.75, "tt": 2, "vol": 9, "ask_size": 3}
    back = {"t": "10:01:14.500", "price": 100.25, "tt": 1}
    out = buyers(find(frame(buy_stall(higher, back, LATER))))
    assert out["ending"].to_list() == ["eaten", "break_back"]
    assert out["price"].to_list() == [100.5, 100.75]
    assert out["arrival_index"].to_list() == [2, 6]
    assert out["end_index"].to_list() == [6, 7]
    assert out["TradeType"].to_list() == [None, 1]


def test_a_stall_times_out_at_the_first_tick_later_than_max_wait():
    late = {"t": "10:02:11", "price": 100.25, "tt": 1}  # 61 s after the arrival
    sell = {"t": "10:02:12", "price": 100.00, "tt": 1}  # would have been a break-back
    out = buyers(find(frame(buy_stall(late, sell, {"t": "10:02:13", "price": 100.00, "tt": 1}))))
    assert out["ending"].to_list() == ["timeout"]
    assert out["end_index"].to_list() == [6]
    assert out["duration_s"].to_list() == [61.0]
    assert out["TradeType"].to_list() == [None]
    on_time = {"t": "10:02:10", "price": 100.00, "tt": 1}  # exactly 60 s: not later
    assert buyers(find(frame(buy_stall(on_time, late))))["ending"].to_list() == ["break_back"]


def test_a_stall_still_open_at_the_last_tick_of_the_day_times_out_there():
    next_day = [{"day": "2025-09-16", "t": "10:00:00", "price": 99.00, "tt": 1},
                {"day": "2025-09-16", "t": "10:00:10", "price": 101.00, "tt": 2}]
    out = buyers(find(frame(buy_stall() + next_day), max_wait_s=300.0))
    assert out["ending"].to_list() == ["timeout"]
    assert out["Date"].to_list() == ["2025-09-15"]
    assert out["end_index"].to_list() == [5]  # the day's last tick, never a tick of the next day
    assert out["absorbed_volume"].to_list() == [35]  # the ending tick is not counted


def test_an_arrival_on_the_last_tick_of_the_day_is_closed_where_it_started():
    rows = [{"t": "10:00:00", "price": 100.00, "tt": 1},
            {"t": "10:00:30", "price": 100.00, "tt": 1},
            {"t": "10:01:10", "price": 100.50, "tt": 2, "vol": 5}]
    out = buyers(find(frame(rows)))
    assert out["ending"].to_list() == ["timeout"]
    assert out["arrival_index"].to_list() == out["end_index"].to_list() == [2]
    assert out["absorbed_volume"].to_list() == [5]


def test_the_break_back_tick_alone_turns_a_timeout_into_an_inversion():
    quiet = [{"t": "10:01:20", "price": 100.25, "tt": 1}, {"t": "10:03:00", "price": 100.25, "tt": 1}]
    without = buyers(find(frame(buy_stall(*quiet))))
    with_break = buyers(find(frame(buy_stall(BREAK_BACK, *quiet))))
    assert without["ending"].to_list() == ["timeout"]
    assert with_break["ending"].to_list() == ["break_back"]
    assert without["arrival_index"].to_list() == with_break["arrival_index"].to_list() == [2]
