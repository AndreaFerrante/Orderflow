"""Tests for absorption stalls: direction first, then the arrival, the absorbed volume, the three
endings, the mirror, and the event-study tools."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from orderflow.market import absorption_inversion as ai
from orderflow.market.absorption_inversion import (
    STALL_COLUMNS,
    attach_expected_move,
    find_absorption_stalls,
    measure_forward_moves,
)

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


def test_the_sell_side_is_the_mirror_of_the_buy_side():
    higher = {"t": "10:01:14", "price": 100.75, "tt": 2, "vol": 9, "ask_size": 3}
    back = {"t": "10:01:14.500", "price": 100.25, "tt": 1}
    tape = buy_stall(higher, back, LATER)
    up, down = find(frame(tape)), find(frame(mirror(tape)))
    assert up["side"].to_list() == [1, 1]
    expected = up.with_columns((-pl.col("side")).alias("side"),
                               (200.0 - pl.col("price")).alias("price"),
                               (3 - pl.col("TradeType")).alias("TradeType"))
    assert down.equals(expected)


def test_compiled_and_plain_python_scans_agree():
    pytest.importorskip("numba")  # without Numba the plain Python scan is the only one
    assert hasattr(ai._scan_side, "py_func"), "the scan is not compiled"
    rng = np.random.default_rng(7)
    n = 4000
    t = np.cumsum(rng.integers(1, 2_000_000, n)).astype(np.int64)
    px = (400 + np.cumsum(rng.integers(-1, 2, n))).astype(np.int64)
    vol = rng.integers(1, 40, n).astype(np.int64)
    buy = rng.random(n) < 0.5
    shown = rng.integers(0, 60, n).astype(np.int64)
    args = (t, px, vol, buy, ~buy, px, shown, 60_000_000, 2, 5_000_000)  # 60 s window, 5 s wait
    plain = ai._scan_side.py_func(*args)
    compiled = ai._scan_side(*args)
    assert len(compiled[0]) > 50
    assert set(compiled[6].tolist()) == {0, 1, 2}  # every ending occurs
    for a, b in zip(compiled, plain):
        assert np.array_equal(a, b)


def test_a_missing_column_is_refused():
    with pytest.raises(ValueError, match="Missing required columns.*AskSize"):
        find(frame(buy_stall(BREAK_BACK, LATER)).drop("AskSize"))


def test_a_clock_that_goes_backwards_in_index_order_is_refused():
    ticks = frame(buy_stall(BREAK_BACK, LATER)).with_columns(
        pl.when(pl.col("Index") == 4).then(datetime(2025, 9, 15, 10, 0, 45))
        .otherwise(pl.col("Datetime")).alias("Datetime"))
    with pytest.raises(ValueError, match="non-decreasing.*Index 4"):
        find(ticks)


def test_a_null_in_a_required_column_is_refused():
    ticks = frame(buy_stall(BREAK_BACK, LATER)).with_columns(
        pl.when(pl.col("Index") == 3).then(None).otherwise(pl.col("Price")).alias("Price"))
    with pytest.raises(ValueError, match="Null values.*Price"):
        find(ticks)


@pytest.mark.parametrize("bad", [
    {"tick_size": 0}, {"break_ticks": 0}, {"break_ticks": 2.0}, {"max_wait_s": 0}, {"push_window_s": -1.0},
])
def test_parameters_that_cannot_describe_a_stall_are_refused(bad):
    kwargs = {"tick_size": TICK, "push_window_s": 60.0, "break_ticks": 2, "max_wait_s": 60.0}
    kwargs.update(bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        find_absorption_stalls(frame(buy_stall(BREAK_BACK, LATER)), **kwargs)


def test_only_rth_ticks_are_read_and_a_lazy_frame_is_accepted():
    eth_high = {"t": "10:01:12.500", "price": 101.00, "tt": 2, "session": "ETH"}  # would eat the stall
    eth_low = {"t": "10:01:13.500", "price": 99.00, "tt": 1, "session": "ETH"}
    ticks = frame(buy_stall(eth_high, eth_low, BREAK_BACK, LATER))
    out = buyers(find(ticks.lazy()))
    assert out["ending"].to_list() == ["break_back"]
    assert out["absorbed_volume"].to_list() == [50]
    assert out["end_index"].to_list() == [8]


def test_a_finished_stall_does_not_change_when_later_ticks_change():
    # the stall ends on the break-back tick, Index 6; every tick after it differs between the two tapes
    tail = [{"t": "10:01:15", "price": 100.00, "tt": 1}, {"t": "10:03:00", "price": 100.00, "tt": 1}]
    wild = [dict(row, price=250.0, tt=2, vol=999, ask_size=77, bid_size=77) for row in tail]
    before = find(frame(buy_stall(BREAK_BACK, *tail))).filter(pl.col("end_index") <= 6)
    after = find(frame(buy_stall(BREAK_BACK, *wild))).filter(pl.col("end_index") <= 6)
    assert before.height == 1
    assert after.equals(before)


def test_a_tape_with_no_stall_gives_an_empty_frame_with_every_column():
    quiet = find(frame([{"t": "10:00:00", "price": 100.0, "tt": 1}, {"t": "10:00:01", "price": 100.0, "tt": 1}]))
    assert quiet.height == 0
    assert quiet.columns == STALL_COLUMNS
    full = find(frame(buy_stall(BREAK_BACK, LATER)))
    assert quiet.schema == full.schema
    assert pl.concat([quiet, full]).height == full.height
    no_rth = find(frame([{"t": "10:00:00", "price": 100.0, "tt": 1, "session": "ETH"}]))
    assert no_rth.height == 0
    assert no_rth.schema == full.schema


def test_a_boolean_is_not_a_size():
    with pytest.raises(ValueError, match="tick_size"):
        find_absorption_stalls(frame(buy_stall(BREAK_BACK, LATER)), tick_size=True, push_window_s=60.0,
                               break_ticks=2, max_wait_s=60.0)


def test_a_null_in_a_column_the_scan_does_not_read_is_ignored():
    ticks = frame(buy_stall(BREAK_BACK, LATER)).with_columns(pl.lit(None, dtype=pl.Float64).alias("LVN"))
    assert buyers(find(ticks))["ending"].to_list() == ["break_back"]


def test_ticks_handed_over_out_of_index_order_give_the_same_stalls():
    ticks = frame(buy_stall(BREAK_BACK, LATER))
    out = find(ticks.reverse())
    assert out.height == 1
    assert out.equals(find(ticks))


def test_a_price_lands_on_its_own_tick_when_floats_cannot_hold_the_grid_exactly():
    rows = [{"t": "10:00:00", "price": 100.0, "tt": 1},
            {"t": "10:00:30", "price": 100.0, "tt": 1},
            {"t": "10:01:10", "price": 100.3, "tt": 2, "vol": 5},  # 100.3 / 0.1 is 1002.9999999999999
            {"t": "10:01:14", "price": 100.1, "tt": 1},            # two ticks of 0.1 below
            {"t": "10:01:15", "price": 100.1, "tt": 1}]
    out = buyers(find_absorption_stalls(frame(rows), tick_size=0.1, push_window_s=60.0, break_ticks=2,
                                        max_wait_s=60.0))
    assert out["ending"].to_list() == ["break_back"]
    assert out["price"].to_list() == pytest.approx([100.3])


def test_the_window_reaches_back_exactly_push_window_s():
    at_the_high = [{"t": "10:00:00", "price": 100.50, "tt": 1},  # 60 s old at the buy below: still the high
                   {"t": "10:00:30", "price": 100.00, "tt": 1},
                   {"t": "10:01:00", "price": 100.50, "tt": 2, "vol": 50},
                   {"t": "10:01:01", "price": 100.00, "tt": 1},
                   {"t": "10:01:02", "price": 100.00, "tt": 1}]
    above = [dict(row, price=100.00) if i == 0 else row for i, row in enumerate(at_the_high)]
    assert buyers(find(frame(at_the_high))).height == 0
    # the same buy, 60 s into the day and above everything before it: one full window has passed
    assert buyers(find(frame(above)))["arrival_index"].to_list() == [2]


def test_only_the_trades_older_than_the_window_leave_it():
    rows = [{"t": "10:00:00", "price": 101.00, "tt": 1},  # out of the window at the buy below
            {"t": "10:00:50", "price": 100.75, "tt": 1},  # still in it: the high the buy would have to beat
            {"t": "10:00:55", "price": 100.25, "tt": 1},
            {"t": "10:01:20", "price": 100.50, "tt": 2, "vol": 50},
            {"t": "10:01:21", "price": 100.00, "tt": 1},
            {"t": "10:01:22", "price": 100.00, "tt": 1}]
    assert buyers(find(frame(rows))).height == 0


def test_after_a_hole_in_the_data_the_window_fills_again():
    rows = [{"t": "10:00:00", "price": 100.00, "tt": 1},
            {"t": "10:05:00", "price": 100.00, "tt": 1},  # nothing for five minutes: the window is empty here
            {"t": "10:05:30", "price": 100.50, "tt": 2, "vol": 5},  # ... and holds the tick above again here
            {"t": "10:05:31", "price": 100.00, "tt": 1},
            {"t": "10:05:32", "price": 100.00, "tt": 1}]
    assert buyers(find(frame(rows)))["arrival_index"].to_list() == [2]


def test_a_stall_closed_where_it_started_carries_its_price_its_trade_and_the_size_shown():
    quiet = [{"t": f"10:00:{s:02d}", "price": 100.00, "tt": 1} for s in range(0, 50, 5)]
    rows = quiet + [{"t": "10:01:10", "price": 103.25, "tt": 2, "vol": 37, "ask_size": 13}]  # the day's last tick
    out = buyers(find(frame(rows)))
    assert out["price"].to_list() == [103.25]
    assert out["n_trades"].to_list() == [1]
    assert out["displayed_on_arrival"].to_list() == [13]


def test_a_break_of_one_tick_is_allowed():
    out = buyers(find(frame(buy_stall(BREAK_BACK, LATER)), break_ticks=1))
    assert out["ending"].to_list() == ["break_back"]
    assert out["end_index"].to_list() == [4]  # the sell one tick below the stall


def test_a_stall_on_the_second_day_ends_on_a_tick_of_that_day():
    first_day = [{"t": "10:00:00", "price": 100.00, "tt": 1}, {"t": "10:00:10", "price": 100.00, "tt": 1},
                 {"t": "10:00:20", "price": 100.00, "tt": 1}]
    out = buyers(find(frame(first_day + buy_stall(BREAK_BACK, LATER, day="2025-09-16"))))
    assert out["arrival_index"].to_list() == [5]
    assert out["end_index"].to_list() == [9]
    assert out["end_datetime"].to_list() == [datetime(2025, 9, 16, 10, 1, 14)]
    assert out["duration_s"].to_list() == [4.0]


def test_stalls_of_both_sides_come_in_the_order_of_their_arrival_ticks():
    rows = [{"t": "10:00:00", "price": 100.25, "tt": 1},
            {"t": "10:00:30", "price": 100.25, "tt": 1},
            {"t": "10:01:10", "price": 100.50, "tt": 2},  # buyers arrive; their stall ends last, on its timeout
            {"t": "10:01:20", "price": 100.00, "tt": 1},  # sellers arrive
            {"t": "10:01:25", "price": 99.75, "tt": 1},   # ... are eaten at once, and arrive again
            {"t": "10:02:15", "price": 100.00, "tt": 1},
            {"t": "10:02:30", "price": 100.00, "tt": 1}]
    out = find(frame(rows), break_ticks=4)
    assert out["side"].to_list() == [1, -1, -1]
    assert out["arrival_index"].to_list() == [2, 3, 4]
    assert out["end_index"].to_list() == [5, 4, 6]


def test_one_lot_shown_at_the_stall_price_is_enough_for_a_refill_ratio():
    rows = buy_stall(BREAK_BACK, LATER)
    rows[2]["ask_size"] = 1
    out = buyers(find(frame(rows)))
    assert out["displayed_on_arrival"].to_list() == [1]
    assert out["refill_ratio"].to_list() == [50.0]


def test_every_column_of_a_stall_has_a_fixed_type():
    out = find(frame(buy_stall(BREAK_BACK, LATER)))
    assert out.schema == {
        "Date": pl.String, "side": pl.Int64, "arrival_index": pl.Int64, "arrival_datetime": pl.Datetime("us"),
        "price": pl.Float64, "absorbed_volume": pl.Int64, "displayed_on_arrival": pl.Int64,
        "refill_ratio": pl.Float64, "n_trades": pl.Int64, "duration_s": pl.Float64, "ending": pl.String,
        "end_index": pl.Int64, "end_datetime": pl.Datetime("us"), "TradeType": pl.Int64,
    }


def minute_tape(day="2025-09-15"):
    """Forty ticks, one a minute from 10:00:30, 10 lots each; the bid steps 100.00, 100.50, 100.00 ...

    From 10:35 on everything is 40 ticks higher: the arrival's own minute holds a jump that must
    not be part of its sigma.
    """
    return [{"day": day, "t": f"10:{m:02d}:30", "tt": 1, "vol": 10,
             "price": 100.0 + 0.5 * (m % 2) + (10.0 if m >= 35 else 0.0)} for m in range(40)]


def stalls_at(*arrival, day="2025-09-15"):
    return pl.DataFrame({"Date": [day] * len(arrival), "arrival_index": list(arrival),
                         "absorbed_volume": [90] * len(arrival)},
                        schema={"Date": pl.String, "arrival_index": pl.Int64, "absorbed_volume": pl.Int64})


def test_expected_move_is_sigma_times_the_root_of_the_volume_share():
    out = attach_expected_move(frame(minute_tape()), stalls_at(35), tick_size=TICK)
    sigma = float(np.std([2.0, -2.0] * 15, ddof=1))  # the thirty one-minute changes before 10:35
    assert out["sigma_ticks"].to_list() == pytest.approx([sigma])
    assert out["session_volume_before"].to_list() == [350]  # 35 ticks of 10 lots before the arrival
    assert out["expected_move"].to_list() == pytest.approx([sigma * (90 / 350) ** 0.5])


def test_expected_move_is_null_until_thirty_minutes_of_the_same_day_exist():
    ticks = frame(minute_tape() + minute_tape(day="2025-09-16"))
    stalls = pl.concat([stalls_at(29, 30), stalls_at(45, day="2025-09-16")])  # 45 = sixth tick of day two
    out = attach_expected_move(ticks, stalls, tick_size=TICK)
    assert out["sigma_ticks"].is_null().to_list() == [True, False, True]
    assert out["expected_move"].is_null().to_list() == [True, False, True]


def test_expected_move_reads_nothing_after_the_arrival():
    calm = minute_tape()
    wild = [dict(row, price=row["price"] + 50.0, vol=999) if i > 35 else row for i, row in enumerate(calm)]
    before = attach_expected_move(frame(calm), stalls_at(35), tick_size=TICK)
    after = attach_expected_move(frame(wild), stalls_at(35), tick_size=TICK)
    assert after.equals(before)


def test_expected_move_keeps_the_rows_in_order():
    out = attach_expected_move(frame(minute_tape()), stalls_at(38, 31, 35), tick_size=TICK)
    assert out["arrival_index"].to_list() == [38, 31, 35]
    assert out["session_volume_before"].to_list() == [380, 310, 350]


def test_an_expected_move_that_cannot_be_measured_is_refused():
    with pytest.raises(ValueError, match="arrival_index.*not an RTH tick"):
        attach_expected_move(frame(minute_tape()), stalls_at(99), tick_size=TICK)
    with pytest.raises(ValueError, match="tick_size"):
        attach_expected_move(frame(minute_tape()), stalls_at(35), tick_size=0)


def test_a_sigma_window_that_is_not_a_positive_number_is_refused():
    with pytest.raises(ValueError, match="sigma_minutes"):
        attach_expected_move(frame(minute_tape()), stalls_at(35), tick_size=TICK, sigma_minutes=0)


def two_quotes_a_minute(day="2025-09-15"):
    """The minute tape with an earlier tick in every minute, at 10:MM:10, always at 99.00."""
    early = [{"day": day, "t": f"10:{m:02d}:10", "tt": 1, "vol": 10, "price": 99.0} for m in range(40)]
    return early + minute_tape(day)


def test_the_ticks_of_one_minute_are_one_bar_measured_at_its_last_quote():
    out = attach_expected_move(frame(two_quotes_a_minute()), stalls_at(71), tick_size=TICK)  # 71: 10:35:30
    assert out["sigma_ticks"].to_list() == pytest.approx([float(np.std([2.0, -2.0] * 15, ddof=1))])


def test_the_session_volume_before_an_arrival_counts_its_own_day_only():
    ticks = frame(minute_tape() + minute_tape(day="2025-09-16"))
    out = attach_expected_move(ticks, stalls_at(45, day="2025-09-16"), tick_size=TICK)  # 45: sixth tick of day two
    assert out["session_volume_before"].to_list() == [50]


def test_the_first_minute_of_every_day_is_measured_from_its_own_first_quote():
    ticks = frame(minute_tape() + two_quotes_a_minute(day="2025-09-16"))
    out = attach_expected_move(ticks, stalls_at(101, day="2025-09-16"), tick_size=TICK)  # 101: 10:30:30, day two
    # day two opens at 99.00 and its first minute closes at 100.00: +4 ticks, then +2, -2, ... as on any day
    assert out["sigma_ticks"].to_list() == pytest.approx([float(np.std([4.0] + [2.0, -2.0] * 14 + [2.0], ddof=1))])


@pytest.mark.parametrize("stray", [99, 36])
def test_one_arrival_that_is_not_an_rth_tick_is_enough_to_refuse_the_stalls(stray):
    tape = minute_tape()
    tape[36]["session"] = "ETH"  # Index 36 is in the tape, and it is not an RTH tick; 99 is past its end
    with pytest.raises(ValueError, match="arrival_index.*not an RTH tick"):
        attach_expected_move(frame(tape), stalls_at(35, stray), tick_size=TICK)


def test_the_expected_move_needs_some_volume_traded_before_the_arrival():
    quiet = [dict(row, vol=0) for row in minute_tape()]
    one_lot = [dict(row, vol=int(m == 0)) for m, row in enumerate(minute_tape())]
    none = attach_expected_move(frame(quiet), stalls_at(35), tick_size=TICK)
    some = attach_expected_move(frame(one_lot), stalls_at(35), tick_size=TICK)
    sigma = float(np.std([2.0, -2.0] * 15, ddof=1))
    assert none["expected_move"].to_list() == [None]
    assert some["expected_move"].to_list() == pytest.approx([sigma * 90 ** 0.5])


def quotes(rows):
    """rows: (t, bid) or (t, bid, day); the ask is one tick above the bid."""
    return frame([{"t": r[0], "price": r[1], "tt": 1, **({"day": r[2]} if len(r) > 2 else {})} for r in rows])


SESSION = [("10:00:00", 100.00),  # 0 the anchor
           ("10:00:01", 100.00),  # 1 the entry: bid 100.00, ask 100.25
           ("10:01:00", 99.50),   # 2 mid 99.625
           ("10:05:00", 99.00),   # 3 mid 99.125
           ("10:15:00", 100.50),  # 4 mid 100.625
           ("10:20:00", 101.00)]  # 5 keeps every horizon inside the day


def events_at(anchors, directions, **more):
    return pl.DataFrame({"end_index": anchors, "trade_dir": directions, **more},
                        schema_overrides={"end_index": pl.Int64, "trade_dir": pl.Int64})


def moves(rows, direction, anchor=0, **kwargs):
    return measure_forward_moves(quotes(rows), events_at([anchor], [direction]), tick_size=TICK,
                                 anchor_col="end_index", direction_col="trade_dir", **kwargs)


def test_a_short_enters_at_the_bid_and_a_long_at_the_ask_of_the_next_tick():
    short, long = moves(SESSION, -1), moves(SESSION, 1)
    assert short["entry_index"].to_list() == long["entry_index"].to_list() == [1]
    assert short["entry_price"].to_list() == [100.00]
    assert long["entry_price"].to_list() == [100.25]
    assert short["entry_datetime"].to_list() == [datetime(2025, 9, 15, 10, 0, 1)]


def test_an_event_with_no_later_tick_that_day_is_kept_with_nulls():
    rows = [("10:00:00", 100.0), ("10:00:05", 100.0), ("10:00:00", 99.0, "2025-09-16"), ("10:30:00", 99.0, "2025-09-16")]
    out = moves(rows, -1, anchor=1)
    assert out.height == 1
    assert out["entry_index"].to_list() == [None]
    assert out["entry_price"].to_list() == [None]
    assert out["move_1m"].to_list() == [None]


def test_ticks_sharing_the_anchor_timestamp_are_not_the_entry():
    rows = [("10:00:00", 100.0), ("10:00:00", 99.75), ("10:00:01", 99.50), ("10:30:00", 99.50)]
    assert moves(rows, -1)["entry_index"].to_list() == [2]


def test_forward_moves_are_in_ticks_and_signed_in_the_trade_direction():
    short, long = moves(SESSION, -1), moves(SESSION, 1)
    assert [short[c][0] for c in ("move_1m", "move_5m", "move_15m")] == [1.5, 3.5, -2.5]
    assert [long[c][0] for c in ("move_1m", "move_5m", "move_15m")] == [-2.5, -4.5, 1.5]


def test_a_horizon_past_the_last_tick_of_the_day_is_null():
    rows = SESSION[:4] + [("10:10:00", 99.00), ("10:00:00", 90.00, "2025-09-16"), ("10:30:00", 90.00, "2025-09-16")]
    out = moves(rows, -1)
    assert (out["move_1m"][0], out["move_5m"][0]) == (1.5, 3.5)
    assert out["move_15m"][0] is None  # 10:15:01 is after the day's last tick; tomorrow is not read
    assert out["mfe_15m"][0] is None and out["mae_15m"][0] is None


def test_mfe_and_mae_are_the_best_and_worst_mid_over_the_longest_horizon():
    short, long = moves(SESSION, -1), moves(SESSION, 1)
    assert (short["mfe_15m"][0], short["mae_15m"][0]) == (3.5, -2.5)
    assert (long["mfe_15m"][0], long["mae_15m"][0]) == (1.5, -4.5)
    rising = [("10:00:00", 100.00), ("10:00:01", 100.00), ("10:10:00", 101.00), ("10:20:00", 101.00)]
    against = moves(rising, -1)
    assert (against["mfe_15m"][0], against["mae_15m"][0]) == (0.0, -4.5)  # never in profit: MFE 0


def test_compiled_and_plain_python_window_extremes_agree():
    pytest.importorskip("numba")
    assert hasattr(ai._window_extremes, "py_func"), "the window extremes are not compiled"
    rng = np.random.default_rng(11)
    mid = 100.0 + np.cumsum(rng.integers(-2, 3, 3000)) * 0.125
    start = np.sort(rng.integers(0, 2500, 200)).astype(np.int64)
    stop = start + rng.integers(0, 400, 200)
    plain = ai._window_extremes.py_func(mid, start, stop)
    compiled = ai._window_extremes(mid, start, stop)
    assert np.array_equal(compiled[0], plain[0]) and np.array_equal(compiled[1], plain[1])
    assert (compiled[0] >= mid[start]).all() and (compiled[1] <= mid[start]).all()
    assert (compiled[0] > compiled[1]).any()


def test_an_event_with_no_direction_gets_nulls():
    ticks = quotes(SESSION)
    kwargs = dict(tick_size=TICK, anchor_col="end_index", direction_col="trade_dir")
    out = measure_forward_moves(ticks, events_at([0, 0], [None, -1]), **kwargs)
    assert out["entry_index"].to_list() == [None, 1]
    assert out["move_5m"].to_list() == [None, 3.5]
    none = measure_forward_moves(ticks, events_at([0], [None]), **kwargs)
    assert none["entry_index"].to_list() == [None]
    assert none.schema == out.schema


def test_an_entry_delay_skips_the_ticks_inside_it():
    rows = [("10:00:00", 100.0), ("10:00:00.400", 99.75), ("10:00:01", 99.50), ("10:00:01.200", 99.25),
            ("10:30:00", 99.25)]
    assert moves(rows, -1)["entry_index"].to_list() == [1]
    delayed = moves(rows, -1, entry_delay_s=1.0)
    assert delayed["entry_index"].to_list() == [3]  # the first tick LATER than the anchor plus 1 s
    assert delayed["entry_price"].to_list() == [99.25]


def test_an_anchor_that_is_not_an_rth_tick_is_refused():
    with pytest.raises(ValueError, match="end_index.*not an RTH tick"):
        moves(SESSION, -1, anchor=99)


@pytest.mark.parametrize("bad", [{"tick_size": 0}, {"entry_delay_s": -1.0}])
def test_forward_moves_refuse_parameters_that_mean_nothing(bad):
    kwargs = {"tick_size": TICK, "anchor_col": "end_index", "direction_col": "trade_dir"}
    kwargs.update(bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        measure_forward_moves(quotes(SESSION), events_at([0], [-1]), **kwargs)


def test_forward_moves_keep_the_rows_and_the_columns_of_the_events():
    events = events_at([2, 0], [1, -1], tag=["b", "a"])
    out = measure_forward_moves(quotes(SESSION), events, tick_size=TICK, anchor_col="end_index",
                                direction_col="trade_dir")
    assert out["tag"].to_list() == ["b", "a"]
    assert out["entry_index"].to_list() == [3, 1]
    assert out.columns == ["end_index", "trade_dir", "tag", "entry_index", "entry_datetime", "entry_price",
                           "move_1m", "move_5m", "move_15m", "mfe_15m", "mae_15m"]


def test_the_excursions_start_at_the_entry_not_at_the_anchor():
    rows = [("10:00:00", 105.00),  # the anchor, far above everything after it
            ("10:00:01", 100.00), ("10:10:00", 99.00), ("10:20:00", 99.00)]
    short = moves(rows, -1)
    assert (short["mfe_15m"][0], short["mae_15m"][0]) == (3.5, -0.5)


def test_a_boolean_is_not_an_entry_delay():
    with pytest.raises(ValueError, match="entry_delay_s"):
        moves(SESSION, -1, entry_delay_s=True)


@pytest.mark.parametrize("stray", [99, 2])
def test_one_anchor_that_is_not_an_rth_tick_is_enough_to_refuse_the_events(stray):
    ticks = quotes(SESSION).with_columns(  # Index 2 is in the tape, and it is not an RTH tick; 99 is past its end
        pl.when(pl.col("Index") == 2).then(pl.lit("ETH")).otherwise(pl.col("SessionType")).alias("SessionType"))
    with pytest.raises(ValueError, match="end_index.*not an RTH tick"):
        measure_forward_moves(ticks, events_at([0, stray], [-1, -1]), tick_size=TICK, anchor_col="end_index",
                              direction_col="trade_dir")


def test_the_last_tick_of_the_day_can_be_the_entry():
    rows = [("10:00:00", 100.0), ("10:00:05", 99.75),  # the first day ends on the tick after the anchor
            ("10:00:00", 99.0, "2025-09-16"), ("10:30:00", 99.0, "2025-09-16")]
    out = moves(rows, -1)
    assert out["entry_index"].to_list() == [1]
    assert out["entry_price"].to_list() == [99.75]
    assert out["move_1m"].to_list() == [None]


def test_an_event_on_the_last_tick_of_the_tape_is_kept_with_nulls():
    out = moves([("10:00:00", 100.0), ("10:00:05", 100.0)], -1, anchor=1)
    assert out["entry_index"].to_list() == [None]
    assert out["entry_price"].to_list() == [None]
    assert out["move_5m"].to_list() == [None]


def test_a_horizon_is_counted_in_minutes_of_sixty_seconds():
    rows = [("10:00:00", 100.00), ("10:00:01", 100.00),
            ("10:01:00", 99.50),      # the last quote within one minute of the entry
            ("10:01:01.500", 99.00),  # half a second too late for it
            ("10:30:00", 99.00)]
    assert moves(rows, -1)["move_1m"].to_list() == [1.5]
