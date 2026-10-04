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
