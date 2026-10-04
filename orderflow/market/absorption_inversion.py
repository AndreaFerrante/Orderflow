"""Absorption stalls and their inversion: where aggressive traders get trapped by resting orders.

A **buy-side stall** starts when an aggressive buy (``TradeType 2``) prints at a price ``P``
strictly above every trade of the trailing ``push_window_s`` seconds: the *arrival*. It stays
open while no trade prints above ``P``. The aggressive buy volume traded at ``P`` meanwhile is
the *absorbed volume*: buyers kept lifting the offer and the offer did not move. The stall ends
in one of three ways, first to occur:

* ``timeout``    -- a tick later than ``max_wait_s`` after the arrival, or the day's last RTH tick;
* ``eaten``      -- a trade above ``P``: the resting seller gave way;
* ``break_back`` -- an aggressive sell at ``P - break_ticks`` ticks or lower. The buyers absorbed
  at ``P`` are trapped; this tick is the inversion trigger and it points SHORT (``TradeType 1``).

The sell side is the exact mirror: aggressive sells (``TradeType 1``) at a new low, eaten by a
trade below, break-back on an aggressive buy at ``P + break_ticks`` ticks or higher, pointing
LONG (``TradeType 2``).

``side`` is the side of the absorbed aggressors (+1 buyers, -1 sellers). ``TradeType`` is the side
of the trade the inversion points to, the opposite one, and only a ``break_back`` carries it.

Only RTH ticks are read and every quantity is computed inside one ``Date``. The scan reads each
tick once, in tape order, so a stall never depends on anything after its ending tick.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_US = 1_000_000

_BREAK_BACK, _EATEN, _TIMEOUT = 0, 1, 2
_ENDINGS = ("break_back", "eaten", "timeout")

_STALL_INPUT = ("Index", "Date", "Datetime", "SessionType", "Price", "Volume", "TradeType",
                "AskPrice", "BidPrice", "AskSize", "BidSize")

#: Columns of :func:`find_absorption_stalls`, in order. An empty result carries them too.
STALL_COLUMNS = [
    "Date", "side", "arrival_index", "arrival_datetime", "price", "absorbed_volume",
    "displayed_on_arrival", "refill_ratio", "n_trades", "duration_s", "ending", "end_index",
    "end_datetime", "TradeType",
]


def _rth_ticks(ticks: pl.DataFrame | pl.LazyFrame, columns) -> pl.DataFrame:
    """RTH rows of ``columns`` in tape order."""
    return ticks.lazy().filter(pl.col("SessionType") == "RTH").select(columns).collect().sort("Index")


def _day_starts(rth: pl.DataFrame) -> np.ndarray:
    """Position of the first tick of every ``Date`` block."""
    return (rth["Date"] != rth["Date"].shift(1)).fill_null(True).arg_true().to_numpy().astype(np.int64)


def _to_ticks(prices: pl.Series, tick_size: float) -> np.ndarray:
    """Prices as whole ticks, so that every comparison is exact on any tick size."""
    return np.rint(prices.to_numpy() / tick_size).astype(np.int64)


def _scan_side(t, px, vol, aggressor, counter, quote_px, quote_sz, window_us, break_ticks, wait_us):
    """Stalls of one side on one day. Written for buyers: every comparison looks upward.

    The sell side is this same scan on negated prices with the trade sides swapped, so the two
    sides cannot drift apart. Works on positions and whole ticks; -1 means "no size displayed".
    """
    n = t.size
    arrival = np.empty(n, np.int64)
    end = np.empty(n, np.int64)
    level = np.empty(n, np.int64)
    absorbed = np.empty(n, np.int64)
    trades = np.empty(n, np.int64)
    displayed = np.empty(n, np.int64)
    ending = np.empty(n, np.int64)
    count = 0

    high = np.zeros(n, np.int64)  # positions of the window's ticks, prices falling: front = highest
    head = 0
    tail = 0

    is_open = False
    a = 0   # position of the arrival tick
    p = 0   # the stall price
    v = 0   # absorbed volume
    k = 0   # aggressive trades at the stall price
    d = -1  # size displayed at the stall price on arrival
    for i in range(n):
        while head < tail and t[high[head]] < t[i] - window_us:
            head += 1

        if is_open:
            how = -1
            if counter[i] and px[i] <= p - break_ticks:
                how = _BREAK_BACK
            if how >= 0:
                arrival[count] = a
                end[count] = i
                level[count] = p
                absorbed[count] = v
                trades[count] = k
                displayed[count] = d
                ending[count] = how
                count += 1
                is_open = False
            elif aggressor[i] and px[i] == p:
                v += vol[i]
                k += 1

        if (not is_open and aggressor[i] and head < tail and t[i] - t[0] >= window_us
                and px[i] > px[high[head]]):
            is_open = True
            a = i
            p = px[i]
            v = vol[i]
            k = 1
            d = quote_sz[i] if quote_px[i] == p else -1

        while head < tail and px[high[tail - 1]] <= px[i]:
            tail -= 1
        high[tail] = i
        tail += 1

    return (arrival[:count].copy(), end[:count].copy(), level[:count].copy(),
            absorbed[:count].copy(), trades[:count].copy(), displayed[:count].copy(),
            ending[:count].copy())


def find_absorption_stalls(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    tick_size: float,
    push_window_s: float,
    break_ticks: int,
    max_wait_s: float,
) -> pl.DataFrame:
    """One row per stall, both sides, in tape order of the arrival tick.

    ``ticks`` needs ``Index``, ``Date``, ``Datetime``, ``SessionType``, ``Price``, ``Volume``,
    ``TradeType``, ``AskPrice``, ``BidPrice``, ``AskSize``, ``BidSize``. Only RTH rows are read.

    Returns the columns of :data:`STALL_COLUMNS`. ``absorbed_volume`` counts the aggressor-side
    trades at the stall price, the arrival tick included and the ending tick excluded.
    ``displayed_on_arrival`` is the size quoted at the stall price on the arrival tick, null when
    the quote was at another price; ``refill_ratio`` is absorbed over displayed, null when nothing
    was displayed. ``TradeType`` is 1 (short) for a buy-side ``break_back``, 2 (long) for a
    sell-side one, null for every other ending.
    """
    rth = _rth_ticks(ticks, _STALL_INPUT)
    idx = rth["Index"].to_numpy()
    t = rth["Datetime"].dt.epoch("us").to_numpy()
    px = _to_ticks(rth["Price"], tick_size)
    ask = _to_ticks(rth["AskPrice"], tick_size)
    bid = _to_ticks(rth["BidPrice"], tick_size)
    vol = rth["Volume"].to_numpy().astype(np.int64)
    trade_type = rth["TradeType"].to_numpy()
    buy, sell = trade_type == 2, trade_type == 1
    ask_size = rth["AskSize"].to_numpy().astype(np.int64)
    bid_size = rth["BidSize"].to_numpy().astype(np.int64)
    window_us, wait_us = int(push_window_s * _US), int(max_wait_s * _US)

    starts = _day_starts(rth)
    found = []
    for lo, hi in zip(starts, np.append(starts[1:], rth.height)):
        day = slice(lo, hi)
        buyers = _scan_side(t[day], px[day], vol[day], buy[day], sell[day], ask[day],
                            ask_size[day], window_us, break_ticks, wait_us)
        # Sellers: the same scan on the mirrored tape. A new low is a new high of -price.
        sellers = _scan_side(t[day], -px[day], vol[day], sell[day], buy[day], -bid[day],
                             bid_size[day], window_us, break_ticks, wait_us)
        for side, part in ((1, buyers), (-1, sellers)):
            found.append((np.full(part[0].size, side, np.int64), part[0] + lo, part[1] + lo) + part[2:])

    def column(j: int) -> np.ndarray:
        return np.concatenate([part[j] for part in found])

    order = np.argsort(column(1))
    side, arrival, end, level, absorbed, trades, displayed, ending = (column(j)[order] for j in range(8))

    shown = pl.col("displayed_on_arrival")
    return pl.DataFrame({
        "Date": rth["Date"].gather(arrival),
        "side": side,
        "arrival_index": idx[arrival],
        "arrival_datetime": rth["Datetime"].gather(arrival),
        "price": side * level * tick_size,  # sellers were scanned on -price
        "absorbed_volume": absorbed,
        "displayed_on_arrival": displayed,
        "n_trades": trades,
        "duration_s": (t[end] - t[arrival]) / _US,
        "ending": np.array(_ENDINGS)[ending],
        "end_index": idx[end],
        "end_datetime": rth["Datetime"].gather(end),
    }).with_columns(
        pl.when(shown >= 0).then(shown),
    ).with_columns(
        pl.when(shown > 0).then(pl.col("absorbed_volume") / shown).alias("refill_ratio"),
        # Buyers trapped -> SHORT (1); sellers trapped -> LONG (2). Swapping this map trades every
        # inversion the wrong way with no other symptom.
        pl.when(pl.col("ending") == "break_back")
        .then(pl.when(pl.col("side") == 1).then(1).otherwise(2)).cast(pl.Int64).alias("TradeType"),
    ).select(STALL_COLUMNS)
