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

The module also holds the event-study tools the panel needs: ``measure_forward_moves`` (what the
quote did after an event) and ``summarise_cells`` / ``difference_of_means`` (the average move with
days as clusters, because the events of one day share that day's drift).
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Optional Numba import -- graceful degradation
try:
    from numba import njit  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is not installed."""
        def _wrapper(fn):  # type: ignore[return]
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrapper

_US = 1_000_000

_BREAK_BACK, _EATEN, _TIMEOUT = 0, 1, 2
_ENDINGS = ("break_back", "eaten", "timeout")

_STALL_INPUT = ("Index", "Date", "Datetime", "SessionType", "Price", "Volume", "TradeType",
                "AskPrice", "BidPrice", "AskSize", "BidSize")
_QUOTE_INPUT = ("Index", "Date", "Datetime", "SessionType", "AskPrice", "BidPrice")
_VOLUME_INPUT =("Index", "Date", "Datetime", "SessionType", "Volume", "AskPrice", "BidPrice")

#: Columns of :func:`find_absorption_stalls`, in order. An empty result carries them too.
STALL_COLUMNS = [
    "Date", "side", "arrival_index", "arrival_datetime", "price", "absorbed_volume",
    "displayed_on_arrival", "refill_ratio", "n_trades", "duration_s", "ending", "end_index",
    "end_datetime", "TradeType",
]


def _check_positive(**values) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")


def _rth_ticks(ticks: pl.DataFrame | pl.LazyFrame, columns) -> pl.DataFrame:
    """RTH rows of ``columns`` in tape order. Refuses what would silently corrupt the scan."""
    lazy = ticks.lazy()
    names = set(lazy.collect_schema())
    missing = [column for column in columns if column not in names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    rth = lazy.filter(pl.col("SessionType") == "RTH").select(columns).collect().sort("Index")
    holes = [column for column, count in rth.null_count().row(0, named=True).items() if count]
    if holes:
        raise ValueError(f"Null values in required columns: {holes}")
    t = rth["Datetime"].dt.epoch("us").to_numpy()
    back = np.flatnonzero(np.diff(t) < 0)
    if back.size:  # the window and every searchsorted below assume time never goes backwards
        raise ValueError("RTH Datetime must be non-decreasing in Index order; "
                         f"first offending Index {rth['Index'][int(back[0]) + 1]}")
    return rth


def _day_starts(rth: pl.DataFrame) -> np.ndarray:
    """Position of the first tick of every ``Date`` block."""
    return (rth["Date"] != rth["Date"].shift(1)).fill_null(True).arg_true().to_numpy().astype(np.int64)


def _to_ticks(prices: pl.Series, tick_size: float) -> np.ndarray:
    """Prices as whole ticks, so that every comparison is exact on any tick size."""
    return np.rint(prices.to_numpy() / tick_size).astype(np.int64)


@njit(cache=True)
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
        last = i == n - 1

        if is_open:
            how = -1
            if t[i] > t[a] + wait_us:
                how = _TIMEOUT
            elif px[i] > p:
                how = _EATEN
            elif counter[i] and px[i] <= p - break_ticks:
                how = _BREAK_BACK
            elif last:
                how = _TIMEOUT
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

        # The tick that ends a stall may start the next one: a buyer eating the level is at a new high.
        if (not is_open and aggressor[i] and head < tail and t[i] - t[0] >= window_us
                and px[i] > px[high[head]]):
            is_open = True
            a = i
            p = px[i]
            v = vol[i]
            k = 1
            d = quote_sz[i] if quote_px[i] == p else -1
            if last:  # nothing can follow: the stall is closed where it started
                arrival[count] = a
                end[count] = i
                level[count] = p
                absorbed[count] = v
                trades[count] = k
                displayed[count] = d
                ending[count] = _TIMEOUT
                count += 1

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
    _check_positive(tick_size=tick_size, push_window_s=push_window_s, max_wait_s=max_wait_s)
    if isinstance(break_ticks, bool) or not isinstance(break_ticks, int) or break_ticks < 1:
        raise ValueError(f"break_ticks must be a whole number of ticks, 1 or more, got {break_ticks!r}")

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
        return np.concatenate([part[j] for part in found]) if found else np.empty(0, np.int64)

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


def attach_expected_move(
    ticks: pl.DataFrame | pl.LazyFrame,
    stalls: pl.DataFrame,
    *,
    tick_size: float,
    sigma_minutes: int = 30,
) -> pl.DataFrame:
    """``stalls`` plus ``sigma_ticks``, ``session_volume_before`` and ``expected_move``.

    The square-root law says a volume ``Q`` moves price by about ``sigma * sqrt(Q / V)``. Inside a
    stall the realised move is zero by construction, so the law's prediction is what is worth
    keeping: how far the absorbed volume *should* have pushed. The scale constant is 1, so the
    column orders stalls and means nothing in absolute terms.

    ``sigma_ticks`` is the standard deviation of the one-minute mid changes over the
    ``sigma_minutes`` RTH minutes that ended before the arrival's minute; null until that many
    minutes exist. ``session_volume_before`` is the RTH volume of the ``Date`` before the arrival
    tick. Both are known at the arrival: nothing later is read.
    """
    _check_positive(tick_size=tick_size, sigma_minutes=sigma_minutes)
    minute = pl.col("Datetime").dt.truncate("1m")
    rth = _rth_ticks(ticks, _VOLUME_INPUT).with_columns(
        ((pl.col("AskPrice") + pl.col("BidPrice")) / 2).alias("mid"),
        # one number per (Date, minute), counted from 1 in tape order
        (minute != minute.shift(1)).fill_null(True).cum_sum().alias("bar"),
        (pl.col("Volume").cum_sum().over("Date") - pl.col("Volume")).alias("session_volume_before"),
    )
    bars = (
        rth.group_by("Date", "bar", maintain_order=True)  # groups in tape order: row j is bar j + 1
        .agg(pl.col("mid").first().alias("open"), pl.col("mid").last().alias("close"))
        # the first minute of a day has no earlier close: it is measured from its own first quote
        .with_columns(pl.col("close").shift(1).over("Date").fill_null(pl.col("open")).alias("before"))
        .with_columns(((pl.col("close") - pl.col("before")) / tick_size).alias("change"))
        # shift(1): the minutes that ENDED before this one, never this one itself
        .with_columns(pl.col("change").rolling_std(sigma_minutes).shift(1).over("Date").alias("sigma_ticks"))
    )

    idx = rth["Index"].to_numpy()
    arrival = stalls["arrival_index"].to_numpy()
    k = np.searchsorted(idx, arrival)
    if (k >= idx.size).any() or (idx[k] != arrival).any():
        raise ValueError("arrival_index holds an Index that is not an RTH tick of `ticks`")
    volume = pl.col("session_volume_before")
    # Picked by position, not joined: the rows of `stalls` cannot move.
    return stalls.with_columns(
        bars["sigma_ticks"].gather(rth["bar"].gather(k) - 1),
        rth["session_volume_before"].gather(k),
    ).with_columns(
        pl.when(volume > 0)
        .then(pl.col("sigma_ticks") * (pl.col("absorbed_volume") / volume).sqrt())
        .alias("expected_move"))


@njit(cache=True)
def _window_extremes(mid, start, stop):
    """Highest and lowest mid between two positions, both included: one pair per event."""
    high = np.empty(start.size, np.float64)
    low = np.empty(start.size, np.float64)
    for j in range(start.size):
        top = mid[start[j]]
        bottom = top
        for i in range(start[j] + 1, stop[j] + 1):
            top = max(top, mid[i])
            bottom = min(bottom, mid[i])
        high[j] = top
        low[j] = bottom
    return high, low


def measure_forward_moves(
    ticks: pl.DataFrame | pl.LazyFrame,
    events: pl.DataFrame,
    *,
    tick_size: float,
    anchor_col: str,
    direction_col: str,
    horizons_min=(1, 5, 15),
    entry_delay_s: float = 0.0,
) -> pl.DataFrame:
    """``events`` plus the entry a taker gets and what the quote did afterwards, in ticks.

    ``anchor_col`` holds the ``Index`` of the tick that triggers the event, ``direction_col`` +1
    for a long and -1 for a short; a null or zero direction gets nulls.

    The entry tick is the first RTH tick of the same ``Date`` with a ``Datetime`` later than the
    anchor's plus ``entry_delay_s``. A long pays that tick's ``AskPrice``, a short gets its
    ``BidPrice``. ``move_<h>m`` is the mid of the last tick at or before ``h`` minutes after the
    entry minus the entry price, signed in the trade direction; null when that moment is past the
    day's last RTH tick. ``mfe_<H>m`` / ``mae_<H>m`` are the best and worst signed mid move over
    the longest horizon ``H``; MFE is never below 0 and MAE never above 0.

    Rows keep their order. The anchor and everything before it are never read for the move.
    """
    _check_positive(tick_size=tick_size)
    if isinstance(entry_delay_s, bool) or not isinstance(entry_delay_s, (int, float)) or entry_delay_s < 0:
        raise ValueError(f"entry_delay_s must be zero or more, got {entry_delay_s!r}")
    horizons = [int(h) for h in horizons_min]
    longest = max(horizons)
    names = [f"move_{h}m" for h in horizons] + [f"mfe_{longest}m", f"mae_{longest}m"]

    rth = _rth_ticks(ticks, _QUOTE_INPUT)
    rows = events.height
    direction = events[direction_col].fill_null(0.0).to_numpy()
    live = np.flatnonzero(direction != 0)  # the events that are measured

    idx = rth["Index"].to_numpy()
    t = rth["Datetime"].dt.epoch("us").to_numpy()
    ask = rth["AskPrice"].to_numpy()
    bid = rth["BidPrice"].to_numpy()
    mid = (ask + bid) / 2
    n = idx.size
    starts = _day_starts(rth)
    day_end = np.append(starts[1:], n) - 1  # position of each day's last tick

    anchor = events[anchor_col].to_numpy()[live]
    k = np.searchsorted(idx, anchor)
    if (k >= n).any() or (idx[k] != anchor).any():
        raise ValueError(f"{anchor_col} holds an Index that is not an RTH tick of `ticks`")
    last = day_end[np.searchsorted(starts, k, side="right") - 1]
    sign = direction[live]

    e = np.searchsorted(t, t[k] + int(entry_delay_s * _US), side="right")
    entered = e <= last  # a later tick exists on the same Date
    e = np.where(entered, e, k)  # stand-in position, masked out below
    price = np.where(sign > 0, ask[e], bid[e])

    def spread(values: np.ndarray, valid: np.ndarray) -> pl.Series:
        """Per-event values back onto the rows of ``events``; null wherever not valid."""
        full = np.full(rows, np.nan)
        full[live[valid]] = values[valid]
        return pl.Series(full).fill_nan(None)

    moves = {}
    for h in horizons:
        moment = t[e] + h * 60 * _US
        inside = entered & (moment <= t[last])
        at = np.where(inside, np.searchsorted(t, moment, side="right") - 1, e)
        moves[f"move_{h}m"] = spread(sign * (mid[at] - price) / tick_size, inside)
        if h == longest:
            high, low = _window_extremes(mid, e, at)
            up, down = (high - price) / tick_size, (low - price) / tick_size
            moves[f"mfe_{longest}m"] = spread(np.maximum(np.where(sign > 0, up, -down), 0.0), inside)
            moves[f"mae_{longest}m"] = spread(np.minimum(np.where(sign > 0, down, -up), 0.0), inside)

    position = np.zeros(rows, np.int64)
    position[live[entered]] = e[entered]
    has_entry = np.zeros(rows, bool)
    has_entry[live[entered]] = True
    keep = pl.Series(has_entry)
    return events.with_columns(
        pl.when(keep).then(pl.Series(idx[position])).alias("entry_index"),
        pl.when(keep).then(rth["Datetime"].gather(position)).alias("entry_datetime"),
        spread(price, entered).alias("entry_price"),
        *[moves[name].alias(name) for name in names],
    )


def _clustered(x: np.ndarray, day: np.ndarray):
    """Mean of ``x`` and its t with days as clusters; t is None with fewer than two days."""
    mean = float(x.mean())
    codes = np.unique(day, return_inverse=True)[1]
    days = int(codes.max()) + 1
    residual = np.bincount(codes, x - mean)  # each day's summed deviation
    error = float(np.sqrt((residual ** 2).sum() * days / (days - 1))) / x.size
    return mean, mean / error


def summarise_cells(
    events: pl.DataFrame,
    *,
    by,
    move_col: str,
    date_col: str = "Date",
    exclude_dates=(),
    drop_best_days: int = 5,
) -> pl.DataFrame:
    """One row per cell of ``by``: how many events, their mean move, and how sure the mean is.

    Rows with a null ``move_col`` are left out. Columns: the ``by`` columns, ``events``,
    ``days``, ``mean``, ``t`` (day-clustered: the events of one day are one observation of that
    day's drift, so the standard error sums each day's deviations before squaring; null with
    fewer than two days), ``mean_<year>`` for every year in ``events``, ``mean_ex_dates`` (without
    ``exclude_dates``) and ``mean_ex_best_days`` (without the ``drop_best_days`` dates whose summed
    move is largest). An empty ``by`` gives one row for the whole frame.
    """
    data = events
    schema = {name: events.schema[name] for name in by}
    schema.update({"events": pl.Int64, "days": pl.Int64, "mean": pl.Float64, "t": pl.Float64})

    groups = data.partition_by(by, as_dict=True) if by else {(): data}
    rows = []
    for key, group in groups.items():
        x = group[move_col].to_numpy()
        day = group[date_col].to_numpy()
        mean, t_value = _clustered(x, day)
        row = dict(zip(by, key))
        row.update(events=int(x.size), days=int(np.unique(day).size), mean=mean, t=t_value)
        rows.append(row)
    out = pl.DataFrame(rows, schema=schema)
    return out.sort(by) if by else out
