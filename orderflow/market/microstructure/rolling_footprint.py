"""A footprint with no bars: the last N contracts traded, sliding tick by tick.

Every other footprint of this library is built per bar. Here the footprint is the volume of the last
``window_contracts`` contracts at each price, split by who was aggressive, and it moves with every
tick: a tick adds its volume at its price and the oldest volume leaves, partly when the boundary
falls inside a tick. The window is measured in contracts only, not in seconds and not in bars.

``TradeType 2`` is ask volume (buy aggression), ``TradeType 1`` is bid volume (sell aggression). A
diagonal sets the ask at a price against the bid one level below it (buyers) or the bid at a price
against the ask one level above it (sellers).

At the price ``P`` of the last trade:

* a **buy stack** is ``n_levels`` consecutive levels ``P, P-1, ...`` that are all buy imbalances:
  the ask is at least ``min_diagonal_volume`` and at least ``imbalance_ratio`` times the bid one
  level below, and that level below has traded on one side or the other inside the window (a price
  that never traded had no auction to be imbalanced against; a bid of zero next to a real ask is
  an imbalance);
* a **sell stack** is the mirror: ``P, P+1, ...``, the bid against the ask one level above.

A stack signals once, when it turns from false to true, and signals again only after it has been
false. Buy and sell are tracked apart, so one tick can signal both. The window is cleared when
``SessionType`` changes (the halt and the weekend end a session) and starts empty at the first tick;
slow tape inside a session does not clear it, and ``window_age_s`` says how old its oldest contract is.

A signal at tick ``t`` reads ticks up to and including ``t``.
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


def _check_positive(**values) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")


@njit(cache=True)
def _scan(level, volume, is_ask, segment, window, ratio, floor, n_levels, width):
    """Signals of one tape. Positions and whole ticks only, no clock.

    ``level`` is the price of each tick on a ladder of ``width`` cells with room for ``n_levels + 1``
    cells beyond the lowest and the highest price. Returns one row per signal: the position of the
    signal tick, its direction, the position of the oldest tick still in the window, then the
    winning-side volume of the stack's levels, nearest the last trade price first.
    """
    n = level.size
    ask = np.zeros(width, np.int64)
    bid = np.zeros(width, np.int64)
    capacity = window + 2  # every contract in the window is one or more, plus the tick being added
    r_level = np.zeros(capacity, np.int64)
    r_ask = np.zeros(capacity, np.bool_)
    r_left = np.zeros(capacity, np.int64)
    r_pos = np.zeros(capacity, np.int64)
    head = 0
    size = 0
    total = 0
    was_buy = False
    was_sell = False
    out = np.zeros((1024, 3 + n_levels), np.int64)
    count = 0
    for i in range(n):
        if i > 0 and segment[i] != segment[i - 1]:
            while size > 0:
                if r_ask[head]:
                    ask[r_level[head]] -= r_left[head]
                else:
                    bid[r_level[head]] -= r_left[head]
                head = (head + 1) % capacity
                size -= 1
            total = 0
        if is_ask[i]:
            ask[level[i]] += volume[i]
        else:
            bid[level[i]] += volume[i]
        tail = (head + size) % capacity
        r_level[tail] = level[i]
        r_ask[tail] = is_ask[i]
        r_left[tail] = volume[i]
        r_pos[tail] = i
        size += 1
        total += volume[i]
        while total > window:
            take = min(r_left[head], total - window)
            if r_ask[head]:
                ask[r_level[head]] -= take
            else:
                bid[r_level[head]] -= take
            r_left[head] -= take
            total -= take
            if r_left[head] == 0:
                head = (head + 1) % capacity
                size -= 1
        p = level[i]
        buy = True
        sell = True
        for k in range(n_levels):
            if not (ask[p - k] >= floor and ask[p - k] >= ratio * bid[p - k - 1]
                    and ask[p - k - 1] + bid[p - k - 1] > 0):
                buy = False
            if not (bid[p + k] >= floor and bid[p + k] >= ratio * ask[p + k + 1]
                    and ask[p + k + 1] + bid[p + k + 1] > 0):
                sell = False
        fire_buy = buy and not was_buy
        fire_sell = sell and not was_sell
        was_buy = buy
        was_sell = sell
        for direction in (1, -1):
            if (direction == 1 and fire_buy) or (direction == -1 and fire_sell):
                if count == out.shape[0]:
                    bigger = np.zeros((2 * count, 3 + n_levels), np.int64)
                    bigger[:count] = out[:count]
                    out = bigger
                out[count, 0] = i
                out[count, 1] = direction
                out[count, 2] = r_pos[head]
                for k in range(n_levels):
                    out[count, 3 + k] = ask[p - k] if direction == 1 else bid[p + k]
                count += 1
    return out[:count].copy()


def _segments(session: pl.Series) -> np.ndarray:
    """Run number of every tick's ``SessionType``: 0, 0, 0, 1, 1, 2 ... Needs at least one tick."""
    rth = (session == "RTH").to_numpy()
    return np.concatenate(([0], np.cumsum(rth[1:] != rth[:-1]))).astype(np.int64)


_TAPE = ("Index", "Date", "Datetime", "SessionType", "Price", "Volume", "TradeType")


def _tape(ticks: pl.DataFrame | pl.LazyFrame, columns) -> pl.DataFrame:
    """The ``columns`` of a tape in tape order. Refuses what would silently corrupt a scan."""
    lazy = ticks.lazy()
    names = set(lazy.collect_schema())
    missing = [column for column in columns if column not in names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = lazy.select(columns).collect()
    holes = [column for column, count in frame.null_count().row(0, named=True).items() if count]
    if holes:
        raise ValueError(f"Null values in required columns: {holes}")
    if (np.diff(frame["Index"].to_numpy()) <= 0).any():
        raise ValueError("Index must be strictly increasing: a tape is read in tape order")
    if not frame["SessionType"].is_in(["RTH", "ETH"]).all():
        raise ValueError("SessionType must be 'RTH' or 'ETH'")
    return frame


def event_columns(n_levels: int = 3) -> list[str]:
    """Columns of :func:`find_rolling_stacked_imbalances`, in order. An empty result carries them too."""
    return (["signal_index", "Date", "Datetime", "SessionType", "direction"]
            + [f"vol_{k}" for k in range(n_levels)] + ["window_age_s"])


def find_rolling_stacked_imbalances(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    tick_size: float,
    window_contracts: int = 2000,
    imbalance_ratio: float = 3.0,
    min_diagonal_volume: float = 400,
    n_levels: int = 3,
) -> pl.DataFrame:
    """One row per stack signal, in tape order (a buy before a sell on the same tick).

    ``ticks`` needs ``Index``, ``Date``, ``Datetime``, ``SessionType``, ``Price``, ``Volume`` and
    ``TradeType`` (1 or 2), in tape order. Columns: ``signal_index`` (the ``Index`` of the tick
    that completes the stack), that tick's ``Date``, ``Datetime`` and ``SessionType``,
    ``direction`` (+1 buy stack, -1 sell stack), ``vol_0`` .. ``vol_<n_levels-1>`` (the winning-side
    volume at ``P`` and at each level behind it) and ``window_age_s`` (seconds between the oldest
    contract still in the window and the signal tick; a diagnostic, never a rule).
    """
    _check_positive(tick_size=tick_size, window_contracts=window_contracts, imbalance_ratio=imbalance_ratio,
                    min_diagonal_volume=min_diagonal_volume, n_levels=n_levels)
    if not isinstance(window_contracts, int) or not isinstance(n_levels, int):
        raise ValueError("window_contracts and n_levels must be whole numbers")
    frame = _tape(ticks, list(_TAPE))
    if not frame.height:
        raise ValueError("the tape has no ticks")
    if not frame["TradeType"].is_in([1, 2]).all():
        raise ValueError("TradeType must be 1 (bid, sell aggression) or 2 (ask, buy aggression)")
    if (frame["Volume"] < 1).any():
        raise ValueError("Volume must be at least 1")
    ticks_of_price = np.rint(frame["Price"].to_numpy() / tick_size).astype(np.int64)
    level = ticks_of_price - ticks_of_price.min() + n_levels + 1
    rows = _scan(level, frame["Volume"].to_numpy().astype(np.int64), (frame["TradeType"] == 2).to_numpy(),
                 _segments(frame["SessionType"]), window_contracts, float(imbalance_ratio),
                 float(min_diagonal_volume), n_levels, int(level.max()) + n_levels + 2)
    at = rows[:, 0]
    clock = frame["Datetime"].dt.epoch("us").to_numpy()
    columns = {
        "signal_index": frame["Index"].gather(at),
        "Date": frame["Date"].gather(at),
        "Datetime": frame["Datetime"].gather(at),
        "SessionType": frame["SessionType"].gather(at),
        "direction": rows[:, 1],
    }
    for k in range(n_levels):
        columns[f"vol_{k}"] = rows[:, 3 + k]
    columns["window_age_s"] = (clock[at] - clock[rows[:, 2]]) / 1_000_000
    return pl.DataFrame(columns)


def forward_moves_by_tick(
    ticks: pl.DataFrame | pl.LazyFrame,
    events: pl.DataFrame,
    *,
    tick_size: float,
    horizons=(5, 20, 100),
    anchor_col: str = "signal_index",
    direction_col: str = "direction",
) -> pl.DataFrame:
    """``events`` plus what the tape did after them, in ticks, signed in the event's direction.

    ``anchor_col`` holds the ``Index`` of the tick that triggers the event, ``direction_col`` +1 for a
    long and -1 for a short. The entry is the tick after the anchor, at its trade price: where
    ``BacktestEngine`` fills, with no spread and no slippage. ``move_<h>`` is the trade price ``h``
    records after the entry minus the entry price. ``mfe_<H>`` and ``mae_<H>``, ``H`` the longest
    horizon, are the best and the worst signed move over the records after the entry; MFE is never
    below 0 and MAE never above 0.

    A move is null when the record it needs is past the end of the tape or in another run of
    ``SessionType`` than the anchor. Rows keep their order. The anchor and everything before it are
    never read for a move.
    """
    raise NotImplementedError


def systematic_events(ticks: pl.DataFrame | pl.LazyFrame, *, step: int) -> pl.DataFrame:
    """The base rate: every ``step``-th tick of the tape, once as a buy (+1) and once as a sell (-1).

    The columns are the first five of :func:`find_rolling_stacked_imbalances`, so the same tools read
    both. No random numbers: the sample is a function of the tape.
    """
    raise NotImplementedError
