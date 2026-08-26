import warnings

import pandas as pd
import numpy as np
import polars as pl


def _buy_flag(ask, bid, p, ratio, eps):
    """Buy imbalance at level ``p``: ask at P against bid one level BELOW.

    ``TradeType 2`` is ask volume, buy aggression, LONG.  The level below is
    ``p - 1`` because the ladder index increases with price.

    A level with zero on *both* sides is untouched ladder padding, not an
    observed zero, and must not trip the zero-denominator bypass below it —
    otherwise every padded cell next to a real print would spuriously flag.
    In market terms: a price that never traded had no auction there to be
    imbalanced against, so it does not flag. This is deliberate, not an
    artefact of the fixed-width ladder array.

    ``eps`` is added to the denominator deliberately: an exact
    ``ask == ratio * bid`` must reject, not accept, so the zero-denominator
    bypass above needs no special case.

    ``p`` must have a real neighbour on both sides (``1 <= p <= len - 2``);
    out of that range a Python negative index would silently wrap to the far
    end of the ladder instead of raising, so it is rejected explicitly.
    """
    if p < 1 or p >= len(ask) - 1:
        return False
    if bid[p - 1] <= 0 and ask[p - 1] <= 0:
        return False
    return bool(ask[p] >= ratio * (bid[p - 1] + eps))


def _sell_flag(ask, bid, p, ratio, eps):
    """Sell imbalance at level ``p``: bid at P against ask one level ABOVE.

    ``TradeType 1`` is bid volume, sell aggression, SHORT. Same untouched-cell
    and boundary guards as ``_buy_flag``, mirrored on the level above: a
    price that never traded had no auction there to be imbalanced against,
    so it does not flag, and ``eps`` makes an exact ratio match reject, not
    accept.
    """
    if p < 1 or p >= len(ask) - 1:
        return False
    if ask[p + 1] <= 0 and bid[p + 1] <= 0:
        return False
    return bool(bid[p] >= ratio * (ask[p + 1] + eps))


def _flag_at(ask, bid, p, direction, ratio, eps):
    if direction == 1:
        return _buy_flag(ask, bid, p, ratio, eps)
    return _sell_flag(ask, bid, p, ratio, eps)


def _pair_volume(ask, bid, p, direction):
    """Both legs of the diagonal pair at level ``p``."""
    if direction == 1:
        return ask[p] + bid[p - 1]
    return bid[p] + ask[p + 1]


def _scan_stack(ask, bid, p, direction, ratio, n_consecutive, min_diagonal_volume, eps):
    """Return ``(lo, hi, diagonal_volume)`` for the qualifying run containing ``p``.

    Returns ``(-1, -1, 0.0)`` when ``p`` is unflagged, when the run is shorter
    than ``n_consecutive``, or when the summed diagonal volume is below
    ``min_diagonal_volume``.
    """
    n = ask.shape[0]
    if p < 1 or p >= n - 1:
        return -1, -1, 0.0
    if not _flag_at(ask, bid, p, direction, ratio, eps):
        return -1, -1, 0.0

    lo = p
    while lo - 1 >= 1 and _flag_at(ask, bid, lo - 1, direction, ratio, eps):
        lo -= 1
    hi = p
    while hi + 1 < n - 1 and _flag_at(ask, bid, hi + 1, direction, ratio, eps):
        hi += 1

    if (hi - lo + 1) < n_consecutive:
        return -1, -1, 0.0

    total = 0.0
    for lvl in range(lo, hi + 1):
        total += _pair_volume(ask, bid, lvl, direction)

    if total < min_diagonal_volume:
        return -1, -1, 0.0
    return lo, hi, total


def filter_big_prints_on_ask(
    data: pd.DataFrame, volume_filter: int = 100
) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this functions returns filtered volume dataframe on the ASK
    :param data: canonical dataframe recorded
    :param volume_filter: time and sales dataframe recorded volume filter
    :return: dataframe with the given filter
    """

    filtered_on_ask = data.query("TradeType == 2").query(
        "Volume >= " + str(volume_filter) + ""
    )

    return filtered_on_ask


def filter_big_prints_on_bid(
    data: pd.DataFrame, volume_filter: int = 100
) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this functions returns filtered volume dataframe on the BID
    :param data: canonical dataframe recorded
    :param volume_filter: time and sales dataframe recorded volume filter
    :return: dataframe with the given filter
    """

    filtered_on_bid = data.query("TradeType == 1").query(
        "Volume >= " + str(volume_filter) + ""
    )

    return filtered_on_bid


_REQUIRED_COLUMNS = ("Index", "Datetime", "Price", "Volume", "TradeType", "SessionType")


def _detect_python(
    prices, volumes, trade_types, bar_ids, session_ids, indices,
    tick_size, ratio, n_consecutive, min_diagonal_volume, eps, ladder_width,
):
    n = prices.shape[0]
    half = ladder_width // 2
    bid = np.zeros(ladder_width, dtype=np.float64)
    ask = np.zeros(ladder_width, dtype=np.float64)

    out_signal_idx, out_entry_idx, out_dir = [], [], []
    out_levels, out_lo_px, out_hi_px, out_vol, out_bar = [], [], [], [], []
    out_row = []

    origin = 0.0
    armed = {1: [], -1: []}   # direction -> list of fired (lo, hi) ranges, concurrently active
    overflow = 0

    for i in range(n):
        new_bar = (i == 0) or (bar_ids[i] != bar_ids[i - 1]) or (session_ids[i] != session_ids[i - 1])
        if new_bar:
            bid[:] = 0.0
            ask[:] = 0.0
            origin = prices[i]
            armed = {1: [], -1: []}

        p = int(round((prices[i] - origin) / tick_size)) + half
        if p < 1 or p >= ladder_width - 1:
            overflow += 1
            continue

        if trade_types[i] == 2:
            ask[p] += volumes[i]
        elif trade_types[i] == 1:
            bid[p] += volumes[i]

        # A tick at p can only move four flags: buy at p and p+1, sell at p and p-1.
        for direction, candidates in ((1, (p, p + 1)), (-1, (p, p - 1))):
            # drop any armed range whose stack has broken; ranges that still
            # qualify stay armed so a persisting stack fires only once, while
            # a second, disjoint stack is still free to fire independently
            armed[direction] = [
                (lo_a, hi_a)
                for (lo_a, hi_a) in armed[direction]
                if all(
                    _flag_at(ask, bid, lvl, direction, ratio, eps)
                    for lvl in range(lo_a, hi_a + 1)
                )
            ]

            for cand in candidates:
                lo, hi, vol = _scan_stack(
                    ask, bid, cand, direction, ratio, n_consecutive, min_diagonal_volume, eps
                )
                if lo < 0:
                    continue
                # suppress only if this candidate range overlaps a range
                # already armed for this direction; a disjoint stack is not suppressed
                if any(not (hi < a_lo or lo > a_hi) for (a_lo, a_hi) in armed[direction]):
                    continue
                if i + 1 >= n:
                    continue                    # no entry tick exists
                armed[direction].append((lo, hi))
                out_signal_idx.append(indices[i])
                out_entry_idx.append(indices[i + 1])
                out_dir.append(direction)
                out_levels.append(hi - lo + 1)
                out_lo_px.append(origin + (lo - half) * tick_size)
                out_hi_px.append(origin + (hi - half) * tick_size)
                out_vol.append(vol)
                out_bar.append(bar_ids[i])
                out_row.append(i)

    return (
        out_signal_idx, out_entry_idx, out_dir, out_levels,
        out_lo_px, out_hi_px, out_vol, out_bar, out_row, overflow,
    )


def find_stacked_imbalances(
    ticks: pl.DataFrame,
    tick_size: float,
    imbalance_ratio: float = 3.0,
    n_consecutive: int = 3,
    min_diagonal_volume: float = 0.0,
    bar_id_col: str = "volume_bar_id",
    eps: float = 1e-6,
    ladder_width: int = 512,
) -> pl.DataFrame:
    """Find stacked diagonal imbalances tick by tick.

    A stack is ``n_consecutive`` or more adjacent price levels where the same
    side dominates its diagonal counterpart by at least ``imbalance_ratio``,
    and where the summed volume of both legs of every diagonal pair reaches
    ``min_diagonal_volume``.

    Buy stacks yield ``direction = +1`` (LONG, ``TradeType 2``); sell stacks
    yield ``direction = -1`` (SHORT, ``TradeType 1``).  Detection is
    edge-triggered: a stack that persists across many ticks emits one signal
    and re-arms only once it breaks.  Two disjoint stacks in the same
    direction and the same bar can be armed concurrently — a candidate only
    gets suppressed when its price range overlaps an already-armed one, not
    merely because some other stack for that direction is still active.  The
    ladder resets on every bar roll and session change, so no stack spans
    either boundary.

    ``Datetime`` on the output is the trigger tick's timestamp (the tick
    that completed the stack), not the entry tick's — the entry tick is
    identified separately by ``entry_index``. Hourly/regime splits in the
    downstream analysis are meant to bucket by when the signal fired.

    ``ladder_width`` bounds the price window held per bar.
    ponytail: fixed 512-level window, grow-on-overflow if any instrument ever
    prints a bar spanning more than that; overflowing ticks are counted and
    skipped, never written out of bounds.
    """
    missing = [c for c in (*_REQUIRED_COLUMNS, bar_id_col) if c not in ticks.columns]
    if missing:
        raise ValueError(f"ticks is missing required columns: {missing}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    if n_consecutive < 1:
        raise ValueError(f"n_consecutive must be >= 1, got {n_consecutive}")
    if ticks[bar_id_col].null_count() > 0:
        raise ValueError(f"{bar_id_col} must not contain nulls")

    bar_ids = ticks[bar_id_col].to_numpy()
    session_codes = ticks["SessionType"].cast(pl.Categorical).to_physical().to_numpy()

    (sig, ent, dirs, levels, lo_px, hi_px, vols, bars, rows, overflow) = _detect_python(
        ticks["Price"].to_numpy().astype(np.float64),
        ticks["Volume"].to_numpy().astype(np.float64),
        ticks["TradeType"].to_numpy().astype(np.int64),
        bar_ids.astype(np.int64),
        session_codes.astype(np.int64),
        ticks["Index"].to_numpy().astype(np.int64),
        float(tick_size), float(imbalance_ratio), int(n_consecutive),
        float(min_diagonal_volume), float(eps), int(ladder_width),
    )

    if overflow:
        warnings.warn(
            f"{overflow} tick(s) fell outside the {ladder_width}-level ladder "
            f"and were skipped; raise ladder_width if this is not negligible.",
            RuntimeWarning,
            stacklevel=2,
        )

    datetimes = ticks["Datetime"].to_list()
    return pl.DataFrame(
        {
            "signal_index": pl.Series(sig, dtype=pl.Int64),
            "entry_index": pl.Series(ent, dtype=pl.Int64),
            "direction": pl.Series(dirs, dtype=pl.Int64),
            "n_levels": pl.Series(levels, dtype=pl.Int64),
            "stack_low_price": pl.Series(lo_px, dtype=pl.Float64),
            "stack_high_price": pl.Series(hi_px, dtype=pl.Float64),
            "diagonal_volume": pl.Series(vols, dtype=pl.Float64),
            "bar_id": pl.Series(bars, dtype=pl.Int64),
            "Datetime": pl.Series(
                [datetimes[r] for r in rows], dtype=ticks["Datetime"].dtype
            ),
        }
    ).sort("entry_index")
