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


def _winning_volume(ask, bid, p, direction):
    """The dominant leg only at level ``p`` -- the side that won the imbalance.

    ``_pair_volume`` sums both legs, so it is inflated by the losing leg it is
    measured against; a 3:1 stack carries a third of its pair volume in size
    that was, by construction, overwhelmed. This returns just the aggressive
    side: ask (``TradeType 2``, buy aggression) for ``direction == 1``, bid
    (``TradeType 1``, sell aggression) for ``direction == -1``.

    The dominant leg sits at ``p`` itself in both directions -- only the
    denominator is diagonal.
    """
    if direction == 1:
        return ask[p]
    return bid[p]


def _scan_stack(
    ask, bid, p, direction, ratio, n_consecutive,
    min_diagonal_volume, min_winning_volume, min_level_diagonal_volume, eps,
):
    """Return ``(lo, hi, diagonal_volume, winning_volume)`` for the run at ``p``.

    Returns ``(-1, -1, 0.0, 0.0)`` when ``p`` is unflagged, when the run is
    shorter than ``n_consecutive``, or when any of the three size floors is
    unmet.  All three are independent gates:

    * ``min_diagonal_volume`` -- both-legs pair volume SUMMED over the stack.
    * ``min_winning_volume`` -- winning-leg volume summed over the stack.
    * ``min_level_diagonal_volume`` -- both-legs pair volume at EVERY level
      individually.  A stack of thin levels can clear the summed floor; this
      one disqualifies the whole stack on a single thin level.
    """
    n = ask.shape[0]
    if p < 1 or p >= n - 1:
        return -1, -1, 0.0, 0.0
    if not _flag_at(ask, bid, p, direction, ratio, eps):
        return -1, -1, 0.0, 0.0

    lo = p
    while lo - 1 >= 1 and _flag_at(ask, bid, lo - 1, direction, ratio, eps):
        lo -= 1
    hi = p
    while hi + 1 < n - 1 and _flag_at(ask, bid, hi + 1, direction, ratio, eps):
        hi += 1

    if (hi - lo + 1) < n_consecutive:
        return -1, -1, 0.0, 0.0

    total = 0.0
    winning = 0.0
    for lvl in range(lo, hi + 1):
        pair = _pair_volume(ask, bid, lvl, direction)
        if pair < min_level_diagonal_volume:
            return -1, -1, 0.0, 0.0
        total += pair
        winning += _winning_volume(ask, bid, lvl, direction)

    if total < min_diagonal_volume:
        return -1, -1, 0.0, 0.0
    if winning < min_winning_volume:
        return -1, -1, 0.0, 0.0
    return lo, hi, total, winning


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


def _detect(
    prices, volumes, trade_types, bar_ids, session_ids, indices,
    tick_size, ratio, n_consecutive, min_diagonal_volume, min_winning_volume,
    min_level_diagonal_volume, require_breakout, eps, ladder_width,
):
    n = prices.shape[0]
    half = ladder_width // 2
    bid = np.zeros(ladder_width, dtype=np.float64)
    ask = np.zeros(ladder_width, dtype=np.float64)

    out_signal_idx, out_entry_idx, out_dir = [], [], []
    out_levels, out_lo_px, out_hi_px, out_vol, out_bar = [], [], [], [], []
    out_win, out_row = [], []

    origin = 0.0
    armed = {1: [], -1: []}   # direction -> list of fired (lo, hi) ranges, concurrently active
    # Stacks awaiting breakout confirmation: direction -> list of
    # (lo, hi, vol, win). Only used when require_breakout is set. Cleared on
    # every bar roll, because the ladder they refer to is cleared too.
    pending = {1: [], -1: []}
    overflow = 0

    def _emit(i_, direction_, lo_, hi_, vol_, win_):
        out_signal_idx.append(indices[i_])
        out_entry_idx.append(indices[i_ + 1])
        out_dir.append(direction_)
        out_levels.append(hi_ - lo_ + 1)
        out_lo_px.append(origin + (lo_ - half) * tick_size)
        out_hi_px.append(origin + (hi_ - half) * tick_size)
        out_vol.append(vol_)
        out_win.append(win_)
        out_bar.append(bar_ids[i_])
        out_row.append(i_)

    for i in range(n):
        new_bar = (i == 0) or (bar_ids[i] != bar_ids[i - 1]) or (session_ids[i] != session_ids[i - 1])
        if new_bar:
            bid[:] = 0.0
            ask[:] = 0.0
            origin = prices[i]
            armed = {1: [], -1: []}
            pending = {1: [], -1: []}

        p = int(round((prices[i] - origin) / tick_size)) + half
        if p < 1 or p >= ladder_width - 1:
            overflow += 1
            continue

        if trade_types[i] == 2:
            ask[p] += volumes[i]
        elif trade_types[i] == 1:
            bid[p] += volumes[i]

        # Breakout confirmation. A buy stack is confirmed by a print strictly
        # above its top level, a sell stack by one strictly below its bottom.
        # Touching the boundary is not breaking it.
        if require_breakout and i + 1 < n:
            for direction in (1, -1):
                if not pending[direction]:
                    continue
                still_pending = []
                for (lo_p, hi_p, vol_p, win_p) in pending[direction]:
                    broke = (p > hi_p) if direction == 1 else (p < lo_p)
                    if broke:
                        _emit(i, direction, lo_p, hi_p, vol_p, win_p)
                    else:
                        still_pending.append((lo_p, hi_p, vol_p, win_p))
                pending[direction] = still_pending

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
                lo, hi, vol, win = _scan_stack(
                    ask, bid, cand, direction, ratio, n_consecutive,
                    min_diagonal_volume, min_winning_volume,
                    min_level_diagonal_volume, eps,
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
                if require_breakout:
                    # Hold it back: the stack has formed, but nothing has yet
                    # traded through it. It signals only once price does.
                    pending[direction].append((lo, hi, vol, win))
                else:
                    _emit(i, direction, lo, hi, vol, win)

    return (
        out_signal_idx, out_entry_idx, out_dir, out_levels,
        out_lo_px, out_hi_px, out_vol, out_win, out_bar, out_row, overflow,
    )


def find_stacked_imbalances(
    ticks: pl.DataFrame,
    tick_size: float,
    imbalance_ratio: float = 3.0,
    n_consecutive: int = 3,
    min_diagonal_volume: float = 0.0,
    min_winning_volume: float = 0.0,
    min_level_diagonal_volume: float = 0.0,
    require_breakout: bool = False,
    bar_id_col: str = "volume_bar_id",
    eps: float = 1e-6,
    ladder_width: int = 512,
) -> pl.DataFrame:
    """Find stacked diagonal imbalances tick by tick.

    A stack is ``n_consecutive`` or more adjacent price levels where the same
    side dominates its diagonal counterpart by at least ``imbalance_ratio``,
    and where the summed volume of both legs of every diagonal pair reaches
    ``min_diagonal_volume``.

    ``min_winning_volume`` is a second, independent size floor applied to the
    winning side alone -- the summed ask across a buy stack, the summed bid
    across a sell stack.  ``min_diagonal_volume`` includes the losing leg it
    was measured against, so at a 3:1 ratio up to a third of it is size that
    lost; this floor filters on the aggressive size only.  Both default to
    ``0.0`` (inert).

    ``require_breakout`` withholds a completed stack until price trades
    strictly beyond it -- above the top level for a buy stack, below the
    bottom level for a sell stack.  A stack that is never broken through never
    signals: the dominant side's aggression was absorbed rather than achieving
    anything, so the imbalance marks trapped participants, not initiative.
    When set, ``signal_index`` is the confirming tick, not the tick that
    completed the stack, and ``entry_index`` is the tick after it.  A pending
    stack expires when the bar rolls, since the ladder it refers to resets.

    Both floors gate *arming*, not just reporting: a stack rejected by either
    never arms, so its price range stays free and the run can fire later,
    deeper, once volume accumulates.  Raising a floor therefore changes which
    stacks fire, not merely which are kept -- results cannot be reproduced by
    filtering the output of a lower-floor run.

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
    if min_diagonal_volume < 0:
        raise ValueError(
            f"min_diagonal_volume must be >= 0, got {min_diagonal_volume}"
        )
    if min_winning_volume < 0:
        raise ValueError(
            f"min_winning_volume must be >= 0, got {min_winning_volume}"
        )
    if min_level_diagonal_volume < 0:
        raise ValueError(
            "min_level_diagonal_volume must be >= 0, got "
            f"{min_level_diagonal_volume}"
        )
    if ticks[bar_id_col].null_count() > 0:
        raise ValueError(f"{bar_id_col} must not contain nulls")

    bar_ids = ticks[bar_id_col].to_numpy()
    session_codes = ticks["SessionType"].cast(pl.Categorical).to_physical().to_numpy()

    (sig, ent, dirs, levels, lo_px, hi_px, vols, wins, bars, rows, overflow) = _detect(
        ticks["Price"].to_numpy().astype(np.float64),
        ticks["Volume"].to_numpy().astype(np.float64),
        ticks["TradeType"].to_numpy().astype(np.int64),
        bar_ids.astype(np.int64),
        session_codes.astype(np.int64),
        ticks["Index"].to_numpy().astype(np.int64),
        float(tick_size), float(imbalance_ratio), int(n_consecutive),
        float(min_diagonal_volume), float(min_winning_volume),
        float(min_level_diagonal_volume), bool(require_breakout),
        float(eps), int(ladder_width),
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
            "winning_volume": pl.Series(wins, dtype=pl.Float64),
            "bar_id": pl.Series(bars, dtype=pl.Int64),
            "Datetime": pl.Series(
                [datetimes[r] for r in rows], dtype=ticks["Datetime"].dtype
            ),
        }
    ).sort("entry_index")
