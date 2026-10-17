"""
Numba kernels for the per-tick, session-reset loops of ``volume_profile`` and ``vwap``.

Every kernel reproduces, operation for operation, the pure-Python loop kept beside its public
function (``_<name>_py``); ``tests/test_session_kernels.py`` holds the two equal on random tapes.
The speed comes from prices becoming integer *levels* (the rank of each price among the frame's
distinct prices), so a session's profile is a flat array instead of a dict that is sorted again
on every tick. The KDE loop also works on several threads, as a profile never crosses the
RTH -> ETH reset; the reference it replaces already ran its density in parallel on every tick.

A kernel runs only on a frame it is known to reproduce: ``session_arrays`` and
``kde_parameters_ok`` say which. Every other input goes to the reference loop, which then
behaves, raises, prints and warns as it always did.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from numba import get_num_threads  # noqa: F401  (how many threads ``peaks_valleys`` may use: its caller asks)

from orderflow.core._kernel_calls import WATCH_EVERY, cached_njit

# Gaussian terms this many bandwidths away are exactly 0.0 in float64 (exp(-0.5 * 39**2) lies
# far below the smallest subnormal), so leaving them out of a sum changes no bit of it.
_KDE_ZERO_Z = 39.0
# Sessions with at most this many price levels keep their Gaussian terms in a table
# (levels**2 * 8 bytes per thread, 34 MB at the cap); wider ones call exp() for every term.
KDE_TABLE_MAX_LEVELS = 2048

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
# Up to this total the running sums of an integer volume are exact in float64, as they are in
# the reference's integer arithmetic; far beyond it lies the overflow of int(total * 0.68).
_MAX_TOTAL_VOLUME = 2.0 ** 53
# Above this a squared price difference times a volume overflows: the reference then warns.
_MAX_PRICE = 1e100
# The half distance between two extrema, in ticks, must fit the kernel's int64.
_MAX_TICKS = 2.0 ** 62

# The look-out every kernel that walks the ticks keeps for ``orderflow.core._kernel_calls.run_kernel``,
# which hands it ``watch`` and ``slot`` as its last two arguments. On tick ``i`` with ``i & _LOOK == 0``
# the kernel returns if ``watch[0]`` is set (its caller throws away whatever it did) and leaves the
# number of ticks behind it in ``watch[slot]``. They are compiled ``nogil``: the thread that waits
# for one has to run meanwhile, as every other thread did while the loop ran.
_LOOK = WATCH_EVERY - 1
# A tick costs the KDE kernel about a thousand times more: it looks more often.
_KDE_LOOK = 127


def _column(data, name):
    """The column as an array, or ``None`` when it is missing or is not one plain column."""
    if name not in data.columns:
        return None
    column = data[name]
    if not isinstance(column, pd.Series):
        return None
    return np.asarray(column)


def numeric_column(data, name):
    """A column of numbers or booleans as an array, else ``None``."""
    values = _column(data, name)
    if values is None or values.dtype.kind not in "iufb":
        return None
    return values


def session_arrays(data, with_levels: bool = True, with_volume: bool = True):
    """
    Arrays the kernels work on, or ``None`` when only the reference loop reproduces the frame.

    Returns ``(price, volume, reset, level, levels)``: ``price`` and ``volume`` as float64,
    ``reset[i]`` True where tick ``i`` opens a session (a label ending in ``ETH`` right after a
    different one ending in ``RTH``), ``level[i]`` the rank of ``price[i]`` in the sorted distinct
    prices ``levels``. ``with_levels=False`` skips the ranking and returns ``None`` for both;
    ``with_volume=False`` does not ask for a ``Volume`` column and returns ``None`` for it.

    The kernels take what the enrichment produces: a pandas frame with a float64 ``Price``, an
    int64 or float64 ``Volume`` and string session labels. ``None`` covers everything else,
    where array arithmetic and the reference part ways or the reference raises or warns: another
    frame type, a missing or duplicated column, another dtype (the reference sums a float32 or
    int8 volume in that type and computes the density of an integer price in integers), an empty
    frame, a price that is not finite, is a negative zero or is beyond 1e100, a volume that is
    not finite or is negative, a total volume beyond 2**53, a session label that is not a string.
    """
    if not isinstance(data, pd.DataFrame) or len(data) == 0:
        return None
    price = _column(data, "Price")
    if price is None or price.dtype != _FLOAT64:
        return None
    price = np.ascontiguousarray(price)
    # ``not ... <`` rather than ``>=``: a NaN fails the comparison and is refused with the rest
    if not np.abs(price).max() < _MAX_PRICE or (np.signbit(price) & (price == 0)).any():
        return None

    volume = None
    if with_volume:
        volume = _column(data, "Volume")
        if volume is None or (volume.dtype != _FLOAT64 and volume.dtype != _INT64):
            return None
        volume = np.ascontiguousarray(volume, dtype=np.float64)
        if np.signbit(volume).any() or not volume.sum() <= _MAX_TOTAL_VOLUME:  # NaN and inf fail the sum
            return None

    if "SessionType" not in data.columns or not isinstance(data["SessionType"], pd.Series):
        return None
    session, labels = pd.factorize(data["SessionType"])
    if (session < 0).any() or not all(isinstance(label, str) for label in labels):
        return None
    eth = np.array([label.endswith("ETH") for label in labels], dtype=np.bool_)
    rth = np.array([label.endswith("RTH") for label in labels], dtype=np.bool_)
    reset = np.zeros(len(price), dtype=np.bool_)
    reset[1:] = (session[1:] != session[:-1]) & eth[session[1:]] & rth[session[:-1]]

    if not with_levels:
        return price, volume, reset, None, None
    level, levels = pd.factorize(price, sort=True)
    return price, volume, reset, level.astype(np.int64, copy=False), np.asarray(levels, dtype=np.float64)


def kde_parameters_ok(price, tick_size, bandwidth) -> bool:
    """
    Whether ``peaks_valleys`` reproduces the reference for this tick size and bandwidth.

    Both must be plain positive numbers: the reference multiplies a float32 bandwidth in float32,
    and divides by whatever else it is handed. The bandwidth must leave the density finite, and
    the price range must be a number of ticks an int64 holds.
    """
    for value in (tick_size, bandwidth):
        if not isinstance(value, (int, float, np.integer)) or not 0 < value < _MAX_TICKS:
            return False
    return bool(np.isfinite(_MAX_TOTAL_VOLUME / bandwidth) and (price.max() - price.min()) / tick_size < _MAX_TICKS)


def session_starts(reset: np.ndarray) -> np.ndarray:
    """Position of the first tick of every session: 0, then each reset."""
    return np.concatenate((np.zeros(1, dtype=np.int64), np.flatnonzero(reset).astype(np.int64)))


# The kernels whose arrays go straight back to the caller of the public function (``session_low_high``,
# ``moving_poc``, ``node_volume``, ``peaks_valleys``) fill arrays of zeros that caller made with NumPy.
# An array Numba allocates and returns does not own its memory (``flags.owndata`` False, ``base``
# set, no ``resize`` in place); the loops returned plain arrays, and a copy would cost their size again.


@cached_njit(nogil=True)
def cumulative_delta(volume, is_bid, reset, watch, slot):
    """Running ask, bid and total volume of the session: (cd_ask, cd_bid, total)."""
    n = volume.size
    cd_ask = np.zeros(n)
    cd_bid = np.zeros(n)
    total = np.zeros(n)
    ask = 0.0
    bid = 0.0
    both = 0.0
    for i in range(n):
        if i & _LOOK == 0:
            if watch[0] != 0:
                break
            watch[slot] = i
        if reset[i]:
            ask = 0.0
            bid = 0.0
            both = 0.0
        v = volume[i]
        both += v
        if is_bid[i]:
            bid += v
        else:
            ask += v
        cd_ask[i] = ask
        cd_bid[i] = bid
        total[i] = both
    return cd_ask, cd_bid, total


@cached_njit(nogil=True)
def session_low_high(price, reset, lows, highs, watch, slot):
    """Running low and high of the session, written into ``lows`` and ``highs``."""
    n = price.size
    low = price[0]
    high = price[0]
    for i in range(n):
        if i & _LOOK == 0:
            if watch[0] != 0:
                return
            watch[slot] = i
        p = price[i]
        if reset[i]:
            low = p
            high = p
        if p > high:
            high = p
        highs[i] = high
        if p < low:
            low = p
        lows[i] = low


@cached_njit(nogil=True)
def moving_poc(volume, reset, level, levels, poc, prev, watch, slot):
    """
    Running point of control of the session and the final one of the session before, written into
    ``poc`` and ``prev``.

    The reference takes ``max`` over a dict, which keeps the first price traded among those tied
    on volume. Volume only grows, so the leader changes only when the level just traded passes it,
    or ties it having been traded earlier in the session: ``order`` holds that arrival rank.
    """
    n = volume.size
    n_levels = levels.size
    vol = np.zeros(n_levels)
    order = np.full(n_levels, -1, np.int64)
    traded = np.empty(n_levels, np.int64)
    n_traded = 0
    best = -1
    prev_poc = 0.0
    for i in range(n):
        if i & _LOOK == 0:
            if watch[0] != 0:
                return
            watch[slot] = i
        g = level[i]
        if reset[i]:
            prev_poc = poc[i - 1]
            for k in range(n_traded):
                vol[traded[k]] = 0.0
                order[traded[k]] = -1
            n_traded = 0
            best = -1
        if order[g] < 0:
            order[g] = n_traded
            traded[n_traded] = g
            n_traded += 1
        vol[g] += volume[i]
        if best < 0 or vol[g] > vol[best] or (vol[g] == vol[best] and order[g] < order[best]):
            best = g
        poc[i] = levels[best]
        prev[i] = prev_poc


@cached_njit(nogil=True)
def node_volume(volume, is_type_1, is_type_2, reset, level, n_levels, out_total, out_1, out_2, out_session,
                watch, slot):
    """
    Session volume at the tick's price, written into ``out_total``, ``out_1`` (slot 1), ``out_2``
    (slot 2) and ``out_session``, four arrays of zeros.

    ``slot_1`` adds the volume of ``TradeType`` 1 and ``slot_2`` that of ``TradeType`` 2, as the
    reference does from the second tick on. For the very first tick of the frame the reference
    fills the two slots the other way round and writes its own outputs; both are kept.
    """
    n = volume.size
    at_price = np.zeros(n_levels)
    slot_1 = np.zeros(n_levels)
    slot_2 = np.zeros(n_levels)
    is_traded = np.zeros(n_levels, np.bool_)
    traded = np.empty(n_levels, np.int64)

    g = level[0]
    v = volume[0]
    is_traded[g] = True
    traded[0] = g
    n_traded = 1
    at_price[g] = v
    if is_type_2[0]:
        slot_1[g] = v
    if is_type_1[0]:
        slot_2[g] = v
        out_2[0] = v
    else:
        out_1[0] = v
    out_total[0] = v
    out_session[0] = v
    session_volume = v

    for i in range(1, n):
        if i & _LOOK == 0:
            if watch[0] != 0:
                return
            watch[slot] = i
        g = level[i]
        v = volume[i]
        if reset[i]:
            for k in range(n_traded):
                at_price[traded[k]] = 0.0
                slot_1[traded[k]] = 0.0
                slot_2[traded[k]] = 0.0
                is_traded[traded[k]] = False
            n_traded = 0
            session_volume = 0.0
        session_volume += v
        if not is_traded[g]:
            is_traded[g] = True
            traded[n_traded] = g
            n_traded += 1
        at_price[g] += v
        if is_type_1[i]:
            slot_1[g] += v
        elif is_type_2[i]:
            slot_2[g] += v
        out_total[i] = at_price[g]
        out_1[i] = slot_1[g]
        out_2[i] = slot_2[g]
        out_session[i] = session_volume


@cached_njit(nogil=True, error_model="numpy")
def session_vwap(price, volume, reset, offsets, watch, slot):
    """
    Session VWAP and its bands: (vwap, top, bottom), the bands as one row per offset.

    ``error_model="numpy"`` keeps the reference's arithmetic on a session that opens with
    zero-volume ticks: 0 / 0 is NaN there, not an exception.
    """
    n = price.size
    n_bands = offsets.size
    vwap = np.zeros(n)
    top = np.zeros((n_bands, n))
    bottom = np.zeros((n_bands, n))
    sum_volume = volume[0]
    sum_price_volume = price[0] * volume[0]
    sum_squares = 0.0
    for i in range(1, n):
        if i & _LOOK == 0:
            if watch[0] != 0:
                break
            watch[slot] = i
        if reset[i]:
            sum_price_volume = 0.0
            sum_volume = 0.0
            sum_squares = 0.0
        p = price[i]
        v = volume[i]
        sum_price_volume += p * v
        sum_volume += v
        w = sum_price_volume / sum_volume
        vwap[i] = w
        sum_squares += (p - w) * (p - w) * v
        if sum_volume != 0.0:
            band = np.sqrt(sum_squares / sum_volume)
        else:
            band = 1.0
        for k in range(n_bands):
            top[k, i] = w + offsets[k] * band
            bottom[k, i] = w - offsets[k] * band
    vwap[0] = price[0]
    for k in range(n_bands):
        top[k, 0] = price[0]
        bottom[k, 0] = price[0]
    return vwap, top, bottom


@cached_njit(nogil=True)
def value_area_labels(volume, level, starts, value_area, watch, slot):
    """
    Per-tick value-area label of the session profile: 0 outside, 1 a new point of control,
    2 inside the value area.

    The reference sorts the session's prices on every tick and walks out from the point of
    control one traded price at a time. Here the profile is an array over the session's price
    range and the walk steps over the prices not traded yet.

    No ``parallel=True``: the loop this replaces was plain Python, safe to call from several Python
    threads at once, and a parallel region is not under every Numba threading layer.
    """
    n = volume.size
    n_sessions = starts.size
    out = np.zeros(n, np.int8)
    for s in range(n_sessions):
        a = starts[s]
        b = starts[s + 1] if s + 1 < n_sessions else n
        g_low = level[a]
        g_high = level[a]
        for i in range(a + 1, b):
            g = level[i]
            if g < g_low:
                g_low = g
            elif g > g_high:
                g_high = g
        vol = np.zeros(g_high - g_low + 1)
        is_traded = np.zeros(g_high - g_low + 1, np.bool_)
        low = -1
        high = -1
        total = 0.0
        poc_volume = 0.0
        poc = 0
        for i in range(a, b):
            if i & _LOOK == 0:
                if watch[0] != 0:
                    return out
                watch[slot] = i
            here = level[i] - g_low
            if not is_traded[here]:
                is_traded[here] = True
                if low < 0:
                    low = here
                    high = here
                elif here < low:
                    low = here
                elif here > high:
                    high = here
            vol[here] += volume[i]
            total += volume[i]
            if poc_volume == 0.0 or vol[here] > poc_volume:
                poc_volume = vol[here]
                poc = here
                out[i] = 1
                continue

            target = int(total * value_area)
            covered = poc_volume
            advance_up = True
            advance_down = True
            upper_volume = 0.0
            lower_volume = 0.0
            upper = poc
            lower = poc
            while True:
                moved = False
                if advance_up:
                    if upper >= high:
                        upper_volume = 0.0
                    else:
                        upper += 1
                        while not is_traded[upper]:
                            upper += 1
                        upper_volume += vol[upper]
                        moved = True
                if advance_down:
                    if lower <= low:
                        lower_volume = 0.0
                    else:
                        lower -= 1
                        while not is_traded[lower]:
                            lower -= 1
                        lower_volume += vol[lower]
                        moved = True

                if (upper_volume > 0.0 or lower_volume > 0.0) and upper_volume > lower_volume:
                    advance_up = True
                    advance_down = False
                    covered += upper_volume
                    upper_volume = 0.0
                    if here == upper:
                        out[i] = 2
                        break
                elif upper_volume > 0.0 or lower_volume > 0.0:
                    advance_up = False
                    advance_down = True
                    covered += lower_volume
                    lower_volume = 0.0
                    if here == lower:
                        out[i] = 2
                        break
                elif not moved:
                    if advance_up and advance_down:
                        break
                    advance_up = True
                    advance_down = True

                if covered > target:
                    break
    return out


@cached_njit(error_model="numpy")
def _kde_at(j, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table):
    """
    Kernel density at level ``j`` for the profile as it stands on ``tick``, once per tick.

    The terms are added in rising price order, as the reference does, so the sum carries the
    same rounding. Levels not traded yet have weight 0 and add exactly 0.0.
    """
    if stamp[j] == tick:
        return cache[j]
    start = first[j] if first[j] > low else low
    stop = last[j] if last[j] < high else high
    total = 0.0
    xj = px[j]
    if use_table:
        # Each Gaussian term is computed on first use, here, one exp() call at a time: filled ahead
        # in a loop of its own it could be vectorised (SVML on x86) and round differently from the
        # scalar exp() the reference calls.
        row = table[j]
        for t in range(start, stop + 1):
            gauss = row[t]
            if gauss < 0.0:
                expo = (xj - px[t]) / h
                gauss = np.exp(-0.5 * expo * expo)
                row[t] = gauss
            val = g_const * gauss
            val *= weight[t]
            total += val
    else:
        for t in range(start, stop + 1):
            expo = (xj - px[t]) / h
            val = g_const * np.exp(-0.5 * expo * expo)
            val *= weight[t]
            total += val
    stamp[j] = tick
    cache[j] = total
    return total


@cached_njit(error_model="numpy")
def _extremum_above(c, below, tick, is_traded, stamp, cache, px, weight, low, high, first, last,
                    g_const, h, table, use_table):
    """First level from ``c`` upwards that is a strict local extremum of the density, or the top one."""
    while c != high:
        above = c + 1
        while not is_traded[above]:
            above += 1
        kc = _kde_at(c, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        kb = _kde_at(below, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        ka = _kde_at(above, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        if (kc > kb and kc > ka) or (kc < kb and kc < ka):
            return c
        below = c
        c = above
    return c


@cached_njit(error_model="numpy")
def _extremum_below(c, above, tick, is_traded, stamp, cache, px, weight, low, high, first, last,
                    g_const, h, table, use_table):
    """First level from ``c`` downwards that is a strict local extremum of the density, or the bottom one."""
    while c != low:
        below = c - 1
        while not is_traded[below]:
            below -= 1
        kc = _kde_at(c, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        kb = _kde_at(below, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        ka = _kde_at(above, tick, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
        if (kc > kb and kc > ka) or (kc < kb and kc < ka):
            return c
        above = c
        c = below
    return c


@cached_njit()
def _traded_level_below(c, steps, is_traded, low, high, count):
    """
    The traded level ``steps`` places below ``c`` in the sorted profile, with Python's index rules:
    past the bottom it counts back from the top, and -1 stands for the reference's IndexError.
    """
    while steps > 0 and c > low:
        c -= 1
        while not is_traded[c]:
            c -= 1
        steps -= 1
    if steps == 0:
        return c
    if steps > count:
        return -1
    c = high
    steps -= 1
    while steps > 0:
        c -= 1
        while not is_traded[c]:
            c -= 1
        steps -= 1
    return c


def peaks_valleys(volume, level, levels, starts, tick_size, h, table_max_levels, threads, out, watch, slot):
    """
    Per-tick position of the price in the kernel-smoothed session profile, written into ``out`` (an
    array of zeros); returns ``unresolved``.

    Labels are those of the reference: 2 / -2 on a peak or a valley of the density, 1 / -1 on the
    side of one, 0 undefined. ``unresolved`` counts the ticks where the reference hits an
    IndexError and writes 0.

    On ``threads`` threads (the caller's ``numba.get_num_threads()``, which the reference's parallel
    density obeyed too) the sessions are dealt one at a time, the biggest first, each counting its
    ticks in a slot of its own from ``slot`` on: ``watch`` then has one per session. They are Python
    threads around a ``nogil`` kernel, not a Numba ``prange``: two parallel regions started from two
    Python threads abort the process under Numba's workqueue layer, and a static split of the
    sessions kept a 56.8M-tick file waiting 65 s on the thread with the busiest run, against 36 s
    for its longest session alone. A frame too short to be run on a thread stays on this one.
    """
    n_sessions = starts.size
    if threads > 1 and n_sessions > 1 and volume.size >= WATCH_EVERY and watch.size >= slot + n_sessions:
        ends = np.append(starts[1:], volume.size)
        biggest_first = np.argsort(starts - ends, kind="stable")

        def session(s):
            return _peaks_valleys_of(volume, level, levels, starts, s, s + 1, tick_size, h, table_max_levels,
                                     out, watch, slot + s)

        try:
            with ThreadPoolExecutor(min(threads, n_sessions)) as pool:
                return sum(pool.map(session, biggest_first))
        except RuntimeError:  # no thread to be had (see ``_kernel_calls._started``): all of it right here
            pass
    return _peaks_valleys_of(volume, level, levels, starts, 0, n_sessions, tick_size, h, table_max_levels,
                             out, watch, slot)


@cached_njit(nogil=True, error_model="numpy")
def _peaks_valleys_of(volume, level, levels, starts, first, last, tick_size, h, table_max_levels, out,
                      watch, slot):
    """
    ``peaks_valleys`` for the sessions ``first`` .. ``last - 1``; returns their ``unresolved``.

    The reference computes the density at every level on every tick and then looks only at the
    extrema next to the price. Here the density is computed where that search actually reads it:
    at the price and outwards to the first extremum on each side. A session of at most
    ``table_max_levels`` price levels reads its Gaussian terms from a table (``KDE_TABLE_MAX_LEVELS``).
    """
    n = volume.size
    n_sessions = starts.size
    unresolved = 0
    behind = 0  # ticks of the sessions this call is through with
    for s in range(first, last):
        a = starts[s]
        b = starts[s + 1] if s + 1 < n_sessions else n
        if b > n - 1:
            b = n - 1  # the reference loop stops one tick before the end of the frame
        if a >= b:
            continue
        behind -= a
        g_low = level[a]
        g_high = level[a]
        for i in range(a + 1, b):
            g = level[i]
            if g < g_low:
                g_low = g
            elif g > g_high:
                g_high = g
        n_px = g_high - g_low + 1
        px = levels[g_low:g_high + 1]
        weight = np.zeros(n_px)
        is_traded = np.zeros(n_px, np.bool_)
        cache = np.empty(n_px)
        stamp = np.full(n_px, -1, np.int64)

        # first[j]..last[j]: the levels whose Gaussian term at j is not exactly 0.0
        first = np.empty(n_px, np.int64)
        last = np.empty(n_px, np.int64)
        left = 0
        right = 0
        for j in range(n_px):
            while (px[j] - px[left]) / h >= _KDE_ZERO_Z:
                left += 1
            if right < j:
                right = j
            while right + 1 < n_px and (px[right + 1] - px[j]) / h < _KDE_ZERO_Z:
                right += 1
            first[j] = left
            last[j] = right

        use_table = n_px <= table_max_levels
        if use_table:
            table = np.full((n_px, n_px), -1.0)  # -1: not computed yet (a Gaussian term is never negative)
        else:
            table = np.empty((1, 1))

        low = -1
        high = -1
        count = 0
        g_const = 0.0
        for i in range(a, b):
            if i & _KDE_LOOK == 0:
                if watch[0] != 0:
                    return unresolved
                watch[slot] = behind + i
            here = level[i] - g_low
            if not is_traded[here]:
                is_traded[here] = True
                count += 1
                if low < 0:
                    low = here
                    high = here
                elif here < low:
                    low = here
                elif here > high:
                    high = here
                # One factor for every density of the tick, so no label depends on it beyond the
                # last bit of a product: it is here to carry the reference's rounding.
                g_const = 1.0 / np.sqrt(2.0 * np.pi) / (count * h)
            weight[here] += volume[i]
            if count < 2:
                continue  # one level has no extrema; this covers the frame's first tick, never labelled
            if here == low or here == high:
                out[i] = -1.0
                continue

            below = here - 1
            while not is_traded[below]:
                below -= 1
            above = here + 1
            while not is_traded[above]:
                above += 1
            k_here = _kde_at(here, i, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
            k_below = _kde_at(below, i, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)
            k_above = _kde_at(above, i, stamp, cache, px, weight, low, high, first, last, g_const, h, table, use_table)

            next_extremum = _extremum_above(above, here, i, is_traded, stamp, cache, px, weight, low, high,
                                            first, last, g_const, h, table, use_table)
            if (k_here > k_below and k_here > k_above) or (k_here < k_below and k_here < k_above):
                if weight[here] < weight[next_extremum]:
                    out[i] = -2.0
                    continue
                if weight[here] > weight[next_extremum]:
                    out[i] = 2.0
                    continue
                upper = here
            else:
                upper = next_extremum
            lower = _extremum_below(below, here, i, is_traded, stamp, cache, px, weight, low, high,
                                    first, last, g_const, h, table, use_table)

            half = int((px[upper] - px[lower]) / tick_size / 2)
            threshold = _traded_level_below(upper, half, is_traded, low, high, count)
            if threshold < 0:
                unresolved += 1
                continue
            rising = weight[upper] > weight[lower]
            falling = weight[upper] < weight[lower]
            if px[here] >= px[threshold]:
                if rising:
                    out[i] = 1.0
                elif falling:
                    out[i] = -1.0
            else:
                if rising:
                    out[i] = -1.0
                elif falling:
                    out[i] = 1.0
        behind += b
    return unresolved
