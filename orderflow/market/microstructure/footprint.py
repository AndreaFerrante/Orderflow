import pandas as pd
import numpy as np


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
