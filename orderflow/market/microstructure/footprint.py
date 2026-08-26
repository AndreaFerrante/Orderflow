import pandas as pd
import numpy as np


def _buy_flag(ask, bid, p, ratio, eps):
    """Buy imbalance at level ``p``: ask at P against bid one level BELOW.

    ``TradeType 2`` is ask volume, buy aggression, LONG.  The level below is
    ``p - 1`` because the ladder index increases with price.

    A level with zero on *both* sides is untouched ladder padding, not an
    observed zero, and must not trip the zero-denominator bypass below it —
    otherwise every padded cell next to a real print would spuriously flag.
    """
    if bid[p - 1] <= 0 and ask[p - 1] <= 0:
        return False
    return bool(ask[p] >= ratio * (bid[p - 1] + eps))


def _sell_flag(ask, bid, p, ratio, eps):
    """Sell imbalance at level ``p``: bid at P against ask one level ABOVE.

    ``TradeType 1`` is bid volume, sell aggression, SHORT. Same untouched-cell
    guard as ``_buy_flag``, mirrored on the level above.
    """
    if ask[p + 1] <= 0 and bid[p + 1] <= 0:
        return False
    return bool(bid[p] >= ratio * (ask[p + 1] + eps))


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
