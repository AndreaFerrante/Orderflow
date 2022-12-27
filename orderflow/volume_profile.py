import pandas as pd
import numpy as np
import operator
from tqdm import tqdm


def get_dynamic_cumulative_delta(data: pd.DataFrame) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this function returns a dataframe with volume in ASK / BID restarted at the
    of a given day.
    :param data: canonical dataframe recorded (tick-by-tick wth the DOM attached)
    :return: canonical dataframe with the addition of the column with Ask / Bid volume
    """

    data.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
    dates = data.Date.unique()
    datas = list()

    for date in tqdm(dates):

        print(f"Dynamic VP processing date {date}")
        ask = data[
            (data.Date == date) & (data.TradeType == 2)
        ]  # get the ask per date...
        bid = data[
            (data.Date == date) & (data.TradeType == 1)
        ]  # get the bid per date...

        ask["AskVolume_VP"] = np.cumsum(ask.Volume)
        ask["BidVolume_VP"] = np.zeros(ask.shape[0])
        bid["AskVolume_VP"] = np.zeros(bid.shape[0])
        bid["BidVolume_VP"] = np.cumsum(bid.Volume)

        single_date = pd.concat([ask, bid], axis=0)
        single_date.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
        single_date["AskVolume_VP"] = (
            single_date["AskVolume_VP"]
            .replace(to_replace=0, method="ffill")
            .astype(np.int64)
        )  # Fill zeros with last cumulative value
        single_date["BidVolume_VP"] = (
            single_date["BidVolume_VP"]
            .replace(to_replace=0, method="ffill")
            .astype(np.int64)
        )  # Fill zeros with last cumulative value

        datas.append(single_date)

    datas = pd.concat(datas)
    datas.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)

    return datas


def get_dynamic_cumulative_delta_with_volume_filter(
    data: pd.DataFrame, volume_filter: int
) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this function returns a dataframe with volume in ASK / BID restarted at the
    beginning of a given day.
    :param data: canonical dataframe recorded (tick-by-tick wth the DOM attached)
    :return: canonical dataframe with the addition of the column with Ask / Bid volume filtered
    """

    # Check if index, as unique trade on the T&S key, is present in the DataFrame...
    # ... this is crucial to have a unique key for every single trade recorded.
    index_present = "index".upper() in [str(x).upper() for x in data.columns]
    if not index_present:
        data.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
        data["Index"] = np.arange(0, data.shape[0], 1)

    dates = data.Date.unique()
    datas = list()

    for date in dates:

        print(f"Dynamic VP filtered processing date {date}")
        ask = data[
            (data.Date == date) & (data.TradeType == 2) & (data.Volume >= volume_filter)
        ]  # get the ask per date...
        bid = data[
            (data.Date == date) & (data.TradeType == 1) & (data.Volume >= volume_filter)
        ]  # get the bid per date...

        ask = ask.assign(AskVolume_VP_f=np.cumsum(ask.Volume))
        ask = ask.assign(BidVolume_VP_f=[np.nan] * ask.shape[0])
        bid = bid.assign(AskVolume_VP_f=[np.nan] * bid.shape[0])
        bid = bid.assign(BidVolume_VP_f=np.cumsum(bid.Volume))

        single_date = pd.concat([ask, bid], axis=0)
        single_date.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
        datas.append(single_date)

    datas = pd.concat(datas)
    datas.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)

    data = data.merge(
        right=datas[["AskVolume_VP_f", "BidVolume_VP_f", "Index"]],
        how="left",
        on="Index",  # index is the unique single trade
    )

    data.fillna(0, inplace=True)
    data["AskVolume_VP_f_" + str(volume_filter)] = (
        data["AskVolume_VP_f"].replace(to_replace=0, method="ffill").astype(np.int64)
    )  # Fill zeros with last cumulative value
    data["BidVolume_VP_f_" + str(volume_filter)] = (
        data["BidVolume_VP_f"].replace(to_replace=0, method="ffill").astype(np.int64)
    )  # Fill zeros with last cumulative value

    return data.drop(["AskVolume_VP_f", "BidVolume_VP_f", "Index"], axis=1)


def get_daily_moving_POC(df: pd.DataFrame) -> np.array:

    """
    Given the canonical dataframe recorded, this function returns the Point of Control (i.e. POC) that is moving during
    the day giving the volume sentiment of uptrending market or choppy market of downtrending market.
    :param df: anonical dataframe recorded
    :return: numpy array for the daily moving poc
    """

    volume = np.array(df.Volume)
    price = np.array(df.Price)
    date = np.array(df.Date)
    poc_final = {}
    len_ = len(price)
    poc_ = np.zeros(len_)

    poc_final[price[0]] = volume[0]
    poc_[0] = price[0]

    for i in tqdm(range(1, len_ - 1)):
        cp = price[i]
        if date[i] != date[i - 1]:
            poc_final.clear()
            poc_final[cp] = volume[i]
            poc_[i] = cp
        else:
            if cp in poc_final:
                poc_final[cp] += volume[i]
            else:
                poc_final[cp] = volume[i]
            poc_[i] = max(poc_final.items(), key=operator.itemgetter(1))[0]

    poc_[len_ - 1] = price[len_ - 1]

    return poc_




