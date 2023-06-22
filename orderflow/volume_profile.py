import pandas as pd
import numpy as np
import operator
from tqdm import tqdm
import orderflow as of
from pytictoc import TicToc


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
    :param df: canonical dataframe recorded
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


def get_volume_profile_areas(df: pd.DataFrame) -> np.array:
    """
    Given the canonical dataframe recorded, this function returns an array with info if the price is in the VA or not
    :param df: canonical dataframe recorded
    :return: numpy array with values: 1 = VA, 0 = No VA
    """
    price = np.array(df.Price)
    volume = np.array(df.Volume)
    date = np.array(df.Date)
    total_volume = 0
    poc = 0.00
    poc_volume = 0
    poc_index = 0
    len_ = len(price)
    value_area = np.zeros(len_)
    volume_profile = {}

    print(f"Calculate VP Value Area...")

    for i in tqdm(range(len_ - 1)):
        if date[i + 1] == date[i]:
            pass
        else:
            volume_profile.clear()
            total_volume = 0
            poc = 0
            poc_volume = 0
            poc_index = 0

        if price[i] in volume_profile.keys():
            volume_profile[price[i]] += volume[i]
        else:
            volume_profile[price[i]] = volume[i]

        if volume_profile[price[i]] > poc_volume:
            poc_volume = volume_profile[price[i]]
            poc = price[i]
            value_area[i] = 1
            continue

        total_volume += volume[i]

        vp_prices = np.array(sorted(volume_profile.keys()))
        vp_volumes = np.array([volume_profile[key] for key in vp_prices])

        poc_index = np.where(vp_prices == poc)[0][0]
        percentage_of_total_volume = int(total_volume * 0.8)
        current_sum_of_volume = poc_volume

        advance_up = True
        advance_down = True
        upper_volume = 0
        lower_volume = 0
        upper_index = poc_index
        lower_index = poc_index

        while True:
            if advance_up:
                if upper_index >= len(vp_prices) - 1:
                    upper_volume = 0
                else:
                    upper_index += 1
                    upper_volume += vp_volumes[upper_index]
            if advance_down:
                if lower_index <= 0:
                    lower_index = 0
                else:
                    lower_index -= 1
                    lower_volume += vp_volumes[lower_index]
            if (upper_volume > 0 or lower_volume > 0) and (upper_volume > lower_volume):
                advance_up = True
                advance_down = False
                current_sum_of_volume += upper_volume
                upper_volume = 0
                if price[i] == vp_prices[upper_index]:
                    value_area[i] = 1
                    break
            elif upper_volume > 0 or lower_volume > 0:
                advance_up = False
                advance_down = True
                current_sum_of_volume += lower_volume
                lower_volume = 0
                if price[i] == vp_prices[lower_index]:
                    value_area[i] = 1
                    break
            if (current_sum_of_volume <= percentage_of_total_volume) and (
                    (upper_index < len(vp_prices)) or (lower_index > 0)):
                break
    return value_area


def get_volume_profile_peaks_valleys(df: pd.DataFrame) -> np.array:
    """
    Given the canonical dataframe recorded, this function returns an array with info if the price is in a peak or valley
    of volumes
    :param df: canonical dataframe recorded
    :return: numpy array with values: High Peak = 2, High Peak Area = 1, Valley Peak = -2, Valley Peak Area = -1
    """

    price = np.array(df.Price)
    volume = np.array(df.Volume)
    date = np.array(df.Date)
    len_ = len(price)
    peaks_valleys = np.zeros(len_)
    volume_profile = {}

    print(f"Calculate VP peaks and valleys...")

    for i in tqdm(range(len_ - 1)):
        if date[i + 1] == date[i]:
            pass
        else:
            volume_profile.clear()

        if price[i] in volume_profile.keys():
            volume_profile[price[i]] += volume[i]
        else:
            volume_profile[price[i]] = volume[i]

        source = np.array(sorted(volume_profile.keys()))
        weight = np.array([volume_profile[key] for key in source])

        kde = of.gaussian_kde_2(source, weight, 0.8)

        peaks_valleys[i] = 0

        if len(kde) > 2:
            ii = 0
            peaks = []
            peaks_weight = []
            peaks_indexes = []
            for it in source:
                if ii == 0 or ii == len(kde) - 1:
                    peaks.append(source[ii])
                    peaks_weight.append(weight[ii])
                    peaks_indexes.append(ii)
                else:
                    if kde[ii] > kde[ii - 1] and kde[ii] > kde[ii + 1]:
                        peaks.append(it)
                        peaks_weight.append(weight[ii])
                        peaks_indexes.append(ii)
                        if price[i] == it:
                            peaks_valleys[i] = 2
                            break
                        if price[i] < it:
                            break
                    if kde[ii] < kde[ii - 1] and kde[ii] < kde[ii + 1]:
                        peaks.append(it)
                        peaks_weight.append(weight[ii])
                        peaks_indexes.append(ii)
                        if price[i] == it:
                            peaks_valleys[i] = -2
                            break
                        if price[i] < it:
                            break
                ii += 1

            if peaks_valleys[i] not in [-2, 2]:
                tick_size = source[1] - source[0]
                distance_in_element = (peaks[len(peaks) - 1] - peaks[len(peaks) - 2]) / tick_size
                if distance_in_element > 0:
                    half_distance_in_element = int(distance_in_element / 2)
                    if (price[i] >= source[peaks_indexes[(len(peaks_indexes) - 1)] - half_distance_in_element] and
                        peaks_weight[len(peaks_weight) - 1] < peaks_weight[len(peaks_weight) - 2]) \
                            or \
                            (price[i] < source[peaks_indexes[(len(peaks_indexes) - 1)] - half_distance_in_element] and
                             peaks_weight[len(peaks_weight) - 1] > peaks_weight[len(peaks_weight) - 2]):
                        peaks_valleys[i] = -1
                    elif (price[i] >= source[peaks_indexes[(len(peaks_indexes) - 1)] - half_distance_in_element] and
                          peaks_weight[len(peaks_weight) - 1] > peaks_weight[len(peaks_weight) - 2]) \
                            or \
                            (price[i] < source[peaks_indexes[(len(peaks_indexes) - 1)] - half_distance_in_element] and
                             peaks_weight[len(peaks_weight) - 1] < peaks_weight[len(peaks_weight) - 2]):
                        peaks_valleys[i] = 1
                    else:
                        peaks_valleys[i] = 0

    return peaks_valleys
