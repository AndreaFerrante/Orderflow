import numpy as np
import pandas as pd
from tqdm import tqdm
import math


def get_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP ( Volume Weighted Average Price ) := sum( Volume(i) * Price(i) ) / sum( Volume(i) )

    Given the canonical data structure recorded, this function returns the VWAP DAILY calculated using:
    :param price: this is the price array recorded
    :param volume: this is the single volume calculation per tick
    :param data: this is the date in which volume calculation is recorded
    :return: pandas dataframe of the VWAP and 4 VWAP bands via variance
    """

    price = np.array(data.Price)
    volume = np.array(data.Volume)
    date = np.array(data.Date)

    len_ = len(price)
    vwap = np.zeros(len_)
    vwap_sd1_top = np.zeros(len_)
    vwap_sd2_top = np.zeros(len_)
    vwap_sd3_top = np.zeros(len_)
    vwap_sd4_top = np.zeros(len_)
    vwap_sd1_bottom = np.zeros(len_)
    vwap_sd2_bottom = np.zeros(len_)
    vwap_sd3_bottom = np.zeros(len_)
    vwap_sd4_bottom = np.zeros(len_)
    sum_vol = 0
    sum_pri_vol = 0
    sum_price_vwap_difference_squared_times_volume = 0.0
    band_distance = 1.0

    for i in tqdm(range(len_ - 1)):

        if date[i + 1] == date[i]:
            sum_pri_vol += price[i] * volume[i]
            sum_vol += volume[i]
            vwap[i] = sum_pri_vol / sum_vol

            price_vwap_difference_squared_times_volume = (price[i] - vwap[i]) * (
                    price[i] - vwap[i]) * volume[i]

            sum_price_vwap_difference_squared_times_volume += price_vwap_difference_squared_times_volume

            if sum_vol != 0:
                band_distance = math.sqrt(sum_price_vwap_difference_squared_times_volume / sum_vol)
            else:
                band_distance = 1.0
        else:
            sum_pri_vol += price[i] * volume[i]
            sum_vol += volume[i]
            vwap[i] = sum_pri_vol / sum_vol

            sum_pri_vol = 0
            sum_vol = 0
            sum_price_vwap_difference_squared_times_volume = 0

            sum_pri_vol += price[i + 1] * volume[i + 1]
            sum_vol += volume[i + 1]
            vwap[i + 1] = sum_pri_vol / sum_vol

            price_vwap_difference_squared_times_volume = (price[i] - vwap[i]) * (
                    price[i] - vwap[i]) * volume[i]
            sum_price_vwap_difference_squared_times_volume += price_vwap_difference_squared_times_volume

            if sum_vol != 0:
                band_distance = math.sqrt(sum_price_vwap_difference_squared_times_volume / sum_vol)
            else:
                band_distance = 1.0

        vwap_sd1_top[i] = vwap[i] + (1 * band_distance)
        vwap_sd1_bottom[i] = vwap[i] - (1 * band_distance)

        vwap_sd2_top[i] = vwap[i] + (2 * band_distance)
        vwap_sd2_bottom[i] = vwap[i] - (2 * band_distance)

        vwap_sd3_top[i] = vwap[i] + (3 * band_distance)
        vwap_sd3_bottom[i] = vwap[i] - (3 * band_distance)

        vwap_sd4_top[i] = vwap[i] + (4 * band_distance)
        vwap_sd4_bottom[i] = vwap[i] - (5 * band_distance)

    vwap[-1:] = vwap[-2:-1]
    vwap_sd1_top[-1:] = vwap_sd1_top[-2:-1]
    vwap_sd1_bottom[-1:] = vwap_sd1_bottom[-2:-1]
    vwap_sd2_top[-1:] = vwap_sd2_top[-2:-1]
    vwap_sd2_bottom[-1:] = vwap_sd2_bottom[-2:-1]
    vwap_sd3_top[-1:] = vwap_sd3_top[-2:-1]
    vwap_sd3_bottom[-1:] = vwap_sd3_bottom[-2:-1]
    vwap_sd4_top[-1:] = vwap_sd4_top[-2:-1]
    vwap_sd4_bottom[-1:] = vwap_sd4_bottom[-2:-1]

    result_df = pd.DataFrame({'vwap': vwap, 'vwap_sd1_top': vwap_sd1_top, 'vwap_sd1_bottom': vwap_sd1_bottom,
                              'vwap_sd2_top': vwap_sd2_top, 'vwap_sd2_bottom': vwap_sd2_bottom,
                              'vwap_sd3_top': vwap_sd3_top, 'vwap_sd3_bottom': vwap_sd3_bottom,
                              'vwap_sd4_top': vwap_sd4_top, 'vwap_sd4_bottom': vwap_sd4_bottom})

    return result_df
