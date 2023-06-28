import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from .configuration import *
from .exceptions import SessionTypeAbsent


def get_vwap(data: pd.DataFrame) -> pd.DataFrame:

    """
    VWAP ( Volume Weighted Average Price ) := sum( Volume(i) * Price(i) ) / sum( Volume(i) )

    This function needs to receive a DataFrame with the session defined within session inside (the name of this column
    must be the following: SessionType)
    :return: DataFrame of the VWAP and 4 VWAP bands via variance.
    """

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price     = np.array(data.Price)
    volume    = np.array(data.Volume)
    session   = np.array(data.SessionType) # <--------------

    len_            = len(price)
    vwap            = np.zeros(len_)
    vwap_sd1_top    = np.zeros(len_)
    vwap_sd2_top    = np.zeros(len_)
    vwap_sd3_top    = np.zeros(len_)
    vwap_sd4_top    = np.zeros(len_)
    vwap_sd1_bottom = np.zeros(len_)
    vwap_sd2_bottom = np.zeros(len_)
    vwap_sd3_bottom = np.zeros(len_)
    vwap_sd4_bottom = np.zeros(len_)

    sum_vol                                        = volume[0]
    sum_pri_vol                                    = price[0] * volume[0]
    sum_price_vwap_difference_squared_times_volume = 0.0


    for i in tqdm(range(1, len_)):

        ########################################################################################################
        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):
            sum_pri_vol  = 0
            sum_vol      = 0
            sum_price_vwap_difference_squared_times_volume = 0
        ########################################################################################################

        sum_pri_vol += price[i] * volume[i]
        sum_vol     += volume[i]
        vwap[i]      = sum_pri_vol / sum_vol

        price_vwap_difference_squared_times_volume      = (price[i] - vwap[i]) * (price[i] - vwap[i]) * volume[i]
        sum_price_vwap_difference_squared_times_volume += price_vwap_difference_squared_times_volume

        if sum_vol:
            band_distance = np.sqrt(sum_price_vwap_difference_squared_times_volume / sum_vol)
        else:
            band_distance = 1.0

        vwap_sd1_top[i]    = vwap[i] + (VWAP_BAND_OFFSET_1 * band_distance)
        vwap_sd1_bottom[i] = vwap[i] - (VWAP_BAND_OFFSET_1 * band_distance)

        vwap_sd2_top[i]    = vwap[i] + (VWAP_BAND_OFFSET_2 * band_distance)
        vwap_sd2_bottom[i] = vwap[i] - (VWAP_BAND_OFFSET_2 * band_distance)

        vwap_sd3_top[i]    = vwap[i] + (VWAP_BAND_OFFSET_3 * band_distance)
        vwap_sd3_bottom[i] = vwap[i] - (VWAP_BAND_OFFSET_3 * band_distance)

        vwap_sd4_top[i]    = vwap[i] + (VWAP_BAND_OFFSET_4 * band_distance)
        vwap_sd4_bottom[i] = vwap[i] - (VWAP_BAND_OFFSET_4 * band_distance)


    vwap[0]            = price[0]
    vwap_sd1_top[0]    = price[0]
    vwap_sd1_bottom[0] = price[0]
    vwap_sd2_top[0]    = price[0]
    vwap_sd2_bottom[0] = price[0]
    vwap_sd3_top[0]    = price[0]
    vwap_sd3_bottom[0] = price[0]
    vwap_sd4_top[0]    = price[0]
    vwap_sd4_bottom[0] = price[0]


    result_df = pd.DataFrame({'vwap': vwap,
                              'vwap_sd1_top': vwap_sd1_top, 'vwap_sd1_bottom': vwap_sd1_bottom,
                              'vwap_sd2_top': vwap_sd2_top, 'vwap_sd2_bottom': vwap_sd2_bottom,
                              'vwap_sd3_top': vwap_sd3_top, 'vwap_sd3_bottom': vwap_sd3_bottom,
                              'vwap_sd4_top': vwap_sd4_top, 'vwap_sd4_bottom': vwap_sd4_bottom})


    return result_df
