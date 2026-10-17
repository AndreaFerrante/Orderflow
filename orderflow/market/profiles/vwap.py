import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from orderflow.core.configuration import *
from orderflow.core.exceptions import SessionTypeAbsent
from orderflow.core._kernel_calls import run_kernel as _run_kernel
from orderflow.market.profiles import _session_kernels as _kernels


def get_vwap(data: pd.DataFrame) -> pd.DataFrame:

    """
    VWAP ( Volume Weighted Average Price ) := sum( Volume(i) * Price(i) ) / sum( Volume(i) )

    This function needs to receive a DataFrame with the session defined within session inside (the name of this column
    must be the following: SessionType)
    :return: DataFrame of the VWAP and 4 VWAP bands via variance.
    """

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    # Numba kernel; the loop below stays as its reference and as the path for a frame the kernel
    # does not reproduce exactly (see ``_session_kernels.session_arrays``).
    arrays = _kernels.session_arrays(data, with_levels=False)
    if arrays is None:
        return _get_vwap_py(data)
    price, volume, reset, _, _ = arrays
    # A session opening on a zero-volume tick divides 0 by 0: the loop warns there, so it keeps it.
    if volume[0] == 0 or (volume[reset] == 0).any():
        return _get_vwap_py(data)
    offsets = np.array([VWAP_BAND_OFFSET_1, VWAP_BAND_OFFSET_2, VWAP_BAND_OFFSET_3, VWAP_BAND_OFFSET_4],
                       dtype=np.float64)
    with tqdm(total=len(price) - 1) as bar:
        vwap, top, bottom = _run_kernel(
            _kernels.session_vwap, price, volume, reset, offsets, ticks=len(price), bar=bar
        )
    return pd.DataFrame({'vwap': vwap,
                         'vwap_sd1_top': top[0], 'vwap_sd1_bottom': bottom[0],
                         'vwap_sd2_top': top[1], 'vwap_sd2_bottom': bottom[1],
                         'vwap_sd3_top': top[2], 'vwap_sd3_bottom': bottom[2],
                         'vwap_sd4_top': top[3], 'vwap_sd4_bottom': bottom[3]})


def _get_vwap_py(data: pd.DataFrame) -> pd.DataFrame:
    """Reference loop of :func:`get_vwap`."""

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
