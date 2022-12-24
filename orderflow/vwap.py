import numpy as np
from tqdm import tqdm


def get_vwap(price:np.array, volume:np.array, date:np.array) -> np.array:

    '''
    VWAP ( Volume Weighted Average Price ) := sum( Volume(i) * Price(i) ) / sum( Volume(i) )

    Given the canonical data structure recorded, this function returns the VWAP DAILY calculated using:
    :param price: this is the price array recorded
    :param volume: this is the single volume calculation per tick
    :param date: this is the date in which volume calculation is recorded
    :return: numpy array of the VWAP
    '''

    len_        = len(price)
    vwap        = np.zeros(len_)
    sum_vol     = 0
    sum_pri_vol = 0

    for i in tqdm(range(len_ - 1)):

        if date[i+1] == date[i]:
            sum_pri_vol += price[i] * volume[i]
            sum_vol     += volume[i]
            vwap[i]     = sum_pri_vol / sum_vol
        else:
            sum_pri_vol += price[i] * volume[i]
            sum_vol     += volume[i]
            vwap[i]     = sum_pri_vol / sum_vol

            sum_pri_vol = 0
            sum_vol     = 0

            sum_pri_vol += price[i + 1] * volume[i + 1]
            sum_vol     += volume[i + 1]
            vwap[i + 1]  = sum_pri_vol / sum_vol

    vwap[-1:] = vwap[-2:-1]

    return np.array( vwap )


