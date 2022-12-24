import pandas as pd
import numpy as np
import operator
from tqdm import tqdm


def get_dynamic_vp(df:pd.DataFrame) -> pd.DataFrame:

    '''
    Given the canonical dataframe recorded, this function returns a dataframe with volume in ASK / BID restarted at the
    of a given day.
    :param df: canonical dataframe recorded (tick-by-tick wth the DOM attached)
    :return: dataframe of
    '''

    volume    = np.array(df.Volume)
    price     = np.array(df.Price)
    side      = np.array(df.TradeType)
    sday      = np.array(df.Index_DayShift)

    volume_a  = {}
    volume_b  = {}
    len_final = df.shape[0]
    voa       = np.zeros(len_final)
    vob       = np.zeros(len_final)

    if side[0] == 2:
        voa[0]             = volume[0]
        volume_a[price[0]] = volume[0]
        volume_b[price[0]] = 0
    elif side[0] == 1:
        vob[0]             = volume[0]
        volume_a[price[0]] = 0
        volume_b[price[0]] = volume[0]

    for idx in tqdm(range(1, len_final)):

        if sday[idx]:
            if side[idx] == 2:
                if price[idx] in volume_a:
                    volume_a[price[idx]] += volume[idx]
                    voa[idx] = volume_a[price[idx]]
                    try:
                        vob[idx] = volume_b[price[idx]]
                    except Exception as ex:
                        print(ex)
                else:
                    volume_a[price[idx]] = volume[idx]
                    voa[idx] = volume[idx]
                    try:
                        vob[idx] = volume_b[price[idx]]
                    except Exception as ex:
                        print(ex)
            elif side[idx] == 1:
                if price[idx] in volume_b:
                    volume_b[price[idx]] += volume[idx]
                    vob[idx] = volume_b[price[idx]]
                    try:
                        voa[idx] = volume_a[price[idx]]
                    except Exception as ex:
                        print(ex)
                else:
                    volume_b[price[idx]] = volume[idx]
                    vob[idx] = volume[idx]
                    try:
                        voa[idx] = volume_a[price[idx]]
                    except Exception as ex:
                        print(ex)
            else:
                print('no side')
        else:
            volume_a = {}
            volume_b = {}
            if side[idx] == 2:
                volume_a[price[idx]] = volume[idx]
                voa[idx] = volume_a[price[idx]]
            elif side[idx] == 1:
                volume_b[price[idx]] = volume[idx]
                vob[idx] = volume_b[price[idx]]
            else:
                print('no side')

    df = df.assign(VP_Ask=pd.Series(voa, dtype=int).reset_index(drop=True))
    df = df.assign(VP_Bid=pd.Series(vob, dtype=int).reset_index(drop=True))

    return df


def get_daily_moving_POC(df:pd.DataFrame) -> np.array:

    '''
    Given the canonical dataframe recorded, this function returns the Point of Control (i.e. POC) that is moving during
    the day giving the volume sentiment of uptrending market or choppy market of downtrending market.
    :param df: anonical dataframe recorded
    :return: numpy array for the daily moving poc
    '''

    volume    = np.array(df.Volume)
    price     = np.array(df.Price)
    date      = np.array(df.Date)
    poc_final = {}
    len_      = len(price)
    poc_      = np.zeros(len_)

    poc_final[price[0]] = volume[0]
    poc_[0]             = price[0]

    for i in tqdm(range(1, len_ - 1)):
        cp = price[i]
        if date[i] != date[i-1]:
            poc_final.clear()
            poc_final[ cp ] = volume[i]
            poc_[i]         = cp
        else:
            if cp in poc_final:
                poc_final[ cp ] += volume[i]
            else:
                poc_final[ cp ] = volume[i]
            poc_[i] = max(poc_final.items(), key=operator.itemgetter(1))[0]

    poc_[len_-1] = price[len_-1]

    return poc_

