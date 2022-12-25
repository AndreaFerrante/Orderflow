import pandas as pd
import numpy as np


def identify_WG_position(
        data:pd.DataFrame
) -> (np.array, np.array):

    '''
    Given usual recordedd data for analysis, this function tells us if the biggest volume on the DOM was on the
    first level of the Depth of the Market (DOM)
    :param data: dataframe with all DOM columns
    :return:
    '''

    dom_cols = [x for x in data.columns if 'DOM_' in x]
    if len(dom_cols) == 0:
        raise Exception('Dataframe passed has no DOM columns ! Provide a dataframe with DOM columns.')
        return
    else:
        dom_cols_ask = [x for x in data.columns if 'AskDOM_' in x] + ['AskSize']
        dom_cols_bid = [x for x in data.columns if 'BidDOM_' in x] + ['BidSize']

        ask_WG = np.array(data[ dom_cols_ask ].idxmax(axis=1))
        bid_WG = np.array(data[ dom_cols_bid ].idxmax(axis=1))

    return ask_WG, bid_WG


def remove_DOM_columns(
        data:pd.DataFrame
) -> pd.DataFrame:

    '''
    Given the usual recorded data for analysis, this functions removes all DOM columns for better performance.
    DOM columns can be identified by removing columns with "DOM_" in it.
    :param data: dataframe racorded data
    :return: recorded dataframe with no DOM columns
    '''

    dom_cols = [x for x in data.columns if 'DOM_' in x]
    return ( data.drop( dom_cols, axis=1 ) )

    pass





