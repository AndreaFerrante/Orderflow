import pandas as pd
import numpy as np


def identify_WG_position(
        data: pd.DataFrame
) -> (np.array, np.array):

    """
    Given usual recordedd data for analysis, this function tells us if the biggest volume on the DOM was on the
    first level of the Depth of the Market (DOM)
    :param data: dataframe with all DOM columns
    :return:
    """

    dom_cols = [x for x in data.columns if "DOM_" in x]
    if len(dom_cols) == 0:
        raise Exception(
            "Dataframe passed has no DOM columns ! Provide a dataframe with DOM columns."
        )
        return
    else:
        dom_cols_ask = [x for x in data.columns if "AskDOM_" in x] + ["AskSize"]
        dom_cols_bid = [x for x in data.columns if "BidDOM_" in x] + ["BidSize"]

        ask_WG = np.array(data[dom_cols_ask].idxmax(axis=1))
        bid_WG = np.array(data[dom_cols_bid].idxmax(axis=1))

    return ask_WG, bid_WG


def remove_DOM_columns(
        data: pd.DataFrame
) -> pd.DataFrame:

    """
    Given the usual recorded data for analysis, this functions removes all DOM columns for better performance.
    DOM columns can be identified by removing columns with "DOM_" in it.
    :param data: dataframe recorded data
    :return: recorded dataframe with no DOM columns
    """

    dom_cols = [x for x in data.columns if "DOM_" in x]
    return data.drop(dom_cols, axis=1)

    pass


def sum_first_n_DOM_levels(
    data:pd.DataFrame, l1_side_to_sum:str='ask', l1_level_to_sum:int=5
) -> pd.DataFrame:

    '''
    This function, given the canonical dataframe, sums the first N levels of the DOM
    :param data: dataframe recorded data
    :param l1_side_to_sum: side of levels to be sum up (use str ask for summing the ASK, bid for summing the BID)
    :param l1_level_to_sum: number of levels to be sum up (default is 10 levels)
    :return: dataframe of levels summed up per side choose (Ask / Bid)

    Attention ! Using str().upper() function to prevent capital letter error...
    '''

    # 1. Convert summation...
    # 2. Select only relevant ASK / BID columns inside the data...
    # 3. Filter only relevant ASK / BID columns inside the data...
    l1_side_to_sum = str(l1_side_to_sum).upper()
    dom_side       = 'AskDOM_' if l1_side_to_sum == 'ASK' else 'BidDOM_'
    dom_cols       = [x for x in data.columns if dom_side in x]

    if l1_level_to_sum > len(dom_cols):
        raise Exception('Data provided has less DOM levels then the ones to sum up.')
    else:
        dom_cols = dom_cols[:l1_level_to_sum]

    if l1_side_to_sum == 'ASK':
        data[ 'DOMSumAsk_' + str(l1_level_to_sum) ] = data[ dom_cols ].sum(axis=1)
    elif l1_side_to_sum == 'BID':
        data['DOMSumBid_' + str(l1_level_to_sum)]   = data[dom_cols].sum(axis=1)
    else:
        raise Exception('No correct side provided !')

    return data




