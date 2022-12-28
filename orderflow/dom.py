import pandas as pd
import numpy as np


def identify_WG_position(data: pd.DataFrame) -> pd.DataFrame:

    """
    Given usual recorded data for analysis, this function tells us if the biggest volume on the DOM was on the
    first level of the Depth of the Market (DOM)
    :param data: dataframe with all DOM columns
    :return: given dataframe with WG on ask and bid addition
    """

    dom_cols = [x for x in data.columns if "DOM_" in x]
    if len(dom_cols) == 0:
        raise Exception("Dataframe passed has no DOM columns ! Provide a dataframe with DOM columns.")
    else:

        dom_cols_ask = [x for x in data.columns if "AskDOM_" in x] + ["AskSize"]
        dom_cols_bid = [x for x in data.columns if "BidDOM_" in x] + ["BidSize"]

        data['Ask_WG'] = np.array( data[dom_cols_ask].idxmax(axis=1) )
        data['Bid_WG'] = np.array( data[dom_cols_bid].idxmax(axis=1) )

    return data


def remove_DOM_columns(data: pd.DataFrame) -> pd.DataFrame:

    """
    Given the usual recorded data for analysis, this functions removes all DOM columns for better performance.
    DOM columns can be identified by removing columns with "DOM_" in it.
    :param data: dataframe recorded data
    :return: recorded dataframe with no DOM columns
    """

    dom_cols = [x for x in data.columns if "DOM_" in x]
    return data.drop(dom_cols, axis=1)


def sum_first_n_DOM_levels(
    data: pd.DataFrame, l1_side_to_sum: str = "ask", l1_level_to_sum: int = 5
) -> pd.DataFrame:

    """
    This function, given the canonical dataframe, sums the first N levels of the DOM
    :param data: dataframe recorded data
    :param l1_side_to_sum: side of levels to be sum up (use str ask for summing the ASK, bid for summing the BID)
    :param l1_level_to_sum: number of levels to be sum up (default is 10 levels)
    :return: dataframe of levels summed up per side choose (Ask / Bid)

    Attention ! Using str().upper() function to prevent capital letter error...
    """

    l1_side_to_sum = str(l1_side_to_sum).upper()
    dom_side = "AskDOM_" if l1_side_to_sum == "ASK" else "BidDOM_"
    dom_cols = [x for x in data.columns if dom_side in x]

    if l1_level_to_sum > len(dom_cols):
        raise Exception("Data provided has less DOM levels then the ones to sum up.")
    else:
        dom_cols = dom_cols[:l1_level_to_sum]

    if l1_side_to_sum == "ASK":
        data["DOMSumAsk_" + str(l1_level_to_sum)] = data[dom_cols].sum(axis=1)
    elif l1_side_to_sum == "BID":
        data["DOMSumBid_" + str(l1_level_to_sum)] = data[dom_cols].sum(axis=1)
    else:
        raise Exception("No correct DOM side provided !")

    return data


def get_dom_shape_for_n_levels(
    data: pd.DataFrame, l1_level_to_watch: int = 5
) -> pd.DataFrame:

    """
    This function, given the canonical dataframe, tells if the DOM shape has a "rectangular" form:
    rectangular_shape := { get_sum_of_n_levels_on_the_dom } / { ( get_max_volume_on_first_n_levels ) * ( l1_level_to_watch ) }

    The closer this function is to one, the more robust and "thick" the DOM is. This function can spot DOM outliers.
    The closer the value is to 1, the thicker the DOM is...

    :param data: dataframe recorded data
    :param l1_side_to_sum: side of levels to be sum up (use str ask for summing the ASK, bid for summing the BID)
    :param l1_level_to_watch: number of levels to be sum up (default is 10 levels)
    :return: dataframe of levels summed up for both side (Ask / Bid)
    """

    ask_dom_columns = [x for x in data.columns if "AskDOM_" in x]
    bid_dom_columns = [x for x in data.columns if "BidDOM_" in x]

    if l1_level_to_watch > len(ask_dom_columns) or l1_level_to_watch > len(
        bid_dom_columns
    ):
        raise Exception("Data provided has less DOM levels then the ones to sum up.")
    else:
        ask_dom_columns = ask_dom_columns[:l1_level_to_watch]
        bid_dom_columns = bid_dom_columns[:l1_level_to_watch]

    data["DOMSumAsk_" + str(l1_level_to_watch) + "_Shape"] = ( data[ask_dom_columns].sum(axis=1) ) / (np.max(data[ask_dom_columns], axis=1) * l1_level_to_watch)
    data["DOMSumBid_" + str(l1_level_to_watch) + "_Shape"] = ( data[bid_dom_columns].sum(axis=1) ) / (np.max(data[bid_dom_columns], axis=1) * l1_level_to_watch)

    return data


