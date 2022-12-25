import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def half_hour(x):

    if x >= 30:
        return "30"
    else:
        return "00"


def prepare_data(
        data: pd.DataFrame
) -> pd.DataFrame:

    """
    Given usual data recorded, this function returns it corrected since its CSV recoding so it adds pandas datatypes
    fro data reshaping and data plotting
    :param data: given usual data recorded
    :return: dataframe corrected
    """

    # es = es.assign(Index    = np.arange(0, es.shape[0], 1))  # Set an index fro reference plotting...
    # es = es.assign(Hour     = es.Time.str[:2].astype(str))   # Extract the hour...
    # es = es.assign(Minute   = es.Time.str[3:5].astype(int))  # Extract the minute...
    # es = es.assign(HalfHour = es.Hour.str.zfill(2) + es.Minute.apply(half_hour)) # Identifies half hours...

    data = (
        data.assign(Index=np.arange(0, data.shape[0], 1))
        .assign(Hour=data.Time.str[:2].astype(str))
        .assign(Minute=data.Time.str[3:5].astype(int))
        .assign(HalfHour=data.Hour.str.zfill(2) + data.Minute.apply(half_hour))
        .assign(DateTime=data.Date.astype(str) + " " + data.Time.astype(str))
        .assign(DateTime_TS=pd.to_datetime(data.DateTime))
    )

    return data


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
        dom_cols_ask = [x for x in data.columns if 'AskDOM_' in x]
        dom_cols_bid = [x for x in data.columns if 'BidDOM_' in x]

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


def get_tickers_in_folder(
    path: str, ticker: str = "ES", cols: list = None, break_at: int = 99999
) -> pd.DataFrame:

    """
    Given a path and a ticker sign, this functions read all file in it starting with the ticker symbol (e.g. ES)
    :param path: path to ticker data to read
    :param ticker: ticker to read in the form of ES, ZN, ZB . . .
    :param cols: columns to import...
    :param break_at: how many ticker files to read ?
    :return: DataFrame of all read ticker files
    """

    files = os.listdir(path)
    stacked = []
    counter = 0
    for file in tqdm.tqdm(files):

        if file.startswith(ticker):

            print(f"Reading file named {file} ...")
            if cols is None:
                stacked.append(pd.read_csv(os.path.join(path, file), sep=";"))
            else:
                stacked.append(
                    pd.read_csv(os.path.join(path, file), sep=";", usecols=cols)
                )

            counter += 1
            if counter >= break_at:
                return pd.concat(stacked)

    return pd.concat(stacked)


def plot_half_hour_volume(
    data_already_read: bool,
    data: pd.DataFrame,
    data_path: str = "",
    data_name: str = "",
) -> None:

    """
    This function helps understanding the "volume smile" so that the peak in volume given hal hours is the market open
    :param data_path: file system path to the data
    :param data_name: file name to import
    :return: a plot in matplotlib with bars per half-hour (the bigger counting bar is the one that finds market opens)
    """

    if not data_already_read:
        try:
            data = pd.read_csv(os.path.join(data_path, data_name), sep=";")
        except:
            data = pd.read_csv(os.path.join(data_path, data_name), sep=",")
    else:
        data = data

    data = data[data.Price != 0]  # Remove recording impurities...
    data = data.assign(
        Index=np.arange(0, data.shape[0], 1)
    )  # Set an index fro reference plotting...
    data = data.assign(Hour=data.Time.str[:2].astype(str))  # Extract the hour...
    data = data.assign(Minute=data.Time.str[3:5].astype(int))  # Extract the minute...
    data = data.assign(
        HalfHour=data.Hour.str.zfill(2) + data.Minute.apply(half_hour)
    )  # Identifies half hours...

    # Plot bar chart in which the
    max_volume = data.groupby(["HalfHour"]).agg({"Volume": "sum"}).reset_index()
    plt.bar(max_volume.HalfHour, max_volume.Volume)
    plt.xlabel("HalfHour")
    plt.ylabel("Volume")
    plt.xticks(rotation=90)
    plt.tight_layout()


def filter_big_prints_on_ask(
    data: pd.DataFrame, volume_filter: int = 100
) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this functions returns filtered volume dataframe on the ASK
    :param data: canonical dataframe recorded
    :param volume_filter: time and sales dataframe recorded volume filter
    :return: dataframe with the given filter
    """

    filtered_on_ask = data.query("TradeType == 2").query(
        "Volume >= " + str(volume_filter) + ""
    )

    return filtered_on_ask


def filter_big_prints_on_bid(
    data: pd.DataFrame, volume_filter: int = 100
) -> pd.DataFrame:

    """
    Given the canonical dataframe recorded, this functions returns filtered volume dataframe on the BID
    :param data: canonical dataframe recorded
    :param volume_filter: time and sales dataframe recorded volume filter
    :return: dataframe with the given filter
    """

    filtered_on_bid = data.query("TradeType == 1").query(
        "Volume >= " + str(volume_filter) + ""
    )

    return filtered_on_bid
