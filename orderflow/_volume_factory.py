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


