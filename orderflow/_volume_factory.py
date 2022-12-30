import os
import datatable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datatable import fread, f
from tqdm import tqdm


def half_hour(x) -> str:

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


def get_longest_columns_dataframe(
    path: str, ticker: str = "ES"
) -> list:

    files = [x for x in os.listdir(path) if x.startswith(ticker)]
    cols = [
        x for x in range(99999)
    ]  # Dummy list for having the cols as big as possible...

    for file in files:

        single = pd.read_csv(
            os.path.join(path, file), sep=";", nrows=2
        )  # Read only first two rows to read teh columns !
        if len(single.columns) < len(cols):
            cols = single.columns

    return list(cols)


def get_tickers_in_folder(
    path: str, ticker: str = "ES", cols: list = None, break_at: int = 99999
) -> pd.DataFrame:

    """
    Given a path and a ticker sign, this functions read all file in it starting with the ticker symbol (e.g. ES).
    This package leverages a ot datatable speed !
    :param path: path to ticker data to read
    :param ticker: ticker to read in the form of ES, ZN, ZB, AAPL, MSFT. . .
    :param cols: columns to import...
    :param break_at: how many ticker files do we have to read ?
    :return: DataFrame of all read ticker files

    Attention ! Recorded dataframes have 19 / 39 DOM levels: this function reads the ones with less DOM cols for all of them.
    """

    if cols is None:
        cols = get_longest_columns_dataframe(path=path, ticker=ticker)

    ticker  = str(ticker).upper()
    files   = [str(x).upper() for x in os.listdir(path) if x.startswith(ticker)]
    stacked = datatable.Frame()

    for idx, file in tqdm(enumerate(files)):

        print(f"Reading file {file} ...")

        read_file = fread(os.path.join(path, file), sep=";", fill=True)
        read_file = read_file[:, list(cols)]    # Select all rows and specific columns...
        read_file["Date"] = datatable.str64     # Convert string for filtering...
        read_file["Price"] = datatable.float64  # Convert float for filtering...
        read_file = read_file[(f.Date != "1899-12-30") & (f.Price != 0), :]  # Filter impurities...
        stacked.rbind(read_file)

        if idx >= break_at:
            return stacked.sort(["Date", "Time"]).to_pandas()

    return stacked.sort(["Date", "Time"]).to_pandas()


def get_orders_in_row(
    trades: pd.DataFrame, seconds_split: int=2
) -> pd.DataFrame:

    '''
    This function gets prints "anxiety" over the tape :-)
    :param trades: canonical trades executed
    :param seconds_split: seconds to measure the speed of the tape
    :return: anxiety over the market on both ask/bid sides
    '''

    # seconds_split = 2
    # trades = pd.concat([big_shoot_ask, big_shoot_bid], axis=0)

    present = 0
    for el in ['Date', 'Time']:
        if el in trades.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a trade dataframe that has Date and Time columns')

    if 'Datetime' not in trades.columns:
        trades.insert( 0, 'Datetime', pd.to_datetime( trades['Date'] + ' ' + trades['Time'] ) )
        trades.sort_values(['Datetime'], ascending=True, inplace=True)
    elif 'Datetime' in trades.columns:
        trades.sort_values(['Datetime'], ascending=True, inplace=True)

    def manage_speed_of_tape(trades_on_on_side:pd.DataFrame,
                             side: int = 2) -> pd.DataFrame:

        trades_on_on_side = trades_on_on_side[ (trades_on_on_side.TradeType == side) ]
        trades_on_on_side.sort_values(['Datetime'], ascending=True, inplace=True)

        vol_, dt_, count_, price_, idx_ = list(), list(), list(), list(), list()
        len_ = trades_on_on_side.shape[0]
        i    = 0

        while i < len_:

            start_time = trades_on_on_side.Datetime[i]
            start_vol  = trades_on_on_side.Volume[i]
            counter    = 0

            for j in range(i + 1, len_):
                delta_time = trades_on_on_side.Datetime[j] - start_time
                ##############################################
                if delta_time.total_seconds() <= seconds_split:
                    start_vol += trades_on_on_side.Volume[j]
                    counter   += 1
                else:
                    break
                ##############################################

            if counter:
                vol_.append(   start_vol)
                dt_.append(    trades_on_on_side.Datetime[j - 1])
                price_.append( trades_on_on_side.Price[j - 1])
                idx_.append(   trades_on_on_side.Index[j - 1])
                count_.append( counter + 1)
                i = i + counter + 1
            else:
                i += 1

        res = pd.DataFrame({'LastDatetime': dt_,
                            'Volume':       vol_,
                            'Counter':      count_,
                            'Price':        price_,
                            'Side':         [side] * len(price_),
                            'Index':        idx_})

        return res

    # Manage speed of tape on the ASK, first
    try:
        ask = manage_speed_of_tape(trades, 2)
    except Exception as e:
        print(e)

    # Manage speed of tape on the BID, secondly.
    try:
        bid = manage_speed_of_tape(trades, 1)
    except Exception as e:
        print(e)

    return pd.concat( [ask, bid], axis=0).sort_values(['LastDatetime'], ascending=True)


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


def get_volume_distribution(
    data: pd.DataFrame
) -> pd.DataFrame:

    value_counts_num = pd.DataFrame(data["Volume"].value_counts()).reset_index()
    value_counts_num = value_counts_num.rename(
        columns={"Volume": "VolumeCount", "index": "VolumeSize"}
    )
    value_counts_per = pd.DataFrame(
        data["Volume"].value_counts(normalize=True)
    ).reset_index()
    value_counts_per = value_counts_per.rename(
        columns={"Volume": "VolumePerc", "index": "VolumeSize"}
    )
    value_counts_per = value_counts_per.assign(
        VolumePerc=value_counts_per.VolumePerc * 100
    )

    stats = value_counts_num.merge(
        right=value_counts_per, how="left", on="VolumeSize"
    ).reset_index(drop=True)
    stats.sort_values(["VolumeSize"], ascending=True, inplace=True)
    stats = stats.assign(VolumePercCumultaive=np.cumsum(stats.VolumePerc))

    return stats


def get_new_start_date(
    data:pd.DataFrame, sort_values:bool=False
) -> pd.DataFrame:

    '''
    This function marks with one the start of a new date
    :param data: canonical dataframe
    :return: canonical dataframe with the addition of the column for new day start
    '''

    ########################################################################
    # Sort by date and time for clarity...
    if sort_values:
        data.sort_values(['Date', 'Time'], ascending=[True, True], inplace=True)
    ########################################################################

    data['Date_Shift']   = data.Date.shift(1)
    data_last_date_notna = data['Date_Shift'].head(2).values[1] # Take first two elements and the second one must be not empty...
    data['Date_Shift']   = data['Date_Shift'].fillna( data_last_date_notna )
    data['DayStart']     = np.where(data.Date != data.Date_Shift, 1, 0)

    return data.drop(['Date_Shift'], axis=1)




