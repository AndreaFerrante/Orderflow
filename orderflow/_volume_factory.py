import os
import polars
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from orderflow.configuration import *
from dateutil.parser import parse


def half_hour(x) -> str:

    print(f"Half hour...")

    if x >= 30:
        return "30"
    else:
        return "00"


def prepare_data(
        data: pd.DataFrame
) -> pd.DataFrame:

    """
    Given usual data recorded, this function returns it corrected since its CSV recoding so it adds pandas datatypes
    for data reshaping and data plotting
    :param data: given usual data recorded
    :return: dataframe corrected
    """

    # es = es.assign(Index    = np.arange(0, es.shape[0], 1))  # Set an index fro reference plotting...
    # es = es.assign(Hour     = es.Time.str[:2].astype(str))   # Extract the hour...
    # es = es.assign(Minute   = es.Time.str[3:5].astype(int))  # Extract the minute...
    # es = es.assign(HalfHour = es.Hour.str.zfill(2) + es.Minute.apply(half_hour)) # Identifies half hours...

    print(f"Prepare data...")

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
    cols  = [x for x in range(99999)]  # Dummy list for having the cols as big as possible...

    for file in files:

        single = pd.read_csv(os.path.join(path, file), sep=";", nrows=2)  # Read only first two rows to read teh columns !
        if len(single.columns) < len(cols):
            cols = single.columns

    return list(cols)


def get_tickers_in_pg_table(
        connection,
        schema: str,
        table_name: str,
        start_date: str,
        end_date: str,
        date_column: str,
        offset: int = 0) -> pd.DataFrame:

    """
    Given a connection to a Database PostgreSQL and a table read all the ticker data
    :param connection: a psycopg2 object connection
    :param schema: database schema
    :param table_name: table with ticker data
    :param start_date: data start date format YYYY-MM-DD
    :param end_date: data end date format YYYY-MM-DD
    :param date_column: column at date filter is applied to
    :param offset: offset to a created datetime column
    :return: DataFrame of all read ticker data
    """

    print(f"Get longest column in dataframes...")

    colum_dict     = {
        'original_date': 'Date',
        'original_time': 'Time',
        'sequence': 'Sequence',
        'depthsequence': 'DepthSequence',
        'price': 'Price',
        'volume': 'Volume',
        'tradetype': 'TradeType',
        'askprice': 'AskPrice',
        'bidprice': 'BidPrice',
        'asksize': 'AskSize',
        'bidsize': 'BidSize',
        'totalaskdepth': 'TotalAskDepth',
        'totalbiddepth': 'TotalBidDepth',
        'askdomprice': 'AskDOMPrice',
        'biddomprice': 'BidDOMPrice',
        'askdom_0': 'AskDOM_0',
        'biddom_0': 'BidDOM_0',
        'askdom_1': 'AskDOM_1',
        'biddom_1': 'BidDOM_1',
        'askdom_2': 'AskDOM_2',
        'biddom_2': 'BidDOM_2',
        'askdom_3': 'AskDOM_3',
        'biddom_3': 'BidDOM_3',
        'askdom_4': 'AskDOM_4',
        'biddom_4': 'BidDOM_4',
        'askdom_5': 'AskDOM_5',
        'biddom_5': 'BidDOM_5',
        'askdom_6': 'AskDOM_6',
        'biddom_6': 'BidDOM_6',
        'askdom_7': 'AskDOM_7',
        'biddom_7': 'BidDOM_7',
        'askdom_8': 'AskDOM_8',
        'biddom_8': 'BidDOM_8',
        'askdom_9': 'AskDOM_9',
        'biddom_9': 'BidDOM_9',
        'askdom_10': 'AskDOM_10',
        'biddom_10': 'BidDOM_10',
        'askdom_11': 'AskDOM_11',
        'biddom_11': 'BidDOM_11',
        'askdom_12': 'AskDOM_12',
        'biddom_12': 'BidDOM_12',
        'askdom_13': 'AskDOM_13',
        'biddom_13': 'BidDOM_13',
        'askdom_14': 'AskDOM_14',
        'biddom_14': 'BidDOM_14',
        'askdom_15': 'AskDOM_15',
        'biddom_15': 'BidDOM_15',
        'askdom_16': 'AskDOM_16',
        'biddom_16': 'BidDOM_16',
        'askdom_17': 'AskDOM_17',
        'biddom_17': 'BidDOM_17',
        'askdom_18': 'AskDOM_18',
        'biddom_18': 'BidDOM_18',
        'askdom_19': 'AskDOM_19',
        'biddom_19': 'BidDOM_19'
    }
    chunk_size     = 500000
    records_offset = 0
    df_list        = []

    while True:

        sql = "SELECT * FROM " + schema + "." + table_name + " WHERE " + date_column + " >= '" + start_date + \
              "' AND " + date_column + " <= '" + end_date + "' LIMIT " + str(chunk_size) + \
              " OFFSET " + str(records_offset) + ";"

        chunk_df = pd.read_sql(sql=sql, con=connection)

        if len(chunk_df) == 0:
            break  # Exit the loop when no more data is returned

        df_list.append(chunk_df)
        records_offset += chunk_size

    df = pd.concat(df_list, ignore_index=True)
    df.rename(columns=colum_dict, inplace=True)

    if offset:
        df['Datetime'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['Datetime'] = df['Datetime'].apply(lambda x: parse(x) if '.' in x else parse(x + '.000'))
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['Datetime'] = df['Datetime'] + pd.DateOffset(hours=offset)

    df = df.sort_values(by=['Date', 'Time'])
    return df


def get_tickers_in_folder(
        path: str, ticker: str = "ES", cols: list = None, break_at: int = 99999, offset: int = 0, extension:str = 'txt'
):

    """
    Given a path and a ticker sign, this functions read all file in it starting with the ticker symbol (e.g. ES).
    This package leverages a polars speed !
    :param path: path to ticker data to read
    :param ticker: ticker to read in the form of ES, ZN, ZB, AAPL, MSFT. . .
    :param cols: columns to import...
    :param break_at: how many ticker files do we have to read ?
    :param offset: offset to a created datetime column
    :return: DataFrame of all read ticker files

    Attention ! Recorded dataframes have 19 / 39 DOM levels: this function reads the ones with less DOM cols for all of them.
    """

    print(f"Get tickers in folder...")

    if cols is None:
        cols = get_longest_columns_dataframe(path=path, ticker=ticker)

    ticker  = str(ticker).upper()
    files   = [str(x).upper() for x in os.listdir(path) if x.startswith(ticker)]
    stacked = polars.DataFrame()

    for idx, file in tqdm(enumerate(files)):

        print(f"Reading file {file} ...")

        if file.endswith(str(extension).upper()):
            single_file = polars.read_csv(os.path.join(path, file), separator=';', columns=cols, infer_schema_length=10_000)
            single_file = single_file.filter((single_file['Date'] != "1899-12-30") & (single_file['Price'] > 0))
            stacked     = polars.concat([stacked, single_file])

        if idx >= break_at:
            break

    print(f"Get tickers in folder, adding Datetime...")
    stacked = stacked.with_columns(Datetime=stacked['Date'] + ' ' + stacked['Time'])
    stacked = stacked.with_columns(Datetime=stacked['Datetime'].str.ljust(26, '0'))
    stacked = stacked.with_columns(Datetime=stacked['Datetime'].str.to_datetime())

    if offset:
        stacked = stacked.with_columns(Datetime = stacked['Datetime'].dt.offset_by("-" + str(offset) + "h"))
        return stacked.sort(["Date", "Time"]).to_pandas()
    else:
        return stacked.sort(["Date", "Time"]).to_pandas()


def get_orders_in_row(
        trades: pd.DataFrame, seconds_split: int = 1, orders_on_same_price_level: bool = False
) -> (pd.DataFrame, pd.DataFrame):

    '''
    This function gets prints "anxiety" over the tape :-)
    !!! Attention !!! Pass to this function a dataset in which the  the "Datetime" columns is in datetime format.

    :param trades: canonical trades executed
    :param seconds_split: seconds to measure the speed of the tape
    :param orders_on_same_price_level: if True, the anxiety is considered on orders at same price level
    :return: anxiety over the market on both ask/bid sides
    '''

    print(f"Get orders in row...")

    present = 0
    for el in ['Date', 'Time']:
        if el in trades.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a trade dataframe that has Date and Time columns')

    if 'Datetime' not in trades.columns:
        trades.insert(0, 'Datetime', pd.to_datetime(trades['Date'] + ' ' + trades['Time']))
        trades.sort_values(['Datetime'], ascending=True, inplace=True)
    elif 'Datetime' in trades.columns:
        trades.sort_values(['Datetime'], ascending=True, inplace=True)

    def manage_speed_of_tape(trades_on_side: pd.DataFrame, side: int = 2,
                             same_price_level: bool = False) -> pd.DataFrame:

        ############################## EXECUTE TRADES ON SIDE SEPARATELY ####################################
        trades_on_side = trades_on_side[(trades_on_side.TradeType == side)].reset_index(drop=True)
        trades_on_side.sort_values(['Datetime'], ascending=True, inplace=True)
        #####################################################################################################

        vol_, dt_, count_, price_, idx_ = list(), list(), list(), list(), list()
        len_ = trades_on_side.shape[0]
        i    = 0

        while i < len_:

            start_time  = trades_on_side.Datetime[i]
            start_vol   = trades_on_side.Volume[i]
            start_price = trades_on_side.Price[i]
            counter     = 0

            for j in range(i + 1, len_):
                delta_time = trades_on_side.Datetime[j] - start_time
                ############################################################################################
                if delta_time.total_seconds() <= seconds_split and \
                   ((not same_price_level) or (same_price_level and start_price == trades_on_side.Price[j])):
                    start_vol += trades_on_side.Volume[j]
                    counter   += 1
                else:
                    break
                ############################################################################################

            if counter:
                vol_.append(start_vol)
                dt_.append(trades_on_side.Datetime[j - 1])
                price_.append(trades_on_side.Price[j - 1])
                idx_.append(trades_on_side.Index[j - 1])
                count_.append(counter + 1)
                i = i + counter + 1
            else:
                i += 1

        return pd.DataFrame({'Datetime':   dt_,
                             'Volume':     vol_,
                             'Counter':    count_,
                             'Price':      price_,
                             'TradeType':  [side] * len(price_),
                             'Index':      idx_})

    ask = None
    bid = None

    # Manage speed of tape on the ASK, first
    try:
        ask = manage_speed_of_tape(trades, 2, orders_on_same_price_level).sort_values(['Datetime'], ascending=True)
    except Exception as e:
        print(e)

    # Manage speed of tape on the BID, secondly.
    try:
        bid = manage_speed_of_tape(trades, 1, orders_on_same_price_level).sort_values(['Datetime'], ascending=True)
    except Exception as e:
        print(e)

    return ask, bid


def plot_half_hour_volume(
        data_already_read: bool,
        data: pd.DataFrame,
        data_path: str = "",
        data_name: str = "",
) -> None:

    """
    This function helps to understand the "volume smile" so that the peak in volume given hal hours is the market open
    :param data_path: file system path to the data
    :param data_name: file name to import
    :return: a plot in matplotlib with bars per half-hour (the bigger counting bar is the one that finds market opens)
    """

    print(f"Plot half hour volume...")

    if not data_already_read:
        try:
            data = pd.read_csv(os.path.join(data_path, data_name), sep=";")
        except:
            data = pd.read_csv(os.path.join(data_path, data_name), sep=",")
    else:
        data = data

    data = data[data.Price != 0]  # Remove recording impurities...
    data = data.assign(Index=np.arange(0, data.shape[0], 1))  # Set an index fro reference plotting...
    data = data.assign(Hour=data.Time.str[:2].astype(str))  # Extract the hour...
    data = data.assign(Minute=data.Time.str[3:5].astype(int))  # Extract the minute...
    data = data.assign(HalfHour=data.Hour.str.zfill(2) + data.Minute.apply(half_hour))  # Identifies half hours...

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
    value_counts_num = value_counts_num.rename(columns={"Volume": "VolumeCount", "index": "VolumeSize"})
    value_counts_per = pd.DataFrame(data["Volume"].value_counts(normalize=True)).reset_index()
    value_counts_per = value_counts_per.rename(columns={"Volume": "VolumePerc", "index": "VolumeSize"})
    value_counts_per = value_counts_per.assign(VolumePerc=value_counts_per.VolumePerc * 100)

    stats = value_counts_num.merge(right=value_counts_per, how="left", on="VolumeSize").reset_index(drop=True)
    stats.sort_values(["VolumeSize"], ascending=True, inplace=True)
    stats = stats.assign(VolumePercCumultaive=np.cumsum(stats.VolumePerc))

    return stats


def get_new_start_date(
        data: pd.DataFrame, sort_values: bool = False
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
    data_last_date_notna = data['Date_Shift'].head(2).values[1]  # Take first two elements and the second one must be not empty...
    data['Date_Shift']   = data['Date_Shift'].fillna(data_last_date_notna)
    data['DayStart']     = np.where(data.Date != data.Date_Shift, 1, 0)

    return data.drop(['Date_Shift'], axis=1)


def get_market_evening_session(
        data: pd.DataFrame
):
    '''
    This function defines session start and end given Chicago Time.
    Pass to this function a DataFrame with Datetime offset !
    '''

    print(f"Assign sessions labels...")

    condlist   = [(data.Datetime.dt.time >= SESSION_START_TIME) & (data.Datetime.dt.time <= SESSION_END_TIME),
                (data.Datetime.dt.time <= SESSION_START_TIME) | (data.Datetime.dt.time >= SESSION_END_TIME)]
    choicelist = ['RTH', 'ETH']

    return np.select(condlist, choicelist)


def print_constants():
    print(SESSION_START_TIME)
    print(SESSION_END_TIME)
    print(EVENING_START_TIME)
    print(EVENING_END_TIME)
    print(KDE_VARIANCE_VALUE)
    print(VALUE_AREA)
    print(VWAP_BAND_OFFSET_1)
    print(VWAP_BAND_OFFSET_2)
    print(VWAP_BAND_OFFSET_3)
    print(VWAP_BAND_OFFSET_4)
