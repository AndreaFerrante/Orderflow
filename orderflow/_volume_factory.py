import os
import pytz
import polars
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from orderflow.configuration import *
from datetime import datetime, timedelta
from dateutil.parser import parse


def half_hour(x) -> str:

    """
    Determines whether a given minute falls in the first or second half of an hour.

    Args:
        x (int): The minute part of a time, expected to be between 0 and 59.

    Returns:
        str: Returns "30" if the given minute is 30 or more, indicating the second half of an hour.
             Returns "00" if the minute is less than 30, indicating the first half of an hour.

    Example:
        >>> half_hour(45)
        '30'
        >>> half_hour(10)
        '00'
    """

    print(f"Half hour...")

    if x >= 30:
        return "30"
    else:
        return "00"


def quarter_hour(x):

    if x <= 15:
        return "15"
    elif x <= 30:
        return "30"
    elif x <= 45:
        return "45"
    else:
        return "60"


def correct_time_nanoseconds(ticker_to_correct: polars.DataFrame = None):

    '''
    OS sometimes record incorrectly the immediate initial nanoseconds at the beginning of a single second.
    This function gets all the nanoseconds recorded at the same padding length.
    '''

    if ticker_to_correct is None:
        raise Exception("Pass a DataFrame to clear the Time column on !")

    if isinstance(ticker_to_correct, pd.DataFrame):
        ticker_to_correct = polars.DataFrame(ticker_to_correct)

    def pad_after_period(value):
        if '.' in value:
            before_dot, after_dot = value.split('.', 1)
            padded_after_dot = after_dot.rjust(6, '0')
            return f"{before_dot}.{padded_after_dot}"
        else:
            return value

    #####################################################################################################################################
    '''This is the apply pandas function but in Polars, much faster'''
    ticker_to_correct = ticker_to_correct.with_columns(
        Time=ticker_to_correct['Time'].map_elements(pad_after_period))
    #####################################################################################################################################

    return ticker_to_correct


def apply_offset_given_dataframe(pl_df:polars.DataFrame, market:str=None):

    """
    Adjusts the 'Datetime' column in the provided Polars DataFrame by applying a time offset. The offset amount
    is determined by the 'market' parameter and the last hour recorded in the 'Hour' column of the DataFrame.

    The offset is computed to align the 'Datetime' values with a standard trading closing time, depending on the
    specified market ('CBOT' or 'CME'). For instance, if the last hour is 23, and the market is 'CBOT', 8 hours
    are subtracted; if 'CME', 7 hours are subtracted. This adjustment is aimed at standardizing the time to a
    reference market close time, providing a uniform time series data irrespective of the actual closing times
    recorded in the data.

    Args:
        pl_df (polars.DataFrame): A DataFrame with at least 'Datetime' and 'Hour' columns. 'Datetime' should be
                                  in datetime format, and 'Hour' should be extracted from 'Datetime' if not present.
        market (str, optional): Market identifier, should be either 'CBOT' or 'CME'. This is required to determine
                                the correct offset to apply. Defaults to an empty string.

    Returns:
        polars.DataFrame: A DataFrame with the 'Datetime' column adjusted according to the specified market's
                          standard closing time.

    Raises:
        ValueError: If 'Datetime' column is not present in the DataFrame.
        ValueError: If 'market' is not 'CBOT' or 'CME'.
        Exception: If the DataFrame cannot be processed due to incorrect or missing market information.

    Examples:
        data = {
                "Datetime": pl.date_range(low=pl.datetime(2023, 1, 1), high=pl.datetime(2023, 1, 1, 23), every='1h'),
                "Hour": list(range(24))
            }
        df = pl.DataFrame(data)
        modified_df = apply_offset_given_dataframe(df, market='CBOT')
        print(modified_df)

    Notes:
        - The function requires that 'market' be specified accurately to ensure correct time adjustments.
        - It is assumed that the input DataFrame is properly formatted with the necessary columns.
        - The function includes error handling to ensure robust processing against common data issues.

    """

    if market is None:
        raise Exception("To offset datetime you must pass the market from which the ticker has been extracted from.")

    ####################################################################################################################
    '''
    1. Extract hour from datetime columns
    2. Select the very last hour as reference !
    '''
    pl_df_gb  = pl_df.group_by('Date').agg([polars.col("Datetime").last()])
    pl_df_gb  = pl_df_gb.with_columns(Hour=pl_df_gb['Datetime'].dt.hour())
    last_hour = pl_df_gb.select(polars.col("Hour").mode())
    last_hour = int(last_hour['Hour'][0]) # ---> We take the most frequent LAST HOUR in the dataframe <---
    ####################################################################################################################

    if 'Datetime' not in pl_df.columns:
        '''Add Datetime column (datetime datatype, too) inside Polars DataFrame'''
        return None

    '''CBOT closes at 15:59:59, CME closes at 16:59:59'''
    if str(market).lower() == 'cbot':
        offset_addition = 1
    elif str(market).lower() == 'cme':
        offset_addition = 0
    else:
        raise Exception("Attention ! Pass a market that is CME or CBOT")

    if last_hour == 23:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(7 + offset_addition) + "h"))
    elif last_hour == 22:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(6 + offset_addition) + "h"))
    elif last_hour == 21:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(5 + offset_addition) + "h"))
    elif last_hour == 20:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(4 + offset_addition) + "h"))
    elif last_hour == 19:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(3 + offset_addition) + "h"))
    elif last_hour == 18:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(2 + offset_addition) + "h"))
    elif last_hour == 17:
        pl_df = pl_df.with_columns(Datetime=pl_df['Datetime'].dt.offset_by("-" + str(1 + offset_addition) + "h"))
    else:
        '''If we had a possible issue in file recorded, better to skip the timestamp correction'''
        return None

    return pl_df.sort(['Datetime'], descending=False)


def get_days_tz_diff(start_date, end_date, tz_from_str:str='Europe/Rome', tz_to_str:str='America/Chicago'):

    """
    Calculates and prints the time difference in hours between two timezones for each day in a specified date range.
    
    This function iterates through each day from the start_date to the end_date, calculates the time difference
    between the start timezone (defaulting to Europe/Rome) and the end timezone (defaulting to America/Chicago),
    and prints the difference in hours along with the date. This can be useful for analyzing the impact of daylight
    saving time changes over the specified period.

    Parameters:
    - start_date (datetime.date or datetime.datetime): The start date of the period for which to calculate time differences.
    - end_date (datetime.date or datetime.datetime): The end date of the period for which to calculate time differences.
    - tz_from_str (str, optional): The IANA timezone database string for the from timezone. Defaults to 'Europe/Rome'.
    - tz_to_str (str, optional): The IANA timezone database string for the to timezone. Defaults to 'America/Chicago'.

    Returns:
    - None: This function prints the time difference for each day in the specified range but does not return any value.

    Note:
    - The function assumes that both start_date and end_date are provided as timezone-naive objects and that
      the times for comparison at both locations are equivalent (i.e., same local time in both timezones).
    - The time difference calculation accounts for daylight saving time changes, if any, in the specified timezones.
    """

    #####################################
    # # Start and end dates
    # start_date = datetime(2023, 10, 29, 1, 0 , 0)
    # end_date   = datetime(2023, 11, 29, 1, 0 , 0)
    #####################################

    # Define the timezones for Chicago and Rome
    from_tz   = pytz.timezone(tz_from_str)
    to_tz     = pytz.timezone(tz_to_str)
    current_date = start_date

    while current_date < end_date:

        ref_from_tz  = from_tz.localize(current_date)
        ref_to_tz    = ref_from_tz.astimezone(to_tz)

        time_difference_current_date_from_tz      = int(ref_from_tz.strftime('%z')[1:3])
        time_difference_current_date_from_tz_sign = ref_from_tz.strftime('%z')[0]
        time_difference_current_date_to_tz        = int(ref_to_tz.strftime('%z')[1:3])
        time_difference_current_date_to_tz_sign   = ref_to_tz.strftime('%z')[0]

        if time_difference_current_date_from_tz_sign != time_difference_current_date_to_tz_sign:
            time_difference = time_difference_current_date_from_tz + time_difference_current_date_to_tz
        elif time_difference_current_date_from_tz > time_difference_current_date_to_tz:
            time_difference = time_difference_current_date_from_tz - time_difference_current_date_to_tz
        else:
            time_difference = time_difference_current_date_to_tz - time_difference_current_date_from_tz

        print(f"Datetime {ref_from_tz.strftime('%Y-%m-%d  %H:%M:%S')}, Chicago to Rome time difference: {time_difference} hours")
        current_date += timedelta(days=1)


def convert_datetime_tz(datetime_array:np.array, tz_from_str:str='Europe/Rome', tz_to_str:str='America/Chicago') -> np.array:
    """
    Calculates and convert naive datetime array from one timezone to another.

    :param datetime_array: an array of naive datetime elements to convert
    :param tz_from_str: The IANA timezone database string for the from timezone. Defaults to 'Europe/Rome'
    :param tz_to_str: The IANA timezone database string for the to timezone. Defaults to 'America/Chicago'.
    :return: an array of converted aware datetime elements
    """

    # Define the timezones for Chicago and Rome
    from_tz = pytz.timezone(tz_from_str)
    to_tz   = pytz.timezone(tz_to_str)

    len_         = len(datetime_array)
    result_array = []

    for dt in enumerate(datetime_array):
        ref_from_tz = from_tz.localize(dt[1])
        ref_to_tz = ref_from_tz.astimezone(to_tz)
        result_array.append(ref_to_tz)

    return result_array


def get_longest_columns_dataframe(path: str, ticker: str = "ES", single_file: str = '') -> list:

    """
    Scans CSV files in a given directory (optionally, a single file) to identify the file with the least number of columns.
    This is useful for determining a consistent set of columns when dealing with multiple CSV files that may have different structures.

    Args:
        path (str): The directory path containing CSV files to scan. Ignored if `single_file` is specified.
        ticker (str): A filter to select files starting with this ticker symbol. Defaults to "ES".
                      Only used when scanning multiple files in a directory.
        single_file (str): Path to a single CSV file to scan. If specified, `path` is ignored, and only this file is scanned.

    Returns:
        list: A list containing the names of the columns of the file with the least number of columns. 
              If `single_file` is specified, it returns the columns from that file.

    Note:
        - This function assumes that all CSV files are delimited by semicolons (`;`).
        - Only the first few rows of each file (2 for multiple files, 5 for a single file) are read to determine the columns,
          which improves performance when working with large files.

    Example:
        # For scanning all files in a directory
        columns = get_longest_columns_dataframe('/path/to/csv/files', ticker='AAPL')
        
        # For scanning a single file
        columns = get_longest_columns_dataframe('/path/to/csv/files', single_file='/path/to/csv/file.csv')
    """

    cols = [x for x in range(99999)]

    '''Get one file only to scan'''
    if single_file is not None:
        
        single = pd.read_csv( path + single_file, sep=";", nrows=5)

        if len(single.columns) < len(cols):
            cols = single.columns
        
        return list(cols)

    '''Get multiple file to scan'''
    files = [x for x in os.listdir(path) if x.startswith(ticker)]
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
        path:           str  = None,
        single_file:    str  = None,
        market:         str  = None,
        future_letters: list = None,
        cols:           list = None,
        ticker:         str  = "ES",
        year:           int  = 0,
        break_at:       int  = 99999,
        extension:      str  = 'txt',
        separator:      str  = ';',
) -> polars.DataFrame:

    """
    Processes files within a specified directory or a single file to extract and adjust financial ticker data,
    returning a Polars DataFrame with the adjusted data.

    This function reads multiple files specified by the combination of ticker symbols, future letters, and year,
    or a single specified file. It applies data corrections, filters out invalid data, and adjusts the datetime
    information based on the last recorded hour to align with a standard time (like Chicago Time for trading data).

    Args:
        path (str, optional): The path to the directory containing the files to be processed. Required if 'single_file' is not provided.
        single_file (str, optional): Specific single file to be processed. If provided, 'path' must also be specified.
        ticker (str, optional): The root ticker symbol used to identify files. Defaults to 'ES'.
        year (int, optional): The year associated with the futures contracts to help identify files. Defaults to 0, which must be updated by the user.
        future_letters (list, optional): List of future letters to identify specific contracts within the files.
        cols (list, optional): List of columns to read from the files. If not provided, it will be determined by calling 'get_longest_columns_dataframe'.
        break_at (int, optional): The maximum number of files to process before stopping. Defaults to a very large number to process all files.
        extension (str, optional): File extension of the files to be processed. Defaults to 'txt'.
        separator (str, optional): The character used to separate values in the file. Defaults to ';'.
        market (str, optional): The market identifier from which the ticker has been extracted. Required to apply the correct datetime offset.

    Returns:
        polars.DataFrame: A DataFrame containing the processed ticker data with additional datetime columns like 'Hour', 'Minute', and 'Second',
        and adjustments based on the last recorded hour.

    Raises:
        Exception: If 'path' is not specified when required.
        Exception: If 'future_letters' is not provided but is required for processing multiple files.
        Exception: If there is an issue with the datetime data in the file being processed.
        Exception: If no year of the ticker has been passed (i.e., year == 0).
        Exception: If 'market' is not specified.

    Example:
        df = get_tickers_in_folder(path="/data/tickers", ticker="ES", year=2023, future_letters=["H", "M", "U", "Z"])
        print(df.shape)
        (500, 8)

    Note:
        This function assumes the presence of helper functions 'apply_offset_given_dataframe' to adjust the datetime
        columns based on trading hours and another 'correct_time_nanoseconds' to correct the timestamps. Ensure these
        functions are correctly implemented and available in the scope.

        The function performs the following steps:
        1. Reads a single file if 'single_file' is specified, applying necessary data corrections and transformations.
        2. Reads and processes multiple files in a directory if 'path' and 'future_letters' are provided.
        3. Applies filters to remove invalid data entries.
        4. Constructs a 'Datetime' column by combining 'Date' and 'Time' columns and applies offset corrections.
        5. Concatenates data from all processed files into a single Polars DataFrame.

    """

    if path is None:
        raise Exception("Pass to the function a path where the files are stored in.")

    if cols is None:
        cols = get_longest_columns_dataframe(path=path, ticker=ticker, single_file=single_file)

    def correct_time_nanoseconds(ticker_to_correct: polars.DataFrame = None):

        '''
        OS sometimes record incorrectly the immediate initial nanoseconds at the beginning of a single second.
        This function gets all the nanoseconds recorded at the same padding length.
        '''

        if ticker_to_correct is None:
            raise Exception("Pass a DataFrame to clear the Time column on !")

        if isinstance(ticker_to_correct, pd.DataFrame):
            ticker_to_correct = polars.DataFrame(ticker_to_correct)

        def pad_after_period(value):
            if '.' in value:
                before_dot, after_dot = value.split('.', 1)
                padded_after_dot = after_dot.rjust(6, '0')
                return f"{before_dot}.{padded_after_dot}"
            else:
                return value

        #####################################################################################################################################
        '''This is the apply pandas function but in Polars, much faster'''
        ticker_to_correct = ticker_to_correct.with_columns(
            Time=ticker_to_correct['Time'].map_elements(pad_after_period))
        #####################################################################################################################################

        return ticker_to_correct


    '''-------------------------'''
    '''Read one file only'''
    if single_file is not None:
        
        print("Reading one single file, only...")
        
        single_file_polars = polars.read_csv(os.path.join(path, single_file), separator=separator, columns=cols, infer_schema_length=10_000)
        single_file_polars = single_file_polars.filter((polars.col('Date') != "1899-12-30") & (polars.col('Price') > 0))
        single_file_polars = correct_time_nanoseconds(single_file_polars)
        single_file_polars = single_file_polars.with_columns(Datetime = single_file_polars['Date'] + ' ' + single_file_polars['Time'])
        single_file_polars = single_file_polars.with_columns(Datetime = single_file_polars['Datetime'].str.to_datetime())
        
        return apply_offset_given_dataframe(pl_df=single_file_polars, market=market)


    '''-------------------------'''
    '''Select only desired files'''
    ticker  = str(ticker).upper()
    files   = [str(x).upper() for x in os.listdir(path) if x.startswith(ticker)]
    if year > 0:
        year  = str(year)
        files = [file for file in files if year in file]
    if future_letters is not None:
        future_letters = [str(letter).upper() for letter in future_letters] # Check all future letter are capital...
        files          = [file for file in files if any([letter for letter in future_letters if str(letter) in file])]


    '''-------------------------'''
    '''Read multiple files'''
    stacked = list()
    for idx, file in tqdm(enumerate(files)):

        print(f"Reading file {file} ... \n")

        if file.endswith(str(extension).upper()):
            single_file = polars.read_csv(os.path.join(path, file), separator=separator, columns=cols, infer_schema_length=10_000)
            single_file = single_file.filter((polars.col('Date') != "1899-12-30") & (polars.col('Price') > 0))
            single_file = apply_offset_given_dataframe(pl_df=single_file, market=market)

            if single_file is None:
                print(f"File named {file}, was not time offset ! Please check it.")
                continue

            stacked.append(single_file)

        if idx >= break_at:
            break

    print(f"Correcting Time and adding Datetime...")
    stacked = polars.concat(stacked, how="vertical")
    stacked = correct_time_nanoseconds(stacked)
    stacked = stacked.with_columns(Datetime = stacked['Date'] + ' ' + stacked['Time'])
    stacked = stacked.with_columns(Datetime = stacked['Datetime'].str.to_datetime())
    
    return stacked


def get_orders_in_row(trades: pd.DataFrame,
                      seconds_split: float = 1.0,
                      orders_on_same_price_level: bool = False,
                      min_volume_summation:int = 100000) -> (pd.DataFrame, pd.DataFrame):

    '''
    This function gets prints "anxiety" over the tape :-)
    !!! Attention !!! Pass to this function a dataset in which the  the "Datetime" columns is in datetime format.

    :param trades: canonical trades executed
    :param seconds_split: seconds to measure the speed of the tape
    :param orders_on_same_price_level: if True, the anxiety is considered on orders at same price level
    :param min_volume_summation: minimum value of volume summation to reach in order to trigger a new order
    :return: anxiety over the market on both ask/bid sides
    '''

    print(f"Get orders in row...")

    present = 0
    for el in ['Date', 'Time']:
        if el in trades.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a trade dataframe that has Date and Time columns.')

    if 'Datetime' not in trades.columns:
        trades.insert(0, 'Datetime', pd.to_datetime(trades['Date'] + ' ' + trades['Time']))
        trades.sort_values(['Datetime'], ascending=True, inplace=True)
    elif 'Datetime' in trades.columns:
        trades.sort_values(['Datetime'], ascending=True, inplace=True)

    def manage_speed_of_tape(trades_on_side: pd.DataFrame,
                             side: int              = 2,
                             same_price_level: bool = False,
                             min_vol_summation:int = 100000) -> pd.DataFrame:

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

                    if start_vol >= min_vol_summation:
                        break
                else:
                    break
                ############################################################################################

            if counter:
                if start_vol >= min_vol_summation:
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
        ask = manage_speed_of_tape(trades,
                                   2,
                                   orders_on_same_price_level,
                                   min_volume_summation).sort_values(['Datetime'], ascending=True)
    except Exception as e:
        print(f"While managing ASK the Speed of Tape, this error couured: {e}")

    # Manage speed of tape on the BID, secondly.
    try:
        bid = manage_speed_of_tape(trades,
                                   1,
                                   orders_on_same_price_level,
                                   min_volume_summation).sort_values(['Datetime'], ascending=True)
    except Exception as e:
        print(f"While managing BID the Speed of Tape, this error couured: {e}")

    return ask, bid


def get_orders_in_row_v2(trades: pd.DataFrame,
                         seconds_split: float              = 1.0,
                         orders_on_prices_level_range    = 0,
                         tick_size: float                  = 0.25,
                         min_volume_summation: int       = 1_000_000,
                         min_num_of_trades               = 1,
                         reset_counter_at_summation: bool = True) -> (pd.DataFrame, pd.DataFrame):

    '''
    This function gets prints "anxiety" over the tape :-)
    !!! Attention !!! Pass to this function a dataset in which the "Datetime" columns is in datetime format.
    This version includes also single trade with volume greater than min summation

    :param trades: canonical trades executed
    :param seconds_split: seconds to measure the speed of the tape
    :param orders_on_prices_level_range: it is the range in ticks within the volume summation is valid, default = 0 = same price level
    :param tick_size: single tick size (e.g. for the ES ticker, tick_value=0.25)
    :param min_volume_summation: minimum value of volume summation to reach in order to trigger a new order
    :param reset_counter_at_summation: if True when the summation of volume reaches the min the counter restart
    :return: anxiety over the market on both ask/bid sides
    '''

    present = 0
    for el in ['Date', 'Time']:
        if el in trades.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a trade dataframe that has Date and Time columns.')

    if 'Datetime' not in trades.columns:
        trades.insert(0, 'Datetime', pd.to_datetime(trades['Date'] + ' ' + trades['Time']))
        trades.sort_values(['Datetime'], ascending=True, inplace=True)
    elif 'Datetime' in trades.columns:
        trades.sort_values(['Datetime'], ascending=True, inplace=True)

    def manage_speed_of_tape(trades_on_side:         pd.DataFrame,
                             side:                   int   = 1,
                             prices_level_range:     int   = 0,
                             tick_size:              float = 0.25,
                             reset_cnt_at_summation: bool  = True,
                             min_num_of_trades:      int   = 1,
                             min_vol_summation:      int   = 0) -> pd.DataFrame:

        trades_on_side = trades_on_side[(trades_on_side['TradeType'] == side)].reset_index(drop=True)
        trades_on_side.sort_values(['Datetime'], ascending=True, inplace=True)

        vol_, dt_, count_, price_, idx_ = list(), list(), list(), list(), list()
        len_ = trades_on_side.shape[0]

        counter       = 0
        start_vol     = 0
        volume_arr    = np.array( trades_on_side['Volume'] )
        price_arr     = np.array( trades_on_side['Price'] )
        index_arr     = np.array( trades_on_side['Index'] )
        datetime_arr  = trades_on_side['Datetime']
        start_time    = datetime_arr[0]
        start_price   = price_arr[0]

        print('Processing get_order_in_row_v2 . . .')

        for j in tqdm(range(len_)):
            
            ###########################################################
            delta_time = (datetime_arr[j] - start_time).total_seconds()
            ###########################################################

            if (delta_time <= seconds_split) & ((abs(start_price - price_arr[j]) / tick_size) <= prices_level_range):
                start_vol  += volume_arr[j]
                counter    += 1
            else:
                counter     = 1
                start_vol   = volume_arr[j]
                start_time  = datetime_arr[j]
                start_price = price_arr[j]

            if (start_vol >= min_vol_summation) & (counter >= min_num_of_trades):
                count_.append(counter)
                vol_.append(start_vol)
                idx_.append(index_arr[j])
                dt_.append(datetime_arr[j])
                price_.append(price_arr[j])

                if j<len_ and reset_cnt_at_summation:
                    start_vol   = 0
                    counter     = 0
                    start_time  = datetime_arr[j + 1]
                    start_price = price_arr[j + 1]

        return pd.DataFrame({'Datetime':   dt_,
                             'Volume':     vol_,
                             'Counter':    count_,
                             'Price':      price_,
                             'TradeType': [side] * len(price_),
                             'Index':      idx_})

    ask = None
    bid = None

    # Manage speed of tape on the ASK, first
    try:
        ask = manage_speed_of_tape(trades,
                                   2,
                                   orders_on_prices_level_range,
                                   tick_size,
                                   reset_counter_at_summation,
                                   min_num_of_trades,
                                   min_volume_summation).sort_values(['Datetime'], ascending=True)
    except Exception as e:
        print(f"While managing ASK the Speed of Tape, this error couured: {e}")
        
    # Manage speed of tape on the BID, secondly.
    try:
        bid = manage_speed_of_tape(trades,
                                   1,
                                   orders_on_prices_level_range,
                                   tick_size,
                                   reset_counter_at_summation,
                                   min_num_of_trades,
                                   min_volume_summation).sort_values(['Datetime'], ascending=True)
    except Exception as e:
       print(f"While managing BID the Speed of Tape, this error couured: {e}")

    return ask, bid


def plot_half_hour_volume(data_already_read: bool, data: pd.DataFrame, data_path: str = "", data_name: str = "" ) -> None:

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


def get_volume_distribution(data: pd.DataFrame) -> pd.DataFrame:

    value_counts_num = pd.DataFrame(data["Volume"].value_counts()).reset_index()
    value_counts_num = value_counts_num.rename(columns={"Volume": "VolumeCount", "index": "VolumeSize"})
    value_counts_per = pd.DataFrame(data["Volume"].value_counts(normalize=True)).reset_index()
    value_counts_per = value_counts_per.rename(columns={"Volume": "VolumePerc", "index": "VolumeSize"})
    value_counts_per = value_counts_per.assign(VolumePerc=value_counts_per.VolumePerc * 100)

    stats = value_counts_num.merge(right=value_counts_per, how="left", on="VolumeSize").reset_index(drop=True)
    stats.sort_values(["VolumeSize"], ascending=True, inplace=True)
    stats = stats.assign(VolumePercCumultaive=np.cumsum(stats.VolumePerc))

    return stats


def get_new_start_date(data: pd.DataFrame, sort_values: bool = False) -> pd.DataFrame:
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


def get_market_evening_session(data: pd.DataFrame, ticker: str):
    '''
    This function defines session start and end given Chicago Time.
    Pass to this function a DataFrame with Datetime offset !
    '''

    print(f"Assign sessions labels...")

    condlist = [
          (data.Datetime.dt.time >= FUTURE_VALUES.loc[FUTURE_VALUES['Ticker'] == ticker, 'RTH_StartTime'].values[0]) & (
           data.Datetime.dt.time <= FUTURE_VALUES.loc[FUTURE_VALUES['Ticker'] == ticker, 'RTH_EndTime'].values[0]),
          (data.Datetime.dt.time <= FUTURE_VALUES.loc[FUTURE_VALUES['Ticker'] == ticker, 'RTH_StartTime'].values[0]) | (
           data.Datetime.dt.time >= FUTURE_VALUES.loc[FUTURE_VALUES['Ticker'] == ticker, 'RTH_EndTime'].values[0])
         ]
    choicelist = ['RTH', 
                  'ETH']

    return np.select(condlist, choicelist)


def get_rolling_mean_by_datetime(pl_df, rolling_column_name, window_size='1m', return_pandas_series=False):

    import pandas, polars

    if isinstance(pl_df, pandas.core.frame.DataFrame):
        pl_df = polars.DataFrame(pl_df)
    elif not isinstance(pl_df, polars.dataframe.frame.DataFrame):
        raise Exception(f"Pass to function a polars or a pandas dataframe.")

    if "Date" not in pl_df.columns or "Datetime" not in pl_df.columns:
        raise Exception(f"Pass to the 'get_rolling_mean_by_date' a dataframe with Date and Datetime column names.")

    dates  = pl_df['Date'].unique().sort()
    rolled = None

    print(f'Getting the rolling mena for {rolling_column_name} by Datetime done...')
    for date in tqdm(dates):
        subset = pl_df.filter(pl_df['Date'] == date)
        subset = subset.with_columns(polars.
                                     col(rolling_column_name).
                                     rolling_mean_by(window_size = window_size,
                                                     by          = 'Datetime',
                                                     closed      = 'both').
                                     fill_null(strategy='backward').
                                     alias('RolledColumn'))
        if rolled is not None:
            rolled = polars.concat([rolled, subset['RolledColumn']])
        else:
            rolled = subset['RolledColumn']

    if return_pandas_series:
        return pd.Series(rolled)

    return rolled


def get_next_tick(signal_df, ticker_df) -> list:

    """
    Computes the indices in the ticker DataFrame where the next favorable price movement occurs for each signal in the signal DataFrame.

    For each index in `signal_df['Index']`, starting from that index, the function searches forward in `ticker_df` to find the next tick where:

    - **Short Trade (`TradeType` == 1):**
        - If the price **decreases** from the initial price, the index where this occurs is recorded.
        - If the price **increases** from the initial price, the search stops without recording.

    - **Long Trade (`TradeType` == 2):**
        - If the price **increases** from the initial price, the index where this occurs is recorded.
        - If the price **decreases** from the initial price, the search stops without recording.

    Parameters
    ----------
    signal_df : pandas.DataFrame or polars.DataFrame
        DataFrame containing signal indices. Must contain an 'Index' column.
    ticker_df : pandas.DataFrame or polars.DataFrame
        DataFrame containing ticker data. Must contain 'Index', 'Price', and 'TradeType' columns.

    Returns
    -------
    list of int
        A list of indices in `ticker_df` where the next favorable price movement occurs for each signal.

    Raises
    ------
    Exception
        If 'Index' column is not present in `signal_df` or `ticker_df`.

    Notes
    -----
    - The function converts input DataFrames to Polars DataFrames for efficient computation if they are not already.
    - The 'TradeType' column in `ticker_df` should contain `1` for Short trades and `2` for Long trades.
    - The function assumes that `ticker_df` is ordered by 'Index'.
    - The function uses `tqdm` for progress indication during iteration; ensure `tqdm` is installed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> signal_df = pd.DataFrame({'Index': [0, 5, 10]})
    >>> ticker_df = pd.DataFrame({
    ...     'Index': np.arange(20),
    ...     'Price': np.random.rand(20),
    ...     'TradeType': [1, 2] * 10
    ... })
    >>> correct_indices = get_next_tick(signal_df, ticker_df)
    >>> print(correct_indices)
    [1, 6, 11]
    """

    import polars

    if 'Index' not in signal_df.columns:
        raise Exception(f'Pass to function "get_next_tick" a "signal_df" with Index column inside.')

    if 'Index' not in ticker_df.columns:
        raise Exception(f'Pass to function "get_next_tick" a "ticker_df" with Index column inside.')

    if not isinstance(ticker_df, polars.dataframe.frame.DataFrame):
        ticker_df = polars.DataFrame(ticker_df)

    if not isinstance(signal_df, polars.dataframe.frame.DataFrame):
        signal_df = polars.DataFrame(signal_df)

    correct_index    = list()
    signal_indexes   = signal_df['Index']
    ticker_df_single = ticker_df

    for index in tqdm(signal_indexes):

        ticker_df_single         = ticker_df_single.filter(pl.col('Index') >= index)
        ticker_df_single_side    = ticker_df_single['TradeType'][0]
        ticker_df_single_price   = ticker_df_single['Price'][0]

        for single_index in range(len(ticker_df_single)):

            # Short trade ...
            if ticker_df_single_side == 1:
                if ticker_df_single['Price'][single_index] > ticker_df_single_price:
                    break # We are short, we had a tick up the current short price so we stop.
                elif ticker_df_single['Price'][single_index] < ticker_df_single_price:
                    correct_index.append(index + single_index)
                    break
                else:
                    continue

            # Long trade ...
            elif ticker_df_single_side == 2:
                if ticker_df_single['Price'][single_index] < ticker_df_single_price:
                    break # We are long, we had a tick down the current short price so we stop.
                elif ticker_df_single['Price'][single_index] > ticker_df_single_price:
                    correct_index.append(single_index + index)
                    break
                else:
                    continue

    return correct_index


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




