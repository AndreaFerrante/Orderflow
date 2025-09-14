from collections import deque
from typing import Optional
from tqdm import tqdm
import polars as pl
import pandas as pd
import numpy as np


def _get_datetime_fixed_pl(df: pl.DataFrame) -> pl.DataFrame:

    '''

    :param df: dataframe that has Date and Time as columns in string format
    :return: same input dataframe with a column named Datetime in "pl.Datetime" format (nanosecond format '%Y-%m-%d %H:%M:%S%.f')
    '''

    required_columns = {"Date", "Time"}
    missing_columns  = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Input Polars DataFrame is missing required columns: {missing_columns}")

    df = (df
          .with_columns(
                pl.concat_str([pl.col("Date"), pl.col("Time")], separator=" ")
                .alias("Datetime")
             )
          .with_columns(
                pl.col("Datetime").str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f')
             )
         )

    return df.sort("Datetime")


def _get_datetime_fixed_pd(df: pd.DataFrame) -> pl.DataFrame:

    '''

    :param df: dataframe that has Date and Time as columns in string format
    :return: same input dataframe with a column named Datetime in "pl.Datetime" format (nanosecond format '%Y-%m-%d %H:%M:%S%.f')
    '''

    required_columns = {"Date", "Time"}
    missing_columns  = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Input Pandas DataFrame is missing required columns: {missing_columns}")

    df = df.assign(Datetime = df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.assign(Datetime = pd.to_datetime(df['Datetime']))

    return df.sort_values("Datetime", ascending=True)


def compress_to_bar_once_range_met(tick_data:pd.DataFrame, price_range:int, tick_size:float=0.25) -> pd.DataFrame:

    """
    Transforms tick-by-tick data into range bars using NumPy.

    Parameters:
    - tick_data: DataFrame containing tick-by-tick data. Must have a 'Price' column.
    - price_range: The price range that each bar should represent.

    Returns:
    - range_bars_df: DataFrame containing range bars.
    """

    if not isinstance(tick_data, pd.DataFrame):
        raise ValueError("Attention, pass to compress in range function a Pandas DataFrame in type.")

    required_columns = {"Date", "Time", "Price", "Volume", "TradeType"}
    missing_columns  = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    #############################################
    tick_data = _get_datetime_fixed_pd(tick_data)
    #############################################

    price_array = tick_data['Price'].to_numpy()
    vol_array   = tick_data['Volume'].to_numpy()
    date_array  = tick_data['Date'].to_numpy()
    trade_type  = tick_data['TradeType'].to_numpy()
    time_array  = tick_data['Datetime'].dt.time.astype(str).to_numpy()

    dates   = deque()
    times   = deque()
    opens   = deque()
    highs   = deque()
    lows    = deque()
    closes  = deque()
    volumes = deque()
    volask  = deque()
    volbid  = deque()
    trades  = deque()

    # Initialize variables for the running bar
    running_open = price_array[0]
    running_high = price_array[0]
    running_low  = price_array[0]
    running_vol  = 0
    vol_ask      = 0
    vol_bid      = 0
    counter      = 0

    for price, volume, date, time, side in tqdm(zip(price_array, vol_array, date_array, time_array, trade_type)):

        running_high = max(running_high, price)
        running_low  = min(running_low, price)
        running_vol += volume

        if side == 1:
            vol_bid += volume
        else:
            vol_ask += volume

        counter += 1

        if abs(running_high - running_low) / tick_size >= price_range:

            dates.append(date)
            times.append(time)
            opens.append(running_open)
            highs.append(running_high)
            lows.append(running_low)
            closes.append(price)
            volumes.append(running_vol)
            volask.append(vol_ask)
            volbid.append(vol_bid)
            trades.append(counter)

            # Reset variables for the next bar
            running_open = price
            running_high = price
            running_low  = price
            running_vol  = 0
            counter      = 0
            vol_bid      = 0
            vol_ask      = 0

    range_bars_df = pd.DataFrame({
        'Date':           dates,
        'Time':           times,
        'Open':           opens,
        'High':           highs,
        'Low':            lows,
        'Close':          closes,
        'Volume':         volumes,
        'NumberOfTrades': trades,
        'BidVolume':      volbid,
        'AskVolume':      volask
    })

    return range_bars_df


def compress_to_volume_bars_pl(tick_data: pl.DataFrame, volume_threshold: float) -> pl.DataFrame:

    """
    Reshapes tick-by-tick trading data into volume bars.

    Parameters:
        tick_data (pl.DataFrame): Tick-by-tick data. Must contain at least the following columns:
                           - 'Datetime': Timestamps of the ticks.
                           - 'Price': Trade price.
                           - 'Volume': Volume traded at each tick.
        volume_threshold (float): The cumulative volume threshold for each volume bar.

    Returns:
        pl.DataFrame: A DataFrame with one row per volume bar containing:
                      - 'timestamp_open': Timestamp of the first tick in the bar.
                      - 'timestamp_close': Timestamp of the last tick in the bar.
                      - 'price_open': Price of the first tick in the bar.
                      - 'price_high': Highest price in the bar.
                      - 'price_low': Lowest price in the bar.
                      - 'price_close': Price of the last tick in the bar.
                      - 'volume': Total traded volume within the bar.
    Raises:
        ValueError: If the input DataFrame does not contain the required columns.
    """

    if not isinstance(tick_data, pl.DataFrame):
        raise ValueError("Attention, pass to the compress in range function a Polars DataFrame in type.")

    required_columns = {"Date", "Time", "Price", "Volume"}
    missing_columns  = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    #############################################
    tick_data = _get_datetime_fixed_pl(tick_data)
    #############################################

    try:

        df = tick_data.with_columns(
            (
                (pl.col("Volume").cum_sum() / volume_threshold)
                .ceil()
                .cast(pl.Int64)
            ).alias("group")
        )

        volume_bars = (
            df.group_by("group", maintain_order=False)
            .agg([
                pl.col("Datetime").first().alias("DatetimeOpen"),
                pl.col("Datetime").last().alias("DatetimeClose"),
                pl.col("Price").first().alias("Open"),
                pl.col("Price").max().alias("High"),
                pl.col("Price").min().alias("Low"),
                pl.col("Price").last().alias("Close"),
                pl.col("Volume").sum().alias("Volume"),
                pl.when(pl.col("TradeType") == 2)
                .then(pl.col("Volume"))
                .otherwise(0)
                .sum()
                .alias("AskVolume"),
                pl.when(pl.col("TradeType") == 1)
                .then(pl.col("Volume"))
                .otherwise(0)
                .sum()
                .alias("BidVolume"),
                pl.len().alias("NumberOfTrades")
            ])
            .sort("group")
            .drop("group")
        )

        return volume_bars

    except Exception as e:
        raise e


def compress_to_minute_bars_pl(
    tick_data: pl.DataFrame,
    win_compression: str         = '1m',
    time_column: str             = "Datetime",
    price_column: str            = "Price",
    volume_column: Optional[str] = "Volume",
    side_column: Optional[str]   = "TradeType",
    ask_value: int               = 2,
    bid_value: int               = 1,
) -> pl.DataFrame:

    """
        Aggregates tick-by-tick data into minute bars with extended volume details.

        This function takes tick-level trading data and groups it into minute intervals,
        returning the following aggregated details for each minute:
          - Open: the first price in the interval
          - High: the maximum price during the interval
          - Low: the minimum price during the interval
          - Close: the last price in the interval
          - Volume: the total volume during the interval (if volume_column is provided)
          - AskVolume: total volume for trades with side equal to `ask_value` (if side_column is provided)
          - BidVolume: total volume for trades with side equal to `bid_value` (if side_column is provided)
          - NumberOfTrades: the total number of trades executed in the interval

        The data is grouped by the specified time interval (`win_compression`) and
        the aggregation occurs over that interval.

        Parameters
        ----------
        tick_data : pl.DataFrame
            DataFrame containing tick-by-tick trading data, with at least the following columns:
            - time_column: timestamp for each trade
            - price_column: price at which the trade occurred
            - volume_column: the volume of the trade (optional)
            - side_column: the type of the trade (optional, e.g., 'ask' or 'bid')

        win_compression : str, default '1m'
            The time interval for aggregating the data (e.g., '1m' for one-minute bars).

        time_column : str, default 'Datetime'
            The name of the column containing timestamps in the tick data.

        price_column : str, default 'Price'
            The name of the column containing trade prices.

        volume_column : Optional[str], default 'Volume'
            The name of the column containing trade volumes. If None, volume aggregation is skipped.

        side_column : Optional[str], default 'TradeType'
            The name of the column indicating trade side (e.g., 'ask' or 'bid').
            If None, ask/bid volume aggregations are skipped.

        ask_value : str, default '2'
            The value in the side_column representing an ask trade.

        bid_value : str, default '1'
            The value in the side_column representing a bid trade.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing one row per minute with the aggregated bar data:
            - Open
            - High
            - Low
            - Close
            - Volume (if available)
            - AskVolume (if side_column is provided)
            - BidVolume (if side_column is provided)
            - NumberOfTrades
        """

    if not isinstance(tick_data, pl.DataFrame):
        raise ValueError("Attention, pass to the compress in range function a Polars DataFrame in type.")

    required_columns = {"Date", "Time", "Price", "Volume", "TradeType"}
    missing_columns = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    #############################################
    tick_data = _get_datetime_fixed_pl(tick_data)
    #############################################

    # Build a list of aggregation expressions.
    aggregations = [
        pl.col(price_column).first().alias("Open"),

        pl.col(price_column).max().alias("High"),

        pl.col(price_column).min().alias("Low"),

        pl.col(price_column).last().alias("Close"),

        pl.col(volume_column).sum().alias("Volume"),

        pl.len().alias("NumberOfTrades"),

        pl.when(pl.col(side_column) == ask_value)
        .then(pl.col(volume_column))
        .otherwise(0)
        .sum()
        .alias("AskVolume"),

        pl.when(pl.col(side_column) == bid_value)
        .then(pl.col(volume_column))
        .otherwise(0)
        .sum()
        .alias("BidVolume")
    ]

    # Group the data into minute-time intervals using groupby_dynamic. . .
    minute_bars = (
        tick_data.group_by_dynamic(
            time_column,
            every  = win_compression, # Group in one-minute intervals.
            closed = "left",          # Intervals are left-closed (include start, exclude end).
            label  = "left"           # Label each group by the left (start) boundary.
        )
        .agg(aggregations)
        .sort(time_column)
    )

    return minute_bars
