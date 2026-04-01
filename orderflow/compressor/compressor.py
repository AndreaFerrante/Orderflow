from collections import deque
from typing import Optional, Union
from tqdm import tqdm
import polars as pl
import pandas as pd


def _get_datetime_fixed_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Date and Time columns to Datetime in Polars DataFrame."""
    required_columns = {"Date", "Time"}
    missing_columns = required_columns - set(df.columns)

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


def _get_datetime_fixed_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date and Time columns to Datetime in Pandas DataFrame."""
    required_columns = {"Date", "Time"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Input Pandas DataFrame is missing required columns: {missing_columns}")

    df = df.copy()
    # Ensure Date and Time are strings, then combine and parse
    date_str = df['Date'].astype(str)
    time_str = df['Time'].astype(str)
    combined = date_str + ' ' + time_str
    
    # Parse with format that handles microseconds (HH:MM:SS.ffffff)
    # Use infer_datetime_format for better performance on large datasets
    df['Datetime'] = pd.to_datetime(combined, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    # Check for any parsing failures
    if df['Datetime'].isna().any():
        # Fallback to infer_datetime_format if format string didn't work
        df['Datetime'] = pd.to_datetime(combined, infer_datetime_format=True)
    
    return df.sort_values("Datetime", ascending=True)


def compress_to_bar_once_range_met(tick_data:pd.DataFrame, price_range:int, tick_size:float=0.25,
    overwrite_time_with_sierras = False) -> pd.DataFrame:

    """
    Transforms tick-by-tick data into range bars using NumPy.

    Parameters:
    - tick_data: Pandas DataFrame containing tick-by-tick data. 
                 Required columns: 'Date', 'Time', 'Price', 'Volume', 'TradeType'
                 Note: Date/Time are automatically converted to Datetime if not present.
    - price_range: The price range (in ticks) that each bar should represent.
                   Dollar range = price_range × tick_size
                   Example: price_range=16, tick_size=0.25 → 4-point range bars
    - tick_size: The minimum price movement (tick size) for the instrument. Default is 0.25.
                 (ES/NQ typically use 0.25, adjust for other instruments)
    - overwrite_time_with_sierras: bool
            If True, ensures timestamps are in Sierra Chart format.
            If False (default), automatically creates Datetime from Date+Time columns.
    
    Returns:
    - range_bars_df: Pandas DataFrame with columns:
                     Date, Time, Open, High, Low, Close, Volume, NumberOfTrades, BidVolume, AskVolume
    
    Notes:
    - Automatically handles Datetime creation from Date+Time if Datetime column missing
    - Efficiently processes tick data using NumPy arrays
    - Uses tqdm progress bar for visibility on large datasets
    """

    if not isinstance(tick_data, pd.DataFrame):
        raise ValueError("Attention, pass to compress in range function a Pandas DataFrame in type.")

    required_columns = {"Date", "Time", "Price", "Volume", "TradeType"}
    missing_columns  = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

    #############################################
    if overwrite_time_with_sierras:
        tick_data = _get_datetime_fixed_pd(tick_data)
    else:
        # Ensure Datetime column exists and is datetime dtype
        if "Datetime" not in tick_data.columns:
            tick_data = _get_datetime_fixed_pd(tick_data)
        elif tick_data['Datetime'].dtype == 'object':
            # Datetime column exists but is a string - convert it to datetime
            tick_data = tick_data.copy()
            tick_data['Datetime'] = pd.to_datetime(tick_data['Datetime'])
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


def compress_to_volume_bars(
    tick_data: Union[pd.DataFrame, pl.DataFrame],
    volume_amount: float
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Compress tick-by-tick trading data into volume bars.

    Parameters:
        tick_data: Tick-by-tick data (Pandas or Polars DataFrame).
                   Must contain: 'Price', 'Volume', 'TradeType' columns.
                   Optionally 'Datetime', 'Date', 'Time' for timestamps.
        volume_amount: The cumulative volume threshold for each volume bar.

    Returns:
        DataFrame with volume bars containing:
        - Open, High, Low, Close: OHLC prices
        - Volume: Total volume in bar
        - AskVolume, BidVolume: Volume breakdown (TradeType: 1=bid, 2=ask)
        - NumberOfTrades: Tick count
        - Timestamps (if available in input)
    """
    # Handle Polars
    if isinstance(tick_data, pl.DataFrame):
        return _compress_to_volume_bars_pl(tick_data, volume_amount)
    
    # Handle Pandas
    elif isinstance(tick_data, pd.DataFrame):
        return _compress_to_volume_bars_pd(tick_data, volume_amount)
    
    else:
        raise TypeError("tick_data must be Pandas or Polars DataFrame")


def _compress_to_volume_bars_pl(
    tick_data: pl.DataFrame,
    volume_amount: float
) -> pl.DataFrame:
    """Compress volume bars using Polars."""
    required_columns = {"Price", "Volume"}
    missing_columns = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame missing: {missing_columns}")
    
    # Ensure Datetime exists
    if "Datetime" not in tick_data.columns:
        if "Date" in tick_data.columns and "Time" in tick_data.columns:
            tick_data = _get_datetime_fixed_pl(tick_data)
        else:
            raise ValueError("Need either 'Datetime' or ('Date' + 'Time') columns")
    
    # Create volume bars using cumulative sum
    df = tick_data.with_columns(
        (
            (pl.col("Volume").cum_sum() / volume_amount)
            .floor()
            .cast(pl.Int64)
        ).alias("bar_group")
    )
    
    # Aggregate by bar group
    volume_bars = (
        df.group_by("bar_group", maintain_order=True)
        .agg([
            pl.col("Datetime").first().alias("OpenTime"),
            pl.col("Datetime").last().alias("CloseTime"),
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
        .drop("bar_group")
    )
    
    return volume_bars


def _compress_to_volume_bars_pd(
    tick_data: pd.DataFrame,
    volume_amount: float
) -> pd.DataFrame:
    """Compress volume bars using Pandas."""
    required_columns = {"Price", "Volume"}
    missing_columns = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame missing: {missing_columns}")
    
    tick_data = tick_data.copy()
    
    # Ensure Datetime exists
    if "Datetime" not in tick_data.columns:
        if "Date" in tick_data.columns and "Time" in tick_data.columns:
            tick_data = _get_datetime_fixed_pd(tick_data)
        else:
            raise ValueError("Need either 'Datetime' or ('Date' + 'Time') columns")
    
    # Create volume bars using cumulative sum
    tick_data['bar_group'] = (tick_data['Volume'].cumsum() // volume_amount).astype(int)
    
    # Aggregate by bar group
    volume_bars = tick_data.groupby('bar_group', sort=False).agg({
        'Datetime': ['first', 'last'],
        'Price': ['first', 'max', 'min', 'last'],
        'Volume': 'sum'
    }).reset_index(drop=True)
    
    # Flatten column names
    volume_bars.columns = ['OpenTime', 'CloseTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Calculate bid/ask volumes
    bid_vol = tick_data[tick_data['TradeType'] == 1].groupby('bar_group')['Volume'].sum()
    ask_vol = tick_data[tick_data['TradeType'] == 2].groupby('bar_group')['Volume'].sum()
    num_trades = tick_data.groupby('bar_group').size()
    
    volume_bars['BidVolume'] = bid_vol.values
    volume_bars['AskVolume'] = ask_vol.values
    volume_bars['NumberOfTrades'] = num_trades.values
    
    # Fill NaN values
    volume_bars = volume_bars.fillna(0)
    
    return volume_bars


def compress_to_minute_bars_pl(
    tick_data: pl.DataFrame,
    win_compression: str         = '1m',
    time_column: str             = "Datetime",
    price_column: str            = "Price",
    volume_column: Optional[str] = "Volume",
    side_column: Optional[str]   = "TradeType",
    ask_value: int               = 2,
    bid_value: int               = 1,
    overwrite_time_with_sierras = False
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
            The value in the side_column representing a bid trade

        overwrite_time_with_sierras: bool
            If True, the function will overwrite the time_column with Sierra Chart's timestamp format.
            This is useful when the input data has timestamps in a different format and needs to be standardized.

            
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
    if overwrite_time_with_sierras:
        tick_data = _get_datetime_fixed_pl(tick_data)
    #############################################
    
    if not tick_data[time_column].is_sorted():
        tick_data = tick_data.sort(time_column)

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


def compress_to_delta_bars(
    tick_data: Union[pd.DataFrame, pl.DataFrame],
    delta_threshold: float
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Compress tick-by-tick trading data into delta bars.

    Delta bars close when the absolute value of cumulative delta (BidVolume - AskVolume)
    reaches or exceeds the specified threshold. This matches SierraChart's Delta Volume
    Per Bar implementation.

    Parameters:
        tick_data: Tick-by-tick data (Pandas or Polars DataFrame).
                   Must contain: 'Price', 'Volume', 'TradeType' columns.
                   Optionally 'Datetime', 'Date', 'Time' for timestamps.
                   TradeType: 1=bid (buy), 2=ask (sell)
        delta_threshold: Absolute delta value threshold. New bar opens when |cumulative_delta| >= threshold.
                        Example: delta_threshold=1000 → bar closes when |bid_vol - ask_vol| >= 1000

    Returns:
        DataFrame with delta bars containing:
        - Open, High, Low, Close: OHLC prices
        - Volume: Total volume in bar
        - AskVolume, BidVolume: Volume breakdown (TradeType: 1=bid, 2=ask)
        - Delta: Net delta (BidVolume - AskVolume)
        - NumberOfTrades: Tick count
        - Timestamps (if available in input)

    Notes:
        - Implements SierraChart's Delta Volume Per Bar compression
        - Bars close when |delta| >= delta_threshold
        - Delta resets to zero at bar boundaries
    """
    # Handle Polars
    if isinstance(tick_data, pl.DataFrame):
        return _compress_to_delta_bars_pl(tick_data, delta_threshold)

    # Handle Pandas
    elif isinstance(tick_data, pd.DataFrame):
        return _compress_to_delta_bars_pd(tick_data, delta_threshold)

    else:
        raise TypeError("tick_data must be Pandas or Polars DataFrame")


def _compress_to_delta_bars_pd(
    tick_data: pd.DataFrame,
    delta_threshold: float
) -> pd.DataFrame:
    """Compress delta bars using Pandas."""
    required_columns = {"Price", "Volume", "TradeType"}
    missing_columns = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame missing: {missing_columns}")

    tick_data = tick_data.copy()

    # Ensure Datetime exists
    if "Datetime" not in tick_data.columns:
        if "Date" in tick_data.columns and "Time" in tick_data.columns:
            tick_data = _get_datetime_fixed_pd(tick_data)
        else:
            raise ValueError("Need either 'Datetime' or ('Date' + 'Time') columns")

    # Vectorized delta calculation: bid (+volume) vs ask (-volume)
    tick_data['tick_delta'] = tick_data['TradeType'].apply(
        lambda x: 1 if x == 1 else -1
    ) * tick_data['Volume']

    # Cumulative delta
    tick_data['cumsum_delta'] = tick_data['tick_delta'].cumsum()
    tick_data['abs_delta'] = tick_data['cumsum_delta'].abs()

    # Create bar groups: new bar closes when |cumsum_delta| >= threshold
    # Track bar boundaries: each time we hit/exceed threshold, increment bar group
    threshold_hit = (tick_data['abs_delta'] >= delta_threshold).astype(int)
    # Detect transitions from False to True
    bar_closes = threshold_hit & (threshold_hit.shift(fill_value=0) == 0)
    tick_data['bar_group'] = bar_closes.cumsum()

    # Aggregate by bar group
    delta_bars = tick_data.groupby('bar_group', sort=False).agg({
        'Datetime': ['first', 'last'],
        'Price': ['first', 'max', 'min', 'last'],
        'Volume': 'sum'
    }).reset_index(drop=True)

    # Flatten column names
    delta_bars.columns = ['OpenTime', 'CloseTime', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Calculate bid/ask volumes and delta
    bid_vol = tick_data[tick_data['TradeType'] == 1].groupby('bar_group')['Volume'].sum()
    ask_vol = tick_data[tick_data['TradeType'] == 2].groupby('bar_group')['Volume'].sum()
    num_trades = tick_data.groupby('bar_group').size()

    delta_bars['BidVolume'] = bid_vol.values
    delta_bars['AskVolume'] = ask_vol.values
    delta_bars['Delta'] = delta_bars['BidVolume'] - delta_bars['AskVolume']
    delta_bars['NumberOfTrades'] = num_trades.values

    # Fill NaN values
    delta_bars = delta_bars.fillna(0)

    return delta_bars


def _compress_to_delta_bars_pl(
    tick_data: pl.DataFrame,
    delta_threshold: float
) -> pl.DataFrame:
    """Compress delta bars using Polars."""
    required_columns = {"Price", "Volume", "TradeType"}
    missing_columns = required_columns - set(tick_data.columns)
    if missing_columns:
        raise ValueError(f"Input DataFrame missing: {missing_columns}")

    # Ensure Datetime exists
    if "Datetime" not in tick_data.columns:
        if "Date" in tick_data.columns and "Time" in tick_data.columns:
            tick_data = _get_datetime_fixed_pl(tick_data)
        else:
            raise ValueError("Need either 'Datetime' or ('Date' + 'Time') columns")

    # Calculate tick-level delta: bid (+) vs ask (-)
    df = tick_data.with_columns(
        pl.when(pl.col("TradeType") == 1)
        .then(pl.col("Volume"))
        .otherwise(-pl.col("Volume"))
        .alias("tick_delta")
    )

    # Cumulative delta
    df = df.with_columns(
        pl.col("tick_delta").cum_sum().alias("cumsum_delta"),
        pl.col("tick_delta").cum_sum().abs().alias("abs_delta")
    )

    # Create bar groups: new bar closes when |cumsum_delta| >= threshold
    # Detect when threshold is first hit in each bar
    df = df.with_columns(
        (pl.col("abs_delta") >= delta_threshold).cast(pl.Int32).alias("threshold_hit")
    )
    df = df.with_columns(
        (
            pl.col("threshold_hit") &
            (pl.col("threshold_hit").shift(1).fill_null(0) == 0)
        ).cast(pl.Int32).cum_sum().alias("bar_group")
    )

    # Aggregate by bar group
    delta_bars = (
        df.group_by("bar_group", maintain_order=True)
        .agg([
            pl.col("Datetime").first().alias("OpenTime"),
            pl.col("Datetime").last().alias("CloseTime"),
            pl.col("Price").first().alias("Open"),
            pl.col("Price").max().alias("High"),
            pl.col("Price").min().alias("Low"),
            pl.col("Price").last().alias("Close"),
            pl.col("Volume").sum().alias("Volume"),
            pl.when(pl.col("TradeType") == 1)
            .then(pl.col("Volume"))
            .otherwise(0)
            .sum()
            .alias("BidVolume"),
            pl.when(pl.col("TradeType") == 2)
            .then(pl.col("Volume"))
            .otherwise(0)
            .sum()
            .alias("AskVolume"),
            pl.len().alias("NumberOfTrades")
        ])
        .drop("bar_group")
    )

    # Calculate delta
    delta_bars = delta_bars.with_columns(
        (pl.col("BidVolume") - pl.col("AskVolume")).alias("Delta")
    )

    return delta_bars
