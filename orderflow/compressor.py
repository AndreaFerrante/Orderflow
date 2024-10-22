from collections import deque
from tqdm import tqdm
import pandas as pd
import numpy as np


def compress_to_range_bars(tick_data:pd.DataFrame, price_range:int, tick_size:float=None) -> pd.DataFrame:

    """
    Transforms tick-by-tick data into range bars using NumPy.

    Parameters:
    - tick_data: DataFrame containing tick-by-tick data. Must have a 'Price' column.
    - price_range: The price range that each bar should represent.

    Returns:
    - range_bars_df: DataFrame containing range bars.
    """

    if 'Price' not in tick_data.columns:
        raise Exception('Price is not in columns, pass it to the compress_to_range_bars function.')

    if 'Volume' not in tick_data.columns:
        raise Exception('Volume is not in columns, pass it to the compress_to_range_bars function.')

    if tick_size is None:
        raise Exception('Attention, yuo must pass tick_size parameter specific for the instrument used.')

    #############################################
    price_array = tick_data['Price'].to_numpy()
    vol_array   = tick_data['Volume'].to_numpy()
    date_array  = tick_data['Date'].to_numpy()
    time_array  = tick_data['Datetime'].dt.time.astype(str).to_numpy()
    #############################################

    dates   = deque()
    times   = deque()
    opens   = deque()
    highs   = deque()
    lows    = deque()
    closes  = deque()
    volumes = deque()

    # Initialize variables for the running bar
    running_open = price_array[0]
    running_high = price_array[0]
    running_low  = price_array[0]
    running_vol  = 0

    for price, volume, date, time in tqdm(zip(price_array, vol_array, date_array, time_array)):

        running_high = max(running_high, price)
        running_low  = min(running_low, price)
        running_vol += volume

        if abs(running_high - running_low) / tick_size >= price_range:

            dates.append(date)
            times.append(time)
            opens.append(running_open)
            highs.append(running_high)
            lows.append(running_low)
            closes.append(price)
            volumes.append(running_vol)

            # Reset variables for the next bar
            running_open = price
            running_high = price
            running_low  = price
            running_vol  = 0

    range_bars_df = pd.DataFrame({
        'Date':   dates,
        'Time':   times,
        'Open':   opens,
        'High':   highs,
        'Low':    lows,
        'Close':  closes,
        'Volume': volumes
    })

    return range_bars_df


def compress_to_volume_bars(tick_data:pd.DataFrame, volume_threshold:int) -> pd.DataFrame:

    """
    Transforms tick-by-tick data into volume bars.

    Parameters:
    - tick_data: DataFrame containing tick-by-tick data. Must have 'Price' and 'Volume' columns.
    - volume_threshold: The volume size that each bar should represent.

    Returns:
    - volume_bars: DataFrame containing volume bars.
    """

    if 'Price' not in tick_data.columns:
        raise Exception('Price is not in columns, pass it to the compress_to_volume_bars function.')

    if 'Volume' not in tick_data.columns:
        raise Exception('Volume is not in columns, pass it to the compress_to_volume_bars function.')

    # Initialize variables
    volume_bars    = deque()
    running_volume = 0
    running_open   = tick_data['Price'].iloc[0]
    running_high   = tick_data['Price'].iloc[0]
    running_low    = tick_data['Price'].iloc[0]


    for index, row in tqdm(tick_data.iterrows()):

        running_volume += row['Volume']
        running_high    = max(running_high, row['Price'])
        running_low     = min(running_low, row['Price'])
        running_close   = row['Price']

        if running_volume >= volume_threshold:

            volume_bars.append({
                'Open': running_open,
                'High': running_high,
                'Low': running_low,
                'Close': running_close,
                'Volume': running_volume
            })

            # Reset variables for the next bar
            running_volume = 0
            running_open   = running_close
            running_high   = running_close
            running_low    = running_close

    # Create a DataFrame from the bars
    volume_bars_df = pd.DataFrame(volume_bars)

    return volume_bars_df

