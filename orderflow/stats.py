'''
Inside this file we collect some statistic over the data that we are interested to.
In this file we collect all the analysis like if they were pure Python functions.
'''


import os
import datetime

import numpy as np
import polars as pl
from tqdm import tqdm
from Orderflow.orderflow.paths import get_current_os
from Orderflow.orderflow.configuration import FUTURE_VALUES, FUTURE_LETTERS


ticker         = 'ZN'
year           = '23'
current_path   = get_current_os()
current_ticker = FUTURE_VALUES['Ticker_Future_Letters'][2].split(' ')
current_ticker = [ticker + str(x) + year for x in current_ticker]
all_files      = os.listdir(current_path)
all_files      = [item for item in all_files if any(sub in item for sub in current_ticker)]


dataframe = pl.DataFrame()
for file in tqdm(all_files):
    dataframe = pl.concat([dataframe, pl.read_csv(current_path + '/' + file, separator=';')])
    print(dataframe['Time'][-1], '\n')

dataframe     = dataframe.with_columns(pl.col("Time").str.strptime(pl.Time))
dataframe     = dataframe.with_columns(pl.col("Time").dt.hour().alias('Hour'))


dates = dataframe['Date'].unique()
hours = list()
for date in tqdm(dates):
    # Filter the DataFrame
    filtered_df = dataframe.filter((pl.col("Date") == date))
    result      = (
                  filtered_df.group_by("Hour").
                  agg(pl.col("Volume").sum().alias("TotVol")))
    result      = result.sort('TotVol', descending=True)
    hours.append(result['Hour'][0])


pl.Series(hours).value_counts(sort=True)




tmp = pl.read_csv(current_path + '/' + file, separator=';')
tmp = tmp.with_columns(Index       = pl.Series(np.arange(0, tmp.shape[0], 1)))
tmp = tmp.with_columns(Datetime    = tmp['Date'] + ' ' + tmp['Time'])


tmp = tmp.with_columns(Datetime_DT = pl.col("date_str").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S.%f"))



