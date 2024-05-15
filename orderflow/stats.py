'''
Inside this file we collect some statistic over the data that we are interested to.
In this file we collect all the analysis like if they were pure Python functions.
'''


import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime, time
from Orderflow.orderflow.paths import get_current_os
from Orderflow.orderflow._volume_factory import get_tickers_in_folder


ticker = get_tickers_in_folder(ticker         = "ZN",
                               future_letters = ['M'],
                               year           = 24,
                               market         = "CBOT",
                               path           = get_current_os())


ticker_gb = (ticker.
             group_by(['Hour']).
             agg([pl.col('Volume').sum().alias('VolSum'),
                  pl.col('Volume').max().alias('VolMax')]).
             sort(['VolSum'], descending=True))
ticker_gb


pre_market  = ticker.filter((pl.col("Datetime").dt.time() > time(7, 30, 00)) &
                            (pl.col("Datetime").dt.time() < time(8, 30, 00)))
post_market = ticker.filter((pl.col("Datetime").dt.time() >  time(8, 30, 00)) &
                            (pl.col("Datetime").dt.time() < time(10, 30, 00)))

pre_market_gb  = (pre_market.
                  group_by(['Date']).
                  agg([pl.col('Volume').sum().alias('VolSumPre'),
                       pl.col('Price').max().alias('PriceMaxPre'),
                       pl.col('Price').min().alias('PriceMinPre')])).sort('Date')
pre_market_gb  = pre_market_gb.with_columns(PrePriceDelta     = pre_market_gb['PriceMaxPre'] - pre_market_gb['PriceMinPre'])
pre_market_gb  = pre_market_gb.with_columns(VolPricePreRatio  = (pre_market_gb['PrePriceDelta'] / pre_market_gb['VolSumPre']) * 1_000_000)

post_market_gb = (post_market.
                  group_by(['Date']).
                  agg([pl.col('Volume').sum().alias('VolSumPost'),
                       pl.col('Price').max().alias('PriceMaxPost'),
                       pl.col('Price').min().alias('PriceMinPost')])).sort('Date')
post_market_gb = post_market_gb.with_columns(PostPriceDelta    = post_market_gb['PriceMaxPost'] - post_market_gb['PriceMinPost'])
post_market_gb = post_market_gb.with_columns(VolPricePostRatio = (post_market_gb['PostPriceDelta'] / post_market_gb['VolSumPost']) * 1_000_000)

merged = pre_market_gb.join(
    post_market_gb,
    how = 'left',
    on  = 'Date'
).drop_nulls()

##############################################################################
np.corrcoef(merged['VolPricePreRatio'], merged['VolPricePostRatio'])
##############################################################################


# print_med_big     = ticker.filter( pl.col('Volume') >= 1000 )
# print_med_big_ask = print_med_big.filter( pl.col("TradeType") == 2 )
# print_med_big_bid = print_med_big.filter( pl.col("TradeType") == 1 )
# print_med_big.shape
#
# plt.plot(ticker['Datetime'], ticker['Price'], zorder=0)
# plt.scatter(print_med_big_ask['Datetime'], print_med_big_ask['Price'], zorder=1, s=50, edgecolor='black', color='lime')
# plt.scatter(print_med_big_bid['Datetime'], print_med_big_bid['Price'], zorder=1, s=50, edgecolor='black', color='red')






