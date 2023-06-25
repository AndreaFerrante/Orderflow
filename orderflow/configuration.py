# Typically, a configuration file for session times includes the following information:
#
#     - Market Opening Time: This specifies the time at which the financial market opens for trading.
#     It could be represented as a specific hour, minute, and second of the day.
#
#     - Market Closing Time: This indicates the time at which the financial market closes for trading.
#     Similar to the opening time, it is represented as an hour, minute, and second of the day.
#
#     - Pre-Market and Post-Market Sessions: Some markets may have additional trading sessions that occur
#     before the official market opening or after the market closing. These sessions are often used by
#     traders to place orders or react to news events outside regular trading hours.
#
#     - Break Periods: Break periods refer to the intervals during which trading is temporarily halted
#     or limited. They can be scheduled within the trading day to allow for system maintenance,
#     news releases, or to handle extreme market volatility.
#
#     - Holidays and Non-Trading Days: The configuration file may also include a list of holidays and
#     non-trading days on which the financial market is closed. This ensures that trading systems and
#     algorithms account for these days and do not attempt to trade when the market is not available.


import pandas as pd


SESSION_START_TIME     = pd.to_datetime('08:30:00', format='%H:%M:%S').time() # Chicago timezone
SESSION_END_TIME       = pd.to_datetime('15:14:59', format='%H:%M:%S').time() # Chicago timezone
EVENING_START_TIME     = pd.to_datetime('15:15:00', format='%H:%M:%S').time() # Chicago timezone
EVENING_END_TIME       = pd.to_datetime('08:29:59', format='%H:%M:%S').time() # Chicago timezone
KDE_VARIANCE_VALUE     = 0.9
VALUE_AREA             = 0.68
VWAP_BAND_OFFSET_1     = 1
VWAP_BAND_OFFSET_2     = 2
VWAP_BAND_OFFSET_3     = 3
VWAP_BAND_OFFSET_4     = 4


