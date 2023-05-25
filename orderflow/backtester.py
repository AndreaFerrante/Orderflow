from tqdm import tqdm
import random
import numpy as np
import pandas as pd

def get_tick_size(price: np.array):
    prices = pd.Series(price).unique()
    return abs(prices[0] - prices[1])


def update_datetime_signal_index(datetime_all, datetime_signal, index_, signal_idx_):

    # ---> We do not enter if a trade is already in place.

    while True:
        try:
            if datetime_signal[signal_idx_] > datetime_all[index_]:
                return signal_idx_
            else:
                signal_idx_ += 1
        except:
            return signal_idx_ - 1


def backtester(
    data: pd.DataFrame,
    signal: pd.DataFrame,
    tp: int,
    sl: int,
    tick_value: float,
    commission: float = 4.5,
    n_contacts: int   = 1,
    slippage_max: int = 0
) -> (pd.DataFrame, pd.DataFrame):

    '''
    This function is a high speed for loop to tick by tick check all the trades given take profit and stop loss.

    :param data: canonical tick by tick recorder dataset
    :param signal: dataframe of the all signals occured
    :param tp: take profit in ticks
    :param sl: stop loss in ticks
    :param tick_value: single tick value (e.g. for the ES ticker, tick_value=12.5 dollars)
    :param commission: commission value per dollars
    :param n_contacts: number of contracts per entry
    :return: 2 dataframes: one for the backtest, and one with all single dataframes ticks
    '''

    if not 'Index' in data.columns:
        raise Exception('Please, provide DataFrame with Index column !')

    present = 0
    for el in ['Date', 'Time']:
        if el in data.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a dataset with Date and Time columns.')

    ############################################
    len_             = data.shape[0]
    tick_size        = get_tick_size(data.Price)
    price_array      = np.array(data.Price)
    datetime_all     = np.array(data.Index)
    datetime_signal  = np.array(signal.Index)
    signal_tradetype = np.array(signal.TradeType)
    ############################################

    entry_time_      = []
    exit_time_       = []
    entry_index_     = []
    exit_index_      = []
    entry_price_     = []
    entry_price_pure = []
    entry_type_      = []
    exit_price_      = []
    success          = 0
    loss             = 0
    signal_idx       = 0
    entry_counter    = 0
    entry_price      = 0.0

    #################### SPEED IS LOOPING OVER BOOLEAN ARRAY ######################
    entries_times = np.where( np.isin(datetime_all, datetime_signal), True, False )
    ###############################################################################

    print('\n')
    for i in tqdm(range(len_)):

        if entries_times[i] and not entry_price:

            entry_counter += 1
            trade_type     = signal_tradetype[signal_idx]

            ####################################################################
            # Let's add slippage given the type of entry (1 == short, 2 == long)
            if trade_type == 1:
                slippage    = float(tick_size * random.randint(0, slippage_max))
                entry_price = float(price_array[i]) - slippage
                print(f'\nSHORT - Price array {price_array[i]}, slippage {slippage}, so price is {entry_price}')
            else:
                slippage    = float(tick_size * random.randint(0, slippage_max))
                entry_price = float(price_array[i]) + slippage
                print(f'\nLONG - Price array {price_array[i]}, slippage {slippage}, so price is {entry_price}')
            ####################################################################

            entry_index_.append( datetime_signal[signal_idx] )
            entry_time_.append(  data.Date[i] + ' ' + data.Time[i] )
            entry_price_.append( entry_price )
            entry_price_pure.append( price_array[i] )

        # ---> Long trade...
        elif entry_price != 0 and trade_type == 2:

            if entry_price - price_array[i] >= sl * tick_size:
                exit_index_.append( datetime_all[i] )
                exit_time_.append(  data.Date[i] + ' ' + data.Time[i] )
                exit_price_.append( price_array[i] )
                entry_type_.append( 'LONG' )
                entry_price = 0
                loss += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(
                        datetime_all, datetime_signal, i, signal_idx
                    )

            elif price_array[i] - entry_price > tp * tick_size:
                exit_index_.append( datetime_all[i] )
                exit_time_.append(  data.Date[i] + ' ' + data.Time[i] )
                exit_price_.append( price_array[i] )
                entry_type_.append('LONG')
                entry_price = 0
                success += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(
                        datetime_all, datetime_signal, i, signal_idx
                    )

        # ---> Short trade...
        elif entry_price != 0 and trade_type == 1:

            if entry_price - price_array[i] > tp * tick_size:
                exit_index_.append( datetime_all[i])
                exit_time_.append(  data.Date[i] + ' ' + data.Time[i])
                exit_price_.append( price_array[i])
                entry_type_.append( 'SHORT')
                entry_price = 0
                success += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(
                        datetime_all, datetime_signal, i, signal_idx
                    )

            elif price_array[i] - entry_price >= sl * tick_size:
                exit_index_.append( datetime_all[i] )
                exit_time_.append(  data.Date[i] + ' ' + data.Time[i] )
                exit_price_.append( price_array[i] )
                entry_type_.append( 'SHORT' )
                entry_price = 0
                loss += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(
                        datetime_all, datetime_signal, i, signal_idx
                    )

    profit_     = success * tp * n_contacts * tick_value
    loss_       = loss * sl * n_contacts * tick_value
    commission_ = entry_counter * n_contacts * commission
    net_profit_ = profit_ - loss_ - commission_

    print("\n")
    print(
        "-- PROFIT:",
        round(profit_, 2),
        "\n",
        "-- LOSS: ",
        round(loss_, 2),
        "\n",
        "-- COMMISSIONS: ",
        round(commission_, 2),
        "\n",
        "-- NET PROFIT",
        round(net_profit_, 2),
        "\n",
        "-- TOTAL TRADES",
        entry_counter,
        "\n",
        "-- PROFIT NET FACTOR",
        round( net_profit_ / loss_, 2 ),
        "\n",
        "-- PROFIT RATE",
        round( success / (success + loss), 2),
        "\n",
        "-- MIN DATE",
        data.Date.min(),
        "\n",
        "-- MAX DATE",
        data.Date.max(),
        "\n",
        "-- NUM. DATES",
        len( data.Date.unique() ),
        "\n",
        "-- NUM. CONTRACTS",
        n_contacts,
    )

    # Define backtest results here...
    backtest =  pd.DataFrame(
                {
                    "ENTRY_TIMES":           entry_time_,
                    "EXIT_TIMES":            exit_time_,
                    "ENTRY_PRICES_SLIPPAGE": entry_price_,
                    "ENTRY_PRICES_PURE":     entry_price_pure,
                    "EXIT_PRICES":           exit_price_,
                    "ENTRY_INDEX":           entry_index_,
                    "EXIT_INDEX":            exit_index_,
                    "ORDER_TYPE":            entry_type_
                }
            )
    backtest.insert(0, 'TRADE_INDEX', np.arange(1, backtest.shape[0] + 1, 1))
    backtest = backtest.assign(TRADE_GAIN = np.where( backtest.ORDER_TYPE == 'LONG',
                                                      backtest.EXIT_PRICES  - backtest.ENTRY_PRICES_SLIPPAGE,
                                                      backtest.ENTRY_PRICES_SLIPPAGE - backtest.EXIT_PRICES))


    # Define single trade snapshots here...
    trades = list()
    for idx, price in enumerate(backtest.ENTRY_PRICES_SLIPPAGE):

        single_trade = data[ ( data.Index >= backtest.ENTRY_INDEX[idx] ) & ( data.Index <= backtest.EXIT_INDEX[idx] ) ]
        single_trade.insert(0, 'TRADE_INDEX', idx + 1)
        single_trade.insert(1, 'MAE', price - np.min(single_trade.Price))
        single_trade.insert(2, 'MFE', np.max(single_trade.Price) - price)
        single_trade.insert(3, 'TRADE_DIRECTION', backtest.ORDER_TYPE[idx])
        trades.append( single_trade )

    return backtest, trades




