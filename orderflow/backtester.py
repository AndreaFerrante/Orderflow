from tqdm import tqdm
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
    n_contacts: int   = 1
) -> (pd.DataFrame, pd.DataFrame):

    if not 'Index' in data.columns:
        raise Exception('Please, provide DataFrame with Index column !')

    ############################################
    len_             = data.shape[0]
    tick_size        = get_tick_size(data.Price)
    price_array      = np.array(data.Price)
    datetime_all     = np.array(data.Index)
    datetime_signal  = np.array(signal.Index)
    signal_tradetype = np.array(signal.TradeType)
    ############################################

    entry_time_     = []
    exit_time_      = []
    entry_index_    = []
    exit_index_     = []
    entry_price_    = []
    entry_type_     = []
    exit_price_     = []
    success         = 0
    loss            = 0
    signal_idx      = 0
    entry_counter   = 0
    entry_price     = 0

    #################### SPEED IS LOOPING OVER BOOLEAN ARRAY ######################
    entries_times = np.where( np.isin(datetime_all, datetime_signal), True, False )
    ###############################################################################

    for i in tqdm(range(len_)):

        if entries_times[i] and not entry_price:

            entry_counter += 1
            entry_price    = price_array[i]
            trade_type     = signal_tradetype[signal_idx]
            entry_index_.append( datetime_signal[signal_idx] )
            entry_time_.append(  data.Date[i] + ' ' + data.Time[i] )
            entry_price_.append( entry_price )

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
        profit_,
        "\n",
        "-- LOSS: ",
        loss_,
        "\n",
        "-- COMMISSIONS: ",
        commission_,
        "\n",
        "-- NET PROFIT",
        net_profit_,
        "\n",
        "-- TOTAL TRADES",
        entry_counter,
        "\n",
        "-- PROFIT NET FACTOR",
        round( net_profit_ / loss_, 2 ),
    )

    # Define backtest results here...
    backtest =  pd.DataFrame(
                {
                    "ENTRY_TIMES":  entry_time_,
                    "EXIT_TIMES":   exit_time_,
                    "ENTRY_PRICES": entry_price_,
                    "EXIT_PRICES":  exit_price_,
                    "ENTRY_INDEX":  entry_index_,
                    "EXIT_INDEX":   exit_index_,
                    "ORDER_TYPE":   entry_type_
                }
            )
    backtest.insert(0, 'TRADE_INDEX', np.arange(1, backtest.shape[0] + 1, 1))
    backtest = backtest.assign(TRADE_GAIN = np.where( backtest.ORDER_TYPE == 'LONG',
                                                      backtest.EXIT_PRICES  - backtest.ENTRY_PRICES,
                                                      backtest.ENTRY_PRICES - backtest.EXIT_PRICES))


    # Define single trade snapshots here...
    trades = list()
    for idx, price in enumerate(backtest.ENTRY_PRICES):

        single_trade = data[ ( data.Index >= backtest.ENTRY_INDEX[idx] ) & ( data.Index <= backtest.EXIT_INDEX[idx] ) ]
        single_trade.insert(0, 'TRADE_INDEX', idx + 1)
        single_trade.insert(1, 'MAE', price - np.min(single_trade.Price))
        single_trade.insert(2, 'MFE', np.max(single_trade.Price) - price)
        single_trade.insert(3, 'TRADE_DIRECTION', 'Long')
        trades.append( single_trade )

    return backtest, pd.concat( trades, axis=0 )





