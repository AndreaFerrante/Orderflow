from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import polars as pl
from .exceptions import SessionTypeAbsent, IndexAbsent


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
    tick_size: float  = 0.25,
    commission: float = 4.5,
    n_contacts: int   = 1,
    slippage_max: int = 0,
    save_path: str    = '',
    adapt_sl_tp_to_slippage: bool = False,
    trade_in_RTH: bool = False
    ) -> (pd.DataFrame, pd.DataFrame):


    '''
    This function is a high speed for loop to tick by tick check all the trades given take profit and stop loss.

    :param data: canonical tick by tick recorder dataset
    :param signal: dataframe of the all signals occured
    :param tp: take profit in ticks
    :param sl: stop loss in ticks
    :param tick_size: single tick size (e.g. for the ES ticker, tick_value=0.25)
    :param tick_value: single tick value (e.g. for the ES ticker, tick_value=12.5 dollars)
    :param commission: commission value per dollars
    :param n_contacts: number of contracts per entry
    :param slippage_max: max number of random ticks of slippage to pick
    :param save_path: if not empty, path where to save the final trades
    :param adapt_sl_tp_to_slippage: move tp and sl given slippage
    :param trade_in_RTH: close/open trades just during RTH trading session (not before/after RTH session)
    :return: 2 dataframes: one for the backtest, and one with all single trades ticks
    '''


    #region UNCOMMENT TO TEST THE BACKTESTER FUNCTION
    # ##########################################
    # # Uncomment below to test this function...
    # data          = ticker
    # signal        = all_trades
    # tp            = 9
    # sl            = 9
    # tick_value    = 12.5
    # tick_size     = 0.25
    # commission    = 4.0
    # n_contacts    = 1
    # slippage_max  = 1
    #endregion


    if not 'Index' in data.columns:
        raise IndexAbsent('Please, provide DataFrame with Index column to procede.')

    if trade_in_RTH and 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column inside DataFrame passed but trade_in_RTH set to True: provide SessionType column.')

    present = 0
    for el in ['Date', 'Time']:
        if el in data.columns:
            present += 1

    if present < 2:
        raise Exception('Please, provide a dataset with Date and Time columns.')


    #############################################
    len_             = data.shape[0]
    price_array      = np.array(data.Price)
    datetime_all     = np.array(data.Index)
    datetime_signal  = np.array(signal.Index)
    signal_tradetype = np.array(signal.TradeType)
    #############################################


    RTH_indexes = pd.DataFrame()
    if trade_in_RTH:
        
        data_ = pl.from_pandas(data[["SessionType", "Date", "Index"]])
        data_ = data_.filter(pl.col("SessionType") == "RTH")
        data_ = data_.with_columns(IndexFirst = data_['Index'])
        data_ = data_.with_columns(IndexLast  = data_['Index'])
        data_ = (data_.
                group_by(["Date", "SessionType"]).
                agg([pl.col("IndexFirst").first(),
                    pl.col("IndexLast").last()]))
        data_       = data_.sort("IndexFirst")
        RTH_indexes = data_.to_pandas()
                
        # Let's filter the entries to avoid ENTERING OUTSIDE RTH hours...
        datetime_signal_RTH = np.zeros(0)
        for idx, row in RTH_indexes.iterrows():
            datetime_signal_filtered = datetime_signal[(datetime_signal >= row['IndexFirst']) & (datetime_signal <= row['IndexLast'])]
            datetime_signal_RTH      = np.append(datetime_signal_RTH, datetime_signal_filtered)

        # We have filtered out all those trading indexes that were NOT in the RTH, so preventing ENTERING during incorrect time...
        datetime_signal = datetime_signal_RTH


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
    position         = '' # This string keeps track if we are LONG or SHORT.
    SL_CONSTANT      = sl
    TP_CONSTANT      = tp


    #################### SPEED IS LOOPING OVER BOOLEAN ARRAY ######################
    entries_times = np.where( np.isin(datetime_all, datetime_signal), True, False )
    ###############################################################################


    for i in tqdm(range(len_)):

        if entries_times[i] and not entry_price:

            entry_counter += 1
            trade_type     = signal_tradetype[signal_idx]

            ######################################################################################################
            # Let's add slippage given the type of entry (1 == short, 2 == long)
            if trade_type == 1:
                slippage_tick = random.randint(0, slippage_max)
                slippage      = float(tick_size * slippage_tick)
                entry_price   = float(price_array[i]) - slippage
                #print(f'\nSHORT - Price array {price_array[i]}, slippage {slippage}, so price is {entry_price}')
            else:
                slippage_tick = random.randint(0, slippage_max)
                slippage      = float(tick_size * slippage_tick)
                entry_price   = float(price_array[i]) + slippage
                #print(f'\nLONG - Price array {price_array[i]}, slippage {slippage}, so price is {entry_price}')
            ######################################################################################################

            entry_index_.append( datetime_signal[signal_idx] )
            entry_time_.append(  data.Date[i] + ' ' + data.Time[i] )
            entry_price_.append( entry_price )
            entry_price_pure.append( price_array[i] )

            if adapt_sl_tp_to_slippage and slippage_tick > 0:
                sl = SL_CONSTANT + slippage_tick
                tp = TP_CONSTANT - slippage_tick
            else:
                sl = SL_CONSTANT
                tp = TP_CONSTANT

        # Check if we must trade during RTH...
        elif trade_in_RTH:
            
            datetime_signal_RTH = 0
            for idx, row in RTH_indexes.iterrows():
                datetime_signal_filtered =  (datetime_all[i] >= row['IndexFirst']) & (datetime_all[i] <= row['IndexLast'])
                datetime_signal_RTH      += datetime_signal_filtered

            if datetime_signal_RTH == 0:

                # ---> We are having a long trade AND real time turned on...
                if entry_price != 0 and trade_type == 2:

                    position = 'LONG'

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
                        entry_type_.append( 'LONG' )
                        entry_price = 0
                        success += 1

                        if signal_idx < len(datetime_signal) - 1:
                            signal_idx = update_datetime_signal_index(
                                datetime_all, datetime_signal, i, signal_idx
                            )

                # ---> We are having a short AND real time turned on...
                elif entry_price != 0 and trade_type == 1:

                    position = 'SHORT'

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

        # ---> We are having a long trade...
        elif entry_price != 0 and trade_type == 2:

            position = 'LONG'

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
                entry_type_.append( 'LONG' )
                entry_price = 0
                success += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(
                        datetime_all, datetime_signal, i, signal_idx
                    )

        # ---> We are having a short trade...
        elif entry_price != 0 and trade_type == 1:

            position = 'SHORT'

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
    profit_net_factor =  net_profit_ / loss_ if loss_ else net_profit_


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
        round( profit_net_factor, 2 ),
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

    backtest_results = pd.DataFrame({'Profit': [round(profit_, 2)],
                                     'Loss': [round(loss_, 2)],
                                     'Commissions': [round(commission_, 2)],
                                     'Net Profit': [round(net_profit_, 2)],
                                     'Total Trades': [entry_counter],
                                     'Profit Net Factor': [round( profit_net_factor, 2 )],
                                     'Profit Rate': [round( success / (success + loss), 2)],
                                     'Min Date': [data['Date'].min()],
                                     'Max Date': [data['Date'].max()]})


    # Manage we are in a position not closed at the end of the dataframe...
    if len(entry_time_) != len(exit_time_):

        '''
        If we are still in position once the main backtest is looping, we have a pending position open.
        We fix this here assuming that we close it with ending values.
        '''

        exit_index_.append( datetime_all[ len_ - 1 ])
        exit_time_.append( data.Date[ len_ - 1 ] + ' ' + data.Time[ len_ - 1 ])
        exit_price_.append( price_array[ len_ - 1 ])
        entry_type_.append( position )


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
    backtest['TRADE_GAIN'] = np.where( backtest.ORDER_TYPE == 'LONG',
                                       backtest.EXIT_PRICES - backtest.ENTRY_PRICES_SLIPPAGE,
                                       backtest.ENTRY_PRICES_SLIPPAGE - backtest.EXIT_PRICES)


    # Define single trade snapshots here...
    trades = list()
    for idx, price in enumerate(backtest.ENTRY_PRICES_SLIPPAGE):

        single_trade = data[ ( data.Index >= backtest.ENTRY_INDEX[idx] ) & ( data.Index <= backtest.EXIT_INDEX[idx] ) ]
        single_trade.insert(0, 'TRADE_INDEX', idx + 1)
        single_trade.insert(1, 'MAE', price - np.min(single_trade.Price))
        single_trade.insert(2, 'MFE', np.max(single_trade.Price) - price)
        single_trade.insert(3, 'TRADE_DIRECTION', backtest.ORDER_TYPE[idx])
        trades.append( single_trade )

    if save_path != '':
        backtest.to_csv(save_path, sep=';')

    return backtest, trades, backtest_results




