from tqdm import tqdm
import numpy as np
import pandas as pd


def get_tick_size(
        price: np.array
):
    prices = pd.Series(price).unique()
    return abs(prices[0] - prices[1])


def update_datetime_signal_index(
        datetime_all, datetime_signal, index_, signal_idx_
):
    while True:
        try:
            if datetime_signal[signal_idx_] > datetime_all[index_]:
                return signal_idx_
            else:
                signal_idx_ += 1
        except:
            return signal_idx_ - 1


def backtester(
        datetime_all: np.array,
        datetime_signal: np.array,
        price_array: np.array,
        len_: int,
        tp: int,
        sl: int,
        tick_size: float,
        tick_value: int,
        n_contacts: int,
        commission: int
) -> pd.DataFrame:

    entry_time_   = []
    exit_time_    = []
    entry_price_  = []
    exit_price_   = []
    success       = 0
    loss          = 0
    signal_idx    = 0
    entry_counter = 0
    entry_price   = 0

    entries_times = np.where(np.isin(datetime_all, datetime_signal), True, False)

    for i in tqdm(range(len_)):

        # if datetime_all[i] == datetime_signal[signal_idx] and entry_price == 0:
        if entries_times[i] and entry_price == 0:

            entry_counter += 1
            entry_price = price_array[i]
            entry_time_.append(datetime_signal[signal_idx])
            entry_price_.append(entry_price)

        elif entry_price != 0:
            if entry_price - price_array[i] >= sl * tick_size:
                exit_time_.append(datetime_all[i])
                exit_price_.append(price_array[i])
                entry_price = 0
                loss += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(datetime_all, datetime_signal, i, signal_idx)

            elif price_array[i] - entry_price >= tp * tick_size:
                exit_time_.append(datetime_all[i])
                exit_price_.append(price_array[i])
                entry_price = 0
                success += 1

                if signal_idx < len(datetime_signal) - 1:
                    signal_idx = update_datetime_signal_index(datetime_all, datetime_signal, i, signal_idx)

    profit_     = success * tp * n_contacts * tick_value
    loss_       = loss * sl * n_contacts * tick_value
    commission_ = entry_counter * n_contacts * commission
    net_profit_ = profit_ - loss_ - commission_

    print('\n')
    print('-- SUCCESS PROFIT', profit_, '\n',
          '-- LOSS PROFIT', loss_, '\n',
          '-- COMMISSIONS', commission_, '\n',
          '-- NET PROFIT', net_profit_, '\n',
          '-- TOTAL TRADES', entry_counter)

    return (pd.DataFrame({'ENTRY_TIMES':  entry_time_,
                          'EXIT_TIMES':   exit_time_,
                          'ENTRY_PRICES': entry_price_,
                          'EXIT_PRICES':  exit_price_}))




