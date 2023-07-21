from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_montecarlo_analysis(trades:pd.DataFrame, n_simulations:int, n_rows_sample:int):

    '''
    This function performs MonteCarlo charting.
    Random sampling the trades and creating the cumulative summation of all the trades simulates scenarios to
    emulate "n_simulations" equity curve patterns. For better visualisation, trades has to contain at least 50 trades.
    A very good MonteCarlo chart should have positive overall slope and none of the pattern should be below zero.

    trades: Trades as returned by OrderFlow package backtester
    n_simulations: Number of simulations (i.e. random sampling with replacement) to be performed
    n_rows_sample: Number of samples (i.e. rows extracted from the trades DataFrame) extracted at every iteration
    '''

    if not isinstance(trades, pd.DataFrame):
        raise Exception('Attention: pass trades as a Pandas DataFrame.')

    if not isinstance(n_simulations, int):
        raise Exception('Attention: pass n_simulations as an integer.')

    if not isinstance(n_rows_sample, int):
        raise Exception('Attention: pass n_rows_sample as an integer.')

    trades          = trades.sort_values(['Datetime'], ascending=True, inplace=False)
    equity_patterns = list()

    for simulation in tqdm(range(n_simulations)):

        pass







