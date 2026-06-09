from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_montecarlo_analysis(trades:pd.DataFrame,
                            n_rows_sample:int,
                            n_simulations:int  = 100,
                            entry_col_name:str = 'Entry_Gains'):

    '''
    This function performs MonteCarlo charting.
    Random sampling the trades and creating the cumulative summation of all the trades simulates scenarios to
    emulate "n_simulations" equity curve patterns. For better visualisation, trades has to contain at least 50 trades.
    A very good MonteCarlo chart should have positive overall slope and none of the pattern should be below zero.

    trades: Trades as returned by OrderFlow backtester function
    n_simulations: Number of simulations (i.e. random sampling with replacement) to be performed
    n_rows_sample: Number of samples (i.e. rows extracted from the trades DataFrame) extracted at every iteration
    entry_col_name: Column trades's name containing the gain per single trade
    '''

    if not isinstance(trades, pd.DataFrame):
        raise Exception('Attention: pass trades as a Pandas DataFrame.')

    if not isinstance(n_simulations, int):
        raise Exception('Attention: pass n_simulations as an integer.')

    if not isinstance(n_rows_sample, int):
        raise Exception('Attention: pass n_rows_sample as an integer.')

    if not 'Datetime' in trades.columns:
        raise Exception('Attention, trades Dataframe has no Datetime column.')



    n_simulations  = 500
    n_rows_sample  = 100
    date_time      = pd.date_range(start='2022-01-01', end='2022-07-31', freq='h')
    gain_pod       = [-2, 3]
    gains          = [np.random.choice(gain_pod) for x in range(len(date_time))]
    trades         = pd.DataFrame({'Datetime': date_time, 'Entry_Gains': gains})
    entry_col_name = 'Entry_Gains'

    equity_patterns = list()
    ec_results      = list()

    print('\n Performing MonteCarlo sampling...')
    for simulation in tqdm(range(n_simulations)):

        sample = trades.sample(n = n_rows_sample).sort_values(['Datetime'], ascending=True)
        equity_patterns.append( sample[ entry_col_name ].cumsum() )
        ec_results.append( sample[ entry_col_name ].sum() )



    min_       = min(ec_results)
    max_       = max(ec_results)
    ec_results = pd.DataFrame({'EcPattern': ec_results})

    bins_      = [x for x in range(min_ - 1, max_ + 1, 5)]
    labels_    = [f'{i + 1}-{j}' for i, j in zip(bins_[:-1], bins_[1:])]
    s          = pd.cut(ec_results['EcPattern'], bins=bins_, labels=labels_)
    ec_results = ec_results.assign(s = s)
    df         = ec_results.groupby('s')['EcPattern'].sum().reset_index()
    df         = df.assign(CumPatterns = df.EcPattern.cumsum())
    plt.plot(df.CumPatterns); plt.show()



    print('\n Performing MonteCarlo plotting...')
    for simulation in tqdm(range(n_simulations)):

        plt.plot( equity_patterns[simulation], alpha=0.5, lw=1 )
        plt.xlabel('Trades Index')
        plt.ylabel('Cumulative Gain per Pattern')
        plt.title('Montecarlo Chart')

    plt.show()





