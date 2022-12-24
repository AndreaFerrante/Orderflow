import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_ticker_in_folder(path:str, ticker:str='ES', cols:list=None) -> pd.DataFrame:

    '''
    Given a path and a ticker sign, this functions read all file in it starting with the ticker symbol (e.g. ES)
    :param path: path to ticker data to read
    :param ticker: ticker to read in the form of ES, ZN, ZB . . .
    :param cols: columns to import...
    :return: DataFrame of all read ticker files
    '''

    files   = os.listdir( path )
    stacked = []
    for file in tqdm.tqdm(files):

        if file.startswith(ticker):

            print(f'Reading file named {file} ...')
            if cols is None:
                stacked.append( pd.read_csv( os.path.join(path, file), sep=';') )
            else:
                stacked.append(pd.read_csv(os.path.join(path, file), sep=';', usecols=cols))

    return pd.concat( stacked )


def plot_half_hour_volume(data_path:str, data_name:str):

    '''
    This function helps understanding the "volume smile" so that the peak in volume given hal hours is the market open
    :param data_path: file system path to the data
    :param data_name: file name to import
    :return: a plot in matplotlib with bars per half-hour (the bigger counting bar is the one that finds market opens)
    '''

    def half_hour(x):
        if x >= 30:
            return "30"
        else:
            return "00"

    try:
        es = pd.read_csv( os.path.join(data_path, data_name), sep=';' )
    except:
        es = pd.read_csv(os.path.join(data_path, data_name), sep=',')

    es = es[es.Price != 0] # Remove recording impurities...
    es = es.assign(Index    = np.arange(0, es.shape[0], 1))  # Set an index fro reference plotting...
    es = es.assign(Hour     = es.Time.str[:2].astype(str))   # Extract the hour...
    es = es.assign(Minute   = es.Time.str[3:5].astype(int))  # Extract the minute...
    es = es.assign(HalfHour = es.Hour.str.zfill(2) + es.Minute.apply(half_hour)) # Identifies half hours...

    # Plot bar chart in which the
    max_volume = es.groupby(['HalfHour']).agg({'Volume':'sum'}).reset_index()
    plt.bar(max_volume.HalfHour, max_volume.Volume)
    plt.xlabel('HalfHour')
    plt.ylabel('Volume')
    plt.xticks(rotation=90)
    plt.tight_layout()



