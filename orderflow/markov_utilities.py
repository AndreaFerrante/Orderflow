import os
import matplotlib
import numpy as np
import pandas as pd
from typing import List
from hmmlearn import hmm
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')


def threshold_prices_states(prices: List[float], threshold: float = 1e-8) -> List[str]:
    if len(prices) < 2:
        raise ValueError("Two prices min are needed to obtain the states.")

    states = list()

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > threshold:
            states.append("UP")
        elif diff < -threshold:
            states.append("DOWN")
        else:
            states.append("FLAT")
    return states


def adaptive_threshold_prices_states(prices: List[float], window: int = 20) -> List[str]:
    """
    Converts a sequence of prices into UP/DOWN/FLAT states with an adaptive threshold.
    The threshold is scaled by the volatility (standard deviation) of the returns
    over a moving window.

    :param prices: List of prices
    :param window: Window for volatility calculation
    :param base_threshold: Base threshold that will be multiplied by the volatility
    :return: List of states ('UP', 'DOWN', 'FLAT')
    """

    if len(prices) < 2:
        raise ValueError("At least two prices are needed to get the states done.")

    returns = np.diff(prices)
    states = list()

    for i in range(1, len(prices)):

        start_idx = max(0, i - window)
        local_returns = returns[start_idx:i] if i > 0 else returns[:1]
        vol = np.std(local_returns) if len(local_returns) > 1 else 1e-8
        mean = np.mean(local_returns) if len(local_returns) > 1 else 1e-8

        diff = prices[i] - prices[i - 1]

        if diff > (mean + 0.5 * vol):
            states.append("UP")
        elif diff < -(mean - 0.5 * vol):
            states.append("DOWN")
        else:
            states.append("FLAT")

    return states


def simulate_market_data(num_steps: int = 10000, seed: int = 123) -> pd.DataFrame:

    """
    It is going to simulate prices regime over a num_steps of candles by including prices and fake volume.
    It returns a DataFrame with the following columns: ['price', 'volume'].
    """

    np.random.seed(seed)

    # Prices simulated with some noise and a small trend...
    prices = [100.0]
    for _ in range(num_steps):
        prices.append(prices[-1] + np.random.normal(0.05, 0.1))

    # Simulated volume with average 1e5 and standard dev 1e4
    volume = np.random.normal(1e5, 1e4, size=len(prices))
    volume = np.maximum(volume, 0.0)  # no negative volumes

    return pd.DataFrame({
                        'price':  prices,
                        'volume': volume
                        })


def compute_df_features(df: pd.DataFrame, window_volatility: int = 20, window_slope: int = 5) -> pd.DataFrame:

    """
    Given a df with 'price' and 'volume' columns, we get:
    - Return (prices diff)
    - Moving window volatility (dev standard over returns given a window_volatility)
    - Slope of the moving averages
    - Log-volume (volume trasformation to avoid zeros)
    """

    df = df.copy()

    # 1. Return (diff) - We could use log return, here we adopt simple differences
    df['return'] = df['price'].diff().fillna(0.0)

    # 2. Volatility window
    rolling_std = df['return'].rolling(window=window_volatility).std().fillna(method='bfill')
    df['volatility'] = rolling_std

    # 3. Moving average slopes (fit a regression line):
    slopes = []
    prices_array = df['price'].values
    for i in range(len(df)):

        start_idx = max(0, i - window_slope + 1)
        segment = prices_array[start_idx: i + 1]

        if len(segment) < 2:
            slopes.append(0.0)
            continue
        # Linear fit over a range 0..len(segment)-1
        x = np.arange(len(segment))
        # a, b => y = a*x + b, e a è la pendenza
        a, b = np.polyfit(x, segment, 1)
        slopes.append(a)

    df['slope'] = slopes

    # 4. Log of the volume
    df['log_volume'] = np.log1p(df['volume'])  # log(1 + volume)

    # Fill in optional NaN
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df


def select_best_hmm_model(data: np.ndarray, n_states_range: List[int], covariance_type: str = 'full', criterion: str = 'bic', random_state: int = 42) -> hmm.GaussianHMM:

    """
    Given a feature matrix with shape (n_samples, n_features),
    this method tries different values to find the hidden states and select the model that
    minimizes BIC (or AIC).

    :param data: array 2D with shape (n_samples, n_features)
    :param n_states_range: list of all possible hidden states (example [2,3,4,5])
    :param covariance_type: 'full', 'diag', 'tied', 'spherical'
    :param criterion: 'bic' or 'aic'
    :param random_state: to be easy to reproduce
    :return: best fitted model

    """

    best_model = None
    best_score = np.inf

    for n_states in n_states_range:

        # Gaussian HMM gaussiano with n_states and n_components
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state
        )

        model.fit(data)  # fit on all the data !

        # Get the log-likelihood and the number of parameters
        log_likelihood = model.score(data)
        n_params = (
                n_states * (n_states - 1)   # transactions
                + n_states * data.shape[1]  # avergae for each state
        )

        # Covariance parameters
        if covariance_type == 'full':
            # Ogni stato ha n_features * (n_features+1)/2 param
            n_params += n_states * (data.shape[1] * (data.shape[1] + 1) // 2)
        elif covariance_type == 'diag':
            n_params += n_states * data.shape[1]
        elif covariance_type == 'tied':
            n_params += (data.shape[1] * (data.shape[1] + 1) // 2)
        elif covariance_type == 'spherical':
            n_params += n_states

        # AIC = 2*n_params - 2*log_likelihood
        # BIC = n_params*log(n_samples) - 2*log_likelihood
        n_samples = data.shape[0]
        if criterion.lower() == 'aic':
            current_criterion = 2 * n_params - 2 * log_likelihood
        else:
            # BIC by default
            current_criterion = n_params * np.log(n_samples) - 2 * log_likelihood

        # Let's get the best one !
        if current_criterion < best_score:
            best_score = current_criterion
            best_model = model

    if best_model is None:
        raise ValueError("No model selected, check the data.")

    return best_model


def concat_sc_bar_data(data_path:str, file_extension:str='txt'):

    '''
    This function reads files extracted from SierraChart.
    The name of the instrument is deducted from the file name, for instance, given this file name
    ESH24-CME.scid_BarData.txt, the added colum will be "ESH24-CME.scid_BarData.txt".

    :param data_path: path where the files are saved in
    :return: dataframe of stacked data
    '''

    if data_path == '':
        raise ValueError('data_path must not be null, it has to be a value.')

    files = os.listdir(data_path)
    data  = list()

    for file in files:

        if file.endswith(file_extension):

            single_file         = pd.read_csv(os.path.join(data_path, file), sep=',')
            single_file.columns = [str(x).strip() for x in single_file.columns]
            single_file.insert(0, 'Instrument', file.split('.')[0])
            data.append( single_file )

    ###############################################################################
    r_data = pd.concat(data)
    r_data = r_data.map(lambda x: x.strip() if isinstance(x, str) else x)
    r_data = r_data.assign(Date = pd.to_datetime(r_data['Date']))
    r_data.sort_values(['Date', 'Time'], ascending=[True, True], inplace=True)
    r_data.reset_index(drop=True, inplace=True)
    ###############################################################################

    return r_data


def plot_distribution_of_float_series(series: pd.Series, bins: int = 75, title: str = "Series Distribution") -> None:

    """
    Create a histogram plot with matplotlib to show
    the distribution of a pandas Series of float values.

    Parameters
    ----------
    series : pd.Series
        The Series whose distribution we want to view
    bins : int, optional
        Number of 'bins' for the histogram (default: 30)
    title : str, optional
        Chart title (default: "Series distribution")
    """

    if not pd.api.types.is_float_dtype(series):
        raise ValueError("The given series is not a float one. Pass a float series !")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(series, bins=bins, edgecolor='black', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    plt.show()
