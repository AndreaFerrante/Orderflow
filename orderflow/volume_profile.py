import pandas as pd
import numpy as np
import operator
from tqdm import tqdm
from .exceptions import SessionTypeAbsent
from .configuration import *
from .volume_profile_kde import gaussian_kde_numba, get_kde_high_low_price_peaks


def get_dynamic_cumulative_delta(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given the canonical dataframe recorded, this function returns a dataframe with volume in ASK / BID restarted at the
    of a given day.
    :param data: canonical dataframe recorded (tick-by-tick wth the DOM attached)
    :return: canonical dataframe with the addition of the column with Ask / Bid volume
    """

    data.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
    dates = data.Date.unique()
    datas = list()

    for date in tqdm(dates):
        print(f"Dynamic VP processing date {date}")
        ask = data[
            (data.Date == date) & (data.TradeType == 2)
            ]  # get the ask per date...
        bid = data[
            (data.Date == date) & (data.TradeType == 1)
            ]  # get the bid per date...

        ask["AskVolume_VP"] = np.cumsum(ask.Volume)
        ask["BidVolume_VP"] = np.zeros(ask.shape[0])
        bid["AskVolume_VP"] = np.zeros(bid.shape[0])
        bid["BidVolume_VP"] = np.cumsum(bid.Volume)

        single_date = pd.concat([ask, bid], axis=0)
        single_date.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)
        single_date["AskVolume_VP"] = (
            single_date["AskVolume_VP"]
            .replace(to_replace=0, method="ffill")
            .astype(np.int64)
        )  # Fill zeros with last cumulative value
        single_date["BidVolume_VP"] = (
            single_date["BidVolume_VP"]
            .replace(to_replace=0, method="ffill")
            .astype(np.int64)
        )  # Fill zeros with last cumulative value

        datas.append(single_date)

    datas = pd.concat(datas)
    datas.sort_values(["Date", "Time"], ascending=[True, True], inplace=True)

    return datas


def get_dynamic_cumulative_delta_per_session_with_volume_filter(data: pd.DataFrame, volume_filter:int=35, skip_session_control:bool=False) -> pd.DataFrame:

    """
        Compute cumulative delta metrics per trading session with a volume filter, distinguishing between bid and ask volumes,
        and resetting the cumulative sums when transitioning from Regular Trading Hours (RTH) to Extended Trading Hours (ETH).

        This function processes a DataFrame containing trade data and calculates the cumulative delta
        of ask and bid volumes, as well as the total cumulative delta, over time. Only trades with volumes
        equal to or exceeding the specified `volume_filter` are included in the cumulative sums. The cumulative sums
        are reset when the session changes from RTH to ETH to reflect the start of a new trading session.

        Parameters
        ----------
        data : pandas.DataFrame
            The input DataFrame must contain the following columns:
            - 'Price': float, the trade price.
            - 'Volume': float or int, the trade volume.
            - 'TradeType': int, indicating the aggressor side of the trade:
                - 1 for trades at the bid price (aggressive sell orders).
                - Any other value for trades at the ask price (aggressive buy orders).
            - 'SessionType': str, indicating the type of trading session (e.g., 'RTH', 'ETH').
              The cumulative sums reset when the session changes from 'RTH' to 'ETH'.

        volume_filter : int, optional
            The minimum volume threshold for trades to be included in the cumulative calculations.
            Trades with volumes less than this threshold are ignored (treated as zero volume). Default is 100.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:
            - 'CD_Ask_Filtered': float, cumulative delta of ask volumes after applying the volume filter.
            - 'CD_Bid_Filtered': float, cumulative delta of bid volumes after applying the volume filter.
            - 'CD_Total_Filtered': float, cumulative delta of total volumes (CD_Bid_Filtered + CD_Ask_Filtered).

        Raises
        ------
        SessionTypeAbsent
            If the 'SessionType' column is not present in the input DataFrame.

        Notes
        -----
        - The cumulative delta is a technical indicator used in trading that measures the net difference
          between buying and selling pressure.
        - In this function, 'CD_Bid_Filtered' represents the cumulative volume of trades occurring at the bid price
          (aggressive sell orders) with volumes greater than or equal to the `volume_filter`, and 'CD_Ask_Filtered'
          represents the cumulative volume of trades occurring at the ask price (aggressive buy orders) with volumes
          greater than or equal to the `volume_filter`.
        - The function assumes that the 'TradeType' column uses the value 1 to represent trades at the bid price (sells)
          and other values for trades at the ask price (buys).
        - The cumulative delta calculations reset when transitioning from RTH to ETH sessions. This reset
          reflects the start of a new trading day or session, which is significant in intraday trading analysis.
        - The `volume_filter` parameter allows users to focus on trades with significant volume, filtering out smaller trades
          that may be considered noise.
        - The function uses numpy arrays for efficient computation and tqdm for progress display during the loop.

        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'Price': [100.5, 100.6, 100.7, 100.8],
        ...     'Volume': [50, 200, 150, 80],
        ...     'TradeType': [1, 0, 1, 0],
        ...     'SessionType': ['RTH', 'RTH', 'ETH', 'ETH']
        ... })
        >>> get_dynamic_cumulative_delta_per_session_with_volume_filter(data, volume_filter=100)
           CD_Ask_Filtered  CD_Bid_Filtered  CD_Total_Filtered
        0              0.0              0.0                0.0
        1            200.0              0.0              200.0
        2              0.0            150.0              150.0
        3              0.0            150.0              150.0

        In this example, only trades with volumes greater than or equal to 100 are included in the cumulative delta calculations.
        The first trade with volume 50 is ignored, while the subsequent trades are included or excluded based on the `volume_filter`.
        The cumulative deltas reset when the session changes from 'RTH' to 'ETH' between the second and third rows.
        """

    if not skip_session_control:
        if 'SessionType' not in data.columns:
            raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price          = np.array(data.Price)
    volume         = np.array(data.Volume)
    type           = np.array(data.TradeType)
    session        = np.array(data.SessionType)
    cd_ask         = np.zeros( data.shape[0] )
    cd_bid         = np.zeros( data.shape[0] )
    total          = np.zeros( data.shape[0] )
    len_           = len(price)

    if type[0] == 1:
        total[0]  = volume[0] if volume[0] >= volume_filter else 0
        cd_bid[0] = volume[0] if volume[0] >= volume_filter else 0
        cd_ask[0] = 0
    else:
        total[0]  = volume[0] if volume[0] >= volume_filter else 0
        cd_bid[0] = 0
        cd_ask[0] = volume[0] if volume[0] >= volume_filter else 0

    for i in tqdm(range(1, len_)):

        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):

            if type[i] == 1:
                total[i]  = volume[i] if volume[i] >= volume_filter else 0
                cd_bid[i] = volume[i] if volume[i] >= volume_filter else 0
                cd_ask[i] = 0
            else:
                total[i]  = volume[i] if volume[i] >= volume_filter else 0
                cd_bid[i] = 0
                cd_ask[i] = volume[i] if volume[i] >= volume_filter else 0

            continue

        volume_ = volume[i] if volume[i] >= volume_filter else 0

        if type[i] == 1:
            total[i]  = total[i - 1]  + volume_
            cd_bid[i] = cd_bid[i - 1] + volume_
            cd_ask[i] = cd_ask[i - 1]
        else:
            total[i]  = total[i - 1]  + volume_
            cd_ask[i] = cd_ask[i - 1] + volume_
            cd_bid[i] = cd_bid[i - 1]

    cd_filtered = pd.DataFrame({'CD_Ask_Filtered_' + str(volume_filter):   cd_ask,
                                'CD_Bid_Filtered_' + str(volume_filter):   cd_bid,
                                'CD_Total_Filtered_' + str(volume_filter): total})

    return cd_filtered.fillna(0)


def get_dynamic_cumulative_delta_per_session(data: pd.DataFrame) -> pd.DataFrame:

    """
    Compute cumulative delta metrics per trading session, distinguishing between bid and ask volumes,
    and resetting the cumulative sums when transitioning from Regular Trading Hours (RTH) to Extended
    Trading Hours (ETH).

    This function processes a DataFrame containing trade data and calculates the cumulative delta
    of ask and bid volumes, as well as the total cumulative delta, over time. The cumulative sums
    are reset when the session changes from RTH to ETH to reflect the start of a new trading session.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame must contain the following columns:
        - 'Price': float, the trade price.
        - 'Volume': float or int, the trade volume.
        - 'TradeType': int, indicating the aggressor side of the trade:
            - 1 for trades at the bid price (aggressive sell orders).
            - Any other value for trades at the ask price (aggressive buy orders).
        - 'SessionType': str, indicating the type of trading session (e.g., 'RTH', 'ETH').
          The cumulative sums reset when the session changes from 'RTH' to 'ETH'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'CD_Ask': float, cumulative delta of ask volumes.
        - 'CD_Bid': float, cumulative delta of bid volumes.
        - 'CD_Total': float, cumulative delta of total volumes (CD_Bid + CD_Ask).

    Raises
    ------
    SessionTypeAbsent
        If the 'SessionType' column is not present in the input DataFrame.

    Notes
    -----
    - The cumulative delta is a technical indicator used in trading that measures the net difference
      between buying and selling pressure.
    - In this function, 'CD_Bid' represents the cumulative volume of trades occurring at the bid price
      (aggressive sell orders), and 'CD_Ask' represents the cumulative volume of trades occurring at
      the ask price (aggressive buy orders).
    - The function assumes that the 'TradeType' column uses the value 1 to represent trades at the bid
      price (sells) and other values for trades at the ask price (buys).
    - The cumulative delta calculations reset when transitioning from RTH to ETH sessions. This reset
      reflects the start of a new trading day or session, which is significant in intraday trading analysis.
    - The function uses numpy arrays for efficient computation and tqdm for progress display during the loop.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Price': [100.5, 100.6, 100.7, 100.8],
    ...     'Volume': [10, 20, 15, 25],
    ...     'TradeType': [1, 0, 1, 0],
    ...     'SessionType': ['RTH', 'RTH', 'ETH', 'ETH']
    ... })
    >>> get_dynamic_cumulative_delta_per_session(data)
       CD_Ask  CD_Bid  CD_Total
    0     0.0    10.0      10.0
    1    20.0    10.0      30.0
    2     0.0    15.0      15.0
    3    25.0    15.0      40.0

    In this example, the cumulative deltas reset when the session changes from 'RTH' to 'ETH'
    between the second and third rows.
    """

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price          = np.array(data.Price)
    volume         = np.array(data.Volume)
    type           = np.array(data.TradeType)
    session        = np.array(data.SessionType)
    cd_ask         = np.zeros( data.shape[0] )
    cd_bid         = np.zeros( data.shape[0] )
    total          = np.zeros( data.shape[0] )
    len_           = len(price)

    if type[0] == 1:
        total[0]  = volume[0]
        cd_bid[0] = volume[0]
        cd_ask[0] = 0
    else:
        total[0]  = volume[0]
        cd_bid[0] = 0
        cd_ask[0] = volume[0]

    for i in tqdm(range(1, len_)):

        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):

            if type[i] == 1:
                total[i]  = volume[i]
                cd_bid[i] = volume[i]
                cd_ask[i] = 0
            else:
                total[i]  = volume[i]
                cd_bid[i] = 0
                cd_ask[i] = volume[i]

            continue

        if type[i] == 1:
            total[i]  = total[i - 1]  + volume[i]
            cd_bid[i] = cd_bid[i - 1] + volume[i]
            cd_ask[i] = cd_ask[i - 1]
        else:
            total[i]   = total[i - 1]  + volume[i]
            cd_bid[i]  = cd_bid[i - 1]
            cd_ask[i]  = cd_ask[i - 1] + volume[i]


    return pd.DataFrame({'CD_Ask':   cd_ask,
                         'CD_Bid':   cd_bid,
                         'CD_Total': total})


def get_daily_session_moving_POC(data: pd.DataFrame) -> np.array:

    """
    Given the canonical dataframe recorded, this function returns the Point of Control (i.e. POC) that is moving during
    the day giving the volume sentiment of uptrending market or choppy market of downtrending market.
    :param df: canonical dataframe recorded
    :return: numpy array for the daily moving poc
    """

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    volume    = np.array(data.Volume)
    price     = np.array(data.Price)
    session   = np.array(data.SessionType)
    poc_final = dict()
    len_      = len(price)
    poc_      = np.zeros(len_)

    poc_final[price[0]] = volume[0]
    poc_[0]             = price[0]


    for i in tqdm(range(1, len_)):

        cp = price[i]

        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):
            poc_final.clear()
            poc_final[cp] = volume[i]
            poc_[i]       = cp

            continue

        if cp in poc_final:
            poc_final[cp] += volume[i]
        else:
            poc_final[cp] = volume[i]
        poc_[i] = max(poc_final.items(), key=operator.itemgetter(1))[0]

    #poc_[len_ - 1] = price[len_ - 1]

    return poc_


def get_volume_profile_areas(data: pd.DataFrame) -> np.array:

    """
    Given the canonical dataframe recorded, this function returns an array with info if the price is in the VA or not
    :param df: canonical dataframe recorded
    :return: numpy array with values: POC, VA, na
    """

    print(f'Get volume profile areas...')

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price          = np.array(data.Price)
    volume         = np.array(data.Volume)
    session        = np.array(data.SessionType)

    total_volume   = 0
    poc_volume     = 0
    poc_index      = 0
    poc            = 0.00
    len_           = len(price)
    value_area     = np.array( ['na' for x in range(len_)] )
    volume_profile = {}


    for i in tqdm(range(1, len_)):

        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):

            volume_profile.clear()
            total_volume = 0
            poc          = 0
            poc_volume   = 0
            poc_index    = 0

        if price[i] in volume_profile.keys():
            volume_profile[price[i]] += volume[i]
        else:
            volume_profile[price[i]] = volume[i]

        if volume_profile[price[i]] > poc_volume:

            poc_volume    = volume_profile[price[i]]
            poc           = price[i]
            value_area[i] = 'POC'
            total_volume += volume[i]
            continue

        total_volume              += volume[i]
        vp_prices                  = np.array(sorted(volume_profile.keys()))
        vp_volumes                 = np.array([volume_profile[key] for key in vp_prices])
        poc_index                  = np.where(vp_prices == poc)[0][0]
        percentage_of_total_volume = int(total_volume * VALUE_AREA)
        current_sum_of_volume      = poc_volume
        advance_up                 = True
        advance_down               = True
        upper_volume               = 0
        lower_volume               = 0
        upper_index                = poc_index
        lower_index                = poc_index

        while True:

            if advance_up:
                if upper_index >= len(vp_prices) - 1:
                    upper_volume = 0
                else:
                    upper_index += 1
                    upper_volume += vp_volumes[upper_index]

            if advance_down:
                if lower_index <= 0:
                    lower_volume = 0
                else:
                    lower_index -= 1
                    lower_volume += vp_volumes[lower_index]

            if (upper_volume > 0 or lower_volume > 0) and (upper_volume > lower_volume):
                advance_up = True
                advance_down = False
                current_sum_of_volume += upper_volume
                upper_volume = 0
                if price[i] == vp_prices[upper_index]:
                    value_area[i] = 'VA'
                    break
            elif upper_volume > 0 or lower_volume > 0:
                advance_up = False
                advance_down = True
                current_sum_of_volume += lower_volume
                lower_volume = 0
                if price[i] == vp_prices[lower_index]:
                    value_area[i] = 'VA'
                    break

            if current_sum_of_volume > percentage_of_total_volume:
                break


    value_area[0] = 'POC'

    return value_area


def get_volume_profile_peaks_valleys(data: pd.DataFrame, tick_size: float = 0.25) -> np.array:

    """
    Given the canonical dataframe recorded, this function returns an array with info if the price is in a
    peak or valley of volumes
    :param df: canonical dataframe recorded
    :return: numpy array with values: High Peak = 2, High Peak Area = 1, Valley Peak = -2, Valley Peak Area = -1
    """

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price          = np.array(data.Price)
    volume         = np.array(data.Volume)
    session        = np.array(data.SessionType)
    len_           = len(price)
    peaks_valleys  = np.zeros(len_)
    volume_profile = {}

    for i in tqdm(range(len_ - 1)):

        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):
            volume_profile.clear()

        if price[i] in volume_profile.keys():
            volume_profile[price[i]] += volume[i]
        else:
            volume_profile[price[i]] = volume[i]

        source        = np.array(sorted(volume_profile.keys()))
        weight        = np.array([volume_profile[key] for key in source])
        kde           = gaussian_kde_numba(source=source, weight=weight, h=KDE_VARIANCE_VALUE)
        peaks_indexes = get_kde_high_low_price_peaks(kde)

        if np.any(peaks_indexes):

            # tick_size           = abs(source[1] - source[0])
            peaks_volumes       = weight[peaks_indexes]
            peaks_prices        = source[peaks_indexes]
            curr_price_position = np.searchsorted(peaks_prices, price[i])

            if curr_price_position == 0 and peaks_volumes[0] >= peaks_volumes[1]:
                peaks_valleys[i] = 2
            elif curr_price_position == 0 and peaks_volumes[0] < peaks_volumes[1]:
                peaks_valleys[i] = -2
            elif curr_price_position == len(peaks_prices) - 1 and peaks_volumes[-2] > peaks_volumes[-1]:
                peaks_valleys[i] = -2
            elif curr_price_position == len(peaks_prices) - 1 and peaks_volumes[-2] <= peaks_volumes[-1]:
                peaks_valleys[i] = 2
            elif peaks_prices[curr_price_position] == price[i] and peaks_volumes[curr_price_position] < peaks_volumes[
                curr_price_position + 1]:
                peaks_valleys[i] = -2
            elif peaks_prices[curr_price_position] == price[i] and peaks_volumes[curr_price_position] > peaks_volumes[
                curr_price_position + 1]:
                peaks_valleys[i] = 2
            else:
                distance_in_element      = (peaks_prices[curr_price_position] - peaks_prices[curr_price_position - 1]) / tick_size
                half_distance_in_element = int(distance_in_element / 2)

                try:
                    if price[i] >= source[peaks_indexes[curr_price_position] - half_distance_in_element] and \
                        peaks_volumes[curr_price_position] > peaks_volumes[curr_price_position - 1]:
                        peaks_valleys[i] = 1
                    elif price[i] >= source[peaks_indexes[curr_price_position] - half_distance_in_element] and \
                        peaks_volumes[curr_price_position] < peaks_volumes[curr_price_position - 1]:
                        peaks_valleys[i] = -1
                    elif price[i] < source[peaks_indexes[curr_price_position] - half_distance_in_element] and \
                        peaks_volumes[curr_price_position] > peaks_volumes[curr_price_position - 1]:
                        peaks_valleys[i] = -1
                    elif price[i] < source[peaks_indexes[curr_price_position] - half_distance_in_element] and \
                        peaks_volumes[curr_price_position] < peaks_volumes[curr_price_position - 1]:
                        peaks_valleys[i] = 1
                    else:
                        peaks_valleys[i] = 0
                except Exception as ex:
                    peaks_valleys[i] = 0
                    print(f'Issue, i = {i}, curr_price_position = {curr_price_position}, half_distance_in_element = {half_distance_in_element}')
                    print(f'Exception equal to {ex}')

    return peaks_valleys


def get_daily_high_and_low_by_date(data: pd.DataFrame):

    '''
    This function returns highs / lows for a given date in an incremental manner for every
    progression of Prices in a single date (not in a single session !)
    '''

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    dates    = data['Date'].astype(str).unique()
    highs    = list()
    lows     = list()

    for date in dates:

        single_date = data[ data['Date'] == date ]
        highs.append( np.maximum.accumulate(single_date['Price'], axis=0) )
        lows.append(  np.minimum.accumulate(single_date['Price'], axis=0) )

    highs = np.hstack(highs)
    lows  = np.hstack(lows)

    return lows, highs


def get_daily_high_and_low_by_session(data: pd.DataFrame):

    if 'SessionType' not in data.columns:
        raise SessionTypeAbsent('No SessionType column present into the DataFrame passed. Execution stops.')

    price       = np.array(data.Price)
    session     = np.array(data.SessionType)
    len_        = len(price)
    lows        = np.zeros(len_)
    highs       = np.zeros(len_)
    current_low = current_high = price[0]

    for i in tqdm(range(1, len_)):

        ########################################################################################################
        if (session[i] != session[i - 1]) & session[i].endswith('ETH') & session[i - 1].endswith('RTH'):
            current_low = current_high = price[i]
        ########################################################################################################

        if price[i] > current_high:
            current_high = price[i]
        highs[i] = current_high

        if price[i] < current_low:
            current_low = price[i]
        lows[i] = current_low

    return lows, highs


