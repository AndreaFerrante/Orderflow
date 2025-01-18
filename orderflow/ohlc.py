'''
Overalll function to manage OHLC SC data for trend and mean reversion analysis, mainly.
'''

import os
import datetime
import pandas as pd


def get_third_friday_three_months_ago(ref_date:datetime.date):

    """
    Returns a datetime.date object representing the third Friday
    from the month that was three months ago (relative to today).
    """

    if not isinstance(ref_date, datetime.date):
        raise Exception(f"Attention, pas parameter ref_date like a 'datetime.date' object.")

    target_year  = ref_date.year
    target_month = ref_date.month - 3

    if target_month < 1:
        target_month += 12
        target_year -= 1

    first_day_of_month = datetime.date(target_year, target_month, 1)
    first_day_weekday  = first_day_of_month.weekday()

    offset_to_first_friday = (4 - first_day_weekday) % 7
    first_friday = first_day_of_month + datetime.timedelta(days=offset_to_first_friday)

    third_friday = first_friday + datetime.timedelta(days=14)

    return third_friday

def trim_df_columns(df:pd.DataFrame):

    if not isinstance(df, pd.DataFrame):
        raise Exception("Paramter named df, must be a DataFrame.")

    columns = [str(x).strip() for x in df.columns]
    df.columns = columns

    return df

def read_and_clean_all_files(path_to_read_files:str):

    if not isinstance(path_to_read_files, str):
        raise Exception("Please, pass to the function parameter 'path_to_read_files' as a string object.")

    files   = os.listdir(path_to_read_files)
    stacked = list()

    for file in files:

        if file.endswith('txt'):

            single_file = pd.read_csv(os.path.join(path_to_read_files, file))
            single_file = trim_df_columns(single_file)
            single_file = single_file.assign(Date = pd.to_datetime(single_file.Date))
            last_friday = get_third_friday_three_months_ago(single_file.Date.max())
            last_monday = last_friday + datetime.timedelta(days=3)
            single_file = single_file[single_file['Date'] >= pd.to_datetime(last_monday)]
            stacked.append(single_file)
            print(f'For file named {file}, last date is {single_file.Date.max()}. 3 Months Friday: {last_friday}')

    stacked = pd.concat(stacked)
    stacked = stacked.sort_values('Date', ascending=True)
    stacked = stacked.reset_index(drop=True, inplace=False)

    return stacked
