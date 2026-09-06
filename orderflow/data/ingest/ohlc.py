'''
Overalll function to manage OHLC SC data for trend and mean reversion analysis, mainly.
'''

import os
import datetime
import polars as pl

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

def trim_df_columns_polars(df: pl.DataFrame) -> pl.DataFrame:

    if not isinstance(df, pl.DataFrame):
        raise ValueError("Parameter named df, must be a DataFrame from the Polars package.")

    trimmed_cols = [col.strip() for col in df.columns]
    df.columns   = trimmed_cols

    # for col_name in df.columns:
    #     if df.schema[col_name] == pl.Utf8:
    #         df = df.with_column(pl.col(col_name).str.strip().alias(col_name))

    return df

def read_and_clean_all_files_polars(path_to_read_files: str,
                                    separator:str    = ',',
                                    file_ext:str     = '.txt',
                                    date_format:str  = '%Y/%m/%d',
                                    schema           = {
                                        "Date": pl.Utf8,  # or pl.Date
                                        "Time": pl.Utf8,  # or plf.Time
                                        "Open": pl.Float64,
                                        "High": pl.Float64,
                                        "Low": pl.Float64,
                                        "Last": pl.Float64,
                                        "Volume": pl.Int64,
                                        "NumberOfTrades": pl.Int64,
                                        "BidVolume": pl.Int64,
                                        "AskVolume": pl.Int64
                                    }
                                    ) -> pl.DataFrame:

    if not isinstance(path_to_read_files, str):
        raise ValueError("Please, pass to the function parameter 'path_to_read_files' as a string object.")

    files       = os.listdir(path_to_read_files)
    stacked_dfs = list()

    for file_name in files:

        if file_name.endswith(file_ext):

            full_path = os.path.join(path_to_read_files, file_name)

            df = pl.read_csv( full_path, has_header=True, separator=separator, schema=schema)
            df = trim_df_columns_polars(df)
            df = df.with_columns(pl.col("Date").str.strptime(pl.Date, date_format, strict=False))

            max_date         = df.select(pl.col("Date").max()).item()
            last_friday      = get_third_friday_three_months_ago(max_date)
            last_monday      = last_friday + datetime.timedelta(days=3)
            df               = df.filter(pl.col("Date") >= pl.lit(last_monday))
            current_max_date = df.select(pl.col("Date").max()).item()

            print(
                f"For file '{file_name}', last date is {current_max_date}. "
                f"3 Months Friday: {last_friday}"
            )

            stacked_dfs.append(df)

    if not stacked_dfs:
        return pl.DataFrame()
    else:
        stacked = pl.concat(stacked_dfs, how="vertical")

    # Get, in Polars, the column Datetime.
    stacked = stacked.with_columns(
        (
                pl.col("Date").cast(pl.Utf8) + pl.lit(" ") + pl.col("Time").cast(pl.Utf8)
        ).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .alias("Datetime")
    )

    # Override the column Time then sort and return the final dataframe !
    stacked = stacked.with_columns(
        [
            pl.col("Datetime").dt.strftime("%H:%M:%S").alias("Time"),
            pl.col("Datetime").dt.weekday().alias("Weekday")
        ]
    )

    stacked = stacked.sort(["Datetime"])

    return stacked
