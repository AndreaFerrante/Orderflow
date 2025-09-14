import polars as pl

def is_skewed(series: pl.Series, threshold: float = 0.5) -> bool:

    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    if series.dtype not in numeric_dtypes:
        raise ValueError("The input series must be numeric.")

    clean_series = series.drop_nulls()
    if len(clean_series) == 0:
        raise ValueError("The input series is empty after dropping missing values.")

    # Calculate skewness using the built-in Polars method Andrea !
    skew_val = clean_series.skew()

    return abs(skew_val) > threshold

def get_kurtosis(series: pl.Series):

    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    if series.dtype not in numeric_dtypes:
        raise ValueError("The input series must be numeric.")

    clean_series = series.drop_nulls()
    if len(clean_series) == 0:
        raise ValueError("The input series is empty after dropping missing values.")

    # Calculate skewness using the built-in Polars method Andrea !
    kurtosis_val = clean_series.kurtosis()

    return kurtosis_val
