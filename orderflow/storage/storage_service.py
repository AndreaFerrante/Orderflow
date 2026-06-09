"""
PostgreSQL storage service for the Orderflow project.

Provides two load paths and two query paths:

Load paths:
    load_raw_txt_to_table   — reads Sierra Chart tick txt files, stages into
                              futures.staging_ticks_raw, then upserts into
                              futures.ticks_raw via the DB function.
    load_enriched_parquet_to_table — reads monthly enriched parquet files,
                              stages into futures.staging_ticks_enriched, then
                              upserts into futures.ticks_enriched.

Query paths:
    get_ticks_raw           — fetch raw ticks by month_key or date range.
    get_ticks_enriched      — fetch enriched ticks by month_key or date range.

Usage example:
    from orderflow.storage.storage_service import (
        load_raw_txt_to_table,
        load_enriched_parquet_to_table,
        get_ticks_raw,
        get_ticks_enriched,
    )

    result = load_raw_txt_to_table("/data/ES/ticks/", symbol="ES")
    df = get_ticks_raw("ES", month_key="202501")
"""
from __future__ import annotations

import gc
import io
import logging
import logging.handlers
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import psycopg2
import polars as pl

from orderflow.storage.configuration import (
    SCHEMA,
    STAGING_RAW_TABLE,
    STAGING_ENRICHED_TABLE,
    UPSERT_RAW_FN,
    UPSERT_ENRICHED_FN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORAGE_DIR = Path(__file__).resolve().parent
SQL_DIR = STORAGE_DIR / "sql"
LOG_DIR = STORAGE_DIR / "log"
DEFAULT_MARKET = "CME"
BULK_CHUNK_SIZE = 50_000

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

RAW_RENAME_MAP: dict[str, str] = {
    "Date": "date",
    "Time": "time",
    "Sequence": "sequence",
    "DepthSequence": "depth_sequence",
    "Price": "price",
    "Volume": "volume",
    "TradeType": "trade_type",
    "AskPrice": "ask_price",
    "BidPrice": "bid_price",
    "AskSize": "ask_size",
    "BidSize": "bid_size",
    "TotalAskDepth": "total_ask_depth",
    "TotalBidDepth": "total_bid_depth",
    "AskDOMPrice": "ask_dom_price",
    "BidDOMPrice": "bid_dom_price",
    **{f"AskDOM_{i}": f"ask_dom_{i}" for i in range(30)},
    **{f"BidDOM_{i}": f"bid_dom_{i}" for i in range(30)},
}

STAGING_RAW_COLUMNS: list[str] = [
    "symbol",
    "month_key",
    "date",
    "time",
    "sequence",
    "depth_sequence",
    "price",
    "volume",
    "trade_type",
    "ask_price",
    "bid_price",
    "ask_size",
    "bid_size",
    "total_ask_depth",
    "total_bid_depth",
    "ask_dom_price",
    "bid_dom_price",
    *[f"ask_dom_{i}" for i in range(30)],
    *[f"bid_dom_{i}" for i in range(30)],
]

STAGING_ENRICHED_COLUMNS: list[str] = [
    "symbol",
    "month_key",
    "Date",
    "Time",
    "Sequence",
    "DepthSequence",
    "Price",
    "Volume",
    "TradeType",
    "AskPrice",
    "BidPrice",
    "AskSize",
    "BidSize",
    "TotalAskDepth",
    "TotalBidDepth",
    "AskDOMPrice",
    "BidDOMPrice",
    *[f"AskDOM_{i}" for i in range(30)],
    *[f"BidDOM_{i}" for i in range(30)],
    "Datetime",
    "Hour",
    "SessionType",
    "POC",
    "Prev_POC",
    "VA_Areas",
    "ValleysPeaks",
    "CD_Ask",
    "CD_Bid",
    "CD_Total",
    "Session_High",
    "Session_Low",
    "Node_Volume",
    "Node_Ask_Volume",
    "Node_Bid_Volume",
    "Session_Volume",
    "LVN",
    "Index",
    "current_bar_open",
    "current_bar_high",
    "current_bar_low",
    "current_bar_close",
    "current_bar_volume",
    "current_bar_numberoftrades",
    "current_bar_askvolume",
    "current_bar_bidvolume",
    "current_bar_datetime",
    "next_bar_datetime",
    "next_bar_open",
    "next_bar_high",
    "next_bar_low",
    "next_bar_close",
    "next_bar_volume",
    "next_bar_num_trades",
    "next_bar_ask_volume",
    "next_bar_bid_volume",
    "vwap",
    "vwap_sd1_top",
    "vwap_sd1_bottom",
    "vwap_sd2_top",
    "vwap_sd2_bottom",
    "vwap_sd3_top",
    "vwap_sd3_bottom",
    "vwap_sd4_top",
    "vwap_sd4_bottom",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger: Optional[logging.Logger] = None


def _configure_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("orderflow.storage")
    if logger.handlers:
        _logger = logger
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "storage_service.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)

    _logger = logger
    return logger


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def _connect() -> psycopg2.extensions.connection:
    from orderflow.storage.configuration import (  # noqa: PLC0415
        PG_USER, PG_PWD, PG_HOST, PG_PORT, PG_DATABASE,
    )
    return psycopg2.connect(
        user=PG_USER,
        password=PG_PWD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


# ---------------------------------------------------------------------------
# SQL loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Core load routine
# ---------------------------------------------------------------------------

def _run_load(
    staging_table: str,
    upsert_fn: str,
    df: pl.DataFrame,
    columns: list[str],
) -> dict:
    quoted_cols = ", ".join(f'"{c}"' for c in columns)
    copy_sql = (
        f"COPY {SCHEMA}.\"{staging_table}\" ({quoted_cols}) "
        "FROM STDIN WITH (FORMAT CSV, DELIMITER '\t', NULL '\\N')"
    )
    conn = _connect()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute(f'TRUNCATE {SCHEMA}."{staging_table}"')
            for chunk in df.select(columns).iter_slices(n_rows=BULK_CHUNK_SIZE):
                cur.copy_expert(
                    copy_sql,
                    io.StringIO(chunk.write_csv(separator="\t", null_value="\\N", include_header=False)),
                )
            cur.execute(f'SELECT {SCHEMA}."{upsert_fn}"()')
            rows_upserted = cur.fetchone()[0]
    finally:
        conn.close()
    return {"rows_staged": len(df), "rows_upserted": rows_upserted}


# ---------------------------------------------------------------------------
# Public API — load
# ---------------------------------------------------------------------------

def load_raw_txt_to_table(folder: str | Path, symbol: str) -> dict:
    """Load Sierra Chart tick txt files into futuress.ticks_raw.

    Args:
        folder: Directory containing the raw txt tick files.
        symbol: Instrument symbol, e.g. ``"ES"``.

    Returns:
        Summary dict with keys: symbol, month_keys, rows_staged, rows_upserted.
    """
    from orderflow._volume_factory import get_tickers_in_folder_mem_optim

    logger = _configure_logger()
    symbol = symbol.upper()
    folder = Path(folder)

    logger.info("Loading raw txt for symbol=%s from %s", symbol, folder)

    df: pl.DataFrame = get_tickers_in_folder_mem_optim(
        path=str(folder) + "/",
        ticker=symbol,
        extension=".txt",
        separator=";",
        market=DEFAULT_MARKET,
    )

    existing_cols = set(df.columns)
    rename_map = {k: v for k, v in RAW_RENAME_MAP.items() if k in existing_cols}
    df = df.rename(rename_map)

    if "Datetime" in df.columns:
        df = df.drop("Datetime")

    df = df.with_columns(
        pl.col("date").str.slice(0, 7).str.replace("-", "").alias("month_key"),
        pl.lit(symbol).alias("symbol"),
    )

    missing = [pl.lit(None).alias(col) for col in STAGING_RAW_COLUMNS if col not in df.columns]
    if missing:
        df = df.with_columns(missing)

    df = df.select(STAGING_RAW_COLUMNS)

    month_keys = sorted(df["month_key"].unique().to_list())
    logger.info("Found month_keys: %s — %d rows total", month_keys, len(df))

    result = _run_load(STAGING_RAW_TABLE, UPSERT_RAW_FN, df, STAGING_RAW_COLUMNS)

    summary = {
        "symbol": symbol,
        "month_keys": month_keys,
        "rows_staged": result["rows_staged"],
        "rows_upserted": result["rows_upserted"],
    }
    logger.info("Load complete: %s", summary)
    return summary


def load_enriched_parquet_to_table(
    folder: str | Path,
    symbol: Optional[str] = None,
) -> dict:
    """Load monthly enriched parquet files into futuress.ticks_enriched.

    Processes one file at a time to stay within the 32 GB memory budget.
    Supported filename patterns:
      - ``YYYYMM_SYMBOL.parquet``
      - ``YYYYMMDD_to_YYYYMMDD_SYMBOL.parquet``

    Args:
        folder: Directory containing enriched parquet files.
        symbol: Fallback symbol if it cannot be inferred from the filename.

    Returns:
        Aggregate summary dict with keys: files_processed, total_rows_staged,
        total_rows_upserted, errors.
    """
    logger = _configure_logger()
    folder = Path(folder)

    pattern_month = re.compile(r"^(\d{6})_([A-Za-z0-9]+)$")
    pattern_range = re.compile(r"^(\d{8})_to_(\d{8})_([A-Za-z0-9]+)$")

    total_staged = 0
    total_upserted = 0
    files_processed = 0
    errors: list[str] = []

    for path in sorted(folder.glob("*.parquet")):
        stem = path.stem
        file_symbol: Optional[str] = None

        m = pattern_month.match(stem)
        if m:
            file_symbol = m.group(2).upper()

        if file_symbol is None:
            m = pattern_range.match(stem)
            if m:
                file_symbol = m.group(3).upper()

        if file_symbol is None:
            if symbol is not None:
                file_symbol = symbol.upper()
            else:
                msg = f"Cannot determine symbol from filename '{path.name}' and no fallback provided."
                logger.error(msg)
                errors.append(msg)
                continue

        logger.info("Processing %s (symbol=%s)", path.name, file_symbol)

        df = None
        try:
            df = pl.read_parquet(path)

            if "Datetime" in df.columns and df["Datetime"].dtype in (
                pl.Datetime,
                pl.Datetime("us"),
                pl.Datetime("ns"),
                pl.Datetime("ms"),
            ):
                month_key_expr = pl.col("Datetime").dt.strftime("%Y%m").alias("month_key")
            elif "Date" in df.columns:
                month_key_expr = (
                    pl.col("Date").cast(pl.Utf8).str.slice(0, 7).str.replace("-", "").alias("month_key")
                )
            else:
                raise ValueError(f"File {path.name} has neither 'Datetime' nor 'Date' column.")

            df = df.with_columns(
                month_key_expr,
                pl.lit(file_symbol).alias("symbol"),
            )

            missing = [pl.lit(None).alias(col) for col in STAGING_ENRICHED_COLUMNS if col not in df.columns]
            if missing:
                df = df.with_columns(missing)

            df = df.select(STAGING_ENRICHED_COLUMNS)

            result = _run_load(
                STAGING_ENRICHED_TABLE,
                UPSERT_ENRICHED_FN,
                df,
                STAGING_ENRICHED_COLUMNS,
            )

            total_staged += result["rows_staged"]
            total_upserted += result["rows_upserted"]
            files_processed += 1
            logger.info(
                "File %s done: staged=%d upserted=%d",
                path.name,
                result["rows_staged"],
                result["rows_upserted"],
            )

        except Exception as exc:
            msg = f"Failed to process {path.name}: {exc}"
            logger.exception(msg)
            errors.append(msg)
        finally:
            if df is not None:
                del df
            gc.collect()

    summary = {
        "files_processed": files_processed,
        "total_rows_staged": total_staged,
        "total_rows_upserted": total_upserted,
        "errors": errors,
    }
    logger.info("Enriched load complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Public API — query
# ---------------------------------------------------------------------------

def get_ticks_raw(
    symbol: str,
    month_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.DataFrame:
    """Fetch raw ticks from futuress.ticks_raw.

    Exactly one of ``month_key`` or the pair ``(start_date, end_date)``
    must be provided.

    Args:
        symbol: Instrument symbol, e.g. ``"ES"``.
        month_key: Six-digit month string, e.g. ``"202501"``.
        start_date: ISO date string ``"YYYY-MM-DD"`` (inclusive).
        end_date: ISO date string ``"YYYY-MM-DD"`` (inclusive).

    Returns:
        Polars DataFrame with all ticks_raw columns.

    Raises:
        ValueError: If the argument combination is invalid.
    """
    if month_key is not None and (start_date is not None or end_date is not None):
        raise ValueError("Provide either month_key OR (start_date, end_date), not both.")
    if month_key is None and (start_date is None or end_date is None):
        raise ValueError("Provide either month_key or both start_date and end_date.")

    if month_key is not None:
        sql = _load_sql("select_ticks_raw_by_month.sql")
        params = {"symbol": symbol, "month_key": month_key}
    else:
        sql = _load_sql("select_ticks_raw_by_date_range.sql")
        params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if not rows:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        return pl.DataFrame(rows, schema=cols, orient="row")
    finally:
        conn.close()


def get_ticks_enriched(
    symbol: str,
    month_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.DataFrame:
    """Fetch enriched ticks from futuress.ticks_enriched.

    Exactly one of ``month_key`` or the pair ``(start_date, end_date)``
    must be provided.

    Args:
        symbol: Instrument symbol, e.g. ``"ES"``.
        month_key: Six-digit month string, e.g. ``"202501"``.
        start_date: ISO date string ``"YYYY-MM-DD"`` (inclusive).
        end_date: ISO date string ``"YYYY-MM-DD"`` (inclusive).

    Returns:
        Polars DataFrame with all ticks_enriched columns.

    Raises:
        ValueError: If the argument combination is invalid.
    """
    if month_key is not None and (start_date is not None or end_date is not None):
        raise ValueError("Provide either month_key OR (start_date, end_date), not both.")
    if month_key is None and (start_date is None or end_date is None):
        raise ValueError("Provide either month_key or both start_date and end_date.")

    if month_key is not None:
        sql = _load_sql("select_ticks_enriched_by_month.sql")
        params = {"symbol": symbol, "month_key": month_key}
    else:
        sql = _load_sql("select_ticks_enriched_by_date_range.sql")
        params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if not rows:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        return pl.DataFrame(rows, schema=cols, orient="row")
    finally:
        conn.close()
