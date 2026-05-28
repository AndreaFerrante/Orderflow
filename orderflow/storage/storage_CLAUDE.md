# CLAUDE.md — orderflow/storage

> **Scope boundary:** This folder is a self-contained service module.
> When working here, do not read, reference, or import logic from other
> `orderflow/` submodules (backtester, compressor, stats, volume_profile, etc.).
> **Exception:** `load_raw_txt_to_table` imports `orderflow._volume_factory` to
> parse Sierra Chart txt files — this is the only permitted cross-module import.
> Everything needed to implement storage connectors is defined in this file.
> For project-level conventions (Python version, Polars/Pandas policy, commit style),
> refer to the root `CLAUDE.md`.

---

## Purpose

Persist and retrieve trading data from PostgreSQL. Two load paths, two query paths.
No trading strategy logic. No signal generation. No backtest logic.
Module knows data shapes (DataFrames, Parquet files) — nothing more.

---

## Module Structure

```
orderflow/storage/
    __init__.py                              ← empty (no public re-exports)
    configuration.py                         ← DB credentials + table/schema/function name constants
    storage_service.py                       ← all load and query logic
    load_ticks_raw_cli.py                    ← CLI: txt files → ticks_raw
    load_ticks_enriched_cli.py               ← CLI: parquet files → ticks_enriched
    log/                                     ← rotating log files
    sql/
        create_table_ticks_raw.sql           ← DDL: ticks_raw + partitions
        create_table_ticks_enriched.sql      ← DDL: ticks_enriched + partitions
        create_table_staging_ticks_raw.sql   ← DDL: staging_ticks_raw
        create_table_staging_ticks_enriched.sql
        create_function_upsert_ticks_raw.sql ← PG function: stage → upsert
        create_function_upsert_ticks_enriched.sql
        select_ticks_raw_by_month.sql
        select_ticks_raw_by_date_range.sql
        select_ticks_enriched_by_month.sql
        select_ticks_enriched_by_date_range.sql
```

---

## One-Time Setup

Run DDL scripts once against the PostgreSQL instance before any loads:

```bash
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_table_ticks_raw.sql
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_table_ticks_enriched.sql
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_table_staging_ticks_raw.sql
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_table_staging_ticks_enriched.sql
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_function_upsert_ticks_raw.sql
psql -U remora_user -h 100.87.5.16 -f orderflow/storage/sql/create_function_upsert_ticks_enriched.sql
```

---

## Database Design

### Connection & Configuration

All configuration lives in `configuration.py`:

```python
# Credentials
PG_USER = 'remora_user'
PG_PWD  = 'remora'
PG_HOST = '100.87.5.16'
PG_PORT = '5432'
PG_DATABASE = 'remora'

# Schema
SCHEMA = 'futures'

# Table names
TICKS_RAW_TABLE = 'ticks_raw'
TICKS_ENRICHED_TABLE = 'ticks_enriched'
STAGING_RAW_TABLE = 'staging_ticks_raw'
STAGING_ENRICHED_TABLE = 'staging_ticks_enriched'

# PostgreSQL upsert function names
UPSERT_RAW_FN = 'upsert_ticks_raw'
UPSERT_ENRICHED_FN = 'upsert_ticks_enriched'
```

`storage_service.py` imports `SCHEMA`, `STAGING_RAW_TABLE`, `STAGING_ENRICHED_TABLE`,
`UPSERT_RAW_FN`, `UPSERT_ENRICHED_FN` at module level. `PG_*` credentials are imported
lazily inside `_connect()` (psycopg2 connection) and `_connection_uri()` (ADBC URI string).

### Schema: `futures`

All tables live in the `futures` schema.

### ticks_raw

Raw Sierra Chart tick data. PK: `(symbol, month_key, date, time, sequence)`.

Partition hierarchy: `PARTITION BY LIST (symbol)` → `PARTITION BY RANGE (month_key)`.

Partition naming: `futures.ticks_raw_es` → `futures.ticks_raw_es_202501`, etc.
Pre-created partitions: ES, 202501–202612.

Key columns: `symbol`, `month_key`, `date`, `time`, `sequence`, `depth_sequence`,
`price`, `volume`, `trade_type`, `ask_price`, `bid_price`, `ask_size`, `bid_size`,
`total_ask_depth`, `total_bid_depth`, `ask_dom_price`, `bid_dom_price`,
`ask_dom_0..29`, `bid_dom_0..29`, `created_at`, `updated_at`.

### ticks_enriched

Enriched tick data with all indicators. PK: `(symbol, month_key, "Date", "Time", "Sequence")`.
Note: enriched columns use PascalCase (preserved from parquet source).

Same partition structure as ticks_raw:
`futures.ticks_enriched_es` → `futures.ticks_enriched_es_202501`, etc.

Columns beyond raw: `Datetime`, `Hour`, `SessionType`, `POC`, `Prev_POC`,
`VA_Areas`, `ValleysPeaks`, `CD_Ask`, `CD_Bid`, `CD_Total`, `Session_High`,
`Session_Low`, `Node_Volume`, `Node_Ask_Volume`, `Node_Bid_Volume`, `Session_Volume`,
`LVN`, `Index`, `current_bar_*` (OHLCV + askvolume/bidvolume/datetime),
`next_bar_*` (OHLCV + num_trades/ask_volume/bid_volume/datetime),
`vwap`, `vwap_sd1_top/bottom` through `vwap_sd4_top/bottom`.

### Staging Tables

`futures.staging_ticks_raw` and `futures.staging_ticks_enriched` — same column
sets as their target tables, no PK. Used as load buffer: TRUNCATE → bulk INSERT → upsert.

### Load Flow

Single psycopg2 connection, single transaction in `_run_load`:

```
with conn:
    TRUNCATE futures."staging_table"
    → COPY FROM STDIN (tab-separated CSV, chunked at 50,000 rows)
    → SELECT futures."upsert_fn"()
       INSERT INTO target SELECT FROM staging ON CONFLICT DO UPDATE
       Returns row count.
```

Atomic: all three steps commit together or roll back together.

---

## Public API

All public functions are in `storage_service.py`.

```python
def load_raw_txt_to_table(folder: str | Path, symbol: str) -> dict:
    """
    Load Sierra Chart tick txt files into futures.ticks_raw.
    Uses _volume_factory.get_tickers_in_folder_mem_optim to parse files.
    Returns: {"symbol", "month_keys", "rows_staged", "rows_upserted"}
    """

def load_enriched_parquet_to_table(
    folder: str | Path,
    symbol: str | None = None,
) -> dict:
    """
    Load monthly enriched parquet files into futures.ticks_enriched.
    Processes one file at a time (32 GB memory budget — gc.collect() after each).
    Filename patterns supported:
        YYYYMM_SYMBOL.parquet
        YYYYMMDD_to_YYYYMMDD_SYMBOL.parquet
    Symbol fallback via `symbol` arg if not inferrable from filename.
    Returns: {"files_processed", "total_rows_staged", "total_rows_upserted", "errors"}
    """

def get_ticks_raw(
    symbol: str,
    month_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Fetch raw ticks from futures.ticks_raw.
    Provide either month_key OR (start_date + end_date) — not both, not neither.
    Returns Polars DataFrame.
    """

def get_ticks_enriched(
    symbol: str,
    month_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Fetch enriched ticks from futures.ticks_enriched.
    Same argument rules as get_ticks_raw.
    Returns Polars DataFrame.
    """
```

---

## CLI Entry Points

```bash
# Raw tick txt files → futures.ticks_raw
python -m orderflow.storage.load_ticks_raw_cli --folder "sources/ES/unarchive/" --symbol ES

# Enriched parquet files → futures.ticks_enriched (symbol inferred from filename)
python -m orderflow.storage.load_ticks_enriched_cli --folder "sources/ES/parquet/"

# With explicit symbol fallback
python -m orderflow.storage.load_ticks_enriched_cli --folder "sources/ES/parquet/" --symbol ES
```

---

## Implementation Rules

**psycopg2 only:** connection built directly in `_connect()`. SQLAlchemy is a declared
dependency but not used in the data path.

**Polars for I/O:** Polars reads parquet files and receives query results.
No Pandas in the data path.

**SQL files:** all queries loaded from `sql/` via `_load_sql(name)` (LRU-cached).
No inline SQL strings in Python code.

**Idempotent writes:** TRUNCATE staging → ADBC insert → upsert. Safe to rerun.

**Memory management:** `load_enriched_parquet_to_table` processes one file at a time,
deletes the DataFrame, and calls `gc.collect()` after each file. Do not accumulate
DataFrames across loop iterations.

**Staging insert uses COPY:** `_run_load` streams data via `cursor.copy_expert` with
tab-separated CSV, chunked at 50,000 rows. PostgreSQL handles all type coercion from
text — robust for wide mixed-type tables. Do not revert to `execute_values`.
ADBC was attempted but rejected: Arrow `null` dtype (from `pl.lit(None)` missing columns)
has 0 buffers; nanoarrow crashes when the PostgreSQL schema expects a typed column.

**No business logic:** this module does not know what `TradeType`, `market_state`,
`Prev_POC`, or `CVD` mean. It moves bytes from DataFrames to tables and back.

**Schema evolution:** add new columns to the relevant `create_table_*.sql` with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Add the column name to the relevant
`STAGING_*_COLUMNS` list in `storage_service.py`.

**Error handling:** failed files in `load_enriched_parquet_to_table` are logged and
appended to the `errors` list — load continues. Caller checks `result["errors"]`.

---

## Dependencies

```toml
[project.optional-dependencies]
storage = [
    "sqlalchemy>=2.0",
    "psycopg2-binary",
]
```

Install with: `pip install -e ".[storage]"`
