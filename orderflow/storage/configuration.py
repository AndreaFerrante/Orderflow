# ------------------ PostgreSQL ----------------------
PG_USER = 'remora_user'
PG_PWD = 'remora'
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
