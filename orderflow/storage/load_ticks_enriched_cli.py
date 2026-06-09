"""CLI: load enriched tick parquet files into futures.ticks_enriched via staging.

Usage::

    # symbol inferred from filename (e.g. 202501_ES.parquet)
    python load_ticks_enriched_cli.py --folder "C:/data/ES/parquet/"

    # explicit symbol fallback for non-standard filenames
    python load_ticks_enriched_cli.py --folder "C:/data/ES/parquet/" --symbol ES

    # from project root (editable install)
    python -m orderflow.storage.load_ticks_enriched_cli --folder "C:/data/ES/parquet/"
"""
from __future__ import annotations

import argparse
import json

from orderflow.storage.storage_service import load_enriched_parquet_to_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Load enriched tick parquet files to PostgreSQL.")
    parser.add_argument("--folder", required=True, help="Folder containing parquet files")
    parser.add_argument(
        "--symbol",
        required=False,
        default=None,
        help="Fallback instrument symbol if not inferrable from filename, e.g. ES",
    )
    args = parser.parse_args()
    result = load_enriched_parquet_to_table(folder=args.folder, symbol=args.symbol)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
