"""CLI: load raw tick txt files into futures.ticks_raw via staging.

Usage::

    python load_ticks_raw_cli.py --folder "C:/data/ES/unarchive/" --symbol ES

    # from project root (editable install)
    python -m orderflow.storage.load_ticks_raw_cli --folder "C:/data/ES/unarchive/" --symbol ES
"""
from __future__ import annotations

import argparse
import json

from orderflow.storage.storage_service import load_raw_txt_to_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Load raw tick txt files to PostgreSQL.")
    parser.add_argument("--folder", required=True, help="Folder containing txt files")
    parser.add_argument("--symbol", required=True, help="Instrument symbol, e.g. ES")
    args = parser.parse_args()
    result = load_raw_txt_to_table(folder=args.folder, symbol=args.symbol)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
