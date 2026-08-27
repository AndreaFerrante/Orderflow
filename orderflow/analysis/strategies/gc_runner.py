"""CLI for bounded-memory evaluation of the GC auction-acceptance strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from orderflow.analysis.strategies.gc_auction_acceptance import (
    GCAuctionAcceptanceConfig, GCContractSpec, evaluate_gc_auction_acceptance,
)


SCHEMA = {
    "Date": pl.String, "Time": pl.String, "Sequence": pl.Int64,
    "Price": pl.Float64, "Volume": pl.Float64, "TradeType": pl.Int8,
    "AskPrice": pl.Float64, "BidPrice": pl.Float64, "AskSize": pl.Float64,
    "BidSize": pl.Float64, "TotalAskDepth": pl.Float64, "TotalBidDepth": pl.Float64,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Sierra .txt file or directory (searched recursively)")
    parser.add_argument("--output", type=Path, default=Path("gc_auction_acceptance_results"))
    parser.add_argument("--side", choices=["short", "long", "both"], default="short")
    parser.add_argument("--tick-size", type=float, default=0.10)
    parser.add_argument("--tick-value", type=float, default=10.0)
    parser.add_argument("--commission", type=float, default=4.50)
    args = parser.parse_args()

    files = [args.data] if args.data.is_file() else sorted(args.data.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    parts = args.output / "trade_parts"
    parts.mkdir(exist_ok=True)
    if any(parts.glob("*.parquet")):
        raise FileExistsError(f"Refusing to mix a new run with existing results in {parts}")

    config = GCAuctionAcceptanceConfig()
    contract = GCContractSpec(args.tick_size, args.tick_value, args.commission)
    sides = ("long", "short") if args.side == "both" else (args.side,)
    counts = []
    for number, path in enumerate(files):
        print(f"[{number + 1}/{len(files)}] {path.name}", flush=True)
        frame = (
            pl.scan_csv(path, separator=";", schema_overrides=SCHEMA, infer_schema_length=0, low_memory=True)
            .select(list(SCHEMA)).collect(engine="streaming")
        )
        trades = evaluate_gc_auction_acceptance(frame, config, contract, sides)
        counts.append({"file": path.name, "trades": trades.height})
        if not trades.is_empty():
            trades.with_columns(pl.lit(path.name).alias("file")).write_parquet(parts / f"part_{number:05d}.parquet")

    pl.DataFrame(counts).write_csv(args.output / "counts_by_file.csv")
    if any(parts.glob("*.parquet")):
        summary = (
            pl.scan_parquet(parts / "*.parquet").group_by("setup").agg(
                pl.len().alias("trades"), pl.col("net_ticks").mean().alias("mean_net_ticks"),
                pl.col("net_ticks").median().alias("median_net_ticks"),
                pl.col("net_ticks").clip(-5, 5).mean().alias("capped5_mean_ticks"),
                (pl.col("net_ticks") > 0).mean().alias("win_rate"),
                pl.col("net_ticks").sum().alias("total_net_ticks"), pl.col("file").n_unique().alias("active_files"),
            ).collect(engine="streaming")
        )
        summary.write_csv(args.output / "summary.csv")
        print(summary.write_csv())
    (args.output / "manifest.json").write_text(json.dumps({
        "files": len(files), "sides": sides, "strategy": config.to_dict(),
        "contract": asdict(contract), "engine": {"polars": pl.__version__, "numba": "enabled"},
    }, indent=2), encoding="utf-8")
