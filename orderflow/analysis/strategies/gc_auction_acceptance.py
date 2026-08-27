"""Causal GC auction-acceptance strategy.

The strategy combines a completed prior-session volume profile, developing
volume outside prior value, trade-price auctions, and a completed footprint.
It is intentionally implemented with Polars transformations and Numba kernels
so weekly Sierra Chart files can be evaluated without loading a full year.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numba as nb
import numpy as np
import polars as pl


@dataclass(frozen=True)
class GCContractSpec:
    tick_size: float = 0.10
    tick_value: float = 10.0
    round_turn_commission: float = 4.50

    @property
    def commission_ticks(self) -> float:
        return self.round_turn_commission / self.tick_value


@dataclass(frozen=True)
class GCAuctionAcceptanceConfig:
    value_area_fraction: float = 0.70
    outside_value_share: float = 0.15
    minimum_auction_trades: int = 20
    executed_imbalance: float = 0.80
    footprint_ms: int = 1_000
    footprint_ratio: float = 3.0
    maximum_footprint_age_ms: int = 2_000
    holding_ms: int = 1_000
    session_gap_minutes: int = 30

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _timestamp_ns_expression() -> pl.Expr:
    parts = pl.col("Time").str.split_exact(".", 1)
    whole = pl.concat_str([pl.col("Date"), pl.lit(" "), parts.struct.field("field_0")]).str.strptime(
        pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=True
    )
    micros = (
        parts.struct.field("field_1").fill_null("").str.slice(0, 6)
        .str.pad_start(6, "0").cast(pl.Int64)
    )
    return (whole.cast(pl.Int64) * 1_000 + micros * 1_000).alias("timestamp_ns")


def prepare_sierra_ticks(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and chronologically order the common Sierra trade/BBO fields."""
    required = {
        "Date", "Time", "Sequence", "Price", "Volume", "TradeType",
        "AskPrice", "BidPrice", "AskSize", "BidSize",
        "TotalAskDepth", "TotalBidDepth",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Sierra columns: {sorted(missing)}")
    if "timestamp_ns" not in frame.columns:
        frame = frame.with_columns(_timestamp_ns_expression())
    if "row_number" in frame.columns:
        frame = frame.drop("row_number")
    return frame.sort("timestamp_ns", "Sequence").with_row_index("row_number")


def _add_sessions(frame: pl.DataFrame, gap_minutes: int) -> pl.DataFrame:
    boundary = pl.col("timestamp_ns").diff().gt(gap_minutes * 60 * 1_000_000_000).fill_null(True)
    return frame.with_columns((boundary.cast(pl.Int64).cum_sum() - 1).alias("session_id"))


@nb.njit(cache=True)
def _value_area_by_session(
    groups: np.ndarray, ticks: np.ndarray, volume: np.ndarray, value_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = groups.size
    poc = np.empty(n, np.int64)
    val = np.empty(n, np.int64)
    vah = np.empty(n, np.int64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and groups[end] == groups[start]:
            end += 1
        poc_i = start
        total = 0.0
        for i in range(start, end):
            total += volume[i]
            if volume[i] > volume[poc_i] or (volume[i] == volume[poc_i] and ticks[i] < ticks[poc_i]):
                poc_i = i
        lo = poc_i
        hi = poc_i
        included = volume[poc_i]
        target = total * value_fraction
        while included < target and (lo > start or hi + 1 < end):
            down = volume[lo - 1] if lo > start else -1.0
            up = volume[hi + 1] if hi + 1 < end else -1.0
            if up > down:
                hi += 1
                included += volume[hi]
            else:
                lo -= 1
                included += volume[lo]
        for i in range(start, end):
            poc[i] = ticks[poc_i]
            val[i] = ticks[lo]
            vah[i] = ticks[hi]
        start = end
    return poc, val, vah


def _attach_prior_value(
    frame: pl.DataFrame, tick_size: float, value_fraction: float,
) -> pl.DataFrame:
    work = frame.with_columns((pl.col("Price") / tick_size).round(0).cast(pl.Int64).alias("price_tick"))
    levels = work.group_by("session_id", "price_tick").agg(pl.col("Volume").sum().alias("profile_volume")).sort(
        "session_id", "price_tick"
    )
    poc, val, vah = _value_area_by_session(
        levels["session_id"].to_numpy(), levels["price_tick"].to_numpy(),
        levels["profile_volume"].to_numpy(), value_fraction,
    )
    profiles = (
        levels.with_columns(pl.Series("poc_tick", poc), pl.Series("val_tick", val), pl.Series("vah_tick", vah))
        .group_by("session_id", maintain_order=True).agg(
            pl.col("poc_tick").first(), pl.col("val_tick").first(), pl.col("vah_tick").first()
        )
        .with_columns((pl.col("session_id") + 1).alias("session_id"))
        .rename({"poc_tick": "prior_poc_tick", "val_tick": "prior_val_tick", "vah_tick": "prior_vah_tick"})
    )
    return work.join(profiles, on="session_id", how="left")


def _trade_price_auctions(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame.with_columns(
        (pl.col("Price").ne(pl.col("Price").shift(1)).fill_null(True).cast(pl.Int64).cum_sum() - 1).alias("auction_id"),
        pl.when(pl.col("TradeType") == 2).then(pl.col("Volume")).otherwise(0.0).alias("buy_volume"),
        pl.when(pl.col("TradeType") == 1).then(pl.col("Volume")).otherwise(0.0).alias("sell_volume"),
    )
    return (
        work.group_by("auction_id", maintain_order=True).agg(
            pl.col("row_number").last().alias("end_row"),
            pl.col("timestamp_ns").last().alias("end_timestamp_ns"),
            pl.col("Price").first().alias("auction_price"),
            pl.col("buy_volume").sum(), pl.col("sell_volume").sum(),
            pl.col("Volume").sum().alias("auction_volume"), pl.len().alias("trade_count"),
        )
        .with_columns(
            pl.when(pl.col("buy_volume") + pl.col("sell_volume") > 0)
            .then((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("buy_volume") + pl.col("sell_volume")))
            .otherwise(0.0).alias("executed_imbalance"),
            pl.when(pl.col("auction_price").shift(-1) > pl.col("auction_price")).then(1)
            .when(pl.col("auction_price").shift(-1) < pl.col("auction_price")).then(-1)
            .otherwise(0).cast(pl.Int8).alias("winner"),
            (pl.col("end_row") + 1).alias("decision_row"),
            (pl.col("end_row") + 2).alias("entry_row"),
        )
    )


def _completed_footprints(
    frame: pl.DataFrame, tick_size: float, footprint_ms: int, ratio: float,
) -> pl.DataFrame:
    period_ns = footprint_ms * 1_000_000
    levels = (
        frame.with_columns(
            (pl.col("timestamp_ns") // period_ns).alias("footprint_id"),
            (pl.col("Price") / tick_size).round(0).cast(pl.Int64).alias("price_tick"),
            pl.when(pl.col("TradeType") == 2).then(pl.col("Volume")).otherwise(0.0).alias("ask_volume"),
            pl.when(pl.col("TradeType") == 1).then(pl.col("Volume")).otherwise(0.0).alias("bid_volume"),
        )
        .group_by("footprint_id", "price_tick").agg(pl.col("ask_volume").sum(), pl.col("bid_volume").sum())
        .sort("footprint_id", "price_tick")
    )
    lower_bid = levels.select(
        "footprint_id", (pl.col("price_tick") + 1).alias("price_tick"),
        pl.col("bid_volume").alias("lower_bid_volume"),
    )
    higher_ask = levels.select(
        "footprint_id", (pl.col("price_tick") - 1).alias("price_tick"),
        pl.col("ask_volume").alias("higher_ask_volume"),
    )
    levels = (
        levels.join(lower_bid, on=["footprint_id", "price_tick"], how="left")
        .join(higher_ask, on=["footprint_id", "price_tick"], how="left")
        .with_columns(pl.col("lower_bid_volume").fill_null(0.0), pl.col("higher_ask_volume").fill_null(0.0))
        .with_columns(
            ((pl.col("ask_volume") >= ratio * pl.col("lower_bid_volume"))
             & (pl.col("ask_volume") >= 1.0) & (pl.col("lower_bid_volume") >= 1.0)).alias("ask_diagonal"),
            ((pl.col("bid_volume") >= ratio * pl.col("higher_ask_volume"))
             & (pl.col("bid_volume") >= 1.0) & (pl.col("higher_ask_volume") >= 1.0)).alias("bid_diagonal"),
        )
    )
    return levels.group_by("footprint_id", maintain_order=True).agg(
        pl.col("ask_diagonal").any().alias("has_ask_diagonal"),
        pl.col("bid_diagonal").any().alias("has_bid_diagonal"),
    ).with_columns(((pl.col("footprint_id") + 1) * period_ns).alias("footprint_end_ns"))


def generate_gc_auction_acceptance_signals(
    frame: pl.DataFrame,
    config: GCAuctionAcceptanceConfig = GCAuctionAcceptanceConfig(),
    contract: GCContractSpec = GCContractSpec(),
    sides: Iterable[str] = ("short",),
) -> pl.DataFrame:
    """Generate causal signals; the researched candidate is the short side."""
    requested = set(sides)
    unknown = requested - {"long", "short"}
    if unknown:
        raise ValueError(f"Unknown sides: {sorted(unknown)}")
    ticks = _add_sessions(prepare_sierra_ticks(frame), config.session_gap_minutes)
    contextual = _attach_prior_value(ticks, contract.tick_size, config.value_area_fraction).with_columns(
        pl.col("Volume").cum_sum().over("session_id").alias("developing_total_volume"),
        pl.when(pl.col("price_tick") > pl.col("prior_vah_tick")).then(pl.col("Volume")).otherwise(0.0)
        .cum_sum().over("session_id").alias("developing_volume_above_prior_value"),
        pl.when(pl.col("price_tick") < pl.col("prior_val_tick")).then(pl.col("Volume")).otherwise(0.0)
        .cum_sum().over("session_id").alias("developing_volume_below_prior_value"),
    )
    decision = contextual.select(
        pl.col("row_number").alias("decision_row"), pl.col("timestamp_ns").alias("decision_timestamp_ns"),
        pl.col("price_tick").alias("decision_tick"), "prior_val_tick", "prior_vah_tick",
        "developing_total_volume", "developing_volume_above_prior_value", "developing_volume_below_prior_value",
    )
    events = (
        _trade_price_auctions(ticks).filter(pl.col("winner") != 0)
        .join(decision, on="decision_row", how="inner").sort("decision_timestamp_ns")
        .join_asof(
            _completed_footprints(ticks, contract.tick_size, config.footprint_ms, config.footprint_ratio)
            .sort("footprint_end_ns"),
            left_on="decision_timestamp_ns", right_on="footprint_end_ns", strategy="backward",
        )
        .with_columns(
            ((pl.col("decision_timestamp_ns") - pl.col("footprint_end_ns")) / 1_000_000).alias("footprint_age_ms"),
            (pl.col("developing_volume_above_prior_value") / pl.col("developing_total_volume")).alias("above_share"),
            (pl.col("developing_volume_below_prior_value") / pl.col("developing_total_volume")).alias("below_share"),
        )
        .filter(
            pl.col("prior_val_tick").is_not_null()
            & (pl.col("trade_count") >= config.minimum_auction_trades)
            & (pl.col("footprint_age_ms") <= config.maximum_footprint_age_ms)
        )
    )
    pieces: list[pl.DataFrame] = []
    if "short" in requested:
        pieces.append(events.filter(
            (pl.col("winner") == -1)
            & (pl.col("executed_imbalance") <= -config.executed_imbalance)
            & pl.col("has_bid_diagonal")
            & (pl.col("decision_tick") < pl.col("prior_val_tick"))
            & (pl.col("below_share") >= config.outside_value_share)
        ).with_columns(pl.lit("gc_acceptance_below_short").alias("setup"), pl.lit(-1, dtype=pl.Int8).alias("direction")))
    if "long" in requested:
        pieces.append(events.filter(
            (pl.col("winner") == 1)
            & (pl.col("executed_imbalance") >= config.executed_imbalance)
            & pl.col("has_ask_diagonal")
            & (pl.col("decision_tick") > pl.col("prior_vah_tick"))
            & (pl.col("above_share") >= config.outside_value_share)
        ).with_columns(pl.lit("gc_acceptance_above_long").alias("setup"), pl.lit(1, dtype=pl.Int8).alias("direction")))
    if not pieces:
        return pl.DataFrame(schema={"setup": pl.String, "entry_row": pl.Int64, "direction": pl.Int8})
    return pl.concat(pieces, how="vertical_relaxed").select(
        "setup", "entry_row", "direction", "decision_timestamp_ns", "decision_tick",
        "prior_val_tick", "prior_vah_tick", "below_share", "above_share",
        "trade_count", "auction_volume", "executed_imbalance", "winner",
        "has_bid_diagonal", "has_ask_diagonal", "footprint_age_ms",
    ).sort("entry_row")


@nb.njit(cache=True)
def _score_executable(
    rows: np.ndarray, directions: np.ndarray, ask: np.ndarray, bid: np.ndarray,
    timestamps: np.ndarray, holding_ns: int, max_lag_ns: int,
    tick_size: float, commission_ticks: float,
) -> tuple[np.ndarray, ...]:
    size = rows.size
    out_entry = np.empty(size, np.int64); out_exit = np.empty(size, np.int64)
    out_direction = np.empty(size, np.int8); out_gross = np.empty(size, np.float64)
    count = 0; last_exit = -1
    for i in range(size):
        entry = int(rows[i])
        if entry <= last_exit or entry < 0 or entry >= ask.size:
            continue
        target = timestamps[entry] + holding_ns
        exit_row = np.searchsorted(timestamps, target)
        if exit_row >= ask.size or timestamps[exit_row] - target > max_lag_ns:
            continue
        direction = int(directions[i])
        gross = ((bid[exit_row] - ask[entry]) if direction > 0 else (bid[entry] - ask[exit_row])) / tick_size
        out_entry[count] = entry; out_exit[count] = exit_row; out_direction[count] = direction; out_gross[count] = gross
        count += 1; last_exit = exit_row
    return out_entry[:count], out_exit[:count], out_direction[:count], out_gross[:count], out_gross[:count] - commission_ticks


def evaluate_gc_auction_acceptance(
    frame: pl.DataFrame,
    config: GCAuctionAcceptanceConfig = GCAuctionAcceptanceConfig(),
    contract: GCContractSpec = GCContractSpec(),
    sides: Iterable[str] = ("short",),
) -> pl.DataFrame:
    """Generate signals and score a marketable entry/exit at the fixed horizon."""
    ticks = prepare_sierra_ticks(frame)
    signals = generate_gc_auction_acceptance_signals(ticks, config, contract, sides)
    if signals.is_empty():
        return pl.DataFrame(schema={"setup": pl.String, "net_ticks": pl.Float64})
    outputs = []
    for setup in signals["setup"].unique(maintain_order=True).to_list():
        subset = signals.filter(pl.col("setup") == setup)
        entry, exit_, direction, gross, net = _score_executable(
            subset["entry_row"].to_numpy(), subset["direction"].to_numpy(),
            ticks["AskPrice"].to_numpy(), ticks["BidPrice"].to_numpy(), ticks["timestamp_ns"].to_numpy(),
            config.holding_ms * 1_000_000, max(1_000, min(5_000, config.holding_ms)) * 1_000_000,
            contract.tick_size, contract.commission_ticks,
        )
        outputs.append(pl.DataFrame({
            "setup": [setup] * len(entry), "entry_row": entry, "exit_row": exit_,
            "direction": direction, "gross_ticks": gross, "net_ticks": net,
        }))
    return pl.concat(outputs, how="vertical_relaxed") if outputs else pl.DataFrame()
