import polars as pl
import pytest

from orderflow.analysis.strategies.gc_auction_acceptance import (
    GCAuctionAcceptanceConfig, GCContractSpec, _score_executable, prepare_sierra_ticks,
)


def test_frozen_defaults_are_explicit() -> None:
    config = GCAuctionAcceptanceConfig()
    assert config.value_area_fraction == 0.70
    assert config.outside_value_share == 0.15
    assert config.minimum_auction_trades == 20
    assert config.executed_imbalance == 0.80
    assert config.footprint_ms == 1_000
    assert config.footprint_ratio == 3.0
    assert config.holding_ms == 1_000


def test_commission_is_converted_to_ticks() -> None:
    assert GCContractSpec().commission_ticks == 0.45


def test_sierra_fractional_suffix_is_microseconds() -> None:
    frame = pl.DataFrame({
        "Date": ["2024-01-01"], "Time": ["12:00:00.001"], "Sequence": [1],
        "Price": [2000.0], "Volume": [1.0], "TradeType": [2],
        "AskPrice": [2000.1], "BidPrice": [2000.0], "AskSize": [10.0], "BidSize": [10.0],
        "TotalAskDepth": [100.0], "TotalBidDepth": [100.0],
    })
    timestamp = prepare_sierra_ticks(frame)["timestamp_ns"][0]
    base = prepare_sierra_ticks(frame.with_columns(pl.lit("12:00:00.000").alias("Time")))["timestamp_ns"][0]
    assert timestamp - base == 1_000


def test_executable_quotes_charge_spread_and_commission() -> None:
    entry, exit_, direction, gross, net = _score_executable(
        rows=pl.Series([0]).to_numpy(), directions=pl.Series([-1], dtype=pl.Int8).to_numpy(),
        ask=pl.Series([2000.1, 2000.1]).to_numpy(), bid=pl.Series([2000.0, 2000.0]).to_numpy(),
        timestamps=pl.Series([0, 1_000_000_000]).to_numpy(), holding_ns=1_000_000_000,
        max_lag_ns=1_000_000_000, tick_size=0.1, commission_ticks=0.45,
    )
    assert entry.tolist() == [0]
    assert exit_.tolist() == [1]
    assert direction.tolist() == [-1]
    assert gross[0] == pytest.approx(-1.0)
    assert net[0] == pytest.approx(-1.45)
