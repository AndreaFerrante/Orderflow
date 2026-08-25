"""Reusable strategy definitions built from order-flow primitives."""

from orderflow.analysis.strategies.gc_auction_acceptance import (
    GCAuctionAcceptanceConfig,
    GCContractSpec,
    evaluate_gc_auction_acceptance,
    generate_gc_auction_acceptance_signals,
)

__all__ = [
    "GCAuctionAcceptanceConfig",
    "GCContractSpec",
    "evaluate_gc_auction_acceptance",
    "generate_gc_auction_acceptance_signals",
]
