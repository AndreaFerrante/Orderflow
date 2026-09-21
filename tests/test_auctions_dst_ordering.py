"""Auction/block ordering must survive the November DST fall-back, where local wall-clock time
repeats an hour - two real, distinct ticks can carry the same (or even a reversed) local
timestamp. Sorting or joining on that local time directly can silently invert true order or
mismatch a tick to the wrong block; sorting/joining on `Index` (assigned once, in true
source-instant order, before any timezone conversion) cannot."""

from datetime import datetime

import polars as pl
import pytest

from orderflow.market.microstructure.auctions import (
    aggregate_auctions,
    attach_block_info,
    get_valid_blocks,
)


def dst_fallback_ticks():
    """Two auctions, fed in true chronological order (Index 0..3), shaped like the real
    November fall-back: auction 2 happens strictly LATER in real time than auction 1, but its
    local timestamps (01:05, 01:10 - the second pass through the repeated hour) are numerically
    SMALLER than auction 1's (01:30, 01:35 - the first pass). A sort keyed on local Datetime
    value alone puts auction 2 before auction 1: backward in true time."""
    return pl.DataFrame(
        {
            "Index": [0, 1, 2, 3],
            "Datetime": [
                datetime(2025, 11, 2, 1, 30),
                datetime(2025, 11, 2, 1, 35),
                datetime(2025, 11, 2, 1, 5),
                datetime(2025, 11, 2, 1, 10),
            ],
            "BidPrice": [100.0, 100.0, 102.0, 102.0],
            "AskPrice": [101.0, 101.0, 103.0, 103.0],
            "Volume": [50.0, 60.0, 70.0, 80.0],
            "TradeType": [2, 2, 1, 1],
        }
    )


def test_aggregate_auctions_orders_by_auction_id_not_local_time():
    agg = aggregate_auctions(df=dst_fallback_ticks())
    # Auction 1 (Index 0-1) happened first in true time and must stay first in the output,
    # even though its local StartTime (01:30) is numerically larger than auction 2's (01:05).
    assert agg["AuctionId"].to_list() == [1, 2]
    assert agg["StartIndex"].to_list() == [0, 2]
    assert agg["EndIndex"].to_list() == [1, 3]


def test_get_valid_blocks_carries_end_index():
    agg = aggregate_auctions(df=dst_fallback_ticks())
    blocks = get_valid_blocks(agg=agg, n_consecutive=1, vol_thresh=10, require_nonzero_imbalance=False)
    assert "EndIndex" in blocks.columns
    assert blocks.sort("BlockId")["EndIndex"].to_list() == [1, 3]


def test_attach_block_info_matches_by_index_not_local_time():
    """The whole point: a tick from auction 1 (Index 0-1, true-earlier) must inherit auction 1's
    block, and a tick from auction 2 (Index 2-3, true-later) must inherit auction 2's - regardless
    of local Datetime numerically going backward between them."""
    ticks = dst_fallback_ticks()
    agg = aggregate_auctions(df=ticks)
    blocks = get_valid_blocks(agg=agg, n_consecutive=1, vol_thresh=10, require_nonzero_imbalance=False)

    out = attach_block_info(ticks, blocks).sort("Index")
    block_ids = out["block_blockid"].to_list()
    # "backward" join: a tick only inherits a block that has already CLOSED as of that tick
    # (no lookahead), so Index 0 - block 0's own first tick, before block 0 closes at Index 1 -
    # correctly gets None. What matters for this test: Index 2 and 3 (auction 2, true-later)
    # must NOT inherit block 0 (auction 1's block) just because their local Datetime (01:05,
    # 01:10) is numerically smaller than block 0's local EndTime (01:35).
    assert block_ids == [None, 0, 0, 1]


def test_attach_block_info_empty_blocks_yields_null_columns():
    ticks = dst_fallback_ticks()
    empty_blocks = get_valid_blocks(
        agg=aggregate_auctions(df=ticks), n_consecutive=99, vol_thresh=10, require_nonzero_imbalance=False
    )
    assert empty_blocks.height == 0
    out = attach_block_info(ticks, empty_blocks)
    assert out["block_blockid"].null_count() == out.height
