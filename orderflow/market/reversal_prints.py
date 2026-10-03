"""Reversal prints: a large aggressive print that hits against the move of the last minute.

A reversal print is one trade line of the tape that is

* large (``Volume >= min_print_size``) and aggressive (``TradeType`` 1 or 2);
* against the move: the mid quote is lower than ``lookback_s`` seconds earlier for a buy
  print, higher for a sell print;
* at a stretched or defended price, either
  - VWAP variant: the mid is ``vwap_sd_min`` standard deviations or more from the session
    VWAP and the print pushes back toward it, or
  - book variant: the size resting at the execution level is the largest of the first
    ``book_levels`` ladder sizes on both sides and the print is at least that size.

Buy print (``TradeType 2``) is side +1, sell print (``TradeType 1``) is side -1. Everything is
read on the print's own tape line or earlier, so the detection is causal; only the entry
tick lies after the print, by construction.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import polars as pl

_US = 1_000_000


def find_reversal_prints(
    ticks: pl.DataFrame | pl.LazyFrame,
    *,
    tick_size: float,
    min_print_size: int,
    lookback_s: float = 60.0,
    vwap_sd_min: float = 2.0,
    book_levels: int = 20,
    entry_delay_s: float = 1.0,
    last_entry_time: str = "15:40:00",
) -> pl.DataFrame:
    """One row per reversal print, in tape order, with the tick to enter on.

    ``ticks`` needs ``Index``, ``Date``, ``Datetime``, ``SessionType``, ``Price``, ``Volume``,
    ``TradeType``, ``AskPrice``, ``BidPrice``, ``AskSize``, ``BidSize``, ``vwap``,
    ``vwap_sd1_top`` and ``AskDOM_i`` / ``BidDOM_i`` for ``i < book_levels``. A LazyFrame is
    read in two passes, so the ladder columns are only materialised for the big prints.

    The entry tick is the first RTH tick of the same ``Date`` at least ``entry_delay_s`` after
    the print. A print with no such tick, or whose entry tick is at or after
    ``last_entry_time``, is dropped. Prints that share an entry tick keep the earliest.
    """
    rth = ticks.lazy().filter(pl.col("SessionType") == "RTH")
    book = [f"{side}DOM_{i}" for side in ("Ask", "Bid") for i in range(book_levels)]
    base = (
        rth.select("Index", "Date", "Datetime", "Price",
                   ((pl.col("AskPrice") + pl.col("BidPrice")) / 2).alias("mid"))
        .collect().sort("Index").with_row_index("pos")
        .with_columns(pl.col("pos").min().over("Date").alias("day_start"))
    )
    cand = (
        rth.filter((pl.col("Volume") >= min_print_size) & pl.col("TradeType").is_in([1, 2]))
        .select("Index", "Volume", "TradeType", "AskSize", "BidSize", "vwap", "vwap_sd1_top",
                pl.max_horizontal(book).alias("book_max"))
        .collect().sort("Index")
    )

    idx = base["Index"].to_numpy()
    t = base["Datetime"].dt.epoch("us").to_numpy()
    back = np.flatnonzero(np.diff(t) < 0)
    if back.size:  # searchsorted below assumes time never goes backwards
        raise ValueError("RTH Datetime must be non-decreasing in Index order; "
                         f"first offending Index {idx[back[0] + 1]}")
    mid = base["mid"].to_numpy()
    k = np.searchsorted(idx, cand["Index"].to_numpy())
    side = np.where(cand["TradeType"].to_numpy() == 2, 1, -1).astype(np.int64)
    volume = cand["Volume"].to_numpy()

    # Against the move: the reference is the last tick strictly older than the lookback, never
    # one from an earlier day.
    ref = np.maximum(
        np.searchsorted(t, t[k] - int(lookback_s * _US), side="left") - 1,
        base["day_start"].to_numpy().astype(np.int64)[k],
    )
    pre_move = side * (mid[k] - mid[ref]) / tick_size

    # VWAP variant: stretched, and the print pushes back toward VWAP.
    vwap = cand["vwap"].to_numpy()
    sd = cand["vwap_sd1_top"].to_numpy() - vwap
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (mid[k] - vwap) / sd, np.nan)
    variant_vwap = (np.abs(z) >= vwap_sd_min) & (side * np.sign(z) < 0)

    # Book variant: the execution level was the largest size shown and the print took all of it.
    level = np.where(side == 1, cand["AskSize"].to_numpy(), cand["BidSize"].to_numpy())
    book_max = cand["book_max"].to_numpy()
    variant_book = (level >= book_max) & (volume >= level)

    # Entry: first tick at least `entry_delay_s` later, same day, before the cutoff.
    e = np.searchsorted(t, t[k] + int(entry_delay_s * _US), side="left")
    has_tick = e < len(t)
    e = np.where(has_tick, e, 0)
    same_day = (base["Date"].gather(e) == base["Date"].gather(k)).to_numpy()
    in_time = (base["Datetime"].gather(e).dt.time() < time.fromisoformat(last_entry_time)).to_numpy()

    keep = np.flatnonzero(
        (pre_move < 0) & (variant_vwap | variant_book) & has_tick & same_day & in_time
    )
    out = pl.DataFrame({
        "Date": base["Date"].gather(k[keep]),
        "side": side[keep],
        "trigger_index": idx[k[keep]],
        "trigger_datetime": base["Datetime"].gather(k[keep]),
        "volume": volume[keep],
        "mid": mid[k[keep]],
        "pre_move_ticks": pre_move[keep],
        "z": z[keep],
        "variant_vwap": variant_vwap[keep],
        "variant_book": variant_book[keep],
        "level_size": level[keep],
        "book_max": book_max[keep],
        "entry_index": idx[e[keep]],
        "entry_datetime": base["Datetime"].gather(e[keep]),
        "entry_price": base["Price"].gather(e[keep]),
    })
    return out.unique(subset=["entry_index"], keep="first", maintain_order=True)
