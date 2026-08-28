"""Tests for the low-volume-node gate on initial balance breaks.

A break that triggers on a low volume node is price moving through a vacuum:
there is nothing at that level to reject it, so it keeps going. Those are the
breaks a reversion strategy must not fade.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from orderflow.market.profiles.initial_balance import (
    compute_initial_balance,
    find_ib_breakouts,
)

IB_PRICES = [4998.0, 5002.0] * 900


def build_day(after_prices, lvn_at=(), after_cvd_step=-1.0, date_str="2025-09-15"):
    """One RTH session. ``lvn_at`` marks post-IB offsets as low volume nodes."""
    open_dt = datetime.fromisoformat(f"{date_str}T08:30:00")
    rows, ask, bid = [], 100_000.0, 100_000.0
    for i, p in enumerate(IB_PRICES):
        rows.append({"Index": i, "Datetime": open_dt + timedelta(seconds=i),
                     "Date": date_str, "Price": float(p), "SessionType": "RTH",
                     "CD_Ask": ask, "CD_Bid": bid, "LVN": 0})
    base = len(IB_PRICES)
    after_open = open_dt + timedelta(minutes=31)
    for i, p in enumerate(after_prices):
        if after_cvd_step >= 0:
            ask += after_cvd_step
        else:
            bid += -after_cvd_step
        rows.append({"Index": base + i, "Datetime": after_open + timedelta(seconds=i),
                     "Date": date_str, "Price": float(p), "SessionType": "RTH",
                     "CD_Ask": ask, "CD_Bid": bid, "LVN": 1 if i in lvn_at else 0})
    rows.append({"Index": base + len(after_prices),
                 "Datetime": open_dt + timedelta(minutes=390), "Date": date_str,
                 "Price": float(rows[-1]["Price"]), "SessionType": "RTH",
                 "CD_Ask": ask, "CD_Bid": bid, "LVN": 0})
    return pl.DataFrame(rows)


def run(ticks, **kw):
    ib = compute_initial_balance(ticks, window_minutes=30, tick_size=0.25)
    kw.setdefault("cvd_lookback_ticks", 10)
    kw.setdefault("flow_gate", "divergent")
    return find_ib_breakouts(ticks, ib, **kw)


BREAK = [5000.0] * 20 + [5003.0] * 20      # offset 20 is the first break


def test_a_session_whose_break_is_all_node_produces_nothing():
    """Every tick beyond the edge is a node, so there is no qualifying
    trigger and the session is skipped rather than fudged."""
    ticks = build_day(BREAK, lvn_at=tuple(range(20, 40)))

    assert run(ticks, lvn_policy="skip").height == 0


def test_the_same_break_off_a_node_is_kept():
    """The control: identical ticks, the node flag alone decides."""
    ticks = build_day(BREAK, lvn_at=())

    assert run(ticks, lvn_policy="skip")["direction"].to_list() == [1]


def test_the_gate_is_off_by_default():
    ticks = build_day(BREAK, lvn_at=(20,))

    assert run(ticks)["direction"].to_list() == [1]


def test_a_later_break_off_a_node_is_still_taken():
    """Dropping the trigger tick must not forfeit the session's signal: the
    first *qualifying* tick is the signal, exactly as with the flow gate."""
    ticks = build_day(BREAK, lvn_at=(20, 21, 22))

    out = run(ticks, lvn_policy="skip")

    assert out.height == 1
    # offsets 20-22 are nodes, so the signal is the tick at offset 23
    assert out["signal_index"].to_list() == [len(IB_PRICES) + 23]


def test_the_gate_reads_the_trigger_tick_not_the_entry_tick():
    """The setup is 'the break happened at a level with volume behind it'.
    The entry tick is merely the next print and says nothing about that."""
    ticks = build_day(BREAK, lvn_at=(21,))     # the ENTRY tick is the node

    assert run(ticks, lvn_policy="skip")["signal_index"].to_list() == [len(IB_PRICES) + 20]


def test_missing_lvn_column_raises_rather_than_passing_everything():
    """Silently ungating is the dangerous failure: the run would look like a
    filtered one and be an unfiltered one."""
    ticks = build_day(BREAK).drop("LVN")

    with pytest.raises(ValueError, match="LVN"):
        run(ticks, lvn_policy="skip")


def test_missing_lvn_column_is_fine_when_the_gate_is_off():
    ticks = build_day(BREAK).drop("LVN")

    assert run(ticks)["direction"].to_list() == [1]


# --------------------------------------------------------------------------
# policy: skip to the next clean trigger, or abandon the session's break
# --------------------------------------------------------------------------

def test_skip_policy_moves_the_signal_to_the_next_clean_tick():
    """`skip` keeps the session: the first *qualifying* tick is the signal."""
    ticks = build_day(BREAK, lvn_at=(20, 21, 22))

    out = run(ticks, lvn_policy="skip")

    assert out["signal_index"].to_list() == [len(IB_PRICES) + 23]


def test_drop_policy_abandons_the_break_entirely():
    """`drop` treats a node trigger as a verdict on the break itself, not on
    the tick: if the break began through a vacuum, the whole extension is
    disqualified and no later tick rehabilitates it."""
    ticks = build_day(BREAK, lvn_at=(20,))

    assert run(ticks, lvn_policy="drop").height == 0


def test_drop_policy_keeps_a_break_that_began_off_a_node():
    """The control: only the FIRST tick beyond the edge is consulted."""
    ticks = build_day(BREAK, lvn_at=(21, 22, 23))

    assert run(ticks, lvn_policy="drop")["signal_index"].to_list() == [len(IB_PRICES) + 20]


def test_the_two_policies_disagree_on_the_same_session():
    """The distinction is the point -- if they never differed, one would do."""
    ticks = build_day(BREAK, lvn_at=(20,))

    assert run(ticks, lvn_policy="skip").height == 1
    assert run(ticks, lvn_policy="drop").height == 0


def test_unknown_lvn_policy_raises():
    ticks = build_day(BREAK)

    with pytest.raises(ValueError, match="lvn_policy"):
        run(ticks, lvn_policy="exclude")
