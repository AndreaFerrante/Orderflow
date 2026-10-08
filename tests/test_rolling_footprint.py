"""Tests of the bar-free rolling footprint: synthetic tapes only."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from orderflow.market.microstructure.rolling_footprint import (
    _scan,
    event_columns,
    find_rolling_stacked_imbalances,
    forward_moves_by_tick,
    systematic_events,
)

TICK = 0.25
ASK, BID = 2, 1  # TradeType: 2 = ask trade = buy aggression, 1 = bid trade = sell aggression
START = datetime(2025, 1, 2, 9, 30)

# A buy stack on levels 0, 1, 2 (the last trade is at 2); level -1 has traded, so level 0 has a diagonal.
BUY_STACK = [(-1, 10, BID), (0, 400, ASK), (1, 400, ASK), (2, 400, ASK)]
# Its mirror: a sell stack on levels 2, 1, 0 (the last trade is at 0); level 3 has traded.
SELL_STACK = [(3, 10, ASK), (2, 400, BID), (1, 400, BID), (0, 400, BID)]


def price(level):
    return 5000.0 + TICK * level


def tape(rows, first_index=0):
    """Ticks one second apart. A row is ``(level, contracts, trade_type)``, optionally ``+ (session,)``."""
    return pl.DataFrame({
        "Index": [first_index + i for i in range(len(rows))],
        "Date": ["2025-01-02"] * len(rows),
        "Datetime": [START + timedelta(seconds=i) for i in range(len(rows))],
        "SessionType": [row[3] if len(row) > 3 else "RTH" for row in rows],
        "Price": [price(row[0]) for row in rows],
        "Volume": [row[1] for row in rows],
        "TradeType": [row[2] for row in rows],
    })


def find(rows, first_index=0, **kwargs):
    return find_rolling_stacked_imbalances(tape(rows, first_index), tick_size=TICK, **kwargs)


def test_a_buy_stack_points_long():
    assert find(BUY_STACK)["direction"].to_list() == [1]


def test_a_sell_stack_points_short():
    assert find(SELL_STACK)["direction"].to_list() == [-1]


def test_events_carry_the_signal_tick():
    events = find(BUY_STACK, first_index=1000)
    assert events["signal_index"].to_list() == [1003]
    assert events["Date"].to_list() == ["2025-01-02"]
    assert events["Datetime"].to_list() == [START + timedelta(seconds=3)]
    assert events["SessionType"].to_list() == ["RTH"]


def test_a_buy_stack_reports_the_ask_of_each_level_nearest_the_last_trade_first():
    events = find([(-1, 10, BID), (0, 500, ASK), (1, 450, ASK), (2, 400, ASK)])
    assert events.select("vol_0", "vol_1", "vol_2").row(0) == (400, 450, 500)


def test_a_sell_stack_reports_the_bid_of_each_level_nearest_the_last_trade_first():
    events = find([(3, 10, ASK), (2, 500, BID), (1, 450, BID), (0, 400, BID)])
    assert events.select("vol_0", "vol_1", "vol_2").row(0) == (400, 450, 500)


def test_a_heavy_bid_one_level_below_blocks_the_buy_stack():
    # 400 at the ask of level 2 against 200 at the bid of level 1 is 2:1, under 3:1
    assert find([(-1, 10, BID), (0, 400, ASK), (1, 400, ASK), (1, 200, BID), (2, 400, ASK)]).is_empty()


def test_a_heavy_ask_one_level_above_blocks_the_sell_stack():
    assert find([(3, 10, ASK), (2, 400, BID), (1, 400, BID), (1, 200, ASK), (0, 400, BID)]).is_empty()


def test_a_bid_on_the_same_level_or_above_does_not_block_the_buy_stack():
    rows = [(-1, 10, BID), (3, 200, BID), (2, 200, BID), (0, 400, ASK), (1, 400, ASK), (2, 400, ASK)]
    assert find(rows)["direction"].to_list() == [1]


def test_an_ask_on_the_same_level_or_below_does_not_block_the_sell_stack():
    rows = [(3, 10, ASK), (-1, 200, ASK), (0, 200, ASK), (2, 400, BID), (1, 400, BID), (0, 400, BID)]
    assert find(rows)["direction"].to_list() == [-1]


def test_exactly_three_to_one_is_a_stack_and_just_under_is_not():
    buy = [(-1, 10, BID), (0, 400, ASK), (1, 400, ASK), (1, 150, BID), (2, 450, ASK)]
    sell = [(3, 10, ASK), (2, 400, BID), (1, 400, BID), (1, 150, ASK), (0, 450, BID)]
    assert find(buy)["direction"].to_list() == [1]
    assert find(sell)["direction"].to_list() == [-1]
    assert find(buy[:3] + [(1, 151, BID), (2, 450, ASK)]).is_empty()
    assert find(sell[:3] + [(1, 151, ASK), (0, 450, BID)]).is_empty()


def test_every_level_needs_the_volume_floor_by_itself():
    buy = [(-1, 10, BID), (0, 400, ASK), (1, 400, ASK), (2, 399, ASK)]
    sell = [(3, 10, ASK), (2, 400, BID), (1, 400, BID), (0, 399, BID)]
    assert find(buy).is_empty()
    assert find(sell).is_empty()
    assert find(buy, min_diagonal_volume=399)["direction"].to_list() == [1]
    assert find(sell, min_diagonal_volume=399)["direction"].to_list() == [-1]


def test_a_level_beyond_the_stack_that_never_traded_gives_no_diagonal():
    assert find(BUY_STACK[1:]).is_empty()
    assert find(SELL_STACK[1:]).is_empty()


def test_volume_older_than_the_window_stops_counting():
    rows = [(-1, 10, BID), (0, 400, ASK), (10, 600, ASK), (1, 400, ASK), (2, 400, ASK)]
    assert find(rows, window_contracts=5000)["direction"].to_list() == [1]
    assert find(rows, window_contracts=1000).is_empty()


def test_the_oldest_tick_is_evicted_only_as_far_as_needed():
    rows = [(0, 900, ASK), (-1, 10, BID), (1, 400, ASK), (2, 400, ASK)]
    assert find(rows, window_contracts=1210)["vol_2"].to_list() == [400]  # 400 of the 900 are left
    assert find(rows, window_contracts=1209).is_empty()  # 399 are not enough


def test_the_default_window_is_2000_contracts():
    rows = [(0, 400, ASK), (-1, 10, BID), (1, 400, ASK), (20, 790, ASK), (2, 400, ASK)]  # 2000 contracts
    assert find(rows)["direction"].to_list() == [1]
    assert find(rows[:3] + [(20, 791, ASK), (2, 400, ASK)]).is_empty()  # 2001: one contract of level 0 leaves


def test_a_tick_bigger_than_the_window_is_cut_to_the_window():
    rows = [(0, 800, ASK), (-1, 10, BID), (0, 1, ASK)]
    assert find(rows, window_contracts=500, n_levels=1)["vol_0"].to_list() == [490]


def test_a_stack_signals_once_while_it_lasts():
    assert find(BUY_STACK + [(2, 10, ASK)])["signal_index"].to_list() == [3]


def test_a_stack_that_broke_signals_again_when_it_forms_again():
    assert find(BUY_STACK + [(3, 10, ASK), (2, 10, ASK)])["signal_index"].to_list() == [3, 5]


def test_a_change_of_session_clears_the_window():
    rows = [(-1, 10, BID, "ETH"), (0, 400, ASK, "ETH"), (1, 400, ASK, "ETH"), (2, 400, ASK, "RTH")]
    assert find(rows).is_empty()


def test_a_change_of_session_on_the_second_tick_clears_the_window():
    rows = [(-1, 10, BID, "ETH"), (0, 400, ASK), (1, 400, ASK), (2, 400, ASK)]
    assert find(rows).is_empty()


def test_a_buy_and_a_sell_stack_can_signal_on_the_same_tick():
    rows = [(-3, 10, BID), (3, 10, ASK), (-2, 400, ASK), (-1, 400, ASK), (0, 400, ASK),
            (0, 400, BID), (1, 400, BID), (2, 400, BID), (5, 1, ASK), (0, 1, ASK)]
    events = find(rows, window_contracts=5000)
    assert events.filter(pl.col("signal_index") == 9)["direction"].to_list() == [1, -1]


def test_n_levels_sets_the_height_of_a_stack():
    rows = [(-1, 10, BID), (0, 400, ASK), (1, 400, ASK)]
    two = find(rows, n_levels=2)
    assert two["direction"].to_list() == [1]
    assert two.columns == event_columns(2)
    assert find(rows).is_empty()


def test_window_age_is_the_age_of_the_oldest_contract_in_the_window():
    rows = [(5, 20, ASK), (-1, 10, BID), (0, 400, ASK), (1, 400, ASK), (2, 400, ASK)]
    assert find(rows, window_contracts=1215)["window_age_s"].to_list() == [4.0]  # 5 of the first 20 are left
    assert find(rows, window_contracts=1210)["window_age_s"].to_list() == [3.0]  # all of them are gone


def test_a_tape_with_no_stack_still_has_every_column():
    events = find(BUY_STACK[:2])
    assert events.is_empty()
    assert events.columns == event_columns()
    assert events.schema == find(BUY_STACK).schema


def test_a_lazy_tape_is_read_like_a_frame():
    lazy = find_rolling_stacked_imbalances(tape(BUY_STACK).lazy(), tick_size=TICK)
    assert lazy.equals(find(BUY_STACK))


# --- the definition, recomputed from scratch at every tick, against the running scan -----------------

SMALL = dict(window_contracts=200, imbalance_ratio=2.0, min_diagonal_volume=20)


def random_rows(seed, n, spread=6, unit=False):
    """A random walk that leans on its last move, with heavy-tailed volumes and rare session changes.

    It stays within ``spread`` levels of where it started: a narrow walk makes a narrow ladder, which is where
    an index that wraps around the ladder would show. With ``unit`` every tick is one contract, so a full window
    holds as many ticks as it can: that is where a ring buffer that is too small would show."""
    rng = np.random.default_rng(seed)
    level, lean, session, rows = 0, 1, "RTH", []
    for _ in range(n):
        if rng.random() < 0.01:
            session = "ETH" if session == "RTH" else "RTH"
        step = int(rng.choice([-1, 0, 1]))
        if rng.random() < 0.6:
            step = lean
        level = int(np.clip(level + step, -spread, spread))
        lean = step or lean
        side = ASK if (step > 0) == (rng.random() < 0.8) else BID
        volume = 1 if unit else int(rng.integers(1, 12) ** 2 // 3 + 1)
        rows.append((level, volume, side, session))
    return rows


def reference(rows, window, imbalance_ratio, min_diagonal_volume, n_levels=3):
    """What a signal is, with no ring buffer and no running ladder."""
    out, kept, was = [], [], {1: False, -1: False}
    for i, (level, volume, trade_type, session) in enumerate(rows):
        if i and session != rows[i - 1][3]:
            kept, was = [], {1: False, -1: False}
        kept.append([level, volume, trade_type == ASK, i])
        excess = sum(entry[1] for entry in kept) - window
        while excess > 0:
            cut = min(kept[0][1], excess)
            kept[0][1] -= cut
            excess -= cut
            if kept[0][1] == 0:
                kept.pop(0)
        ask, bid = {}, {}
        for entry_level, left, is_ask, _ in kept:
            book = ask if is_ask else bid
            book[entry_level] = book.get(entry_level, 0) + left
        a, b = (lambda at: ask.get(at, 0)), (lambda at: bid.get(at, 0))
        buy = all(a(level - k) >= min_diagonal_volume and a(level - k) >= imbalance_ratio * b(level - k - 1)
                  and a(level - k - 1) + b(level - k - 1) > 0 for k in range(n_levels))
        sell = all(b(level + k) >= min_diagonal_volume and b(level + k) >= imbalance_ratio * a(level + k + 1)
                   and a(level + k + 1) + b(level + k + 1) > 0 for k in range(n_levels))
        for direction, now in ((1, buy), (-1, sell)):
            if now and not was[direction]:
                volumes = [a(level - k) if direction == 1 else b(level + k) for k in range(n_levels)]
                out.append([i, direction, kept[0][3], *volumes])
            was[direction] = now
    return np.array(out, dtype=np.int64).reshape(-1, 3 + n_levels)


def scan_rows(scan, rows, window_contracts, imbalance_ratio, min_diagonal_volume, n_levels=3):
    levels = np.array([row[0] for row in rows], dtype=np.int64)
    sessions = np.array([row[3] == "RTH" for row in rows])
    level = levels - levels.min() + n_levels + 1
    return scan(level, np.array([row[1] for row in rows], dtype=np.int64),
                np.array([row[2] == ASK for row in rows]),
                np.concatenate(([0], np.cumsum(sessions[1:] != sessions[:-1]))).astype(np.int64),
                window_contracts, float(imbalance_ratio), float(min_diagonal_volume), n_levels,
                int(level.max()) + n_levels + 2)


@pytest.mark.parametrize("scan", [_scan, getattr(_scan, "py_func", _scan)], ids=["compiled", "python"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("spread, n_levels", [(6, 2), (6, 3), (2, 2)])
def test_the_running_scan_matches_the_definition_recomputed_at_every_tick(scan, seed, spread, n_levels):
    rows = random_rows(seed, 800, spread)
    expected = reference(rows, n_levels=n_levels, window=SMALL["window_contracts"],
                         imbalance_ratio=SMALL["imbalance_ratio"],
                         min_diagonal_volume=SMALL["min_diagonal_volume"])
    assert {1, -1} <= set(expected[:, 1].tolist()), "the random tape must produce both kinds of stack"
    assert np.array_equal(scan_rows(scan, rows, n_levels=n_levels, **SMALL), expected)


@pytest.mark.parametrize("scan", [_scan, getattr(_scan, "py_func", _scan)], ids=["compiled", "python"])
@pytest.mark.parametrize("seed", [0, 1, 8, 10])
def test_a_window_full_of_one_contract_ticks_matches_the_definition(scan, seed):
    rows = random_rows(seed, 800, unit=True)
    expected = reference(rows, window=20, imbalance_ratio=2.0, min_diagonal_volume=4, n_levels=2)
    assert len(expected), "the random tape must produce a stack"
    assert np.array_equal(scan_rows(scan, rows, 20, 2.0, 4, n_levels=2), expected)


def test_a_tape_narrower_than_the_stack_is_read_inside_its_ladder(monkeypatch):
    import orderflow.market.microstructure.rolling_footprint as footprint

    # compiled code reads outside an array without a word; the Python body raises, so the Python body is what runs
    monkeypatch.setattr(footprint, "_scan", getattr(footprint._scan, "py_func", footprint._scan))
    rows = random_rows(0, 500, spread=1)  # three levels, and a stack of four
    assert find(rows, n_levels=4, **SMALL).is_empty()


def test_more_signals_than_the_first_buffer_holds_are_all_returned():
    # after each signal the price steps off the stack, to one of 50 levels so that none of them fills up, and back
    pairs = [row for k in range(1100) for row in ((3 + k % 50, 1, ASK), (2, 1, ASK))]
    rows = [(level, volume, trade_type, "RTH") for level, volume, trade_type in BUY_STACK + pairs]
    found = scan_rows(_scan, rows, window_contracts=10**6, imbalance_ratio=3.0, min_diagonal_volume=400)
    assert found.shape == (1101, 6)
    assert found[:3, 0].tolist() == [3, 5, 7]


def test_a_signal_does_not_change_when_later_ticks_change():
    rows = random_rows(7, 800)
    whole = find(rows, n_levels=2, **SMALL)
    assert whole.height > 5
    for signal in whole["signal_index"].to_list()[:8]:
        cut = signal + 1  # the signal tick is the last tick that signal may read
        assert find(rows[:cut], n_levels=2, **SMALL).equals(whole.filter(pl.col("signal_index") < cut))


# --- what is refused ----------------------------------------------------------------------------------

def test_a_missing_column_is_refused():
    with pytest.raises(ValueError, match="Missing required columns"):
        find_rolling_stacked_imbalances(tape(BUY_STACK).drop("Volume"), tick_size=TICK)


def test_a_null_value_is_refused():
    ticks = tape(BUY_STACK).with_columns(
        pl.when(pl.col("Index") == 1).then(None).otherwise(pl.col("Price")).alias("Price"))
    with pytest.raises(ValueError, match="Null values"):
        find_rolling_stacked_imbalances(ticks, tick_size=TICK)


def test_a_trade_type_other_than_1_or_2_is_refused():
    with pytest.raises(ValueError, match="TradeType"):
        find([(0, 5, 0)])


def test_a_tape_out_of_tape_order_is_refused():
    ticks = tape(BUY_STACK).with_columns((pl.col("Index") // 2).alias("Index"))
    with pytest.raises(ValueError, match="strictly increasing"):
        find_rolling_stacked_imbalances(ticks, tick_size=TICK)


def test_an_unknown_session_is_refused():
    with pytest.raises(ValueError, match="SessionType"):
        find([(0, 5, ASK, "EVE")])


def test_a_volume_below_one_is_refused():
    with pytest.raises(ValueError, match="Volume"):
        find([(0, 0, ASK)])


def test_an_empty_tape_is_refused():
    with pytest.raises(ValueError, match="no ticks"):
        find_rolling_stacked_imbalances(tape(BUY_STACK).clear(), tick_size=TICK)


@pytest.mark.parametrize("name, value", [
    ("tick_size", 0), ("window_contracts", 0), ("window_contracts", 1000.5), ("imbalance_ratio", 0),
    ("min_diagonal_volume", 0), ("n_levels", 0), ("n_levels", 2.0), ("n_levels", True),
])
def test_a_parameter_that_cannot_work_is_refused(name, value):
    kwargs = dict(tick_size=TICK, window_contracts=2000, imbalance_ratio=3.0, min_diagonal_volume=400,
                  n_levels=3)
    kwargs[name] = value
    with pytest.raises(ValueError):
        find_rolling_stacked_imbalances(tape(BUY_STACK), **kwargs)


# --- forward moves, in records ------------------------------------------------------------------------

def walk(levels, session="RTH"):
    """One contract at the ask at each level: a tape where only the price path matters."""
    return [(level, 1, ASK, session) for level in levels]


def moves(rows, anchors, directions, first_index=0, **kwargs):
    kwargs.setdefault("horizons", (2,))
    kwargs.setdefault("tick_size", TICK)
    events = pl.DataFrame({"signal_index": anchors, "direction": directions})
    return forward_moves_by_tick(tape(rows, first_index), events, **kwargs)


def test_a_long_event_gains_when_the_price_rises_after_the_entry():
    assert moves(walk([0, 0, 1, 2, 3]), [0], [1])["move_2"].to_list() == [2.0]


def test_a_short_event_gains_when_the_price_falls_after_the_entry():
    assert moves(walk([0, 0, 1, 2, 3]), [0], [-1])["move_2"].to_list() == [-2.0]


def test_the_entry_is_the_tick_after_the_anchor_not_the_anchor():
    assert moves(walk([0, 5, 6, 7]), [0], [1], horizons=(1,))["move_1"].to_list() == [1.0]


def test_the_horizon_counts_records_after_the_entry():
    result = moves(walk([0, 0, 1, 3, 6, 10]), [0], [1], horizons=(1, 3))
    assert result["move_1"].to_list() == [1.0]
    assert result["move_3"].to_list() == [6.0]


def test_a_move_is_null_when_its_record_is_past_the_end_of_the_tape():
    result = moves(walk([0, 0, 1]), [0], [1], horizons=(1, 5))
    assert result["move_1"].to_list() == [1.0]
    assert result["move_5"].to_list() == [None]


def test_a_move_is_null_when_its_record_is_in_another_session():
    rows = walk([0, 0, 1], "RTH") + walk([2, 3], "ETH")
    result = moves(rows, [0], [1], horizons=(1, 2))
    assert result["move_1"].to_list() == [1.0]
    assert result["move_2"].to_list() == [None]


def test_a_move_is_null_when_its_record_is_the_one_just_past_the_end():
    assert moves(walk([0, 0, 1]), [0], [1], horizons=(2,))["move_2"].to_list() == [None]


def test_an_event_on_the_last_tick_is_kept_with_null_moves():
    result = moves(walk([0, 0, 1]), [2], [1])
    assert result.height == 1
    assert result["move_2"].to_list() == [None]


def test_mfe_and_mae_are_the_best_and_the_worst_signed_move_after_the_entry():
    rows = walk([0, 0, 6, -2, 1, -7])  # the first record is the best for a long, the last is the worst
    long = moves(rows, [0], [1], horizons=(2, 4))
    short = moves(rows, [0], [-1], horizons=(2, 4))
    assert (long["mfe_4"].to_list(), long["mae_4"].to_list()) == ([6.0], [-7.0])
    assert (short["mfe_4"].to_list(), short["mae_4"].to_list()) == ([7.0], [-6.0])


def test_mfe_and_mae_are_null_when_the_longest_horizon_is_cut():
    result = moves(walk([0, 0, 3, -2]), [0], [1], horizons=(1, 4))
    assert result["move_1"].to_list() == [3.0]
    assert result["mfe_4"].to_list() == [None]
    assert result["mae_4"].to_list() == [None]


def test_the_default_horizons_are_5_20_and_100():
    events = pl.DataFrame({"signal_index": [0], "direction": [1]})
    result = forward_moves_by_tick(tape(walk(range(110))), events, tick_size=TICK)
    assert result.columns == ["signal_index", "direction", "move_5", "move_20", "move_100", "mfe_100", "mae_100"]
    assert result.row(0)[2:] == (5.0, 20.0, 100.0, 100.0, 0.0)


def test_the_anchor_is_an_index_not_a_row_position():
    assert moves(walk([0, 0, 1, 2]), [1000], [1], first_index=1000)["move_2"].to_list() == [2.0]


def test_events_keep_their_order_and_their_columns():
    events = pl.DataFrame({"signal_index": [2, 0], "direction": [1, -1], "x": ["a", "b"]})
    result = forward_moves_by_tick(tape(walk([0, 0, 1, 2, 3, 4])), events, tick_size=TICK, horizons=(1,))
    assert result["x"].to_list() == ["a", "b"]
    assert result["signal_index"].to_list() == [2, 0]
    assert result["move_1"].to_list() == [1.0, -1.0]


def test_an_anchor_that_is_not_in_the_tape_is_refused():
    with pytest.raises(ValueError, match="not in the tape"):
        moves(walk([0, 0, 1]), [99], [1])


def test_a_direction_other_than_plus_or_minus_one_is_refused():
    with pytest.raises(ValueError, match="must be"):
        moves(walk([0, 0, 1]), [0], [0])


@pytest.mark.parametrize("horizons", [(), (0,), (2.5,), (True,), 5])
def test_horizons_that_cannot_work_are_refused(horizons):
    with pytest.raises(ValueError, match="horizons"):
        moves(walk([0, 0, 1]), [0], [1], horizons=horizons)


def test_a_tick_size_that_cannot_work_is_refused_by_the_forward_moves():
    with pytest.raises(ValueError, match="tick_size"):
        moves(walk([0, 0, 1]), [0], [1], tick_size=0)


def test_the_forward_moves_refuse_a_tape_out_of_tape_order():
    ticks = tape(walk([0, 0, 1])).with_columns((pl.col("Index") // 2).alias("Index"))
    events = pl.DataFrame({"signal_index": [0], "direction": [1]})
    with pytest.raises(ValueError, match="strictly increasing"):
        forward_moves_by_tick(ticks, events, tick_size=TICK, horizons=(1,))


def test_the_forward_moves_refuse_an_empty_tape():
    events = pl.DataFrame({"signal_index": [0], "direction": [1]})
    with pytest.raises(ValueError, match="no ticks"):
        forward_moves_by_tick(tape(walk([0, 0, 1])).clear(), events, tick_size=TICK, horizons=(1,))


# --- the base rate ------------------------------------------------------------------------------------

def test_systematic_events_take_every_step_th_tick_once_long_and_once_short():
    result = systematic_events(tape(walk(range(7)), first_index=10), step=3)
    assert result.columns == ["signal_index", "Date", "Datetime", "SessionType", "direction"]
    assert result["signal_index"].to_list() == [10, 13, 16, 10, 13, 16]
    assert result["direction"].to_list() == [1, 1, 1, -1, -1, -1]


@pytest.mark.parametrize("step", [0, -3, 2.5, True])
def test_a_step_that_cannot_work_is_refused(step):
    with pytest.raises(ValueError, match="step"):
        systematic_events(tape(walk(range(7))), step=step)


def test_a_step_of_one_takes_every_tick():
    result = systematic_events(tape(walk(range(4))), step=1)
    assert result["signal_index"].to_list() == [0, 1, 2, 3, 0, 1, 2, 3]
