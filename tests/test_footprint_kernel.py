"""The Numba path of the stacked-imbalance detector against the Python loop it stands in for."""

import itertools
import threading
import time

import numpy as np
import polars as pl
import pytest

from orderflow.market.microstructure import footprint

TICK = 0.25


def _fast(*arguments):
    """``_detect_fast`` as ``find_stacked_imbalances`` has it run, with nobody watching the kernel."""
    return footprint._detect_fast(*arguments, np.zeros(2, np.int64), 1)


def _tape(seed: int, n: int, *, jump: int = 2, bar: int = 120):
    """Ticks in bars of about ``bar`` ticks: a price walk, mixed sizes, a few prints that are neither side."""
    rng = np.random.default_rng(seed)
    prices = 5000.0 + np.cumsum(rng.integers(-jump, jump + 1, size=n)) * TICK
    volumes = rng.integers(0, 25, size=n).astype(np.float64)
    trade_types = rng.choice(np.array([1, 2, 2, 1, 0, 3]), size=n).astype(np.int64)
    bar_ids = np.cumsum(rng.random(n) < 1.0 / bar).astype(np.int64)
    session_ids = (np.arange(n) // max(n // 3, 1) % 2).astype(np.int64)
    indices = (np.arange(n) * 3 + 11).astype(np.int64)
    return prices, volumes, trade_types, bar_ids, session_ids, indices


# ratio, n_consecutive, min_diagonal_volume, min_winning_volume, min_level_diagonal_volume
RULES = [
    (3.0, 3, 0.0, 0.0, 0.0),
    (1.5, 1, 0.0, 0.0, 0.0),
    (2.0, 2, 40.0, 0.0, 0.0),
    (2.0, 2, 0.0, 25.0, 6.0),
]
CASES = list(itertools.product(range(4), RULES, (False, True), (512, 24)))


@pytest.mark.parametrize("seed, rule, require_breakout, ladder_width", CASES)
def test_the_kernel_emits_the_signals_of_the_reference(seed, rule, require_breakout, ladder_width):
    tape = _tape(seed, 2500, jump=2 if ladder_width == 512 else 5)
    arguments = (*tape, TICK, *rule, require_breakout, 1e-6, ladder_width)
    fast = _fast(*arguments)
    reference = footprint._detect(*arguments)
    assert fast[-1] == reference[-1]  # ticks that fell off the ladder
    for got, expected in zip(fast[:-1], reference[:-1]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("seed", range(3))
def test_a_decimal_tick_size_lands_on_the_same_ladder_levels(seed):
    """0.1 is not a power of two: (price - origin) / tick comes out a hair off a whole number, and the
    level is its nearest integer, as in the reference."""
    rng = np.random.default_rng(seed)
    n = 2500
    prices = 131.0 + np.cumsum(rng.integers(-2, 3, size=n)) * 0.1
    tape = (prices, rng.integers(0, 25, size=n).astype(np.float64), rng.integers(1, 3, size=n).astype(np.int64),
            np.cumsum(rng.random(n) < 0.01).astype(np.int64), np.zeros(n, np.int64), np.arange(n, dtype=np.int64))
    assert np.any(np.abs((prices - prices[0]) / 0.1 - np.rint((prices - prices[0]) / 0.1)) > 0)
    arguments = (*tape, 0.1, 1.5, 2, 0.0, 0.0, 0.0, False, 1e-6, 512)
    fast = _fast(*arguments)
    reference = footprint._detect(*arguments)
    assert len(reference[0]) > 50
    for got, expected in zip(fast[:-1], reference[:-1]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_the_tapes_reach_signals_on_both_sides_breakouts_and_the_ladder_edge():
    """The equality above only binds if the tapes make the detector do all of its work."""
    directions, overflow, signals, confirmed = set(), 0, 0, 0
    for seed, rule, require_breakout, ladder_width in CASES:
        tape = _tape(seed, 2500, jump=2 if ladder_width == 512 else 5)
        out = _fast(*tape, TICK, *rule, require_breakout, 1e-6, ladder_width)
        directions |= set(np.asarray(out[2]).tolist())
        overflow += out[-1]
        signals += len(out[0])
        confirmed += len(out[0]) if require_breakout else 0
    assert directions == {1, -1}
    assert overflow > 0 and signals > 1000 and confirmed > 100


def test_many_signals_grow_the_output_past_its_first_allocation():
    """More than 1024 signals in one call: the kernel's tables have to grow without losing a row."""
    tape = _tape(9, 30_000)
    arguments = (*tape, TICK, 1.5, 1, 0.0, 0.0, 0.0, False, 1e-6, 512)
    fast = _fast(*arguments)
    reference = footprint._detect(*arguments)
    assert len(reference[0]) > 1024
    for got, expected in zip(fast[:-1], reference[:-1]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_many_pending_stacks_in_one_bar_grow_the_pending_table():
    """
    Ninety one-level buy stacks down a staircase, none broken through until the last print lifts
    above all of them: they wait together (the kernel's table starts at 64) and confirm on one tick.
    """
    steps = 90
    prices, volumes, trade_types = [], [], []
    for k in range(steps):
        level = 5000.0 - 3 * k * TICK
        prices += [level - TICK, level]      # a small sell below, then the buying that stacks on it
        volumes += [1.0, 50.0]
        trade_types += [1, 2]
    prices += [5000.0 + TICK, 5000.0 + TICK]  # the breakout print, and the entry tick after it
    volumes += [1.0, 1.0]
    trade_types += [2, 2]
    n = len(prices)
    tape = (np.array(prices), np.array(volumes), np.array(trade_types, dtype=np.int64),
            np.zeros(n, np.int64), np.zeros(n, np.int64), np.arange(n, dtype=np.int64))
    arguments = (*tape, TICK, 1.2, 1, 0.0, 0.0, 0.0, True, 1e-6, 1024)
    fast = _fast(*arguments)
    reference = footprint._detect(*arguments)
    assert len(reference[0]) == steps and set(reference[9]) == {n - 2}
    for got, expected in zip(fast[:-1], reference[:-1]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def _ticks(seed: int = 5, n: int = 1200, nanoseconds: int = 0) -> pl.DataFrame:
    prices, volumes, trade_types, bar_ids, session_ids, indices = _tape(seed, n)
    stamps = pl.datetime_range(pl.datetime(2025, 1, 2, 9), pl.datetime(2025, 1, 2, 9) + pl.duration(seconds=n - 1),
                               "1s", eager=True).cast(pl.Datetime("ns"))
    return pl.DataFrame({
        "Index": indices,
        "Datetime": (stamps.cast(pl.Int64) + nanoseconds * (np.arange(n) % 997)).cast(pl.Datetime("ns")),
        "Price": prices, "Volume": volumes, "TradeType": trade_types,
        "SessionType": np.where(session_ids == 0, "RTH", "ETH"), "volume_bar_id": bar_ids,
    })


LOOSE = {"imbalance_ratio": 1.5, "n_consecutive": 1}


def test_the_public_function_keeps_the_trigger_timestamp_and_the_reference_on_a_missing_price():
    ticks = _ticks().with_columns(pl.col("Datetime").cast(pl.Datetime("us")))
    signals = footprint.find_stacked_imbalances(ticks, tick_size=TICK, **LOOSE)
    assert signals.height > 0
    by_index = dict(zip(ticks["Index"].to_list(), ticks["Datetime"].to_list()))
    assert signals["Datetime"].to_list() == [by_index[i] for i in signals["signal_index"].to_list()]

    with pytest.raises(ValueError):  # the reference loop cannot place a NaN price on the ladder
        footprint.find_stacked_imbalances(
            ticks.with_columns(pl.when(pl.col("Index") == ticks["Index"][40]).then(None).otherwise(pl.col("Price")).alias("Price")),
            tick_size=TICK,
        )


def test_a_nanosecond_timestamp_comes_back_as_it_always_did():
    """The trigger timestamps went through Python datetimes, which stop at the microsecond: a
    nanosecond column keeps its type and loses the digits below. Still so."""
    ticks = _ticks(nanoseconds=1)
    assert (ticks["Datetime"].dt.nanosecond() % 1000 != 0).any()
    signals = footprint.find_stacked_imbalances(ticks, tick_size=TICK, **LOOSE)
    assert signals.height > 50 and signals["Datetime"].dtype == pl.Datetime("ns")
    datetimes = ticks["Datetime"].to_list()
    row_of = {index: row for row, index in enumerate(ticks["Index"].to_list())}
    expected = pl.Series([datetimes[row_of[i]] for i in signals["signal_index"].to_list()], dtype=pl.Datetime("ns"))
    assert signals["Datetime"].to_list() == expected.to_list()
    assert (signals["Datetime"].dt.nanosecond() % 1000 == 0).all()


def _detector_used(monkeypatch, ticks, **kwargs) -> str:
    used = []
    kernel, loop = footprint._detect_fast, footprint._detect
    monkeypatch.setattr(footprint, "_detect_fast", lambda *a: used.append("kernel") or kernel(*a))
    monkeypatch.setattr(footprint, "_detect", lambda *a: used.append("loop") or loop(*a))
    try:
        footprint.find_stacked_imbalances(ticks, **{"tick_size": TICK, **LOOSE, **kwargs})
    except Exception:  # noqa: BLE001  (what the loop raises is its own business)
        pass
    return ",".join(used)


def _with_price(ticks, row, value):
    return ticks.with_columns(pl.when(pl.int_range(pl.len()) == row).then(value).otherwise(pl.col("Price")).alias("Price"))


@pytest.mark.filterwarnings("ignore:.*fell outside the")
@pytest.mark.parametrize("what, change, kwargs", [
    ("a tick size that is not a number", None, {"tick_size": float("nan")}),
    ("an infinite tick size", None, {"tick_size": float("inf")}),
    ("a tick size the price range overflows", None, {"tick_size": 1e-300}),
    ("a ladder of two levels", None, {"ladder_width": 2}),
    ("a ladder of no level", None, {"ladder_width": 0}),
    ("a negative ladder", None, {"ladder_width": -5}),
    ("a NaN price", lambda t: _with_price(t, 40, float("nan")), {}),
    ("an infinite price", lambda t: _with_price(t, 40, float("inf")), {}),
])
def test_what_the_kernel_does_not_place_like_the_loop_stays_with_the_loop(monkeypatch, what, change, kwargs):
    ticks = _ticks() if change is None else change(_ticks())
    assert _detector_used(monkeypatch, ticks, **kwargs) == "loop"


@pytest.mark.filterwarnings("ignore:.*fell outside the")
@pytest.mark.parametrize("kwargs", [{}, {"ladder_width": 3}, {"tick_size": 1}, {"require_breakout": True}], ids=repr)
def test_the_usual_frame_takes_the_kernel(monkeypatch, kwargs):
    assert _detector_used(monkeypatch, _ticks(), **kwargs) == "kernel"
    assert _detector_used(monkeypatch, _ticks().head(0), **kwargs) == "kernel"


@pytest.mark.parametrize("kwargs, message", [
    ({"tick_size": float("nan")}, "cannot convert float NaN to integer"),
    ({"ladder_width": -5}, "negative dimensions are not allowed"),
])
def test_the_loops_own_errors_reach_the_caller(kwargs, message):
    with pytest.raises(ValueError, match=message):
        footprint.find_stacked_imbalances(_ticks(), **{"tick_size": TICK, **LOOSE, **kwargs})


def test_ctrl_c_during_the_kernel_is_still_a_keyboard_interrupt(monkeypatch):
    """The loop was Python: Ctrl-C was a ``KeyboardInterrupt``. Out of a Numba call it can come as
    ``SystemError: ... returned a result with an exception set``, which ``except Exception`` swallows."""
    def interrupted(*args):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt as interrupt:
            raise SystemError("CPUDispatcher(<function kernel>) returned a result with an exception set") from interrupt

    monkeypatch.setattr(footprint, "_detect_fast", interrupted)
    with pytest.raises(KeyboardInterrupt):
        footprint.find_stacked_imbalances(_ticks(), tick_size=TICK, **LOOSE)


def test_loaded_on_its_own_without_numba_the_loop_still_finds_the_signals():
    """This file never needed Numba by itself (its package does, through the rolling footprint).
    Where Numba is missing the kernels are plain functions nobody has tested as such: the loop does
    the work there, as it always did."""
    import json
    import os
    import subprocess
    import sys

    import orderflow

    frame = (
        "rng = np.random.default_rng(5)\n"
        "n = 600\n"
        "ticks = pl.DataFrame({\n"
        "    'Index': np.arange(n), 'Datetime': np.arange(n).astype('datetime64[us]'),\n"
        "    'Price': 5000.0 + np.cumsum(rng.integers(-2, 3, size=n)) * 0.25,\n"
        "    'Volume': rng.integers(0, 25, size=n).astype(np.float64),\n"
        "    'TradeType': rng.integers(1, 3, size=n), 'SessionType': ['RTH'] * n,\n"
        "    'volume_bar_id': np.arange(n) // 120,\n"
        "})\n"
        "signals = module.find_stacked_imbalances(ticks, tick_size=0.25, imbalance_ratio=1.5, n_consecutive=1)\n"
    )
    script = (
        "import sys\n"
        "class NoNumba:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'numba' or name.startswith('numba.'):\n"
        "            raise ImportError('numba blocked')\n"
        "sys.meta_path.insert(0, NoNumba())\n"
        "import importlib.util, json\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "spec = importlib.util.spec_from_file_location('footprint_alone', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'numba' not in sys.modules\n"
        "def no_kernel(*args):\n"
        "    raise AssertionError('the kernel was called without Numba')\n"
        "module._detect_fast = no_kernel\n"
        + frame +
        "print(json.dumps(signals.drop('Datetime').to_dict(as_series=False)))\n"
    )
    root = os.path.dirname(os.path.dirname(orderflow.__file__))
    done = subprocess.run([sys.executable, "-c", script, footprint.__file__], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": root})
    assert done.returncode == 0, done.stderr[-600:]

    scope = {"np": np, "pl": pl, "module": footprint}
    exec(frame, scope)  # the same ticks here, where the kernel runs
    assert scope["signals"].height > 20
    assert json.loads(done.stdout) == scope["signals"].drop("Datetime").to_dict(as_series=False)


def _loose(n: int):
    """The kernel's arguments on a tape of ``n`` ticks with a signal every few ticks."""
    return (*_tape(9, n), TICK, 1.5, 1, 0.0, 0.0, 0.0, False, 1e-6, 512)


def test_the_kernel_counts_its_ticks_and_leaves_at_its_first_look_when_told_to_stop():
    watch = np.zeros(2, np.int64)
    assert len(footprint._detect_fast(*_loose(5000), watch, 1)[0]) > 100
    assert watch.tolist() == [0, 4096]
    watch = np.array([1, 0], np.int64)
    assert len(footprint._detect_fast(*_loose(5000), watch, 1)[0]) == 0
    assert watch.tolist() == [1, 0]


def test_the_kernel_stops_when_told_to_while_it_runs():
    """As ``run_kernel`` tells it: from another thread, by setting ``watch[0]``. That takes a kernel
    that lets other threads run and reads the flag again at every look."""
    arguments = _loose(1_000_000)
    full_run = np.zeros(2, np.int64)
    footprint._detect_fast(*arguments, full_run, 1)
    for _ in range(5):  # (the thread below could be given no time before the kernel is through)
        watch = np.zeros(2, np.int64)
        worker = threading.Thread(target=footprint._detect_fast, args=(*arguments, watch, 1))
        worker.start()
        while watch[1] == 0 and worker.is_alive():
            pass
        watch[0] = 1
        worker.join()
        if watch[1] < full_run[1]:
            return
    pytest.fail(f"ran to its last look ({full_run[1]}) five times out of five")


def test_ctrl_c_stops_the_search_while_the_kernel_runs(monkeypatch, ctrl_c_after):
    told = []

    def endless(*arguments):
        watch, deadline = arguments[-2], time.monotonic() + 5
        while not watch[0] and time.monotonic() < deadline:
            time.sleep(0.001)
        told.append(bool(watch[0]))

    monkeypatch.setattr(footprint, "_detect_fast", endless)
    started = time.perf_counter()
    with ctrl_c_after(0.05), pytest.raises(KeyboardInterrupt):
        footprint.find_stacked_imbalances(_ticks(n=3000), tick_size=TICK, **LOOSE)
    assert told == [True] and time.perf_counter() - started < 2.0
