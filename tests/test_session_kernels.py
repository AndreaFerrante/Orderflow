"""
The Numba session kernels against the pure-Python loops they stand in for.

Every public function of ``volume_profile`` / ``vwap`` that resets at the RTH -> ETH change runs a
kernel and keeps its original loop as ``_<name>_py``. These tests hold the two equal, bit for bit,
on random tapes built to reach the awkward cases: ties on volume, zero-volume ticks, sessions of
one tick, prices that skip levels, a tick size that is not a power of two.

A kernel only runs on a frame it is known to reproduce. The second half of this file holds that
line: every other input reaches the reference loop, so it behaves, raises and prints as before.
"""

import re
import threading
import time

import numba
import numpy as np
import pandas as pd
import polars as pl
import pytest

from orderflow.core.configuration import VALUE_AREA
from orderflow.market.profiles import _session_kernels as kernels
from orderflow.market.profiles import volume_profile as vp
from orderflow.market.profiles import vwap as vwap_module


def _tape(seed: int, n: int, *, tick: float = 0.25, step: int = 3, zero_volume: float = 0.1,
          cuts: int = 7) -> pd.DataFrame:
    """A price walk on the tick grid, cut into alternating ETH / RTH blocks."""
    rng = np.random.default_rng(seed)
    price = 5000.0 + np.cumsum(rng.integers(-step, step + 1, size=n)) * tick
    volume = rng.integers(1, 30, size=n)
    volume[rng.random(n) < zero_volume] = 0
    edges = np.sort(rng.choice(np.arange(1, n), size=min(cuts, n - 1), replace=False)) if n > 1 else []
    block = np.searchsorted(edges, np.arange(n), side="right")
    return pd.DataFrame({
        "Price": price,
        "Volume": volume,
        "TradeType": rng.integers(1, 3, size=n),
        "SessionType": np.where(block % 2 == 0, "ETH", "RTH"),
    })


# (seed, ticks, tape options): short frames, wide and narrow walks, gaps, a decimal tick size.
TAPES = [
    (0, 1, {}),
    (1, 2, {}),
    (2, 3, {}),
    (3, 60, {"step": 1}),
    (4, 400, {"step": 1, "zero_volume": 0.4}),
    (5, 1500, {}),
    (6, 1500, {"step": 12, "cuts": 3}),
    (7, 1500, {"tick": 0.01, "step": 2, "zero_volume": 0.0}),
    (8, 3000, {"step": 2, "cuts": 21}),
    (9, 800, {"step": 40, "cuts": 1}),
]


@pytest.fixture(params=TAPES, ids=lambda t: f"seed{t[0]}-n{t[1]}")
def tape(request) -> pd.DataFrame:
    seed, n, options = request.param
    frame = _tape(seed, n, **options)
    # The comparison below means nothing if the public function falls back to the reference.
    assert kernels.session_arrays(frame) is not None
    return frame


def _idle() -> np.ndarray:
    """The ``watch`` array of a kernel nobody watches: the last two arguments of each are it and 1."""
    return np.zeros(2, np.int64)


def _peaks_valleys(volume, level, levels, starts, tick, bandwidth, table_max_levels):
    """The kernel with the array of zeros it fills: (labels, unresolved)."""
    labels = np.zeros(volume.size)
    unresolved = kernels.peaks_valleys(
        volume, level, levels, starts, tick, bandwidth, table_max_levels, 1, labels, _idle(), 1
    )
    return labels, unresolved


def _same_arrays(fast, reference) -> None:
    assert len(fast) == len(reference)
    for got, expected in zip(fast, reference):
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def test_cumulative_delta_equals_reference(tape):
    pd.testing.assert_frame_equal(
        vp.get_dynamic_cumulative_delta_per_session(tape),
        vp._get_dynamic_cumulative_delta_per_session_py(tape),
        check_exact=True,
    )


def test_moving_poc_equals_reference(tape):
    _same_arrays(vp.get_daily_session_moving_POC(tape), vp._get_daily_session_moving_POC_py(tape))


def test_value_area_equals_reference(tape):
    fast = vp.get_volume_profile_areas(tape)
    reference = vp._get_volume_profile_areas_py(tape)
    assert fast.dtype == reference.dtype == np.dtype("<U2")
    np.testing.assert_array_equal(fast, reference)


@pytest.mark.parametrize("bandwidth", [0.8, 0.05])
def test_peaks_valleys_equal_reference(tape, bandwidth, capsys):
    """The kernel is called directly: the public function hands a tape with unresolved ticks to the
    reference, and comparing the reference with itself would prove nothing."""
    tick = float(np.min(np.diff(np.unique(tape["Price"])))) if tape["Price"].nunique() > 1 else 0.25
    _, volume, reset, level, levels = kernels.session_arrays(tape)
    labels, unresolved = _peaks_valleys(
        volume, level, levels, kernels.session_starts(reset), tick, bandwidth, kernels.KDE_TABLE_MAX_LEVELS
    )
    capsys.readouterr()
    reference = vp._get_volume_profile_peaks_valleys_py(tape, tick, False, bandwidth)
    assert labels.dtype == reference.dtype
    np.testing.assert_array_equal(labels, reference)
    # ``unresolved`` counts exactly the ticks the reference reports
    assert unresolved == capsys.readouterr().out.count("Issue, i = ")
    np.testing.assert_array_equal(
        vp.get_volume_profile_peaks_valleys(tape, tick_size=tick, kde_bandwidth=bandwidth), reference
    )


def test_peaks_valleys_without_the_table_equal_reference(tape):
    """``table_max_levels=0`` sends every session down the branch that calls exp() per term."""
    _, volume, reset, level, levels = kernels.session_arrays(tape)
    labels, _ = _peaks_valleys(volume, level, levels, kernels.session_starts(reset), 0.25, 0.8, 0)
    np.testing.assert_array_equal(labels, vp._get_volume_profile_peaks_valleys_py(tape, 0.25, False, 0.8))


def test_session_low_high_equals_reference(tape):
    _same_arrays(vp.get_daily_high_and_low_by_session(tape), vp._get_daily_high_and_low_by_session_py(tape))


def test_node_volume_equals_reference(tape):
    _same_arrays(vp.get_volume_profile_node_volume(tape), vp._get_volume_profile_node_volume_py(tape))


def test_vwap_equals_reference(tape):
    """The kernel is called directly too: a session opening on a zero-volume tick goes to the
    reference in the public function, and the kernel must still carry its NaN the same way."""
    price, volume, reset, _, _ = kernels.session_arrays(tape, with_levels=False)
    vwap, top, bottom = kernels.session_vwap(price, volume, reset, np.array([1.0, 2.0, 3.0, 4.0]), _idle(), 1)
    with np.errstate(invalid="ignore"):
        reference = vwap_module._get_vwap_py(tape)
    np.testing.assert_array_equal(vwap, reference["vwap"].to_numpy())
    for band in range(4):
        np.testing.assert_array_equal(top[band], reference[f"vwap_sd{band + 1}_top"].to_numpy())
        np.testing.assert_array_equal(bottom[band], reference[f"vwap_sd{band + 1}_bottom"].to_numpy())
    with np.errstate(invalid="ignore"):
        pd.testing.assert_frame_equal(vwap_module.get_vwap(tape), reference, check_exact=True)


def test_tapes_reach_every_label_and_the_unresolved_ticks():
    """The equalities above only bind if the tapes walk through every branch of the two heavy kernels."""
    areas, shapes, unresolved = set(), set(), 0
    for seed, n, options in TAPES:
        frame = _tape(seed, n, **options)
        _, volume, reset, level, levels = kernels.session_arrays(frame)
        starts = kernels.session_starts(reset)
        areas |= set(kernels.value_area_labels(volume, level, starts, VALUE_AREA, _idle(), 1).tolist())
        labels, missed = _peaks_valleys(volume, level, levels, starts, 0.25, 0.8, 2048)
        shapes |= set(labels.tolist())
        unresolved += missed
    assert areas == {0, 1, 2}
    assert shapes == {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert unresolved > 0


def test_the_half_distance_level_follows_python_indexing():
    """``source[rank - steps]`` of the reference: a negative index counts back from the top, and
    one below ``-len`` is its IndexError (-1 here)."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        is_traded = rng.random(int(rng.integers(2, 12))) < 0.6
        traded = np.flatnonzero(is_traded)
        for rank, level in enumerate(traded):
            for steps in range(2 * traded.size + 3):
                index = rank - steps
                expected = int(traded[index]) if index >= -traded.size else -1
                assert kernels._traded_level_below(
                    int(level), steps, is_traded, int(traded[0]), int(traded[-1]), traded.size
                ) == expected


def test_a_tie_on_volume_keeps_the_price_traded_first():
    """Two prices at 10 lots: the point of control stays on the one that got there first."""
    frame = pd.DataFrame({
        "Price": [100.0, 101.0, 101.0, 100.0],
        "Volume": [10, 4, 6, 0],
        "TradeType": [2, 2, 2, 2],
        "SessionType": ["ETH"] * 4,
    })
    poc, _ = vp.get_daily_session_moving_POC(frame)
    assert poc.tolist() == [100.0, 100.0, 100.0, 100.0]


def test_sessions_do_not_leak_into_each_other():
    """Reversing the order of the ticks inside the first session leaves the second one untouched."""
    boundary = 300
    frame = _tape(11, 600, cuts=0)
    frame["SessionType"] = np.where(np.arange(600) < boundary, "RTH", "ETH")
    reversed_first = pd.concat([frame.iloc[:boundary].iloc[::-1], frame.iloc[boundary:]], ignore_index=True)
    np.testing.assert_array_equal(
        vp.get_volume_profile_areas(frame)[boundary:], vp.get_volume_profile_areas(reversed_first)[boundary:]
    )
    np.testing.assert_array_equal(
        vp.get_volume_profile_peaks_valleys(frame)[boundary:],
        vp.get_volume_profile_peaks_valleys(reversed_first)[boundary:],
    )


# ---------------------------------------------------------------------------
# The line between kernel and reference loop
# ---------------------------------------------------------------------------

LOOPS = {
    "cd": (vp, "get_dynamic_cumulative_delta_per_session", "_get_dynamic_cumulative_delta_per_session_py"),
    "poc": (vp, "get_daily_session_moving_POC", "_get_daily_session_moving_POC_py"),
    "areas": (vp, "get_volume_profile_areas", "_get_volume_profile_areas_py"),
    "pv": (vp, "get_volume_profile_peaks_valleys", "_get_volume_profile_peaks_valleys_py"),
    "hl": (vp, "get_daily_high_and_low_by_session", "_get_daily_high_and_low_by_session_py"),
    "node": (vp, "get_volume_profile_node_volume", "_get_volume_profile_node_volume_py"),
    "vwap": (vwap_module, "get_vwap", "_get_vwap_py"),
}
ALL = tuple(LOOPS)
WITH_VOLUME = ("cd", "poc", "areas", "pv", "node", "vwap")
REFERENCE = object()


def _clean_tape(seed: int = 20, n: int = 300) -> pd.DataFrame:
    """A tape every kernel takes: no zero volume, prices on a dense 0.25 grid."""
    return _tape(seed, n, step=1, zero_volume=0.0)


def _takes_the_reference(monkeypatch, key, frame, **kwargs) -> bool:
    module, public, loop = LOOPS[key]
    monkeypatch.setattr(module, loop, lambda *args, **kw: REFERENCE)
    return getattr(module, public)(frame, **kwargs) is REFERENCE


def _set(column, row, value, dtype=None):
    def change(frame):
        frame = frame.copy()
        values = frame[column].to_numpy(copy=True)
        if dtype is not None:
            values = values.astype(dtype)
        values[row] = value
        frame[column] = values
        return frame
    return change


def _cast(column, dtype):
    return lambda frame: frame.astype({column: dtype})


def _zero_volume_where_a_session_opens(frame):
    frame = frame.copy()
    session = frame["SessionType"].to_numpy()
    opens = np.flatnonzero((session[1:] == "ETH") & (session[:-1] == "RTH")) + 1
    frame.loc[opens[0], "Volume"] = 0
    return frame


class _LookAlike:
    """Columns by name like a pandas frame, but not one: the loops read ``data.Price`` and fail on it."""

    def __init__(self, frame):
        self._frame = frame
        self.columns, self.shape = frame.columns, frame.shape

    def __getitem__(self, name):
        return self._frame[name]

    def __len__(self):
        return len(self._frame)


# (what is wrong with the frame, how to build it, the functions that must leave it to the reference)
OUTSIDE = [
    ("a polars frame", pl.from_pandas, ALL),
    ("a frame look-alike", _LookAlike, ALL),
    ("no Price column", lambda f: f.drop(columns="Price"), ALL),
    ("no Volume column", lambda f: f.drop(columns="Volume"), WITH_VOLUME),
    ("no TradeType column", lambda f: f.drop(columns="TradeType"), ("cd", "node")),
    ("two Price columns", lambda f: pd.concat([f, f[["Price"]]], axis=1), ALL),
    ("two SessionType columns", lambda f: pd.concat([f, f[["SessionType"]]], axis=1), ALL),
    ("an empty frame", lambda f: f.iloc[:0], ALL),
    ("float32 prices", _cast("Price", "float32"), ALL),
    ("integer prices", lambda f: f.assign(Price=(f["Price"] * 4).round().astype("int64")), ALL),
    ("a NaN price", _set("Price", 50, np.nan), ALL),
    ("an infinite price", _set("Price", 50, np.inf), ALL),
    ("a negative-zero price", _set("Price", 50, -0.0), ALL),
    ("a price beyond 1e100", _set("Price", 50, 1e200), ALL),
    ("float32 volume", lambda f: f.assign(Volume=(f["Volume"] * 0.37).astype("float32")), WITH_VOLUME),
    ("int8 volume", _cast("Volume", "int8"), WITH_VOLUME),
    ("int32 volume", _cast("Volume", "int32"), WITH_VOLUME),
    ("a NaN volume", _set("Volume", 50, np.nan, "float64"), WITH_VOLUME),
    ("an infinite volume", _set("Volume", 50, np.inf, "float64"), WITH_VOLUME),
    ("a negative volume", _set("Volume", 50, -5), WITH_VOLUME),
    ("a negative-zero volume", _set("Volume", 50, -0.0, "float64"), WITH_VOLUME),
    ("a total volume beyond 2**53", _set("Volume", 50, 2 ** 58), WITH_VOLUME),
    ("a missing trade type", lambda f: f.assign(TradeType=f["TradeType"].astype(object).where(np.arange(len(f)) != 4, None)), ("cd", "node")),
    ("trade types as text", _cast("TradeType", "str"), ("cd", "node")),
    ("a missing session label", lambda f: f.assign(SessionType=f["SessionType"].astype(object).where(np.arange(len(f)) != 6, None)), ALL),
    ("session labels that are not text", lambda f: f.assign(SessionType=(f["SessionType"] == "RTH").astype("int64")), ALL),
    ("zero volume on the first tick", _set("Volume", 0, 0), ("vwap",)),
    ("zero volume where a session opens", _zero_volume_where_a_session_opens, ("vwap",)),
]


@pytest.mark.parametrize("change, key", [
    pytest.param(change, key, id=f"{what} / {key}") for what, change, keys in OUTSIDE for key in keys
])
def test_a_frame_outside_the_kernels_is_left_to_the_reference(monkeypatch, change, key):
    assert _takes_the_reference(monkeypatch, key, change(_clean_tape()))


@pytest.mark.parametrize("key", ALL)
def test_a_clean_tape_takes_the_kernel(monkeypatch, key):
    """The other side of the line: without it every test above would pass with the kernels switched off."""
    kwargs = {"tick_size": 0.25, "kde_bandwidth": 0.8} if key == "pv" else {}
    assert not _takes_the_reference(monkeypatch, key, _clean_tape(), **kwargs)


@pytest.mark.parametrize("key", ["poc", "areas", "pv", "hl", "node"])
def test_the_arrays_handed_back_own_their_memory_as_the_loops_arrays_do(monkeypatch, key):
    """Numba returns arrays whose memory belongs to its runtime: ``flags.owndata`` is False, ``base``
    is set, and ``resize`` in place refuses. The loops always handed back plain arrays."""
    module, public, loop = LOOPS[key]
    kwargs = {"tick_size": 0.25, "kde_bandwidth": 0.8} if key == "pv" else {}
    frame = _clean_tape()
    expected = getattr(module, loop)(frame, **kwargs)

    def no_loop(*args, **kw):
        raise AssertionError("the reference loop ran: this compares nothing")
    monkeypatch.setattr(module, loop, no_loop)
    got = getattr(module, public)(frame, **kwargs)

    got, expected = (got, expected) if isinstance(got, tuple) else ((got,), (expected,))
    assert len(got) == len(expected)
    for fast, slow in zip(got, expected):
        assert slow.flags.owndata and slow.base is None  # what the loops hand back
        assert fast.flags.owndata and fast.base is None
        assert (fast.flags.writeable, fast.flags.c_contiguous) == (slow.flags.writeable, slow.flags.c_contiguous)
        np.testing.assert_array_equal(fast, slow)
        fast.resize(len(fast) + 1, refcheck=False)  # allowed on the loops' arrays


def _screen(stderr: str) -> list:
    """What a terminal shows of ``stderr``: a carriage return goes back to the start of the line.
    The elapsed time and the rate of a progress bar, in square brackets, are left out."""
    rows = []
    for line in stderr.split("\n"):
        row = ""
        for piece in line.split("\r"):
            row = piece + row[len(piece):]
        rows.append(re.sub(r"\[.*\]", "[]", row).rstrip())
    return rows


@pytest.mark.parametrize("ticks", [1, 2, 300, 3000])
@pytest.mark.parametrize("key", ALL)
def test_the_progress_bar_of_the_loop_is_still_shown(monkeypatch, capsys, key, ticks):
    """Every loop walked ``tqdm(range(...))``: a bar on stderr that ends on the last tick."""
    module, public, loop = LOOPS[key]
    kwargs = {"tick_size": 0.25, "kde_bandwidth": 0.8} if key == "pv" else {}
    frame = _clean_tape(n=ticks)
    getattr(module, loop)(frame, **kwargs)
    expected = _screen(capsys.readouterr().err)
    assert expected[0].endswith("it []") or expected[0].endswith(f"{len(range(1, ticks - (key == 'pv')))} []")

    def no_loop(*args, **kw):
        raise AssertionError("the reference loop ran: this compares nothing")
    monkeypatch.setattr(module, loop, no_loop)
    getattr(module, public)(frame, **kwargs)
    assert _screen(capsys.readouterr().err) == expected


@pytest.mark.parametrize("what, change", [
    ("no Volume column", lambda f: f.drop(columns="Volume")),
    ("a NaN volume", _set("Volume", 50, np.nan, "float64")),
    ("float32 volume", _cast("Volume", "float32")),
])
def test_session_high_and_low_never_look_at_the_volume(monkeypatch, what, change):
    """The loop reads ``Price`` and ``SessionType`` only, so the kernel must not ask for more."""
    frame = change(_clean_tape())
    _same_arrays(vp.get_daily_high_and_low_by_session(frame), vp._get_daily_high_and_low_by_session_py(frame))
    assert not _takes_the_reference(monkeypatch, "hl", frame)


@pytest.mark.parametrize("kwargs", [
    {"fast": True},
    {"tick_size": np.float32(0.25)},
    {"tick_size": "0.25"},
    {"tick_size": None},
    {"tick_size": 0},
    {"tick_size": -0.25},
    {"tick_size": float("nan")},
    {"tick_size": float("inf")},
    {"tick_size": 1e-300},
    {"kde_bandwidth": np.float32(0.8)},
    {"kde_bandwidth": "0.8"},
    {"kde_bandwidth": None},
    {"kde_bandwidth": 0},
    {"kde_bandwidth": -1.0},
    {"kde_bandwidth": float("nan")},
    {"kde_bandwidth": float("inf")},
    {"kde_bandwidth": 1e-320},
], ids=repr)
def test_kde_parameters_the_kernel_does_not_reproduce_go_to_the_reference(monkeypatch, kwargs):
    assert _takes_the_reference(monkeypatch, "pv", _clean_tape(), **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"tick_size": 0.25, "kde_bandwidth": 0.8},
    {"tick_size": 1, "kde_bandwidth": 5},
    {"tick_size": np.float64(0.25), "kde_bandwidth": np.float64(0.05)},
    {"tick_size": np.int64(1), "kde_bandwidth": True},
], ids=repr)
def test_plain_kde_parameters_take_the_kernel_and_match_the_reference(monkeypatch, kwargs):
    frame = _clean_tape(21)
    np.testing.assert_array_equal(
        vp.get_volume_profile_peaks_valleys(frame, **kwargs),
        vp._get_volume_profile_peaks_valleys_py(frame, kwargs["tick_size"], False, kwargs["kde_bandwidth"]),
    )
    assert not _takes_the_reference(monkeypatch, "pv", frame, **kwargs)


def test_a_price_range_of_more_ticks_than_an_int64_holds_is_not_for_the_kernel():
    """Asked of the check itself: what an overflowing ``int()`` returns in the kernel depends on the processor."""
    price = _clean_tape()["Price"].to_numpy()
    assert kernels.kde_parameters_ok(price, 0.25, 0.8)
    assert not kernels.kde_parameters_ok(price, 1e-300, 0.8)
    assert not kernels.kde_parameters_ok(price, np.ptp(price) / 2.0 ** 62, 0.8)
    assert kernels.kde_parameters_ok(price, np.ptp(price) / 2.0 ** 61, 0.8)


def test_ticks_the_reference_reports_are_reported_the_same_way(capsys):
    """A tick size far too small sends the half distance off the profile: the loop prints two lines
    per such tick, and so must the public function."""
    frame = _clean_tape(22)
    capsys.readouterr()
    reference = vp._get_volume_profile_peaks_valleys_py(frame, 0.01, False, 0.8)
    expected = capsys.readouterr().out
    assert expected.count("Issue, i = ") > 10 and "Exception equal to index" in expected
    labels = vp.get_volume_profile_peaks_valleys(frame, tick_size=0.01, kde_bandwidth=0.8)
    printed = capsys.readouterr().out
    assert printed.partition("...\n")[2] == expected
    np.testing.assert_array_equal(labels, reference)


def test_a_frame_the_kde_kernel_hands_back_shows_the_bar_of_the_loop_alone(capsys):
    """The kernel finds out at the end that only the loop knows what to print: its own bar must be
    off the screen by the time the loop draws one."""
    frame = _clean_tape(22)
    vp._get_volume_profile_peaks_valleys_py(frame, 0.01, False, 0.8)
    expected = _screen(capsys.readouterr().err)
    assert expected[0].endswith("298/298 []")
    vp.get_volume_profile_peaks_valleys(frame, tick_size=0.01, kde_bandwidth=0.8)
    assert _screen(capsys.readouterr().err) == expected


def test_a_frame_outside_the_kernels_still_gets_the_reference_result():
    frame = _tape(13, 300).astype({"Volume": float})
    frame.loc[40, "Volume"] = -3.0
    _same_arrays(vp.get_daily_session_moving_POC(frame), vp._get_daily_session_moving_POC_py(frame))
    pd.testing.assert_frame_equal(vwap_module.get_vwap(frame), vwap_module._get_vwap_py(frame), check_exact=True)


def test_a_session_opening_on_zero_volume_warns_as_the_loop_does():
    frame = _zero_volume_where_a_session_opens(_clean_tape())
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        result = vwap_module.get_vwap(frame)
    assert result["vwap"].isna().any()


def test_value_areas_can_be_computed_from_several_threads_at_once():
    """The loop was plain Python, so this always worked; a parallel kernel aborts the process here
    under Numba's workqueue threading layer, which is why the value-area kernel uses one thread."""
    assert not kernels.value_area_labels.targetoptions.get("parallel", False)
    frames = [_clean_tape(seed, 4000) for seed in range(4)]
    expected = [vp._get_volume_profile_areas_py(frame) for frame in frames]
    results = [None] * len(frames)

    def work(k):
        results[k] = vp.get_volume_profile_areas(frames[k])

    threads = [threading.Thread(target=work, args=(k,)) for k in range(len(frames))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for got, wanted in zip(results, expected):
        np.testing.assert_array_equal(got, wanted)


def test_other_threads_keep_running_while_the_value_area_kernel_works():
    """The loop was Python: another thread never waited more than a few milliseconds for it. A
    kernel that keeps the interpreter lock holds every other Python thread up until it returns,
    seconds on a full file."""
    frame = _tape(1, 600_000, cuts=3)
    _, volume, reset, level, _ = kernels.session_arrays(frame)
    starts = kernels.session_starts(reset)
    kernels.value_area_labels(volume[:10], level[:10], starts[:1], VALUE_AREA, _idle(), 1)  # compiled before the clock starts
    beats, running = [0], [True]

    def heart():
        while running[0]:
            time.sleep(0.001)
            beats[0] += 1

    thread = threading.Thread(target=heart, daemon=True)
    thread.start()
    time.sleep(0.02)
    before, started = beats[0], time.perf_counter()
    kernels.value_area_labels(volume, level, starts, VALUE_AREA, _idle(), 1)
    seconds, during = time.perf_counter() - started, beats[0] - before
    running[0] = False
    thread.join()
    assert seconds > 0.05, "the tape is too small to tell"
    assert during >= 5, f"another thread ran {during} times in {seconds:.2f}s of kernel"


def _kernels_on(frame) -> dict:
    """Every kernel on ``frame`` as its public function calls it: key -> ``call(watch)`` -> its output arrays."""
    price, volume, reset, level, levels = kernels.session_arrays(frame)
    starts = kernels.session_starts(reset)
    trade_type = frame["TradeType"].to_numpy()
    offsets = np.array([1.0, 2.0, 3.0, 4.0])

    def filling(count, kernel, *arguments):
        def call(watch):
            out = [np.zeros(len(volume)) for _ in range(count)]
            kernel(*arguments, *out, watch, 1)
            return out
        return call

    return {
        "cd": lambda watch: list(kernels.cumulative_delta(volume, trade_type == 1, reset, watch, 1)),
        "poc": filling(2, kernels.moving_poc, volume, reset, level, levels),
        "areas": lambda watch: [kernels.value_area_labels(volume, level, starts, VALUE_AREA, watch, 1)],
        "pv": filling(1, kernels.peaks_valleys, volume, level, levels, starts, 0.25, 0.8, kernels.KDE_TABLE_MAX_LEVELS, 1),
        "hl": filling(2, kernels.session_low_high, price, reset),
        "node": filling(4, kernels.node_volume, volume, trade_type == 1, trade_type == 2, reset, level, levels.size),
        "vwap": lambda watch: list(kernels.session_vwap(price, volume, reset, offsets, watch, 1)),
    }


# The last look of each kernel on a frame of 5,000 ticks: they look every 1,024 ticks, the KDE
# kernel, where a tick costs a thousand times more, every 128 (its loop ends one tick early).
LAST_LOOK = {key: 4096 for key in ALL} | {"pv": 4992}


@pytest.mark.parametrize("key", ALL)
def test_a_kernel_counts_its_ticks_where_its_caller_can_read_them(key):
    """What moves the progress bar while the kernel runs."""
    watch = _idle()
    out = _kernels_on(_clean_tape(n=5000))[key](watch)
    assert watch.tolist() == [0, LAST_LOOK[key]]
    assert all(a[..., 1024:].any() for a in out)  # (what the next test finds untouched)


@pytest.mark.parametrize("key", ALL)
def test_a_kernel_told_to_stop_before_it_starts_leaves_at_its_first_look(key):
    """Which comes before it has walked 1,024 ticks (the loops that start on the second tick look
    there for the first time)."""
    watch = np.array([1, 0], np.int64)
    out = _kernels_on(_clean_tape(n=5000))[key](watch)
    assert watch.tolist() == [1, 0]
    assert not any(a[..., 1024:].any() for a in out)


@pytest.mark.parametrize("key", ALL)
def test_a_kernel_stops_when_told_to_while_it_runs(key):
    """As ``run_kernel`` tells it: from another thread, by setting ``watch[0]``. That takes a kernel
    that lets other threads run, and that reads the flag again at every look: a compiler that knows
    the flag's place and the count's reads it once, before the loop."""
    call = _kernels_on(_tape(3, 1_000_000, step=1, zero_volume=0.0, cuts=5))[key]
    full_run = _idle()
    call(full_run)
    for _ in range(5):  # (the thread below could be given no time before the kernel is through)
        watch, errors = _idle(), []

        def work():
            try:
                call(watch)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        worker = threading.Thread(target=work)
        worker.start()
        while watch[1] == 0 and worker.is_alive():
            pass
        watch[0] = 1
        worker.join()
        assert not errors, errors
        if watch[1] < full_run[1]:
            return
    pytest.fail(f"ran to its last look ({full_run[1]}) five times out of five")


KERNELS = {"cd": "cumulative_delta", "poc": "moving_poc", "areas": "value_area_labels", "pv": "peaks_valleys",
           "hl": "session_low_high", "node": "node_volume", "vwap": "session_vwap"}


def _interrupted(*args):
    """What a Numba call does when Ctrl-C is pending as it hands its arrays back."""
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt as interrupt:
        raise SystemError("CPUDispatcher(<function kernel>) returned a result with an exception set") from interrupt


@pytest.mark.parametrize("key", ALL)
def test_ctrl_c_during_a_kernel_is_still_a_keyboard_interrupt(monkeypatch, key):
    """Numba calls a Python helper while it hands arrays back; an interrupt pending by then is raised
    inside it and leaves the call as ``SystemError: ... returned a result with an exception set``.
    The loops raised ``KeyboardInterrupt``, which ``except Exception`` does not swallow."""
    module, public, _ = LOOPS[key]
    kwargs = {"tick_size": 0.25, "kde_bandwidth": 0.8} if key == "pv" else {}
    monkeypatch.setattr(kernels, KERNELS[key], _interrupted)
    with pytest.raises(KeyboardInterrupt):
        getattr(module, public)(_clean_tape(), **kwargs)


def test_any_other_system_error_from_a_kernel_is_left_alone(monkeypatch):
    def broken(*args):
        raise SystemError("something else")

    monkeypatch.setattr(kernels, "moving_poc", broken)
    with pytest.raises(SystemError, match="something else"):
        vp.get_daily_session_moving_POC(_clean_tape())


@pytest.mark.parametrize("key", ALL)
def test_ctrl_c_stops_a_public_function_while_its_kernel_runs(monkeypatch, ctrl_c_after, key):
    """The loops stopped between two ticks. A compiled kernel cannot be interrupted: it is run on a
    thread of its own and told to stop, and the public function raises without waiting for the
    rest of the file. Here a kernel that would otherwise never end."""
    module, public, _ = LOOPS[key]
    kwargs = {"tick_size": 0.25, "kde_bandwidth": 0.8} if key == "pv" else {}
    told = []

    def endless(*arguments):
        watch, deadline = arguments[-2], time.monotonic() + 5
        while not watch[0] and time.monotonic() < deadline:
            time.sleep(0.001)
        told.append(bool(watch[0]))

    monkeypatch.setattr(kernels, KERNELS[key], endless)
    started = time.perf_counter()
    with ctrl_c_after(0.05), pytest.raises(KeyboardInterrupt):
        getattr(module, public)(_clean_tape(n=3000), **kwargs)
    assert told == [True] and time.perf_counter() - started < 2.0


def _many_sessions() -> pd.DataFrame:
    """22 sessions of 150 and 300 ticks: each long enough for its thread to look up at least once."""
    frame = _tape(31, 6300, step=1, zero_volume=0.0, cuts=0)
    frame["SessionType"] = np.where(np.arange(6300) // 150 % 2 == 0, "RTH", "ETH")
    return frame


@pytest.mark.parametrize("threads", [1, 2, 3, 7, 64])
def test_the_kde_kernel_labels_the_same_on_any_number_of_threads(threads):
    frame = _many_sessions()
    _, volume, reset, level, levels = kernels.session_arrays(frame)
    starts = kernels.session_starts(reset)
    assert starts.size == 22
    labels, watch = np.zeros(len(volume)), np.zeros(1 + starts.size, np.int64)
    unresolved = kernels.peaks_valleys(volume, level, levels, starts, 0.25, 0.8, kernels.KDE_TABLE_MAX_LEVELS,
                                       threads, labels, watch, 1)
    assert unresolved == 0
    np.testing.assert_array_equal(labels, vp._get_volume_profile_peaks_valleys_py(frame, 0.25, False, 0.8))
    # on threads every session counts its ticks in a slot of its own; on one, all go in the first
    assert watch[0] == 0 and np.count_nonzero(watch[1:]) == (starts.size if threads > 1 else 1)


def _sessions_of(sizes) -> pd.DataFrame:
    """One session per size: ETH ticks, then RTH ticks (each ETH after an RTH opens a session)."""
    frame = _tape(31, sum(sizes), step=1, zero_volume=0.0, cuts=0)
    labels = []
    for size in sizes:
        labels += ["ETH"] * (size // 2) + ["RTH"] * (size - size // 2)
    frame["SessionType"] = labels
    return frame


def test_the_kde_kernel_deals_its_sessions_biggest_first_to_as_many_threads_as_numba_is_set_to(monkeypatch):
    """The reference ran its density in a Numba parallel region on every tick: ``NUMBA_NUM_THREADS``
    and ``numba.set_num_threads`` in the calling thread bound the cores it took, and still do. One
    session at a time, the biggest first: equal runs of sessions left a 56.8M-tick file waiting 65 s
    on the thread that got the busiest run, against 36 s for its longest session alone."""
    sizes = [300, 900, 300, 1500, 300, 600, 300, 1200, 300, 300]
    calls = []
    real = kernels._peaks_valleys_of

    def recorded(*arguments):
        calls.append((int(arguments[4]), int(arguments[5]), threading.get_ident()))
        return real(*arguments)

    monkeypatch.setattr(kernels, "_peaks_valleys_of", recorded)
    frame = _sessions_of(sizes)
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(2)
        labels = vp.get_volume_profile_peaks_valleys(frame, tick_size=0.25, kde_bandwidth=0.8)
    finally:
        numba.set_num_threads(before)
    np.testing.assert_array_equal(labels, vp._get_volume_profile_peaks_valleys_py(frame, 0.25, False, 0.8))
    assert sorted(call[:2] for call in calls) == [(s, s + 1) for s in range(len(sizes))]
    assert {call[0] for call in calls[:2]} == {3, 7}  # 1,500 and 1,200 ticks
    threads = {call[2] for call in calls}
    assert len(threads) == 2 and threading.get_ident() not in threads


def test_without_a_slot_for_every_session_the_kde_kernel_stays_on_one_thread(monkeypatch):
    """On threads each session counts in ``watch[slot + session]``: a ``watch`` too short for that
    would be written past its end, so the sessions stay on the calling thread, all counting in ``slot``."""
    calls = []
    real = kernels._peaks_valleys_of
    monkeypatch.setattr(kernels, "_peaks_valleys_of", lambda *a: calls.append(threading.get_ident()) or real(*a))
    frame = _many_sessions()
    _, volume, reset, level, levels = kernels.session_arrays(frame)
    starts = kernels.session_starts(reset)
    labels, watch = np.zeros(len(volume)), np.zeros(starts.size, np.int64)  # one slot short
    kernels.peaks_valleys(volume, level, levels, starts, 0.25, 0.8, kernels.KDE_TABLE_MAX_LEVELS, 4, labels, watch, 1)
    assert calls == [threading.get_ident()]
    np.testing.assert_array_equal(labels, vp._get_volume_profile_peaks_valleys_py(frame, 0.25, False, 0.8))


def test_a_short_frame_is_not_split_over_threads(monkeypatch):
    shares = []
    real = kernels._peaks_valleys_of
    monkeypatch.setattr(kernels, "_peaks_valleys_of", lambda *a: shares.append(threading.get_ident()) or real(*a))
    frame = _tape(31, 1000, step=1, zero_volume=0.0, cuts=40)
    vp.get_volume_profile_peaks_valleys(frame, tick_size=0.25, kde_bandwidth=0.8)
    assert shares == [threading.get_ident()]


def test_where_no_thread_can_be_started_the_kde_kernel_still_labels_every_tick(monkeypatch):
    """An interpreter that is shutting down, a process at its thread limit: nothing to split the
    sessions over, and nothing to run the kernel on beside the caller."""
    def no_thread(self):
        raise RuntimeError("can't start new thread")

    frame = _many_sessions()
    expected = vp._get_volume_profile_peaks_valleys_py(frame, 0.25, False, 0.8)
    monkeypatch.setattr(threading.Thread, "start", no_thread)
    np.testing.assert_array_equal(vp.get_volume_profile_peaks_valleys(frame, tick_size=0.25, kde_bandwidth=0.8), expected)
