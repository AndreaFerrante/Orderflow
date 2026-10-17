"""
The Numba bar-id kernels of the compressor against the Python loops they stand in for, and the
line between the two: an input the kernels are not known to reproduce stays with the loop.
"""

import threading
import time
from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import pytest

from orderflow.compressor import compressor


def _tape(seed: int, n: int, sessions=("ETH", "RTH")):
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.choice(np.arange(1, max(n, 2)), size=min(6, max(n - 1, 0)), replace=False))
    block = np.searchsorted(edges, np.arange(n), side="right")
    session = np.array([sessions[b % len(sessions)] for b in block], dtype=object)
    return rng.integers(1, 3, size=n), rng.integers(0, 40, size=n).astype(np.float64), session


CASES = [(seed, n) for seed, n in enumerate([0, 1, 2, 50, 700, 4000])]


@pytest.mark.parametrize("seed, n", CASES)
@pytest.mark.parametrize("threshold", [1.0, 55.0, 500.0, float("inf")])
def test_delta_bar_ids_equal_reference(seed, n, threshold):
    trade_types, volumes, session = _tape(seed, n)
    for sessions in (session, None):
        fast = compressor._compute_delta_bar_ids(trade_types, volumes, sessions, threshold)
        reference = compressor._compute_delta_bar_ids_py(trade_types, volumes, sessions, threshold)
        for got, expected in zip(fast, reference):
            assert got.dtype == expected.dtype
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("seed, n", CASES)
@pytest.mark.parametrize("threshold", [1.0, 55.0, 500.0, float("inf")])
def test_volume_bar_ids_equal_reference(seed, n, threshold):
    _, volumes, session = _tape(seed, n)
    for sessions in (session, None):
        fast = compressor._compute_volume_bar_ids(volumes, sessions, threshold)
        reference = compressor._compute_volume_bar_ids_py(volumes, sessions, threshold)
        for got, expected in zip(fast, reference):
            assert got.dtype == expected.dtype
            np.testing.assert_array_equal(got, expected)


def test_a_missing_session_label_counts_as_leaving_rth_and_as_a_change():
    """``None`` is not "RTH": it ends an RTH delta bar, and any change of label ends a volume bar."""
    session = np.array(["RTH", "RTH", None, None, "RTH", "ETH"], dtype=object)
    volumes = np.full(6, 1.0)
    trade_types = np.full(6, 2)
    for fast, reference in (
        (compressor._compute_delta_bar_ids(trade_types, volumes, session, 100.0),
         compressor._compute_delta_bar_ids_py(trade_types, volumes, session, 100.0)),
        (compressor._compute_volume_bar_ids(volumes, session, 100.0),
         compressor._compute_volume_bar_ids_py(volumes, session, 100.0)),
    ):
        np.testing.assert_array_equal(fast[0], reference[0])
        np.testing.assert_array_equal(fast[1], reference[1])
    assert compressor._compute_delta_bar_ids(trade_types, volumes, session, 100.0)[0].tolist() == [0, 0, 1, 1, 1, 2]
    assert compressor._compute_volume_bar_ids(volumes, session, 100.0)[0].tolist() == [0, 0, 1, 1, 2, 3]


def test_pandas_and_polars_frames_get_the_same_bar_ids():
    """A pandas frame hands the kernels an Arrow-backed string column: same ids as the polars one."""
    trade_types, volumes, session = _tape(7, 900)
    frame = pd.DataFrame({"TradeType": trade_types, "Volume": volumes, "SessionType": pd.Series(session, dtype="str")})
    from_pandas = compressor.assign_delta_bar_ids(frame, 60)
    from_polars = compressor.assign_delta_bar_ids(pl.from_pandas(frame), 60)
    np.testing.assert_array_equal(from_pandas["delta_bar_id"].to_numpy(), from_polars["delta_bar_id"].to_numpy())
    np.testing.assert_array_equal(
        compressor.assign_volume_bar_ids(frame, 200)["volume_bar_complete"].to_numpy(),
        compressor.assign_volume_bar_ids(pl.from_pandas(frame), 200)["volume_bar_complete"].to_numpy(),
    )


def _outcome(call):
    try:
        return "returned", call()
    except Exception as ex:  # noqa: BLE001
        return "raised", type(ex), str(ex)


def _same_outcome(fast, reference):
    assert fast[0] == reference[0], (fast, reference)
    if fast[0] == "raised":
        assert fast[1:] == reference[1:]
        return
    for got, expected in zip(fast[1], reference[1]):
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


_TT, _VOL, _SES = _tape(3, 300)
# what is unusual about the call -> (trade_types, volumes, session_types, threshold)
UNUSUAL = {
    "no threshold, no ticks": (_TT[:0], _VOL[:0], _SES[:0], None),
    "no threshold": (_TT, _VOL, _SES, None),
    "a threshold given as text": (_TT, _VOL, _SES, "50"),
    "a Decimal threshold": (_TT, _VOL, _SES, Decimal("50.5")),
    "a float32 threshold": (_TT, _VOL, _SES, np.float32(50.5)),
    "an integer threshold beyond 2**53": (_TT, _VOL, _SES, 2 ** 70),
    "volumes as text": (_TT, _VOL.astype(str), _SES, 50.0),
    "float32 volumes": (_TT, (_VOL * 0.37).astype(np.float32), _SES, 50.0),
    "volumes as objects": (_TT, _VOL.astype(object), _SES, 50.0),
    "volumes in two columns": (_TT, np.stack([_VOL, _VOL], axis=1), _SES, 50.0),
    "volumes in a list": (_TT, _VOL.tolist(), _SES, 50.0),
    "fewer volumes than trade types": (_TT, _VOL[:200], _SES, 50.0),
    "fewer volumes than trade types, no labels": (_TT, _VOL[:200], None, 50.0),
    "fewer session labels than ticks": (_TT, _VOL, _SES[:200], 1e9),
    "session labels in a relabelled Series": (_TT, _VOL, pd.Series(_SES, index=np.arange(300)[::-1]), 50.0),
    "a missing trade type": (pd.array([None if k == 5 else int(t) for k, t in enumerate(_TT)], dtype="Int64"), _VOL, _SES, 50.0),
    "trade types as text": (_TT.astype(str), _VOL, _SES, 50.0),
    "trade types in a list": (_TT.tolist(), _VOL, _SES, 50.0),
}


@pytest.mark.parametrize("name", UNUSUAL)
def test_an_unusual_call_behaves_as_the_loop_does(name, monkeypatch):
    trade_types, volumes, session, threshold = UNUSUAL[name]
    _same_outcome(
        _outcome(lambda: compressor._compute_delta_bar_ids(trade_types, volumes, session, threshold)),
        _outcome(lambda: compressor._compute_delta_bar_ids_py(trade_types, volumes, session, threshold)),
    )
    if "trade type" not in name:
        _same_outcome(
            _outcome(lambda: compressor._compute_volume_bar_ids(volumes, session, threshold)),
            _outcome(lambda: compressor._compute_volume_bar_ids_py(volumes, session, threshold)),
        )
    # ...because the loop is what ran: the kernels must not even be tried
    def no_kernel(*args):
        raise AssertionError("kernel called")
    monkeypatch.setattr(compressor, "_delta_bar_kernel", no_kernel)
    monkeypatch.setattr(compressor, "_volume_bar_kernel", no_kernel)
    assert _outcome(lambda: compressor._compute_delta_bar_ids(trade_types, volumes, session, threshold))[1:2] != (AssertionError,)
    if "trade type" not in name:
        assert _outcome(lambda: compressor._compute_volume_bar_ids(volumes, session, threshold))[1:2] != (AssertionError,)


@pytest.mark.parametrize("volumes", [_VOL, _VOL.astype(np.int64), _VOL.astype(np.uint8), _VOL > 20], ids=["float64", "int64", "uint8", "bool"])
@pytest.mark.parametrize("threshold", [50, 50.5, np.int64(50), np.float64(50.5), True, float("nan")], ids=repr)
def test_the_usual_calls_take_the_kernels(volumes, threshold, monkeypatch):
    """The other side of the line, with the result still the loop's."""
    for session in (_SES, pd.Series(_SES, dtype="str").values, None):
        _same_outcome(
            _outcome(lambda: compressor._compute_delta_bar_ids(_TT, volumes, session, threshold)),
            _outcome(lambda: compressor._compute_delta_bar_ids_py(_TT, volumes, session, threshold)),
        )
        _same_outcome(
            _outcome(lambda: compressor._compute_volume_bar_ids(volumes, session, threshold)),
            _outcome(lambda: compressor._compute_volume_bar_ids_py(volumes, session, threshold)),
        )
    def no_loop(*args):
        raise AssertionError("loop called")
    monkeypatch.setattr(compressor, "_compute_delta_bar_ids_py", no_loop)
    monkeypatch.setattr(compressor, "_compute_volume_bar_ids_py", no_loop)
    compressor._compute_delta_bar_ids(_TT, volumes, _SES, threshold)
    compressor._compute_volume_bar_ids(volumes, _SES, threshold)


@pytest.mark.parametrize("dtype", ["str", "string[pyarrow]", "string[python]", "category"])
def test_a_missing_label_in_a_pandas_array_is_compared_by_the_loop(dtype, monkeypatch):
    """How a missing value compares is the array's own business (NaN differs from NaN, ``pd.NA``
    refuses to answer): the loop asks it, tick by tick, as it always did."""
    labels = pd.Series(_SES, dtype=dtype).mask(np.arange(len(_SES)) % 41 == 7).values
    _same_outcome(
        _outcome(lambda: compressor._compute_delta_bar_ids(_TT, _VOL, labels, 50.0)),
        _outcome(lambda: compressor._compute_delta_bar_ids_py(_TT, _VOL, labels, 50.0)),
    )
    _same_outcome(
        _outcome(lambda: compressor._compute_volume_bar_ids(_VOL, labels, 50.0)),
        _outcome(lambda: compressor._compute_volume_bar_ids_py(_VOL, labels, 50.0)),
    )

    def no_kernel(*args):
        raise AssertionError("kernel called")
    monkeypatch.setattr(compressor, "_delta_bar_kernel", no_kernel)
    monkeypatch.setattr(compressor, "_volume_bar_kernel", no_kernel)
    assert _outcome(lambda: compressor._compute_delta_bar_ids(_TT, _VOL, labels, 50.0))[1:2] != (AssertionError,)
    assert _outcome(lambda: compressor._compute_volume_bar_ids(_VOL, labels, 50.0))[1:2] != (AssertionError,)


@pytest.mark.parametrize("dtype", ["str", "string[pyarrow]", "string[python]", "category", object])
def test_complete_pandas_labels_take_the_kernels(dtype, monkeypatch):
    labels = pd.Series(_SES, dtype=dtype).values
    for threshold in (1.0, 50.0, 1e9):
        _same_outcome(
            _outcome(lambda: compressor._compute_delta_bar_ids(_TT, _VOL, labels, threshold)),
            _outcome(lambda: compressor._compute_delta_bar_ids_py(_TT, _VOL, labels, threshold)),
        )
        _same_outcome(
            _outcome(lambda: compressor._compute_volume_bar_ids(_VOL, labels, threshold)),
            _outcome(lambda: compressor._compute_volume_bar_ids_py(_VOL, labels, threshold)),
        )

    def no_loop(*args):
        raise AssertionError("loop called")
    monkeypatch.setattr(compressor, "_compute_delta_bar_ids_py", no_loop)
    monkeypatch.setattr(compressor, "_compute_volume_bar_ids_py", no_loop)
    compressor._compute_delta_bar_ids(_TT, _VOL, labels, 50.0)
    compressor._compute_volume_bar_ids(_VOL, labels, 50.0)


def test_a_pandas_string_column_is_not_turned_into_a_python_string_per_tick():
    """The loop read one label at a time. Unpacking the column instead builds about 50 bytes of
    Python string per tick, gigabytes on a full file: the kernels' wrapper compares integer codes."""
    import tracemalloc

    n = 200_000
    rng = np.random.default_rng(0)
    labels = pd.Series(np.where(np.arange(n) // 5000 % 2 == 0, "ETH", "RTH"), dtype="str").values
    volumes, trade_types = rng.integers(1, 9, size=n).astype(np.float64), rng.integers(1, 3, size=n)
    compressor._compute_volume_bar_ids(volumes[:10], labels[:10], 50.0)
    compressor._compute_delta_bar_ids(trade_types[:10], volumes[:10], labels[:10], 50.0)
    for call in (lambda: compressor._compute_volume_bar_ids(volumes, labels, 50.0),
                 lambda: compressor._compute_delta_bar_ids(trade_types, volumes, labels, 50.0)):
        tracemalloc.start()
        call()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak / n < 30, f"{peak / n:.0f} bytes per tick"


def test_the_module_loads_and_assigns_bars_without_numba():
    """Numba is a declared dependency, yet this module (and ``orderflow.data``, which imports it)
    loaded and worked without one before it had kernels; a Numba that does not match the installed
    NumPy raises ``ImportError`` as well. There the loops assign the bars, as they always did."""
    import json
    import os
    import subprocess
    import sys

    import orderflow

    tape = (
        "rng = np.random.default_rng(3)\n"
        "trade_types, volumes = rng.integers(1, 3, size=400), rng.integers(0, 40, size=400).astype(np.float64)\n"
        "session = np.where(np.arange(400) // 90 % 2 == 0, 'RTH', 'ETH').astype(object)\n"
        "bars = (compressor._compute_delta_bar_ids(trade_types, volumes, session, 55.0)\n"
        "        + compressor._compute_volume_bar_ids(volumes, session, 55.0))\n"
    )
    script = (
        "import sys\n"
        "class NoNumba:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'numba' or name.startswith('numba.'):\n"
        "            raise ImportError('numba blocked')\n"
        "sys.meta_path.insert(0, NoNumba())\n"
        "import json\n"
        "import numpy as np\n"
        "import orderflow.data\n"
        "from orderflow.compressor import compressor\n"
        "assert 'numba' not in sys.modules\n"
        "def no_kernel(*args):\n"
        "    raise AssertionError('a kernel was called without Numba')\n"
        "compressor._delta_bar_kernel = compressor._volume_bar_kernel = no_kernel\n"
        + tape +
        "print(json.dumps([a.tolist() for a in bars]))\n"
    )
    root = os.path.dirname(os.path.dirname(orderflow.__file__))
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": root})
    assert done.returncode == 0, done.stderr[-600:]

    scope = {"np": np, "compressor": compressor}
    exec(tape, scope)  # the same tape here, where the kernels run
    assert json.loads(done.stdout) == [a.tolist() for a in scope["bars"]]


def _interrupted(*args):
    """What a Numba call does when Ctrl-C is pending as it hands its arrays back."""
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt as interrupt:
        raise SystemError("CPUDispatcher(<function kernel>) returned a result with an exception set") from interrupt


@pytest.mark.parametrize("kernel", ["_delta_bar_kernel", "_volume_bar_kernel"])
def test_ctrl_c_during_a_kernel_is_still_a_keyboard_interrupt(monkeypatch, kernel):
    """The loops were Python: Ctrl-C was a ``KeyboardInterrupt``. Out of a Numba call it can come as
    ``SystemError: ... returned a result with an exception set``, which ``except Exception`` swallows."""
    monkeypatch.setattr(compressor, kernel, _interrupted)
    with pytest.raises(KeyboardInterrupt):
        if kernel == "_delta_bar_kernel":
            compressor._compute_delta_bar_ids(_TT, _VOL, _SES, 50.0)
        else:
            compressor._compute_volume_bar_ids(_VOL, _SES, 50.0)


def _bar_kernels(n: int) -> dict:
    """Both kernels on a tape of ``n`` ticks as their wrappers call them: name -> ``call(watch)``."""
    trade_types, volumes, session = _tape(8, n)
    changed, leaves_rth = compressor._session_flags(session, n)
    return {
        "_delta_bar_kernel": lambda watch: compressor._delta_bar_kernel(trade_types == 2, volumes, leaves_rth, 55.0, watch, 1),
        "_volume_bar_kernel": lambda watch: compressor._volume_bar_kernel(volumes, changed, 55.0, watch, 1),
    }


@pytest.mark.parametrize("kernel", ["_delta_bar_kernel", "_volume_bar_kernel"])
def test_a_kernel_counts_its_ticks_and_leaves_at_its_first_look_when_told_to_stop(kernel):
    call = _bar_kernels(5000)[kernel]
    watch = np.zeros(2, np.int64)
    call(watch)
    assert watch.tolist() == [0, 4096]
    watch = np.array([1, 0], np.int64)
    call(watch)
    assert watch.tolist() == [1, 0]


@pytest.mark.parametrize("kernel", ["_delta_bar_kernel", "_volume_bar_kernel"])
def test_a_kernel_stops_when_told_to_while_it_runs(kernel):
    """As ``run_kernel`` tells it: from another thread, by setting ``watch[0]``. That takes a kernel
    that lets other threads run and reads the flag again at every look."""
    call = _bar_kernels(2_000_000)[kernel]
    full_run = np.zeros(2, np.int64)
    call(full_run)
    for _ in range(5):  # (the thread below could be given no time before the kernel is through)
        watch = np.zeros(2, np.int64)
        worker = threading.Thread(target=call, args=(watch,))
        worker.start()
        while watch[1] == 0 and worker.is_alive():
            pass
        watch[0] = 1
        worker.join()
        if watch[1] < full_run[1]:
            return
    pytest.fail(f"ran to its last look ({full_run[1]}) five times out of five")


@pytest.mark.parametrize("kernel", ["_delta_bar_kernel", "_volume_bar_kernel"])
def test_ctrl_c_stops_the_bar_assigner_while_its_kernel_runs(monkeypatch, ctrl_c_after, kernel):
    told = []

    def endless(*arguments):
        watch, deadline = arguments[-2], time.monotonic() + 5
        while not watch[0] and time.monotonic() < deadline:
            time.sleep(0.001)
        told.append(bool(watch[0]))

    monkeypatch.setattr(compressor, kernel, endless)
    trade_types, volumes, session = _tape(8, 3000)
    started = time.perf_counter()
    with ctrl_c_after(0.05), pytest.raises(KeyboardInterrupt):
        if kernel == "_delta_bar_kernel":
            compressor._compute_delta_bar_ids(trade_types, volumes, session, 50.0)
        else:
            compressor._compute_volume_bar_ids(volumes, session, 50.0)
    assert told == [True] and time.perf_counter() - started < 2.0
