"""
How a Numba kernel is compiled and run (``orderflow.core._kernel_calls``).

A kernel stands in for a Python loop, and beyond the time it takes its caller must not be able to
tell. The loop let Ctrl-C through between two ticks, kept its progress bar moving and let other
threads run; a module with a loop in it imported wherever Python did. Compiled code gives none of
that by itself.
"""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from numba import njit
from tqdm import tqdm

import orderflow
from orderflow.core import _kernel_calls

ROOT = os.path.dirname(os.path.dirname(orderflow.__file__))


def test_where_numba_has_no_cache_folder_an_import_ends_as_it_did_before_the_kernels():
    """A read-only install without a user cache folder: ``njit(cache=True)`` raises while the module
    is imported. ``orderflow.market`` already stopped there, on ``_scan`` of the rolling footprint,
    and the message says so; the compressor had nothing compiled and imported."""
    script = (
        "import importlib\n"
        "import numba.core.caching as caching\n"
        "caching.CacheImpl._locator_classes = []\n"
        "for name in ('orderflow.compressor.compressor', 'orderflow.market.microstructure.footprint',\n"
        "             'orderflow.market.profiles._session_kernels'):\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        print('imports')\n"
        "    except RuntimeError as error:\n"
        "        print(str(error).split(':')[0])\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ROOT})
    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.splitlines() == ["imports", "cannot cache function '_scan'", "cannot cache function '_scan'"]


def test_a_kernel_compiles_without_a_cache_where_numba_cannot_keep_one(monkeypatch):
    real = _kernel_calls.njit

    def nowhere_to_cache(*args, **kwargs):
        if kwargs.get("cache"):
            raise RuntimeError("cannot cache function 'kernel': no locator available for file 'module.py'")
        return real(*args, **kwargs)

    monkeypatch.setattr(_kernel_calls, "njit", nowhere_to_cache)
    doubled = _kernel_calls.cached_njit(nogil=True)(lambda x: x * 2)
    assert doubled(21) == 42
    assert doubled.targetoptions.get("nogil") is True


# ---------------------------------------------------------------------------
# Running a kernel: its last two arguments are the ``watch`` array ``run_kernel`` makes for it and
# the place in it where the kernel counts its ticks
# ---------------------------------------------------------------------------

MANY = 10 * _kernel_calls.WATCH_EVERY  # a frame long enough to be run on a thread of its own


@njit(nogil=True)
def _spin(rounds, watch, slot):
    """A kernel that only burns time and keeps the look-out every kernel keeps: (rounds done, sum)."""
    total = 0.0
    for r in range(rounds):
        if watch[0] != 0:
            return r, total
        watch[slot] = r
        for k in range(50_000):
            total += np.sqrt(k + r)
    return rounds, total


def _rounds_lasting(seconds: float) -> int:
    _spin(1, np.zeros(2, np.int64), 1)  # compiled before the clock starts
    started = time.perf_counter()
    _spin(200, np.zeros(2, np.int64), 1)
    return int(seconds / ((time.perf_counter() - started) / 200))


def test_ctrl_c_gets_through_while_a_kernel_runs_and_stops_it(ctrl_c_after):
    """The loops were Python: Ctrl-C got in between two ticks. Compiled code cannot be interrupted,
    so the kernel runs on a thread of its own while the caller's stays free to be, and is told to stop."""
    rounds = _rounds_lasting(3.0)
    done = []

    def kernel(count, watch, slot):
        rounds_done = _spin(count, watch, slot)[0]
        time.sleep(0.2)  # still busy when its caller is interrupted: it has to be waited for
        done.append(rounds_done)

    started = time.perf_counter()
    with ctrl_c_after(0.05), pytest.raises(KeyboardInterrupt):
        _kernel_calls.run_kernel(kernel, rounds, ticks=MANY)
    assert time.perf_counter() - started < 1.5
    assert done and 0 < done[0] < rounds  # told to stop and waited for: nothing runs on behind the caller


def test_a_frame_too_short_to_report_on_is_run_where_the_caller_is():
    """A thread costs more than such a kernel takes, and it is over before anyone could interrupt it."""
    threads = []

    def kernel(watch, slot):
        threads.append(threading.get_ident())
        return watch.tolist() + [slot]

    assert _kernel_calls.run_kernel(kernel, ticks=_kernel_calls.WATCH_EVERY - 1) == [0, 0, 1]
    assert _kernel_calls.run_kernel(kernel, ticks=_kernel_calls.WATCH_EVERY) == [0, 0, 1]
    assert threads[0] == threading.get_ident() and threads[1] != threading.get_ident()


@pytest.mark.parametrize("ticks", [5, MANY])
def test_what_the_kernel_returns_or_raises_reaches_the_caller(ticks):
    assert _kernel_calls.run_kernel(lambda a, b, watch, slot: (a + b, watch.size), 2, 3, ticks=ticks, slots=4) == (5, 5)

    error = ZeroDivisionError("as raised")

    def broken(watch, slot):
        raise error

    with pytest.raises(ZeroDivisionError) as caught:
        _kernel_calls.run_kernel(broken, ticks=ticks)
    assert caught.value is error


def test_the_bar_follows_the_kernel_while_it_runs_and_ends_on_the_last_tick(capsys):
    seen = []

    def kernel(watch, slot):
        watch[1], watch[2] = 2000, 3000          # two threads of a kernel, each with its own count
        deadline = time.monotonic() + 5
        while bar.n != 5000 and time.monotonic() < deadline:
            time.sleep(0.001)
        seen.append(bar.n)
        watch[1] = 10 ** 9                       # a count past the end must not push the bar past it
        while bar.n != MANY and time.monotonic() < deadline:
            time.sleep(0.001)
        seen.append(bar.n)

    with tqdm(total=MANY) as bar:
        _kernel_calls.run_kernel(kernel, ticks=MANY, bar=bar, slots=2)
        assert bar.n == MANY
    assert seen == [5000, MANY]

    with tqdm(total=7) as bar:                  # the direct call leaves it on the last tick too
        _kernel_calls.run_kernel(lambda watch, slot: None, ticks=8, bar=bar)
        assert bar.n == 7


def test_where_no_thread_can_be_started_the_kernel_is_still_run(monkeypatch):
    """An interpreter that is shutting down, a process at its thread limit: the loops needed none."""
    def no_thread(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_thread)
    assert _kernel_calls.run_kernel(lambda watch, slot: threading.get_ident(), ticks=MANY) == threading.get_ident()
