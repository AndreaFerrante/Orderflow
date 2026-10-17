"""
How the library compiles and runs its Numba kernels.

A kernel stands in for a Python loop, which stays beside it as the reference. Beyond the time it
takes, the caller of the public function must not be able to tell the two apart.
"""

import threading

import numpy as np

from orderflow.core.exceptions import _call_kernel

try:
    from numba import njit
except ImportError:
    # ``orderflow.compressor`` loaded without Numba before it had kernels: there the loops do the work.
    njit = None

# A kernel reports how far it is, and looks whether it has been told to stop, once in this many
# ticks. A frame shorter than that is over before it could do either.
WATCH_EVERY = 1024
# Seconds between two looks of the waiting thread at a kernel running on another: the bar moves this
# often, and where a signal does not wake a waiting thread (Windows) Ctrl-C waits this long at most.
_PAUSE = 0.02


def cached_njit(**options):
    """
    Decorator: ``njit(cache=True, **options)``, or the same without the cache where Numba has nowhere
    to keep one (a read-only install without a user cache folder: ``njit(cache=True)`` raises there
    while the module is imported). Importing then ends where it did before the kernel existed.

    Without Numba the function comes back as it is, and nothing may call it (``njit is None``).
    """
    def decorate(function):
        if njit is None:
            return function
        try:
            return njit(cache=True, **options)(function)
        except RuntimeError:
            return njit(**options)(function)
    return decorate


def _started(work):
    """A thread running ``work``, or ``None`` where there is none to be had (an interpreter that is
    shutting down, a process at its thread limit): the loops needed no thread."""
    worker = threading.Thread(target=work)
    try:
        worker.start()
    except RuntimeError:
        return None
    return worker


def run_kernel(kernel, *arguments, ticks, bar=None, slots=1):
    """
    ``kernel(*arguments, watch, 1)``, behaving towards its caller as the Python loop it stands in for.

    That loop let Ctrl-C, or any other signal handler, in between two ticks, and moved its ``tqdm``
    bar. Compiled code does neither and cannot be interrupted. So the kernel, compiled ``nogil``,
    runs on a thread of its own while this one waits for it: free for signals, moving ``bar`` to the
    tick count the kernel keeps in ``watch[1:]`` (``slots`` counts: a kernel working on several
    threads uses one each, from the slot it is handed). Whatever is raised here sets ``watch[0]``,
    and the kernel returns at its next look. ``bar`` ends on its last tick.

    The kernel is handed its slot as an argument, never a constant in its code: knowing both places,
    the compiler reads ``watch[0]`` once before the loop and the kernel never sees it change. A
    count written to a place it cannot tell from ``watch[0]`` makes it read the flag again.

    A frame of fewer than ``WATCH_EVERY`` ticks is run right here: it is over before anyone could
    interrupt it, and a thread would cost more than the kernel.
    """
    watch = np.zeros(1 + slots, dtype=np.int64)
    outcome = []

    def work():
        try:
            outcome.append((_call_kernel(kernel, *arguments, watch, 1), None))
        except BaseException as error:  # noqa: BLE001  (raised again below, where the caller is)
            outcome.append((None, error))

    worker = None
    try:
        if ticks >= WATCH_EVERY:
            worker = _started(work)
        if worker is None:
            result = _call_kernel(kernel, *arguments, watch, 1)
        else:
            while worker.is_alive():
                worker.join(_PAUSE)
                if bar is not None:
                    bar.update(min(int(watch[1:].sum()), bar.total) - bar.n)
            result, error = outcome[0]
            if error is not None:
                raise error
    except BaseException:
        watch[0] = 1  # a kernel still running returns at its next look
        if worker is not None:
            worker.join()
        raise
    if bar is not None:
        bar.update(bar.total - bar.n)
    return result
