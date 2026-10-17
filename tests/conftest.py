import _thread
import contextlib
import signal
import threading

import pytest


@pytest.fixture
def ctrl_c_after():
    """
    ``with ctrl_c_after(seconds):`` Ctrl-C reaches the main thread that many seconds into the block,
    as if pressed; leaving the block earlier calls it off.

    Python turns SIGINT into ``KeyboardInterrupt`` only if SIGINT was not ignored when it started, and
    a job started in the background starts with it ignored: the handler is put in place for the test.
    """
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)

    @contextlib.contextmanager
    def pressed(seconds: float):
        timer = threading.Timer(seconds, _thread.interrupt_main)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            timer.join()

    yield pressed
    signal.signal(signal.SIGINT, previous)
