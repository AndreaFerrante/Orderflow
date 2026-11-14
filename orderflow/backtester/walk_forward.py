"""
Walk-forward check of parameter selection on a finished trade blotter.

Walk-forward answers one question: do parameters chosen on past days still
earn on the days that follow? Each parameter set is run once over the whole
history. The blotters are then cut by date: in every fold the best set of the
train window is picked and its trades of the next test window are kept.

Design calls:

- The check is post hoc on blotters. The engine is never re-run. A trade
  belongs to the day of its ``entry_datetime``.
- The day is the unit. Windows are counted in days present in the calendar,
  not in calendar days, so a weekend is not a day.
- Test windows tile: the window steps forward by ``test_days``, so
  out-of-sample windows never overlap and leave no gap.

A fold's test window never reads a day of its own train window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_folds(dates, *, train_days: int, test_days: int, anchored: bool = False) -> pd.DataFrame:
    """
    Return the fold calendar: one row per fold, in trading days.

    ``dates`` is any iterable of dates (strings, ``date``, ``Timestamp`` or
    ``numpy.datetime64``); it is sorted and reduced to distinct ``YYYY-MM-DD``
    days. The train window is ``train_days`` days long and the test window
    ``test_days`` days long and follows it directly. Each fold moves both
    windows forward by ``test_days``. With ``anchored`` the train window
    always starts on the first day. The last test window may be shorter than
    ``test_days``; its real length is reported.

    Columns: ``fold``, ``train_start``, ``train_end``, ``test_start``,
    ``test_end`` (both ends included), ``n_train_days``, ``n_test_days``.
    Raises ``ValueError`` when a window is not a whole number of days, 1 or
    more, or when there are too few days for one fold.
    """
    for n in (train_days, test_days):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("train_days and test_days must be whole numbers of days, 1 or more")
    calendar = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in set(dates)})
    if len(calendar) <= train_days:
        raise ValueError("not enough days for one fold")
    rows = []
    k = 0
    while (end := k * test_days + train_days) < len(calendar):
        train = calendar[:end] if anchored else calendar[end - train_days:end]
        test = calendar[end:end + test_days]
        rows.append((k, train[0], train[-1], test[0], test[-1], len(train), len(test)))
        k += 1
    return pd.DataFrame(rows, columns=["fold", "train_start", "train_end", "test_start",
                                       "test_end", "n_train_days", "n_test_days"])
