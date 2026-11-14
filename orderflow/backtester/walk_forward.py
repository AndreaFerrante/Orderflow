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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from orderflow.stats import trade_sharpe


@dataclass(frozen=True)
class WalkForwardResult:
    """
    Outcome of ``walk_forward``.

    ``folds`` is the fold calendar of ``walk_forward_folds`` plus ``chosen``,
    ``train_trades``, ``train_score``, ``train_net_pnl``, ``test_trades`` and
    ``test_net_pnl``. ``oos_trades`` is the stitched out-of-sample blotter:
    the chosen candidate's trades of every test window, with ``fold`` and
    ``candidate`` columns, in ``entry_datetime`` order (empty when no fold traded).
    """

    folds: pd.DataFrame
    oos_trades: pd.DataFrame


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


def walk_forward(candidates, *, train_days: int, test_days: int, anchored: bool = False,
                 score=trade_sharpe, min_train_trades: int = 30, dates=None) -> WalkForwardResult:
    """
    Pick the best candidate of every train window and stitch its test-window trades.

    ``candidates`` maps a key (a name or a tuple of parameters) to a blotter.
    Each blotter needs the columns ``entry_datetime`` and ``net_pnl`` and comes
    from one engine run over the whole history. A trade belongs to the day of
    its ``entry_datetime``; trades are read in that order whatever the row order.
    A blotter with no trades (an empty frame) never wins.

    In each fold, a candidate is eligible when it has at least
    ``min_train_trades`` trades in the train window and ``score(pnl)`` of those
    trades is a number. The eligible candidate with the highest score is chosen;
    the first one wins a tie. A fold with no eligible candidate sits out: no
    choice, no trades.

    ``dates`` should be the tick frame's ``Date`` values. Without it the calendar
    is the days on which some candidate traded, so days on which nothing traded
    are not counted. The stitched blotter is exact only when every candidate is
    flat at each day boundary.
    """
    if not candidates:
        raise ValueError("no candidate blotters")
    prepared = []
    for key, frame in candidates.items():
        if frame.empty:
            continue
        frame = frame.sort_values("entry_datetime", kind="stable")
        frame = frame.assign(_day=pd.to_datetime(frame["entry_datetime"]).dt.strftime("%Y-%m-%d"))
        prepared.append((key, frame))
    if dates is None:
        dates = [d for _, frame in prepared for d in frame["_day"]]
    folds = walk_forward_folds(dates, train_days=train_days, test_days=test_days, anchored=anchored)

    chosen_keys, out_rows, oos = [], [], []
    for f in folds.itertuples():
        best = None
        for key, frame in prepared:
            train = frame[(frame["_day"] >= f.train_start) & (frame["_day"] <= f.train_end)]
            pnl = train["net_pnl"].to_numpy(np.float64)
            if len(pnl) < min_train_trades:
                continue
            s = float(score(pnl))
            if np.isnan(s):
                continue
            if best is None or s > best[1]:
                best = (key, s, pnl, frame)
        if best is None:
            chosen_keys.append(None)
            out_rows.append((0, float("nan"), 0.0, 0, 0.0))
            continue
        key, s, pnl, frame = best
        test = frame[(frame["_day"] >= f.test_start) & (frame["_day"] <= f.test_end)]
        # a tuple key must stay one object per cell, not be spread over cells
        test_rows = test.drop(columns="_day")
        test_rows = test_rows.assign(fold=f.fold, candidate=pd.Series([key] * len(test_rows), dtype=object, index=test_rows.index))
        oos.append(test_rows)
        chosen_keys.append(key)
        out_rows.append((len(pnl), s, float(pnl.sum()), len(test), float(test["net_pnl"].sum())))

    folds["chosen"] = pd.Series(chosen_keys, dtype=object, index=folds.index)
    for i, name in enumerate(["train_trades", "train_score", "train_net_pnl", "test_trades", "test_net_pnl"]):
        folds[name] = [row[i] for row in out_rows]
    oos_trades = (pd.concat(oos).sort_values("entry_datetime", kind="stable").reset_index(drop=True)
                  if oos else pd.DataFrame())
    return WalkForwardResult(folds=folds, oos_trades=oos_trades)
