"""
Monte Carlo check of the equity path of a finished trade blotter.

One history is one draw. This module shows what other mixes of the same
trading days would have looked like: it resamples days with replacement and
measures net P&L and max drawdown of every simulated path.

Whole days are drawn, not single trades, because the trades of one day share
that day's drift. Within a day the trades keep their entry order. Days without
trades are not in the pool.

The drawdown starts from zero: the equity curve begins at 0 and that start is
a peak, so a loss on the very first trade is a drawdown.
``stats.max_drawdown_absolute`` does not count that loss and stays as it is.
The days are drawn with the same generator call as
``stats.bootstrap_total_by_group``, so the totals of the two match to float rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _day_summaries(pnl_by_day):
    """Per day, from its running sum c: total, lowest point, highest point, deepest fall inside the day."""
    total, low, high, fall = (np.empty(len(pnl_by_day)) for _ in range(4))
    for d, pnl in enumerate(pnl_by_day):
        c = np.cumsum(pnl)
        total[d], low[d], high[d] = c[-1], c.min(), c.max()
        fall[d] = (np.maximum.accumulate(c) - c).max()
    return total, low, high, fall


def _paths(picks, total, low, high, fall):
    """Net P&L and max drawdown of every row of ``picks`` (day positions), the curve starting at a peak of 0."""
    level = np.zeros(len(picks))
    peak = np.zeros(len(picks))
    worst = np.zeros(len(picks))
    for d in picks.T:
        worst = np.maximum(worst, np.maximum(peak - level - low[d], fall[d]))
        peak = np.maximum(peak, level + high[d])
        level = level + total[d]
    return level, worst


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Outcome of ``monte_carlo``.

    ``paths`` has one row per simulated path, with columns ``net_pnl`` and
    ``max_drawdown``. ``observed_net_pnl`` and ``observed_max_drawdown`` are
    those of the blotter as it was traded.
    """

    paths: pd.DataFrame
    observed_net_pnl: float
    observed_max_drawdown: float

    @property
    def p_loss(self) -> float:
        """Share of simulated paths that end at or below zero."""
        return float((self.paths["net_pnl"] <= 0).mean())


def monte_carlo(trades: pd.DataFrame, *, n_resamples: int = 10_000, seed: int = 42) -> MonteCarloResult:
    """
    Resample the trading days of a blotter and return the paths they make.

    ``trades`` needs the columns ``entry_datetime`` and ``net_pnl``. A trade
    belongs to the day of its ``entry_datetime``. Each path draws as many days
    as the blotter has, with replacement, and lays their trades end to end in
    entry order. The same inputs and seed give the same paths. Raises
    ``ValueError`` when the blotter has no trades.
    """
    if trades.empty:
        raise ValueError("no trades to resample")
    ordered = trades.sort_values("entry_datetime", kind="stable")
    day = pd.to_datetime(ordered["entry_datetime"]).dt.strftime("%Y-%m-%d")
    pnl_by_day = [g.to_numpy(np.float64) for _, g in ordered["net_pnl"].groupby(day)]
    summaries = _day_summaries(pnl_by_day)
    n_days = len(pnl_by_day)
    picks = np.random.default_rng(seed).integers(0, n_days, size=(n_resamples, n_days))
    net, worst = _paths(picks, *summaries)
    obs_net, obs_worst = _paths(np.arange(n_days)[None, :], *summaries)
    return MonteCarloResult(pd.DataFrame({"net_pnl": net, "max_drawdown": worst}),
                            float(obs_net[0]), float(obs_worst[0]))
