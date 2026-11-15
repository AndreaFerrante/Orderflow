"""
Monte Carlo check of the equity path of a finished trade blotter.

One history is one draw. This module shows what other mixes of the same
trading days would have looked like: it resamples days with replacement and
measures net P&L and max drawdown of every simulated path.

Whole days are drawn, not single trades, because the trades of one day share
that day's drift. Within a day the trades keep their entry order. Days without
trades are not in the pool.

The drawdown starts from zero: the equity curve begins at 0 and that start is
a peak, so a loss on the very first trade is a drawdown, as in
``stats.max_drawdown_absolute``.
The days are drawn with the same generator call as
``stats.bootstrap_total_by_group``, so the totals of the two match to float rounding.

``MonteCarloResult.plot`` draws the paths and the drawdowns. The curves it
draws are day-end equity: a fall inside a day is counted in ``max_drawdown``
but is not seen on the curve.
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
    """Day-end equity and max drawdown of every row of ``picks`` (day positions), the curve starting at a peak of 0."""
    equity = np.zeros((picks.shape[0], picks.shape[1] + 1))
    peak = np.zeros(len(picks))
    worst = np.zeros(len(picks))
    for i, d in enumerate(picks.T):
        level = equity[:, i]
        worst = np.maximum(worst, np.maximum(peak - level - low[d], fall[d]))
        peak = np.maximum(peak, level + high[d])
        equity[:, i + 1] = level + total[d]
    return equity, worst


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Outcome of ``monte_carlo``.

    ``paths`` has one row per simulated path, with columns ``net_pnl`` and
    ``max_drawdown``. ``observed_net_pnl`` and ``observed_max_drawdown`` are
    those of the blotter as it was traded.

    ``equity`` is the day-end equity of every path, one row per path and one
    column per day drawn, after a first column of zeros (the start).
    ``observed_equity`` is that curve for the blotter as it was traded.
    ``equity`` holds ``n_resamples * (days + 1)`` floats: 10,000 paths over
    250 days are 20 MB.
    """

    paths: pd.DataFrame
    observed_net_pnl: float
    observed_max_drawdown: float
    equity: np.ndarray
    observed_equity: np.ndarray

    @property
    def p_loss(self) -> float:
        """Share of simulated paths that end at or below zero."""
        return float((self.paths["net_pnl"] <= 0).mean())

    def plot(self):
        """
        Draw the paths and their drawdowns; return the matplotlib figure.

        Left: day-end equity of the first 200 paths, the band that holds 90%
        of all paths with their median, and the observed path. Right: max
        drawdown of all paths, with the observed value and the value 95% of
        the paths stay below.

        The figure is not shown and pyplot does not keep it, so a long job
        can draw one per strategy. Save it with ``fig.savefig(path)``; a
        notebook shows it when it is the last value of a cell.
        """
        from matplotlib.figure import Figure  # here, so that a backtest never loads matplotlib

        blue, orange, ink = "#2a78d6", "#eb6834", "#52514e"
        days = np.arange(self.equity.shape[1])
        low, median, high = np.percentile(self.equity, [5, 50, 95], axis=0)
        drawdown = self.paths["max_drawdown"]
        drawdown_95 = float(drawdown.quantile(0.95))

        fig = Figure(figsize=(12, 5), layout="constrained")
        left, right = fig.subplots(1, 2)
        left.plot(days, self.equity[:200].T, color=ink, linewidth=0.5, alpha=0.2)[0].set_label("one simulated path")
        left.fill_between(days, low, high, color=blue, alpha=0.2, linewidth=0, label="5% to 95% of paths")
        left.plot(days, median, color=blue, linewidth=2, label="median path")
        left.plot(days, self.observed_equity, color=orange, linewidth=2, label="observed")
        left.axhline(0.0, color=ink, linewidth=0.8)
        left.set(title=f"Equity paths: {self.p_loss:.0%} end at or below zero",
                 xlabel="trading days drawn", ylabel="net P&L")
        right.hist(drawdown, bins=50, color=blue)
        right.axvline(self.observed_max_drawdown, color=orange, linewidth=2,
                      label=f"observed: {self.observed_max_drawdown:,.0f}")
        right.axvline(drawdown_95, color=ink, linewidth=2, linestyle="--",
                      label=f"95% of paths stay below: {drawdown_95:,.0f}")
        right.set(title="Max drawdown of the paths", xlabel="max drawdown", ylabel="paths")
        for ax in (left, right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
        return fig


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
    equity, worst = _paths(picks, *summaries)
    obs_equity, obs_worst = _paths(np.arange(n_days)[None, :], *summaries)
    return MonteCarloResult(pd.DataFrame({"net_pnl": equity[:, -1], "max_drawdown": worst}),
                            float(obs_equity[0, -1]), float(obs_worst[0]), equity, obs_equity[0])
