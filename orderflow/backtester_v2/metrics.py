"""
Post-trade performance analytics.

Computes institutional-grade metrics from a sequence of ``TradeRecord``
objects produced by the backtesting engine.

All calculations follow industry-standard definitions used by CME/CFTC
reporting and common quant-finance textbooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from orderflow.backtester_v2.models import ExitReason, Side, TradeRecord


# ---------------------------------------------------------------------------
# Metrics data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """
    Comprehensive performance report.

    All dollar amounts are in the instrument's native currency.  Tick-based
    metrics (MAE/MFE) are in **ticks**, not price units.

    Attributes
    ----------
    total_trades : int
    winning_trades : int
    losing_trades : int
    break_even_trades : int
    win_rate : float
        ``winning / total`` (0–1 scale).
    profit_factor : float
        ``gross_profit / gross_loss`` (∞ if no losses).
    gross_profit : float
        Sum of net PnL on winning trades (after commissions).
    gross_loss : float
        Sum of |net PnL| on losing trades (after commissions).
    net_profit : float
        ``gross_profit - gross_loss``.
    total_commissions : float
    avg_trade_pnl : float
    max_win : float
    max_loss : float
    sharpe_ratio : float
        Annualised Sharpe using trade-level returns.  If < 30 trades the
        estimate is unreliable — the value is still computed but the user
        should interpret with caution.
    max_drawdown : float
        Maximum peak-to-trough decline of the equity curve (in dollars).
    max_drawdown_pct : float
        Same as ``max_drawdown`` but as fraction of peak equity.
    avg_mae_ticks : float
        Average Maximum Adverse Excursion (in ticks).
    avg_mfe_ticks : float
        Average Maximum Favorable Excursion (in ticks).
    max_mae_ticks : float
    max_mfe_ticks : float
    avg_ticks_in_trade : float
    long_trades : int
    short_trades : int
    long_win_rate : float
    short_win_rate : float
    exit_reason_counts : Dict[str, int]
    equity_curve : np.ndarray
        Cumulative PnL after each trade.
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commissions: float = 0.0
    avg_trade_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_mae_ticks: float = 0.0
    avg_mfe_ticks: float = 0.0
    max_mae_ticks: float = 0.0
    max_mfe_ticks: float = 0.0
    avg_ticks_in_trade: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------ #
    #  Pretty printing                                                    #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            "=" * 60,
            "  BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"  Total Trades        : {self.total_trades}",
            f"  Winning Trades      : {self.winning_trades}",
            f"  Losing Trades       : {self.losing_trades}",
            f"  Break-Even Trades   : {self.break_even_trades}",
            f"  Win Rate            : {self.win_rate:.2%}",
            "-" * 60,
            f"  Gross Profit        : {self.gross_profit:>12.2f}",
            f"  Gross Loss          : {self.gross_loss:>12.2f}",
            f"  Net Profit          : {self.net_profit:>12.2f}",
            f"  Total Commissions   : {self.total_commissions:>12.2f}",
            f"  Profit Factor       : {self.profit_factor:>12.2f}",
            f"  Avg Trade PnL       : {self.avg_trade_pnl:>12.2f}",
            f"  Max Win             : {self.max_win:>12.2f}",
            f"  Max Loss            : {self.max_loss:>12.2f}",
            "-" * 60,
            f"  Sharpe Ratio (ann.) : {self.sharpe_ratio:>12.4f}",
            f"  Max Drawdown ($)    : {self.max_drawdown:>12.2f}",
            f"  Max Drawdown (%)    : {self.max_drawdown_pct:>12.2%}",
            "-" * 60,
            f"  Avg MAE (ticks)     : {self.avg_mae_ticks:>12.2f}",
            f"  Avg MFE (ticks)     : {self.avg_mfe_ticks:>12.2f}",
            f"  Max MAE (ticks)     : {self.max_mae_ticks:>12.2f}",
            f"  Max MFE (ticks)     : {self.max_mfe_ticks:>12.2f}",
            f"  Avg Ticks in Trade  : {self.avg_ticks_in_trade:>12.1f}",
            "-" * 60,
            f"  Long Trades         : {self.long_trades}",
            f"  Short Trades        : {self.short_trades}",
            f"  Long Win Rate       : {self.long_win_rate:.2%}",
            f"  Short Win Rate      : {self.short_win_rate:.2%}",
            "-" * 60,
            "  Exit Reasons:",
        ]
        for reason, count in sorted(self.exit_reason_counts.items()):
            lines.append(f"    {reason:<24s}: {count}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame for easy export / concatenation."""
        d = {k: v for k, v in self.__dict__.items() if k not in ("equity_curve", "exit_reason_counts")}
        d.update({f"exit_{k}": v for k, v in self.exit_reason_counts.items()})
        return pd.DataFrame([d])


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: List[TradeRecord],
    initial_capital: float = 0.0,
    annualisation_factor: float = 252.0,
) -> PerformanceMetrics:
    """
    Compute ``PerformanceMetrics`` from a list of closed ``TradeRecord`` objects.

    Parameters
    ----------
    trades : list[TradeRecord]
        Chronologically ordered trade records.
    initial_capital : float
        Starting capital (used only for percentage-based drawdown).
    annualisation_factor : float
        Trading days per year (252 for equities / futures, 365 for crypto).

    Returns
    -------
    PerformanceMetrics
    """
    if not trades:
        return PerformanceMetrics()

    n = len(trades)
    net_pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)
    mae_arr  = np.array([t.mae_ticks for t in trades], dtype=np.float64)
    mfe_arr  = np.array([t.mfe_ticks for t in trades], dtype=np.float64)
    ticks_arr = np.array([t.ticks_in_trade for t in trades], dtype=np.float64)
    commissions = np.array([t.commission for t in trades], dtype=np.float64)

    winning_mask = net_pnls > 0
    losing_mask  = net_pnls < 0
    be_mask      = net_pnls == 0

    winning_trades = int(winning_mask.sum())
    losing_trades  = int(losing_mask.sum())
    be_trades      = int(be_mask.sum())

    gross_profit = float(net_pnls[winning_mask].sum()) if winning_trades else 0.0
    gross_loss   = float(np.abs(net_pnls[losing_mask]).sum()) if losing_trades else 0.0
    net_profit   = float(net_pnls.sum())
    total_comm   = float(commissions.sum())

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    win_rate = winning_trades / n if n > 0 else 0.0

    # --- Sharpe ---
    if n > 1 and np.std(net_pnls) > 0:
        sharpe = float(np.mean(net_pnls) / np.std(net_pnls, ddof=1) * np.sqrt(annualisation_factor))
    else:
        sharpe = 0.0

    # --- Equity curve & drawdown ---
    equity = np.cumsum(net_pnls) + initial_capital
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    peak_at_max_dd = float(running_max[np.argmax(drawdowns)]) if len(drawdowns) > 0 else 0.0
    max_dd_pct = (max_dd / peak_at_max_dd) if peak_at_max_dd > 0 else 0.0

    # --- Direction splits ---
    long_mask  = np.array([t.side == Side.LONG for t in trades])
    short_mask = ~long_mask
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    long_wins  = int((winning_mask & long_mask).sum())
    short_wins = int((winning_mask & short_mask).sum())

    # --- Exit reasons ---
    exit_counts: Dict[str, int] = {}
    for t in trades:
        key = t.exit_reason.value
        exit_counts[key] = exit_counts.get(key, 0) + 1

    return PerformanceMetrics(
        total_trades=n,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        break_even_trades=be_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        total_commissions=total_comm,
        avg_trade_pnl=float(net_pnls.mean()),
        max_win=float(net_pnls.max()),
        max_loss=float(net_pnls.min()),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_mae_ticks=float(mae_arr.mean()),
        avg_mfe_ticks=float(mfe_arr.mean()),
        max_mae_ticks=float(mae_arr.max()),
        max_mfe_ticks=float(mfe_arr.max()),
        avg_ticks_in_trade=float(ticks_arr.mean()),
        long_trades=long_count,
        short_trades=short_count,
        long_win_rate=long_wins / long_count if long_count > 0 else 0.0,
        short_win_rate=short_wins / short_count if short_count > 0 else 0.0,
        exit_reason_counts=exit_counts,
        equity_curve=equity,
    )
