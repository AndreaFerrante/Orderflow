"""
Position-level risk management.

The ``RiskManager`` is called on every tick while a position is open.
It updates the mutable ``PositionState`` (trailing-stop level, break-even
flag, etc.) and returns an ``ExitSignal`` when a risk threshold is breached.

Separation from the exit-strategy layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Exit strategies* encode the trader's **alpha logic** ("close after 30 min",
"close when ATR spikes").  The *risk manager* encodes **mechanical guards**
(TP, SL, trailing stop, break-even) that override or supplement alpha logic.
The engine checks *risk first, then strategy* on each tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from orderflow.backtester.models import (
    ExitReason,
    ExitSignal,
    PositionState,
    Side,
    Tick,
)


@dataclass(slots=True)
class RiskManager:
    """
    Mechanical risk overlay applied every tick while a position is live.

    Parameters
    ----------
    tp_ticks : float | None
        Take-profit distance in ticks from entry.
    sl_ticks : float | None
        Stop-loss distance in ticks from entry.
    trailing_stop_ticks : float | None
        Trailing stop distance in ticks.  The stop follows the price at
        ``trail`` ticks behind the best price observed so far, and **never
        moves against** the position.
    break_even_ticks : float | None
        When the position moves *this many ticks* in-the-money, the stop
        is automatically moved to the entry price (adjusted by
        ``break_even_offset_ticks`` if desired).
    break_even_offset_ticks : float
        Extra ticks beyond entry price for break-even stop (to cover costs).
    tick_size : float
        The instrument's minimum price increment.
    tick_value : float
        Dollar value of one tick.
    fill_on_touch : bool
        ``True``: the target fills when a trade prints at its level. ``False``:
        only when a trade prints beyond it, the stricter test of a resting
        limit order that sits at the back of the queue.

    Notes
    -----
    * All distances are in **ticks** (not price units) for clarity.
    * The manager mutates ``PositionState`` in-place for speed.
    * When multiple conditions trigger on the same tick, priority is:
      ``stop_loss > break_even > trailing_stop > take_profit``.
    * A stop is a market order: its signal names no price, so the engine fills
      it at the tick that triggered it, however far beyond the stop that is.
      A target is a limit order: it fills at its own level, never better.
    """
    tp_ticks: Optional[float] = None
    sl_ticks: Optional[float] = None
    trailing_stop_ticks: Optional[float] = None
    break_even_ticks: Optional[float] = None
    break_even_offset_ticks: float = 0.0
    tick_size: float = 0.25
    tick_value: float = 12.5
    fill_on_touch: bool = True

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def initialize_position(self, pos: PositionState) -> None:
        """
        Called once on entry to set initial stop/target levels.
        """
        # No stop loss is a stop infinitely far away, not a stop at 0: at 0 a short's break-even
        # or trailing level, which must be *below* the stop to replace it, never took hold.
        if self.sl_ticks is not None:
            if pos.side == Side.LONG:
                pos.current_stop = pos.entry_price - self.sl_ticks * self.tick_size
            else:
                pos.current_stop = pos.entry_price + self.sl_ticks * self.tick_size
        else:
            pos.current_stop = -math.inf if pos.side == Side.LONG else math.inf

        if self.tp_ticks is not None:
            if pos.side == Side.LONG:
                pos.current_target = pos.entry_price + self.tp_ticks * self.tick_size
            else:
                pos.current_target = pos.entry_price - self.tp_ticks * self.tick_size

        if self.trailing_stop_ticks is not None:
            # The trailing stop is live from the entry, not from the first new high.
            if pos.side == Side.LONG:
                pos.trailing_stop_level = pos.entry_price - self.trailing_stop_ticks * self.tick_size
                pos.current_stop = max(pos.current_stop, pos.trailing_stop_level)
            else:
                pos.trailing_stop_level = pos.entry_price + self.trailing_stop_ticks * self.tick_size
                pos.current_stop = min(pos.current_stop, pos.trailing_stop_level)

        pos.break_even_triggered = False

    def check(self, tick: Tick, pos: PositionState) -> ExitSignal:
        """
        Evaluate all risk conditions against the current tick.

        Called on **every tick** while a position is open.  Returns an
        ``ExitSignal`` with ``should_exit=True`` when a risk level is hit.

        The method first updates dynamic levels (trailing stop, break-even)
        and then checks exit conditions in priority order.
        """
        price = tick.price

        # --- update running MAE / MFE ---
        pos.update_extremes(price)

        # No level configured (the exit strategy owns TP and SL): every step below does nothing and
        # answers "stay". Only for this class itself: a subclass may have overridden a step.
        if (type(self) is RiskManager and self.sl_ticks is None and self.tp_ticks is None
                and self.trailing_stop_ticks is None and self.break_even_ticks is None):
            return ExitSignal(should_exit=False)

        # --- dynamic levels ---
        self._update_trailing_stop(price, pos)
        self._update_break_even(price, pos)

        # --- exit checks (priority order) ---
        # 1. Hard stop loss
        sl_signal = self._check_stop_loss(price, pos)
        if sl_signal.should_exit:
            return sl_signal

        # 2. Trailing stop
        ts_signal = self._check_trailing_stop(price, pos)
        if ts_signal.should_exit:
            return ts_signal

        # 3. Take profit
        tp_signal = self._check_take_profit(price, pos)
        if tp_signal.should_exit:
            return tp_signal

        return ExitSignal(should_exit=False)

    # ------------------------------------------------------------------ #
    #  Dynamic level updates                                              #
    # ------------------------------------------------------------------ #

    def _update_trailing_stop(self, price: float, pos: PositionState) -> None:
        if self.trailing_stop_ticks is None:
            return

        trail_distance = self.trailing_stop_ticks * self.tick_size

        if pos.side == Side.LONG:
            new_level = price - trail_distance
            if new_level > pos.trailing_stop_level:
                pos.trailing_stop_level = new_level
                # Also move the hard stop up if trailing surpasses it
                if pos.trailing_stop_level > pos.current_stop:
                    pos.current_stop = pos.trailing_stop_level
        else:  # SHORT
            new_level = price + trail_distance
            if new_level < pos.trailing_stop_level or pos.trailing_stop_level == 0:
                # For short, a *lower* trailing stop is better
                if pos.trailing_stop_level == 0:
                    pos.trailing_stop_level = new_level
                elif new_level < pos.trailing_stop_level:
                    pos.trailing_stop_level = new_level
                    if pos.trailing_stop_level < pos.current_stop:
                        pos.current_stop = pos.trailing_stop_level

    def _update_break_even(self, price: float, pos: PositionState) -> None:
        if self.break_even_ticks is None or pos.break_even_triggered:
            return

        be_distance = self.break_even_ticks * self.tick_size
        offset = self.break_even_offset_ticks * self.tick_size

        if pos.side == Side.LONG:
            if price >= pos.entry_price + be_distance:
                be_stop = pos.entry_price + offset
                if be_stop > pos.current_stop:
                    pos.current_stop = be_stop
                pos.break_even_triggered = True
        else:  # SHORT
            if price <= pos.entry_price - be_distance:
                be_stop = pos.entry_price - offset
                if be_stop < pos.current_stop:
                    pos.current_stop = be_stop
                pos.break_even_triggered = True

    # ------------------------------------------------------------------ #
    #  Exit condition checkers                                            #
    # ------------------------------------------------------------------ #

    def _check_stop_loss(self, price: float, pos: PositionState) -> ExitSignal:
        if self.sl_ticks is None and self.trailing_stop_ticks is None and self.break_even_ticks is None:
            return ExitSignal(should_exit=False)

        if pos.current_stop == 0:
            return ExitSignal(should_exit=False)

        if pos.side == Side.LONG and price <= pos.current_stop:
            trailed = self.trailing_stop_ticks is not None and price <= pos.trailing_stop_level
        elif pos.side == Side.SHORT and price >= pos.current_stop:
            trailed = self.trailing_stop_ticks is not None and price >= pos.trailing_stop_level
        else:
            return ExitSignal(should_exit=False)

        # The same label as the kernel: break-even once armed, else trailing if that level was hit.
        if pos.break_even_triggered:
            reason = ExitReason.BREAK_EVEN
        else:
            reason = ExitReason.TRAILING_STOP if trailed else ExitReason.STOP_LOSS
        return ExitSignal(should_exit=True, reason=reason, metadata={"stop_level": pos.current_stop})

    def _check_trailing_stop(self, price: float, pos: PositionState) -> ExitSignal:
        if self.trailing_stop_ticks is None:
            return ExitSignal(should_exit=False)

        if pos.side == Side.LONG and price <= pos.trailing_stop_level:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                metadata={"trailing_level": pos.trailing_stop_level},
            )
        if pos.side == Side.SHORT and price >= pos.trailing_stop_level:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                metadata={"trailing_level": pos.trailing_stop_level},
            )

        return ExitSignal(should_exit=False)

    def _check_take_profit(self, price: float, pos: PositionState) -> ExitSignal:
        if self.tp_ticks is None:
            return ExitSignal(should_exit=False)

        beyond = price - pos.current_target if pos.side == Side.LONG else pos.current_target - price
        if beyond > 0 or (self.fill_on_touch and beyond == 0):
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=pos.current_target,
                metadata={"target_level": pos.current_target},
            )

        return ExitSignal(should_exit=False)


def apply_daily_stop(trades: pd.DataFrame, *, max_stops: int = 2, stop_reason: str = "stop_loss") -> pd.DataFrame:
    """Drop every trade entered after the day's ``max_stops``-th stopped trade has exited.

    ``trades`` is the engine's ``trades_df`` (``entry_datetime``, ``exit_datetime``,
    ``exit_reason``). With one position at a time this is exactly what a daily stop inside
    the engine would produce: removing later trades never changes an earlier one, and any
    signal skipped while a removed trade was open came after the stop as well.
    """
    if trades.empty:
        return trades
    ordered = trades.sort_values("entry_datetime")
    day = pd.to_datetime(ordered["entry_datetime"]).dt.date
    is_stop = ordered["exit_reason"] == stop_reason
    last_allowed = is_stop & (is_stop.groupby(day).cumsum() == max_stops)
    if not last_allowed.any():
        return ordered.reset_index(drop=True)
    cutoff = day.map(ordered.loc[last_allowed].groupby(day[last_allowed])["exit_datetime"].first())
    return ordered[cutoff.isna() | (ordered["entry_datetime"] <= cutoff)].reset_index(drop=True)
