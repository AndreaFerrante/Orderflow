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

from dataclasses import dataclass
from typing import Optional

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

    Notes
    -----
    * All distances are in **ticks** (not price units) for clarity.
    * The manager mutates ``PositionState`` in-place for speed.
    * When multiple conditions trigger on the same tick, priority is:
      ``stop_loss > break_even > trailing_stop > take_profit``.
    """
    tp_ticks: Optional[float] = None
    sl_ticks: Optional[float] = None
    trailing_stop_ticks: Optional[float] = None
    break_even_ticks: Optional[float] = None
    break_even_offset_ticks: float = 0.0
    tick_size: float = 0.25
    tick_value: float = 12.5

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def initialize_position(self, pos: PositionState) -> None:
        """
        Called once on entry to set initial stop/target levels.
        """
        if self.sl_ticks is not None:
            if pos.side == Side.LONG:
                pos.current_stop = pos.entry_price - self.sl_ticks * self.tick_size
            else:
                pos.current_stop = pos.entry_price + self.sl_ticks * self.tick_size

        if self.tp_ticks is not None:
            if pos.side == Side.LONG:
                pos.current_target = pos.entry_price + self.tp_ticks * self.tick_size
            else:
                pos.current_target = pos.entry_price - self.tp_ticks * self.tick_size

        if self.trailing_stop_ticks is not None:
            if pos.side == Side.LONG:
                pos.trailing_stop_level = pos.entry_price - self.trailing_stop_ticks * self.tick_size
            else:
                pos.trailing_stop_level = pos.entry_price + self.trailing_stop_ticks * self.tick_size

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
            reason = ExitReason.BREAK_EVEN if pos.break_even_triggered else ExitReason.STOP_LOSS
            return ExitSignal(
                should_exit=True,
                reason=reason,
                exit_price=pos.current_stop,
                metadata={"stop_level": pos.current_stop},
            )
        if pos.side == Side.SHORT and price >= pos.current_stop:
            reason = ExitReason.BREAK_EVEN if pos.break_even_triggered else ExitReason.STOP_LOSS
            return ExitSignal(
                should_exit=True,
                reason=reason,
                exit_price=pos.current_stop,
                metadata={"stop_level": pos.current_stop},
            )

        return ExitSignal(should_exit=False)

    def _check_trailing_stop(self, price: float, pos: PositionState) -> ExitSignal:
        if self.trailing_stop_ticks is None:
            return ExitSignal(should_exit=False)

        if pos.side == Side.LONG and price <= pos.trailing_stop_level:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                exit_price=pos.trailing_stop_level,
                metadata={"trailing_level": pos.trailing_stop_level},
            )
        if pos.side == Side.SHORT and price >= pos.trailing_stop_level:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                exit_price=pos.trailing_stop_level,
                metadata={"trailing_level": pos.trailing_stop_level},
            )

        return ExitSignal(should_exit=False)

    def _check_take_profit(self, price: float, pos: PositionState) -> ExitSignal:
        if self.tp_ticks is None:
            return ExitSignal(should_exit=False)

        if pos.side == Side.LONG and price >= pos.current_target:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=pos.current_target,
                metadata={"target_level": pos.current_target},
            )
        if pos.side == Side.SHORT and price <= pos.current_target:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=pos.current_target,
                metadata={"target_level": pos.current_target},
            )

        return ExitSignal(should_exit=False)
