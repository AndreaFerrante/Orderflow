"""
Pluggable exit-strategy library (Strategy Pattern).

Every exit strategy inherits from ``BaseExitStrategy`` and implements:

.. code-block:: python

    def on_tick(self, tick, position, price_history, indicators) -> ExitSignal

The engine calls ``on_tick`` on every market tick while a position is open
*after* the ``RiskManager`` has been evaluated.  If the risk manager already
triggered an exit, the strategy is **not** called.

Users can combine multiple strategies via ``CompositeExit`` (first-to-fire
wins) or write entirely custom logic by subclassing ``BaseExitStrategy``.

Example
-------
>>> exit_logic = CompositeExit([
...     FixedTPSLExit(tp=10, sl=8),
...     TrailingStopExit(trail_ticks=6),
...     TimeBasedExit(max_ticks_in_trade=5000),
... ])
>>> engine = BacktestEngine(...)
>>> result = engine.run(data, signals, exit_strategy=exit_logic)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from orderflow.backtester.models import (
    ExitReason,
    ExitSignal,
    PositionState,
    Side,
    Tick,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseExitStrategy(abc.ABC):
    """
    Interface every exit strategy must implement.

    Parameters passed to ``on_tick``
    --------------------------------
    tick : Tick
        The current market tick.
    position : PositionState
        Mutable position state (read-only access recommended).
    price_history : np.ndarray
        Rolling window of recent prices (length configurable via engine).
    indicators : dict
        Arbitrary indicator values the user attached to each tick row.

    Returns
    -------
    ExitSignal
        ``should_exit=True`` to close the position.
    """

    @abc.abstractmethod
    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        ...

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        """
        Optional hook called once when a new position is opened.
        Override to initialise strategy-specific state.
        """
        pass

    def on_exit(self, tick: Tick, position: PositionState, reason: ExitReason) -> None:
        """
        Optional hook called after a position is closed.
        Override to perform cleanup or bookkeeping.
        """
        pass


# ---------------------------------------------------------------------------
# Fixed TP / SL
# ---------------------------------------------------------------------------

@dataclass
class FixedTPSLExit(BaseExitStrategy):
    """
    Classic fixed take-profit / stop-loss exit.

    Parameters
    ----------
    tp : float
        Take-profit distance in ticks.
    sl : float
        Stop-loss distance in ticks.
    tick_size : float
        Minimum price increment.

    Notes
    -----
    This strategy is largely redundant when the ``RiskManager`` is configured
    with ``tp_ticks`` / ``sl_ticks``, but is included for completeness and as
    a reference implementation.
    """
    tp: float = 10.0
    sl: float = 8.0
    tick_size: float = 0.25

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        price = tick.price
        entry = position.entry_price

        if position.side == Side.LONG:
            if price - entry >= self.tp * self.tick_size:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)
            if entry - price >= self.sl * self.tick_size:
                return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)

        elif position.side == Side.SHORT:
            if entry - price >= self.tp * self.tick_size:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)
            if price - entry >= self.sl * self.tick_size:
                return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)

        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# Trailing Stop (strategy-layer variant)
# ---------------------------------------------------------------------------

@dataclass
class TrailingStopExit(BaseExitStrategy):
    """
    Trail the stop by a fixed distance in ticks from the best price observed.

    Parameters
    ----------
    trail_ticks : float
        Distance to trail behind the extreme favorable price.
    tick_size : float
        Minimum price increment.
    activation_ticks : float | None
        If set, trailing only activates once the position is in profit
        by at least this many ticks.
    """
    trail_ticks: float = 6.0
    tick_size: float = 0.25
    activation_ticks: Optional[float] = None
    _trailing_level: float = field(default=0.0, init=False, repr=False)
    _activated: bool = field(default=False, init=False, repr=False)

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        self._trailing_level = 0.0
        self._activated = self.activation_ticks is None

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        price = tick.price
        trail_dist = self.trail_ticks * self.tick_size

        # Check activation
        if not self._activated and self.activation_ticks is not None:
            activation_dist = self.activation_ticks * self.tick_size
            if position.side == Side.LONG and price - position.entry_price >= activation_dist:
                self._activated = True
            elif position.side == Side.SHORT and position.entry_price - price >= activation_dist:
                self._activated = True

        if not self._activated:
            return ExitSignal(should_exit=False)

        if position.side == Side.LONG:
            new_level = price - trail_dist
            if new_level > self._trailing_level:
                self._trailing_level = new_level
            if price <= self._trailing_level:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    exit_price=self._trailing_level,
                )

        elif position.side == Side.SHORT:
            new_level = price + trail_dist
            if self._trailing_level == 0.0 or new_level < self._trailing_level:
                self._trailing_level = new_level
            if price >= self._trailing_level:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    exit_price=self._trailing_level,
                )

        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# Break-Even Stop (strategy-layer variant)
# ---------------------------------------------------------------------------

@dataclass
class BreakEvenExit(BaseExitStrategy):
    """
    Move the stop to break-even once price moves favorably by *threshold*.

    Parameters
    ----------
    activation_ticks : float
        Ticks in-the-money before the break-even triggers.
    offset_ticks : float
        Extra offset from entry to cover costs.
    tick_size : float
        Minimum price increment.
    """
    activation_ticks: float = 5.0
    offset_ticks: float = 0.0
    tick_size: float = 0.25
    _triggered: bool = field(default=False, init=False, repr=False)
    _be_stop: float = field(default=0.0, init=False, repr=False)

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        self._triggered = False
        self._be_stop = 0.0

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        price = tick.price
        act_dist = self.activation_ticks * self.tick_size
        offset = self.offset_ticks * self.tick_size

        if not self._triggered:
            if position.side == Side.LONG and price >= position.entry_price + act_dist:
                self._triggered = True
                self._be_stop = position.entry_price + offset
            elif position.side == Side.SHORT and price <= position.entry_price - act_dist:
                self._triggered = True
                self._be_stop = position.entry_price - offset

        if self._triggered:
            if position.side == Side.LONG and price <= self._be_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.BREAK_EVEN,
                    exit_price=self._be_stop,
                )
            if position.side == Side.SHORT and price >= self._be_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.BREAK_EVEN,
                    exit_price=self._be_stop,
                )

        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------

@dataclass
class TimeBasedExit(BaseExitStrategy):
    """
    Close the position after a maximum number of ticks (or bars) in trade.

    Parameters
    ----------
    max_ticks_in_trade : int
        Maximum duration in ticks/bars.
    """
    max_ticks_in_trade: int = 5000

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        if position.ticks_in_trade >= self.max_ticks_in_trade:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_EXIT,
                metadata={"ticks_in_trade": position.ticks_in_trade},
            )
        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# Volatility-based exit
# ---------------------------------------------------------------------------

@dataclass
class VolatilityExit(BaseExitStrategy):
    """
    Close when recent realised volatility exceeds a threshold.

    Uses the standard deviation of the last *window* prices.

    Parameters
    ----------
    vol_threshold : float
        Maximum tolerated standard deviation (in price units).
    window : int
        Lookback length for the rolling standard deviation.
    """
    vol_threshold: float = 2.0
    window: int = 100

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        if len(price_history) < self.window:
            return ExitSignal(should_exit=False)

        recent = price_history[-self.window:]
        vol = float(np.std(recent))
        if vol > self.vol_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.VOLATILITY_EXIT,
                metadata={"realised_vol": vol, "threshold": self.vol_threshold},
            )
        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# Composite (first-to-fire wins)
# ---------------------------------------------------------------------------

@dataclass
class CompositeExit(BaseExitStrategy):
    """
    Combine multiple exit strategies.  The **first** strategy to fire wins.

    Parameters
    ----------
    strategies : list[BaseExitStrategy]
        Ordered list of strategies to evaluate.

    Example
    -------
    >>> combo = CompositeExit([
    ...     FixedTPSLExit(tp=10, sl=8),
    ...     TrailingStopExit(trail_ticks=6),
    ...     TimeBasedExit(max_ticks_in_trade=5000),
    ... ])
    """
    strategies: List[BaseExitStrategy] = field(default_factory=list)

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        for s in self.strategies:
            s.on_entry(tick, position)

    def on_exit(self, tick: Tick, position: PositionState, reason: ExitReason) -> None:
        for s in self.strategies:
            s.on_exit(tick, position, reason)

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        for strategy in self.strategies:
            signal = strategy.on_tick(tick, position, price_history, indicators)
            if signal.should_exit:
                return signal
        return ExitSignal(should_exit=False)
