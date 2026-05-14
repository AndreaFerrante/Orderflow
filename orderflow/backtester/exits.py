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
from datetime import datetime
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


@dataclass
class DynamicTPSLExit(BaseExitStrategy):
    """
    Exit strategy that uses per-signal TP and SL values from dedicated columns.

    Expects signals DataFrame to have columns: Index, TradeType, TP_Ticks, SL_Ticks
    """
    signals_df: pd.DataFrame
    tick_size: float = 0.25
    _signal_lookup: Dict[int, tuple] = field(default_factory=dict, init=False, repr=False)
    _current_tp: float = field(default=None, init=False, repr=False)
    _current_sl: float = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Build a lookup dict: signal Index -> (TP_Ticks, SL_Ticks)
        for _, row in self.signals_df.iterrows():
            self._signal_lookup[int(row['Index'])] = (
                float(row['TP_Ticks']),
                float(row['SL_Ticks'])
            )

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        """Called when position opens — retrieve TP/SL for this signal."""
        if tick.index in self._signal_lookup:
            self._current_tp, self._current_sl = self._signal_lookup[tick.index]

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        price = tick.price
        entry = position.entry_price

        if self._current_tp is None or self._current_sl is None:
            return ExitSignal(should_exit=False)

        tp_distance = self._current_tp * self.tick_size
        sl_distance = self._current_sl * self.tick_size

        if position.side == Side.LONG:
            if price - entry >= tp_distance:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)
            if entry - price >= sl_distance:
                return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)

        elif position.side == Side.SHORT:
            if entry - price >= tp_distance:
                return ExitSignal(should_exit=True, reason=ExitReason.TAKE_PROFIT)
            if price - entry >= sl_distance:
                return ExitSignal(should_exit=True, reason=ExitReason.STOP_LOSS)

        return ExitSignal(should_exit=False)


# ---------------------------------------------------------------------------
# CVD-confirmed break-even
# ---------------------------------------------------------------------------

@dataclass
class CVDBreakEvenExit(BaseExitStrategy):
    """
    Move the stop to break-even once session CVD confirms the trade direction.

    Activation logic (one-shot, never reverts) — **both** gates must open on
    the same tick:

    * **Gate 1 — minimum profit**: price has moved at least
      ``min_profit_ticks * tick_size`` favourably from entry.
    * **Gate 2 — CVD confirmation**: current CVD has shifted by at least
      ``cvd_delta_threshold`` beyond the baseline CVD recorded at the trigger
      tick (long: increase; short: decrease).

    Once activated the break-even stop is fixed at
    ``entry_price ± offset_ticks * tick_size`` and never reverts.  If neither
    gate ever opens, the original TP/SL (from another strategy in the
    composite) remains in force.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Signals DataFrame **as passed to the engine** (after ``drop_nulls``).
        Must contain columns ``'Index'`` (entry index) and ``cvd_col``
        (session CVD at the trigger tick — used as the per-trade baseline).
    cvd_col : str
        Name of the CVD column in both ``signals_df`` and the ``indicators``
        dict.  The engine populates ``indicators`` from the tick data when
        ``indicator_columns=[cvd_col]`` is passed to ``engine.run()``.
    min_profit_ticks : float
        Minimum favourable price move in ticks before CVD is even evaluated.
        Prevents premature break-even activation immediately after entry.
    cvd_delta_threshold : float
        Minimum CVD shift beyond the signal baseline required for activation.
        ``0.0`` activates on *any* confirming movement.
    offset_ticks : float
        Extra ticks above/below entry price for the break-even stop.
        ``0.0`` = exact break-even (no profit lock-in).
    tick_size : float
        Minimum price increment.

    Notes
    -----
    Place this strategy *after* ``DynamicTPSLExit`` (or similar) and *before*
    ``HourBasedExit`` inside ``CompositeExit``.  Because the break-even stop
    sits between the entry price and the original SL, it fires before the SL
    once activated, without interfering with take-profit logic.

    ``indicator_columns=[cvd_col]`` **must** be passed to ``engine.run()`` and
    the tick-data DataFrame must contain that column.

    Example
    -------
    >>> exit_strategy = CompositeExit([
    ...     DynamicTPSLExit(signals_df=signals, tick_size=0.25),
    ...     CVDBreakEvenExit(signals_df=signals, cvd_col="CVD", min_profit_ticks=4),
    ...     HourBasedExit(close_hour=15, close_minute=0),
    ... ])
    >>> result = engine.run(data, signals, exit_strategy=exit_strategy,
    ...                     indicator_columns=["CVD"])
    """

    signals_df: pd.DataFrame
    cvd_col: str = "CVD"
    min_profit_ticks: float = 4.0
    cvd_delta_threshold: float = 0.0
    offset_ticks: float = 0.0
    tick_size: float = 0.25

    _baseline_lookup: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _baseline_cvd: float = field(default=0.0, init=False, repr=False)
    _activated: bool = field(default=False, init=False, repr=False)
    _be_stop: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        for _, row in self.signals_df.iterrows():
            self._baseline_lookup[int(row["Index"])] = float(row[self.cvd_col])

    def on_entry(self, tick: Tick, position: PositionState) -> None:
        self._baseline_cvd = self._baseline_lookup.get(tick.index, float("nan"))
        self._activated = False
        self._be_stop = 0.0

    def on_tick(
        self,
        tick: Tick,
        position: PositionState,
        price_history: np.ndarray,
        indicators: Dict[str, Any],
    ) -> ExitSignal:
        current_cvd = indicators.get(self.cvd_col)
        if current_cvd is None or np.isnan(self._baseline_cvd):
            return ExitSignal(should_exit=False)

        # Try to activate (one-shot, irreversible)
        # Gate 1: minimum profit reached; Gate 2: CVD confirms direction
        if not self._activated:
            min_move = self.min_profit_ticks * self.tick_size
            offset = self.offset_ticks * self.tick_size
            if (
                position.side == Side.LONG
                and tick.price >= position.entry_price + min_move
                and current_cvd > self._baseline_cvd + self.cvd_delta_threshold
            ):
                self._activated = True
                position.break_even_triggered = True
                self._be_stop = position.entry_price + offset
            elif (
                position.side == Side.SHORT
                and tick.price <= position.entry_price - min_move
                and current_cvd < self._baseline_cvd - self.cvd_delta_threshold
            ):
                self._activated = True
                position.break_even_triggered = True
                self._be_stop = position.entry_price - offset

        # Check BE stop once activated
        if self._activated:
            if position.side == Side.LONG and tick.price <= self._be_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.BREAK_EVEN,
                    exit_price=self._be_stop,
                    metadata={
                        "break_even_activated": True,
                        "baseline_cvd": self._baseline_cvd,
                        "trigger_cvd": float(current_cvd),
                    },
                )
            if position.side == Side.SHORT and tick.price >= self._be_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.BREAK_EVEN,
                    exit_price=self._be_stop,
                    metadata={
                        "break_even_activated": True,
                        "baseline_cvd": self._baseline_cvd,
                        "trigger_cvd": float(current_cvd),
                    },
                )

        return ExitSignal(should_exit=False)


@dataclass
class HourBasedExit(BaseExitStrategy):
    """
    Close all positions at a specific hour of the day.

    Parameters
    ----------
    close_hour : int
        Hour (0-23) to force close all open positions.
    close_minute : int
        Minute (0-59) to force close.
    """
    close_hour: int = 15
    close_minute: int = 0

    def on_tick(
            self,
            tick: Tick,
            position: PositionState,
            price_history: np.ndarray,
            indicators: Dict[str, Any],
    ) -> ExitSignal:
        raw = tick.datetime
        if isinstance(raw, (int, np.int64, np.int32)):
            raw_min = int(raw) // (60 * 1_000_000_000)  # nanosecondi → minuti
            tick_hour = raw_min // 60 % 24
            tick_minute = raw_min % 60

        elif isinstance(raw, np.datetime64):
            raw_min = raw.astype('datetime64[m]').astype(np.int64)  # minuti da epoch
            tick_hour = int(raw_min // 60 % 24)
            tick_minute = int(raw_min % 60)

        else:
            tick_hour = raw.hour
            tick_minute = raw.minute

        close_time_minutes = self.close_hour * 60 + self.close_minute
        tick_time_minutes = tick_hour * 60 + tick_minute

        if tick_time_minutes >= close_time_minutes:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_EXIT,
                metadata={"close_time": f"{self.close_hour}:{self.close_minute:02d}"},
            )
        return ExitSignal(should_exit=False)
