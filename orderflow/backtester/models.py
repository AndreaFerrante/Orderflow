"""
Data models, enums, and type definitions for the backtesting engine.

All value objects are intentionally kept immutable (frozen dataclasses / enums)
to prevent accidental mutation during the hot-loop.  ``PositionState`` is the
sole mutable structure — it tracks the *live* position and is updated in-place
by the risk manager for performance.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(enum.IntEnum):
    """Trade direction. IntEnum so it can be used directly in Numba paths."""
    SHORT = -1
    FLAT  = 0
    LONG  = 1


class ExitReason(enum.Enum):
    """Why a position was closed."""
    TAKE_PROFIT     = "take_profit"
    STOP_LOSS       = "stop_loss"
    TRAILING_STOP   = "trailing_stop"
    BREAK_EVEN      = "break_even"
    TIME_EXIT       = "time_exit"
    VOLATILITY_EXIT = "volatility_exit"
    CUSTOM          = "custom"
    SESSION_CLOSE   = "session_close"
    END_OF_DATA     = "end_of_data"


# ---------------------------------------------------------------------------
# Immutable value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExitSignal:
    """
    Returned by an exit strategy to instruct the engine to close a position.

    Attributes
    ----------
    should_exit : bool
        ``True`` if the position should be closed **now**.
    reason : ExitReason
        Categorical reason for the exit.
    exit_price : float | None
        Override price (e.g. for limit-style exits).  When ``None`` the engine
        uses the current tick price adjusted for slippage.
    metadata : dict
        Arbitrary key/value bag for strategy-specific diagnostics.
    """
    should_exit: bool = False
    reason: ExitReason = ExitReason.CUSTOM
    exit_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Tick:
    """
    Single market tick consumed by the engine.

    Attributes
    ----------
    index : int
        Monotonically increasing row ordinal (from the source DataFrame).
    timestamp : np.int64
        Nanosecond-precision epoch or ordinal index for fast comparison.
    datetime : Any
        Human-readable datetime (kept for reporting; not used in hot loop).
    price : float
        Last traded price at this tick.
    bid : float
        Best bid price (use ``price`` if unavailable).
    ask : float
        Best ask price (use ``price`` if unavailable).
    volume : float
        Tick volume (can be 0 for quote-only ticks).
    date : Any
        Date component (kept for session management).
    session_type : str
        ``"RTH"`` / ``"ETH"`` / ``""`` for session filtering.
    extra : dict
        Any additional columns the user wants to carry along.
    """
    index: int
    timestamp: np.int64
    datetime: Any
    price: float
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    date: Any = None
    session_type: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """
    Immutable run configuration.

    Attributes
    ----------
    tick_size : float
        Minimum price increment (e.g. 0.25 for ES futures).
    tick_value : float
        Dollar value of one tick (e.g. 12.50 for ES futures).
    commission : float
        Round-trip commission **per contract** in dollars.
    n_contracts : int
        Number of contracts per entry.
    trade_in_rth : bool
        If ``True``, force-close positions that drift outside RTH hours.
    seed : int | None
        RNG seed for reproducible slippage simulation.
    """
    tick_size: float = 0.25
    tick_value: float = 12.5
    commission: float = 4.5
    n_contracts: int = 1
    trade_in_rth: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Mutable position tracker
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PositionState:
    """
    Mutable state for the *currently open* position.

    The risk manager mutates ``current_stop``, ``current_target``,
    ``trailing_stop_level``, and ``break_even_triggered`` on every tick.
    """
    side: Side = Side.FLAT
    entry_price: float = 0.0
    entry_price_pure: float = 0.0  # before slippage
    entry_tick_idx: int = 0
    entry_timestamp: np.int64 = np.int64(0)
    entry_datetime: Any = None

    # Risk levels (set by exit strategy / risk manager)
    current_stop: float = 0.0
    current_target: float = 0.0
    trailing_stop_level: float = 0.0
    break_even_triggered: bool = False

    # Running extremes for MAE/MFE
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    ticks_in_trade: int = 0

    def reset(self) -> None:
        """Return to flat — called after every trade close."""
        self.side = Side.FLAT
        self.entry_price = 0.0
        self.entry_price_pure = 0.0
        self.entry_tick_idx = 0
        self.entry_timestamp = np.int64(0)
        self.entry_datetime = None
        self.current_stop = 0.0
        self.current_target = 0.0
        self.trailing_stop_level = 0.0
        self.break_even_triggered = False
        self.max_favorable_price = 0.0
        self.max_adverse_price = 0.0
        self.ticks_in_trade = 0

    @property
    def is_flat(self) -> bool:
        return self.side == Side.FLAT

    def update_extremes(self, price: float) -> None:
        """Track running MAE / MFE prices (called every tick while in-trade)."""
        self.ticks_in_trade += 1
        if self.side == Side.LONG:
            if price > self.max_favorable_price or self.ticks_in_trade == 1:
                self.max_favorable_price = price
            if price < self.max_adverse_price or self.ticks_in_trade == 1:
                self.max_adverse_price = price
        elif self.side == Side.SHORT:
            if price < self.max_favorable_price or self.ticks_in_trade == 1:
                self.max_favorable_price = price
            if price > self.max_adverse_price or self.ticks_in_trade == 1:
                self.max_adverse_price = price


# ---------------------------------------------------------------------------
# Trade record (one per closed trade)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TradeRecord:
    """
    Immutable record written once a position is fully closed.
    """
    trade_id: int
    side: Side
    entry_datetime: Any
    exit_datetime: Any
    entry_timestamp: np.int64
    exit_timestamp: np.int64
    entry_price: float
    entry_price_pure: float
    exit_price: float
    pnl_ticks: float
    pnl_dollars: float
    commission: float
    net_pnl: float
    exit_reason: ExitReason
    mae_ticks: float
    mfe_ticks: float
    ticks_in_trade: int
    metadata: Dict[str, Any] = field(default_factory=dict)
