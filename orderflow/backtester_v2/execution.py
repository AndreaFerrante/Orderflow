"""
Slippage and fill-simulation models.

The execution layer sits between the signal generator and the position
manager.  Its job is to turn an *ideal* fill price into a *realistic*
fill price by accounting for:

* Random or deterministic slippage (in ticks)
* Bid/ask spread crossing
* Partial-fill modelling (future extension)

Design
------
``SlippageModel`` is a lightweight callable that returns the number of
ticks of slippage for a given fill attempt.  ``FillSimulator`` composes a
``SlippageModel`` with the current tick data to produce a realised entry
or exit price.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from orderflow.backtester_v2.models import Side, Tick


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------

class SlippageMode(enum.Enum):
    FIXED    = "fixed"
    UNIFORM  = "uniform"
    GAUSSIAN = "gaussian"
    ZERO     = "zero"


@dataclass(slots=True)
class SlippageModel:
    """
    Configurable slippage model.

    Parameters
    ----------
    mode : SlippageMode
        How slippage is drawn.
    max_ticks : int
        Upper bound for uniform / fixed; std-dev for gaussian.
    fixed_ticks : int
        Exact slippage used when ``mode == FIXED``.
    seed : int | None
        RNG seed for reproducibility.

    Examples
    --------
    >>> sm = SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=2, seed=42)
    >>> sm.sample()          # returns 0, 1, or 2
    >>> sm_zero = SlippageModel()  # default: zero slippage
    >>> sm_zero.sample()     # always 0
    """
    mode: SlippageMode = SlippageMode.ZERO
    max_ticks: int = 0
    fixed_ticks: int = 0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self) -> int:
        """Return a non-negative integer number of ticks of slippage."""
        if self.mode == SlippageMode.ZERO:
            return 0
        if self.mode == SlippageMode.FIXED:
            return max(0, self.fixed_ticks)
        if self.mode == SlippageMode.UNIFORM:
            return int(self._rng.integers(0, self.max_ticks + 1))
        if self.mode == SlippageMode.GAUSSIAN:
            return max(0, int(round(abs(self._rng.normal(0, self.max_ticks)))))
        return 0  # pragma: no cover

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Fill simulator
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FillSimulator:
    """
    Combines a ``SlippageModel`` with tick-level data to compute
    realistic fill prices.

    Parameters
    ----------
    slippage_model : SlippageModel
        The slippage model to apply on every fill.
    tick_size : float
        Minimum price increment of the instrument.
    use_spread : bool
        If ``True`` and bid/ask are available, entries cross the spread
        (buys fill at ask, sells fill at bid) *before* slippage is added.

    Notes
    -----
    Slippage always works **against** the trader:
    - LONG entry  → price goes UP   by slippage
    - SHORT entry → price goes DOWN by slippage
    - LONG exit   → price goes DOWN by slippage
    - SHORT exit  → price goes UP   by slippage
    """
    slippage_model: SlippageModel
    tick_size: float = 0.25
    use_spread: bool = False

    def fill_entry(self, tick: Tick, side: Side) -> tuple[float, float, int]:
        """
        Compute the realised entry price.

        Returns
        -------
        (fill_price, pure_price, slippage_ticks) : tuple
            ``pure_price`` is the price *before* slippage (for reporting).
        """
        slippage_ticks = self.slippage_model.sample()
        slippage_amount = slippage_ticks * self.tick_size

        if self.use_spread and tick.ask > 0 and tick.bid > 0:
            base_price = tick.ask if side == Side.LONG else tick.bid
        else:
            base_price = tick.price

        pure_price = base_price

        if side == Side.LONG:
            fill_price = base_price + slippage_amount
        else:  # SHORT
            fill_price = base_price - slippage_amount

        return fill_price, pure_price, slippage_ticks

    def fill_exit(self, tick: Tick, side: Side) -> tuple[float, int]:
        """
        Compute the realised exit price.

        Returns
        -------
        (fill_price, slippage_ticks) : tuple
        """
        slippage_ticks = self.slippage_model.sample()
        slippage_amount = slippage_ticks * self.tick_size

        if self.use_spread and tick.ask > 0 and tick.bid > 0:
            # Exit crosses the spread in the opposite direction
            base_price = tick.bid if side == Side.LONG else tick.ask
        else:
            base_price = tick.price

        if side == Side.LONG:
            # Selling to close — slippage pushes price down
            fill_price = base_price - slippage_amount
        else:  # SHORT
            # Buying to cover — slippage pushes price up
            fill_price = base_price + slippage_amount

        return fill_price, slippage_ticks
