"""
Core backtesting engine — tick-by-tick execution loop.

Architecture
~~~~~~~~~~~~
The engine orchestrates data ingestion, signal matching, position lifecycle,
risk management, and exit-strategy evaluation.
It is deliberately *not* a monolithic function: each concern lives in its own 
module and the engine merely coordinates them.

Hot-path optimisation
~~~~~~~~~~~~~~~~~~~~~
1. **Pre-materialised NumPy arrays** — all tick data is extracted once into
   contiguous arrays before the loop starts.
2. **Numba JIT inner loop** for the pure-TP/SL path (``_numba_core_loop``).
   When the user supplies a custom ``BaseExitStrategy`` the engine falls
   back to the pure-Python path which is still vectorised where possible.
3. **Boolean entry mask** — ``np.isin`` precomputes an O(1) lookup array
   so the inner loop never searches for signal timestamps.
4. Minimal object creation inside the hot loop — ``Tick`` objects are only
   constructed when needed (custom exit strategies).

Usage
-----
>>> from orderflow.backtester import BacktestEngine, FixedTPSLExit
>>> engine = BacktestEngine(tick_size=0.25, tick_value=12.5)
>>> result = engine.run(data, signals, exit_strategy=FixedTPSLExit(tp=10, sl=8))
>>> print(result.metrics.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from orderflow.backtester.execution import (
    FillSimulator,
    SlippageModel,
)
from orderflow.backtester.exits import BaseExitStrategy
from orderflow.backtester.metrics import PerformanceMetrics, compute_metrics
from orderflow.backtester.models import (
    BacktestConfig,
    ExitReason,
    ExitSignal,
    PositionState,
    Side,
    Tick,
    TradeRecord,
)
from orderflow.backtester.risk import RiskManager

# Optional Numba import — graceful degradation
try:
    from numba import njit  # type: ignore[import-untyped]
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is not installed."""
        def _wrapper(fn):  # type: ignore[return]
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrapper


# ======================================================================== #
#  Numba-accelerated inner loop (pure TP/SL — no custom strategy)          #
# ======================================================================== #

@njit(cache=True)
def _numba_core_loop(
    prices: np.ndarray,
    timestamps: np.ndarray,
    entry_mask: np.ndarray,
    signal_sides: np.ndarray,
    slippage_ticks_arr: np.ndarray,
    tick_size: float,
    tp_ticks: float,
    sl_ticks: float,
    trailing_stop_ticks: float,
    break_even_ticks: float,
    break_even_offset_ticks: float,
    tick_value: float,
    commission: float,
    n_contracts: int,
) -> tuple:
    """
    Numba-compiled inner loop for maximum throughput.

    Handles TP, SL, trailing stop, and break-even mechanics.  Does **not**
    support custom ``BaseExitStrategy`` callables — for that the engine falls
    back to the pure-Python loop.

    Returns parallel arrays that are assembled into ``TradeRecord`` objects by
    the caller.
    """
    n = len(prices)

    # Pre-allocate output arrays (upper bound = number of signals)
    max_trades = int(entry_mask.sum()) + 1
    out_entry_idx     = np.empty(max_trades, dtype=np.int64)
    out_exit_idx      = np.empty(max_trades, dtype=np.int64)
    out_entry_ts      = np.empty(max_trades, dtype=np.int64)
    out_exit_ts       = np.empty(max_trades, dtype=np.int64)
    out_entry_price   = np.empty(max_trades, dtype=np.float64)
    out_entry_pure    = np.empty(max_trades, dtype=np.float64)
    out_exit_price    = np.empty(max_trades, dtype=np.float64)
    out_side          = np.empty(max_trades, dtype=np.int64)
    out_exit_reason   = np.empty(max_trades, dtype=np.int64)
    out_mae           = np.empty(max_trades, dtype=np.float64)
    out_mfe           = np.empty(max_trades, dtype=np.float64)
    out_ticks_in      = np.empty(max_trades, dtype=np.int64)

    # Exit reason codes: 0=TP, 1=SL, 2=trailing, 3=break-even, 4=EOD
    trade_count = 0
    signal_ptr  = 0

    # Position state
    in_position   = False
    side          = 0        # +1 LONG, -1 SHORT
    entry_price   = 0.0
    entry_pure    = 0.0
    entry_ts      = np.int64(0)
    entry_row_idx = 0
    current_stop  = 0.0
    current_target = 0.0
    trailing_level = 0.0
    be_triggered   = False
    max_fav_price  = 0.0
    max_adv_price  = 0.0
    ticks_count    = 0

    use_trailing = trailing_stop_ticks > 0
    use_be       = break_even_ticks > 0

    for i in range(n):
        price = prices[i]
        ts    = timestamps[i]

        # --- Try to enter ---
        if not in_position and entry_mask[i] and signal_ptr < len(signal_sides):
            side_val = signal_sides[signal_ptr]
            slip = slippage_ticks_arr[signal_ptr] * tick_size

            entry_pure = price
            if side_val == 1:   # SHORT
                entry_price = price - slip
                side = -1
            else:               # LONG
                entry_price = price + slip
                side = 1

            entry_ts      = ts
            entry_row_idx = i
            in_position   = True
            be_triggered  = False
            ticks_count   = 0
            max_fav_price = price
            max_adv_price = price

            # Initial levels
            if side == 1:
                current_stop   = entry_price - sl_ticks * tick_size
                current_target = entry_price + tp_ticks * tick_size
                if use_trailing:
                    trailing_level = entry_price - trailing_stop_ticks * tick_size
            else:
                current_stop   = entry_price + sl_ticks * tick_size
                current_target = entry_price - tp_ticks * tick_size
                if use_trailing:
                    trailing_level = entry_price + trailing_stop_ticks * tick_size

            signal_ptr += 1
            continue

        if not in_position:
            # Advance signal pointer past stale signals
            if entry_mask[i] and signal_ptr < len(signal_sides):
                signal_ptr += 1
            continue

        # --- In position: update extremes ---
        ticks_count += 1
        if side == 1:
            if price > max_fav_price:
                max_fav_price = price
            if price < max_adv_price:
                max_adv_price = price
        else:
            if price < max_fav_price:
                max_fav_price = price
            if price > max_adv_price:
                max_adv_price = price

        # --- Update trailing stop ---
        if use_trailing:
            trail_dist = trailing_stop_ticks * tick_size
            if side == 1:
                new_trail = price - trail_dist
                if new_trail > trailing_level:
                    trailing_level = new_trail
                    if trailing_level > current_stop:
                        current_stop = trailing_level
            else:
                new_trail = price + trail_dist
                if new_trail < trailing_level:
                    trailing_level = new_trail
                    if trailing_level < current_stop:
                        current_stop = trailing_level

        # --- Update break-even ---
        if use_be and not be_triggered:
            be_dist = break_even_ticks * tick_size
            offset  = break_even_offset_ticks * tick_size
            if side == 1 and price >= entry_price + be_dist:
                be_stop = entry_price + offset
                if be_stop > current_stop:
                    current_stop = be_stop
                be_triggered = True
            elif side == -1 and price <= entry_price - be_dist:
                be_stop = entry_price - offset
                if be_stop < current_stop:
                    current_stop = be_stop
                be_triggered = True

        # --- Check exits ---
        exited = False
        exit_reason = 4  # default EOD

        if side == 1:
            if price <= current_stop:
                exited = True
                exit_reason = 3 if be_triggered else (2 if use_trailing and price <= trailing_level else 1)
            elif price >= current_target:
                exited = True
                exit_reason = 0
        else:
            if price >= current_stop:
                exited = True
                exit_reason = 3 if be_triggered else (2 if use_trailing and price >= trailing_level else 1)
            elif price <= current_target:
                exited = True
                exit_reason = 0

        if exited:
            idx = trade_count
            out_entry_idx[idx]   = entry_row_idx
            out_exit_idx[idx]    = i
            out_entry_ts[idx]    = entry_ts
            out_exit_ts[idx]     = ts
            out_entry_price[idx] = entry_price
            out_entry_pure[idx]  = entry_pure
            out_exit_price[idx]  = price
            out_side[idx]        = side
            out_exit_reason[idx] = exit_reason

            # MAE / MFE in ticks
            if side == 1:
                out_mae[idx] = (entry_price - max_adv_price) / tick_size
                out_mfe[idx] = (max_fav_price - entry_price) / tick_size
            else:
                out_mae[idx] = (max_adv_price - entry_price) / tick_size
                out_mfe[idx] = (entry_price - max_fav_price) / tick_size

            out_ticks_in[idx] = ticks_count
            trade_count += 1
            in_position = False

    # --- Close dangling position at end of data ---
    if in_position:
        idx = trade_count
        last_price = prices[n - 1]
        out_entry_idx[idx]   = entry_row_idx
        out_exit_idx[idx]    = n - 1
        out_entry_ts[idx]    = entry_ts
        out_exit_ts[idx]     = timestamps[n - 1]
        out_entry_price[idx] = entry_price
        out_entry_pure[idx]  = entry_pure
        out_exit_price[idx]  = last_price
        out_side[idx]        = side
        out_exit_reason[idx] = 4  # END_OF_DATA

        if side == 1:
            out_mae[idx] = (entry_price - max_adv_price) / tick_size
            out_mfe[idx] = (max_fav_price - entry_price) / tick_size
        else:
            out_mae[idx] = (max_adv_price - entry_price) / tick_size
            out_mfe[idx] = (entry_price - max_fav_price) / tick_size

        out_ticks_in[idx] = ticks_count
        trade_count += 1

    # Trim to actual trade count
    return (
        out_entry_idx[:trade_count],
        out_exit_idx[:trade_count],
        out_entry_ts[:trade_count],
        out_exit_ts[:trade_count],
        out_entry_price[:trade_count],
        out_entry_pure[:trade_count],
        out_exit_price[:trade_count],
        out_side[:trade_count],
        out_exit_reason[:trade_count],
        out_mae[:trade_count],
        out_mfe[:trade_count],
        out_ticks_in[:trade_count],
    )


# ======================================================================== #
#  Result container                                                        #
# ======================================================================== #

@dataclass
class BacktestResult:
    """
    Container returned by ``BacktestEngine.run()``.

    Attributes
    ----------
    trades : list[TradeRecord]
        Individual trade records.
    trades_df : pd.DataFrame
        Trades in tabular form for analysis / export.
    metrics : PerformanceMetrics
        Aggregated performance statistics.
    config : BacktestConfig
        The configuration used for this run.
    """
    trades: List[TradeRecord] = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    def summary(self) -> str:
        """Print and return the performance summary."""
        s = self.metrics.summary()
        print(s)
        return s


# ======================================================================== #
#  Engine                                                                  #
# ======================================================================== #

class BacktestEngine:
    """
    Professional-grade tick-by-tick backtesting engine.

    Parameters
    ----------
    tick_size : float
        Minimum price increment (e.g. 0.25 for ES).
    tick_value : float
        Dollar value per tick (e.g. 12.50 for ES).
    commission : float
        Commission per contract per round-trip ($).
    n_contracts : int
        Default number of contracts per signal.
    slippage_model : SlippageModel | None
        Slippage model to use for fill simulation.
    use_spread : bool
        If ``True`` and bid/ask columns are available, fill prices cross
        the spread before slippage is applied.
    trade_in_rth : bool
        Force-close (or prevent) entries outside RTH session.
    price_history_window : int
        Number of recent prices passed to exit strategies (for rolling
        volatility, etc.).
    seed : int | None
        Master RNG seed for reproducibility (overrides slippage model seed).
    progress_bar : bool
        Show ``tqdm`` progress bar during the loop.

    Examples
    --------
    >>> engine = BacktestEngine(
    ...     tick_size=0.25,
    ...     tick_value=12.5,
    ...     commission=4.5,
    ...     slippage_model=SlippageModel(mode=SlippageMode.UNIFORM, max_ticks=2, seed=42),
    ... )
    >>> result = engine.run(data, signals, exit_strategy=FixedTPSLExit(tp=10, sl=8))
    >>> result.summary()
    """

    def __init__(
        self,
        tick_size: float = 0.25,
        tick_value: float = 12.5,
        commission: float = 4.5,
        n_contracts: int = 1,
        slippage_model: Optional[SlippageModel] = None,
        use_spread: bool = False,
        trade_in_rth: bool = False,
        price_history_window: int = 200,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.commission = commission
        self.n_contracts = n_contracts
        self.slippage_model = slippage_model or SlippageModel()
        self.use_spread = use_spread
        self.trade_in_rth = trade_in_rth
        self.price_history_window = price_history_window
        self.seed = seed
        self.progress_bar = progress_bar

        if seed is not None:
            self.slippage_model.reseed(seed)

        self._fill_sim = FillSimulator(
            slippage_model=self.slippage_model,
            tick_size=self.tick_size,
            use_spread=self.use_spread,
        )

    # ------------------------------------------------------------------ #
    #  Data validation & preparation                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        required = {"Index", "Datetime", "Price", "Date", "Time", "SessionType"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input data is missing required columns: {missing}. "
                f"Expected columns: {sorted(required)}"
            )

    @staticmethod
    def _validate_signals(signal: pd.DataFrame) -> None:
        required = {"Index", "TradeType"}
        missing = required - set(signal.columns)
        if missing:
            raise ValueError(
                f"Signal DataFrame is missing required columns: {missing}. "
                f"Expected columns: {sorted(required)}"
            )

    def _filter_rth_signals(
        self, data: pd.DataFrame, datetime_signal: np.ndarray, signal_sides: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove signals that fall outside RTH hours.
        """
        if "SessionType" not in data.columns:
            raise ValueError(
                "trade_in_rth=True requires a 'SessionType' column in the data DataFrame."
            )

        data_pl = pl.from_pandas(data[["SessionType", "Date", "Index"]])
        
        rth = (
            data_pl.
            filter(pl.col("SessionType") == "RTH")
            .with_columns(
                IndexFirst=pl.col("Index"),
                IndexLast=pl.col("Index"),
            )
            .group_by(["Date", "SessionType"])
            .agg(
                pl.col("IndexFirst").first(),
                pl.col("IndexLast").last(),
            )
            .sort("IndexFirst")
        ).to_pandas()

        masks = [
            (datetime_signal >= row.IndexFirst) & (datetime_signal <= row.IndexLast)
            for _, row in rth.iterrows()
        ]
        
        if masks:
            combined = np.any(np.stack(masks, axis=0), axis=0)
        else:
            combined = np.zeros(len(datetime_signal), dtype=bool)

        return datetime_signal[combined], signal_sides[combined]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        exit_strategy: Optional[BaseExitStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        tp_ticks: Optional[float] = None,
        sl_ticks: Optional[float] = None,
        trailing_stop_ticks: Optional[float] = None,
        break_even_ticks: Optional[float] = None,
        break_even_offset_ticks: float = 0.0,
        initial_capital: float = 0.0,
        indicator_columns: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        data : pd.DataFrame
            Tick-by-tick market data.  Required columns:
            ``Index``, ``Datetime``, ``Price``, ``Date``, ``Time``, ``SessionType``
            Optional: ``Bid``, ``Ask``, ``Volume``
        signals : pd.DataFrame
            Entry signals.  Required columns:
            ``Index`` (matching ``data.Index``), ``TradeType`` (1=SHORT, 2=LONG).
        exit_strategy : BaseExitStrategy | None
            Custom exit logic.  If ``None`` the engine uses only the
            ``RiskManager`` (TP/SL/trailing/break-even).
        risk_manager : RiskManager | None
            Risk overlay.  If ``None``, one is built from the ``tp_ticks``,
            ``sl_ticks``, etc. arguments.
        tp_ticks : float | None
            Take-profit distance in ticks (convenience; ignored if
            ``risk_manager`` is supplied).
        sl_ticks : float | None
            Stop-loss distance in ticks.
        trailing_stop_ticks : float | None
            Trailing stop distance in ticks.
        break_even_ticks : float | None
            Break-even activation distance in ticks.
        break_even_offset_ticks : float
            Extra offset for break-even stop.
        initial_capital : float
            Starting capital for equity-curve / drawdown calculations.
        indicator_columns : list[str] | None
            Column names from ``data`` to pass as ``indicators`` dict to
            exit strategies.

        Returns
        -------
        BacktestResult
        """
        # --- Validation ---
        self._validate_data(data)
        self._validate_signals(signals)

        # --- Build risk manager if not supplied ---
        if risk_manager is None:
            risk_manager = RiskManager(
                tp_ticks=tp_ticks,
                sl_ticks=sl_ticks,
                trailing_stop_ticks=trailing_stop_ticks,
                break_even_ticks=break_even_ticks,
                break_even_offset_ticks=break_even_offset_ticks,
                tick_size=self.tick_size,
                tick_value=self.tick_value,
            )

        # --- Extract arrays ---
        prices     = np.ascontiguousarray(data["Price"].values, dtype=np.float64)
        timestamps = np.ascontiguousarray(data["Index"].values, dtype=np.int64)
        datetimes  = data["Datetime"].values
        dates      = data["Date"].values

        signal_ts    = np.ascontiguousarray(signals["Index"].values, dtype=np.int64)
        signal_sides = np.ascontiguousarray(signals["TradeType"].values, dtype=np.int64)

        # RTH filtering
        if self.trade_in_rth:
            signal_ts, signal_sides = self._filter_rth_signals(data, signal_ts, signal_sides)

        # Boolean entry mask (vectorised O(n) lookup)
        entry_mask = np.isin(timestamps, signal_ts)

        # Pre-sample slippage for all potential entries (reproducibility)
        n_signals = len(signal_sides)
        slippage_ticks_arr = np.array(
            [self.slippage_model.sample() for _ in range(n_signals)],
            dtype=np.int64,
        )

        # --- Choose execution path ---
        config = BacktestConfig(
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            commission=self.commission,
            n_contracts=self.n_contracts,
            trade_in_rth=self.trade_in_rth,
            seed=self.seed,
        )

        if exit_strategy is None and HAS_NUMBA:
            # Fast Numba path
            trades = self._run_numba(
                prices, timestamps, datetimes, dates, entry_mask,
                signal_sides, slippage_ticks_arr, risk_manager, config,
            )
        else:
            # Flexible Python path with custom exit strategy
            trades = self._run_python(
                data, prices, timestamps, datetimes, dates, entry_mask,
                signal_sides, slippage_ticks_arr, risk_manager, config,
                exit_strategy, indicator_columns,
            )

        # --- Metrics ---
        metrics = compute_metrics(trades, initial_capital=initial_capital)

        # --- Trades DataFrame ---
        trades_df = self._trades_to_dataframe(trades)

        return BacktestResult(
            trades=trades,
            trades_df=trades_df,
            metrics=metrics,
            config=config,
        )

    # ------------------------------------------------------------------ #
    #  Numba fast path                                                    #
    # ------------------------------------------------------------------ #

    def _run_numba(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        datetimes: np.ndarray,
        dates: np.ndarray,
        entry_mask: np.ndarray,
        signal_sides: np.ndarray,
        slippage_ticks_arr: np.ndarray,
        risk_manager: RiskManager,
        config: BacktestConfig,
    ) -> List[TradeRecord]:
        """Dispatch to Numba-compiled loop and assemble TradeRecords."""
        tp = risk_manager.tp_ticks if risk_manager.tp_ticks is not None else 1e18
        sl = risk_manager.sl_ticks if risk_manager.sl_ticks is not None else 1e18
        trail = risk_manager.trailing_stop_ticks if risk_manager.trailing_stop_ticks is not None else 0.0
        be = risk_manager.break_even_ticks if risk_manager.break_even_ticks is not None else 0.0
        be_off = risk_manager.break_even_offset_ticks

        (
            entry_idxs, exit_idxs, entry_ts, exit_ts,
            entry_prices, entry_pures, exit_prices,
            sides, exit_reasons, maes, mfes, ticks_in,
        ) = _numba_core_loop(
            prices, timestamps, entry_mask, signal_sides, slippage_ticks_arr,
            config.tick_size, tp, sl, trail, be, be_off,
            config.tick_value, config.commission, config.n_contracts,
        )

        # Map exit-reason codes → ExitReason enum
        _reason_map = {
            0: ExitReason.TAKE_PROFIT,
            1: ExitReason.STOP_LOSS,
            2: ExitReason.TRAILING_STOP,
            3: ExitReason.BREAK_EVEN,
            4: ExitReason.END_OF_DATA,
        }

        records: List[TradeRecord] = []
        for j in range(len(entry_idxs)):
            s = Side.LONG if sides[j] == 1 else Side.SHORT
            ep = entry_prices[j]
            xp = exit_prices[j]
            pnl_ticks = ((xp - ep) / config.tick_size) if s == Side.LONG else ((ep - xp) / config.tick_size)
            pnl_dollars = pnl_ticks * config.tick_value * config.n_contracts
            comm = config.commission * config.n_contracts
            net = pnl_dollars - comm

            records.append(TradeRecord(
                trade_id=j + 1,
                side=s,
                entry_datetime=datetimes[entry_idxs[j]],
                exit_datetime=datetimes[exit_idxs[j]],
                entry_timestamp=entry_ts[j],
                exit_timestamp=exit_ts[j],
                entry_price=ep,
                entry_price_pure=entry_pures[j],
                exit_price=xp,
                pnl_ticks=pnl_ticks,
                pnl_dollars=pnl_dollars,
                commission=comm,
                net_pnl=net,
                exit_reason=_reason_map.get(int(exit_reasons[j]), ExitReason.CUSTOM),
                mae_ticks=maes[j],
                mfe_ticks=mfes[j],
                ticks_in_trade=int(ticks_in[j]),
            ))

        return records

    # ------------------------------------------------------------------ #
    #  Pure-Python flexible path                                          #
    # ------------------------------------------------------------------ #

    def _run_python(
        self,
        data: pd.DataFrame,
        prices: np.ndarray,
        timestamps: np.ndarray,
        datetimes: np.ndarray,
        dates: np.ndarray,
        entry_mask: np.ndarray,
        signal_sides: np.ndarray,
        slippage_ticks_arr: np.ndarray,
        risk_manager: RiskManager,
        config: BacktestConfig,
        exit_strategy: Optional[BaseExitStrategy],
        indicator_columns: Optional[List[str]],
    ) -> List[TradeRecord]:
        """
        Pure-Python tick loop with full strategy/risk manager support.
        """
        n = len(prices)
        records: List[TradeRecord] = []
        trade_counter = 0
        signal_ptr = 0
        pos = PositionState()

        # Price history ring buffer
        hist_window = self.price_history_window
        price_history = np.empty(hist_window, dtype=np.float64)
        hist_len = 0

        # Pre-extract indicator arrays
        ind_arrays: Dict[str, np.ndarray] = {}
        if indicator_columns:
            for col in indicator_columns:
                if col in data.columns:
                    ind_arrays[col] = data[col].values

        # Optional bid/ask
        has_bid = "Bid" in data.columns
        has_ask = "Ask" in data.columns
        bids = data["Bid"].values if has_bid else np.zeros(n)
        asks = data["Ask"].values if has_ask else np.zeros(n)

        # Session type
        has_session = "SessionType" in data.columns
        session_types = data["SessionType"].values if has_session else np.full(n, "")

        iterator = tqdm(range(n), desc="Backtesting", disable=not self.progress_bar)

        for i in iterator:
            price = prices[i]

            # Update rolling price history
            if hist_len < hist_window:
                price_history[hist_len] = price
                hist_len += 1
            else:
                price_history[:-1] = price_history[1:]
                price_history[-1] = price

            # ---- Not in position: try entry ----
            if pos.is_flat:
                if entry_mask[i] and signal_ptr < len(signal_sides):
                    side_code = signal_sides[signal_ptr]
                    side = Side.LONG if side_code == 2 else Side.SHORT

                    slip_ticks = int(slippage_ticks_arr[signal_ptr])
                    slip_amount = slip_ticks * config.tick_size
                    pure_price = price

                    if side == Side.LONG:
                        fill_price = price + slip_amount
                    else:
                        fill_price = price - slip_amount

                    # Initialise position state
                    pos.side = side
                    pos.entry_price = fill_price
                    pos.entry_price_pure = pure_price
                    pos.entry_tick_idx = i
                    pos.entry_timestamp = timestamps[i]
                    pos.entry_datetime = datetimes[i]
                    pos.max_favorable_price = price
                    pos.max_adverse_price = price
                    pos.ticks_in_trade = 0
                    pos.break_even_triggered = False

                    # Risk manager sets initial levels
                    risk_manager.initialize_position(pos)

                    # Strategy hook
                    if exit_strategy is not None:
                        tick_obj = Tick(
                            index=i, timestamp=timestamps[i], datetime=datetimes[i],
                            price=price, bid=float(bids[i]), ask=float(asks[i]),
                            date=dates[i], session_type=str(session_types[i]),
                        )
                        exit_strategy.on_entry(tick_obj, pos)

                    signal_ptr += 1
                    continue

                # Advance signal pointer for stale entries
                if entry_mask[i] and signal_ptr < len(signal_sides):
                    signal_ptr += 1
                continue

            # ---- In position: evaluate risk + strategy ----
            tick_obj = Tick(
                index=i, timestamp=timestamps[i], datetime=datetimes[i],
                price=price, bid=float(bids[i]), ask=float(asks[i]),
                date=dates[i], session_type=str(session_types[i]),
            )

            # 1) Risk manager first
            risk_signal = risk_manager.check(tick_obj, pos)

            exit_signal: Optional[ExitSignal] = None
            if risk_signal.should_exit:
                exit_signal = risk_signal
            elif exit_strategy is not None:
                # 2) Then custom strategy
                indicators = {col: arr[i] for col, arr in ind_arrays.items()}
                hist_slice = price_history[:hist_len] if hist_len < hist_window else price_history
                strat_signal = exit_strategy.on_tick(tick_obj, pos, hist_slice, indicators)
                if strat_signal.should_exit:
                    exit_signal = strat_signal

            if exit_signal is not None and exit_signal.should_exit:
                # --- Close position ---
                exit_price = exit_signal.exit_price if exit_signal.exit_price is not None else price
                trade_counter += 1

                pnl_ticks = (
                    (exit_price - pos.entry_price) / config.tick_size
                    if pos.side == Side.LONG
                    else (pos.entry_price - exit_price) / config.tick_size
                )
                pnl_dollars = pnl_ticks * config.tick_value * config.n_contracts
                comm = config.commission * config.n_contracts
                net = pnl_dollars - comm

                # MAE / MFE in ticks
                if pos.side == Side.LONG:
                    mae = (pos.entry_price - pos.max_adverse_price) / config.tick_size
                    mfe = (pos.max_favorable_price - pos.entry_price) / config.tick_size
                else:
                    mae = (pos.max_adverse_price - pos.entry_price) / config.tick_size
                    mfe = (pos.entry_price - pos.max_favorable_price) / config.tick_size

                records.append(TradeRecord(
                    trade_id=trade_counter,
                    side=pos.side,
                    entry_datetime=pos.entry_datetime,
                    exit_datetime=datetimes[i],
                    entry_timestamp=pos.entry_timestamp,
                    exit_timestamp=timestamps[i],
                    entry_price=pos.entry_price,
                    entry_price_pure=pos.entry_price_pure,
                    exit_price=exit_price,
                    pnl_ticks=pnl_ticks,
                    pnl_dollars=pnl_dollars,
                    commission=comm,
                    net_pnl=net,
                    exit_reason=exit_signal.reason,
                    mae_ticks=mae,
                    mfe_ticks=mfe,
                    ticks_in_trade=pos.ticks_in_trade,
                    metadata=exit_signal.metadata,
                ))

                # Strategy cleanup hook
                if exit_strategy is not None:
                    exit_strategy.on_exit(tick_obj, pos, exit_signal.reason)

                pos.reset()

                # Advance signal pointer past any signals during this trade
                while signal_ptr < len(signal_sides):
                    if signal_ptr >= len(signal_sides):
                        break
                    # Find next signal that is in the future
                    # (signal_ts is implicit via entry_mask — we just step ptr)
                    if entry_mask[i] and signal_ptr < len(signal_sides):
                        signal_ptr += 1
                    break

        # --- Close dangling position at end of data ---
        if not pos.is_flat:
            trade_counter += 1
            exit_price = prices[n - 1]

            if pos.side == Side.LONG:
                pnl_ticks = (exit_price - pos.entry_price) / config.tick_size
                mae = (pos.entry_price - pos.max_adverse_price) / config.tick_size
                mfe = (pos.max_favorable_price - pos.entry_price) / config.tick_size
            else:
                pnl_ticks = (pos.entry_price - exit_price) / config.tick_size
                mae = (pos.max_adverse_price - pos.entry_price) / config.tick_size
                mfe = (pos.entry_price - pos.max_favorable_price) / config.tick_size

            pnl_dollars = pnl_ticks * config.tick_value * config.n_contracts
            comm = config.commission * config.n_contracts
            net = pnl_dollars - comm

            records.append(TradeRecord(
                trade_id=trade_counter,
                side=pos.side,
                entry_datetime=pos.entry_datetime,
                exit_datetime=datetimes[n - 1],
                entry_timestamp=pos.entry_timestamp,
                exit_timestamp=timestamps[n - 1],
                entry_price=pos.entry_price,
                entry_price_pure=pos.entry_price_pure,
                exit_price=exit_price,
                pnl_ticks=pnl_ticks,
                pnl_dollars=pnl_dollars,
                commission=comm,
                net_pnl=net,
                exit_reason=ExitReason.END_OF_DATA,
                mae_ticks=mae,
                mfe_ticks=mfe,
                ticks_in_trade=pos.ticks_in_trade,
            ))

            pos.reset()

        return records

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
        """Convert a list of ``TradeRecord`` into a Pandas DataFrame."""
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                "trade_id": t.trade_id,
                "side": "LONG" if t.side == Side.LONG else "SHORT",
                "entry_datetime": t.entry_datetime,
                "exit_datetime": t.exit_datetime,
                "entry_timestamp": t.entry_timestamp,
                "exit_timestamp": t.exit_timestamp,
                "entry_price": t.entry_price,
                "entry_price_pure": t.entry_price_pure,
                "exit_price": t.exit_price,
                "pnl_ticks": t.pnl_ticks,
                "pnl_dollars": t.pnl_dollars,
                "commission": t.commission,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason.value,
                "mae_ticks": t.mae_ticks,
                "mfe_ticks": t.mfe_ticks,
                "ticks_in_trade": t.ticks_in_trade,
            })
        return pd.DataFrame(rows)
