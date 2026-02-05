"""Event-driven backtesting engine for trading strategies.

Performance optimizations:
- Numba JIT compilation for core loop (20x-300x speedup)
- Preallocated numpy arrays (no list.append)
- Pure numpy operations in hot path (no pandas)
- Simplified trade storage using arrays
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else args[0]

from config.settings import settings
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS (20x-300x faster)
# =============================================================================

def run_vectorized_backtest(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    actions: np.ndarray,
    initial_capital: float = 10000.0,
    commission: float = 0.62,
    slippage_ticks: float = 1.0,
    tick_size: float = 0.25,
    point_value: float = 2.0,
    fixed_sl_ticks: float = 120.0,
) -> Dict[str, Any]:
    """
    Standalone high-performance backtest function.

    Use this for parameter sweeps or when you have pre-computed actions.
    Target: >=200k bars/sec (acceptable), >=1M bars/sec (good)

    Args:
        close_prices: Close prices as numpy array
        high_prices: High prices as numpy array
        low_prices: Low prices as numpy array
        actions: Action array (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
        initial_capital: Starting capital
        commission: Commission per side
        slippage_ticks: Slippage in ticks
        tick_size: Tick size
        point_value: Point value
        fixed_sl_ticks: Fixed stop loss in ticks

    Returns:
        Dict with equity_curve, pnls, total_trades, final_equity, total_return
    """
    slippage = slippage_ticks * tick_size

    equity_curve, entry_prices, exit_prices, pnls, sides, durations, n_trades = _backtest_core_loop(
        close_prices.astype(np.float64),
        high_prices.astype(np.float64),
        low_prices.astype(np.float64),
        actions.astype(np.int32),
        float(initial_capital),
        float(commission),
        float(slippage),
        float(point_value),
        float(tick_size),
        float(fixed_sl_ticks),
    )

    final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_equity / initial_capital - 1) * 100

    # Calculate basic metrics without Trade objects
    gross_profit = np.sum(pnls[pnls > 0]) if n_trades > 0 else 0
    gross_loss = np.abs(np.sum(pnls[pnls <= 0])) if n_trades > 0 else 0.01
    profit_factor = gross_profit / max(gross_loss, 0.01)
    win_rate = (np.sum(pnls > 0) / n_trades * 100) if n_trades > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    return {
        "equity_curve": equity_curve,
        "entry_prices": entry_prices,
        "exit_prices": exit_prices,
        "pnls": pnls,
        "sides": sides,
        "durations": durations,
        "total_trades": n_trades,
        "final_equity": final_equity,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }

@njit(cache=True)
def _backtest_core_loop(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    actions: np.ndarray,
    initial_capital: float,
    commission: float,
    slippage: float,
    point_value: float,
    tick_size: float,
    fixed_sl_ticks: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Numba-optimized core backtest loop.

    Returns:
        equity_curve, entry_prices, exit_prices, pnls, sides, durations, n_trades
    """
    n_bars = len(close_prices)

    # Preallocated arrays
    equity_curve = np.empty(n_bars, dtype=np.float64)
    max_trades = n_bars // 2  # Upper bound on possible trades
    entry_prices = np.empty(max_trades, dtype=np.float64)
    exit_prices = np.empty(max_trades, dtype=np.float64)
    pnls = np.empty(max_trades, dtype=np.float64)
    sides = np.empty(max_trades, dtype=np.int32)
    durations = np.empty(max_trades, dtype=np.int32)

    # State
    capital = initial_capital
    position = 0  # -1, 0, +1
    pos_entry_price = 0.0
    pos_entry_bar = 0
    stop_loss_price = 0.0
    trade_idx = 0
    sl_distance = fixed_sl_ticks * tick_size

    for i in range(n_bars):
        close = close_prices[i]
        high = high_prices[i]
        low = low_prices[i]
        action = actions[i]

        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if position != 0:
            price_diff = close - pos_entry_price
            unrealized_pnl = price_diff * position * point_value

        # Record equity
        equity_curve[i] = capital + unrealized_pnl

        # Check stop loss
        sl_hit = False
        if position == 1 and low <= stop_loss_price:
            # Long SL hit
            exit_price = stop_loss_price
            pnl = (exit_price - pos_entry_price) * point_value - commission
            capital += pnl
            entry_prices[trade_idx] = pos_entry_price
            exit_prices[trade_idx] = exit_price
            pnls[trade_idx] = pnl
            sides[trade_idx] = 1
            durations[trade_idx] = i - pos_entry_bar
            trade_idx += 1
            position = 0
            sl_hit = True
        elif position == -1 and high >= stop_loss_price:
            # Short SL hit
            exit_price = stop_loss_price
            pnl = (pos_entry_price - exit_price) * point_value - commission
            capital += pnl
            entry_prices[trade_idx] = pos_entry_price
            exit_prices[trade_idx] = exit_price
            pnls[trade_idx] = pnl
            sides[trade_idx] = -1
            durations[trade_idx] = i - pos_entry_bar
            trade_idx += 1
            position = 0
            sl_hit = True

        if sl_hit:
            continue

        # Execute action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        if action == 0:
            pass  # HOLD
        elif action == 3:  # CLOSE
            if position != 0:
                if position == 1:
                    exit_price = close - slippage
                    pnl = (exit_price - pos_entry_price) * point_value - commission
                else:
                    exit_price = close + slippage
                    pnl = (pos_entry_price - exit_price) * point_value - commission
                capital += pnl
                entry_prices[trade_idx] = pos_entry_price
                exit_prices[trade_idx] = exit_price
                pnls[trade_idx] = pnl
                sides[trade_idx] = position
                durations[trade_idx] = i - pos_entry_bar
                trade_idx += 1
                position = 0
        elif action == 1:  # BUY
            if position == -1:
                # Close short
                exit_price = close + slippage
                pnl = (pos_entry_price - exit_price) * point_value - commission
                capital += pnl
                entry_prices[trade_idx] = pos_entry_price
                exit_prices[trade_idx] = exit_price
                pnls[trade_idx] = pnl
                sides[trade_idx] = -1
                durations[trade_idx] = i - pos_entry_bar
                trade_idx += 1
                position = 0
            if position == 0:
                # Open long
                pos_entry_price = close + slippage
                pos_entry_bar = i
                position = 1
                stop_loss_price = pos_entry_price - sl_distance
                capital -= commission
        elif action == 2:  # SELL
            if position == 1:
                # Close long
                exit_price = close - slippage
                pnl = (exit_price - pos_entry_price) * point_value - commission
                capital += pnl
                entry_prices[trade_idx] = pos_entry_price
                exit_prices[trade_idx] = exit_price
                pnls[trade_idx] = pnl
                sides[trade_idx] = 1
                durations[trade_idx] = i - pos_entry_bar
                trade_idx += 1
                position = 0
            if position == 0:
                # Open short
                pos_entry_price = close - slippage
                pos_entry_bar = i
                position = -1
                stop_loss_price = pos_entry_price + sl_distance
                capital -= commission

    # Close final position
    if position != 0:
        close = close_prices[n_bars - 1]
        if position == 1:
            exit_price = close - slippage
            pnl = (exit_price - pos_entry_price) * point_value - commission
        else:
            exit_price = close + slippage
            pnl = (pos_entry_price - exit_price) * point_value - commission
        capital += pnl
        entry_prices[trade_idx] = pos_entry_price
        exit_prices[trade_idx] = exit_price
        pnls[trade_idx] = pnl
        sides[trade_idx] = position
        durations[trade_idx] = n_bars - 1 - pos_entry_bar
        trade_idx += 1

    # Trim arrays to actual size
    return (
        equity_curve,
        entry_prices[:trade_idx],
        exit_prices[:trade_idx],
        pnls[:trade_idx],
        sides[:trade_idx],
        durations[:trade_idx],
        trade_idx,
    )


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(IntEnum):
    """Order side."""
    BUY = 1
    SELL = -1


@dataclass
class Order:
    """Represents a trading order."""
    order_id: int
    timestamp: datetime
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Represents a trading position."""
    quantity: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    commission: float
    duration_bars: int
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading with realistic fills, commissions, and slippage.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.62,  # Per contract per side
        slippage_ticks: float = 1.0,
        point_value: float = 2.0,  # MNQ point value
        tick_size: float = 0.25,
        max_position: int = 5,
    ):
        """
        Initialize backtest engine.

        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            initial_capital: Starting capital
            commission: Commission per contract per side
            slippage_ticks: Slippage in ticks
            point_value: Dollar value per point
            tick_size: Minimum price increment
            max_position: Maximum position size
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.slippage = slippage_ticks * tick_size
        self.point_value = point_value
        self.tick_size = tick_size
        self.max_position = max_position

        # Pre-convert to numpy for speed
        self.close_prices = df["close"].values.astype(np.float32)
        self.high_prices = df["high"].values.astype(np.float32)
        self.low_prices = df["low"].values.astype(np.float32)
        self.timestamps = df.index.values
        self.n_bars = len(df)

        # State
        self.capital = initial_capital
        self.position = Position()
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_bar = 0
        self.order_id_counter = 0

        # Tracking for MFE/MAE
        self._trade_high = 0.0
        self._trade_low = float('inf')

    def reset(self) -> None:
        """Reset the backtest state."""
        self.capital = self.initial_capital
        self.position = Position()
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.current_bar = 0
        self.order_id_counter = 0
        self._trade_high = 0.0
        self._trade_low = float('inf')
        self._time_in_position = 0
        self._fixed_sl_ticks = 200.0  # Default fixed SL
        self._current_sl_ticks = 120.0
        self._stop_loss_price = 0.0
        self._sl_hits = 0
        self._max_unrealized_pnl = 0.0

    def run(
        self,
        strategy: Callable[[pd.DataFrame, int, Position], Optional[OrderSide]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the backtest with given strategy.

        Args:
            strategy: Strategy function that returns OrderSide or None
                      signature: (df, current_bar, position) -> Optional[OrderSide]
            progress_callback: Optional callback for progress updates

        Returns:
            Backtest results dictionary
        """
        self.reset()
        n_bars = self.n_bars

        logger.info(f"Starting backtest with {n_bars} bars")

        for i in tqdm(range(n_bars), desc="Backtesting", unit="bars"):
            self.current_bar = i
            close = self.close_prices[i]
            high = self.high_prices[i]
            low = self.low_prices[i]

            # Update position P&L
            self._update_position_pnl(close)

            # Track MFE/MAE
            if self.position.quantity != 0:
                self._track_excursion_fast(high, low)

            # Record equity
            equity = self.capital + self.position.unrealized_pnl
            self.equity_curve.append(equity)

            # Get strategy signal (still needs df for complex strategies)
            signal = strategy(self.df, i, self.position)

            # Execute signal
            if signal is not None:
                self._execute_signal_fast(signal, i, close)

            # Progress callback
            if progress_callback and i % 1000 == 0:
                progress_callback(i, n_bars)

        # Close any open position at end
        if self.position.quantity != 0:
            self._close_position(self.df.iloc[-1])

        # Calculate metrics
        results = self._compile_results()

        logger.info(
            f"Backtest complete: {len(self.trades)} trades, "
            f"Final equity: ${results['final_equity']:.2f}"
        )

        return results

    def run_with_model(
        self,
        model,
        feature_columns: List[str],
        lookback_window: int = 50,
        deterministic: bool = True,
        fixed_sl_ticks: float = 120.0,
        sl_options: List[float] = None,  # Deprecated - kept for compatibility
    ) -> Dict[str, Any]:
        """
        Run backtest using a trained RL model with fixed SL.

        Args:
            model: Trained agent with predict() method
            feature_columns: Feature columns for model input
            lookback_window: Lookback window for state
            deterministic: Use deterministic policy
            fixed_sl_ticks: Fixed stop loss in ticks (default 120)
            sl_options: Deprecated - kept for compatibility

        Returns:
            Backtest results
        """
        self.reset()
        n_bars = self.n_bars
        self._time_in_position = 0
        self._fixed_sl_ticks = fixed_sl_ticks
        self._current_sl_ticks = fixed_sl_ticks
        self._stop_loss_price = 0.0
        self._sl_hits = 0
        self._max_unrealized_pnl = 0.0

        # Pre-convert feature data to numpy
        self._feature_data = self.df[feature_columns].values.astype(np.float32)

        logger.info(f"Running model backtest with {n_bars} bars")
        logger.info(f"Fixed SL: {self._fixed_sl_ticks} ticks")

        for i in tqdm(range(lookback_window, n_bars), desc="Backtesting", unit="bars"):
            self.current_bar = i
            close = self.close_prices[i]
            high = self.high_prices[i]
            low = self.low_prices[i]

            # Update position P&L
            self._update_position_pnl(close)

            # Track MFE/MAE and max unrealized P&L
            if self.position.quantity != 0:
                self._track_excursion_fast(high, low)
                self._time_in_position += 1
                self._max_unrealized_pnl = max(self._max_unrealized_pnl, self.position.unrealized_pnl)

            # Record equity
            equity = self.capital + self.position.unrealized_pnl
            self.equity_curve.append(equity)

            # Check SL first
            sl_hit = False
            if self.position.quantity != 0:
                sl_hit = self._check_sl_fast(high, low)

            if sl_hit:
                continue

            # Prepare observation (optimized)
            obs = self._prepare_observation_fast(i, lookback_window)

            # Get model prediction (Discrete: single action_type)
            action, _ = model.predict(obs, deterministic=deterministic)

            # Parse action (Discrete space returns single int or 0-d array)
            if isinstance(action, np.ndarray):
                action_type = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action_type = int(action)

            # Execute action
            if action_type == 0:  # HOLD
                pass
            elif action_type == 3:  # CLOSE
                if self.position.quantity != 0:
                    self._close_position_fast(i, close)
                    self._time_in_position = 0
                    self._max_unrealized_pnl = 0.0
            elif action_type in [1, 2]:  # BUY or SELL
                signal = OrderSide.BUY if action_type == 1 else OrderSide.SELL
                current_pos = np.sign(self.position.quantity)

                if (signal == OrderSide.BUY and current_pos == -1) or \
                   (signal == OrderSide.SELL and current_pos == 1):
                    self._close_position_fast(i, close)
                    self._open_position_with_fixed_sl(signal, i, close)
                elif current_pos == 0:
                    self._open_position_with_fixed_sl(signal, i, close)

        # Close any open position
        if self.position.quantity != 0:
            last_bar = n_bars - 1
            self._close_position_fast(last_bar, self.close_prices[last_bar])

        results = self._compile_results()
        results["sl_hits"] = self._sl_hits
        return results

    def run_fast_with_model(
        self,
        model,
        feature_columns: List[str],
        lookback_window: int = 50,
        deterministic: bool = True,
        fixed_sl_ticks: float = 120.0,
        batch_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        High-performance backtest using BATCH inference.

        ~10x-50x faster than run_with_model due to batched GPU/CPU inference.

        Args:
            model: Trained agent with predict() method
            feature_columns: Feature columns for model input
            lookback_window: Lookback window for state
            deterministic: Use deterministic policy
            fixed_sl_ticks: Fixed stop loss in ticks
            batch_size: Batch size for inference (larger = faster but more memory)

        Returns:
            Backtest results
        """
        n_bars = self.n_bars

        # Pre-convert feature data to numpy
        feature_data = self.df[feature_columns].values.astype(np.float32)

        logger.info(f"Running FAST model backtest with {n_bars} bars (batch_size={batch_size})")

        # Pre-compute ALL market observations at once
        logger.info("Pre-computing market observations...")
        n_obs = n_bars - lookback_window
        n_features = len(feature_columns)

        # Preallocate observation array
        all_market_obs = np.zeros((n_obs, lookback_window, n_features), dtype=np.float32)

        for i in range(n_obs):
            all_market_obs[i] = feature_data[i:i + lookback_window]

        # Generate actions in batches
        logger.info("Running batched model inference...")
        actions = np.zeros(n_bars, dtype=np.int32)

        for batch_start in tqdm(range(0, n_obs, batch_size), desc="Batch inference", unit="batch"):
            batch_end = min(batch_start + batch_size, n_obs)
            batch_market = all_market_obs[batch_start:batch_end]
            batch_len = batch_end - batch_start

            # Create batched observations
            # Note: position_info is approximate (flat) for speed
            batch_position = np.zeros((batch_len, 4), dtype=np.float32)
            batch_position[:, 3] = 1.0  # equity_ratio = 1.0

            batch_obs = {
                "market_features": batch_market,
                "position_info": batch_position,
            }

            # Batch prediction
            batch_actions, _ = model.predict(batch_obs, deterministic=deterministic)

            # Store actions
            if isinstance(batch_actions, np.ndarray):
                if batch_actions.ndim == 0:
                    actions[lookback_window + batch_start] = int(batch_actions.item())
                else:
                    actions[lookback_window + batch_start:lookback_window + batch_end] = batch_actions.flatten().astype(np.int32)
            else:
                actions[lookback_window + batch_start] = int(batch_actions)

        # Run numba-optimized core loop
        logger.info("Running optimized backtest loop...")
        (
            equity_curve,
            entry_prices,
            exit_prices,
            pnls,
            sides,
            durations,
            n_trades,
        ) = _backtest_core_loop(
            self.close_prices.astype(np.float64),
            self.high_prices.astype(np.float64),
            self.low_prices.astype(np.float64),
            actions,
            self.initial_capital,
            self.commission,
            self.slippage,
            self.point_value,
            self.tick_size,
            fixed_sl_ticks,
        )

        # Convert to Trade objects for metrics compatibility
        self.trades = []
        for i in range(n_trades):
            trade = Trade(
                entry_time=None,  # Not tracked in fast mode
                exit_time=None,
                side=int(sides[i]),
                entry_price=entry_prices[i],
                exit_price=exit_prices[i],
                quantity=1,
                pnl=pnls[i],
                commission=self.commission * 2,
                duration_bars=int(durations[i]),
            )
            self.trades.append(trade)

        self.equity_curve = equity_curve.tolist()

        # Compile results
        results = {
            "initial_capital": self.initial_capital,
            "final_equity": equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital,
            "total_return": (equity_curve[-1] / self.initial_capital - 1) * 100 if len(equity_curve) > 0 else 0,
            "total_trades": n_trades,
            "equity_curve": equity_curve,
            "trades": self.trades,
        }

        if n_trades > 0:
            metrics = PerformanceMetrics(self.trades, equity_curve, self.initial_capital)
            results.update(metrics.calculate_all())

        logger.info(f"Fast backtest complete: {n_trades} trades, Final equity: ${results['final_equity']:.2f}")

        return results

    def _open_position_with_sl(self, side: OrderSide, bar: pd.Series) -> None:
        """Open position and set fixed stop loss."""
        timestamp = bar.name if hasattr(bar, "name") else self.df.index[self.current_bar]
        self._open_position(side, bar, timestamp)

        # Set fixed SL
        self._current_sl_ticks = self._fixed_sl_ticks
        sl_points = self._current_sl_ticks * self.tick_size

        if side == OrderSide.BUY:
            self._stop_loss_price = self.position.entry_price - sl_points
        else:
            self._stop_loss_price = self.position.entry_price + sl_points

        self._time_in_position = 0
        self._max_unrealized_pnl = 0.0

    def _check_sl(self, bar: pd.Series) -> bool:
        """Check if stop loss was hit. Returns True if position was closed."""
        if self.position.quantity == 0:
            return False

        high = bar["high"]
        low = bar["low"]

        if self.position.quantity > 0:  # Long position
            if low <= self._stop_loss_price:
                self._close_at_price(self._stop_loss_price, bar)
                self._sl_hits += 1
                return True
        else:  # Short position
            if high >= self._stop_loss_price:
                self._close_at_price(self._stop_loss_price, bar)
                self._sl_hits += 1
                return True

        return False

    def _check_sl_fast(self, high: float, low: float) -> bool:
        """Check if stop loss was hit (optimized). Returns True if position was closed."""
        if self.position.quantity == 0:
            return False

        if self.position.quantity > 0:  # Long position
            if low <= self._stop_loss_price:
                self._close_at_price_fast(self._stop_loss_price, self.current_bar)
                self._sl_hits += 1
                return True
        else:  # Short position
            if high >= self._stop_loss_price:
                self._close_at_price_fast(self._stop_loss_price, self.current_bar)
                self._sl_hits += 1
                return True

        return False

    def _close_at_price(self, exit_price: float, bar: pd.Series) -> None:
        """Close position at specified price."""
        if self.position.quantity == 0:
            return

        timestamp = bar.name if hasattr(bar, "name") else self.df.index[self.current_bar]

        # Calculate P&L
        price_diff = exit_price - self.position.entry_price
        pnl = price_diff * self.position.quantity * self.point_value
        pnl -= self.commission  # Exit commission

        # Update capital
        self.capital += pnl
        self.position.realized_pnl += pnl

        # Calculate duration
        if self.position.entry_time is not None:
            entry_idx = self.df.index.get_loc(self.position.entry_time)
            duration = self.current_bar - entry_idx
        else:
            duration = 0

        # Calculate MFE/MAE
        if self.position.quantity > 0:
            mfe = (self._trade_high - self.position.entry_price) * self.point_value
            mae = (self.position.entry_price - self._trade_low) * self.point_value
        else:
            mfe = (self.position.entry_price - self._trade_low) * self.point_value
            mae = (self._trade_high - self.position.entry_price) * self.point_value

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            side=self.position.quantity,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            quantity=abs(self.position.quantity),
            pnl=pnl,
            commission=self.commission * 2,
            duration_bars=duration,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
        )
        self.trades.append(trade)

        # Reset position
        self.position = Position()
        self._time_in_position = 0
        self._stop_loss_price = 0.0
        self._take_profit_price = 0.0

    def _prepare_observation(
        self,
        current_bar: int,
        feature_columns: List[str],
        lookback_window: int,
    ) -> Dict[str, np.ndarray]:
        """Prepare observation for model (4-element position_info)."""
        start_idx = current_bar - lookback_window
        market_data = self.df.iloc[start_idx:current_bar][feature_columns].values

        # Get time in position
        time_in_position = self._time_in_position if hasattr(self, '_time_in_position') else 0

        # Equity ratio
        current_equity = self.capital + self.position.unrealized_pnl
        equity_ratio = current_equity / self.initial_capital

        # 4-element position_info (simplified)
        position_info = np.array([
            float(np.sign(self.position.quantity)),
            self.position.unrealized_pnl / 100.0,  # Normalize ~$100 typical
            min(time_in_position / 50.0, 2.0),  # Normalize time
            equity_ratio,
        ], dtype=np.float32)

        return {
            "market_features": market_data.astype(np.float32),
            "position_info": position_info,
        }
    
    def _prepare_observation_fast(
        self,
        current_bar: int,
        lookback_window: int,
    ) -> Dict[str, np.ndarray]:
        """Prepare observation for model (optimized, 4-element position_info)."""
        start_idx = current_bar - lookback_window
        market_data = self._feature_data[start_idx:current_bar]

        # Equity ratio
        current_equity = self.capital + self.position.unrealized_pnl
        equity_ratio = current_equity / self.initial_capital

        # 4-element position_info (simplified)
        position_info = np.array([
            float(np.sign(self.position.quantity)),
            self.position.unrealized_pnl / 100.0,  # Normalize ~$100 typical
            min(self._time_in_position / 50.0, 2.0),  # Normalize time
            equity_ratio,
        ], dtype=np.float32)

        return {
            "market_features": market_data,
            "position_info": position_info,
        }

    def _execute_signal_fast(self, signal: OrderSide, bar_idx: int, close: float) -> None:
        """Execute a trading signal (optimized)."""
        current_position = np.sign(self.position.quantity)
        timestamp = self.timestamps[bar_idx]

        if signal == OrderSide.BUY:
            if current_position == -1:
                # Close short first
                self._close_position_fast(bar_idx, close)
            if current_position <= 0:
                # Open long
                self._open_position_fast(OrderSide.BUY, bar_idx, close, timestamp)

        elif signal == OrderSide.SELL:
            if current_position == 1:
                # Close long first
                self._close_position_fast(bar_idx, close)
            if current_position >= 0:
                # Open short
                self._open_position_fast(OrderSide.SELL, bar_idx, close, timestamp)

    def _open_position_with_fixed_sl(self, side: OrderSide, bar_idx: int, close: float) -> None:
        """Open position and set fixed stop loss (optimized)."""
        timestamp = self.timestamps[bar_idx]
        self._open_position_fast(side, bar_idx, close, timestamp)

        # Set fixed SL
        self._current_sl_ticks = self._fixed_sl_ticks
        sl_points = self._current_sl_ticks * self.tick_size

        if side == OrderSide.BUY:
            self._stop_loss_price = self.position.entry_price - sl_points
        else:
            self._stop_loss_price = self.position.entry_price + sl_points

        self._time_in_position = 0
        self._max_unrealized_pnl = 0.0

    def _open_position_with_sl_fast(self, side: OrderSide, bar_idx: int, close: float, sl_level: int = 0) -> None:
        """Deprecated - kept for compatibility. Use _open_position_with_fixed_sl instead."""
        self._open_position_with_fixed_sl(side, bar_idx, close)

    def _open_position_fast(
        self, side: OrderSide, bar_idx: int, close: float, timestamp: datetime
    ) -> None:
        """Open a new position (optimized)."""
        # Calculate fill price with slippage
        if side == OrderSide.BUY:
            fill_price = close + self.slippage
        else:
            fill_price = close - self.slippage

        # Deduct commission
        self.capital -= self.commission

        # Update position
        self.position.quantity = int(side)
        self.position.entry_price = fill_price
        self.position.entry_time = timestamp

        # Reset MFE/MAE tracking
        self._trade_high = fill_price
        self._trade_low = fill_price

        logger.debug(
            f"Opened {'LONG' if side == OrderSide.BUY else 'SHORT'} "
            f"at {fill_price:.2f}"
        )

    def _close_position_fast(self, bar_idx: int, close: float) -> None:
        """Close current position (optimized)."""
        if self.position.quantity == 0:
            return

        timestamp = self.timestamps[bar_idx]

        # Calculate fill price with slippage
        if self.position.quantity > 0:  # Closing long
            fill_price = close - self.slippage
        else:  # Closing short
            fill_price = close + self.slippage

        # Calculate P&L
        price_diff = fill_price - self.position.entry_price
        pnl = price_diff * self.position.quantity * self.point_value
        pnl -= self.commission  # Exit commission

        # Update capital
        self.capital += pnl
        self.position.realized_pnl += pnl

        # Calculate duration
        if self.position.entry_time is not None:
            # Use numpy searchsorted for fast index lookup
            entry_idx = np.searchsorted(self.timestamps, self.position.entry_time)
            duration = self.current_bar - entry_idx
        else:
            duration = 0

        # Calculate MFE/MAE
        if self.position.quantity > 0:  # Long position
            mfe = (self._trade_high - self.position.entry_price) * self.point_value
            mae = (self.position.entry_price - self._trade_low) * self.point_value
        else:  # Short position
            mfe = (self.position.entry_price - self._trade_low) * self.point_value
            mae = (self._trade_high - self.position.entry_price) * self.point_value

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            side=self.position.quantity,
            entry_price=self.position.entry_price,
            exit_price=fill_price,
            quantity=abs(self.position.quantity),
            pnl=pnl,
            commission=self.commission * 2,  # Entry + exit
            duration_bars=duration,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
        )
        self.trades.append(trade)

        logger.debug(
            f"Closed position at {fill_price:.2f}, P&L: ${pnl:.2f}"
        )

        # Reset position
        self.position = Position()

    def _close_at_price_fast(self, exit_price: float, bar_idx: int) -> None:
        """Close position at specified price (optimized)."""
        if self.position.quantity == 0:
            return

        timestamp = self.timestamps[bar_idx]

        # Calculate P&L
        price_diff = exit_price - self.position.entry_price
        pnl = price_diff * self.position.quantity * self.point_value
        pnl -= self.commission  # Exit commission

        # Update capital
        self.capital += pnl
        self.position.realized_pnl += pnl

        # Calculate duration
        if self.position.entry_time is not None:
            entry_idx = np.searchsorted(self.timestamps, self.position.entry_time)
            duration = self.current_bar - entry_idx
        else:
            duration = 0

        # Calculate MFE/MAE
        if self.position.quantity > 0:
            mfe = (self._trade_high - self.position.entry_price) * self.point_value
            mae = (self.position.entry_price - self._trade_low) * self.point_value
        else:
            mfe = (self.position.entry_price - self._trade_low) * self.point_value
            mae = (self._trade_high - self.position.entry_price) * self.point_value

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            side=self.position.quantity,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            quantity=abs(self.position.quantity),
            pnl=pnl,
            commission=self.commission * 2,
            duration_bars=duration,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
        )
        self.trades.append(trade)

        # Reset position
        self.position = Position()
        self._time_in_position = 0
        self._stop_loss_price = 0.0
        self._take_profit_price = 0.0


    def _execute_signal(self, signal: OrderSide, bar: pd.Series) -> None:
        """Execute a trading signal."""
        current_position = np.sign(self.position.quantity)
        timestamp = bar.name if hasattr(bar, "name") else self.df.index[self.current_bar]

        if signal == OrderSide.BUY:
            if current_position == -1:
                # Close short first
                self._close_position(bar)
            if current_position <= 0:
                # Open long
                self._open_position(OrderSide.BUY, bar, timestamp)

        elif signal == OrderSide.SELL:
            if current_position == 1:
                # Close long first
                self._close_position(bar)
            if current_position >= 0:
                # Open short
                self._open_position(OrderSide.SELL, bar, timestamp)

    def _open_position(
        self, side: OrderSide, bar: pd.Series, timestamp: datetime
    ) -> None:
        """Open a new position."""
        # Calculate fill price with slippage
        if side == OrderSide.BUY:
            fill_price = bar["close"] + self.slippage
        else:
            fill_price = bar["close"] - self.slippage

        # Deduct commission
        self.capital -= self.commission

        # Update position
        self.position.quantity = int(side)
        self.position.entry_price = fill_price
        self.position.entry_time = timestamp

        # Reset MFE/MAE tracking
        self._trade_high = fill_price
        self._trade_low = fill_price

        logger.debug(
            f"Opened {'LONG' if side == OrderSide.BUY else 'SHORT'} "
            f"at {fill_price:.2f}"
        )

    def _close_position(self, bar: pd.Series) -> None:
        """Close current position."""
        if self.position.quantity == 0:
            return

        timestamp = bar.name if hasattr(bar, "name") else self.df.index[self.current_bar]

        # Calculate fill price with slippage
        if self.position.quantity > 0:  # Closing long
            fill_price = bar["close"] - self.slippage
        else:  # Closing short
            fill_price = bar["close"] + self.slippage

        # Calculate P&L
        price_diff = fill_price - self.position.entry_price
        pnl = price_diff * self.position.quantity * self.point_value
        pnl -= self.commission  # Exit commission

        # Update capital
        self.capital += pnl
        self.position.realized_pnl += pnl

        # Calculate duration
        if self.position.entry_time is not None:
            entry_idx = self.df.index.get_loc(self.position.entry_time)
            duration = self.current_bar - entry_idx
        else:
            duration = 0

        # Calculate MFE/MAE
        if self.position.quantity > 0:  # Long position
            mfe = (self._trade_high - self.position.entry_price) * self.point_value
            mae = (self.position.entry_price - self._trade_low) * self.point_value
        else:  # Short position
            mfe = (self.position.entry_price - self._trade_low) * self.point_value
            mae = (self._trade_high - self.position.entry_price) * self.point_value

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            side=self.position.quantity,
            entry_price=self.position.entry_price,
            exit_price=fill_price,
            quantity=abs(self.position.quantity),
            pnl=pnl,
            commission=self.commission * 2,  # Entry + exit
            duration_bars=duration,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
        )
        self.trades.append(trade)

        logger.debug(
            f"Closed position at {fill_price:.2f}, P&L: ${pnl:.2f}"
        )

        # Reset position
        self.position = Position()

    def _update_position_pnl(self, current_price: float) -> None:
        """Update unrealized P&L."""
        if self.position.quantity == 0:
            self.position.unrealized_pnl = 0.0
            return

        price_diff = current_price - self.position.entry_price
        self.position.unrealized_pnl = (
            price_diff * self.position.quantity * self.point_value
        )

    def _track_excursion(self, bar: pd.Series) -> None:
        """Track maximum favorable/adverse excursion."""
        self._trade_high = max(self._trade_high, bar["high"])
        self._trade_low = min(self._trade_low, bar["low"])
    
    def _track_excursion_fast(self, high: float, low: float) -> None:
        """Track maximum favorable/adverse excursion (optimized)."""
        self._trade_high = max(self._trade_high, high)
        self._trade_low = min(self._trade_low, low)

    def _compile_results(self) -> Dict[str, Any]:
        """Compile backtest results."""
        equity_curve = np.array(self.equity_curve)

        # Basic results
        results = {
            "initial_capital": self.initial_capital,
            "final_equity": equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital,
            "total_return": (equity_curve[-1] / self.initial_capital - 1) * 100 if len(equity_curve) > 0 else 0,
            "total_trades": len(self.trades),
            "equity_curve": equity_curve,
            "trades": self.trades,
        }

        # Calculate performance metrics
        if len(self.trades) > 0:
            metrics = PerformanceMetrics(self.trades, equity_curve, self.initial_capital)
            results.update(metrics.calculate_all())

        return results

    def get_trade_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": "LONG" if t.side > 0 else "SHORT",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "commission": t.commission,
                "duration_bars": t.duration_bars,
                "mfe": t.max_favorable_excursion,
                "mae": t.max_adverse_excursion,
            }
            for t in self.trades
        ])


def simple_momentum_strategy(
    df: pd.DataFrame,
    current_bar: int,
    position: Position,
    lookback: int = 20,
) -> Optional[OrderSide]:
    """
    Simple momentum strategy for testing.

    Args:
        df: Price data
        current_bar: Current bar index
        position: Current position
        lookback: Lookback period for momentum

    Returns:
        Order side or None
    """
    if current_bar < lookback:
        return None

    # Calculate momentum
    returns = df.iloc[current_bar - lookback : current_bar]["close"].pct_change()
    momentum = returns.sum()

    current_position = np.sign(position.quantity)

    # Simple rules
    if momentum > 0.01 and current_position <= 0:
        return OrderSide.BUY
    elif momentum < -0.01 and current_position >= 0:
        return OrderSide.SELL

    return None
