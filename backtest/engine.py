"""Event-driven backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime
import logging
from pathlib import Path

from config.settings import settings
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


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
        self._sl_options = [80.0, 120.0, 160.0, 200.0]
        self._current_sl_ticks = 0.0
        self._current_sl_level = 0
        self._stop_loss_price = 0.0
        self._sl_hits = 0
        self._max_unrealized_pnl = 0.0
        self._sl_choices = []

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
        n_bars = len(self.df)

        logger.info(f"Starting backtest with {n_bars} bars")

        for i in range(n_bars):
            self.current_bar = i
            bar = self.df.iloc[i]

            # Update position P&L
            self._update_position_pnl(bar["close"])

            # Track MFE/MAE
            if self.position.quantity != 0:
                self._track_excursion(bar)

            # Record equity
            equity = self.capital + self.position.unrealized_pnl
            self.equity_curve.append(equity)

            # Get strategy signal
            signal = strategy(self.df, i, self.position)

            # Execute signal
            if signal is not None:
                self._execute_signal(signal, bar)

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
        sl_options: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest using a trained RL model with learnable SL.

        Args:
            model: Trained agent with predict() method
            feature_columns: Feature columns for model input
            lookback_window: Lookback window for state
            deterministic: Use deterministic policy
            sl_options: List of SL options in ticks

        Returns:
            Backtest results
        """
        self.reset()
        n_bars = len(self.df)
        self._time_in_position = 0
        self._sl_options = sl_options or [80.0, 120.0, 160.0, 200.0]
        self._current_sl_ticks = 0.0
        self._current_sl_level = 0
        self._stop_loss_price = 0.0
        self._sl_hits = 0
        self._max_unrealized_pnl = 0.0
        self._sl_choices = []

        logger.info(f"Running model backtest with {n_bars} bars")
        logger.info(f"SL options: {self._sl_options} ticks")

        for i in range(lookback_window, n_bars):
            self.current_bar = i
            bar = self.df.iloc[i]

            # Update position P&L
            self._update_position_pnl(bar["close"])

            # Track MFE/MAE and max unrealized P&L
            if self.position.quantity != 0:
                self._track_excursion(bar)
                self._time_in_position += 1
                self._max_unrealized_pnl = max(self._max_unrealized_pnl, self.position.unrealized_pnl)

            # Record equity
            equity = self.capital + self.position.unrealized_pnl
            self.equity_curve.append(equity)

            # Check SL first
            sl_hit = False
            if self.position.quantity != 0:
                sl_hit = self._check_sl(bar)

            if sl_hit:
                continue

            # Prepare observation
            obs = self._prepare_observation(i, feature_columns, lookback_window)

            # Get model prediction (MultiDiscrete: [action_type, sl_level])
            action, _ = model.predict(obs, deterministic=deterministic)

            # Parse action
            if isinstance(action, np.ndarray) and len(action) >= 2:
                action_type = int(action[0])
                sl_level = int(action[1])
            else:
                action_type = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                sl_level = 1  # Default medium

            # Execute action
            if action_type == 0:  # HOLD
                pass
            elif action_type == 3:  # CLOSE
                if self.position.quantity != 0:
                    self._close_position(bar)
                    self._time_in_position = 0
                    self._max_unrealized_pnl = 0.0
            elif action_type in [1, 2]:  # BUY or SELL
                signal = OrderSide.BUY if action_type == 1 else OrderSide.SELL
                current_pos = np.sign(self.position.quantity)

                if (signal == OrderSide.BUY and current_pos == -1) or \
                   (signal == OrderSide.SELL and current_pos == 1):
                    self._close_position(bar)
                    self._open_position_with_sl(signal, bar, sl_level)
                elif current_pos == 0:
                    self._open_position_with_sl(signal, bar, sl_level)

        # Close any open position
        if self.position.quantity != 0:
            self._close_position(self.df.iloc[-1])

        results = self._compile_results()
        results["sl_hits"] = self._sl_hits
        results["sl_choices"] = self._sl_choices
        return results

    def _open_position_with_sl(self, side: OrderSide, bar: pd.Series, sl_level: int = 0) -> None:
        """Open position and set stop loss based on chosen level."""
        timestamp = bar.name if hasattr(bar, "name") else self.df.index[self.current_bar]
        self._open_position(side, bar, timestamp)

        # Set SL based on chosen level
        sl_level = min(sl_level, len(self._sl_options) - 1)
        self._current_sl_level = sl_level
        self._current_sl_ticks = self._sl_options[sl_level]
        sl_points = self._current_sl_ticks * self.tick_size

        # Track SL choice
        self._sl_choices.append(sl_level)

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
        """Prepare observation for model (7-element position_info with learnable SL)."""
        start_idx = current_bar - lookback_window
        market_data = self.df.iloc[start_idx:current_bar][feature_columns].values

        # Normalize unrealized P&L
        normalized_pnl = self.position.unrealized_pnl / self.initial_capital

        # Get time in position
        time_in_position = self._time_in_position if hasattr(self, '_time_in_position') else 0

        # Balance ratio
        current_equity = self.capital + self.position.unrealized_pnl
        balance_ratio = current_equity / self.initial_capital

        # Calculate distance to SL and max P&L ratio
        if self.position.quantity != 0:
            sl_points = self._current_sl_ticks * self.tick_size
            max_loss = sl_points * self.point_value
            pnl_to_sl_ratio = (self.position.unrealized_pnl + max_loss) / max_loss if max_loss > 0 else 1.0
            max_pnl_ratio = self.position.unrealized_pnl / max(1.0, self._max_unrealized_pnl) if self._max_unrealized_pnl > 0 else 1.0
        else:
            pnl_to_sl_ratio = 1.0
            max_pnl_ratio = 1.0

        # Normalize current SL level
        sl_level_norm = self._current_sl_level / max(1, len(self._sl_options) - 1) if self.position.quantity != 0 else 0.5

        position_info = np.array([
            np.sign(self.position.quantity),
            normalized_pnl,
            min(time_in_position / 100, 1.0),
            balance_ratio,
            pnl_to_sl_ratio,   # Distance to SL
            max_pnl_ratio,     # Current P&L vs max P&L in trade
            sl_level_norm,     # Current SL level (normalized)
        ], dtype=np.float32)

        return {
            "market_features": market_data.astype(np.float32),
            "position_info": position_info,
        }

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
