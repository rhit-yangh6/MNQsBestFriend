"""Clean Gymnasium trading environment for MNQ futures."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from enum import IntEnum

from config.settings import settings
from .rewards import RewardCalculator


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


class TradingEnv(gym.Env):
    """
    Clean trading environment with simple differential rewards.

    Key design principles:
    - Reward = change in equity (immediate feedback every step)
    - No complex bonuses/penalties that confuse learning
    - Agent learns purely from profit/loss signal
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        max_position: int = 1,
        commission: float = 0.62,
        slippage_ticks: float = 1.0,
        reward_scaling: float = 1.0,
        lookback_window: int = 50,
        stop_loss_ticks: float = 200.0,
        render_mode: Optional[str] = None,
        # Legacy parameters (ignored but kept for compatibility)
        reward_type: str = None,
        fixed_sl_ticks: float = None,
        sl_options: List[float] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: List of feature column names
            initial_balance: Starting balance
            max_position: Maximum contracts (always 1 for now)
            commission: Commission per side
            slippage_ticks: Slippage in ticks
            reward_scaling: Scale factor for rewards
            lookback_window: Bars of history in observation
            stop_loss_ticks: Emergency stop loss distance
            render_mode: Rendering mode
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.lookback_window = lookback_window
        self.stop_loss_ticks = stop_loss_ticks if fixed_sl_ticks is None else fixed_sl_ticks
        self.render_mode = render_mode

        # Reward calculator
        self.reward_calculator = RewardCalculator(reward_scaling=reward_scaling)

        # Pre-convert data to numpy
        self.n_features = len(feature_columns)
        self.features_data = self.df[feature_columns].values.astype(np.float32)
        self.close_prices = self.df["close"].values.astype(np.float64)
        self.high_prices = self.df["high"].values.astype(np.float64)
        self.low_prices = self.df["low"].values.astype(np.float64)
        self.data_len = len(self.df)

        # Action space: HOLD, BUY, SELL, CLOSE
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Dict({
            "market_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, self.n_features),
                dtype=np.float32,
            ),
            "position_info": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),  # position, unrealized_pnl_norm, time_norm, equity_ratio
                dtype=np.float32,
            ),
        })

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all state variables."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.time_in_position = 0
        self.unrealized_pnl = 0.0

        # Statistics
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_balance]
        self.peak_equity = self.initial_balance

        # Reset reward calculator
        self.reward_calculator.reset(self.initial_balance)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Simple flow:
        1. Check stop loss
        2. Execute action
        3. Update state
        4. Calculate reward (equity change)
        5. Check termination
        """
        # Parse action
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            action = int(action)

        # Get prices for this bar
        current_price = self.close_prices[self.current_step]
        high = self.high_prices[self.current_step]
        low = self.low_prices[self.current_step]

        # Track trade completion for reward
        trade_closed = False
        trade_pnl = 0.0
        bars_held = 0
        stop_loss_hit = False

        # 1. Check stop loss first
        if self.position != 0:
            sl_hit, sl_pnl = self._check_stop_loss(high, low)
            if sl_hit:
                trade_pnl = sl_pnl
                bars_held = self.time_in_position
                trade_closed = True
                stop_loss_hit = True

        # 2. Execute action (if no SL hit)
        if not trade_closed:
            if action == Action.BUY:
                if self.position == -1:
                    # Close short, open long
                    bars_held = self.time_in_position
                    trade_pnl = self._close_position(current_price)
                    trade_closed = True
                    self._open_position(1, current_price)
                elif self.position == 0:
                    # Open long
                    self._open_position(1, current_price)

            elif action == Action.SELL:
                if self.position == 1:
                    # Close long, open short
                    bars_held = self.time_in_position
                    trade_pnl = self._close_position(current_price)
                    trade_closed = True
                    self._open_position(-1, current_price)
                elif self.position == 0:
                    # Open short
                    self._open_position(-1, current_price)

            elif action == Action.CLOSE:
                if self.position != 0:
                    bars_held = self.time_in_position
                    trade_pnl = self._close_position(current_price)
                    trade_closed = True

            # HOLD: do nothing

        # 3. Move to next step and update unrealized PnL
        self.current_step += 1

        if self.position != 0:
            self.time_in_position += 1
            next_price = self.close_prices[self.current_step]
            self.unrealized_pnl = self._calc_unrealized_pnl(next_price)
        else:
            self.time_in_position = 0
            self.unrealized_pnl = 0.0

        # 4. Calculate reward based on equity change
        current_equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(current_equity)

        reward = self.reward_calculator.calculate_step_reward(
            current_equity=current_equity,
            position=self.position,
            unrealized_pnl=self.unrealized_pnl,
            trade_just_closed=trade_closed,
            trade_pnl=trade_pnl,
            bars_held=bars_held,
            stop_loss_hit=stop_loss_hit,
        )

        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # 5. Check termination
        terminated = False
        truncated = False

        # End of data
        if self.current_step >= self.data_len - 1:
            truncated = True

        # Bankruptcy
        if current_equity <= 0:
            terminated = True
            reward = -10.0  # Terminal penalty

        # Max drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= settings.MAX_DRAWDOWN:
            terminated = True
            reward = -5.0  # Terminal penalty

        # Max daily loss
        if current_equity < self.initial_balance - settings.MAX_DAILY_LOSS:
            terminated = True
            reward = -5.0

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_stop_loss(self, high: float, low: float) -> Tuple[bool, Optional[float]]:
        """Check if stop loss was triggered. Returns (triggered, pnl)."""
        if self.position == 1 and low <= self.stop_loss_price:
            return True, self._close_at_price(self.stop_loss_price, "STOP_LOSS")
        elif self.position == -1 and high >= self.stop_loss_price:
            return True, self._close_at_price(self.stop_loss_price, "STOP_LOSS")
        return False, None

    def _open_position(self, direction: int, price: float) -> None:
        """Open a position."""
        self.position = direction
        self.time_in_position = 0

        # Apply slippage
        slip = self.slippage_ticks * settings.TICK_SIZE
        if direction == 1:
            self.entry_price = price + slip
            self.stop_loss_price = self.entry_price - (self.stop_loss_ticks * settings.TICK_SIZE)
        else:
            self.entry_price = price - slip
            self.stop_loss_price = self.entry_price + (self.stop_loss_ticks * settings.TICK_SIZE)

        # Deduct commission
        self.balance -= self.commission

    def _close_position(self, price: float) -> float:
        """Close position at market price."""
        if self.position == 0:
            return 0.0

        # Apply slippage
        slip = self.slippage_ticks * settings.TICK_SIZE
        if self.position == 1:
            exit_price = price - slip
        else:
            exit_price = price + slip

        return self._close_at_price(exit_price, "CLOSE")

    def _close_at_price(self, exit_price: float, exit_type: str) -> float:
        """Close position at specific price."""
        if self.position == 0:
            return 0.0

        # Calculate PnL
        price_diff = exit_price - self.entry_price
        pnl = price_diff * self.position * settings.POINT_VALUE
        pnl -= self.commission  # Exit commission

        # Update balance
        self.balance += pnl

        # Record trade
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "direction": self.position,
            "pnl": pnl,
            "bars_held": self.time_in_position,
            "exit_type": exit_type,
            "step": self.current_step,
        })

        # Reset position state
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.unrealized_pnl = 0.0

        return pnl

    def _calc_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.position == 0:
            return 0.0
        price_diff = current_price - self.entry_price
        return price_diff * self.position * settings.POINT_VALUE

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        start = self.current_step - self.lookback_window
        market_features = self.features_data[start:self.current_step]

        # Simple position info
        equity = self.balance + self.unrealized_pnl
        position_info = np.array([
            float(self.position),
            self.unrealized_pnl / 100.0,  # Normalize ~$100 typical
            min(self.time_in_position / 50.0, 2.0),  # Normalize time
            equity / self.initial_balance,  # Equity ratio
        ], dtype=np.float32)

        return {
            "market_features": market_features,
            "position_info": position_info,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        winning_trades = sum(1 for t in self.trades if t["pnl"] > 0)
        losing_trades = sum(1 for t in self.trades if t["pnl"] <= 0)
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = sum(abs(t["pnl"]) for t in self.trades if t["pnl"] <= 0)

        equity = self.balance + self.unrealized_pnl
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        realized_pnl = self.balance - self.initial_balance
        total_trades = len(self.trades)

        return {
            "step": self.current_step,
            "balance": self.balance,
            "equity": equity,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": winning_trades / max(1, total_trades),
            "profit_factor": gross_profit / max(0.01, gross_loss),
            "drawdown": drawdown,
        }

    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step {self.current_step} | "
                f"Equity: ${info['equity']:.2f} | "
                f"Position: {self.position} | "
                f"Trades: {info['total_trades']}"
            )

    def close(self) -> None:
        """Clean up."""
        pass

    def get_trade_history(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)
