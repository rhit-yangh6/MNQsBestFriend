"""Custom Gymnasium trading environment for MNQ futures."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from enum import IntEnum
import logging

from config.settings import settings
from .rewards import RewardCalculator
from .position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


class TradingEnv(gym.Env):
    """
    Custom trading environment for MNQ futures.

    Features:
    - Agent learns when to enter and exit (no forced TP)
    - Emergency stop loss as safety net only
    - Simple 4-action discrete space: HOLD, BUY, SELL, CLOSE
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        max_position: int = 1,
        commission: float = 0.62,
        slippage_ticks: float = 1.0,
        reward_type: str = "composite",
        lookback_window: int = 50,
        sl_options: List[float] = None,  # Learnable SL options in ticks
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: List of feature column names to use as state
            initial_balance: Starting account balance
            max_position: Maximum position size (contracts)
            commission: Commission per contract per side
            slippage_ticks: Expected slippage in ticks
            reward_type: Type of reward function
            lookback_window: Number of bars to include in state
            sl_options: List of SL options in ticks (agent learns to choose)
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
        self.render_mode = render_mode

        # Learnable SL options (default: tight, medium, wide)
        self.sl_options = sl_options or [80.0, 120.0, 160.0, 200.0]  # ticks
        self.n_sl_options = len(self.sl_options)

        # Reward calculator
        self.reward_calculator = RewardCalculator(reward_type=reward_type)

        # Position sizer
        self.position_sizer = PositionSizer(
            max_position=max_position,
            risk_per_trade=settings.RISK_PER_TRADE,
        )

        # Number of features
        self.n_features = len(feature_columns)

        # MultiDiscrete action space: [action_type (4), sl_level (n_sl_options)]
        # action_type: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        # sl_level: index into sl_options
        self.action_space = spaces.MultiDiscrete([4, self.n_sl_options])

        # Observation space
        market_shape = (lookback_window, self.n_features)
        # position, unrealized_pnl, time_in_position, balance_ratio, pnl_to_sl_ratio, max_pnl_ratio, current_sl_level
        position_info_size = 7

        self.observation_space = spaces.Dict({
            "market_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=market_shape,
                dtype=np.float32,
            ),
            "position_info": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(position_info_size,),
                dtype=np.float32,
            ),
        })

        # Pre-convert data to numpy for faster access
        self.market_features_data = self.df[self.feature_columns].values.astype(np.float32)
        self.close_prices = self.df["close"].values.astype(np.float32)
        self.high_prices = self.df["high"].values.astype(np.float32)
        self.low_prices = self.df["low"].values.astype(np.float32)
        self.data_len = len(self.df)

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.time_in_position = 0
        self.trades: List[Dict] = []
        self.returns_history: List[float] = []
        self.equity_curve: List[float] = [self.initial_balance]

        # SL tracking
        self.stop_loss_price = 0.0
        self.current_sl_ticks = 0.0
        self.current_sl_level = 0
        self.sl_hits = 0

        # Track max profit in trade (for reward shaping)
        self.max_unrealized_pnl = 0.0

        # Statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

        # Track SL choices for analysis
        self.sl_choices: List[int] = []

        # Drawdown tracking
        self.peak_equity = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        self.trading_restricted = False  # True when in deep drawdown

        # Reset reward calculator tracking for composite reward
        self.reward_calculator.reset_tracking(self.initial_balance)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_state()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: MultiDiscrete [action_type, sl_level]
                    action_type: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
                    sl_level: index into sl_options

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Parse action
        action_type = int(action[0])
        sl_level = int(action[1])

        prev_equity = self.balance + self.unrealized_pnl

        # Get current bar's OHLC
        current_price = self.close_prices[self.current_step]
        high_price = self.high_prices[self.current_step]
        low_price = self.low_prices[self.current_step]

        # Check SL first
        sl_reward = 0.0
        sl_triggered = False

        if self.position != 0:
            sl_reward, sl_triggered = self._check_sl(high_price, low_price)

        # Execute action only if SL wasn't triggered
        action_reward = 0.0
        if not sl_triggered:
            action_reward = self._execute_action(Action(action_type), current_price, sl_level)

        reward = sl_reward + action_reward

        # Update step
        self.current_step += 1

        # Update unrealized P&L
        if self.position != 0:
            self.time_in_position += 1
            new_price = self.close_prices[self.current_step]
            self.unrealized_pnl = self._calculate_unrealized_pnl(new_price)
            # Track max unrealized profit
            self.max_unrealized_pnl = max(self.max_unrealized_pnl, self.unrealized_pnl)
        else:
            self.time_in_position = 0

        # Update equity curve
        current_equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(current_equity)

        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)

        # Update trading restriction based on drawdown
        if self.current_drawdown >= settings.MAX_DRAWDOWN:
            self.trading_restricted = True
        elif self.current_drawdown < settings.DRAWDOWN_REDUCE_THRESHOLD * 0.5:
            # Only lift restriction when recovered to half the threshold
            self.trading_restricted = False

        # Update reward calculator's equity tracking (for composite reward)
        self.reward_calculator.update_equity(current_equity)

        # Calculate return
        step_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        self.returns_history.append(step_return)

        # Check termination
        terminated = False
        truncated = False

        if self.current_step >= self.data_len - 1:
            truncated = True

        if current_equity <= 0:
            terminated = True
            reward = -10.0

        if current_equity < self.initial_balance - settings.MAX_DAILY_LOSS:
            terminated = True
            reward = -5.0

        # Terminate on max drawdown exceeded
        if self.current_drawdown >= settings.MAX_DRAWDOWN:
            terminated = True
            reward = -3.0  # Strong penalty for hitting max drawdown

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _check_sl(self, high_price: float, low_price: float) -> Tuple[float, bool]:
        """Check if stop loss was hit."""
        if self.position == 0:
            return 0.0, False

        triggered = False
        reward = 0.0

        if self.position == 1:  # Long
            if low_price <= self.stop_loss_price:
                reward = self._close_position_at_price(self.stop_loss_price, "STOP_LOSS")
                triggered = True
                self.sl_hits += 1

        elif self.position == -1:  # Short
            if high_price >= self.stop_loss_price:
                reward = self._close_position_at_price(self.stop_loss_price, "STOP_LOSS")
                triggered = True
                self.sl_hits += 1

        return reward, triggered

    def _execute_action(self, action: Action, current_price: float, sl_level: int = 0) -> float:
        """Execute trading action."""
        reward = 0.0

        # Check if in deep drawdown - restrict new positions but allow closing
        in_drawdown_restriction = self.current_drawdown >= settings.DRAWDOWN_REDUCE_THRESHOLD

        if action == Action.HOLD:
            if self.position != 0:
                reward = self._calculate_holding_reward()

        elif action == Action.BUY:
            if self.position == -1:
                # Always allow closing positions
                reward = self._close_position(current_price)
                # Only open new position if not in deep drawdown
                if not in_drawdown_restriction:
                    self._open_position(1, current_price, sl_level)
            elif self.position == 0 and not in_drawdown_restriction:
                # Only open new position if not in deep drawdown
                self._open_position(1, current_price, sl_level)
            elif self.position == 0 and in_drawdown_restriction:
                # Penalty for trying to trade during drawdown restriction
                reward = -0.1

        elif action == Action.SELL:
            if self.position == 1:
                # Always allow closing positions
                reward = self._close_position(current_price)
                # Only open new position if not in deep drawdown
                if not in_drawdown_restriction:
                    self._open_position(-1, current_price, sl_level)
            elif self.position == 0 and not in_drawdown_restriction:
                # Only open new position if not in deep drawdown
                self._open_position(-1, current_price, sl_level)
            elif self.position == 0 and in_drawdown_restriction:
                # Penalty for trying to trade during drawdown restriction
                reward = -0.1

        elif action == Action.CLOSE:
            if self.position != 0:
                reward = self._close_position(current_price)

        return reward

    def _calculate_holding_reward(self) -> float:
        """Calculate reward for holding - encourage letting winners run, cutting losers."""
        if self.position == 0:
            return 0.0

        # Small positive reward for holding profitable positions
        # Small negative reward for holding losing positions
        pnl_normalized = self.unrealized_pnl / 100.0

        if self.unrealized_pnl > 0:
            # Holding a winner - small encouragement
            return pnl_normalized * 0.001
        else:
            # Holding a loser - increasing pressure to close
            # Penalty increases with time in losing position
            time_factor = min(self.time_in_position / 50.0, 1.0)
            return pnl_normalized * 0.002 * (1 + time_factor)

    def _open_position(self, direction: int, current_price: float, sl_level: int = 0) -> None:
        """Open a new position with chosen stop loss level."""
        self.position = direction
        self.max_unrealized_pnl = 0.0

        # Set SL based on chosen level
        sl_level = min(sl_level, len(self.sl_options) - 1)
        self.current_sl_level = sl_level
        self.current_sl_ticks = self.sl_options[sl_level]
        sl_points = self.current_sl_ticks * settings.TICK_SIZE

        # Track SL choice
        self.sl_choices.append(sl_level)

        # Calculate entry with slippage
        if direction == 1:  # Long
            self.entry_price = current_price + self.slippage_ticks * settings.TICK_SIZE
            self.stop_loss_price = self.entry_price - sl_points
        else:  # Short
            self.entry_price = current_price - self.slippage_ticks * settings.TICK_SIZE
            self.stop_loss_price = self.entry_price + sl_points

        self.balance -= self.commission
        self.time_in_position = 0

    def _close_position(self, current_price: float) -> float:
        """Close position at market."""
        if self.position == 0:
            return 0.0

        if self.position == 1:
            exit_price = current_price - self.slippage_ticks * settings.TICK_SIZE
        else:
            exit_price = current_price + self.slippage_ticks * settings.TICK_SIZE

        return self._close_position_at_price(exit_price, "AGENT_CLOSE")

    def _close_position_at_price(self, exit_price: float, exit_type: str) -> float:
        """Close position at specified price."""
        if self.position == 0:
            return 0.0

        # Calculate P&L
        price_diff = exit_price - self.entry_price
        pnl = price_diff * self.position * settings.POINT_VALUE
        pnl -= self.commission

        # Update balance
        self.balance += pnl
        self.realized_pnl += pnl

        # Update statistics
        if pnl > 0:
            self.winning_trades += 1
            self.total_wins += pnl
        else:
            self.losing_trades += 1
            self.total_losses += abs(pnl)

        # Record trade
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "position": self.position,
            "pnl": pnl,
            "time_in_position": self.time_in_position,
            "step": self.current_step,
            "exit_type": exit_type,
            "max_unrealized_pnl": self.max_unrealized_pnl,
        })

        # Calculate reward
        base_reward = self.reward_calculator.calculate_trade_reward(
            pnl=pnl,
            position=self.position,
            time_in_position=self.time_in_position,
            returns_history=self.returns_history,
        )

        # Bonus/penalty for different exit types
        if exit_type == "AGENT_CLOSE":
            if pnl > 0:
                # Reward for taking profit - bonus if captured most of the max
                if self.max_unrealized_pnl > 0:
                    capture_ratio = pnl / self.max_unrealized_pnl
                    base_reward *= (1.0 + 0.3 * capture_ratio)
            else:
                # SIGNIFICANT BONUS for cutting losses early (before SL)
                max_sl_loss = self.current_sl_ticks * settings.TICK_SIZE * settings.POINT_VALUE
                if max_sl_loss > 0:
                    loss_ratio = abs(pnl) / max_sl_loss
                    # The earlier you cut, the bigger the bonus
                    if loss_ratio < 0.3:  # Cut at less than 30% of max loss
                        base_reward *= 0.4  # 60% reduction in penalty - great exit!
                        base_reward += 0.5  # Bonus for smart early exit
                    elif loss_ratio < 0.5:  # Cut at less than 50%
                        base_reward *= 0.6  # 40% reduction in penalty
                        base_reward += 0.3  # Small bonus
                    elif loss_ratio < 0.7:  # Cut at less than 70%
                        base_reward *= 0.8  # 20% reduction
                    # else: no bonus for cutting close to SL

        elif exit_type == "STOP_LOSS":
            # PENALTY for hitting stop loss - you should have closed earlier!
            # Tight SL (low level) = less penalty, Wide SL (high level) = more penalty
            sl_penalty_factor = 1.3 + (self.current_sl_level * 0.2)
            base_reward *= sl_penalty_factor
            base_reward -= 0.3  # Additional flat penalty for hitting SL

        # Reset state
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.time_in_position = 0
        self.stop_loss_price = 0.0
        self.max_unrealized_pnl = 0.0

        return base_reward

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0.0
        price_diff = current_price - self.entry_price
        return price_diff * self.position * settings.POINT_VALUE

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step

        market_features = self.market_features_data[start_idx:end_idx]

        # Position info
        current_equity = self.balance + self.unrealized_pnl
        balance_ratio = current_equity / self.initial_balance

        # How close to SL (0 = at SL, 1 = at entry)
        if self.position != 0:
            sl_points = self.current_sl_ticks * settings.TICK_SIZE
            max_loss = sl_points * settings.POINT_VALUE
            pnl_to_sl_ratio = (self.unrealized_pnl + max_loss) / max_loss if max_loss > 0 else 1.0
            # How much of max profit are we keeping
            max_pnl_ratio = self.unrealized_pnl / max(1.0, self.max_unrealized_pnl) if self.max_unrealized_pnl > 0 else 1.0
        else:
            pnl_to_sl_ratio = 1.0
            max_pnl_ratio = 1.0

        # Normalize current SL level
        sl_level_norm = self.current_sl_level / max(1, len(self.sl_options) - 1) if self.position != 0 else 0.5

        position_info = np.array([
            self.position,
            self.unrealized_pnl / self.initial_balance,
            min(self.time_in_position / 100, 1.0),
            balance_ratio,
            pnl_to_sl_ratio,   # Distance to SL
            max_pnl_ratio,     # Current P&L vs max P&L in trade
            sl_level_norm,     # Current SL level (normalized)
        ], dtype=np.float32)

        return {
            "market_features": market_features,
            "position_info": position_info,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        profit_factor = self.total_wins / max(0.01, self.total_losses)

        # SL choice distribution
        sl_distribution = {}
        if self.sl_choices:
            for i in range(len(self.sl_options)):
                sl_distribution[f"sl_{i}"] = self.sl_choices.count(i) / len(self.sl_choices)

        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_trades": len(self.trades),
            "equity": self.balance + self.unrealized_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sl_hits": self.sl_hits,
            "sl_distribution": sl_distribution,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown_reached,
            "trading_restricted": self.trading_restricted,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            current_price = self.close_prices[self.current_step]
            equity = self.balance + self.unrealized_pnl
            print(
                f"Step: {self.current_step} | "
                f"Price: {current_price:.2f} | "
                f"Position: {self.position} | "
                f"Equity: ${equity:.2f} | "
                f"Unrealized: ${self.unrealized_pnl:.2f} | "
                f"Trades: {len(self.trades)}"
            )
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)
