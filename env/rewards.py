"""Reward function definitions for RL trading environment."""

import numpy as np
from typing import List, Optional
from enum import Enum
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of reward functions."""
    PNL = "pnl"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    RISK_ADJUSTED = "risk_adjusted"
    ASYMMETRIC = "asymmetric"
    PROFIT_FACTOR = "profit_factor"
    COMPOSITE = "composite"


class RewardCalculator:
    """
    Calculates rewards for trading actions.

    Supports multiple reward function types with configurable parameters.
    """

    def __init__(
        self,
        reward_type: str = "pnl",
        risk_penalty: float = 0.1,
        transaction_cost_penalty: float = 0.5,
        holding_penalty: float = 0.0001,
        profit_scale: float = 1.0,
        loss_scale: float = 1.5,  # Penalize losses more than equivalent profits
        sharpe_window: int = 20,
    ):
        """
        Initialize reward calculator.

        Args:
            reward_type: Type of reward function
            risk_penalty: Penalty factor for risk
            transaction_cost_penalty: Additional penalty for trading
            holding_penalty: Small penalty for holding positions (prevents infinite holding)
            profit_scale: Scaling factor for profits
            loss_scale: Scaling factor for losses (asymmetric)
            sharpe_window: Window for Sharpe ratio calculation
        """
        self.reward_type = RewardType(reward_type)
        self.risk_penalty = risk_penalty
        self.transaction_cost_penalty = transaction_cost_penalty
        self.holding_penalty = holding_penalty
        self.profit_scale = profit_scale
        self.loss_scale = loss_scale
        self.sharpe_window = sharpe_window

        # Tracking for composite reward
        self._trade_history: List[float] = []
        self._equity_history: List[float] = []
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._wins: int = 0
        self._losses: int = 0
        self._total_profit: float = 0.0
        self._total_loss: float = 0.0

    def reset_tracking(self, initial_equity: float = 10000.0) -> None:
        """Reset tracking variables for a new episode."""
        self._trade_history = []
        self._equity_history = [initial_equity]
        self._peak_equity = initial_equity
        self._current_drawdown = 0.0
        self._wins = 0
        self._losses = 0
        self._total_profit = 0.0
        self._total_loss = 0.0
        self._consecutive_losses = 0

    def update_equity(self, equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        self._equity_history.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        self._current_drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0

    def calculate_trade_reward(
        self,
        pnl: float,
        position: int,
        time_in_position: int,
        returns_history: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate reward for a completed trade.

        Args:
            pnl: Realized P&L from the trade
            position: Position that was closed
            time_in_position: Number of bars position was held
            returns_history: Historical returns for Sharpe calculation

        Returns:
            Calculated reward
        """
        if self.reward_type == RewardType.PNL:
            return self._pnl_reward(pnl)

        elif self.reward_type == RewardType.SHARPE:
            return self._sharpe_reward(pnl, returns_history)

        elif self.reward_type == RewardType.SORTINO:
            return self._sortino_reward(pnl, returns_history)

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            return self._risk_adjusted_reward(pnl, time_in_position)

        elif self.reward_type == RewardType.ASYMMETRIC:
            return self._asymmetric_reward(pnl)

        elif self.reward_type == RewardType.PROFIT_FACTOR:
            return self._profit_factor_reward(pnl, time_in_position)

        elif self.reward_type == RewardType.COMPOSITE:
            return self._composite_reward(pnl, time_in_position, returns_history)

        else:
            return self._pnl_reward(pnl)

    def calculate_holding_reward(
        self,
        position: int,
        unrealized_pnl: float,
        time_in_position: int,
    ) -> float:
        """
        Calculate reward for holding a position (no trade executed).

        Args:
            position: Current position
            unrealized_pnl: Current unrealized P&L
            time_in_position: Number of bars held

        Returns:
            Holding reward (usually small or zero)
        """
        if position == 0:
            # Flat position, no reward
            return 0.0

        # Small reward based on unrealized P&L
        reward = unrealized_pnl * 0.001

        # Small penalty for holding too long
        if time_in_position > 50:
            reward -= self.holding_penalty * (time_in_position - 50)

        return reward

    def _pnl_reward(self, pnl: float) -> float:
        """
        Simple P&L-based reward.

        Args:
            pnl: Realized P&L

        Returns:
            Normalized reward
        """
        # Normalize by typical trade size
        normalized_pnl = pnl / 100.0  # Assuming typical trades around $100

        # Apply asymmetric scaling
        if normalized_pnl >= 0:
            return normalized_pnl * self.profit_scale
        else:
            return normalized_pnl * self.loss_scale

    def _sharpe_reward(
        self,
        pnl: float,
        returns_history: Optional[List[float]],
    ) -> float:
        """
        Sharpe ratio-based reward.

        Args:
            pnl: Current P&L
            returns_history: Historical returns

        Returns:
            Sharpe-weighted reward
        """
        base_reward = self._pnl_reward(pnl)

        if returns_history is None or len(returns_history) < self.sharpe_window:
            return base_reward

        # Calculate recent Sharpe ratio
        recent_returns = np.array(returns_history[-self.sharpe_window:])
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8

        sharpe = mean_return / std_return * np.sqrt(252 * 78)  # Annualized

        # Weight reward by Sharpe ratio
        sharpe_weight = np.clip(sharpe / 2.0, 0.5, 2.0)  # Clamp between 0.5 and 2.0

        return base_reward * sharpe_weight

    def _sortino_reward(
        self,
        pnl: float,
        returns_history: Optional[List[float]],
    ) -> float:
        """
        Sortino ratio-based reward (penalizes downside volatility only).

        Args:
            pnl: Current P&L
            returns_history: Historical returns

        Returns:
            Sortino-weighted reward
        """
        base_reward = self._pnl_reward(pnl)

        if returns_history is None or len(returns_history) < self.sharpe_window:
            return base_reward

        # Calculate recent Sortino ratio
        recent_returns = np.array(returns_history[-self.sharpe_window:])
        mean_return = np.mean(recent_returns)

        # Downside deviation
        negative_returns = recent_returns[recent_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
        else:
            downside_std = 1e-8

        sortino = mean_return / (downside_std + 1e-8) * np.sqrt(252 * 78)

        # Weight reward by Sortino ratio
        sortino_weight = np.clip(sortino / 2.0, 0.5, 2.0)

        return base_reward * sortino_weight

    def _risk_adjusted_reward(
        self,
        pnl: float,
        time_in_position: int,
    ) -> float:
        """
        Risk-adjusted reward considering time in position.

        Args:
            pnl: Realized P&L
            time_in_position: Number of bars held

        Returns:
            Risk-adjusted reward
        """
        base_reward = self._pnl_reward(pnl)

        # Penalize long holding times (exposure risk)
        time_penalty = 0.0
        if time_in_position > 20:
            time_penalty = (time_in_position - 20) * self.risk_penalty * 0.01

        # Bonus for quick profitable trades
        time_bonus = 0.0
        if pnl > 0 and time_in_position < 10:
            time_bonus = 0.1 * (10 - time_in_position) / 10

        return base_reward - time_penalty + time_bonus

    def _asymmetric_reward(self, pnl: float) -> float:
        """
        Asymmetric reward that penalizes losses more heavily.

        Args:
            pnl: Realized P&L

        Returns:
            Asymmetric reward
        """
        normalized_pnl = pnl / 100.0

        if normalized_pnl >= 0:
            # Standard scaling for profits
            return normalized_pnl * self.profit_scale
        else:
            # Higher scaling for losses
            return normalized_pnl * self.loss_scale * 1.5

    def _profit_factor_reward(self, pnl: float, time_in_position: int) -> float:
        """
        Profit factor focused reward that encourages:
        - Higher win amounts relative to loss amounts
        - Cutting losses short
        - Letting winners run
        - Efficient use of time (profit per bar)

        Args:
            pnl: Realized P&L
            time_in_position: Bars held

        Returns:
            Profit factor optimized reward
        """
        # Base reward scaled by typical trade
        normalized_pnl = pnl / 50.0  # Smaller normalization for more sensitivity

        if pnl > 0:
            # Winning trade: reward more for larger wins
            # Bonus for quick wins (efficient use of capital)
            efficiency_bonus = 1.0
            if time_in_position > 0:
                efficiency_bonus = min(2.0, 10.0 / time_in_position)

            # Progressive scaling - bigger wins get proportionally more reward
            if normalized_pnl > 1.0:
                reward = normalized_pnl * 1.2 * efficiency_bonus
            else:
                reward = normalized_pnl * efficiency_bonus

        else:
            # Losing trade: penalize more heavily
            # But give slight reduction for cutting losses quickly
            time_factor = 1.0
            if time_in_position < 5:
                time_factor = 0.9  # 10% reduction for quick stop

            # Asymmetric penalty - losses hurt more than equivalent wins help
            reward = normalized_pnl * self.loss_scale * 1.8 * time_factor

        return reward

    def _composite_reward(
        self,
        pnl: float,
        time_in_position: int,
        returns_history: Optional[List[float]] = None,
    ) -> float:
        """
        Composite reward combining multiple factors:
        - Profit (40% weight) - most important
        - Win rate (25% weight) - very important
        - Drawdown penalty (15% weight)
        - Risk-adjusted component (10% weight)
        - Trade efficiency (10% weight)

        Args:
            pnl: Realized P&L from trade
            time_in_position: Bars held
            returns_history: Historical returns

        Returns:
            Composite reward
        """
        # Update tracking
        self._trade_history.append(pnl)
        if pnl > 0:
            self._wins += 1
            self._total_profit += pnl
            self._consecutive_losses = 0
        else:
            self._losses += 1
            self._total_loss += abs(pnl)
            self._consecutive_losses = getattr(self, '_consecutive_losses', 0) + 1

        # === 1. PROFIT COMPONENT (40% weight) ===
        normalized_pnl = pnl / 50.0
        if pnl > 0:
            # Progressive bonus for larger wins
            if pnl >= 500:
                # EXCEPTIONAL trade - $500+ is huge
                profit_reward = normalized_pnl * 1.5 + 2.5
            elif pnl >= 300:
                # GOOD big winner - $300+ is solid
                profit_reward = normalized_pnl * 1.3 + 1.0
            elif pnl >= 100:
                # Decent trade
                profit_reward = normalized_pnl * 1.15 + 0.3
            else:
                profit_reward = normalized_pnl
        else:
            # Heavier penalty for losses
            profit_reward = normalized_pnl * 1.8

        # === 2. WIN RATE COMPONENT (25% weight) - INCREASED ===
        total_trades = self._wins + self._losses
        win_rate_reward = 0.0

        if total_trades >= 3:  # Start tracking earlier
            win_rate = self._wins / total_trades

            # Strong rewards/penalties based on win rate
            if win_rate >= 0.60:
                win_rate_reward = 1.5 + (win_rate - 0.60) * 5.0  # Big bonus for 60%+
            elif win_rate >= 0.50:
                win_rate_reward = 0.5 + (win_rate - 0.50) * 10.0  # Good bonus for 50-60%
            elif win_rate >= 0.45:
                win_rate_reward = 0.0  # Acceptable
            elif win_rate >= 0.40:
                win_rate_reward = (win_rate - 0.45) * 6.0  # -0.3 to 0
            else:
                win_rate_reward = -0.5 + (win_rate - 0.40) * 5.0  # Strong penalty below 40%

            # Extra penalty for consecutive losses (losing streaks hurt win rate)
            consecutive_losses = getattr(self, '_consecutive_losses', 0)
            if consecutive_losses >= 3:
                win_rate_reward -= 0.3 * (consecutive_losses - 2)  # Escalating penalty

        # Immediate feedback: bonus for win, penalty for loss
        if pnl > 0:
            win_rate_reward += 0.2  # Immediate win bonus
        else:
            win_rate_reward -= 0.3  # Immediate loss penalty

        # === 3. DRAWDOWN COMPONENT (20% weight) - STRENGTHENED ===
        # Much stronger penalties for drawdowns
        if self._current_drawdown > 0.15:
            # Severe penalty for >15% drawdown - model should stop trading
            drawdown_penalty = -2.0 - (self._current_drawdown - 0.15) * 10.0
        elif self._current_drawdown > 0.10:
            # Strong penalty for >10% drawdown
            drawdown_penalty = -1.0 - (self._current_drawdown - 0.10) * 20.0
        elif self._current_drawdown > 0.05:
            # Moderate penalty for >5% drawdown
            drawdown_penalty = -0.3 - (self._current_drawdown - 0.05) * 14.0
        elif self._current_drawdown > 0.02:
            # Light penalty starting at 2%
            drawdown_penalty = -(self._current_drawdown - 0.02) * 10.0
        else:
            drawdown_penalty = 0.0

        # Extra penalty for losing during drawdown - compounds the pain
        if pnl < 0 and self._current_drawdown > 0.03:
            drawdown_penalty *= 2.0

        # Bonus for reducing drawdown (recovering)
        if pnl > 0 and self._current_drawdown > 0.05:
            drawdown_penalty += 0.5  # Reward for recovering during drawdown

        # === 4. RISK-ADJUSTED COMPONENT (10% weight) ===
        if self._total_loss > 0:
            profit_factor = self._total_profit / self._total_loss
            if profit_factor >= 2.0:
                risk_reward = 0.5
            elif profit_factor >= 1.5:
                risk_reward = 0.25
            elif profit_factor >= 1.0:
                risk_reward = 0.0
            else:
                risk_reward = -0.3 * (1.0 - profit_factor)
        else:
            risk_reward = 0.3 if self._total_profit > 0 else 0.0

        # === 5. TRADE EFFICIENCY COMPONENT (5% weight) ===
        if pnl > 0:
            # Quick wins are good
            if time_in_position <= 5:
                efficiency_reward = 0.3
            elif time_in_position <= 15:
                efficiency_reward = 0.15
            else:
                efficiency_reward = 0.0
        else:
            # Losing trades - penalize very short ones (bad entry signals)
            if time_in_position <= 1:
                # Super fast losing trade = terrible entry, heavy penalty
                efficiency_reward = -1.0
            elif time_in_position <= 3:
                # Very quick loss = bad entry
                efficiency_reward = -0.5
            elif time_in_position <= 10:
                # Normal loss duration
                efficiency_reward = 0.0
            else:
                # Held too long before stopping = slow to cut losses
                efficiency_reward = -0.3

        # === COMBINE WITH WEIGHTS ===
        # REVISED WEIGHTS:
        # Profit: 45% (Increased from 40%) - The ultimate goal
        # Win Rate: 20% (Decreased from 25%) - Important but profit matters more
        # Drawdown: 20% (Same) - critical for survival
        # Risk-Adjusted: 15% (Increased from 10%) - Emphasize consistency
        # Efficiency: 0% (Removed) - Holding time is less important than profit
        
        total_reward = (
            profit_reward * 0.45 +      # Profit: 45%
            win_rate_reward * 0.20 +    # Win rate: 20%
            drawdown_penalty * 0.20 +   # Drawdown: 20%
            risk_reward * 0.15          # Risk-adjusted: 15%
        )

        return total_reward

    def get_reward_info(self) -> dict:
        """Get information about reward configuration."""
        return {
            "type": self.reward_type.value,
            "risk_penalty": self.risk_penalty,
            "transaction_cost_penalty": self.transaction_cost_penalty,
            "holding_penalty": self.holding_penalty,
            "profit_scale": self.profit_scale,
            "loss_scale": self.loss_scale,
            "sharpe_window": self.sharpe_window,
        }


class CompositeReward:
    """
    Combines multiple reward functions with weights.
    """

    def __init__(
        self,
        reward_types: List[str],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize composite reward.

        Args:
            reward_types: List of reward types to combine
            weights: Weights for each reward type (default: equal weights)
        """
        self.calculators = [
            RewardCalculator(reward_type=rt) for rt in reward_types
        ]

        if weights is None:
            weights = [1.0 / len(reward_types)] * len(reward_types)

        self.weights = weights

    def calculate(
        self,
        pnl: float,
        position: int,
        time_in_position: int,
        returns_history: Optional[List[float]] = None,
    ) -> float:
        """Calculate weighted combination of rewards."""
        total_reward = 0.0

        for calculator, weight in zip(self.calculators, self.weights):
            reward = calculator.calculate_trade_reward(
                pnl=pnl,
                position=position,
                time_in_position=time_in_position,
                returns_history=returns_history,
            )
            total_reward += reward * weight

        return total_reward
