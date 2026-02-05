"""
Reward: mnq_equity_drawdown_control_v1

Objective: Maximize equity growth while controlling drawdown and volatility.

Formula:
    reward = profit - drawdown_increase - volatility - turnover
             - overholding - over_idle + close_bonus + stoploss_hit
"""

import numpy as np


class RewardCalculator:
    """
    Equity-based reward with drawdown control.

    reward = profit - drawdown_increase - volatility - turnover
             - overholding - over_idle + close_bonus + stoploss_hit
    """

    # === HYPERPARAMETERS ===
    USD_PER_REWARD_UNIT = 100.0

    PROFIT_SCALE = 1.0
    DRAWDOWN_PENALTY = 4.0
    VOLATILITY_PENALTY = 0.2
    TURNOVER_PENALTY = 0.03

    HOLD_THRESHOLD_BARS = 30
    HOLD_PENALTY = 0.02

    IDLE_THRESHOLD_BARS = 120
    IDLE_PENALTY = 0.01

    CLOSE_BONUS_WEIGHT = 0.2
    CLOSE_BONUS_SCALE_USD = 200.0

    STOPLOSS_EXTRA_PENALTY = 1.0

    def __init__(self, reward_scaling: float = 1.0):
        self.reward_scaling = reward_scaling
        self._equity_prev = 0.0
        self._equity_peak = 0.0
        self._drawdown_prev = 0.0
        self._position_prev = 0
        self._flat_steps = 0
        self._position_steps = 0

    def reset(self, initial_equity: float) -> None:
        """Reset state for new episode."""
        self._equity_prev = initial_equity
        self._equity_peak = initial_equity
        self._drawdown_prev = 0.0
        self._position_prev = 0
        self._flat_steps = 0
        self._position_steps = 0

    def calculate_step_reward(
        self,
        current_equity: float,
        position: int,
        unrealized_pnl: float,
        trade_just_closed: bool,
        trade_pnl: float,
        bars_held: int,
        stop_loss_hit: bool = False,
    ) -> float:
        """
        Calculate reward for current step.

        Args:
            current_equity: E_t (balance + unrealized, after costs)
            position: Current position (-1, 0, +1)
            unrealized_pnl: Current unrealized PnL (unused but kept for interface)
            trade_just_closed: Whether trade closed this step
            trade_pnl: Realized PnL if trade closed
            bars_held: Bars held (unused, we track internally)
            stop_loss_hit: Whether stop loss triggered

        Returns:
            Scalar reward
        """
        # === INPUTS ===
        equity_now = current_equity
        equity_prev = self._equity_prev
        equity_peak = self._equity_peak
        drawdown_prev = self._drawdown_prev
        position_now = position
        position_prev = self._position_prev
        flat_steps = self._flat_steps
        position_steps = self._position_steps

        # === DERIVED METRICS ===
        step_return = (equity_now - equity_prev) / self.USD_PER_REWARD_UNIT

        # Update peak before calculating drawdown
        equity_peak = max(equity_peak, equity_now)
        drawdown_now = max(0.0, 1.0 - equity_now / max(equity_peak, 1e-9))

        # === REWARD TERMS ===

        # 1. Profit (bounded by tanh)
        profit = np.tanh(self.PROFIT_SCALE * step_return)

        # 2. Drawdown increase penalty
        drawdown_increase = self.DRAWDOWN_PENALTY * max(0.0, drawdown_now - drawdown_prev)

        # 3. Volatility penalty (penalize large swings)
        volatility = self.VOLATILITY_PENALTY * step_return * step_return

        # 4. Turnover penalty (penalize position changes)
        turnover = self.TURNOVER_PENALTY * abs(position_now - position_prev)

        # 5. Overholding penalty (in position too long)
        overholding = 0.0
        if position_now != 0 and position_steps > self.HOLD_THRESHOLD_BARS:
            overholding = self.HOLD_PENALTY * (
                (position_steps - self.HOLD_THRESHOLD_BARS) / self.HOLD_THRESHOLD_BARS
            )

        # 6. Over-idle penalty (flat too long)
        over_idle = 0.0
        if position_now == 0 and flat_steps > self.IDLE_THRESHOLD_BARS:
            over_idle = self.IDLE_PENALTY * (
                (flat_steps - self.IDLE_THRESHOLD_BARS) / self.IDLE_THRESHOLD_BARS
            )

        # 7. Close bonus (credit assignment for trade completion)
        close_bonus = 0.0
        if trade_just_closed:
            close_bonus = self.CLOSE_BONUS_WEIGHT * np.tanh(
                trade_pnl / self.CLOSE_BONUS_SCALE_USD
            )

        # 8. Stop loss penalty
        stoploss_hit = -self.STOPLOSS_EXTRA_PENALTY if stop_loss_hit else 0.0

        # === FINAL REWARD ===
        reward = (
            profit
            - drawdown_increase
            - volatility
            - turnover
            - overholding
            - over_idle
            + close_bonus
            + stoploss_hit
        )

        # === STATE UPDATES ===
        self._equity_prev = equity_now
        self._equity_peak = equity_peak
        self._drawdown_prev = drawdown_now
        self._position_prev = position_now

        if position_now == 0:
            self._flat_steps += 1
            self._position_steps = 0
        else:
            self._position_steps += 1
            self._flat_steps = 0

        return reward * self.reward_scaling

    def get_episode_return(self, final_equity: float) -> float:
        """Get total episode return percentage."""
        if self._equity_peak <= 0:
            return 0.0
        return (final_equity / self._equity_peak - 1.0) * 100.0
