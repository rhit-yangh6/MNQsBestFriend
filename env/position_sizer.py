"""Position sizing logic for risk management."""

import numpy as np
from typing import Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates optimal position sizes based on various methods.

    Supports:
    - Fixed fractional (risk % of account)
    - Kelly criterion
    - Volatility-based sizing
    - Fixed size
    """

    def __init__(
        self,
        max_position: int = 5,
        risk_per_trade: float = 0.02,
        method: str = "fixed_fractional",
    ):
        """
        Initialize position sizer.

        Args:
            max_position: Maximum position size in contracts
            risk_per_trade: Fraction of account to risk per trade
            method: Sizing method ('fixed', 'fixed_fractional', 'kelly', 'volatility')
        """
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.method = method

        # Kelly criterion parameters (updated via track_trade)
        self.win_rate = 0.5
        self.avg_win = 1.0
        self.avg_loss = 1.0
        self.trade_history = []

    def calculate_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        current_atr: Optional[float] = None,
        win_probability: Optional[float] = None,
    ) -> int:
        """
        Calculate position size.

        Args:
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            current_atr: Current ATR for volatility-based sizing
            win_probability: Model's predicted win probability (for Kelly)

        Returns:
            Position size in contracts (always >= 1 if trading)
        """
        if self.method == "fixed":
            return self._fixed_size()

        elif self.method == "fixed_fractional":
            return self._fixed_fractional_size(
                account_balance, entry_price, stop_loss_price
            )

        elif self.method == "kelly":
            return self._kelly_size(
                account_balance, entry_price, stop_loss_price, win_probability
            )

        elif self.method == "volatility":
            return self._volatility_size(
                account_balance, current_atr
            )

        else:
            return self._fixed_size()

    def _fixed_size(self) -> int:
        """Return fixed position size."""
        return settings.DEFAULT_POSITION_SIZE

    def _fixed_fractional_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """
        Calculate position size based on fixed fractional method.

        Risk a fixed percentage of account on each trade.

        Args:
            account_balance: Current balance
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            Position size
        """
        # Calculate risk amount
        risk_amount = account_balance * self.risk_per_trade

        # Calculate risk per contract
        price_risk = abs(entry_price - stop_loss_price)
        risk_per_contract = price_risk * settings.POINT_VALUE

        if risk_per_contract <= 0:
            return 1

        # Calculate position size
        position_size = int(risk_amount / risk_per_contract)

        # Apply limits
        position_size = max(1, min(position_size, self.max_position))

        logger.debug(
            f"Fixed fractional: risk=${risk_amount:.2f}, "
            f"risk_per_contract=${risk_per_contract:.2f}, "
            f"size={position_size}"
        )

        return position_size

    def _kelly_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        win_probability: Optional[float] = None,
    ) -> int:
        """
        Calculate position size using Kelly criterion.

        Kelly fraction = W - (1-W)/R
        where W = win probability, R = win/loss ratio

        Args:
            account_balance: Current balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            win_probability: Probability of winning (default: historical)

        Returns:
            Position size
        """
        # Use provided probability or historical win rate
        w = win_probability if win_probability is not None else self.win_rate

        # Win/loss ratio
        if self.avg_loss > 0:
            r = self.avg_win / self.avg_loss
        else:
            r = 1.0

        # Kelly fraction
        kelly_fraction = w - (1 - w) / r

        # Use half-Kelly for safety
        kelly_fraction = kelly_fraction / 2

        # Clamp to reasonable range
        kelly_fraction = max(0, min(kelly_fraction, self.risk_per_trade * 2))

        if kelly_fraction <= 0:
            return 1

        # Calculate risk amount based on Kelly
        risk_amount = account_balance * kelly_fraction

        # Calculate risk per contract
        price_risk = abs(entry_price - stop_loss_price)
        risk_per_contract = price_risk * settings.POINT_VALUE

        if risk_per_contract <= 0:
            return 1

        # Calculate position size
        position_size = int(risk_amount / risk_per_contract)
        position_size = max(1, min(position_size, self.max_position))

        logger.debug(
            f"Kelly: W={w:.2f}, R={r:.2f}, "
            f"kelly_fraction={kelly_fraction:.3f}, "
            f"size={position_size}"
        )

        return position_size

    def _volatility_size(
        self,
        account_balance: float,
        current_atr: Optional[float],
    ) -> int:
        """
        Calculate position size based on volatility (ATR).

        Lower position size in high volatility, higher in low volatility.

        Args:
            account_balance: Current balance
            current_atr: Current ATR value

        Returns:
            Position size
        """
        if current_atr is None or current_atr <= 0:
            return self._fixed_size()

        # Target risk amount
        risk_amount = account_balance * self.risk_per_trade

        # Use 2x ATR as stop distance
        stop_distance = current_atr * 2
        risk_per_contract = stop_distance * settings.POINT_VALUE

        if risk_per_contract <= 0:
            return 1

        # Calculate position size
        position_size = int(risk_amount / risk_per_contract)
        position_size = max(1, min(position_size, self.max_position))

        logger.debug(
            f"Volatility: ATR={current_atr:.2f}, "
            f"stop_distance={stop_distance:.2f}, "
            f"size={position_size}"
        )

        return position_size

    def track_trade(self, pnl: float, is_win: bool) -> None:
        """
        Track trade results for Kelly criterion updates.

        Args:
            pnl: Trade P&L
            is_win: Whether trade was profitable
        """
        self.trade_history.append({
            "pnl": pnl,
            "is_win": is_win,
        })

        # Update statistics
        wins = [t for t in self.trade_history if t["is_win"]]
        losses = [t for t in self.trade_history if not t["is_win"]]

        if len(self.trade_history) > 0:
            self.win_rate = len(wins) / len(self.trade_history)

        if len(wins) > 0:
            self.avg_win = np.mean([t["pnl"] for t in wins])

        if len(losses) > 0:
            self.avg_loss = abs(np.mean([t["pnl"] for t in losses]))

    def get_stats(self) -> dict:
        """Get position sizing statistics."""
        return {
            "method": self.method,
            "max_position": self.max_position,
            "risk_per_trade": self.risk_per_trade,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": len(self.trade_history),
        }

    def calculate_stop_loss(
        self,
        entry_price: float,
        position_direction: int,
        atr: float,
        multiplier: float = 2.0,
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            position_direction: 1 for long, -1 for short
            atr: Current ATR value
            multiplier: ATR multiplier for stop distance

        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier

        if position_direction == 1:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        position_direction: int,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            position_direction: 1 for long, -1 for short
            risk_reward_ratio: Target R:R ratio

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward_ratio

        if position_direction == 1:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
