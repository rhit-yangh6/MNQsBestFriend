"""Tests for the trading environment."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.trading_env import TradingEnv, Action
from env.rewards import RewardCalculator, RewardType
from env.position_sizer import PositionSizer


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    n_bars = 500
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="5T")

    np.random.seed(42)
    close = 15000 + np.cumsum(np.random.randn(n_bars) * 5)

    df = pd.DataFrame({
        "open": close + np.random.randn(n_bars) * 2,
        "high": close + abs(np.random.randn(n_bars) * 3),
        "low": close - abs(np.random.randn(n_bars) * 3),
        "close": close,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)

    # Add some features
    df["returns"] = df["close"].pct_change()
    df["momentum"] = df["close"].pct_change(10)
    df["volatility"] = df["returns"].rolling(20).std()
    df = df.dropna()

    return df


@pytest.fixture
def feature_columns():
    """Feature columns for testing."""
    return ["returns", "momentum", "volatility"]


class TestTradingEnv:
    """Tests for TradingEnv class."""

    def test_initialization(self, sample_data, feature_columns):
        """Test environment initialization."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            initial_balance=10000.0,
            lookback_window=50,
        )

        assert env.initial_balance == 10000.0
        assert env.lookback_window == 50
        assert env.n_features == len(feature_columns)

    def test_reset(self, sample_data, feature_columns):
        """Test environment reset."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        obs, info = env.reset()

        assert "market_features" in obs
        assert "position_info" in obs
        assert obs["market_features"].shape == (50, 3)
        assert obs["position_info"].shape == (4,)
        assert info["position"] == 0
        assert info["balance"] == env.initial_balance

    def test_step_hold(self, sample_data, feature_columns):
        """Test HOLD action."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(Action.HOLD)

        assert info["position"] == 0
        assert info["total_trades"] == 0

    def test_step_buy(self, sample_data, feature_columns):
        """Test BUY action."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(Action.BUY)

        assert info["position"] == 1
        assert env.position == 1
        assert env.entry_price > 0

    def test_step_sell(self, sample_data, feature_columns):
        """Test SELL action."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(Action.SELL)

        assert info["position"] == -1
        assert env.position == -1

    def test_close_position(self, sample_data, feature_columns):
        """Test closing a position."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()

        # Open position
        env.step(Action.BUY)
        assert env.position == 1

        # Hold for a bit
        for _ in range(5):
            env.step(Action.HOLD)

        # Close position
        obs, reward, terminated, truncated, info = env.step(Action.CLOSE)

        assert info["position"] == 0
        assert info["total_trades"] == 1

    def test_reversal(self, sample_data, feature_columns):
        """Test position reversal."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()

        # Open long
        env.step(Action.BUY)
        assert env.position == 1

        # Reverse to short
        env.step(Action.SELL)
        assert env.position == -1
        assert env.entry_price > 0

    def test_episode_end(self, sample_data, feature_columns):
        """Test episode termination."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()

        # Run until end
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(Action.HOLD)
            steps += 1
            if steps > 1000:
                break

        assert truncated or terminated

    def test_trade_history(self, sample_data, feature_columns):
        """Test trade history recording."""
        env = TradingEnv(
            df=sample_data,
            feature_columns=feature_columns,
            lookback_window=50,
        )

        env.reset()

        # Make some trades
        env.step(Action.BUY)
        for _ in range(5):
            env.step(Action.HOLD)
        env.step(Action.CLOSE)

        env.step(Action.SELL)
        for _ in range(5):
            env.step(Action.HOLD)
        env.step(Action.CLOSE)

        trade_df = env.get_trade_history()
        assert len(trade_df) == 2
        assert "pnl" in trade_df.columns


class TestRewardCalculator:
    """Tests for RewardCalculator class."""

    def test_pnl_reward_positive(self):
        """Test positive P&L reward."""
        calc = RewardCalculator(reward_type="pnl")
        reward = calc.calculate_trade_reward(
            pnl=100.0,
            position=1,
            time_in_position=10,
        )
        assert reward > 0

    def test_pnl_reward_negative(self):
        """Test negative P&L reward."""
        calc = RewardCalculator(reward_type="pnl")
        reward = calc.calculate_trade_reward(
            pnl=-100.0,
            position=1,
            time_in_position=10,
        )
        assert reward < 0

    def test_asymmetric_reward(self):
        """Test asymmetric reward scaling."""
        calc = RewardCalculator(
            reward_type="asymmetric",
            profit_scale=1.0,
            loss_scale=1.5,
        )

        win_reward = calc.calculate_trade_reward(pnl=100.0, position=1, time_in_position=5)
        loss_reward = calc.calculate_trade_reward(pnl=-100.0, position=1, time_in_position=5)

        # Loss should be penalized more
        assert abs(loss_reward) > abs(win_reward)

    def test_holding_reward_flat(self):
        """Test holding reward when flat."""
        calc = RewardCalculator()
        reward = calc.calculate_holding_reward(
            position=0,
            unrealized_pnl=0,
            time_in_position=0,
        )
        assert reward == 0


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_fixed_size(self):
        """Test fixed position sizing."""
        sizer = PositionSizer(max_position=5, method="fixed")
        size = sizer.calculate_size(
            account_balance=10000,
            entry_price=15000,
            stop_loss_price=14950,
        )
        assert size == 1  # Default fixed size

    def test_fixed_fractional(self):
        """Test fixed fractional position sizing."""
        sizer = PositionSizer(
            max_position=5,
            risk_per_trade=0.02,
            method="fixed_fractional",
        )
        size = sizer.calculate_size(
            account_balance=10000,
            entry_price=15000,
            stop_loss_price=14950,  # 50 point stop
        )
        assert size >= 1
        assert size <= 5

    def test_position_limit(self):
        """Test maximum position limit."""
        sizer = PositionSizer(max_position=3, method="fixed_fractional")
        size = sizer.calculate_size(
            account_balance=100000,  # Large account
            entry_price=15000,
            stop_loss_price=14990,  # Small stop
        )
        assert size <= 3

    def test_stop_loss_calculation(self):
        """Test stop-loss price calculation."""
        sizer = PositionSizer()

        # Long position
        long_stop = sizer.calculate_stop_loss(
            entry_price=15000,
            position_direction=1,
            atr=20,
            multiplier=2.0,
        )
        assert long_stop < 15000

        # Short position
        short_stop = sizer.calculate_stop_loss(
            entry_price=15000,
            position_direction=-1,
            atr=20,
            multiplier=2.0,
        )
        assert short_stop > 15000

    def test_take_profit_calculation(self):
        """Test take-profit price calculation."""
        sizer = PositionSizer()

        # Long position with 2:1 R:R
        tp = sizer.calculate_take_profit(
            entry_price=15000,
            stop_loss_price=14950,
            position_direction=1,
            risk_reward_ratio=2.0,
        )
        assert tp == 15100  # 50 * 2 = 100 profit target

    def test_track_trade(self):
        """Test trade tracking for Kelly criterion."""
        sizer = PositionSizer()

        # Track some trades
        sizer.track_trade(100, True)
        sizer.track_trade(-50, False)
        sizer.track_trade(75, True)
        sizer.track_trade(-40, False)

        stats = sizer.get_stats()
        assert stats["total_trades"] == 4
        assert stats["win_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
