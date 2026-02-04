"""Tests for backtesting module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine, OrderSide, simple_momentum_strategy
from backtest.metrics import PerformanceMetrics, calculate_sharpe_ratio, calculate_max_drawdown


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    n_bars = 500
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="5T")

    np.random.seed(42)
    # Create trending data
    trend = np.linspace(0, 50, n_bars)
    noise = np.cumsum(np.random.randn(n_bars) * 2)
    close = 15000 + trend + noise

    df = pd.DataFrame({
        "open": close + np.random.randn(n_bars) * 2,
        "high": close + abs(np.random.randn(n_bars) * 3),
        "low": close - abs(np.random.randn(n_bars) * 3),
        "close": close,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)

    return df


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_initialization(self, sample_data):
        """Test engine initialization."""
        engine = BacktestEngine(
            df=sample_data,
            initial_capital=10000.0,
            commission=0.62,
        )

        assert engine.initial_capital == 10000.0
        assert engine.capital == 10000.0
        assert engine.position.quantity == 0

    def test_reset(self, sample_data):
        """Test engine reset."""
        engine = BacktestEngine(df=sample_data, initial_capital=10000.0)

        # Simulate some trading
        engine.capital = 15000.0
        engine.position.quantity = 1

        engine.reset()

        assert engine.capital == 10000.0
        assert engine.position.quantity == 0
        assert len(engine.trades) == 0

    def test_run_with_simple_strategy(self, sample_data):
        """Test running backtest with simple strategy."""
        engine = BacktestEngine(
            df=sample_data,
            initial_capital=10000.0,
            commission=0.62,
        )

        def always_buy(df, current_bar, position):
            if position.quantity == 0 and current_bar > 20:
                return OrderSide.BUY
            return None

        results = engine.run(strategy=always_buy)

        assert "final_equity" in results
        assert "total_trades" in results
        assert len(results["equity_curve"]) > 0

    def test_run_momentum_strategy(self, sample_data):
        """Test momentum strategy."""
        engine = BacktestEngine(
            df=sample_data,
            initial_capital=10000.0,
        )

        results = engine.run(strategy=simple_momentum_strategy)

        assert results["total_trades"] > 0
        assert len(results["trades"]) == results["total_trades"]

    def test_commission_deduction(self, sample_data):
        """Test commission is properly deducted."""
        engine = BacktestEngine(
            df=sample_data,
            initial_capital=10000.0,
            commission=1.0,
        )

        # Strategy that trades once
        trade_count = [0]

        def trade_once(df, current_bar, position):
            if trade_count[0] == 0 and current_bar > 20:
                trade_count[0] = 1
                return OrderSide.BUY
            elif trade_count[0] == 1 and current_bar > 30:
                trade_count[0] = 2
                return None  # Close via CLOSE action would need modification
            return None

        engine.run(strategy=trade_once)

        # At least one commission should be deducted
        assert engine.capital < 10000.0 or len(engine.trades) > 0

    def test_trade_history(self, sample_data):
        """Test trade history recording."""
        engine = BacktestEngine(df=sample_data, initial_capital=10000.0)

        results = engine.run(strategy=simple_momentum_strategy)

        if len(results["trades"]) > 0:
            trade = results["trades"][0]
            assert hasattr(trade, "entry_price")
            assert hasattr(trade, "exit_price")
            assert hasattr(trade, "pnl")

    def test_get_trade_df(self, sample_data):
        """Test trade DataFrame generation."""
        engine = BacktestEngine(df=sample_data, initial_capital=10000.0)
        engine.run(strategy=simple_momentum_strategy)

        trade_df = engine.get_trade_df()

        if len(trade_df) > 0:
            assert "entry_price" in trade_df.columns
            assert "exit_price" in trade_df.columns
            assert "pnl" in trade_df.columns


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        from backtest.metrics import Trade

        return [
            Trade(
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 11, 0),
                side=1,
                entry_price=15000,
                exit_price=15020,
                quantity=1,
                pnl=40,
                commission=1.24,
                duration_bars=12,
            ),
            Trade(
                entry_time=datetime(2024, 1, 1, 12, 0),
                exit_time=datetime(2024, 1, 1, 13, 0),
                side=-1,
                entry_price=15030,
                exit_price=15050,
                quantity=1,
                pnl=-40,
                commission=1.24,
                duration_bars=12,
            ),
            Trade(
                entry_time=datetime(2024, 1, 1, 14, 0),
                exit_time=datetime(2024, 1, 1, 15, 0),
                side=1,
                entry_price=15040,
                exit_price=15080,
                quantity=1,
                pnl=80,
                commission=1.24,
                duration_bars=12,
            ),
        ]

    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve."""
        return np.array([10000, 10040, 10000, 10080])

    def test_return_metrics(self, sample_trades, sample_equity_curve):
        """Test return calculation."""
        metrics = PerformanceMetrics(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_capital=10000,
        )

        results = metrics._return_metrics()

        assert "total_return_pct" in results
        assert results["total_return_pct"] == pytest.approx(0.8, rel=0.1)

    def test_trade_metrics(self, sample_trades, sample_equity_curve):
        """Test trade metrics calculation."""
        metrics = PerformanceMetrics(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_capital=10000,
        )

        results = metrics._trade_metrics()

        assert results["total_trades"] == 3
        assert results["winning_trades"] == 2
        assert results["losing_trades"] == 1
        assert results["win_rate_pct"] == pytest.approx(66.67, rel=0.1)

    def test_drawdown_metrics(self, sample_trades, sample_equity_curve):
        """Test drawdown calculation."""
        # Create equity curve with drawdown
        equity = np.array([10000, 10500, 10200, 9800, 10100])

        metrics = PerformanceMetrics(
            trades=sample_trades,
            equity_curve=equity,
            initial_capital=10000,
        )

        results = metrics._drawdown_metrics()

        assert "max_drawdown_pct" in results
        assert results["max_drawdown_pct"] > 0

    def test_risk_adjusted_metrics(self, sample_trades, sample_equity_curve):
        """Test risk-adjusted metrics."""
        # Need more data points for meaningful calculation
        equity = np.linspace(10000, 10500, 100) + np.random.randn(100) * 50

        metrics = PerformanceMetrics(
            trades=sample_trades,
            equity_curve=equity,
            initial_capital=10000,
        )

        results = metrics._risk_adjusted_metrics()

        assert "sharpe_ratio" in results
        assert "sortino_ratio" in results
        assert "calmar_ratio" in results

    def test_calculate_all(self, sample_trades, sample_equity_curve):
        """Test full metrics calculation."""
        equity = np.linspace(10000, 10500, 100) + np.random.randn(100) * 50

        metrics = PerformanceMetrics(
            trades=sample_trades,
            equity_curve=equity,
            initial_capital=10000,
        )

        results = metrics.calculate_all()

        assert "total_return_pct" in results
        assert "sharpe_ratio" in results
        assert "win_rate_pct" in results
        assert "max_drawdown_pct" in results


class TestMetricFunctions:
    """Tests for standalone metric functions."""

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Generate returns with positive mean
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005  # Slight positive drift

        sharpe = calculate_sharpe_ratio(returns)

        # Should be a reasonable number
        assert -5 < sharpe < 5

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        # Equity curve with 10% drawdown
        equity = np.array([100, 110, 105, 99, 108])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == pytest.approx(10.0, rel=0.1)

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonic increase."""
        equity = np.array([100, 101, 102, 103, 104])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
