"""Performance metrics for backtesting and evaluation."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade data structure for metrics calculation."""
    entry_time: Any
    exit_time: Any
    side: int
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    commission: float
    duration_bars: int
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for trading strategies.
    """

    def __init__(
        self,
        trades: List[Trade],
        equity_curve: np.ndarray,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.02,  # Annual risk-free rate
        periods_per_year: int = 252 * 276,  # 5-min bars per year (full ETH: 23h * 12 bars/hr)
    ):
        """
        Initialize performance metrics calculator.

        Args:
            trades: List of Trade objects
            equity_curve: Array of equity values over time
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year for annualization
        """
        self.trades = trades
        self.equity_curve = np.array(equity_curve)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Calculate returns
        self.returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        self.returns = np.nan_to_num(self.returns, 0)

        # Extract P&L series
        self.pnl_series = np.array([t.pnl for t in trades]) if trades else np.array([])

    def calculate_all(self) -> Dict[str, Any]:
        """Calculate all performance metrics."""
        metrics = {}

        # Return metrics
        metrics.update(self._return_metrics())

        # Risk metrics
        metrics.update(self._risk_metrics())

        # Trade metrics
        metrics.update(self._trade_metrics())

        # Drawdown metrics
        metrics.update(self._drawdown_metrics())

        # Risk-adjusted metrics
        metrics.update(self._risk_adjusted_metrics())

        return metrics

    def _return_metrics(self) -> Dict[str, float]:
        """Calculate return-based metrics."""
        if len(self.equity_curve) < 2:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "cagr_pct": 0.0,
            }

        total_return = (self.equity_curve[-1] / self.initial_capital) - 1
        n_periods = len(self.equity_curve)

        # Annualized return
        years = n_periods / self.periods_per_year
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        return {
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "cagr_pct": annualized_return * 100,
        }

    def _risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if len(self.returns) < 2:
            return {
                "volatility_annual_pct": 0.0,
                "downside_deviation_pct": 0.0,
                "var_95_pct": 0.0,
                "cvar_95_pct": 0.0,
            }

        # Volatility (annualized)
        volatility = np.std(self.returns) * np.sqrt(self.periods_per_year)

        # Downside deviation
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.std(negative_returns) * np.sqrt(self.periods_per_year)
        else:
            downside_dev = 0.0

        # Value at Risk (95%)
        var_95 = np.percentile(self.returns, 5) if len(self.returns) > 0 else 0.0

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(self.returns[self.returns <= var_95]) if len(self.returns[self.returns <= var_95]) > 0 else 0.0

        return {
            "volatility_annual_pct": volatility * 100,
            "downside_deviation_pct": downside_dev * 100,
            "var_95_pct": var_95 * 100,
            "cvar_95_pct": cvar_95 * 100,
        }

    def _trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_trade_duration": 0.0,
                "expectancy": 0.0,
            }

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        total_trades = len(self.trades)

        win_rate = n_wins / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "total_trades": total_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max([t.pnl for t in self.trades]) if self.trades else 0,
            "largest_loss": min([t.pnl for t in self.trades]) if self.trades else 0,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean(self.pnl_series) if len(self.pnl_series) > 0 else 0,
            "avg_trade_duration": np.mean([t.duration_bars for t in self.trades]),
            "expectancy": expectancy,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

    def _drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        if len(self.equity_curve) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "avg_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "recovery_factor": 0.0,
            }

        # Calculate drawdown series
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - running_max) / running_max
        drawdown_pct = drawdown * 100

        max_dd = abs(np.min(drawdown_pct))
        avg_dd = abs(np.mean(drawdown_pct[drawdown_pct < 0])) if np.any(drawdown_pct < 0) else 0

        # Max drawdown duration
        in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        # Recovery factor
        total_return = self.equity_curve[-1] - self.initial_capital
        recovery_factor = total_return / (max_dd * self.initial_capital / 100) if max_dd > 0 else 0

        return {
            "max_drawdown_pct": max_dd,
            "avg_drawdown_pct": avg_dd,
            "max_drawdown_duration": max_duration,
            "recovery_factor": recovery_factor,
            "drawdown_series": drawdown_pct,
        }

    def _risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if len(self.returns) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "omega_ratio": 0.0,
            }

        # Sharpe Ratio
        excess_returns = self.returns - self.risk_free_rate / self.periods_per_year
        sharpe = (np.mean(excess_returns) / (np.std(self.returns) + 1e-8)) * np.sqrt(self.periods_per_year)

        # Sortino Ratio
        negative_returns = self.returns[self.returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-8
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(self.periods_per_year)

        # Calmar Ratio
        dd_metrics = self._drawdown_metrics()
        max_dd = dd_metrics["max_drawdown_pct"]
        return_metrics = self._return_metrics()
        annual_return = return_metrics["annualized_return_pct"]
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Omega Ratio
        threshold = 0  # Zero return threshold
        gains = self.returns[self.returns > threshold].sum()
        losses = abs(self.returns[self.returns < threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": omega,
        }

    def get_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        if len(self.equity_curve) < 2:
            return pd.Series()

        # Assuming equity curve has datetime index
        equity_series = pd.Series(self.equity_curve)
        monthly = equity_series.resample('M').last()
        monthly_returns = monthly.pct_change().dropna()

        return monthly_returns

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Get detailed trade analysis."""
        if not self.trades:
            return {}

        # Consecutive wins/losses
        pnl_signs = [1 if t.pnl > 0 else -1 for t in self.trades]
        max_consecutive_wins = self._max_consecutive(pnl_signs, 1)
        max_consecutive_losses = self._max_consecutive(pnl_signs, -1)

        # By side analysis
        long_trades = [t for t in self.trades if t.side > 0]
        short_trades = [t for t in self.trades if t.side < 0]

        long_pnl = sum(t.pnl for t in long_trades) if long_trades else 0
        short_pnl = sum(t.pnl for t in short_trades) if short_trades else 0

        long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) * 100 if long_trades else 0
        short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) * 100 if short_trades else 0

        # MFE/MAE analysis
        mfe_values = [t.max_favorable_excursion for t in self.trades]
        mae_values = [t.max_adverse_excursion for t in self.trades]

        return {
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_win_rate_pct": long_win_rate,
            "short_win_rate_pct": short_win_rate,
            "avg_mfe": np.mean(mfe_values) if mfe_values else 0,
            "avg_mae": np.mean(mae_values) if mae_values else 0,
        }

    def _max_consecutive(self, values: List[int], target: int) -> int:
        """Find maximum consecutive occurrences of target value."""
        max_count = 0
        current_count = 0

        for v in values:
            if v == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def print_summary(self) -> None:
        """Print a summary of performance metrics."""
        metrics = self.calculate_all()

        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)

        print(f"\nReturn Metrics:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Volatility (Annual): {metrics['volatility_annual_pct']:.2f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  VaR (95%): {metrics['var_95_pct']:.2f}%")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")

        print(f"\nTrade Metrics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Expectancy: ${metrics['expectancy']:.2f}")
        print(f"  Avg Trade P&L: ${metrics['avg_trade_pnl']:.2f}")

        print("=" * 50 + "\n")


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 276,  # Full ETH: 23h * 12 bars/hr
) -> float:
    """Quick Sharpe ratio calculation."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return (np.mean(excess_returns) / (np.std(returns) + 1e-8)) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Quick max drawdown calculation."""
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return abs(np.min(drawdown)) * 100
