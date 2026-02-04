"""Visualization utilities for trading analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)


class TradingVisualizer:
    """
    Creates visualizations for trading performance and analysis.
    """

    def __init__(self, use_plotly: bool = True):
        """
        Initialize visualizer.

        Args:
            use_plotly: Use Plotly for interactive charts (fallback to matplotlib)
        """
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE

        if not self.use_plotly and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither Plotly nor Matplotlib is available")

    def plot_equity_curve(
        self,
        equity_curve: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        title: str = "Equity Curve",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot equity curve with optional benchmark.

        Args:
            equity_curve: Array of equity values
            benchmark: Optional benchmark equity curve
            title: Chart title
            save_path: Path to save figure
        """
        if self.use_plotly:
            self._plot_equity_plotly(equity_curve, benchmark, title, save_path)
        else:
            self._plot_equity_matplotlib(equity_curve, benchmark, title, save_path)

    def _plot_equity_plotly(
        self,
        equity_curve: np.ndarray,
        benchmark: Optional[np.ndarray],
        title: str,
        save_path: Optional[Path],
    ) -> None:
        """Plot equity curve using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, "Drawdown"),
            vertical_spacing=0.1,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                y=equity_curve,
                mode="lines",
                name="Strategy",
                line=dict(color="blue"),
            ),
            row=1, col=1,
        )

        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    y=benchmark,
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="gray", dash="dash"),
                ),
                row=1, col=1,
            )

        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(
                y=drawdown,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red"),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_white",
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def _plot_equity_matplotlib(
        self,
        equity_curve: np.ndarray,
        benchmark: Optional[np.ndarray],
        title: str,
        save_path: Optional[Path],
    ) -> None:
        """Plot equity curve using Matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        # Equity curve
        ax1.plot(equity_curve, label="Strategy", color="blue")
        if benchmark is not None:
            ax1.plot(benchmark, label="Benchmark", color="gray", linestyle="--")
        ax1.set_title(title)
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
        ax2.plot(drawdown, color="red")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Bar")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_trades_on_price(
        self,
        df: pd.DataFrame,
        trades: List[Dict],
        title: str = "Trades on Price Chart",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot trades overlaid on price chart.

        Args:
            df: OHLCV DataFrame
            trades: List of trade dictionaries
            title: Chart title
            save_path: Path to save figure
        """
        if self.use_plotly:
            self._plot_trades_plotly(df, trades, title, save_path)
        else:
            self._plot_trades_matplotlib(df, trades, title, save_path)

    def _plot_trades_plotly(
        self,
        df: pd.DataFrame,
        trades: List[Dict],
        title: str,
        save_path: Optional[Path],
    ) -> None:
        """Plot trades using Plotly with candlestick chart."""
        fig = go.Figure()

        # For large datasets, use OHLC with line for better visibility
        if len(df) > 1000:
            # Use line chart with high-low range shading for better visibility
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["high"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["low"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(200, 200, 200, 0.3)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Close price line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["close"],
                    mode="lines",
                    line=dict(color="blue", width=1),
                    name="Close Price",
                )
            )
        else:
            # Candlestick chart for smaller datasets
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            )

        # Collect all entries and exits for batch plotting (more efficient)
        long_entries_x, long_entries_y = [], []
        long_exits_x, long_exits_y = [], []
        short_entries_x, short_entries_y = [], []
        short_exits_x, short_exits_y = [], []

        for trade in trades:
            side = trade.get("side", trade.get("position", 0))
            if side > 0:  # Long
                long_entries_x.append(trade["entry_time"])
                long_entries_y.append(trade["entry_price"])
                long_exits_x.append(trade["exit_time"])
                long_exits_y.append(trade["exit_price"])
            else:  # Short
                short_entries_x.append(trade["entry_time"])
                short_entries_y.append(trade["entry_price"])
                short_exits_x.append(trade["exit_time"])
                short_exits_y.append(trade["exit_price"])

        # Long entries (green triangles pointing up)
        if long_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=long_entries_x,
                    y=long_entries_y,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="lime",
                                line=dict(width=1, color="darkgreen")),
                    name="Long Entry",
                )
            )

        # Long exits (green x)
        if long_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=long_exits_x,
                    y=long_exits_y,
                    mode="markers",
                    marker=dict(symbol="x", size=6, color="darkgreen"),
                    name="Long Exit",
                )
            )

        # Short entries (red triangles pointing down)
        if short_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=short_entries_x,
                    y=short_entries_y,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="red",
                                line=dict(width=1, color="darkred")),
                    name="Short Entry",
                )
            )

        # Short exits (red x)
        if short_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=short_exits_x,
                    y=short_exits_y,
                    mode="markers",
                    marker=dict(symbol="x", size=6, color="darkred"),
                    name="Short Exit",
                )
            )

        fig.update_layout(
            title=title,
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            yaxis_title="Price",
            xaxis_title="Time",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def _plot_trades_matplotlib(
        self,
        df: pd.DataFrame,
        trades: List[Dict],
        title: str,
        save_path: Optional[Path],
    ) -> None:
        """Plot trades using Matplotlib with candlestick-style chart."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        # Simple candlestick representation using bar chart
        up = df[df["close"] >= df["open"]]
        down = df[df["close"] < df["open"]]

        # Plot wicks
        ax.vlines(df.index, df["low"], df["high"], color="black", linewidth=0.5)

        # Plot bodies
        width = 0.6
        ax.bar(up.index, up["close"] - up["open"], width, bottom=up["open"], color="green", edgecolor="darkgreen")
        ax.bar(down.index, down["open"] - down["close"], width, bottom=down["close"], color="red", edgecolor="darkred")

        # Collect trades by type
        for trade in trades:
            side = trade.get("side", trade.get("position", 0))
            if side > 0:
                entry_color = "lime"
                exit_color = "darkgreen"
                marker = "^"
            else:
                entry_color = "red"
                exit_color = "darkred"
                marker = "v"

            ax.scatter(
                [trade["entry_time"]],
                [trade["entry_price"]],
                color=entry_color,
                marker=marker,
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            ax.scatter(
                [trade["exit_time"]],
                [trade["exit_price"]],
                color=exit_color,
                marker="x",
                s=30,
                linewidths=1,
                zorder=5,
            )

        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        title: str = "Returns Distribution",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot distribution of returns.

        Args:
            returns: Array of returns
            title: Chart title
            save_path: Path to save figure
        """
        if self.use_plotly:
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name="Returns",
                )
            )

            # Add normal distribution overlay
            mean = np.mean(returns) * 100
            std = np.std(returns) * 100
            x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            y = y * len(returns) * (returns.max() - returns.min()) / 50 * 100

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name="Normal Fit",
                    line=dict(color="red"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                template="plotly_white",
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.hist(returns * 100, bins=50, density=True, alpha=0.7, label="Returns")

            # Normal fit
            mean = np.mean(returns) * 100
            std = np.std(returns) * 100
            x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
            plt.plot(
                x,
                (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2),
                "r-",
                label="Normal Fit",
            )

            plt.title(title)
            plt.xlabel("Return (%)")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()

    def plot_monthly_returns_heatmap(
        self,
        equity_curve: np.ndarray,
        dates: pd.DatetimeIndex,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot monthly returns as a heatmap.

        Args:
            equity_curve: Equity curve array
            dates: Datetime index
            title: Chart title
            save_path: Path to save figure
        """
        # Calculate monthly returns
        equity_series = pd.Series(equity_curve, index=dates)
        monthly_equity = equity_series.resample("M").last()
        monthly_returns = monthly_equity.pct_change() * 100

        # Pivot to year x month
        df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")

        if self.use_plotly:
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    y=pivot.index,
                    colorscale="RdYlGn",
                    zmid=0,
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text}%",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Month",
                yaxis_title="Year",
                template="plotly_white",
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            plt.figure(figsize=(12, 6))
            plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            plt.colorbar(label="Return (%)")

            plt.xticks(
                range(12),
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            )
            plt.yticks(range(len(pivot.index)), pivot.index)

            plt.title(title)
            plt.xlabel("Month")
            plt.ylabel("Year")

            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()

    def plot_performance_summary(
        self,
        metrics: Dict[str, Any],
        equity_curve: np.ndarray,
        trades: List[Dict],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create comprehensive performance summary dashboard.

        Args:
            metrics: Performance metrics dictionary
            equity_curve: Equity curve array
            trades: List of trades
            save_path: Path to save figure
        """
        if not self.use_plotly:
            logger.warning("Performance summary requires Plotly")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Equity Curve",
                "Trade P&L Distribution",
                "Drawdown",
                "Cumulative P&L",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(y=equity_curve, mode="lines", name="Equity"),
            row=1, col=1,
        )

        # Trade P&L distribution
        pnls = [t["pnl"] for t in trades]
        fig.add_trace(
            go.Histogram(x=pnls, nbinsx=30, name="Trade P&L"),
            row=1, col=2,
        )

        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(y=drawdown, mode="lines", fill="tozeroy", name="Drawdown"),
            row=2, col=1,
        )

        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        fig.add_trace(
            go.Scatter(y=cumulative_pnl, mode="lines", name="Cumulative P&L"),
            row=2, col=2,
        )

        # Add metrics as annotation
        metrics_text = (
            f"Total Return: {metrics.get('total_return_pct', 0):.1f}%<br>"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}<br>"
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%<br>"
            f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%<br>"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}"
        )

        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=1.02, y=1,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
        )

        fig.update_layout(
            height=700,
            width=1200,
            title_text="Performance Summary",
            showlegend=False,
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            importances: Array of importance values
            title: Chart title
            save_path: Path to save figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        names = [feature_names[i] for i in indices]
        values = importances[indices]

        if self.use_plotly:
            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=names,
                    orientation="h",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis=dict(autorange="reversed"),
                template="plotly_white",
                height=500,
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            plt.figure(figsize=(10, 8))
            plt.barh(names[::-1], values[::-1])
            plt.xlabel("Importance")
            plt.title(title)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
