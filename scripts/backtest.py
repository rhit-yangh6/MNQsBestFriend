#!/usr/bin/env python3
"""Script to run backtests on trained models."""

import argparse
from pathlib import Path
import logging
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from models.agent import TradingAgent
from env.trading_env import TradingEnv
from utils.logger import setup_logger
from utils.visualize import TradingVisualizer

import pandas as pd
import numpy as np

logger = setup_logger("backtest", level="INFO")


def load_model_and_config(model_dir: Path):
    """Load trained model and its configuration."""
    # Load feature columns
    with open(model_dir / "feature_columns.txt", "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    # Load scaler
    preprocessor = DataPreprocessor()
    preprocessor.load_scaler(model_dir / "scaler.joblib")

    # Load training config (if exists)
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            training_config = json.load(f)
    else:
        # Default config for backwards compatibility
        training_config = {
            "lookback_window": settings.LOOKBACK_WINDOW,
            "fixed_sl_ticks": 200.0,
        }

    return feature_columns, preprocessor, training_config


def main():
    parser = argparse.ArgumentParser(description="Run backtest on trained model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory or model file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/MNQ_5min_databento.parquet",
        help="Path to test data (parquet file)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.62,
        help="Commission per contract",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=1.0,
        help="Slippage in ticks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slow per-bar inference (more accurate position tracking, 10-50x slower)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for fast inference (default: 1024)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.LOG_DIR / f"backtest_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Initial capital: ${args.initial_capital}")
    logger.info(f"Commission: ${args.commission}")
    logger.info(f"Slippage: {args.slippage} ticks")
    logger.info("=" * 50)

    # Load data
    logger.info("Loading test data...")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} bars")

    # Load model configuration
    if model_path.is_dir():
        model_dir = model_path
        model_file = model_path / "final_model.zip"
    else:
        model_dir = model_path.parent
        model_file = model_path

    feature_columns, preprocessor, training_config = load_model_and_config(model_dir)
    logger.info(f"Using {len(feature_columns)} features")
    logger.info(f"Training config: {training_config}")

    # Prepare data
    logger.info("Preprocessing data...")
    preprocessor_new = DataPreprocessor()
    df = preprocessor_new.clean_data(df)

    # Compute features
    feature_engineer = FeatureEngineer()
    df = feature_engineer.compute_all_features(df)

    # Normalize using saved scaler
    available_features = [f for f in feature_columns if f in df.columns]
    df[available_features] = preprocessor.transform(df[available_features])

    # Get config values
    lookback_window = training_config.get("lookback_window", settings.LOOKBACK_WINDOW)
    fixed_sl_ticks = training_config.get("fixed_sl_ticks", 120.0)

    # Load model
    logger.info("Loading model...")
    # Create a dummy environment for loading
    dummy_env = TradingEnv(
        df=df.iloc[:lookback_window + 100].reset_index(drop=True),
        feature_columns=available_features,
        fixed_sl_ticks=fixed_sl_ticks,
    )

    agent = TradingAgent(algorithm="PPO")
    agent.load(model_file, env=dummy_env)
    logger.info("Model loaded successfully")

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(
        df=df,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage_ticks=args.slippage,
    )

    if args.slow:
        logger.info("Using SLOW per-bar inference mode")
        results = engine.run_with_model(
            model=agent,
            feature_columns=available_features,
            lookback_window=lookback_window,
            deterministic=args.deterministic,
            fixed_sl_ticks=fixed_sl_ticks,
        )
    else:
        # Fast mode is default
        results = engine.run_fast_with_model(
            model=agent,
            feature_columns=available_features,
            lookback_window=lookback_window,
            deterministic=args.deterministic,
            fixed_sl_ticks=fixed_sl_ticks,
            batch_size=args.batch_size,
        )

    # Print results
    logger.info("=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\nReturn Metrics:")
    print(f"  Initial Capital: ${results['initial_capital']:.2f}")
    print(f"  Final Equity: ${results['final_equity']:.2f}")
    print(f"  Total Return: {results.get('total_return_pct', 0):.2f}%")
    print(f"  Annualized Return: {results.get('annualized_return_pct', 0):.2f}%")

    print(f"\nRisk Metrics:")
    print(f"  Volatility (Annual): {results.get('volatility_annual_pct', 0):.2f}%")
    print(f"  Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
    print(f"  VaR (95%): {results.get('var_95_pct', 0):.2f}%")

    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio: {results.get('sortino_ratio', 0):.3f}")
    print(f"  Calmar Ratio: {results.get('calmar_ratio', 0):.3f}")

    print(f"\nTrade Metrics:")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Win Rate: {results.get('win_rate_pct', 0):.1f}%")
    print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"  Avg Trade P&L: ${results.get('avg_trade_pnl', 0):.2f}")
    print(f"  Expectancy: ${results.get('expectancy', 0):.2f}")

    print("=" * 60 + "\n")

    # Save results
    results_to_save = {k: v for k, v in results.items()
                      if not isinstance(v, (np.ndarray, list))}
    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    # Save trade log
    trade_df = engine.get_trade_df()
    if not trade_df.empty:
        trade_df.to_csv(output_dir / "trades.csv", index=False)
        logger.info(f"Trade log saved to {output_dir / 'trades.csv'}")

    # Generate plots
    if not args.no_plots:
        logger.info("Generating plots...")
        visualizer = TradingVisualizer()

        # Equity curve
        visualizer.plot_equity_curve(
            equity_curve=results['equity_curve'],
            title="Backtest Equity Curve",
            save_path=output_dir / "equity_curve.html",
        )

        # Trades on price
        if len(results['trades']) > 0:
            trades_dict = [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "side": t.side,
                    "pnl": t.pnl,
                }
                for t in results['trades']
            ]

            # Show ~500 bars centered around first trades for clear candlesticks
            window_size = 500
            if len(trades_dict) > 0:
                # Find the start of trading activity
                first_trade_time = trades_dict[0]["entry_time"]
                if first_trade_time in df.index:
                    trade_idx = df.index.get_loc(first_trade_time)
                    start_idx = max(0, trade_idx - 50)  # 50 bars before first trade
                    end_idx = min(len(df), start_idx + window_size)
                else:
                    start_idx = 0
                    end_idx = min(len(df), window_size)
            else:
                start_idx = 0
                end_idx = min(len(df), window_size)

            plot_df = df.iloc[start_idx:end_idx]
            plot_trades = [t for t in trades_dict
                          if t["entry_time"] in plot_df.index or t["exit_time"] in plot_df.index]

            visualizer.plot_trades_on_price(
                df=plot_df,
                trades=plot_trades,
                title=f"Trades on Price Chart ({len(plot_trades)} trades shown)",
                save_path=output_dir / "trades_on_price.html",
            )

            # Performance summary
            visualizer.plot_performance_summary(
                metrics=results,
                equity_curve=results['equity_curve'],
                trades=trades_dict,
                save_path=output_dir / "performance_summary.html",
            )

        # Returns distribution
        returns = np.diff(results['equity_curve']) / results['equity_curve'][:-1]
        visualizer.plot_returns_distribution(
            returns=returns,
            title="Returns Distribution",
            save_path=output_dir / "returns_distribution.html",
        )

        logger.info(f"Plots saved to {output_dir}")

    logger.info(f"Backtest complete! Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
