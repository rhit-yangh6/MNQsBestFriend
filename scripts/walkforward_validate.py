#!/usr/bin/env python3
"""
Walk-forward validation for frozen RL model.

Tests a trained model across multiple time windows to validate
out-of-sample performance stability.

Config: walk_forward_validation
- Rolling time windows (2yr train context, 1yr test)
- Frozen policy (no learning)
- Pass/fail criteria
"""

import argparse
from pathlib import Path
import logging
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer
from backtest.engine import BacktestEngine
from models.agent import TradingAgent
from env.trading_env import TradingEnv
from utils.logger import setup_logger

logger = setup_logger("walkforward_validate", level="INFO")


@dataclass
class WindowMetrics:
    """Metrics for a single validation window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_bars: int

    # Core metrics
    total_profit: float
    total_return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float

    # Trade metrics
    total_trades: int
    trades_per_bar: float
    trades_per_day: float
    win_rate_pct: float
    avg_trade_pnl: float

    # Pass/fail
    passed: bool
    fail_reasons: List[str]


@dataclass
class ValidationConfig:
    """Walk-forward validation configuration."""
    train_years: float = 2.0
    test_years: float = 1.0
    step_years: float = 1.0

    # Pass/fail criteria
    min_profit_factor: float = 1.3
    max_drawdown_pct: float = 35.0
    no_negative_return: bool = True

    # Bars per year (full ETH, 5min)
    bars_per_year: int = 252 * 276  # 69,552


def get_time_windows(
    df: pd.DataFrame,
    config: ValidationConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate rolling time windows.

    Returns list of (train_start, train_end, test_start, test_end) indices.
    """
    n_bars = len(df)
    train_bars = int(config.train_years * config.bars_per_year)
    test_bars = int(config.test_years * config.bars_per_year)
    step_bars = int(config.step_years * config.bars_per_year)

    windows = []
    start = 0

    while start + train_bars + test_bars <= n_bars:
        train_start = start
        train_end = start + train_bars
        test_start = train_end
        test_end = min(test_start + test_bars, n_bars)

        windows.append((train_start, train_end, test_start, test_end))
        start += step_bars

    return windows


def validate_window(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
    preprocessor: DataPreprocessor,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    window_id: int,
    config: ValidationConfig,
    fixed_sl_ticks: float = 200.0,
) -> WindowMetrics:
    """Run validation on a single time window."""

    # Get data slices
    train_df = df.iloc[train_start:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    # Fit scaler on train window only (no leakage)
    window_preprocessor = DataPreprocessor(scaler_type="robust")
    window_preprocessor.fit(train_df, feature_columns)

    # Transform test data
    test_df_scaled = window_preprocessor.transform(test_df)

    # Get date range for reporting
    train_start_date = str(train_df.index[0])[:10] if hasattr(train_df.index[0], 'strftime') else str(train_start)
    train_end_date = str(train_df.index[-1])[:10] if hasattr(train_df.index[-1], 'strftime') else str(train_end)
    test_start_date = str(test_df.index[0])[:10] if hasattr(test_df.index[0], 'strftime') else str(test_start)
    test_end_date = str(test_df.index[-1])[:10] if hasattr(test_df.index[-1], 'strftime') else str(test_end)

    logger.info(f"Window {window_id}: Train [{train_start_date} to {train_end_date}], Test [{test_start_date} to {test_end_date}]")

    # Run backtest on test window
    engine = BacktestEngine(
        df=test_df_scaled,
        initial_capital=10000.0,
        commission=0.62,
        slippage_ticks=1.0,
    )

    results = engine.run_fast_with_model(
        model=model,
        feature_columns=feature_columns,
        lookback_window=settings.LOOKBACK_WINDOW,
        deterministic=True,
        fixed_sl_ticks=fixed_sl_ticks,
    )

    # Calculate metrics
    test_bars = len(test_df)
    total_trades = results.get("total_trades", 0)
    trades_per_bar = total_trades / max(test_bars, 1)
    trades_per_day = trades_per_bar * 276  # Full ETH bars per day

    profit_factor = results.get("profit_factor", 0)
    max_drawdown = results.get("max_drawdown_pct", 100)
    total_return = results.get("total_return_pct", results.get("total_return", 0))
    total_profit = results.get("final_equity", 10000) - 10000

    sharpe = results.get("sharpe_ratio", 0)
    calmar = total_return / max(max_drawdown, 0.01)

    win_rate = results.get("win_rate_pct", 0)
    avg_trade_pnl = results.get("avg_trade_pnl", 0)

    # Check pass/fail criteria
    fail_reasons = []

    if profit_factor < config.min_profit_factor:
        fail_reasons.append(f"PF {profit_factor:.2f} < {config.min_profit_factor}")

    if max_drawdown > config.max_drawdown_pct:
        fail_reasons.append(f"MaxDD {max_drawdown:.1f}% > {config.max_drawdown_pct}%")

    if config.no_negative_return and total_return < 0:
        fail_reasons.append(f"Negative return: {total_return:.2f}%")

    passed = len(fail_reasons) == 0

    return WindowMetrics(
        window_id=window_id,
        train_start=train_start_date,
        train_end=train_end_date,
        test_start=test_start_date,
        test_end=test_end_date,
        test_bars=test_bars,
        total_profit=total_profit,
        total_return_pct=total_return,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        total_trades=total_trades,
        trades_per_bar=trades_per_bar,
        trades_per_day=trades_per_day,
        win_rate_pct=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        passed=passed,
        fail_reasons=fail_reasons,
    )


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for frozen RL model")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--data", type=str, required=True, help="Path to data parquet file")
    parser.add_argument("--train-years", type=float, default=2.0, help="Training context years")
    parser.add_argument("--test-years", type=float, default=1.0, help="Test window years")
    parser.add_argument("--step-years", type=float, default=1.0, help="Step forward years")
    parser.add_argument("--min-pf", type=float, default=1.3, help="Minimum profit factor to pass")
    parser.add_argument("--max-dd", type=float, default=35.0, help="Maximum drawdown % to pass")
    parser.add_argument("--fixed-sl-ticks", type=float, default=200.0, help="Fixed stop loss ticks")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.LOG_DIR / f"wf_validate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = ValidationConfig(
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        min_profit_factor=args.min_pf,
        max_drawdown_pct=args.max_dd,
    )

    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Train: {config.train_years}yr, Test: {config.test_years}yr, Step: {config.step_years}yr")
    logger.info(f"Pass criteria: PF >= {config.min_profit_factor}, MaxDD <= {config.max_drawdown_pct}%")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} bars")

    # Preprocess
    preprocessor = DataPreprocessor(scaler_type="robust")
    df = preprocessor.clean_data(df)

    # Compute features
    feature_engineer = FeatureEngineer()
    df = feature_engineer.compute_all_features(df)
    feature_columns = feature_engineer.feature_names

    # Load model
    model_path = Path(args.model)
    if model_path.is_dir():
        model_file = model_path / "final_model.zip"
        if not model_file.exists():
            model_file = model_path / "best_model_overall.zip"
    else:
        model_file = model_path

    # Load feature columns from model
    feature_cols_file = model_path / "feature_columns.txt" if model_path.is_dir() else model_path.parent / "feature_columns.txt"
    if feature_cols_file.exists():
        with open(feature_cols_file) as f:
            saved_features = [line.strip() for line in f.readlines()]
        feature_columns = [f for f in saved_features if f in df.columns]

    logger.info(f"Using {len(feature_columns)} features")

    # Create dummy env for loading
    dummy_env = TradingEnv(
        df=df.iloc[:settings.LOOKBACK_WINDOW + 100].reset_index(drop=True),
        feature_columns=feature_columns,
        fixed_sl_ticks=args.fixed_sl_ticks,
    )

    agent = TradingAgent(algorithm="PPO")
    agent.load(model_file, env=dummy_env)
    logger.info("Model loaded (frozen policy)")

    # Get time windows
    windows = get_time_windows(df, config)
    logger.info(f"Generated {len(windows)} validation windows")

    if len(windows) == 0:
        logger.error("Not enough data for validation windows")
        return 1

    # Run validation on each window
    all_metrics: List[WindowMetrics] = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        metrics = validate_window(
            model=agent,
            df=df,
            feature_columns=feature_columns,
            preprocessor=preprocessor,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            window_id=i,
            config=config,
            fixed_sl_ticks=args.fixed_sl_ticks,
        )
        all_metrics.append(metrics)

        status = "PASS" if metrics.passed else f"FAIL ({', '.join(metrics.fail_reasons)})"
        logger.info(
            f"  Result: {status} | "
            f"Return: {metrics.total_return_pct:.1f}% | "
            f"PF: {metrics.profit_factor:.2f} | "
            f"MaxDD: {metrics.max_drawdown_pct:.1f}% | "
            f"Trades: {metrics.total_trades}"
        )

    # Aggregate results
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)

    # Per-window results
    print("\nPer-Window Results:")
    print("-" * 80)
    print(f"{'Window':<8} {'Test Period':<25} {'Return':>10} {'PF':>8} {'MaxDD':>8} {'Trades':>8} {'Status':<10}")
    print("-" * 80)

    for m in all_metrics:
        status = "PASS" if m.passed else "FAIL"
        print(f"{m.window_id:<8} {m.test_start} to {m.test_end:<10} {m.total_return_pct:>9.1f}% {m.profit_factor:>8.2f} {m.max_drawdown_pct:>7.1f}% {m.total_trades:>8} {status:<10}")

    print("-" * 80)

    # Aggregate stats
    returns = [m.total_return_pct for m in all_metrics]
    pfs = [m.profit_factor for m in all_metrics]
    dds = [m.max_drawdown_pct for m in all_metrics]
    trades = [m.total_trades for m in all_metrics]

    print("\nAggregate Statistics:")
    print(f"  Return:  Mean={np.mean(returns):.1f}%, Std={np.std(returns):.1f}%, Min={np.min(returns):.1f}%, Max={np.max(returns):.1f}%")
    print(f"  PF:      Mean={np.mean(pfs):.2f}, Std={np.std(pfs):.2f}, Min={np.min(pfs):.2f}, Max={np.max(pfs):.2f}")
    print(f"  MaxDD:   Mean={np.mean(dds):.1f}%, Std={np.std(dds):.1f}%, Min={np.min(dds):.1f}%, Max={np.max(dds):.1f}%")
    print(f"  Trades:  Mean={np.mean(trades):.0f}, Std={np.std(trades):.0f}")

    # Worst window
    worst_idx = np.argmin(returns)
    worst = all_metrics[worst_idx]
    print(f"\nWorst Window: {worst.window_id} ({worst.test_start} to {worst.test_end})")
    print(f"  Return: {worst.total_return_pct:.1f}%, PF: {worst.profit_factor:.2f}, MaxDD: {worst.max_drawdown_pct:.1f}%")

    # Overall pass/fail
    n_passed = sum(1 for m in all_metrics if m.passed)
    n_total = len(all_metrics)
    overall_pass = n_passed == n_total

    print("\n" + "=" * 80)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'} ({n_passed}/{n_total} windows passed)")
    print("=" * 80)

    # Save results
    results_df = pd.DataFrame([asdict(m) for m in all_metrics])
    results_df.to_csv(output_dir / "walkforward_results.csv", index=False)

    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "config": asdict(config),
        "n_windows": n_total,
        "n_passed": n_passed,
        "overall_pass": overall_pass,
        "mean_return": float(np.mean(returns)),
        "mean_pf": float(np.mean(pfs)),
        "mean_maxdd": float(np.mean(dds)),
        "worst_window": worst_idx,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
