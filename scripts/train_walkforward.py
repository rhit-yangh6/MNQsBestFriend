#!/usr/bin/env python3
"""
Walk-forward training script for MNQ RL model.

Config: mnq_walkforward_training_plan
- Multiple train/val/test splits
- Rolling window advancement
- Model selection: score = val_return - 2.0 * val_max_drawdown
- Early stopping with patience
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
from env.trading_env import TradingEnv
from models.agent import TradingAgent
from models.callbacks import TradingCallback, CheckpointCallback
from utils.logger import setup_logger

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logger = setup_logger("train_walkforward", level="INFO")


@dataclass
class EvalMetrics:
    """Evaluation metrics for a model checkpoint."""
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    avg_trade_pnl: float
    total_trades: int
    trades_per_day: float
    win_rate_pct: float
    score: float  # val_return - 2.0 * val_max_drawdown


@dataclass
class WalkForwardConfig:
    """Walk-forward training configuration."""
    # Data splits
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15
    roll_forward_step_pct: float = 0.10

    # PPO settings
    n_envs: int = 8
    n_steps: int = 8192
    batch_size: int = 512

    # Training targets
    max_timesteps: int = 30_000_000
    min_timesteps_before_eval: int = 4_000_000
    checkpoint_every_timesteps: int = 2_000_000

    # Early stopping
    patience_checkpoints: int = 3
    min_improvement_score: float = 0.02
    stop_if_drawdown_worsens_pct: float = 5.0

    # Model selection
    prefer_smoother_equity: bool = True

    # Success criteria
    validation_drawdown_under_pct: float = 15.0


class WalkForwardTrainer:
    """Walk-forward validation trainer."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        config: WalkForwardConfig,
        output_dir: Path,
        fixed_sl_ticks: float = 200.0,
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.config = config
        self.output_dir = output_dir
        self.fixed_sl_ticks = fixed_sl_ticks

        self.total_bars = len(df)
        self.results: List[Dict] = []
        self.best_score = -np.inf
        self.best_model_path: Optional[Path] = None
        self.no_improvement_count = 0

        output_dir.mkdir(parents=True, exist_ok=True)

    def _get_window_indices(self, fold: int) -> Tuple[int, int, int, int]:
        """Get train/val/test indices for a fold."""
        step_size = int(self.total_bars * self.config.roll_forward_step_pct)

        # Starting point advances each fold
        start_offset = fold * step_size

        # Calculate split points
        remaining = self.total_bars - start_offset
        train_end = start_offset + int(remaining * self.config.train_pct)
        val_end = train_end + int(remaining * self.config.val_pct)
        test_end = min(val_end + int(remaining * self.config.test_pct), self.total_bars)

        return start_offset, train_end, val_end, test_end

    def _create_env(self, df: pd.DataFrame) -> TradingEnv:
        """Create a trading environment."""
        return TradingEnv(
            df=df,
            feature_columns=self.feature_columns,
            lookback_window=settings.LOOKBACK_WINDOW,
            stop_loss_ticks=self.fixed_sl_ticks,
        )

    def _evaluate_model(
        self,
        agent: TradingAgent,
        val_df: pd.DataFrame,
        n_episodes: int = 1,
    ) -> EvalMetrics:
        """Evaluate model on validation data."""
        env = self._create_env(val_df)

        all_trades = []
        all_equity = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            all_trades.extend(env.trades)
            all_equity.extend(env.equity_curve)

        # Calculate metrics
        if not all_trades:
            return EvalMetrics(
                total_return_pct=0.0,
                max_drawdown_pct=100.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                avg_trade_pnl=0.0,
                total_trades=0,
                trades_per_day=0.0,
                win_rate_pct=0.0,
                score=-100.0,
            )

        # Returns
        initial = all_equity[0] if all_equity else 10000
        final = all_equity[-1] if all_equity else 10000
        total_return_pct = ((final - initial) / initial) * 100

        # Max drawdown
        equity_arr = np.array(all_equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100
        max_drawdown_pct = np.max(drawdown)

        # Trade stats
        wins = [t['pnl'] for t in all_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in all_trades if t['pnl'] <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss

        total_trades = len(all_trades)
        win_rate_pct = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_trade_pnl = sum(t['pnl'] for t in all_trades) / total_trades if total_trades > 0 else 0

        # Trades per day (assuming 5-min bars, ~78 bars per RTH day)
        bars_in_val = len(val_df)
        trading_days = bars_in_val / 78
        trades_per_day = total_trades / max(trading_days, 1)

        # Sharpe-like ratio (simplified)
        if len(all_equity) > 1:
            returns = np.diff(all_equity) / all_equity[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 78)
        else:
            sharpe_ratio = 0.0

        # Score function: val_return - 2.0 * val_max_drawdown
        score = total_return_pct - 2.0 * max_drawdown_pct

        return EvalMetrics(
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            avg_trade_pnl=avg_trade_pnl,
            total_trades=total_trades,
            trades_per_day=trades_per_day,
            win_rate_pct=win_rate_pct,
            score=score,
        )

    def _should_early_stop(self, metrics: EvalMetrics) -> bool:
        """Check if we should stop early."""
        # Check if score improved
        if metrics.score > self.best_score + self.config.min_improvement_score:
            self.best_score = metrics.score
            self.no_improvement_count = 0
            return False

        self.no_improvement_count += 1

        # Patience exceeded
        if self.no_improvement_count >= self.config.patience_checkpoints:
            logger.info(f"Early stopping: no improvement for {self.no_improvement_count} checkpoints")
            return True

        # Drawdown worsened significantly
        if metrics.max_drawdown_pct > self.config.validation_drawdown_under_pct + self.config.stop_if_drawdown_worsens_pct:
            logger.info(f"Early stopping: drawdown too high ({metrics.max_drawdown_pct:.1f}%)")
            return True

        return False

    def run_fold(
        self,
        fold: int,
        timesteps: int,
        tensorboard_log: Optional[str] = None,
    ) -> Dict:
        """Run a single walk-forward fold."""
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold}")
        logger.info(f"{'='*60}")

        # Get data splits
        train_start, train_end, val_end, test_end = self._get_window_indices(fold)

        if train_end >= self.total_bars - settings.LOOKBACK_WINDOW:
            logger.info("Not enough data for this fold, skipping")
            return {}

        train_df = self.df.iloc[train_start:train_end].reset_index(drop=True)
        val_df = self.df.iloc[train_end:val_end].reset_index(drop=True)

        logger.info(f"Train: bars {train_start}-{train_end} ({len(train_df)} bars)")
        logger.info(f"Val: bars {train_end}-{val_end} ({len(val_df)} bars)")

        if len(train_df) < settings.LOOKBACK_WINDOW * 2 or len(val_df) < settings.LOOKBACK_WINDOW * 2:
            logger.info("Insufficient data, skipping fold")
            return {}

        # Create parallel training environments
        def make_env(df, seed, rank):
            def _init():
                env = self._create_env(df)
                env.reset(seed=seed + rank)
                return Monitor(env)
            return _init

        train_envs = DummyVecEnv([
            make_env(train_df, 42, i) for i in range(self.config.n_envs)
        ])

        # Create agent
        effective_batch = self.config.batch_size * self.config.n_envs
        agent = TradingAgent(
            env=train_envs,
            algorithm="PPO",
            feature_extractor="lstm",
            learning_rate=3e-4,
            batch_size=effective_batch,
            n_steps=self.config.n_steps,
            tensorboard_log=tensorboard_log,
            seed=42,
        )

        # Training loop with periodic evaluation
        fold_dir = self.output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        timesteps_trained = 0
        checkpoint_num = 0
        best_fold_score = -np.inf
        best_fold_model = None

        while timesteps_trained < timesteps:
            # Train for checkpoint interval
            train_steps = min(
                self.config.checkpoint_every_timesteps,
                timesteps - timesteps_trained
            )

            logger.info(f"Training for {train_steps} timesteps...")
            agent.model.learn(
                total_timesteps=train_steps,
                reset_num_timesteps=False,
                progress_bar=True,
            )
            timesteps_trained += train_steps
            checkpoint_num += 1

            # Evaluate if past minimum
            if timesteps_trained >= self.config.min_timesteps_before_eval:
                logger.info(f"Evaluating checkpoint {checkpoint_num}...")
                metrics = self._evaluate_model(agent, val_df)

                logger.info(f"  Return: {metrics.total_return_pct:.2f}%")
                logger.info(f"  Max DD: {metrics.max_drawdown_pct:.2f}%")
                logger.info(f"  PF: {metrics.profit_factor:.2f}")
                logger.info(f"  Trades: {metrics.total_trades}")
                logger.info(f"  Score: {metrics.score:.2f}")

                # Save if best for this fold
                if metrics.score > best_fold_score:
                    best_fold_score = metrics.score
                    best_fold_model = fold_dir / f"best_model_fold_{fold}"
                    agent.save(best_fold_model)
                    logger.info(f"  New best for fold! Saved to {best_fold_model}")

                # Check early stopping
                if self._should_early_stop(metrics):
                    break

        train_envs.close()

        # Final evaluation
        if best_fold_model:
            agent.load(best_fold_model)
            final_metrics = self._evaluate_model(agent, val_df)
        else:
            final_metrics = EvalMetrics(
                total_return_pct=0, max_drawdown_pct=100, profit_factor=0,
                sharpe_ratio=0, avg_trade_pnl=0, total_trades=0,
                trades_per_day=0, win_rate_pct=0, score=-100
            )

        # Save global best
        if final_metrics.score > self.best_score:
            self.best_score = final_metrics.score
            self.best_model_path = self.output_dir / "best_model_overall"
            agent.save(self.best_model_path)
            logger.info(f"New global best! Score: {final_metrics.score:.2f}")

        result = {
            "fold": fold,
            "train_bars": len(train_df),
            "val_bars": len(val_df),
            "timesteps_trained": timesteps_trained,
            **asdict(final_metrics),
        }

        self.results.append(result)
        return result

    def run(
        self,
        timesteps_per_fold: int,
        tensorboard_log: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run full walk-forward training."""
        fold = 0

        while True:
            train_start, train_end, val_end, test_end = self._get_window_indices(fold)

            # Check if we have enough data
            if val_end >= self.total_bars - settings.LOOKBACK_WINDOW:
                logger.info(f"Reached end of data at fold {fold}")
                break

            result = self.run_fold(fold, timesteps_per_fold, tensorboard_log)

            if not result:
                break

            fold += 1

        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / "walk_forward_results.csv", index=False)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"{'='*60}")
        logger.info("WALK-FORWARD TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total folds: {len(self.results)}")
        logger.info(f"Best score: {self.best_score:.2f}")
        logger.info(f"Best model: {self.best_model_path}")

        return results_df


def main():
    parser = argparse.ArgumentParser(description="Walk-forward training for MNQ RL model")
    parser.add_argument("--data", type=str, required=True, help="Path to data parquet file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--timesteps-per-fold", type=int, default=10_000_000, help="Timesteps per fold")
    parser.add_argument("--train-pct", type=float, default=0.70, help="Training data percentage")
    parser.add_argument("--val-pct", type=float, default=0.15, help="Validation data percentage")
    parser.add_argument("--roll-step-pct", type=float, default=0.10, help="Roll forward step percentage")
    parser.add_argument("--fixed-sl-ticks", type=float, default=200.0, help="Fixed stop loss ticks")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.MODEL_DIR / f"walkforward_{timestamp}"

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} bars")

    # Preprocess
    preprocessor = DataPreprocessor(scaler_type="robust")
    df = preprocessor.clean_data(df)

    # Compute features
    feature_engineer = FeatureEngineer()
    df = feature_engineer.compute_all_features(df)
    feature_columns = feature_engineer.feature_names

    # Normalize
    df = preprocessor.fit_transform(df, feature_columns)

    # Config
    config = WalkForwardConfig(
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=1.0 - args.train_pct - args.val_pct,
        roll_forward_step_pct=args.roll_step_pct,
    )

    logger.info(f"Config: {asdict(config)}")

    # Tensorboard
    tensorboard_log = str(output_dir / "tensorboard") if args.tensorboard else None

    # Run walk-forward training
    trainer = WalkForwardTrainer(
        df=df,
        feature_columns=feature_columns,
        config=config,
        output_dir=output_dir,
        fixed_sl_ticks=args.fixed_sl_ticks,
    )

    results = trainer.run(
        timesteps_per_fold=args.timesteps_per_fold,
        tensorboard_log=tensorboard_log,
    )

    print("\nResults:")
    print(results.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
