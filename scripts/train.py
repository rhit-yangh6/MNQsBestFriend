#!/usr/bin/env python3
"""Script to train the RL trading model."""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer
from env.trading_env import TradingEnv
from models.agent import TradingAgent, WalkForwardTrainer
from models.callbacks import TradingCallback, CheckpointCallback
from utils.logger import setup_logger
from utils.visualize import TradingVisualizer

import pandas as pd
import numpy as np

logger = setup_logger("train", level="INFO")


def load_and_prepare_data(data_path: Path, compute_features: bool = True):
    """Load and prepare data for training."""
    logger.info(f"Loading data from {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Preprocess
    preprocessor = DataPreprocessor(scaler_type="robust")
    df = preprocessor.clean_data(df)

    # Compute features if not already present
    if compute_features:
        feature_engineer = FeatureEngineer()
        df = feature_engineer.compute_all_features(df)
        feature_columns = feature_engineer.feature_names
    else:
        # Assume features are already computed
        base_cols = ["open", "high", "low", "close", "volume", "vwap", "bar_count"]
        feature_columns = [col for col in df.columns if col not in base_cols]

    logger.info(f"Using {len(feature_columns)} features")

    # Normalize features
    df = preprocessor.fit_transform(df, feature_columns)

    return df, feature_columns, preprocessor


def main():
    parser = argparse.ArgumentParser(description="Train RL trading model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/MNQ_5min_databento.parquet",
        help="Path to training data (parquet file)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="lstm",
        choices=["lstm", "attention"],
        help="Feature extractor type",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="composite",
        choices=["pnl", "sharpe", "sortino", "risk_adjusted", "profit_factor", "composite"],
        help="Reward function type",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--batch-multiplier",
        type=int,
        default=1,
        help="Multiply batch size for faster training",
    )
    parser.add_argument(
        "--sl-options",
        type=str,
        default="80,120,160,200",
        help="Comma-separated SL options in ticks (agent learns to choose)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model directory to resume training from",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.MODEL_DIR / f"train_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    tensorboard_log = str(output_dir / "tensorboard") if args.tensorboard else None

    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Feature extractor: {args.feature_extractor}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Reward type: {args.reward_type}")

    # Parse SL options
    sl_options = [float(x) for x in args.sl_options.split(",")]
    logger.info(f"SL options (ticks): {sl_options}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 50)

    # Load data
    df, feature_columns, preprocessor = load_and_prepare_data(Path(args.data))

    # Split data
    n = len(df)
    train_end = int(n * settings.TRAIN_SPLIT)
    val_end = int(n * (settings.TRAIN_SPLIT + settings.VAL_SPLIT))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save preprocessor
    preprocessor.save_scaler(output_dir / "scaler.joblib")

    if args.walk_forward:
        # Walk-forward training
        logger.info("Running walk-forward training...")

        trainer = WalkForwardTrainer(
            df=df,
            feature_columns=feature_columns,
            train_window=50000,
            val_window=10000,
            step_size=10000,
            algorithm=args.algorithm,
            feature_extractor=args.feature_extractor,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

        results = trainer.run(
            timesteps_per_fold=args.timesteps // 5,
            save_dir=output_dir,
        )

        results.to_csv(output_dir / "walk_forward_results.csv", index=False)
        logger.info("Walk-forward training completed")
        logger.info(f"Results saved to {output_dir / 'walk_forward_results.csv'}")

    else:
        # Standard training with parallel environments
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        def make_env(df, feature_columns, reward_type, sl_options, seed, rank):
            """Create a single environment instance."""
            def _init():
                env = TradingEnv(
                    df=df,
                    feature_columns=feature_columns,
                    reward_type=reward_type,
                    lookback_window=settings.LOOKBACK_WINDOW,
                    sl_options=sl_options,
                )
                env.reset(seed=seed + rank)
                return Monitor(env)
            return _init

        n_envs = args.n_envs
        logger.info(f"Creating {n_envs} parallel environments...")

        # Create parallel training environments
        if n_envs > 1:
            train_envs = SubprocVecEnv([
                make_env(train_df, feature_columns, args.reward_type, sl_options, args.seed, i)
                for i in range(n_envs)
            ])
        else:
            train_envs = DummyVecEnv([
                make_env(train_df, feature_columns, args.reward_type, sl_options, args.seed, 0)
            ])

        # Single validation environment
        val_env = TradingEnv(
            df=val_df,
            feature_columns=feature_columns,
            reward_type=args.reward_type,
            lookback_window=settings.LOOKBACK_WINDOW,
            sl_options=sl_options,
        )

        # Create agent with parallel envs and larger batch
        effective_batch = args.batch_size * args.batch_multiplier * n_envs
        logger.info(f"Effective batch size: {effective_batch}")

        if args.resume:
            # Resume training from existing model
            resume_path = Path(args.resume)
            if resume_path.is_dir():
                model_file = resume_path / "final_model.zip"
            else:
                model_file = resume_path

            logger.info(f"Resuming training from {model_file}")
            agent = TradingAgent.from_pretrained(
                path=model_file,
                env=train_envs,
                algorithm=args.algorithm,
                feature_extractor=args.feature_extractor,
                learning_rate=args.learning_rate,
                batch_size=effective_batch,
                tensorboard_log=tensorboard_log,
                seed=args.seed,
            )
        else:
            # Create new agent
            agent = TradingAgent(
                env=train_envs,
                algorithm=args.algorithm,
                feature_extractor=args.feature_extractor,
                learning_rate=args.learning_rate,
                batch_size=effective_batch,
                tensorboard_log=tensorboard_log,
                seed=args.seed,
            )

        # Train
        logger.info("Starting training...")
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=val_env,
            eval_freq=max(10000, args.timesteps // 20),  # Evaluate ~20 times during training
            save_freq=args.timesteps // 5,
            save_path=output_dir,
            log_interval=5,  # Log trading metrics every 5 episodes
        )

        # Close parallel envs
        train_envs.close()

        # Save final model
        agent.save(output_dir / "final_model")
        logger.info(f"Model saved to {output_dir / 'final_model'}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_env = TradingEnv(
            df=test_df,
            feature_columns=feature_columns,
            reward_type=args.reward_type,
            lookback_window=settings.LOOKBACK_WINDOW,
            sl_options=sl_options,
        )

        eval_results = agent.evaluate(test_env, n_episodes=10)

        logger.info("=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)
        for key, value in eval_results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 50)

        # Save evaluation results
        import json
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

    # Save feature columns for inference
    with open(output_dir / "feature_columns.txt", "w") as f:
        f.write("\n".join(feature_columns))

    # Save training config for inference
    import json
    training_config = {
        "algorithm": args.algorithm,
        "feature_extractor": args.feature_extractor,
        "reward_type": args.reward_type,
        "lookback_window": settings.LOOKBACK_WINDOW,
        "sl_options": sl_options,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    logger.info(f"Training complete! Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
