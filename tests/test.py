#!/usr/bin/env python3
"""Test script to check model behavior."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from models.agent import TradingAgent
from env.trading_env import TradingEnv
import pandas as pd


def test_model_actions(model_path: str = None, n_steps: int = 100):
    """Test what actions the model outputs."""

    # Find latest model if not specified
    if model_path is None:
        model_dir = Path("models/saved")
        model_dirs = sorted(model_dir.glob("train_*"), reverse=True)
        if not model_dirs:
            print("No trained models found!")
            return
        model_path = model_dirs[0] / "final_model.zip"
        print(f"Using latest model: {model_path}")
    else:
        model_path = Path(model_path)

    # Load data
    print("Loading data...")
    df = pd.read_parquet("data/processed/MNQ_5min_databento.parquet")
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'bar_count']]
    print(f"Features: {len(feature_cols)}")

    # Create environment
    env = TradingEnv(df=df.iloc[:2000], feature_columns=feature_cols)

    # Load model
    print(f"Loading model from {model_path}...")
    agent = TradingAgent(algorithm='PPO')
    agent.load(model_path, env=env)
    print("Model loaded!")

    # Run test
    print(f"\nRunning {n_steps} steps...")
    obs, _ = env.reset()

    actions = []
    rewards = []
    trades = 0

    positions_opened = 0
    for step in range(n_steps):
        action, _ = agent.predict(obs)

        # Handle Discrete action space (single int or 0-d array)
        if isinstance(action, np.ndarray):
            action_type = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            action_type = int(action)

        old_position = env.position
        actions.append(action_type)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        trades = info.get('total_trades', 0)

        # Track if position was opened
        if old_position == 0 and env.position != 0:
            positions_opened += 1
            print(f"  Step {step}: Opened {'LONG' if env.position == 1 else 'SHORT'} position")

        # Show drawdown restriction
        if info.get('trading_restricted', False):
            print(f"  Step {step}: Trading restricted due to drawdown!")

        if terminated or truncated:
            print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            print(f"  Final equity: ${info.get('equity', 0):.2f}, Drawdown: {info.get('current_drawdown', 0)*100:.1f}%")
            obs, _ = env.reset()

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print("\nAction Distribution (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE):")
    for action_type in range(4):
        count = actions.count(action_type)
        pct = count / len(actions) * 100
        action_name = ['HOLD', 'BUY', 'SELL', 'CLOSE'][action_type]
        print(f"  {action_type} ({action_name}): {count} ({pct:.1f}%)")

    print(f"\nPositions Opened: {positions_opened}")
    print(f"Total Trades (closed): {trades}")
    print(f"Total Reward: {sum(rewards):.4f}")
    print(f"Mean Reward: {np.mean(rewards):.6f}")

    # Diagnosis
    print("\n" + "=" * 50)
    print("DIAGNOSIS")
    print("=" * 50)

    buy_pct = actions.count(1) / len(actions) * 100
    sell_pct = actions.count(2) / len(actions) * 100
    trade_actions_pct = buy_pct + sell_pct

    if trade_actions_pct == 0:
        print("CRITICAL: Model NEVER outputs BUY or SELL!")
        print("The model only uses HOLD and CLOSE - it cannot trade.")
        print("This is a training failure. Retrain with inaction penalty.")
    elif trade_actions_pct < 5:
        print(f"WARNING: Model rarely trades (only {trade_actions_pct:.1f}% BUY/SELL)")
        print("Consider retraining with stronger inaction penalty.")
    elif positions_opened == 0:
        print("WARNING: Model outputs BUY/SELL but no positions opened!")
        print("Possible causes: drawdown restriction, or bug in environment.")
    else:
        print(f"Model is trading: {trade_actions_pct:.1f}% BUY/SELL actions")
        print(f"Opened {positions_opened} positions in {n_steps} steps.")

    if trades == 0 and positions_opened > 0:
        print("\nNote: Positions opened but no trades closed.")
        print("Positions may still be open at end of test.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model actions")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to test")

    args = parser.parse_args()
    test_model_actions(args.model, args.steps)
