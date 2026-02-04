"""Training callbacks for monitoring and evaluation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym

logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """
    Custom callback for tracking trading-specific metrics during training.
    """

    def __init__(
        self,
        verbose: int = 0,
        log_freq: int = 100,
    ):
        """
        Initialize trading callback.

        Args:
            verbose: Verbosity level
            log_freq: Frequency of logging (in episodes)
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_trades: List[int] = []
        self.episode_profits: List[float] = []
        self.episode_win_rates: List[float] = []

        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.n_episodes = 0

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []
        self.episode_profits = []
        self.episode_win_rates = []

        # Initialize per-env tracking for parallel environments
        n_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
        self._env_rewards = [0.0] * n_envs
        self._env_lengths = [0] * n_envs

        logger.info(f"TradingCallback initialized for {n_envs} environments, logging every {self.log_freq} episodes")

    def _on_step(self) -> bool:
        """Called after each step - handles parallel environments."""
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        n_envs = len(rewards)

        # Initialize per-env tracking if needed
        if not hasattr(self, '_env_rewards'):
            self._env_rewards = [0.0] * n_envs
            self._env_lengths = [0] * n_envs

        # Update all environments
        for i in range(n_envs):
            self._env_rewards[i] += rewards[i]
            self._env_lengths[i] += 1

            # Check if episode ended for this env
            if dones[i]:
                self.episode_rewards.append(self._env_rewards[i])
                self.episode_lengths.append(self._env_lengths[i])

                # Get trading-specific info
                info = infos[i]
                self.episode_trades.append(info.get("total_trades", 0))
                self.episode_profits.append(info.get("realized_pnl", 0))

                # Track win rate
                total_trades = info.get("total_trades", 0)
                if total_trades > 0:
                    win_rate = info.get("win_rate", 0.5)
                    self.episode_win_rates.append(win_rate)

                self.n_episodes += 1

                # Log periodically
                if self.n_episodes % self.log_freq == 0:
                    self._log_metrics()

                # Reset for this env
                self._env_rewards[i] = 0.0
                self._env_lengths[i] = 0

        return True

    def _log_metrics(self) -> None:
        """Log training metrics."""
        if len(self.episode_rewards) < self.log_freq:
            return

        # Get recent episodes
        recent_rewards = self.episode_rewards[-self.log_freq:]
        recent_profits = self.episode_profits[-self.log_freq:]
        recent_trades = self.episode_trades[-self.log_freq:]
        recent_win_rates = self.episode_win_rates[-self.log_freq:] if self.episode_win_rates else []

        # Calculate metrics
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_profit = np.mean(recent_profits)
        total_profit = np.sum(recent_profits)
        mean_trades = np.mean(recent_trades)
        mean_win_rate = np.mean(recent_win_rates) * 100 if recent_win_rates else 0

        # Calculate profit factor from recent profits
        wins = [p for p in recent_profits if p > 0]
        losses = [abs(p) for p in recent_profits if p < 0]
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0

        # Log to tensorboard if available
        if self.logger is not None:
            self.logger.record("trading/mean_reward", mean_reward)
            self.logger.record("trading/std_reward", std_reward)
            self.logger.record("trading/mean_profit", mean_profit)
            self.logger.record("trading/total_profit", total_profit)
            self.logger.record("trading/mean_trades", mean_trades)
            self.logger.record("trading/win_rate", mean_win_rate)
            self.logger.record("trading/profit_factor", profit_factor)
            self.logger.record("trading/n_episodes", self.n_episodes)

        if self.verbose > 0:
            logger.info(
                f"Episodes: {self.n_episodes} | "
                f"Reward: {mean_reward:.2f} | "
                f"Profit: ${mean_profit:.2f} | "
                f"Trades: {mean_trades:.1f} | "
                f"WinRate: {mean_win_rate:.1f}% | "
                f"PF: {profit_factor:.2f}"
            )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            logger.info(
                f"Training completed | "
                f"Total Episodes: {self.n_episodes} | "
                f"Final Mean Reward: {np.mean(self.episode_rewards[-100:]):.2f} | "
                f"Final Mean Profit: ${np.mean(self.episode_profits[-100:]):.2f}"
            )

    def get_results(self) -> pd.DataFrame:
        """Get training results as DataFrame."""
        return pd.DataFrame({
            "episode": range(len(self.episode_rewards)),
            "reward": self.episode_rewards,
            "length": self.episode_lengths,
            "trades": self.episode_trades,
            "profit": self.episode_profits,
        })


class EvaluationCallback(EvalCallback):
    """
    Extended evaluation callback with trading-specific metrics.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
        early_stopping_patience: int = 10,
        min_improvement: float = 0.01,
    ):
        """
        Initialize evaluation callback.

        Args:
            eval_env: Evaluation environment
            n_eval_episodes: Number of evaluation episodes
            eval_freq: Evaluation frequency
            best_model_save_path: Path to save best model
            log_path: Path for evaluation logs
            deterministic: Use deterministic policy
            verbose: Verbosity level
            early_stopping_patience: Number of evals without improvement before stopping
            min_improvement: Minimum improvement threshold
        """
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            deterministic=deterministic,
            verbose=verbose,
        )

        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        self.no_improvement_count = 0
        self.best_sharpe = -np.inf
        self.eval_history: List[Dict] = []

    def _on_step(self) -> bool:
        """Called after each step."""
        result = super()._on_step()

        # Check for early stopping
        if self.n_calls % self.eval_freq == 0 and len(self.evaluations_results) > 0:
            # Calculate Sharpe ratio from recent evaluations
            recent_returns = self.evaluations_results[-1]
            if len(recent_returns) > 0:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns) + 1e-8
                sharpe = mean_return / std_return

                # Track history
                self.eval_history.append({
                    "timestep": self.num_timesteps,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "sharpe": sharpe,
                })

                # Check for improvement
                if sharpe > self.best_sharpe + self.min_improvement:
                    self.best_sharpe = sharpe
                    self.no_improvement_count = 0

                    if self.verbose > 0:
                        logger.info(f"New best Sharpe ratio: {sharpe:.3f}")
                else:
                    self.no_improvement_count += 1

                # Early stopping check
                if self.no_improvement_count >= self.early_stopping_patience:
                    if self.verbose > 0:
                        logger.info(
                            f"Early stopping triggered after {self.no_improvement_count} "
                            f"evaluations without improvement"
                        )
                    return False

        return result

    def get_eval_history(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame."""
        return pd.DataFrame(self.eval_history)


class CheckpointCallback(BaseCallback):
    """
    Callback for periodic model checkpoints.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: Union[str, Path],
        name_prefix: str = "model",
        verbose: int = 0,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_freq: Checkpoint frequency (in timesteps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint names
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = (
                self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        return True


class SharpeCallback(BaseCallback):
    """
    Callback for tracking and optimizing Sharpe ratio during training.
    """

    def __init__(
        self,
        window_size: int = 100,
        target_sharpe: float = 1.5,
        verbose: int = 0,
    ):
        """
        Initialize Sharpe callback.

        Args:
            window_size: Window for Sharpe calculation
            target_sharpe: Target Sharpe ratio
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.window_size = window_size
        self.target_sharpe = target_sharpe
        self.returns_history: List[float] = []
        self.sharpe_history: List[float] = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Track returns
        reward = self.locals["rewards"][0]
        self.returns_history.append(reward)

        # Calculate Sharpe periodically
        if len(self.returns_history) >= self.window_size:
            recent_returns = np.array(self.returns_history[-self.window_size:])
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8
            sharpe = mean_return / std_return * np.sqrt(252 * 78)  # Annualized

            self.sharpe_history.append(sharpe)

            # Log to tensorboard
            if self.logger is not None:
                self.logger.record("trading/sharpe_ratio", sharpe)

            # Check if target reached
            if sharpe >= self.target_sharpe:
                if self.verbose > 0:
                    logger.info(f"Target Sharpe ratio ({self.target_sharpe}) reached: {sharpe:.3f}")

        return True

    def get_sharpe_history(self) -> np.ndarray:
        """Get Sharpe ratio history."""
        return np.array(self.sharpe_history)


class PositionTrackingCallback(BaseCallback):
    """
    Callback for tracking position statistics during training.
    """

    def __init__(self, verbose: int = 0):
        """Initialize position tracking callback."""
        super().__init__(verbose)
        self.position_history: List[int] = []
        self.long_count = 0
        self.short_count = 0
        self.flat_count = 0

    def _on_step(self) -> bool:
        """Called after each step."""
        info = self.locals["infos"][0]
        position = info.get("position", 0)
        self.position_history.append(position)

        if position > 0:
            self.long_count += 1
        elif position < 0:
            self.short_count += 1
        else:
            self.flat_count += 1

        # Log periodically
        if self.n_calls % 10000 == 0:
            total = self.long_count + self.short_count + self.flat_count
            if total > 0:
                long_pct = self.long_count / total * 100
                short_pct = self.short_count / total * 100
                flat_pct = self.flat_count / total * 100

                if self.logger is not None:
                    self.logger.record("position/long_pct", long_pct)
                    self.logger.record("position/short_pct", short_pct)
                    self.logger.record("position/flat_pct", flat_pct)

        return True

    def get_position_stats(self) -> Dict[str, float]:
        """Get position statistics."""
        total = self.long_count + self.short_count + self.flat_count
        if total == 0:
            return {"long_pct": 0, "short_pct": 0, "flat_pct": 0}

        return {
            "long_pct": self.long_count / total * 100,
            "short_pct": self.short_count / total * 100,
            "flat_pct": self.flat_count / total * 100,
        }
