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

    Metrics corrections applied:
    - profit_factor: sum(gross_profit) / abs(sum(gross_loss)) - NOT averaged
    - win_rate: total_winning_trades / total_trades - NOT averaged
    - trade_frequency: trades_per_bar, trades_per_100_bars, trades_per_1000_bars, trades_per_day, avg_holding_bars
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

        # Episode-level tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_profits: List[float] = []

        # Aggregate trade tracking (for correct PF and WR calculation)
        self.total_gross_profit = 0.0
        self.total_gross_loss = 0.0
        self.total_winning_trades = 0
        self.total_trades = 0
        self.total_bars = 0

        # Per-env tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.n_episodes = 0

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_profits = []

        # Reset aggregate tracking
        self.total_gross_profit = 0.0
        self.total_gross_loss = 0.0
        self.total_winning_trades = 0
        self.total_trades = 0
        self.total_bars = 0

        # Recent window tracking (for logging intervals)
        self._recent_gross_profit = 0.0
        self._recent_gross_loss = 0.0
        self._recent_winning_trades = 0
        self._recent_trades = 0
        self._recent_bars = 0

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

                # Episode profit
                realized_pnl = info.get("realized_pnl", 0)
                self.episode_profits.append(realized_pnl)

                # Aggregate trade stats from info
                ep_trades = info.get("total_trades", 0)
                ep_gross_profit = info.get("gross_profit", 0)
                ep_gross_loss = info.get("gross_loss", 0)
                ep_winning_trades = info.get("winning_trades", 0)
                ep_bars = self._env_lengths[i]

                # Update totals
                self.total_trades += ep_trades
                self.total_gross_profit += ep_gross_profit
                self.total_gross_loss += ep_gross_loss
                self.total_winning_trades += ep_winning_trades
                self.total_bars += ep_bars

                # Update recent window
                self._recent_trades += ep_trades
                self._recent_gross_profit += ep_gross_profit
                self._recent_gross_loss += ep_gross_loss
                self._recent_winning_trades += ep_winning_trades
                self._recent_bars += ep_bars

                self.n_episodes += 1

                # Log periodically
                if self.n_episodes % self.log_freq == 0:
                    self._log_metrics()

                # Reset for this env
                self._env_rewards[i] = 0.0
                self._env_lengths[i] = 0

        return True

    def _log_metrics(self) -> None:
        """Log training metrics with corrected calculations."""
        if len(self.episode_rewards) < self.log_freq:
            return

        # Get recent episodes for reward/profit
        recent_rewards = self.episode_rewards[-self.log_freq:]
        recent_profits = self.episode_profits[-self.log_freq:]

        # === REWARD METRICS ===
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_profit = np.mean(recent_profits)
        total_profit = np.sum(recent_profits)

        # === PROFIT FACTOR (aggregated, not averaged) ===
        # Formula: sum(gross_profit) / abs(sum(gross_loss))
        if self._recent_gross_loss != 0:
            profit_factor = self._recent_gross_profit / abs(self._recent_gross_loss)
        else:
            profit_factor = self._recent_gross_profit if self._recent_gross_profit > 0 else 0

        # === WIN RATE (aggregated, not averaged) ===
        # Formula: winning_trades / total_trades
        if self._recent_trades > 0:
            win_rate = (self._recent_winning_trades / self._recent_trades) * 100
        else:
            win_rate = 0

        # === TRADE FREQUENCY METRICS ===
        avg_trades_per_episode = self._recent_trades / self.log_freq if self.log_freq > 0 else 0

        # Trades per day (full ETH: 23h * 12 bars/hr = 276 bars)
        bars_per_day = 276
        episode_days = self._recent_bars / bars_per_day if bars_per_day > 0 else 1
        trades_per_day = self._recent_trades / max(episode_days, 1)

        # Trades per bar (primary frequency metric for turnover tuning)
        trades_per_bar = self._recent_trades / max(self._recent_bars, 1)
        trades_per_100_bars = trades_per_bar * 100
        trades_per_1000_bars = trades_per_bar * 1000

        # Average holding bars
        avg_holding_bars = self._recent_bars / max(self._recent_trades, 1)

        # === LOG TO TENSORBOARD ===
        if self.logger is not None:
            # Reward metrics
            self.logger.record("trading/mean_reward", mean_reward)
            self.logger.record("trading/std_reward", std_reward)
            self.logger.record("trading/mean_profit", mean_profit)
            self.logger.record("trading/total_profit", total_profit)

            # Corrected metrics
            self.logger.record("trading/profit_factor", profit_factor)
            self.logger.record("trading/win_rate", win_rate)

            # Trade frequency metrics (per-bar metrics for turnover tuning)
            self.logger.record("trading/trades_per_bar", trades_per_bar)
            self.logger.record("trading/trades_per_100_bars", trades_per_100_bars)
            self.logger.record("trading/trades_per_1000_bars", trades_per_1000_bars)
            self.logger.record("trading/trades_per_day", trades_per_day)
            self.logger.record("trading/avg_trades_per_episode", avg_trades_per_episode)
            self.logger.record("trading/avg_holding_bars", avg_holding_bars)

            # Episode count
            self.logger.record("trading/n_episodes", self.n_episodes)

        if self.verbose > 0:
            logger.info(
                f"Episodes: {self.n_episodes} | "
                f"Reward: {mean_reward:.2f} | "
                f"Profit: ${mean_profit:.2f} | "
                f"Trades: {avg_trades_per_episode:.1f} | "
                f"WinRate: {win_rate:.1f}% | "
                f"PF: {profit_factor:.2f} | "
                f"Trades/Day: {trades_per_day:.1f}"
            )

        # Reset recent window tracking
        self._recent_gross_profit = 0.0
        self._recent_gross_loss = 0.0
        self._recent_winning_trades = 0
        self._recent_trades = 0
        self._recent_bars = 0

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            # Final aggregated stats
            final_pf = self.total_gross_profit / abs(self.total_gross_loss) if self.total_gross_loss != 0 else 0
            final_wr = (self.total_winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            logger.info(
                f"Training completed | "
                f"Total Episodes: {self.n_episodes} | "
                f"Total Trades: {self.total_trades} | "
                f"Final PF: {final_pf:.2f} | "
                f"Final WR: {final_wr:.1f}% | "
                f"Final Mean Profit: ${np.mean(self.episode_profits[-100:]):.2f}"
            )

    def get_results(self) -> pd.DataFrame:
        """Get training results as DataFrame."""
        return pd.DataFrame({
            "episode": range(len(self.episode_rewards)),
            "reward": self.episode_rewards,
            "length": self.episode_lengths,
            "profit": self.episode_profits,
        })


class EvaluationCallback(EvalCallback):
    """
    Extended evaluation callback with trading-specific metrics.

    Model selection uses: score = total_profit - 1.5 * max_drawdown
    Primary metric: profit_factor
    Secondary metrics: total_profit, max_drawdown, trades_per_day
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
        min_improvement: float = 50.0,  # Score improvement threshold (dollars)
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
            min_improvement: Minimum score improvement threshold
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
        self.best_score = -np.inf  # score = total_profit - 1.5 * max_drawdown
        self.eval_history: List[Dict] = []

    def _evaluate_with_trading_metrics(self) -> Dict[str, Any]:
        """Run evaluation and collect trading metrics."""
        from tqdm import tqdm

        episode_rewards = []
        episode_lengths = []
        episode_infos = []  # Collect final info from each episode

        obs = self.eval_env.reset()
        n_envs = self.eval_env.num_envs
        episode_counts = np.zeros(n_envs, dtype=int)
        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype=int)

        pbar = tqdm(total=self.n_eval_episodes, desc="Evaluating", unit="ep", leave=False)

        while episode_counts.sum() < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = self.eval_env.step(actions)

            current_rewards += rewards
            current_lengths += 1

            for i, done in enumerate(dones):
                if done and episode_counts[i] < self.n_eval_episodes:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_infos.append(infos[i])  # Store final episode info
                    episode_counts[i] += 1
                    pbar.update(1)
                    pbar.set_postfix({"reward": f"{current_rewards[i]:.2f}"})
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        pbar.close()

        # Aggregate trading metrics from all episodes
        total_gross_profit = sum(info.get("gross_profit", 0) for info in episode_infos)
        total_gross_loss = sum(info.get("gross_loss", 0) for info in episode_infos)
        total_trades = sum(info.get("total_trades", 0) for info in episode_infos)
        total_winning_trades = sum(info.get("winning_trades", 0) for info in episode_infos)
        total_bars = sum(episode_lengths)

        # Calculate metrics
        profit_factor = total_gross_profit / max(abs(total_gross_loss), 0.01)
        win_rate = (total_winning_trades / max(total_trades, 1)) * 100
        total_profit = sum(info.get("realized_pnl", 0) for info in episode_infos)
        max_drawdown = max(info.get("drawdown", 0) for info in episode_infos) * 100

        # Trades per day (full ETH: 276 bars per day)
        trading_days = total_bars / 276
        trades_per_day = total_trades / max(trading_days, 1)

        # Model selection score: total_profit - 1.5 * max_drawdown_dollars
        initial_balance = 10000.0  # Default
        max_drawdown_dollars = (max_drawdown / 100.0) * initial_balance
        score = total_profit - 1.5 * max_drawdown_dollars

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "profit_factor": profit_factor,
            "total_profit": total_profit,
            "max_drawdown": max_drawdown,
            "trades_per_day": trades_per_day,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "score": score,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
        }

    def _on_step(self) -> bool:
        """Called after each step."""
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation with trading metrics
            logger.info(f"Starting evaluation at timestep {self.num_timesteps}...")
            metrics = self._evaluate_with_trading_metrics()

            self.last_mean_reward = metrics["mean_reward"]

            # Store results
            self.evaluations_results.append(metrics["episode_rewards"])
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_length.append(metrics["episode_lengths"])

            # Log results with trading metrics
            if self.verbose >= 1:
                logger.info(
                    f"Eval: PF={metrics['profit_factor']:.2f} | "
                    f"Profit=${metrics['total_profit']:.2f} | "
                    f"MaxDD={metrics['max_drawdown']:.1f}% | "
                    f"Trades/Day={metrics['trades_per_day']:.1f} | "
                    f"WinRate={metrics['win_rate']:.1f}% | "
                    f"Score={metrics['score']:.2f}"
                )

            # Save best model based on SCORE (not mean_reward)
            if metrics["score"] > self.best_score + self.min_improvement:
                self.best_score = metrics["score"]
                self.no_improvement_count = 0
                if self.best_model_save_path is not None:
                    self.model.save(f"{self.best_model_save_path}/best_model")
                    logger.info(f"New best model! Score: {metrics['score']:.2f}")
            else:
                self.no_improvement_count += 1

            # Track history
            self.eval_history.append({
                "timestep": self.num_timesteps,
                "profit_factor": metrics["profit_factor"],
                "total_profit": metrics["total_profit"],
                "max_drawdown": metrics["max_drawdown"],
                "trades_per_day": metrics["trades_per_day"],
                "score": metrics["score"],
            })

            # Early stopping based on score
            if self.no_improvement_count >= self.early_stopping_patience:
                logger.info(f"Early stopping after {self.no_improvement_count} evals without improvement")
                return False

        return continue_training

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
            sharpe = mean_return / std_return * np.sqrt(252 * 276)  # Annualized (full ETH)

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
