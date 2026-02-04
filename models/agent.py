"""RL agent wrapper for trading using Stable-Baselines3."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import torch
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from config.settings import settings
from env.trading_env import TradingEnv
from .networks import (
    LSTMFeatureExtractor, 
    AttentionFeatureExtractor,
    GRUFeatureExtractor,
    ResidualLSTMFeatureExtractor
)
from .callbacks import TradingCallback, EvaluationCallback

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    Wrapper class for RL trading agent.

    Supports PPO, SAC, and A2C algorithms with custom network architectures.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        algorithm: str = "PPO",
        feature_extractor: str = "lstm",
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        """
        Initialize the trading agent.

        Args:
            env: Trading environment (optional, can be set later)
            algorithm: RL algorithm ('PPO', 'SAC', 'A2C')
            feature_extractor: Type of feature extractor ('lstm', 'attention')
            learning_rate: Learning rate
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            tensorboard_log: TensorBoard log directory
            tensorboard_log: TensorBoard log directory
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            seed: Random seed
        """
        self.algorithm_name = algorithm
        self.feature_extractor_type = feature_extractor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log
        
        # Auto-detect MPS
        if device == "auto" and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) acceleration")
        else:
            self.device = device
            
        self.seed = seed

        self.env = env
        self.model = None
        self.vec_env = None
        self.vec_normalize = None

        if env is not None:
            self._setup_model(env)

    def _setup_model(self, env: gym.Env) -> None:
        """Set up the RL model."""
        # Check if already vectorized
        from stable_baselines3.common.vec_env import VecEnv
        if isinstance(env, VecEnv):
            self.vec_env = env
        else:
            self.vec_env = DummyVecEnv([lambda: Monitor(env)])

        # Create policy kwargs with custom feature extractor
        policy_kwargs = self._get_policy_kwargs()

        # Create model based on algorithm
        if self.algorithm_name == "PPO":
            self.model = PPO(
                policy="MultiInputPolicy",
                env=self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=policy_kwargs,
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
        elif self.algorithm_name == "A2C":
            self.model = A2C(
                policy="MultiInputPolicy",
                env=self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=5,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=policy_kwargs,
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

        logger.info(f"Initialized {self.algorithm_name} agent with {self.feature_extractor_type} extractor")

    def _get_policy_kwargs(self) -> Dict[str, Any]:
        """Get policy keyword arguments including feature extractor."""
        if self.feature_extractor_type == "lstm":
            features_extractor_class = LSTMFeatureExtractor
            features_extractor_kwargs = {
                "features_dim": 256,
                "lstm_hidden_size": settings.LSTM_HIDDEN_SIZE,
                "lstm_num_layers": settings.LSTM_NUM_LAYERS,
                "dropout": 0.1,
            }
        elif self.feature_extractor_type == "attention":
            features_extractor_class = AttentionFeatureExtractor
            features_extractor_kwargs = {
                "features_dim": 256,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
            }
        elif self.feature_extractor_type == "gru":
            features_extractor_class = GRUFeatureExtractor
            features_extractor_kwargs = {
                "features_dim": 256,
                "gru_hidden_size": settings.LSTM_HIDDEN_SIZE,
                "gru_num_layers": settings.LSTM_NUM_LAYERS,
                "dropout": 0.1,
            }
        elif self.feature_extractor_type == "residual_lstm":
            features_extractor_class = ResidualLSTMFeatureExtractor
            features_extractor_kwargs = {
                "features_dim": 256,
                "lstm_hidden_size": settings.LSTM_HIDDEN_SIZE,
                "lstm_num_layers": settings.LSTM_NUM_LAYERS,
                "dropout": 0.1,
            }
        else:
            raise ValueError(f"Unknown feature extractor: {self.feature_extractor_type}")

        return {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            "activation_fn": torch.nn.ReLU,
        }

    def set_env(self, env: gym.Env) -> None:
        """Set or replace the environment."""
        from stable_baselines3.common.vec_env import VecEnv
        self.env = env
        if self.model is None:
            self._setup_model(env)
        else:
            if isinstance(env, VecEnv):
                self.vec_env = env
            else:
                self.vec_env = DummyVecEnv([lambda: Monitor(env)])
            self.model.set_env(self.vec_env)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[CallbackList] = None,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 50000,
        save_path: Optional[Path] = None,
        log_interval: int = 10,
    ) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            callback: Training callbacks
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_freq: Model save frequency
            save_path: Path to save models
            log_interval: Logging interval
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set environment first.")

        # Create callbacks
        callbacks = []

        # Trading callback for custom metrics
        # Log every 10 episodes (or less with parallel envs)
        trading_callback = TradingCallback(
            verbose=1,
            log_freq=max(5, log_interval),
        )
        callbacks.append(trading_callback)

        # Evaluation callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
            eval_callback = EvaluationCallback(
                eval_env=eval_vec_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=str(save_path) if save_path else None,
                deterministic=True,
                verbose=1,
            )
            callbacks.append(eval_callback)

        # Add user callbacks
        if callback is not None:
            callbacks.append(callback)

        callback_list = CallbackList(callbacks)

        # Train
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=log_interval,
            progress_bar=True,
        )
        logger.info("Training completed")

    def predict(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            action (array for MultiDiscrete), states
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        action, states = self.model.predict(
            observation,
            deterministic=deterministic,
        )
        # Handle both single actions and MultiDiscrete actions
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action.item()])
            elif action.ndim == 2:
                action = action[0]  # Remove batch dimension
        return action, states

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the agent.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        vec_env = DummyVecEnv([lambda: env])

        episode_rewards = []
        episode_lengths = []
        episode_trades = []
        episode_profits = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                length += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_trades.append(info.get("total_trades", 0))
            episode_profits.append(info.get("realized_pnl", 0))

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_trades": np.mean(episode_trades),
            "mean_profit": np.mean(episode_profits),
            "total_profit": np.sum(episode_profits),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("Model not initialized")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path], env: Optional[gym.Env] = None) -> None:
        """Load a saved model."""
        path = Path(path)

        if env is not None:
            self.set_env(env)

        if self.algorithm_name == "PPO":
            self.model = PPO.load(path, env=self.vec_env, device=self.device)
        elif self.algorithm_name == "A2C":
            self.model = A2C.load(path, env=self.vec_env, device=self.device)

        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        env: gym.Env,
        algorithm: str = "PPO",
        **kwargs,
    ) -> "TradingAgent":
        """Load a pretrained agent for continued training."""
        agent = cls(algorithm=algorithm, **kwargs)
        agent.set_env(env)
        agent.load(path)
        logger.info(f"Loaded pretrained model from {path} for continued training")
        return agent

    def get_action_probabilities(
        self, observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Get action probabilities for given observation."""
        if self.model is None:
            raise ValueError("Model not initialized")

        obs_tensor = {}
        for key, value in observation.items():
            obs_tensor[key] = torch.tensor(value).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            action_logits = self.model.policy.action_net(latent_pi)
            action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs.cpu().numpy()[0]


class WalkForwardTrainer:
    """
    Walk-forward validation trainer for robust model evaluation.

    Trains on rolling windows and validates on subsequent periods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        train_window: int = 50000,  # bars
        val_window: int = 10000,    # bars
        step_size: int = 10000,     # bars to step forward
        **agent_kwargs,
    ):
        """
        Initialize walk-forward trainer.

        Args:
            df: Full dataset
            feature_columns: Feature columns to use
            train_window: Training window size in bars
            val_window: Validation window size in bars
            step_size: Step size for walk-forward
            **agent_kwargs: Arguments for TradingAgent
        """
        self.df = df
        self.feature_columns = feature_columns
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.agent_kwargs = agent_kwargs

        self.results: List[Dict] = []

    def run(
        self,
        timesteps_per_fold: int = 100000,
        save_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Run walk-forward validation.

        Args:
            timesteps_per_fold: Training timesteps per fold
            save_dir: Directory to save models

        Returns:
            DataFrame with results
        """
        n_samples = len(self.df)
        start_idx = 0
        fold = 0

        while start_idx + self.train_window + self.val_window <= n_samples:
            logger.info(f"=== Fold {fold} ===")

            # Define train and validation ranges
            train_start = start_idx
            train_end = start_idx + self.train_window
            val_start = train_end
            val_end = min(val_start + self.val_window, n_samples)

            logger.info(
                f"Train: {train_start} - {train_end}, "
                f"Val: {val_start} - {val_end}"
            )

            # Create environments
            train_df = self.df.iloc[train_start:train_end].reset_index(drop=True)
            val_df = self.df.iloc[val_start:val_end].reset_index(drop=True)

            train_env = TradingEnv(
                df=train_df,
                feature_columns=self.feature_columns,
            )
            val_env = TradingEnv(
                df=val_df,
                feature_columns=self.feature_columns,
            )

            # Train agent
            agent = TradingAgent(env=train_env, **self.agent_kwargs)
            agent.train(
                total_timesteps=timesteps_per_fold,
                eval_env=val_env,
                eval_freq=timesteps_per_fold // 5,
            )

            # Evaluate on validation set
            eval_results = agent.evaluate(val_env, n_episodes=5)
            eval_results["fold"] = fold
            eval_results["train_start"] = train_start
            eval_results["train_end"] = train_end
            eval_results["val_start"] = val_start
            eval_results["val_end"] = val_end

            self.results.append(eval_results)

            # Save model
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                agent.save(save_dir / f"model_fold_{fold}.zip")

            # Move forward
            start_idx += self.step_size
            fold += 1

        return pd.DataFrame(self.results)
