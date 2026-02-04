"""Neural network architectures for RL trading agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, List, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from config.settings import settings


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for sequential market data.

    Processes market features through LSTM layers before policy/value heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: The observation space
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__(observation_space, features_dim)

        # Get observation dimensions
        market_shape = observation_space["market_features"].shape
        position_dim = observation_space["position_info"].shape[0]

        self.seq_len = market_shape[0]
        self.input_dim = market_shape[1]
        self.position_dim = position_dim

        # LSTM for market features
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        # Combine LSTM output with position info
        combined_dim = lstm_hidden_size + position_dim

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: Dict with 'market_features' and 'position_info'

        Returns:
            Feature tensor
        """
        market_features = observations["market_features"]
        position_info = observations["position_info"]

        # LSTM processing
        lstm_out, _ = self.lstm(market_features)

        # Take last hidden state
        lstm_final = lstm_out[:, -1, :]

        # Layer normalization
        lstm_final = self.layer_norm(lstm_final)

        # Combine with position info
        combined = torch.cat([lstm_final, position_info], dim=-1)

        # Final projection
        features = self.fc(combined)

        return features


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for market data.

    Uses self-attention to identify important time steps in the sequence.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize attention feature extractor.

        Args:
            observation_space: The observation space
            features_dim: Output feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__(observation_space, features_dim)

        market_shape = observation_space["market_features"].shape
        position_dim = observation_space["position_info"].shape[0]

        self.seq_len = market_shape[0]
        self.input_dim = market_shape[1]
        self.position_dim = position_dim

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, features_dim)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            self.seq_len, features_dim
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=num_heads,
            dim_feedforward=features_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        combined_dim = features_dim + position_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        market_features = observations["market_features"]
        position_info = observations["position_info"]

        batch_size = market_features.shape[0]

        # Project input
        x = self.input_proj(market_features)

        # Add positional encoding
        x = x + self.pos_encoding[:, : self.seq_len, :].to(x.device)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Combine with position info
        combined = torch.cat([x, position_info], dim=-1)

        # Output projection
        features = self.output_proj(combined)

        return features


class LSTMActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM backbone.

    For use with custom PPO implementation or as reference architecture.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            input_dim: Feature dimension
            seq_len: Sequence length
            action_dim: Number of actions
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            hidden: Optional LSTM hidden state

        Returns:
            action_logits, value, new_hidden
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Take last hidden state
        features = lstm_out[:, -1, :]
        features = self.layer_norm(features)

        # Actor and critic heads
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value, hidden

    def get_action(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Get action from policy.

        Args:
            x: Input tensor
            hidden: LSTM hidden state
            deterministic: If True, return argmax action

        Returns:
            action, log_prob, value, hidden
        """
        action_logits, value, hidden = self.forward(x, hidden)

        # Create distribution
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1), hidden

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            x: Input tensor
            actions: Actions to evaluate
            hidden: LSTM hidden state

        Returns:
            log_probs, values, entropy
        """
        action_logits, value, _ = self.forward(x, hidden)

        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value.squeeze(-1), entropy


class AttentionNetwork(nn.Module):
    """
    Self-attention network for feature importance.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attention network.

        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            output, attention_weights (if return_attention)
        """
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x)

        # Residual connection and layer norm
        x = self.layer_norm(x + self.dropout(attn_out))

        if return_attention:
            return x, attn_weights
        return x, None


def create_feature_extractor(
    observation_space: gym.spaces.Dict,
    extractor_type: str = "lstm",
    **kwargs,
) -> BaseFeaturesExtractor:
    """
    Factory function to create feature extractor.

    Args:
        observation_space: Observation space
        extractor_type: Type of extractor ('lstm', 'attention')
        **kwargs: Additional arguments for the extractor

    Returns:
        Feature extractor instance
    """
    if extractor_type == "lstm":
        return LSTMFeatureExtractor(observation_space, **kwargs)
    elif extractor_type == "attention":
        return AttentionFeatureExtractor(observation_space, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
