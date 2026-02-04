"""Global configuration settings for MNQ Trading System."""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path
import os


@dataclass
class Settings:
    """Global settings for the trading system."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "storage")
    MODEL_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models" / "saved")
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Trading instrument
    SYMBOL: str = "MNQ"
    EXCHANGE: str = "CME"
    CURRENCY: str = "USD"
    TICK_SIZE: float = 0.25
    TICK_VALUE: float = 0.50  # $0.50 per tick for MNQ
    POINT_VALUE: float = 2.0  # $2.00 per point for MNQ

    # Timeframe settings
    TIMEFRAME: str = "5 mins"
    TIMEFRAME_MINUTES: int = 5

    # Market hours (Eastern Time)
    RTH_START: Tuple[int, int] = (9, 30)   # 9:30 AM ET
    RTH_END: Tuple[int, int] = (16, 0)     # 4:00 PM ET
    GLOBEX_START: Tuple[int, int] = (18, 0)  # 6:00 PM ET (Sunday)
    GLOBEX_END: Tuple[int, int] = (17, 0)    # 5:00 PM ET (Friday)

    # Weekend close - close all positions before this time on Friday
    WEEKEND_CLOSE_DAY: int = 4  # Friday (Monday=0)
    WEEKEND_CLOSE_TIME: Tuple[int, int] = (16, 45)  # 4:45 PM ET Friday

    # Trading parameters
    MAX_POSITION_SIZE: int = 5  # Maximum contracts
    DEFAULT_POSITION_SIZE: int = 1
    MAX_DAILY_LOSS: float = 500.0  # Maximum daily loss in USD
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    MAX_DRAWDOWN: float = 0.15  # Maximum drawdown before stopping (15%)
    DRAWDOWN_REDUCE_THRESHOLD: float = 0.08  # Reduce trading at 8% drawdown

    # Transaction costs
    COMMISSION_PER_CONTRACT: float = 0.62  # Per side
    SLIPPAGE_TICKS: float = 1.0  # Expected slippage in ticks

    # Feature engineering
    FEATURE_WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    EMA_PERIODS: List[int] = field(default_factory=lambda: [9, 21, 50])

    # RL Environment
    LOOKBACK_WINDOW: int = 50  # Number of bars for state
    MAX_EPISODE_STEPS: int = 2000

    # Training
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_RATE: float = 3e-4
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5

    # LSTM settings
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 2

    # Logging
    LOG_LEVEL: str = "INFO"
    TENSORBOARD_LOG: bool = True

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables with defaults."""
        return cls(
            MAX_DAILY_LOSS=float(os.getenv("MAX_DAILY_LOSS", 500.0)),
            RISK_PER_TRADE=float(os.getenv("RISK_PER_TRADE", 0.02)),
            MAX_POSITION_SIZE=int(os.getenv("MAX_POSITION_SIZE", 5)),
        )


# Default settings instance
settings = Settings()
