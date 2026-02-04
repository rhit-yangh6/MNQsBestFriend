"""Interactive Brokers connection configuration."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class IBConfig:
    """Configuration for Interactive Brokers connection."""

    # TWS/Gateway connection settings
    HOST: str = "127.0.0.1"

    # Port settings
    # TWS Live: 7496, TWS Paper: 7497
    # Gateway Live: 4001, Gateway Paper: 4002
    LIVE_PORT: int = 7496
    PAPER_PORT: int = 7497
    GATEWAY_LIVE_PORT: int = 4001
    GATEWAY_PAPER_PORT: int = 4002

    # Client ID (unique per connection)
    CLIENT_ID: int = 1

    # Connection timeout in seconds
    TIMEOUT: int = 30

    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS: int = 5
    RECONNECT_DELAY: int = 5  # seconds

    # Account settings (set via environment variables for security)
    ACCOUNT_ID: Optional[str] = None

    # Market data settings
    MARKET_DATA_TYPE: int = 1  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen

    # Order settings
    USE_SMART_ROUTING: bool = True
    DEFAULT_ORDER_TYPE: str = "MKT"  # MKT, LMT, STP, etc.

    @classmethod
    def from_env(cls, paper: bool = True) -> "IBConfig":
        """
        Create config from environment variables.

        Args:
            paper: If True, use paper trading ports
        """
        return cls(
            HOST=os.getenv("IB_HOST", "127.0.0.1"),
            LIVE_PORT=int(os.getenv("IB_LIVE_PORT", 7496)),
            PAPER_PORT=int(os.getenv("IB_PAPER_PORT", 7497)),
            GATEWAY_LIVE_PORT=int(os.getenv("IB_GATEWAY_LIVE_PORT", 4001)),
            GATEWAY_PAPER_PORT=int(os.getenv("IB_GATEWAY_PAPER_PORT", 4002)),
            CLIENT_ID=int(os.getenv("IB_CLIENT_ID", 1)),
            TIMEOUT=int(os.getenv("IB_TIMEOUT", 30)),
            ACCOUNT_ID=os.getenv("IB_ACCOUNT_ID"),
            MARKET_DATA_TYPE=int(os.getenv("IB_MARKET_DATA_TYPE", 1)),
        )

    def get_port(self, paper: bool = True, use_gateway: bool = False) -> int:
        """
        Get the appropriate port based on trading mode.

        Args:
            paper: If True, use paper trading port
            use_gateway: If True, use Gateway ports instead of TWS
        """
        if use_gateway:
            return self.GATEWAY_PAPER_PORT if paper else self.GATEWAY_LIVE_PORT
        return self.PAPER_PORT if paper else self.LIVE_PORT


# Default configuration
ib_config = IBConfig.from_env()
