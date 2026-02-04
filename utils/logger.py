"""Logging utilities for the trading system."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from config.settings import settings


def setup_logger(
    name: str = "mnq_trading",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    file_rotation: str = "size",  # 'size' or 'time'
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a configured logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: logs/trading.log)
        console: Whether to output to console
        file_rotation: Rotation type ('size' or 'time')
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = settings.LOG_DIR / "trading.log"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    if file_rotation == "size":
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
    else:
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            backupCount=backup_count,
        )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class TradingLogger:
    """
    Specialized logger for trading activities.

    Logs trades, signals, and performance metrics in structured format.
    """

    def __init__(
        self,
        name: str = "trading",
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize trading logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.log_dir = log_dir or settings.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main logger
        self.logger = setup_logger(
            name=name,
            level=settings.LOG_LEVEL,
            log_file=self.log_dir / "trading.log",
        )

        # Trade logger (separate file for trades)
        self.trade_logger = self._setup_trade_logger()

        # Signal logger
        self.signal_logger = self._setup_signal_logger()

    def _setup_trade_logger(self) -> logging.Logger:
        """Set up trade-specific logger."""
        logger = logging.getLogger("trades")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        handler = RotatingFileHandler(
            self.log_dir / "trades.log",
            maxBytes=5_000_000,
            backupCount=10,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        return logger

    def _setup_signal_logger(self) -> logging.Logger:
        """Set up signal-specific logger."""
        logger = logging.getLogger("signals")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        handler = RotatingFileHandler(
            self.log_dir / "signals.log",
            maxBytes=5_000_000,
            backupCount=10,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        return logger

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def log_trade(
        self,
        trade_type: str,  # 'OPEN' or 'CLOSE'
        side: str,
        price: float,
        quantity: int,
        pnl: Optional[float] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a trade in structured format.

        Args:
            trade_type: Type of trade (OPEN/CLOSE)
            side: BUY or SELL
            price: Execution price
            quantity: Number of contracts
            pnl: P&L (for closes)
            reason: Trade reason
            **kwargs: Additional fields
        """
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "type": trade_type,
            "side": side,
            "price": price,
            "quantity": quantity,
            "pnl": pnl,
            "reason": reason,
            **kwargs,
        }

        self.trade_logger.info(json.dumps(trade_data))

        # Also log to main logger
        if trade_type == "OPEN":
            self.logger.info(f"TRADE OPEN: {side} {quantity} @ {price:.2f}")
        else:
            pnl_str = f", P&L: ${pnl:.2f}" if pnl is not None else ""
            self.logger.info(f"TRADE CLOSE: {quantity} @ {price:.2f}{pnl_str}")

    def log_signal(
        self,
        action: int,
        price: float,
        features: dict,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Log a trading signal.

        Args:
            action: Action (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
            price: Current price
            features: Model features/state
            confidence: Signal confidence
        """
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}

        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "action": action_names.get(action, str(action)),
            "price": price,
            "confidence": confidence,
            "features": features,
        }

        self.signal_logger.info(json.dumps(signal_data))

    def log_performance(
        self,
        daily_pnl: float,
        total_pnl: float,
        trade_count: int,
        win_rate: float,
        sharpe: float,
    ) -> None:
        """
        Log performance metrics.

        Args:
            daily_pnl: Today's P&L
            total_pnl: Total P&L
            trade_count: Number of trades
            win_rate: Win rate percentage
            sharpe: Sharpe ratio
        """
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "sharpe": sharpe,
        }

        self.logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")

    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log an error with full traceback.

        Args:
            error: Exception object
            context: Additional context
        """
        import traceback
        tb = traceback.format_exc()

        self.logger.error(f"ERROR in {context}: {str(error)}\n{tb}")

    def log_connection(self, status: str, details: str = "") -> None:
        """
        Log connection status.

        Args:
            status: Connection status
            details: Additional details
        """
        self.logger.info(f"CONNECTION: {status} - {details}")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            import traceback
            log_data["exception"] = traceback.format_exception(*record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default configuration."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        setup_logger(name, level=settings.LOG_LEVEL)

    return logger
