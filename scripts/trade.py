#!/usr/bin/env python3
"""Script to run paper or live trading."""

import argparse
from pathlib import Path
import logging
import sys
import signal
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.preprocessor import DataPreprocessor
from data.features import FeatureEngineer
from models.agent import TradingAgent
from trading.paper_trader import PaperTrader
from trading.live_trader import LiveTrader
from env.trading_env import TradingEnv
from utils.logger import setup_logger, TradingLogger

import pandas as pd

logger = setup_logger("trade", level="INFO")


def load_model_and_config(model_dir: Path):
    """Load trained model and its configuration."""
    import json

    # Load feature columns
    with open(model_dir / "feature_columns.txt", "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    # Load scaler
    preprocessor = DataPreprocessor()
    preprocessor.load_scaler(model_dir / "scaler.joblib")

    # Load training config
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            training_config = json.load(f)
    else:
        training_config = {
            "fixed_sl_ticks": 120.0,
            "lookback_window": settings.LOOKBACK_WINDOW,
        }

    return feature_columns, preprocessor, training_config


class TradingManager:
    """Manages trading session with graceful shutdown."""

    def __init__(self, trader, trading_logger: TradingLogger):
        self.trader = trader
        self.trading_logger = trading_logger
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning("Received shutdown signal")
        self.running = False
        self.trader.stop()

    def run(self):
        """Run trading session."""
        logger.info("Starting trading session...")

        try:
            self.trader.start()

            # Main loop - monitor status
            while self.running and self.trader.running:
                status = self.trader.get_status()

                # Log status periodically
                if hasattr(self, '_last_status_log'):
                    if time.time() - self._last_status_log > 60:
                        self._log_status(status)
                        self._last_status_log = time.time()
                else:
                    self._log_status(status)
                    self._last_status_log = time.time()

                time.sleep(1)

        except Exception as e:
            logger.error(f"Trading error: {e}")
            self.trading_logger.log_error(e, "trading_session")
        finally:
            self.trader.stop()
            logger.info("Trading session ended")

    def _log_status(self, status: dict):
        """Log current trading status."""
        logger.info(
            f"Status: Position={status['position']}, "
            f"Unrealized P&L=${status['unrealized_pnl']:.2f}, "
            f"Daily P&L=${status['daily_pnl']:.2f}, "
            f"Trades={status['trade_count']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run paper or live trading")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "live", "simulation"],
        help="Trading mode",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=1,
        help="Maximum position size",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=500.0,
        help="Maximum daily loss in USD",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.02,
        help="Risk per trade as fraction",
    )
    parser.add_argument(
        "--use-stop-loss",
        action="store_true",
        default=True,
        help="Use automatic stop-loss",
    )
    parser.add_argument(
        "--stop-loss-atr",
        type=float,
        default=2.0,
        help="ATR multiplier for stop-loss",
    )
    parser.add_argument(
        "--simulation-data",
        type=str,
        default=None,
        help="Data file for simulation mode",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for trade logs",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    # Setup logging
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = settings.LOG_DIR / f"trading_{args.mode}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    trading_logger = TradingLogger(log_dir=log_dir)

    # Print configuration
    print("\n" + "=" * 60)
    print("TRADING CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {model_path}")
    print(f"Max Position: {args.max_position}")
    print(f"Max Daily Loss: ${args.max_daily_loss}")
    print(f"Risk per Trade: {args.risk_per_trade * 100}%")
    print(f"Use Stop-Loss: {args.use_stop_loss}")
    print(f"Log Directory: {log_dir}")
    print("=" * 60 + "\n")

    if args.mode == "live":
        # Check safety gate
        import os
        allow_live = os.environ.get("ALLOW_LIVE_TRADING", "false").lower() == "true"
        if not allow_live:
            print("*" * 60)
            print("ERROR: Live trading is disabled!")
            print("Set ALLOW_LIVE_TRADING=true in .env to enable.")
            print("*" * 60)
            return 1

        print("*" * 60)
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        print("*" * 60)
        response = input("Type 'CONFIRM' to proceed: ")
        if response != "CONFIRM":
            print("Cancelled.")
            return 1

    # Load model configuration
    feature_columns, preprocessor, training_config = load_model_and_config(model_path)
    fixed_sl_ticks = training_config.get("fixed_sl_ticks", 120.0)
    logger.info(f"Using {len(feature_columns)} features")
    logger.info(f"Fixed SL: {fixed_sl_ticks} ticks")

    # Load model
    logger.info("Loading model...")

    # Create dummy environment for model loading
    dummy_df = pd.DataFrame({
        col: [0.0] * (settings.LOOKBACK_WINDOW + 10)
        for col in feature_columns
    })
    dummy_df["open"] = 15000.0
    dummy_df["high"] = 15010.0
    dummy_df["low"] = 14990.0
    dummy_df["close"] = 15005.0
    dummy_df["volume"] = 1000

    dummy_env = TradingEnv(
        df=dummy_df,
        feature_columns=feature_columns,
        fixed_sl_ticks=fixed_sl_ticks,
    )

    agent = TradingAgent(algorithm="PPO")
    agent.load(model_path / "final_model.zip", env=dummy_env)
    logger.info("Model loaded successfully")

    # Create trader based on mode
    if args.mode == "paper":
        trader = PaperTrader(
            model=agent,
            feature_columns=feature_columns,
            preprocessor=preprocessor,
            fixed_sl_ticks=fixed_sl_ticks,
            lookback_window=settings.LOOKBACK_WINDOW,
            max_position=args.max_position,
            max_daily_loss=args.max_daily_loss,
            log_dir=str(log_dir),
            use_live_data=True,  # Use live data for no delay
        )

        # Connect to IBKR (live data, simulated trades)
        if not trader.connect_sync():
            logger.error("Failed to connect to IBKR")
            return 1

        logger.info("=" * 50)
        logger.info("PAPER TRADING MODE - SIMULATED TRADES ONLY")
        logger.info(f"Using LIVE data feed (no delay)")
        logger.info(f"Trades logged to: {log_dir}")
        logger.info("=" * 50)

    elif args.mode == "live":
        trader = LiveTrader(
            model=agent,
            feature_columns=feature_columns,
            lookback_window=settings.LOOKBACK_WINDOW,
            max_position=args.max_position,
            max_daily_loss=args.max_daily_loss,
            risk_per_trade=args.risk_per_trade,
            use_stop_loss=args.use_stop_loss,
            stop_loss_atr_mult=args.stop_loss_atr,
            log_dir=log_dir,
        )

        # Connect to IBKR
        if not trader.connect_sync():
            logger.error("Failed to connect to IBKR live trading")
            return 1

    else:  # simulation
        if not args.simulation_data:
            logger.error("Simulation mode requires --simulation-data")
            return 1

        from trading.paper_trader import PaperTradingSimulator

        # Load simulation data
        sim_df = pd.read_parquet(args.simulation_data)

        # Compute features
        feature_engineer = FeatureEngineer()
        sim_df = feature_engineer.compute_all_features(sim_df)

        # Normalize
        available_features = [f for f in feature_columns if f in sim_df.columns]
        sim_df[available_features] = preprocessor.transform(sim_df[available_features])

        simulator = PaperTradingSimulator(
            model=agent,
            df=sim_df,
            feature_columns=available_features,
            lookback_window=settings.LOOKBACK_WINDOW,
        )

        logger.info("Running simulation...")
        results = simulator.run()

        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Trade Count: {results['trade_count']}")
        print("=" * 60 + "\n")

        # Save results
        results['results'].to_csv(log_dir / "simulation_results.csv", index=False)
        logger.info(f"Results saved to {log_dir}")

        return 0

    # Setup callbacks
    def on_trade(trade_data: dict):
        trading_logger.log_trade(
            trade_type="CLOSE",
            side="",
            price=trade_data["exit_price"],
            quantity=1,
            pnl=trade_data["pnl"],
            reason=trade_data["reason"],
        )

    def on_alert(level: str, message: str):
        logger.log(
            logging.WARNING if level in ["WARNING", "CRITICAL"] else logging.INFO,
            f"[ALERT] {message}"
        )

    trader.on_trade = on_trade
    if hasattr(trader, 'on_alert'):
        trader.on_alert = on_alert

    # Run trading
    manager = TradingManager(trader, trading_logger)
    manager.run()

    # Disconnect
    trader.disconnect()

    logger.info(f"Trading session complete. Logs saved to {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
