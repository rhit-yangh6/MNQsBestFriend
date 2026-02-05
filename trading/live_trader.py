"""Live trading execution module with full risk management."""

import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Any, Callable, List
import logging
import numpy as np
import pandas as pd
from threading import Thread, Event, Lock
from queue import Queue
import json
from pathlib import Path

from ib_insync import IB, Future, util, Ticker, MarketOrder, LimitOrder, StopOrder
from ib_insync.contract import Contract
from ib_insync.order import Order as IBOrder

from config.settings import settings
from config.ib_config import ib_config
from .order_manager import OrderManager, OrderSide
from data.features import FeatureEngineer

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading execution system with comprehensive risk management.

    Features:
    - Real-time IBKR integration
    - Position sizing based on account equity
    - Maximum daily loss protection
    - Automatic stop-loss orders
    - Reconnection handling
    - Trade logging and alerts
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        preprocessor=None,
        lookback_window: int = 50,
        max_position: int = 1,
        max_daily_loss: float = 500.0,
        fixed_sl_ticks: float = 200.0,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize live trader.

        Args:
            model: Trained trading model
            feature_columns: Feature columns for model input
            preprocessor: Data preprocessor with fitted scaler
            lookback_window: Lookback window for state
            max_position: Maximum position size
            max_daily_loss: Maximum daily loss in USD
            fixed_sl_ticks: Fixed stop loss in ticks (200 = $100)
            log_dir: Directory for trade logs
        """
        self.model = model
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.lookback_window = lookback_window
        self.max_position = max_position
        self.max_daily_loss = max_daily_loss
        self.fixed_sl_ticks = fixed_sl_ticks
        self.log_dir = log_dir or settings.LOG_DIR

        # IBKR connection
        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None

        # Components
        self.order_manager = OrderManager()
        self.feature_engineer = FeatureEngineer()

        # SL order tracking
        self.sl_order_id: Optional[int] = None

        # State
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.account_value = 0.0
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.trade_count = 0
        self.last_trade_time: Optional[datetime] = None

        # Data buffers
        self.bar_buffer: List[Dict] = []
        self.tick_buffer: List[Dict] = []

        # Control
        self.running = False
        self.trading_enabled = True
        self.stop_event = Event()
        self.lock = Lock()

        # Callbacks
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_alert: Optional[Callable[[str, str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Trade log
        self.trade_log: List[Dict] = []

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> bool:
        """Connect to IBKR live trading."""
        try:
            self.ib = IB()
            port = ib_config.get_port(paper=False)  # LIVE trading

            await self.ib.connectAsync(
                host=ib_config.HOST,
                port=port,
                clientId=ib_config.CLIENT_ID,
                timeout=ib_config.TIMEOUT,
            )

            logger.warning("CONNECTED TO LIVE TRADING - REAL MONEY AT RISK")

            # Get account info
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == "NetLiquidation":
                    self.account_value = float(av.value)
                    self.daily_start_equity = self.account_value
                    break

            logger.info(f"Account value: ${self.account_value:.2f}")

            # Get MNQ contract
            self.contract = Future(
                symbol=settings.SYMBOL,
                exchange=settings.EXCHANGE,
                currency=settings.CURRENCY,
            )
            qualified = self.ib.qualifyContracts(self.contract)
            if qualified:
                self.contract = qualified[0]
                logger.info(f"Using contract: {self.contract}")
            else:
                raise ValueError("Could not qualify MNQ contract")

            # Set up event handlers
            self.ib.disconnectedEvent += self._on_disconnect
            self.ib.errorEvent += self._on_error

            self._send_alert("INFO", f"Connected to live trading. Account: ${self.account_value:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._send_alert("ERROR", f"Connection failed: {e}")
            return False

    def connect_sync(self) -> bool:
        """Synchronous connect wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.connect())

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _on_disconnect(self) -> None:
        """Handle disconnect event."""
        logger.warning("Disconnected from IBKR")
        self._send_alert("WARNING", "Disconnected from IBKR - attempting reconnection")

        # Attempt reconnection
        if self.running:
            self._attempt_reconnect()

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle error event."""
        logger.error(f"IBKR Error {errorCode}: {errorString}")

        if errorCode in [1100, 1101, 1102]:  # Connection errors
            self._send_alert("ERROR", f"Connection error: {errorString}")

        if self.on_error:
            self.on_error(errorString)

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to IBKR."""
        max_attempts = ib_config.MAX_RECONNECT_ATTEMPTS

        for attempt in range(max_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(ib_config.RECONNECT_DELAY))

            if self.connect_sync():
                logger.info("Reconnected successfully")
                self._send_alert("INFO", "Reconnected to IBKR")
                return

        logger.error("Failed to reconnect after maximum attempts")
        self._send_alert("CRITICAL", "Failed to reconnect - manual intervention required")
        self.stop()

    def start(self) -> None:
        """Start live trading."""
        if not self.ib or not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return

        self.running = True
        self.stop_event.clear()
        self.trading_enabled = True

        # Subscribe to real-time data
        self._subscribe_market_data()

        # Start monitoring thread
        self._start_monitoring()

        logger.info("Live trading started")
        self._send_alert("INFO", "Live trading started")

    def stop(self) -> None:
        """Stop live trading."""
        self.running = False
        self.stop_event.set()

        # Close any open position
        if self.position != 0:
            self._emergency_close("Trading stopped")

        # Save trade log
        self._save_trade_log()

        logger.info("Live trading stopped")
        self._send_alert("INFO", "Live trading stopped")

    def _subscribe_market_data(self) -> None:
        """Subscribe to real-time market data."""
        # Subscribe to 5-second bars
        self.ib.reqRealTimeBars(
            contract=self.contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=False,
        )

        # Set up handlers
        self.ib.pendingTickersEvent += self._on_tick_update

        logger.info("Subscribed to market data")

    def _on_tick_update(self, tickers: List[Ticker]) -> None:
        """Handle real-time tick updates."""
        for ticker in tickers:
            if ticker.contract == self.contract:
                self._process_tick(ticker)

    def _process_tick(self, ticker: Ticker) -> None:
        """Process a tick update."""
        with self.lock:
            current_time = datetime.now()

            # Build bar from ticks (aggregate to 5-minute)
            tick_data = {
                "timestamp": current_time,
                "price": ticker.last,
                "volume": ticker.volume,
            }
            self.tick_buffer.append(tick_data)

            # Check if we should process a new bar
            if self._should_process_bar():
                self._aggregate_and_process_bar()

            # Check stop-loss
            if self.position != 0 and self.use_stop_loss:
                self._check_stop_loss(ticker.last)

    def _should_process_bar(self) -> bool:
        """Check if we should aggregate ticks into a bar."""
        if not self.tick_buffer:
            return False

        first_tick_time = self.tick_buffer[0]["timestamp"]
        elapsed = (datetime.now() - first_tick_time).total_seconds()

        return elapsed >= settings.TIMEFRAME_MINUTES * 60

    def _aggregate_and_process_bar(self) -> None:
        """Aggregate ticks into a bar and process."""
        if not self.tick_buffer:
            return

        prices = [t["price"] for t in self.tick_buffer]
        volumes = [t["volume"] for t in self.tick_buffer]

        bar_data = {
            "timestamp": self.tick_buffer[0]["timestamp"],
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
        }

        self.bar_buffer.append(bar_data)
        self.tick_buffer = []

        # Keep buffer at appropriate size
        if len(self.bar_buffer) > self.lookback_window + 10:
            self.bar_buffer = self.bar_buffer[-self.lookback_window - 5:]

        # Process trading logic
        if len(self.bar_buffer) >= self.lookback_window:
            self._process_trading_signal()

    def _process_trading_signal(self) -> None:
        """Process trading signal from model."""
        # Check if trading is enabled
        if not self.trading_enabled:
            return

        # Check weekend close first - no holding over weekend
        if self._check_weekend_close():
            return

        # Check daily loss limit
        if self._check_daily_loss_limit():
            return

        # Check market hours
        if not self._is_trading_hours():
            return

        # Don't open new positions near weekend close
        if self._is_weekend_close_time():
            return

        # Get model action
        action = self._get_model_action()

        # Execute action
        current_price = self.bar_buffer[-1]["close"]
        self._execute_action(action, current_price)

    def _get_model_action(self) -> int:
        """Get action from trained model."""
        obs = self._prepare_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def _prepare_observation(self) -> Dict[str, np.ndarray]:
        """Prepare observation for model (4-element position_info)."""
        df = pd.DataFrame(self.bar_buffer[-self.lookback_window:])

        # Calculate features
        df_with_features = self.feature_engineer.compute_all_features(df)

        # Normalize if preprocessor available
        available_features = [f for f in self.feature_columns if f in df_with_features.columns]
        if self.preprocessor is not None and available_features:
            df_with_features[available_features] = self.preprocessor.transform(
                df_with_features[available_features]
            )

        market_features = df_with_features[available_features].values

        # Calculate time in position
        time_in_position = 0
        if self.last_trade_time:
            # Approximate bars since entry
            time_in_position = len([b for b in self.bar_buffer
                                   if b.get('timestamp', datetime.min) >= self.last_trade_time])

        # Position info (4 elements - matches current model)
        unrealized_pnl = self._calculate_unrealized_pnl()
        initial_balance = 10000.0
        current_equity = initial_balance + self.daily_pnl + unrealized_pnl
        equity_ratio = current_equity / initial_balance

        position_info = np.array([
            float(self.position),
            unrealized_pnl / 100.0,  # Normalize ~$100 typical
            min(time_in_position / 50.0, 2.0),  # Normalize time
            equity_ratio,
        ], dtype=np.float32)

        return {
            "market_features": market_features.astype(np.float32),
            "position_info": position_info,
        }

    def _execute_action(self, action: int, current_price: float) -> None:
        """Execute trading action with full risk management."""
        if action == 0:  # HOLD
            return

        elif action == 1:  # BUY
            if self.position == -1:
                self._close_position("Signal reversal to long")
            if self.position <= 0:
                self._open_position(OrderSide.BUY, current_price)

        elif action == 2:  # SELL
            if self.position == 1:
                self._close_position("Signal reversal to short")
            if self.position >= 0:
                self._open_position(OrderSide.SELL, current_price)

        elif action == 3:  # CLOSE
            if self.position != 0:
                self._close_position("Model close signal")

    def _open_position(self, side: OrderSide, price: float) -> None:
        """Open a position with fixed stop-loss placed immediately after fill."""
        try:
            size = self.max_position

            # Create and submit market order
            action = "BUY" if side == OrderSide.BUY else "SELL"
            order = MarketOrder(action, size)

            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)  # Wait for fill

            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice

                # Update state
                self.position = size if side == OrderSide.BUY else -size
                self.entry_price = fill_price
                self.last_trade_time = datetime.now()

                # Calculate fixed stop-loss price
                sl_points = self.fixed_sl_ticks * settings.TICK_SIZE
                if side == OrderSide.BUY:
                    self.stop_loss_price = fill_price - sl_points
                else:
                    self.stop_loss_price = fill_price + sl_points

                # Place SL order IMMEDIATELY after fill
                self._place_stop_loss_order(self.stop_loss_price, size, side)

                # Log trade
                self._log_trade({
                    "type": "OPEN",
                    "side": side.value,
                    "price": fill_price,
                    "size": size,
                    "sl_ticks": self.fixed_sl_ticks,
                    "stop_loss": self.stop_loss_price,
                    "timestamp": datetime.now().isoformat(),
                })

                logger.info(
                    f"Opened {side.value} {size} @ {fill_price:.2f}, "
                    f"SL @ {self.stop_loss_price:.2f} ({self.fixed_sl_ticks} ticks)"
                )
                self._send_alert("TRADE", f"Opened {side.value} @ {fill_price:.2f}, SL @ {self.stop_loss_price:.2f}")

            else:
                logger.warning(f"Order not filled: {trade.orderStatus.status}")

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            self._send_alert("ERROR", f"Failed to open position: {e}")

    def _close_position(self, reason: str) -> None:
        """Close current position."""
        if self.position == 0:
            return

        try:
            # Cancel any existing stop-loss orders
            self._cancel_stop_loss_orders()

            # Submit close order
            action = "SELL" if self.position > 0 else "BUY"
            size = abs(self.position)
            order = MarketOrder(action, size)

            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)

            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice

                # Calculate P&L
                price_diff = fill_price - self.entry_price
                pnl = price_diff * self.position * settings.POINT_VALUE
                pnl -= settings.COMMISSION_PER_CONTRACT * 2

                self.daily_pnl += pnl
                self.trade_count += 1

                # Log trade
                self._log_trade({
                    "type": "CLOSE",
                    "price": fill_price,
                    "size": size,
                    "pnl": pnl,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                })

                logger.info(f"Closed position @ {fill_price:.2f}, P&L: ${pnl:.2f}, Reason: {reason}")
                self._send_alert("TRADE", f"Closed @ {fill_price:.2f}, P&L: ${pnl:.2f}")

                # Reset state
                self.position = 0
                self.entry_price = 0.0
                self.stop_loss_price = 0.0
                self.sl_order_id = None

                if self.on_trade:
                    self.on_trade({
                        "exit_price": fill_price,
                        "pnl": pnl,
                        "reason": reason,
                    })

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            self._send_alert("ERROR", f"Failed to close position: {e}")

    def _emergency_close(self, reason: str) -> None:
        """Emergency close with immediate execution."""
        logger.warning(f"EMERGENCY CLOSE: {reason}")
        self._send_alert("CRITICAL", f"Emergency close: {reason}")
        self._close_position(reason)

    def _place_stop_loss_order(
        self, stop_price: float, size: int, entry_side: OrderSide
    ) -> None:
        """Place a stop-loss order immediately after entry fill."""
        try:
            action = "SELL" if entry_side == OrderSide.BUY else "BUY"
            order = StopOrder(action, size, stop_price)
            order.transmit = True
            order.outsideRth = True  # Allow execution outside regular hours
            order.tif = 'GTC'  # Good till cancelled

            trade = self.ib.placeOrder(self.contract, order)
            self.sl_order_id = trade.order.orderId

            logger.info(f"Placed SL order #{self.sl_order_id}: {action} @ {stop_price:.2f}")

        except Exception as e:
            logger.error(f"Error placing stop-loss: {e}")
            self._send_alert("ERROR", f"Failed to place SL order: {e}")

    def _cancel_stop_loss_orders(self) -> None:
        """Cancel existing stop-loss order."""
        try:
            if self.sl_order_id:
                for trade in self.ib.openTrades():
                    if trade.order.orderId == self.sl_order_id:
                        self.ib.cancelOrder(trade.order)
                        logger.info(f"Cancelled SL order #{self.sl_order_id}")
                        break
                self.sl_order_id = None
            else:
                # Fallback: cancel all stop orders for this contract
                for trade in self.ib.openTrades():
                    if trade.contract == self.contract and isinstance(trade.order, StopOrder):
                        self.ib.cancelOrder(trade.order)
                        logger.info("Cancelled stop-loss order")
        except Exception as e:
            logger.error(f"Error cancelling stop-loss: {e}")

    def _check_stop_loss(self, current_price: float) -> None:
        """Check if stop-loss should be triggered."""
        if self.position == 0 or self.stop_loss_price == 0:
            return

        triggered = False
        if self.position > 0 and current_price <= self.stop_loss_price:
            triggered = True
        elif self.position < 0 and current_price >= self.stop_loss_price:
            triggered = True

        if triggered:
            self._close_position("Stop-loss triggered")

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is reached."""
        if self.daily_pnl <= -self.max_daily_loss:
            if self.trading_enabled:
                self.trading_enabled = False
                logger.warning("Daily loss limit reached - trading disabled")
                self._send_alert("WARNING", f"Daily loss limit reached: ${self.daily_pnl:.2f}")

                if self.position != 0:
                    self._close_position("Daily loss limit")

            return True
        return False

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0 or not self.bar_buffer:
            return 0.0

        current_price = self.bar_buffer[-1]["close"]
        price_diff = current_price - self.entry_price
        return price_diff * self.position * settings.POINT_VALUE

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()

        # Skip weekends
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        rth_start = time(settings.RTH_START[0], settings.RTH_START[1])
        rth_end = time(settings.RTH_END[0], settings.RTH_END[1])

        return rth_start <= current_time <= rth_end

    def _is_weekend_close_time(self) -> bool:
        """Check if it's time to close positions for the weekend."""
        now = datetime.now()
        # Check if it's Friday (weekday 4)
        if now.weekday() != settings.WEEKEND_CLOSE_DAY:
            return False

        # Check if past weekend close time
        close_time = time(settings.WEEKEND_CLOSE_TIME[0], settings.WEEKEND_CLOSE_TIME[1])
        return now.time() >= close_time

    def _check_weekend_close(self) -> bool:
        """Close position if it's weekend close time. Returns True if closed."""
        if self.position != 0 and self._is_weekend_close_time():
            self._close_position("WEEKEND_CLOSE")
            logger.info("Closed position for weekend - no holding over weekend")
            self._send_alert("INFO", "Closed position for weekend")
            return True
        return False

    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        def monitor_loop():
            while self.running:
                try:
                    # Update account value
                    if self.ib and self.ib.isConnected():
                        account_values = self.ib.accountValues()
                        for av in account_values:
                            if av.tag == "NetLiquidation":
                                self.account_value = float(av.value)
                                break

                    # Check for daily reset (new trading day)
                    now = datetime.now()
                    if now.hour == 9 and now.minute < 5:
                        self._reset_daily_stats()

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")

                self.stop_event.wait(60)  # Check every minute

        thread = Thread(target=monitor_loop, daemon=True)
        thread.start()

    def _reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_start_equity = self.account_value
        self.trading_enabled = True
        self.order_manager.reset_daily_count()
        logger.info("Reset daily statistics")

    def _send_alert(self, level: str, message: str) -> None:
        """Send alert notification."""
        logger.log(
            logging.WARNING if level in ["WARNING", "CRITICAL"] else logging.INFO,
            f"[{level}] {message}"
        )

        if self.on_alert:
            self.on_alert(level, message)

    def _log_trade(self, trade_data: Dict) -> None:
        """Log trade to file."""
        self.trade_log.append(trade_data)

    def _save_trade_log(self) -> None:
        """Save trade log to file."""
        if not self.trade_log:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f"trades_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.trade_log, f, indent=2)

        logger.info(f"Saved trade log to {filepath}")

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        return {
            "running": self.running,
            "trading_enabled": self.trading_enabled,
            "connected": self.ib.isConnected() if self.ib else False,
            "position": self.position,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "fixed_sl_ticks": self.fixed_sl_ticks,
            "sl_order_active": self.sl_order_id is not None,
            "unrealized_pnl": self._calculate_unrealized_pnl(),
            "daily_pnl": self.daily_pnl,
            "account_value": self.account_value,
            "trade_count": self.trade_count,
        }
