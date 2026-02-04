"""Paper trading execution module."""

import asyncio
from datetime import datetime, time
from typing import Dict, Optional, Any, Callable, List
import logging
import numpy as np
import pandas as pd
from threading import Thread, Event
from queue import Queue

from ib_insync import IB, Future, util, Ticker
from ib_insync.contract import Contract

from config.settings import settings
from config.ib_config import ib_config
from .order_manager import OrderManager, Order, OrderSide, OrderType, OrderStatus
from env.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading execution system.

    Connects to IBKR paper trading account for realistic simulation.
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        lookback_window: int = 50,
        max_position: int = 1,
        max_daily_loss: float = 500.0,
        use_ibkr: bool = True,
    ):
        """
        Initialize paper trader.

        Args:
            model: Trained trading model
            feature_columns: Feature columns for model input
            lookback_window: Lookback window for state
            max_position: Maximum position size
            max_daily_loss: Maximum daily loss limit
            use_ibkr: Whether to use IBKR connection
        """
        self.model = model
        self.feature_columns = feature_columns
        self.lookback_window = lookback_window
        self.max_position = max_position
        self.max_daily_loss = max_daily_loss
        self.use_ibkr = use_ibkr

        # IBKR connection
        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None
        self.ticker: Optional[Ticker] = None

        # Order management
        self.order_manager = OrderManager()
        self.position_sizer = PositionSizer(max_position=max_position)

        # State tracking
        self.position = 0
        self.entry_price = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0

        # Data buffer
        self.bar_buffer: List[Dict] = []
        self.last_bar_time: Optional[datetime] = None

        # Control
        self.running = False
        self.stop_event = Event()

        # Callbacks
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_position_change: Optional[Callable[[int, float], None]] = None

    async def connect(self) -> bool:
        """Connect to IBKR paper trading."""
        if not self.use_ibkr:
            logger.info("Running in simulation mode (no IBKR)")
            return True

        try:
            self.ib = IB()
            port = ib_config.get_port(paper=True)

            await self.ib.connectAsync(
                host=ib_config.HOST,
                port=port,
                clientId=ib_config.CLIENT_ID + 100,  # Different client ID for paper
                timeout=ib_config.TIMEOUT,
            )

            logger.info(f"Connected to IBKR paper trading on port {port}")

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

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def connect_sync(self) -> bool:
        """Synchronous connect wrapper."""
        return asyncio.get_event_loop().run_until_complete(self.connect())

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def start(self) -> None:
        """Start paper trading loop."""
        self.running = True
        self.stop_event.clear()

        if self.use_ibkr:
            self._start_ibkr_trading()
        else:
            logger.info("Paper trader started in simulation mode")

    def stop(self) -> None:
        """Stop paper trading."""
        self.running = False
        self.stop_event.set()

        # Close any open position
        if self.position != 0:
            self._close_position("End of session")

        logger.info("Paper trader stopped")

    def _start_ibkr_trading(self) -> None:
        """Start IBKR-based paper trading."""
        # Subscribe to real-time bars
        self.ib.reqRealTimeBars(
            contract=self.contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=False,
        )

        # Set up bar handler
        self.ib.pendingTickersEvent += self._on_bar_update

        logger.info("Started real-time bar subscription")

    def _on_bar_update(self, tickers: List[Ticker]) -> None:
        """Handle new bar data from IBKR."""
        for ticker in tickers:
            if ticker.contract == self.contract:
                self._process_bar(ticker)

    def _process_bar(self, ticker: Ticker) -> None:
        """Process a new bar and potentially trade."""
        # Build bar data
        bar_data = {
            "timestamp": datetime.now(),
            "open": ticker.open,
            "high": ticker.high,
            "low": ticker.low,
            "close": ticker.close,
            "volume": ticker.volume,
        }

        # Add to buffer
        self.bar_buffer.append(bar_data)

        # Keep buffer at lookback window size
        if len(self.bar_buffer) > self.lookback_window + 10:
            self.bar_buffer = self.bar_buffer[-self.lookback_window - 5:]

        # Check if we have enough data
        if len(self.bar_buffer) < self.lookback_window:
            return

        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            if self.position != 0:
                self._close_position("Daily loss limit reached")
            return

        # Check market hours
        if not self._is_trading_hours():
            return

        # Get trading decision
        action = self._get_model_action()

        # Execute action
        self._execute_action(action, ticker.close)

    def _get_model_action(self) -> int:
        """Get action from trained model."""
        # Prepare observation
        obs = self._prepare_observation()

        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)

        return int(action)

    def _prepare_observation(self) -> Dict[str, np.ndarray]:
        """Prepare observation for model input."""
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.bar_buffer[-self.lookback_window:])

        # Get features (assuming features are pre-computed or we compute them)
        # For simplicity, using available columns
        market_features = df[["open", "high", "low", "close", "volume"]].values

        # Normalize position info
        position_info = np.array([
            np.sign(self.position),
            self._calculate_unrealized_pnl() / 1000,  # Normalized
            0.0,  # Time in position (simplified)
            1.0,  # Balance ratio (simplified)
        ], dtype=np.float32)

        return {
            "market_features": market_features.astype(np.float32),
            "position_info": position_info,
        }

    def _execute_action(self, action: int, current_price: float) -> None:
        """
        Execute trading action.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
            current_price: Current market price
        """
        if action == 0:  # HOLD
            return

        elif action == 1:  # BUY
            if self.position == -1:
                self._close_position("Signal reversal")
            if self.position <= 0:
                self._open_position(OrderSide.BUY, current_price)

        elif action == 2:  # SELL
            if self.position == 1:
                self._close_position("Signal reversal")
            if self.position >= 0:
                self._open_position(OrderSide.SELL, current_price)

        elif action == 3:  # CLOSE
            if self.position != 0:
                self._close_position("Model signal")

    def _open_position(self, side: OrderSide, price: float) -> None:
        """Open a new position."""
        try:
            # Create order
            order = self.order_manager.create_order(
                symbol=settings.SYMBOL,
                side=side,
                quantity=1,
                order_type=OrderType.MARKET,
            )

            if self.use_ibkr and self.ib:
                # Submit to IBKR
                ib_order = self._create_ib_order(side, 1)
                trade = self.ib.placeOrder(self.contract, ib_order)
                self.order_manager.submit_order(order)

                # Wait for fill (simplified)
                self.ib.sleep(1)

                if trade.orderStatus.status == "Filled":
                    fill_price = trade.orderStatus.avgFillPrice
                    self.order_manager.fill_order(
                        order.order_id,
                        fill_price,
                        commission=settings.COMMISSION_PER_CONTRACT,
                    )
            else:
                # Simulate fill
                slippage = settings.SLIPPAGE_TICKS * settings.TICK_SIZE
                fill_price = price + slippage if side == OrderSide.BUY else price - slippage
                self.order_manager.fill_order(
                    order.order_id,
                    fill_price,
                    commission=settings.COMMISSION_PER_CONTRACT,
                )

            # Update position state
            self.position = 1 if side == OrderSide.BUY else -1
            self.entry_price = order.filled_price
            self.trade_count += 1

            logger.info(
                f"Opened {side.value} position at {order.filled_price:.2f}"
            )

            if self.on_position_change:
                self.on_position_change(self.position, self.entry_price)

        except Exception as e:
            logger.error(f"Error opening position: {e}")

    def _close_position(self, reason: str) -> None:
        """Close current position."""
        if self.position == 0:
            return

        try:
            side = OrderSide.SELL if self.position > 0 else OrderSide.BUY

            order = self.order_manager.create_order(
                symbol=settings.SYMBOL,
                side=side,
                quantity=1,
                order_type=OrderType.MARKET,
            )

            if self.use_ibkr and self.ib:
                ib_order = self._create_ib_order(side, 1)
                trade = self.ib.placeOrder(self.contract, ib_order)
                self.order_manager.submit_order(order)
                self.ib.sleep(1)

                if trade.orderStatus.status == "Filled":
                    fill_price = trade.orderStatus.avgFillPrice
                    self.order_manager.fill_order(
                        order.order_id,
                        fill_price,
                        commission=settings.COMMISSION_PER_CONTRACT,
                    )
            else:
                # Simulate fill
                current_price = self.bar_buffer[-1]["close"] if self.bar_buffer else self.entry_price
                slippage = settings.SLIPPAGE_TICKS * settings.TICK_SIZE
                fill_price = current_price - slippage if side == OrderSide.SELL else current_price + slippage
                self.order_manager.fill_order(
                    order.order_id,
                    fill_price,
                    commission=settings.COMMISSION_PER_CONTRACT,
                )

            # Calculate P&L
            price_diff = order.filled_price - self.entry_price
            pnl = price_diff * self.position * settings.POINT_VALUE
            pnl -= settings.COMMISSION_PER_CONTRACT * 2  # Round trip

            self.daily_pnl += pnl
            self.total_pnl += pnl

            logger.info(
                f"Closed position at {order.filled_price:.2f}, "
                f"P&L: ${pnl:.2f}, Reason: {reason}"
            )

            # Reset position state
            self.position = 0
            self.entry_price = 0.0

            # Callback
            if self.on_trade:
                self.on_trade({
                    "exit_price": order.filled_price,
                    "pnl": pnl,
                    "reason": reason,
                })

            if self.on_position_change:
                self.on_position_change(0, 0.0)

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def _create_ib_order(self, side: OrderSide, quantity: int):
        """Create IB order object."""
        from ib_insync import MarketOrder
        action = "BUY" if side == OrderSide.BUY else "SELL"
        return MarketOrder(action, quantity)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate current unrealized P&L."""
        if self.position == 0 or not self.bar_buffer:
            return 0.0

        current_price = self.bar_buffer[-1]["close"]
        price_diff = current_price - self.entry_price
        return price_diff * self.position * settings.POINT_VALUE

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now().time()
        rth_start = time(settings.RTH_START[0], settings.RTH_START[1])
        rth_end = time(settings.RTH_END[0], settings.RTH_END[1])

        # For now, allow trading during RTH
        return rth_start <= now <= rth_end

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        return {
            "running": self.running,
            "connected": self.ib.isConnected() if self.ib else False,
            "position": self.position,
            "entry_price": self.entry_price,
            "unrealized_pnl": self._calculate_unrealized_pnl(),
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "active_orders": len(self.order_manager.get_active_orders()),
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.order_manager.reset_daily_count()
        logger.info("Reset daily statistics")


class PaperTradingSimulator:
    """
    Offline paper trading simulator for testing without IBKR connection.
    """

    def __init__(
        self,
        model,
        df: pd.DataFrame,
        feature_columns: List[str],
        lookback_window: int = 50,
    ):
        """
        Initialize simulator.

        Args:
            model: Trained model
            df: Historical data for simulation
            feature_columns: Feature columns
            lookback_window: Lookback window
        """
        self.model = model
        self.df = df
        self.feature_columns = feature_columns
        self.lookback_window = lookback_window

        self.trader = PaperTrader(
            model=model,
            feature_columns=feature_columns,
            lookback_window=lookback_window,
            use_ibkr=False,
        )

    def run(self) -> Dict[str, Any]:
        """Run simulation on historical data."""
        results = []

        for i in range(self.lookback_window, len(self.df)):
            # Build bar buffer
            bar_data = self.df.iloc[i].to_dict()
            bar_data["timestamp"] = self.df.index[i]
            self.trader.bar_buffer.append(bar_data)

            if len(self.trader.bar_buffer) > self.lookback_window + 10:
                self.trader.bar_buffer = self.trader.bar_buffer[-self.lookback_window - 5:]

            # Get action and execute
            action = self.trader._get_model_action()
            current_price = self.df.iloc[i]["close"]
            self.trader._execute_action(action, current_price)

            # Record state
            results.append({
                "timestamp": self.df.index[i],
                "price": current_price,
                "position": self.trader.position,
                "pnl": self.trader.total_pnl,
            })

        # Close final position
        if self.trader.position != 0:
            self.trader._close_position("End of simulation")

        return {
            "results": pd.DataFrame(results),
            "total_pnl": self.trader.total_pnl,
            "trade_count": self.trader.trade_count,
            "final_status": self.trader.get_status(),
        }
