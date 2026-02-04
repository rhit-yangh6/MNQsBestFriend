"""Paper trading execution module."""

import asyncio
from datetime import datetime, time, date
from typing import Dict, Optional, Any, Callable, List
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from threading import Thread, Event
from queue import Queue
import csv

from ib_insync import IB, Future, util, Ticker, BarData
from ib_insync.contract import Contract

from config.settings import settings
from config.ib_config import ib_config
from .order_manager import OrderManager, Order, OrderSide, OrderType, OrderStatus
from env.position_sizer import PositionSizer
from data.features import FeatureEngineer

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading execution system.

    Connects to IBKR live data but simulates trades in memory.
    Logs all trades to daily CSV files.
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        preprocessor=None,
        sl_options: List[float] = None,
        lookback_window: int = 50,
        max_position: int = 1,
        max_daily_loss: float = 500.0,
        log_dir: str = "logs/paper_trading",
        use_live_data: bool = True,  # Connect to live for real-time data
    ):
        """
        Initialize paper trader.

        Args:
            model: Trained trading model
            feature_columns: Feature columns for model input
            preprocessor: Data preprocessor with fitted scaler
            sl_options: SL options in ticks (for learnable SL)
            lookback_window: Lookback window for state
            max_position: Maximum position size
            max_daily_loss: Maximum daily loss limit
            log_dir: Directory for trade logs
            use_live_data: Connect to live port for real-time data (no delay)
        """
        self.model = model
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.sl_options = sl_options or [80.0, 120.0, 160.0, 200.0]
        self.lookback_window = lookback_window
        self.max_position = max_position
        self.max_daily_loss = max_daily_loss
        self.use_live_data = use_live_data

        # Log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # IBKR connection
        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None

        # Feature engineering
        self.feature_engineer = FeatureEngineer()

        # Order management (for tracking only, no real orders)
        self.order_manager = OrderManager()
        self.position_sizer = PositionSizer(max_position=max_position)

        # State tracking
        self.position = 0
        self.entry_price = 0.0
        self.entry_time: Optional[datetime] = None
        self.current_sl_ticks = 0.0
        self.current_sl_level = 0
        self.stop_loss_price = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Data buffer for OHLCV bars
        self.bar_buffer: List[Dict] = []
        self.df_buffer: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[datetime] = None

        # Trade history for current day
        self.daily_trades: List[Dict] = []
        self.current_trade_date: Optional[date] = None

        # Control
        self.running = False
        self.stop_event = Event()

        # Callbacks
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_position_change: Optional[Callable[[int, float], None]] = None

    async def connect(self) -> bool:
        """Connect to IBKR for live data (simulated trades only)."""
        try:
            self.ib = IB()

            # Use live port for real-time data, but we won't place real orders
            if self.use_live_data:
                port = ib_config.LIVE_PORT  # 4001 - real-time data
                logger.info("Connecting to LIVE data feed (trades will be SIMULATED)")
            else:
                port = ib_config.PAPER_PORT  # 4002 - delayed data
                logger.info("Connecting to PAPER data feed (15-20 min delay)")

            await self.ib.connectAsync(
                host=ib_config.HOST,
                port=port,
                clientId=ib_config.CLIENT_ID + 200,  # Unique client ID for paper trader
                timeout=ib_config.TIMEOUT,
            )

            logger.info(f"Connected to IBKR on port {port}")

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
        # Save any remaining trades with summary before disconnect
        self._save_daily_trades(include_summary=True)

        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _get_csv_path(self, trade_date: date) -> Path:
        """Get CSV file path for a given date."""
        return self.log_dir / f"trades_{trade_date.strftime('%Y%m%d')}.csv"

    def _save_daily_trades(self, include_summary: bool = False) -> None:
        """Save daily trades to CSV with optional summary row."""
        if not self.daily_trades:
            return

        trade_date = self.current_trade_date or date.today()
        csv_path = self._get_csv_path(trade_date)

        # Check if file exists to determine if we need headers
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'action', 'side', 'entry_price', 'exit_price',
                'sl_level', 'sl_ticks', 'stop_loss_price', 'pnl', 'commission',
                'duration_bars', 'reason', 'position_after', 'daily_pnl', 'total_pnl'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for trade in self.daily_trades:
                writer.writerow(trade)

        logger.info(f"Saved {len(self.daily_trades)} trades to {csv_path}")
        self.daily_trades = []

        # Add summary if requested (end of day or shutdown)
        if include_summary:
            self._write_daily_summary(csv_path)

    def _write_daily_summary(self, csv_path: Path) -> None:
        """Write a summary row at the end of the daily CSV."""
        # Calculate summary stats
        total_trades = self.trade_count
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = self.daily_pnl / total_trades if total_trades > 0 else 0

        summary_path = csv_path.with_suffix('.summary.txt')

        with open(summary_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write(f"DAILY SUMMARY - {self.current_trade_date}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Trades:    {total_trades}\n")
            f.write(f"Winning Trades:  {self.winning_trades}\n")
            f.write(f"Losing Trades:   {self.losing_trades}\n")
            f.write(f"Win Rate:        {win_rate:.1f}%\n")
            f.write(f"Daily P&L:       ${self.daily_pnl:.2f}\n")
            f.write(f"Total P&L:       ${self.total_pnl:.2f}\n")
            f.write(f"Avg P&L/Trade:   ${avg_pnl:.2f}\n")
            f.write("=" * 50 + "\n")

            # SL level distribution
            if hasattr(self, '_sl_level_counts'):
                f.write("\nSL Level Distribution:\n")
                for level, count in self._sl_level_counts.items():
                    f.write(f"  Level {level}: {count} trades\n")

        logger.info(f"Saved daily summary to {summary_path}")

        # Also append summary row to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])  # Empty row
            writer.writerow(['--- SUMMARY ---', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
            writer.writerow([
                f'Trades: {total_trades}',
                f'Wins: {self.winning_trades}',
                f'Losses: {self.losing_trades}',
                f'Win Rate: {win_rate:.1f}%',
                '',
                '',
                '',
                '',
                f'Daily P&L: ${self.daily_pnl:.2f}',
                '',
                '',
                '',
                '',
                '',
                f'Total P&L: ${self.total_pnl:.2f}'
            ])

    def _log_trade(self, trade_data: Dict) -> None:
        """Log a trade to the daily buffer."""
        # Check if day changed
        today = date.today()
        if self.current_trade_date and self.current_trade_date != today:
            # Save previous day's trades
            self._save_daily_trades()

        self.current_trade_date = today
        self.daily_trades.append(trade_data)

        # Also save immediately for safety
        self._save_daily_trades()

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
        """Start IBKR-based paper trading with 5-min bars."""
        # Request historical bars first to fill buffer
        logger.info("Fetching historical bars to fill buffer...")
        bars = self.ib.reqHistoricalData(
            contract=self.contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,  # Keep updating with new bars
        )

        # Process historical bars
        for bar in bars:
            self._process_bar(bar)

        logger.info(f"Loaded {len(bars)} historical bars")

        # Set up handler for new bars
        self.ib.barUpdateEvent += self._on_bar_update

        logger.info("Started 5-min bar subscription")

    def _on_bar_update(self, bars: List[BarData], hasNewBar: bool) -> None:
        """Handle new bar data from IBKR."""
        if hasNewBar and bars:
            # Process the latest bar
            self._process_bar(bars[-1])

    def _process_bar(self, bar: BarData) -> None:
        """Process a new 5-min bar and potentially trade."""
        # Build bar data
        bar_data = {
            "timestamp": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }

        # Skip if same bar
        if self.last_bar_time and bar.date <= self.last_bar_time:
            return

        self.last_bar_time = bar.date

        # Add to buffer
        self.bar_buffer.append(bar_data)

        # Keep buffer at reasonable size
        max_buffer = self.lookback_window + 100
        if len(self.bar_buffer) > max_buffer:
            self.bar_buffer = self.bar_buffer[-max_buffer:]

        # Rebuild DataFrame with features
        self._update_feature_buffer()

        # Check if we have enough data
        if self.df_buffer is None or len(self.df_buffer) < self.lookback_window:
            logger.debug(f"Buffering data: {len(self.bar_buffer)}/{self.lookback_window} bars")
            return

        current_price = bar.close

        # Check weekend close first - no holding over weekend
        if self._check_weekend_close(current_price):
            return  # Position closed for weekend

        # Check SL
        if self.position != 0:
            if self._check_stop_loss(current_price):
                return  # Position was closed by SL

        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            if self.position != 0:
                self._close_position(current_price, "Daily loss limit reached")
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return

        # Check market hours
        if not self._is_trading_hours():
            return

        # Don't open new positions near weekend close
        if self._is_weekend_close_time():
            return

        # Get trading decision
        action_type, sl_level = self._get_model_action()

        # Execute action
        self._execute_action(action_type, sl_level, current_price)

    def _update_feature_buffer(self) -> None:
        """Update the feature DataFrame from bar buffer."""
        if len(self.bar_buffer) < 20:  # Need minimum bars for features
            return

        try:
            # Create DataFrame from buffer
            df = pd.DataFrame(self.bar_buffer)

            # Compute features
            df = self.feature_engineer.compute_all_features(df)

            # Normalize features if preprocessor available
            if self.preprocessor is not None:
                available_features = [f for f in self.feature_columns if f in df.columns]
                if available_features:
                    df[available_features] = self.preprocessor.transform(df[available_features])

            self.df_buffer = df

        except Exception as e:
            logger.error(f"Error computing features: {e}")

    def _get_model_action(self) -> tuple:
        """Get action from trained model (MultiDiscrete: action_type, sl_level)."""
        # Prepare observation
        obs = self._prepare_observation()

        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)

        # Parse MultiDiscrete action
        if isinstance(action, np.ndarray) and len(action) >= 2:
            action_type = int(action[0])
            sl_level = int(action[1])
        else:
            action_type = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            sl_level = 1  # Default medium

        return action_type, sl_level

    def _prepare_observation(self) -> Dict[str, np.ndarray]:
        """Prepare observation for model input (7-element position_info)."""
        # Build DataFrame from buffer
        if self.df_buffer is None or len(self.df_buffer) < self.lookback_window:
            # Not enough data yet, return zeros
            market_features = np.zeros((self.lookback_window, len(self.feature_columns)), dtype=np.float32)
            position_info = np.zeros(7, dtype=np.float32)
            return {"market_features": market_features, "position_info": position_info}

        # Get last lookback_window rows
        df = self.df_buffer.iloc[-self.lookback_window:].copy()

        # Get feature columns
        available_features = [f for f in self.feature_columns if f in df.columns]
        market_features = df[available_features].values.astype(np.float32)

        # Calculate position info (7 elements)
        unrealized_pnl = self._calculate_unrealized_pnl()
        time_in_position = 0
        if self.entry_time:
            time_in_position = len(self.bar_buffer) - self.bar_buffer.index(
                next((b for b in self.bar_buffer if b.get('timestamp') == self.entry_time), self.bar_buffer[0])
            )

        # Balance ratio (assuming $10k initial)
        initial_balance = 10000.0
        current_equity = initial_balance + self.total_pnl + unrealized_pnl
        balance_ratio = current_equity / initial_balance

        # SL distance ratio
        if self.position != 0 and self.current_sl_ticks > 0:
            sl_points = self.current_sl_ticks * settings.TICK_SIZE
            max_loss = sl_points * settings.POINT_VALUE
            pnl_to_sl_ratio = (unrealized_pnl + max_loss) / max_loss if max_loss > 0 else 1.0
        else:
            pnl_to_sl_ratio = 1.0

        # Max P&L ratio (simplified - would need tracking)
        max_pnl_ratio = 1.0

        # Normalized SL level
        sl_level_norm = self.current_sl_level / max(1, len(self.sl_options) - 1) if self.position != 0 else 0.5

        position_info = np.array([
            self.position,
            unrealized_pnl / initial_balance,
            min(time_in_position / 100, 1.0),
            balance_ratio,
            pnl_to_sl_ratio,
            max_pnl_ratio,
            sl_level_norm,
        ], dtype=np.float32)

        return {
            "market_features": market_features,
            "position_info": position_info,
        }

    def _execute_action(self, action_type: int, sl_level: int, current_price: float) -> None:
        """
        Execute trading action (SIMULATED - no real orders).

        Args:
            action_type: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
            sl_level: SL level index
            current_price: Current market price
        """
        if action_type == 0:  # HOLD
            # Check if SL hit while holding
            if self.position != 0:
                self._check_stop_loss(current_price)
            return

        elif action_type == 1:  # BUY
            if self.position == -1:
                self._close_position(current_price, "Signal reversal")
            if self.position <= 0:
                self._open_position(OrderSide.BUY, current_price, sl_level)

        elif action_type == 2:  # SELL
            if self.position == 1:
                self._close_position(current_price, "Signal reversal")
            if self.position >= 0:
                self._open_position(OrderSide.SELL, current_price, sl_level)

        elif action_type == 3:  # CLOSE
            if self.position != 0:
                self._close_position(current_price, "Model signal")

    def _check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit. Returns True if position was closed."""
        if self.position == 0 or self.stop_loss_price == 0:
            return False

        if self.position == 1:  # Long
            if current_price <= self.stop_loss_price:
                self._close_position(self.stop_loss_price, "STOP_LOSS")
                return True
        else:  # Short
            if current_price >= self.stop_loss_price:
                self._close_position(self.stop_loss_price, "STOP_LOSS")
                return True

        return False

    def _open_position(self, side: OrderSide, price: float, sl_level: int) -> None:
        """Open a new position (SIMULATED)."""
        try:
            # Simulate fill with slippage
            slippage = settings.SLIPPAGE_TICKS * settings.TICK_SIZE
            fill_price = price + slippage if side == OrderSide.BUY else price - slippage

            # Set SL based on chosen level
            sl_level = min(sl_level, len(self.sl_options) - 1)
            self.current_sl_level = sl_level
            self.current_sl_ticks = self.sl_options[sl_level]
            sl_points = self.current_sl_ticks * settings.TICK_SIZE

            # Calculate stop loss price
            if side == OrderSide.BUY:
                self.stop_loss_price = fill_price - sl_points
            else:
                self.stop_loss_price = fill_price + sl_points

            # Update position state
            self.position = 1 if side == OrderSide.BUY else -1
            self.entry_price = fill_price
            self.entry_time = datetime.now()
            self.trade_count += 1

            # Deduct commission
            self.daily_pnl -= settings.COMMISSION_PER_CONTRACT
            self.total_pnl -= settings.COMMISSION_PER_CONTRACT

            # Log trade
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'side': side.value,
                'entry_price': fill_price,
                'exit_price': None,
                'sl_level': sl_level,
                'sl_ticks': self.current_sl_ticks,
                'stop_loss_price': self.stop_loss_price,
                'pnl': None,
                'commission': settings.COMMISSION_PER_CONTRACT,
                'duration_bars': None,
                'reason': 'Model signal',
                'position_after': self.position,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
            }
            self._log_trade(trade_data)

            logger.info(
                f"[SIM] Opened {side.value} at {fill_price:.2f}, "
                f"SL={self.stop_loss_price:.2f} (level {sl_level}, {self.current_sl_ticks} ticks)"
            )

            if self.on_position_change:
                self.on_position_change(self.position, self.entry_price)

        except Exception as e:
            logger.error(f"Error opening position: {e}")

    def _close_position(self, exit_price: float, reason: str) -> None:
        """Close current position (SIMULATED)."""
        if self.position == 0:
            return

        try:
            # Simulate fill with slippage
            slippage = settings.SLIPPAGE_TICKS * settings.TICK_SIZE
            if self.position > 0:  # Closing long
                fill_price = exit_price - slippage
            else:  # Closing short
                fill_price = exit_price + slippage

            # Calculate P&L
            price_diff = fill_price - self.entry_price
            pnl = price_diff * self.position * settings.POINT_VALUE
            pnl -= settings.COMMISSION_PER_CONTRACT  # Exit commission

            self.daily_pnl += pnl
            self.total_pnl += pnl

            # Track win/loss
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Calculate duration
            duration_bars = 0
            if self.entry_time:
                # Approximate duration based on bars in buffer since entry
                duration_bars = len([b for b in self.bar_buffer
                                    if b.get('timestamp', datetime.min) >= self.entry_time])

            # Log trade
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'action': 'CLOSE',
                'side': 'LONG' if self.position > 0 else 'SHORT',
                'entry_price': self.entry_price,
                'exit_price': fill_price,
                'sl_level': self.current_sl_level,
                'sl_ticks': self.current_sl_ticks,
                'stop_loss_price': self.stop_loss_price,
                'pnl': pnl,
                'commission': settings.COMMISSION_PER_CONTRACT * 2,
                'duration_bars': duration_bars,
                'reason': reason,
                'position_after': 0,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
            }
            self._log_trade(trade_data)

            logger.info(
                f"[SIM] Closed {'LONG' if self.position > 0 else 'SHORT'} at {fill_price:.2f}, "
                f"P&L: ${pnl:.2f}, Reason: {reason}"
            )

            # Reset position state
            prev_position = self.position
            self.position = 0
            self.entry_price = 0.0
            self.entry_time = None
            self.stop_loss_price = 0.0
            self.current_sl_ticks = 0.0
            self.current_sl_level = 0

            # Callback
            if self.on_trade:
                self.on_trade({
                    "entry_price": self.entry_price,
                    "exit_price": fill_price,
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

    def _is_weekend_close_time(self) -> bool:
        """Check if it's time to close positions for the weekend."""
        now = datetime.now()
        # Check if it's Friday (weekday 4)
        if now.weekday() != settings.WEEKEND_CLOSE_DAY:
            return False

        # Check if past weekend close time
        close_time = time(settings.WEEKEND_CLOSE_TIME[0], settings.WEEKEND_CLOSE_TIME[1])
        return now.time() >= close_time

    def _check_weekend_close(self, current_price: float) -> bool:
        """Close position if it's weekend close time. Returns True if closed."""
        if self.position != 0 and self._is_weekend_close_time():
            self._close_position(current_price, "WEEKEND_CLOSE")
            logger.info("Closed position for weekend - no holding over weekend")
            return True
        return False

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
