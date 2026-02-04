"""Order management for trading execution."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import logging
from threading import Lock
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"
    TRAILING_STOP = "TRAIL"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    ib_order_id: Optional[int] = None
    parent_order_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    notes: str = ""

    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


class OrderManager:
    """
    Manages order lifecycle and tracking.

    Handles order creation, submission, fills, and cancellations.
    """

    def __init__(self, max_orders_per_day: int = 100):
        """
        Initialize order manager.

        Args:
            max_orders_per_day: Maximum orders allowed per day
        """
        self.max_orders_per_day = max_orders_per_day
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.daily_order_count = 0
        self.lock = Lock()

        # Callbacks
        self.on_fill: Optional[Callable[[Order], None]] = None
        self.on_cancel: Optional[Callable[[Order], None]] = None
        self.on_reject: Optional[Callable[[Order], None]] = None

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_amount: Optional[float] = None,
        parent_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: Buy or Sell
            quantity: Number of contracts
            order_type: Type of order
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            trail_amount: Trail amount (for TRAILING_STOP)
            parent_order_id: Parent order for bracket orders

        Returns:
            Created Order object
        """
        with self.lock:
            # Check daily limit
            if self.daily_order_count >= self.max_orders_per_day:
                raise ValueError("Daily order limit reached")

            # Generate unique order ID
            order_id = str(uuid.uuid4())[:8]

            # Validate order parameters
            self._validate_order_params(order_type, price, stop_price)

            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                trail_amount=trail_amount,
                parent_order_id=parent_order_id,
            )

            self.orders[order_id] = order
            self.daily_order_count += 1

            logger.info(
                f"Created order {order_id}: {side.value} {quantity} {symbol} @ {order_type.value}"
            )

            return order

    def _validate_order_params(
        self,
        order_type: OrderType,
        price: Optional[float],
        stop_price: Optional[float],
    ) -> None:
        """Validate order parameters."""
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit orders require a price")

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError("Stop orders require a stop price")

        if order_type == OrderType.STOP_LIMIT and price is None:
            raise ValueError("Stop limit orders require both price and stop price")

    def submit_order(self, order: Order) -> None:
        """
        Mark order as submitted.

        Args:
            order: Order to submit
        """
        with self.lock:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            self.active_orders[order.order_id] = order

            logger.info(f"Submitted order {order.order_id}")

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: Optional[int] = None,
        commission: float = 0.0,
    ) -> None:
        """
        Record an order fill.

        Args:
            order_id: Order ID
            fill_price: Fill price
            fill_quantity: Filled quantity (default: full fill)
            commission: Commission charged
        """
        with self.lock:
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found")
                return

            order = self.orders[order_id]
            fill_qty = fill_quantity or order.quantity

            order.filled_quantity += fill_qty
            order.filled_price = fill_price
            order.commission += commission
            order.filled_at = datetime.now()

            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.filled_orders.append(order)
                self.active_orders.pop(order_id, None)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

            logger.info(
                f"Filled order {order_id}: {fill_qty} @ {fill_price}, "
                f"commission: ${commission:.2f}"
            )

            # Trigger callback
            if self.on_fill:
                self.on_fill(order)

    def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        with self.lock:
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found")
                return False

            order = self.orders[order_id]

            if not order.is_active():
                logger.warning(f"Order {order_id} is not active")
                return False

            order.status = OrderStatus.CANCELLED
            order.notes = reason
            self.active_orders.pop(order_id, None)

            logger.info(f"Cancelled order {order_id}: {reason}")

            if self.on_cancel:
                self.on_cancel(order)

            return True

    def reject_order(self, order_id: str, reason: str) -> None:
        """
        Reject an order.

        Args:
            order_id: Order ID
            reason: Rejection reason
        """
        with self.lock:
            if order_id not in self.orders:
                return

            order = self.orders[order_id]
            order.status = OrderStatus.REJECTED
            order.notes = reason
            self.active_orders.pop(order_id, None)

            logger.warning(f"Rejected order {order_id}: {reason}")

            if self.on_reject:
                self.on_reject(order)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())

    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return self.filled_orders.copy()

    def cancel_all_orders(self, reason: str = "Cancel all") -> int:
        """
        Cancel all active orders.

        Args:
            reason: Cancellation reason

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        for order_id in list(self.active_orders.keys()):
            if self.cancel_order(order_id, reason):
                cancelled += 1
        return cancelled

    def reset_daily_count(self) -> None:
        """Reset daily order count."""
        with self.lock:
            self.daily_order_count = 0
            logger.info("Reset daily order count")

    def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float] = None,
        take_profit_price: float = None,
        stop_loss_price: float = None,
        entry_type: OrderType = OrderType.MARKET,
    ) -> List[Order]:
        """
        Create a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Trading symbol
            side: Entry side
            quantity: Number of contracts
            entry_price: Entry limit price (for limit orders)
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            entry_type: Entry order type

        Returns:
            List of orders [entry, take_profit, stop_loss]
        """
        # Entry order
        entry_order = self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=entry_type,
            price=entry_price,
        )

        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # Take profit order
        tp_order = self.create_order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=take_profit_price,
            parent_order_id=entry_order.order_id,
        )

        # Stop loss order
        sl_order = self.create_order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            parent_order_id=entry_order.order_id,
        )

        entry_order.children = [tp_order.order_id, sl_order.order_id]

        return [entry_order, tp_order, sl_order]

    def get_pnl_summary(self) -> Dict[str, float]:
        """Get P&L summary from filled orders."""
        total_pnl = 0.0
        total_commission = 0.0

        for order in self.filled_orders:
            total_commission += order.commission

        return {
            "total_commission": total_commission,
            "total_orders": len(self.filled_orders),
            "active_orders": len(self.active_orders),
        }
