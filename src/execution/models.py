from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class PyramidingLimitBreachError(Exception):
    """Raised when an entry lot exceeds the configured maximum pyramiding level."""
    pass


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"


@dataclass(frozen=True)
class PriceTick:
    symbol: str
    price: Decimal
    timestamp: datetime


@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    side: OrderSide
    execution_price: Decimal
    filled_quantity: Decimal
    timestamp: Optional[datetime] = None


@dataclass
class PositionLot:
    lot_id: str
    quantity: Decimal
    price: Decimal

    def __post_init__(self) -> None:
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.price <= Decimal("0"):
            raise ValueError(f"Price must be positive, got {self.price}")


@dataclass
class Trade:
    symbol: str
    price: Decimal
    quantity: Decimal
    maker_order_id: str
    taker_order_id: str
    timestamp: Optional[float] = None
    trade_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.price <= Decimal("0"):
            raise ValueError(f"Price must be positive, got {self.price}")


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None
    watermark: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[Decimal] = None
    filled_quantity: Optional[Decimal] = None
    remaining_quantity: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    price: Optional[Decimal] = None
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.price is not None and self.price <= Decimal("0"):
            raise ValueError(f"Price must be positive, got {self.price}")

        if self.limit_price is not None and self.limit_price <= Decimal("0"):
            raise ValueError(f"Limit price must be positive, got {self.limit_price}")

        if self.price is not None and self.limit_price is None:
            self.limit_price = self.price
        elif self.limit_price is not None and self.price is None:
            self.price = self.limit_price

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit_price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop_price")

        if self.order_type == OrderType.TRAILING_STOP and self.trailing_delta is None:
            raise ValueError("Trailing stop orders require a trailing_delta")

        if self.stop_price is not None and self.stop_price <= Decimal("0"):
            raise ValueError(f"Stop price must be positive, got {self.stop_price}")

        if self.trailing_delta is not None and self.trailing_delta <= Decimal("0"):
            raise ValueError(f"Trailing delta must be positive, got {self.trailing_delta}")

        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        elif self.remaining_quantity < Decimal("0"):
            raise ValueError(f"Remaining quantity cannot be negative, got {self.remaining_quantity}")