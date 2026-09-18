"""Domain models and enumerations for trailing stop order management."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    """Side of the order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    ACTIVE = "ACTIVE"
    PENDING = "PENDING"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


class DeltaType(str, Enum):
    """Type of trailing delta measurement."""

    AMOUNT = "AMOUNT"
    PERCENTAGE = "PERCENTAGE"


@dataclass
class TrailingStopOrder:
    """Represents a dynamic trailing stop order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    initial_price: Decimal
    trailing_delta: Decimal
    stop_price: Decimal
    delta_type: DeltaType = DeltaType.AMOUNT
    activation_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.ACTIVE
    highest_price: Optional[Decimal] = None
    lowest_price: Optional[Decimal] = None
    trigger_price: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Validate trailing stop order invariants upon creation."""
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.trailing_delta <= Decimal("0"):
            raise ValueError(
                f"Trailing delta must be positive, got {self.trailing_delta}"
            )

        if self.delta_type == DeltaType.PERCENTAGE and self.trailing_delta > Decimal("1"):
            raise ValueError(
                f"Percentage delta must be between 0 and 1, got {self.trailing_delta}"
            )

        if self.side == OrderSide.SELL and self.stop_price >= self.initial_price:
            raise ValueError(
                f"For SELL (long position), stop price ({self.stop_price}) "
                f"must be strictly less than initial price ({self.initial_price})"
            )

        if self.side == OrderSide.BUY and self.stop_price <= self.initial_price:
            raise ValueError(
                f"For BUY (short position), stop price ({self.stop_price}) "
                f"must be strictly greater than initial price ({self.initial_price})"
            )