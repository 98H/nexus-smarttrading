"""Broker domain data models."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class OrderSide(StrEnum):
    """Order side indicating buy or sell intent."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Order type indicating the execution mechanism."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(StrEnum):
    """Order lifecycle status indicating execution state."""

    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


def _coerce_to_decimal(value: Any, field_name: str) -> Decimal:
    """Coerce a numeric or string input to Decimal, preventing invalid operations and booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} cannot be a boolean")
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, (int, float, str)):
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc
    else:
        raise TypeError(
            f"{field_name} must be a Decimal, int, float, or numeric str, got {type(value).__name__}"
        )

    if dec.is_nan() or dec.is_infinite():
        raise ValueError(f"{field_name} must be a finite number, got {dec}")
    return dec


def _validate_quantity(value: Any) -> Decimal:
    """Validate quantity ensuring it is strictly positive and finite."""
    dec = _coerce_to_decimal(value, "quantity")
    if dec <= Decimal("0"):
        raise ValueError(f"quantity must be positive (greater than 0), got {dec}")
    return dec


def _validate_price(value: Any) -> Decimal:
    """Validate price ensuring it is non-negative and finite."""
    dec = _coerce_to_decimal(value, "price")
    if dec < Decimal("0"):
        raise ValueError(f"price must be non-negative (greater than or equal to 0), got {dec}")
    return dec


def _validate_side(value: Any) -> OrderSide:
    """Validate and coerce order side."""
    if isinstance(value, OrderSide):
        return value
    if isinstance(value, str):
        try:
            return OrderSide(value.upper())
        except ValueError as exc:
            valid = [s.value for s in OrderSide]
            raise ValueError(f"Invalid side '{value}'. Must be one of {valid}") from exc
    raise TypeError(f"side must be an OrderSide or str, got {type(value).__name__}")


def _validate_order_type(value: Any) -> OrderType:
    """Validate and coerce order type."""
    if isinstance(value, OrderType):
        return value
    if isinstance(value, str):
        try:
            return OrderType(value.upper())
        except ValueError as exc:
            valid = [t.value for t in OrderType]
            raise ValueError(f"Invalid order_type '{value}'. Must be one of {valid}") from exc
    raise TypeError(f"order_type must be an OrderType or str, got {type(value).__name__}")


def _validate_status(value: Any) -> OrderStatus:
    """Validate and coerce order status."""
    if isinstance(value, OrderStatus):
        return value
    if isinstance(value, str):
        try:
            return OrderStatus(value.upper())
        except ValueError as exc:
            valid = [s.value for s in OrderStatus]
            raise ValueError(f"Invalid status '{value}'. Must be one of {valid}") from exc
    raise TypeError(f"status must be an OrderStatus or str, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Domain model representing an immutable order placement request."""

    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    order_type: OrderType = OrderType.LIMIT

    def __post_init__(self) -> None:
        if not isinstance(self.client_order_id, str):
            raise TypeError(
                f"client_order_id must be a str, got {type(self.client_order_id).__name__}"
            )
        if not self.client_order_id.strip():
            raise ValueError("client_order_id cannot be empty or whitespace")

        if not isinstance(self.symbol, str):
            raise TypeError(f"symbol must be a str, got {type(self.symbol).__name__}")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty or whitespace")

        object.__setattr__(self, "side", _validate_side(self.side))
        object.__setattr__(self, "order_type", _validate_order_type(self.order_type))
        object.__setattr__(self, "quantity", _validate_quantity(self.quantity))
        object.__setattr__(self, "price", _validate_price(self.price))


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Domain model representing the immutable execution or submission result of an order."""

    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    status: OrderStatus

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str):
            raise TypeError(f"order_id must be a str, got {type(self.order_id).__name__}")
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty or whitespace")

        if not isinstance(self.client_order_id, str):
            raise TypeError(
                f"client_order_id must be a str, got {type(self.client_order_id).__name__}"
            )
        if not self.client_order_id.strip():
            raise ValueError("client_order_id cannot be empty or whitespace")

        if not isinstance(self.symbol, str):
            raise TypeError(f"symbol must be a str, got {type(self.symbol).__name__}")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty or whitespace")

        object.__setattr__(self, "side", _validate_side(self.side))
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "quantity", _validate_quantity(self.quantity))
        object.__setattr__(self, "price", _validate_price(self.price))