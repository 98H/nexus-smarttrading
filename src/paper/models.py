"""Data models for paper trading portfolio tracker."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class PositionSide(str, Enum):
    """Side of an open position."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeSide(str, Enum):
    """Side of an executed trade."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """Represents an active paper trading position."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if not isinstance(self.current_price, Decimal):
            self.current_price = Decimal(str(self.current_price))

        if self.quantity <= Decimal("0"):
            raise ValueError("Position quantity must be positive")
        if self.entry_price <= Decimal("0"):
            raise ValueError("Position entry price must be positive")
        if self.current_price < Decimal("0"):
            raise ValueError("Position current price cannot be negative")

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate mark-to-market unrealized profit and loss."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        return (self.entry_price - self.current_price) * self.quantity

    @property
    def notional_value(self) -> Decimal:
        """Calculate absolute current notional value of the position."""
        return self.quantity * self.current_price


@dataclass
class Trade:
    """Represents an executed order."""

    symbol: str
    side: TradeSide
    quantity: Decimal
    price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))

        if self.quantity <= Decimal("0"):
            raise ValueError("Trade quantity must be positive")
        if self.price <= Decimal("0"):
            raise ValueError("Trade price must be positive")