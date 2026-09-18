"""Execution cost emulation models for spread, slippage, and commission calculations."""

from enum import Enum
from typing import Union


class OrderSide(str, Enum):
    """Execution side for trade orders."""

    BUY = "BUY"
    SELL = "SELL"


def _normalize_side(side: Union[OrderSide, str]) -> OrderSide:
    """Validate and normalize order side input to OrderSide enum.

    Args:
        side: Order side enum instance or string (case-insensitive).

    Returns:
        OrderSide enum value.

    Raises:
        ValueError: If side is not a valid order side or side type.
    """
    if isinstance(side, OrderSide):
        return side
    if isinstance(side, str):
        try:
            return OrderSide(side.upper())
        except ValueError:
            raise ValueError(f"Invalid order side: '{side}'. Expected 'BUY' or 'SELL'.")
    raise ValueError(f"Invalid order side type: {type(side).__name__}")


class SpreadModel:
    """Emulates bid-ask spread impact on trade executions."""

    def __init__(self, spread: float) -> None:
        """Initialize spread model.

        Args:
            spread: Total bid-ask spread amount (must be non-negative).

        Raises:
            ValueError: If spread is negative.
        """
        if spread < 0.0:
            raise ValueError(f"Spread must be non-negative, got {spread}")
        self.spread = float(spread)

    def calculate_fill_price(
        self, mid_price: float, side: Union[OrderSide, str]
    ) -> float:
        """Calculate execution fill price adjusted for the bid-ask spread.

        Buy orders cross the ask at mid + (spread / 2).
        Sell orders cross the bid at mid - (spread / 2).

        Args:
            mid_price: Mid-market reference price (must be positive).
            side: Order execution side (BUY or SELL).

        Returns:
            Spread-adjusted execution fill price.

        Raises:
            ValueError: If mid_price is non-positive or side is invalid.
        """
        if mid_price <= 0.0:
            raise ValueError(f"Mid price must be positive, got {mid_price}")

        order_side = _normalize_side(side)
        half_spread = self.spread / 2.0

        if order_side == OrderSide.BUY:
            return mid_price + half_spread
        return mid_price - half_spread


class SlippageModel:
    """Emulates market impact slippage proportional to trade volume participation."""

    def __init__(self, impact_rate: float) -> None:
        """Initialize slippage model.

        Args:
            impact_rate: Proportional price impact rate (must be non-negative).

        Raises:
            ValueError: If impact_rate is negative.
        """
        if impact_rate < 0.0:
            raise ValueError(f"Impact rate must be non-negative, got {impact_rate}")
        self.impact_rate = float(impact_rate)

    def calculate_price_impact(
        self, order_size: float, market_volume: float
    ) -> float:
        """Calculate proportional price impact fraction.

        Args:
            order_size: Order quantity or size executed (must be non-negative).
            market_volume: Total reference market volume (must be positive).

        Returns:
            Fractional price impact (impact_rate * (order_size / market_volume)).

        Raises:
            ValueError: If order_size is negative or market_volume is non-positive.
        """
        if order_size < 0.0:
            raise ValueError(f"Order size must be non-negative, got {order_size}")
        if market_volume <= 0.0:
            raise ValueError(f"Market volume must be positive, got {market_volume}")

        volume_fraction = order_size / market_volume
        return self.impact_rate * volume_fraction

    def calculate_fill_price(
        self,
        base_price: float,
        order_size: float,
        market_volume: float,
        side: Union[OrderSide, str],
    ) -> float:
        """Calculate execution fill price adjusted for market impact slippage.

        Buy orders shift upward: base_price * (1 + impact).
        Sell orders shift downward: base_price * (1 - impact).

        Args:
            base_price: Reference execution price (must be positive).
            order_size: Order quantity or size executed (must be non-negative).
            market_volume: Reference market volume (must be positive).
            side: Order execution side (BUY or SELL).

        Returns:
            Slippage-adjusted execution fill price.

        Raises:
            ValueError: If base_price <= 0, order_size < 0, market_volume <= 0, or side invalid.
        """
        if base_price <= 0.0:
            raise ValueError(f"Base price must be positive, got {base_price}")

        order_side = _normalize_side(side)
        impact = self.calculate_price_impact(
            order_size=order_size, market_volume=market_volume
        )

        if order_side == OrderSide.BUY:
            return base_price * (1.0 + impact)
        return base_price * (1.0 - impact)


class CommissionModel:
    """Emulates trading commission fees with percentage rate and minimum fee floors."""

    def __init__(self, percentage_rate: float, min_fee: float = 0.0) -> None:
        """Initialize commission model.

        Args:
            percentage_rate: Percentage rate applied to notional value (must be non-negative).
            min_fee: Minimum commission fee charged per order (must be non-negative).

        Raises:
            ValueError: If percentage_rate or min_fee is negative.
        """
        if percentage_rate < 0.0:
            raise ValueError(
                f"Percentage rate must be non-negative, got {percentage_rate}"
            )
        if min_fee < 0.0:
            raise ValueError(f"Minimum fee must be non-negative, got {min_fee}")

        self.percentage_rate = float(percentage_rate)
        self.min_fee = float(min_fee)

    def calculate_commission(self, notional_value: float) -> float:
        """Compute transaction commission cost.

        Args:
            notional_value: Executed trade notional value (must be non-negative).

        Returns:
            Commission fee equal to max(notional_value * percentage_rate, min_fee).

        Raises:
            ValueError: If notional_value is negative.
        """
        if notional_value < 0.0:
            raise ValueError(
                f"Notional value must be non-negative, got {notional_value}"
            )

        proportional_fee = notional_value * self.percentage_rate
        return max(proportional_fee, self.min_fee)


__all__ = [
    "CommissionModel",
    "OrderSide",
    "SlippageModel",
    "SpreadModel",
]