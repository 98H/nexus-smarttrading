"""Dynamic server-side trailing stop processor."""

from datetime import datetime
from decimal import Decimal
from typing import Callable, List, Optional

from src.trading.orders import DeltaType, OrderSide, OrderStatus, TrailingStopOrder


class TrailingStopProcessor:
    """Manages active trailing stop orders and processes market price ticks."""

    def __init__(self) -> None:
        self._orders: dict[str, TrailingStopOrder] = {}
        self._trigger_callback: Optional[
            Callable[[TrailingStopOrder, Decimal], None]
        ] = None

    def register_order(self, order: TrailingStopOrder) -> None:
        """Register a new trailing stop order with the processor."""
        if order.order_id in self._orders:
            raise ValueError(f"Order '{order.order_id}' is already registered.")
        self._orders[order.order_id] = order

    def get_order(self, order_id: str) -> TrailingStopOrder:
        """Retrieve an order by its identifier."""
        if order_id not in self._orders:
            raise KeyError(f"Order '{order_id}' not found.")
        return self._orders[order_id]

    def cancel_order(self, order_id: str) -> TrailingStopOrder:
        """Cancel an existing order and remove it from active evaluation."""
        order = self.get_order(order_id)
        order.status = OrderStatus.CANCELLED
        return order

    def get_active_orders(
        self, symbol: Optional[str] = None
    ) -> List[TrailingStopOrder]:
        """Retrieve all currently active trailing stop orders, optionally filtered by symbol."""
        return [
            order
            for order in self._orders.values()
            if order.status == OrderStatus.ACTIVE
            and (symbol is None or order.symbol == symbol)
        ]

    def set_trigger_callback(
        self, callback: Callable[[TrailingStopOrder, Decimal], None]
    ) -> None:
        """Register a notification callback invoked when any trailing stop triggers."""
        self._trigger_callback = callback

    def process_tick(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> List[TrailingStopOrder]:
        """
        Process a market price tick for a specific symbol.

        Evaluates activation gating, updates trailing stop ratchets,
        and triggers stop breaches.
        """
        if price <= Decimal("0"):
            raise ValueError(f"Market tick price must be positive, got {price}")

        triggered_orders: List[TrailingStopOrder] = []

        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue

            if order.status not in (OrderStatus.ACTIVE, OrderStatus.PENDING):
                continue

            # Check activation gating for PENDING orders
            if order.status == OrderStatus.PENDING:
                if order.activation_price is not None:
                    if order.side == OrderSide.SELL and price >= order.activation_price:
                        order.status = OrderStatus.ACTIVE
                    elif order.side == OrderSide.BUY and price <= order.activation_price:
                        order.status = OrderStatus.ACTIVE
                    else:
                        continue
                else:
                    order.status = OrderStatus.ACTIVE

            # Evaluate stop breach or ratchet stop price
            if order.side == OrderSide.SELL:
                if price <= order.stop_price:
                    self._trigger_order(order, price, triggered_orders)
                else:
                    if order.highest_price is None or price > order.highest_price:
                        order.highest_price = price

                    if order.delta_type == DeltaType.AMOUNT:
                        candidate_stop = order.highest_price - order.trailing_delta
                    else:
                        candidate_stop = order.highest_price * (
                            Decimal("1") - order.trailing_delta
                        )

                    if candidate_stop > order.stop_price:
                        order.stop_price = candidate_stop

            elif order.side == OrderSide.BUY:
                if price >= order.stop_price:
                    self._trigger_order(order, price, triggered_orders)
                else:
                    if order.lowest_price is None or price < order.lowest_price:
                        order.lowest_price = price

                    if order.delta_type == DeltaType.AMOUNT:
                        candidate_stop = order.lowest_price + order.trailing_delta
                    else:
                        candidate_stop = order.lowest_price * (
                            Decimal("1") + order.trailing_delta
                        )

                    if candidate_stop < order.stop_price:
                        order.stop_price = candidate_stop

        return triggered_orders

    def _trigger_order(
        self,
        order: TrailingStopOrder,
        price: Decimal,
        triggered_list: List[TrailingStopOrder],
    ) -> None:
        """Mark order as triggered and execute the registered callback."""
        order.status = OrderStatus.TRIGGERED
        order.trigger_price = price
        triggered_list.append(order)

        if self._trigger_callback is not None:
            self._trigger_callback(order, price)