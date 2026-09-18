from decimal import Decimal
from typing import Optional

from src.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PriceTick,
)


class SimulatedExecutionPipeline:
    """Simulated execution pipeline processing market, limit, stop, and trailing orders."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    def submit_order(self, order: Order) -> None:
        """Submit an order to the pipeline."""
        if order.order_id in self._orders:
            raise ValueError(f"Order with ID '{order.order_id}' already exists.")
        self._orders[order.order_id] = order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[Order]:
        """Return all open (PENDING) orders."""
        return [order for order in self._orders.values() if order.status == OrderStatus.PENDING]

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order."""
        if order_id not in self._orders:
            raise KeyError(f"Order with ID '{order_id}' not found.")
        order = self._orders[order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot cancel order '{order_id}' with status {order.status}.")
        order.status = OrderStatus.CANCELLED
        return order

    def process_tick(self, tick: PriceTick) -> list[ExecutionReport]:
        """Process an incoming price tick against all pending orders for the matching symbol."""
        reports: list[ExecutionReport] = []

        for order in list(self._orders.values()):
            if order.status != OrderStatus.PENDING or order.symbol != tick.symbol:
                continue

            if self._should_execute(order, tick):
                report = self._execute_order(order, tick)
                reports.append(report)

        return reports

    def _should_execute(self, order: Order, tick: PriceTick) -> bool:
        """Evaluate whether an order's execution or trigger condition is met."""
        if order.order_type == OrderType.MARKET:
            return True
        elif order.order_type == OrderType.LIMIT:
            return self._evaluate_limit(order, tick.price)
        elif order.order_type == OrderType.STOP:
            return self._evaluate_stop(order, tick.price)
        elif order.order_type == OrderType.TRAILING_STOP:
            return self._evaluate_trailing_stop(order, tick.price)
        return False

    def _evaluate_limit(self, order: Order, price: Decimal) -> bool:
        """Evaluate limit threshold crossing."""
        if order.limit_price is None:
            return False
        if order.side == OrderSide.BUY:
            return price <= order.limit_price
        elif order.side == OrderSide.SELL:
            return price >= order.limit_price
        return False

    def _evaluate_stop(self, order: Order, price: Decimal) -> bool:
        """Evaluate stop threshold breach."""
        if order.stop_price is None:
            return False
        if order.side == OrderSide.BUY:
            return price >= order.stop_price
        elif order.side == OrderSide.SELL:
            return price <= order.stop_price
        return False

    def _evaluate_trailing_stop(self, order: Order, price: Decimal) -> bool:
        """
        Adjust watermark level on favorable movement or trigger upon adverse breach.
        """
        delta = order.trailing_delta or Decimal("0")

        if order.watermark is None:
            order.watermark = price
            return False

        if order.side == OrderSide.SELL:
            if price > order.watermark:
                order.watermark = price
                return False
            trigger_price = order.watermark - delta
            return price <= trigger_price

        elif order.side == OrderSide.BUY:
            if price < order.watermark:
                order.watermark = price
                return False
            trigger_price = order.watermark + delta
            return price >= trigger_price

        return False

    def _execute_order(self, order: Order, tick: PriceTick) -> ExecutionReport:
        """Mark order as FILLED and create its execution report."""
        order.status = OrderStatus.FILLED
        order.filled_price = tick.price
        order.filled_quantity = order.quantity

        return ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            execution_price=tick.price,
            filled_quantity=order.quantity,
            timestamp=tick.timestamp,
        )