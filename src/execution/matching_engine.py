from decimal import Decimal

from src.execution.models import Order, OrderSide, OrderStatus, Trade


class MatchingEngine:
    """In-memory simulated order matching engine implementing price-time priority."""

    def __init__(self, symbol: str) -> None:
        if symbol is None:
            raise TypeError("Symbol cannot be None")
        if not isinstance(symbol, str):
            raise TypeError("Symbol must be a string")
        if not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        self._symbol: str = symbol
        self._bids: list[tuple[Order, int]] = []
        self._asks: list[tuple[Order, int]] = []
        self._order_ids: set[str] = set()
        self._sequence: int = 0

    @property
    def symbol(self) -> str:
        """Return the trading symbol managed by this matching engine."""
        return self._symbol

    def get_bids(self) -> list[Order]:
        """Return resting buy orders sorted by price-time priority."""
        return [order for order, _ in self._bids]

    def get_asks(self) -> list[Order]:
        """Return resting sell orders sorted by price-time priority."""
        return [order for order, _ in self._asks]

    @staticmethod
    def _bid_sort_key(item: tuple[Order, int]) -> tuple[Decimal, float, int]:
        order, seq = item
        price = order.price if order.price is not None else Decimal("0")
        ts = float(order.timestamp) if order.timestamp is not None else 0.0
        return (-price, ts, seq)

    @staticmethod
    def _ask_sort_key(item: tuple[Order, int]) -> tuple[Decimal, float, int]:
        order, seq = item
        price = order.price if order.price is not None else Decimal("0")
        ts = float(order.timestamp) if order.timestamp is not None else 0.0
        return (price, ts, seq)

    def submit_order(self, order: Order) -> list[Trade]:
        """Submit an order to the matching engine, executing trades or resting on book."""
        if not isinstance(order, Order):
            raise TypeError("Expected Order instance")
        if order.symbol != self._symbol:
            raise ValueError(
                f"Order symbol '{order.symbol}' does not match engine symbol '{self._symbol}'"
            )
        if order.order_id in self._order_ids:
            raise ValueError(f"Duplicate order ID: {order.order_id}")
        if order.quantity <= Decimal("0"):
            raise ValueError("Order quantity must be positive")
        if order.price is not None and order.price <= Decimal("0"):
            raise ValueError("Order price must be positive")

        self._order_ids.add(order.order_id)
        self._sequence += 1
        current_seq = self._sequence

        if order.remaining_quantity is None:
            order.remaining_quantity = order.quantity
        if order.price is None and order.limit_price is not None:
            order.price = order.limit_price
        if order.limit_price is None and order.price is not None:
            order.limit_price = order.price

        trades: list[Trade] = []

        if order.side == OrderSide.BUY:
            while order.remaining_quantity > Decimal("0") and self._asks:
                best_ask, _ = self._asks[0]
                if (
                    order.price is not None
                    and best_ask.price is not None
                    and order.price < best_ask.price
                ):
                    break

                match_qty = min(order.remaining_quantity, best_ask.remaining_quantity)
                match_price = best_ask.price if best_ask.price is not None else Decimal("0")

                trade = Trade(
                    symbol=self._symbol,
                    price=match_price,
                    quantity=match_qty,
                    maker_order_id=best_ask.order_id,
                    taker_order_id=order.order_id,
                    timestamp=order.timestamp,
                )
                trades.append(trade)

                best_ask.remaining_quantity -= match_qty
                order.remaining_quantity -= match_qty

                if best_ask.remaining_quantity == Decimal("0"):
                    best_ask.status = OrderStatus.FILLED
                    self._asks.pop(0)
                else:
                    best_ask.status = OrderStatus.PARTIALLY_FILLED

                if order.remaining_quantity == Decimal("0"):
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED

            if order.remaining_quantity > Decimal("0"):
                self._bids.append((order, current_seq))
                self._bids.sort(key=self._bid_sort_key)

        elif order.side == OrderSide.SELL:
            while order.remaining_quantity > Decimal("0") and self._bids:
                best_bid, _ = self._bids[0]
                if (
                    order.price is not None
                    and best_bid.price is not None
                    and order.price > best_bid.price
                ):
                    break

                match_qty = min(order.remaining_quantity, best_bid.remaining_quantity)
                match_price = best_bid.price if best_bid.price is not None else Decimal("0")

                trade = Trade(
                    symbol=self._symbol,
                    price=match_price,
                    quantity=match_qty,
                    maker_order_id=best_bid.order_id,
                    taker_order_id=order.order_id,
                    timestamp=order.timestamp,
                )
                trades.append(trade)

                best_bid.remaining_quantity -= match_qty
                order.remaining_quantity -= match_qty

                if best_bid.remaining_quantity == Decimal("0"):
                    best_bid.status = OrderStatus.FILLED
                    self._bids.pop(0)
                else:
                    best_bid.status = OrderStatus.PARTIALLY_FILLED

                if order.remaining_quantity == Decimal("0"):
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED

            if order.remaining_quantity > Decimal("0"):
                self._asks.append((order, current_seq))
                self._asks.sort(key=self._ask_sort_key)

        return trades