"""Bracket Order (Take-Profit / Stop-Loss) One-Cancels-the-Other (OCO) Manager."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol


class OrderStatus(str, Enum):
    """Execution status of an individual order leg."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BracketStatus(str, Enum):
    """Lifecycle status of a bracket order."""

    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


@dataclass
class ExecutionReport:
    """Execution report received for an order leg."""

    order_id: str
    status: OrderStatus


@dataclass
class BracketOrder:
    """Represents a bracket order linking take-profit and stop-loss legs."""

    bracket_id: str
    tp_order_id: str
    sl_order_id: str
    status: BracketStatus = BracketStatus.ACTIVE

    def get_opposite_leg(self, order_id: str) -> str:
        """Return the opposite order ID for a given leg in this bracket."""
        if order_id == self.tp_order_id:
            return self.sl_order_id
        if order_id == self.sl_order_id:
            return self.tp_order_id
        raise ValueError(
            f"Order ID '{order_id}' does not belong to bracket '{self.bracket_id}'."
        )


class OrderGateway(Protocol):
    """Protocol defining the external order gateway cancellation interface."""

    def cancel_order(self, order_id: str) -> bool:
        """Issue a cancellation request for an order."""
        ...


class OCOManager:
    """Manages bracket orders and enforces One-Cancels-the-Other logic."""

    def __init__(self, order_gateway: Any) -> None:
        self.order_gateway = order_gateway
        self._brackets: Dict[str, BracketOrder] = {}
        self._order_to_bracket: Dict[str, str] = {}

    def register_bracket(
        self, bracket_id: str, tp_order_id: str, sl_order_id: str
    ) -> BracketOrder:
        """Register a new bracket order with active take-profit and stop-loss legs."""
        if tp_order_id == sl_order_id:
            raise ValueError(
                "Take-profit and stop-loss order IDs cannot be identical."
            )

        if bracket_id in self._brackets:
            raise ValueError(f"Bracket ID '{bracket_id}' already exists.")

        for order_id in (tp_order_id, sl_order_id):
            if order_id in self._order_to_bracket:
                existing_bracket = self._brackets[self._order_to_bracket[order_id]]
                if existing_bracket.status == BracketStatus.ACTIVE:
                    raise ValueError(
                        f"Order ID '{order_id}' is already in use by active bracket "
                        f"'{existing_bracket.bracket_id}'."
                    )

        bracket = BracketOrder(
            bracket_id=bracket_id,
            tp_order_id=tp_order_id,
            sl_order_id=sl_order_id,
            status=BracketStatus.ACTIVE,
        )
        self._brackets[bracket_id] = bracket
        self._order_to_bracket[tp_order_id] = bracket_id
        self._order_to_bracket[sl_order_id] = bracket_id
        return bracket

    def get_bracket(self, bracket_id: str) -> Optional[BracketOrder]:
        """Retrieve a bracket order by its identifier."""
        return self._brackets.get(bracket_id)

    def handle_execution_report(self, report: ExecutionReport) -> None:
        """Handle incoming execution reports and apply OCO cancellation rules."""
        if report.order_id not in self._order_to_bracket:
            raise KeyError(f"Order ID '{report.order_id}' is not tracked.")

        bracket_id = self._order_to_bracket[report.order_id]
        bracket = self._brackets[bracket_id]

        if bracket.status == BracketStatus.CLOSED:
            return

        if report.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            opposite_order_id = bracket.get_opposite_leg(report.order_id)
            bracket.status = BracketStatus.CLOSED
            self.order_gateway.cancel_order(opposite_order_id)

    def cancel_bracket(self, bracket_id: str) -> None:
        """Manually cancel an active bracket order by cancelling both legs."""
        if bracket_id not in self._brackets:
            raise KeyError(f"Bracket ID '{bracket_id}' not found.")

        bracket = self._brackets[bracket_id]
        if bracket.status == BracketStatus.CLOSED:
            return

        bracket.status = BracketStatus.CLOSED
        self.order_gateway.cancel_order(bracket.tp_order_id)
        self.order_gateway.cancel_order(bracket.sl_order_id)