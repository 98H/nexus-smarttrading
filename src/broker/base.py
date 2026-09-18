"""Unified Broker Interface core abstraction."""

from abc import ABC, abstractmethod
from typing import Any, List

from src.broker.models import OrderRequest, OrderResult


class UnifiedBrokerInterface(ABC):
    """Abstract base class defining the Unified Broker Interface contract."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker service."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker service."""
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker and return the execution result."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing active order by its identifier."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[Any]:
        """Retrieve current open positions from the broker."""
        raise NotImplementedError