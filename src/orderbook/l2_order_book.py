"""
In-memory L2 Order Book backed by self-balancing Red-Black trees.
Bids are maintained descending by price and asks ascending by price.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.orderbook.rb_tree import RedBlackTree


class L2OrderBook:
    """
    Level 2 (aggregated price level) in-memory order book.
    Maintains O(log N) price level operations with top-of-book retrieval in O(log N).
    """

    def __init__(self) -> None:
        self.bids_tree: RedBlackTree = RedBlackTree()
        self.asks_tree: RedBlackTree = RedBlackTree()

    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        """Return the highest bid (price, quantity) or None if empty."""
        return self.bids_tree.find_max()

    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        """Return the lowest ask (price, quantity) or None if empty."""
        return self.asks_tree.find_min()

    def update_bid(self, price: float, quantity: float) -> None:
        """
        Add, update, or delete a bid price level.
        Deletes level if quantity is 0. Raises ValueError on non-positive price
        or negative quantity.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if quantity < 0:
            raise ValueError(f"Quantity cannot be negative, got {quantity}")

        if quantity == 0.0:
            if price in self.bids_tree:
                self.bids_tree.delete(price)
        else:
            self.bids_tree.insert(price, quantity)

    def update_ask(self, price: float, quantity: float) -> None:
        """
        Add, update, or delete an ask price level.
        Deletes level if quantity is 0. Raises ValueError on non-positive price
        or negative quantity.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if quantity < 0:
            raise ValueError(f"Quantity cannot be negative, got {quantity}")

        if quantity == 0.0:
            if price in self.asks_tree:
                self.asks_tree.delete(price)
        else:
            self.asks_tree.insert(price, quantity)

    def get_bid_quantity(self, price: float) -> float:
        """Return the available quantity at a given bid price level, or 0.0 if not present."""
        return self.bids_tree.get(price, 0.0)

    def get_ask_quantity(self, price: float) -> float:
        """Return the available quantity at a given ask price level, or 0.0 if not present."""
        return self.asks_tree.get(price, 0.0)

    def get_bids(self, depth: Optional[int] = None) -> List[Tuple[float, float]]:
        """Return bid price levels sorted descending by price up to requested depth."""
        bids = self.bids_tree.items(reverse=True)
        if depth is not None:
            return bids[:max(0, depth)]
        return bids

    def get_asks(self, depth: Optional[int] = None) -> List[Tuple[float, float]]:
        """Return ask price levels sorted ascending by price up to requested depth."""
        asks = self.asks_tree.items(reverse=False)
        if depth is not None:
            return asks[:max(0, depth)]
        return asks

    def clear(self) -> None:
        """Remove all price levels from both sides of the book."""
        self.bids_tree.clear()
        self.asks_tree.clear()