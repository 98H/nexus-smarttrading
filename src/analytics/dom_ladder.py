"""Real-Time Visual Depth of Market (DOM) Ladder analytics."""

from typing import Any, List, Tuple


class DomLadder(dict):
    """Structured DOM ladder containing sorted bids and asks."""

    def __init__(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        super().__init__(bids=bids, asks=asks)

    @property
    def bids(self) -> List[Tuple[float, float]]:
        """Return the sorted bid levels."""
        return self["bids"]

    @property
    def asks(self) -> List[Tuple[float, float]]:
        """Return the sorted ask levels."""
        return self["asks"]


def _parse_level(level: Any) -> Tuple[float, float]:
    """Parse an order book level into a (price, volume) tuple."""
    if isinstance(level, (tuple, list)) and len(level) >= 2:
        return (float(level[0]), float(level[1]))
    if hasattr(level, "price") and hasattr(level, "volume"):
        return (float(level.price), float(level.volume))
    if isinstance(level, dict) and "price" in level and "volume" in level:
        return (float(level["price"]), float(level["volume"]))
    raise TypeError(f"Unexpected price level format: {type(level)}")


def generate_dom_ladder(order_book: Any, depth: int) -> DomLadder:
    """Generate a structured DOM ladder limited to a specified depth limit.

    Args:
        order_book: Order book state containing bids and asks.
        depth: Maximum number of price levels to include for bids and asks.

    Returns:
        DomLadder containing sorted bids (descending) and asks (ascending).

    Raises:
        TypeError: If depth is not an integer or order_book format is invalid.
        ValueError: If depth is negative.
    """
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError(f"Depth must be an integer, got {type(depth).__name__}")
    if depth < 0:
        raise ValueError(f"Depth must be non-negative, got {depth}")

    if hasattr(order_book, "bids") and hasattr(order_book, "asks"):
        raw_bids = order_book.bids
        raw_asks = order_book.asks
    elif isinstance(order_book, dict):
        raw_bids = order_book.get("bids", [])
        raw_asks = order_book.get("asks", [])
    else:
        raise TypeError(f"Invalid order book format: {type(order_book)}")

    parsed_bids = [_parse_level(level) for level in (raw_bids or [])]
    parsed_asks = [_parse_level(level) for level in (raw_asks or [])]

    # Bids sorted descending (highest price first)
    sorted_bids = sorted(parsed_bids, key=lambda lvl: lvl[0], reverse=True)[:depth]
    # Asks sorted ascending (lowest price first)
    sorted_asks = sorted(parsed_asks, key=lambda lvl: lvl[0])[:depth]

    return DomLadder(bids=sorted_bids, asks=sorted_asks)