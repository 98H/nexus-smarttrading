"""Simulation event data structures for tick and bar processing."""

from typing import Optional


class Event:
    """Base class for all simulation events."""

    def __init__(
        self,
        timestamp: int,
        symbol: str = "",
        sequence_id: Optional[int] = 0,
    ) -> None:
        self.timestamp = timestamp
        self.symbol = symbol
        self.sequence_id = sequence_id if sequence_id is not None else 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"timestamp={self.timestamp!r}, "
            f"sequence_id={self.sequence_id!r}, "
            f"symbol={self.symbol!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.sequence_id < other.sequence_id


class TickEvent(Event):
    """Represents a market tick event."""

    def __init__(
        self,
        timestamp: int,
        symbol: str = "",
        price: float = 0.0,
        volume: float = 0.0,
        sequence_id: Optional[int] = 0,
    ) -> None:
        super().__init__(timestamp=timestamp, symbol=symbol, sequence_id=sequence_id)
        self.price = price
        self.volume = volume

    def __repr__(self) -> str:
        return (
            f"TickEvent("
            f"timestamp={self.timestamp!r}, "
            f"sequence_id={self.sequence_id!r}, "
            f"symbol={self.symbol!r}, "
            f"price={self.price!r}, "
            f"volume={self.volume!r})"
        )


class BarEvent(Event):
    """Represents an aggregated OHLCV bar event."""

    def __init__(
        self,
        timestamp: int,
        symbol: str = "",
        open: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        volume: float = 0.0,
        sequence_id: Optional[int] = 0,
    ) -> None:
        super().__init__(timestamp=timestamp, symbol=symbol, sequence_id=sequence_id)
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self) -> str:
        return (
            f"BarEvent("
            f"timestamp={self.timestamp!r}, "
            f"sequence_id={self.sequence_id!r}, "
            f"symbol={self.symbol!r}, "
            f"open={self.open!r}, "
            f"high={self.high!r}, "
            f"low={self.low!r}, "
            f"close={self.close!r}, "
            f"volume={self.volume!r})"
        )