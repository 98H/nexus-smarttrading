"""Domain models for market data trades, ticks, and aggregated bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

__all__ = ["Bar", "OHLCVBar", "Tick", "Trade"]


@dataclass(frozen=True)
class Trade:
    """Represents an individual trade execution event."""

    timestamp: datetime
    price: float
    volume: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise ValueError(
                f"Trade timestamp must be a datetime instance, got {type(self.timestamp).__name__}"
            )

        try:
            price_val = float(self.price)
            if not math.isfinite(price_val) or price_val <= 0.0:
                raise ValueError(f"Price must be positive and finite, got {self.price}")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Price must be positive and finite, got {self.price}") from err

        try:
            vol_val = float(self.volume)
            if not math.isfinite(vol_val) or vol_val < 0.0:
                raise ValueError(f"Volume must be non-negative and finite, got {self.volume}")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Volume must be non-negative and finite, got {self.volume}") from err

        object.__setattr__(self, "price", price_val)
        object.__setattr__(self, "volume", vol_val)


@dataclass(frozen=True)
class Bar:
    """Represents an OHLCV bar aggregation."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    close_timestamp: datetime | None = None


@dataclass(frozen=True)
class Tick:
    """Represents an individual market tick event for a symbol."""

    symbol: str
    timestamp: datetime
    price: float
    volume: float


@dataclass(frozen=True)
class OHLCVBar:
    """Represents an aggregated OHLCV bar with finalization and correction metadata."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_finalized: bool = False
    is_corrected: bool = False