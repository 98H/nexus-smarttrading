"""Date and price range measurement primitive."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CoordinatePoint:
    """A point in date-price coordinate space."""

    timestamp: datetime
    price: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                f"timestamp must be a datetime instance, got {type(self.timestamp).__name__}"
            )
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)):
            raise TypeError(
                f"price must be a numeric type (int or float), got {type(self.price).__name__}"
            )
        object.__setattr__(self, "price", float(self.price))


@dataclass(frozen=True)
class DatePriceRangeMeasurement:
    """Measures time duration and price delta between two coordinate points."""

    start: CoordinatePoint
    end: CoordinatePoint

    def __post_init__(self) -> None:
        if not isinstance(self.start, CoordinatePoint):
            raise TypeError(
                f"start must be a CoordinatePoint instance, got {type(self.start).__name__}"
            )
        if not isinstance(self.end, CoordinatePoint):
            raise TypeError(
                f"end must be a CoordinatePoint instance, got {type(self.end).__name__}"
            )

    @property
    def duration(self) -> timedelta:
        """Return the time duration between start and end timestamps."""
        return self.end.timestamp - self.start.timestamp

    @property
    def price_delta(self) -> float:
        """Return the price difference between end and start prices."""
        return self.end.price - self.start.price

    @property
    def percentage_change(self) -> float:
        """Return the percentage change from start price to end price.

        Raises:
            ValueError: If start price is 0.0, indicating division by zero is invalid.
        """
        if self.start.price == 0.0:
            raise ValueError("division by zero is invalid when start price is 0.0")
        return (self.price_delta / abs(self.start.price)) * 100.0


RangeMeasurement = DatePriceRangeMeasurement

__all__ = [
    "CoordinatePoint",
    "DatePriceRangeMeasurement",
    "RangeMeasurement",
]