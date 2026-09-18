"""Magnetic Snap-to-OHLC Algorithm module.

Provides data structures and algorithms for magnetically snapping cursor positions
to the nearest OHLC component (Open, High, Low, Close) of a candlestick within a
configured threshold radius.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

# Deterministic tie-breaking priority: extremes ('H', 'L') take precedence over
# intermediates ('O', 'C').
COMPONENT_PRIORITY: dict[str, int] = {
    "H": 0,
    "L": 1,
    "O": 2,
    "C": 3,
}


@dataclass(frozen=True)
class Point:
    """Immutable 2D coordinate representing a position on a chart or canvas."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))


@dataclass(frozen=True)
class Candlestick:
    """Immutable candlestick data structure representing an OHLC interval."""

    open: float
    high: float
    low: float
    close: float
    x: float = 0.0

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.x)
        if not all(math.isfinite(val) for val in values):
            raise ValueError("Candlestick values must be finite numbers")

        object.__setattr__(self, "open", float(self.open))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "close", float(self.close))
        object.__setattr__(self, "x", float(self.x))

        if self.high < self.low:
            raise ValueError("High must be greater than or equal to Low")
        if self.high < self.open:
            raise ValueError("High must be greater than or equal to Open")
        if self.high < self.close:
            raise ValueError("High must be greater than or equal to Close")
        if self.low > self.open:
            raise ValueError("Low must be less than or equal to Open")
        if self.low > self.close:
            raise ValueError("Low must be less than or equal to Close")


@dataclass(frozen=True)
class SnapResult:
    """Immutable evaluation result of a magnetic snap operation."""

    snapped: bool
    coordinate: Point
    price: float | None = None
    component: str | None = None


def snap_to_ohlc(
    candlestick: Candlestick,
    cursor: Point,
    snap_radius: float,
    price_to_y: Callable[[float], float] | None = None,
) -> SnapResult:
    """Evaluate cursor proximity to candlestick OHLC levels and snap if within radius.

    Args:
        candlestick: Candlestick containing OHLC values and horizontal coordinate.
        cursor: Current cursor coordinates to evaluate.
        snap_radius: Maximum Euclidean distance allowed for snapping.
        price_to_y: Optional projection callable mapping price to canvas Y coordinate.

    Returns:
        SnapResult containing snap status, target coordinate, price, and component.

    Raises:
        ValueError: If snap_radius is non-positive, or if coordinates are non-finite.
    """
    if not math.isfinite(snap_radius) or snap_radius <= 0:
        raise ValueError(
            f"snap_radius must be a strictly positive finite number, got {snap_radius}"
        )

    if not (math.isfinite(cursor.x) and math.isfinite(cursor.y)):
        raise ValueError(
            f"Cursor coordinates must be finite numbers, got ({cursor.x}, {cursor.y})"
        )

    ohlc_levels = [
        ("H", candlestick.high),
        ("L", candlestick.low),
        ("O", candlestick.open),
        ("C", candlestick.close),
    ]

    best_candidate: tuple[float, int, str, float, Point] | None = None

    for comp, price in ohlc_levels:
        target_y = price_to_y(price) if price_to_y is not None else price
        if not math.isfinite(target_y):
            raise ValueError(f"Projected coordinate must be finite, got {target_y}")

        target_coord = Point(candlestick.x, target_y)
        dist = math.hypot(target_coord.x - cursor.x, target_coord.y - cursor.y)

        if dist <= snap_radius:
            priority = COMPONENT_PRIORITY[comp]
            candidate_key = (dist, priority)
            if best_candidate is None or candidate_key < (
                best_candidate[0],
                best_candidate[1],
            ):
                best_candidate = (dist, priority, comp, price, target_coord)

    if best_candidate is None:
        return SnapResult(
            snapped=False,
            coordinate=cursor,
            price=None,
            component=None,
        )

    _, _, best_comp, best_price, best_coord = best_candidate
    return SnapResult(
        snapped=True,
        coordinate=best_coord,
        price=best_price,
        component=best_comp,
    )


class MagneticSnap:
    """Configurable magnetic snap algorithm for OHLC charts."""

    def __init__(
        self,
        snap_radius: float,
        price_to_y: Callable[[float], float] | None = None,
    ) -> None:
        if not math.isfinite(snap_radius) or snap_radius <= 0:
            raise ValueError(
                f"snap_radius must be a strictly positive finite number, got {snap_radius}"
            )
        self._snap_radius = float(snap_radius)
        self._price_to_y = price_to_y

    @property
    def snap_radius(self) -> float:
        """The configured snap threshold radius."""
        return self._snap_radius

    @property
    def price_to_y(self) -> Callable[[float], float] | None:
        """Optional custom projection function mapping price to coordinate Y."""
        return self._price_to_y

    def snap(self, candlestick: Candlestick, cursor: Point) -> SnapResult:
        """Snap the cursor against a given candlestick using configured settings."""
        return snap_to_ohlc(
            candlestick=candlestick,
            cursor=cursor,
            snap_radius=self._snap_radius,
            price_to_y=self._price_to_y,
        )