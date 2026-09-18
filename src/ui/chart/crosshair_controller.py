"""Crosshair controller coordinating viewport mouse events and axis tracking badges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.ui.chart.axis_badges import PriceBadge, TimeBadge


@dataclass(frozen=True)
class Viewport:
    """Defines the 2D plot area boundaries."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})"
            )

    def contains(self, x: float, y: float) -> bool:
        """Returns True if (x, y) is within the viewport boundaries (inclusive)."""
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)


class CrosshairController:
    """Coordinates crosshair events with axis badges and coordinate transformations."""

    def __init__(
        self,
        viewport: Viewport,
        price_badge: PriceBadge,
        time_badge: TimeBadge,
        price_transform: Callable[[float], float],
        time_transform: Callable[[float], float],
    ) -> None:
        self.viewport = viewport
        self.price_badge = price_badge
        self.time_badge = time_badge
        self.price_transform = price_transform
        self.time_transform = time_transform

    def on_mouse_move(self, x: float, y: float) -> None:
        """Handles cursor movement within the chart viewport."""
        if not self.viewport.contains(x, y):
            self.on_mouse_leave()
            return

        price = self.price_transform(y)
        time_val = self.time_transform(x)

        self.price_badge.update(coordinate=y, value=price)
        self.time_badge.update(coordinate=x, value=time_val)

    def on_mouse_leave(self) -> None:
        """Handles cursor leaving the viewport plot area boundary."""
        self.price_badge.hide()
        self.time_badge.hide()