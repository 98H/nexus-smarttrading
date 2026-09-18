"""High-precision crosshair overlay layer for sub-pixel canvas rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BoundaryPolicy(str, Enum):
    """Policy for handling coordinates outside the viewport boundaries."""

    CLIP = "CLIP"
    EXCLUDE = "EXCLUDE"


@dataclass(frozen=True)
class Viewport:
    """Represents the rectangular bounding box of a canvas viewport."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})."
            )


@dataclass(frozen=True)
class Point:
    """Represents a 2D point with floating-point precision."""

    x: float
    y: float


@dataclass(frozen=True)
class LineSegment:
    """Represents a directed line segment between two 2D points."""

    start: Point
    end: Point


@dataclass(frozen=True)
class CrosshairGeometry:
    """Represents the horizontal and vertical guide segments of a crosshair."""

    horizontal: LineSegment
    vertical: LineSegment


class HighPrecisionCrosshairOverlay:
    """Calculates and renders sub-pixel precision crosshair guides over a viewport."""

    def __init__(
        self,
        viewport: Viewport,
        boundary_policy: BoundaryPolicy = BoundaryPolicy.CLIP,
        visible: bool = True,
        enabled: bool = True,
    ) -> None:
        self.viewport: Viewport = viewport
        self.boundary_policy: BoundaryPolicy = boundary_policy
        self.visible: bool = visible
        self.enabled: bool = enabled
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def set_position(self, x: float, y: float) -> None:
        """Update the crosshair position with sub-pixel floating-point coordinates."""
        self._x = float(x)
        self._y = float(y)

    def calculate_geometry(self) -> Optional[CrosshairGeometry]:
        """Compute the horizontal and vertical line segments within the viewport.

        Raises:
            ValueError: If `set_position` has not been called prior to calculation.

        Returns:
            A CrosshairGeometry instance containing horizontal and vertical line segments,
            or None if the coordinates are excluded by boundary policy.
        """
        if self._x is None or self._y is None:
            raise ValueError("Crosshair position has not been set.")

        x = self._x
        y = self._y

        if self.boundary_policy == BoundaryPolicy.EXCLUDE:
            if (
                x < self.viewport.x_min
                or x > self.viewport.x_max
                or y < self.viewport.y_min
                or y > self.viewport.y_max
            ):
                return None
            target_x = x
            target_y = y
        elif self.boundary_policy == BoundaryPolicy.CLIP:
            target_x = min(max(x, self.viewport.x_min), self.viewport.x_max)
            target_y = min(max(y, self.viewport.y_min), self.viewport.y_max)
        else:
            raise ValueError(f"Unsupported boundary policy: {self.boundary_policy}")

        horizontal = LineSegment(
            start=Point(x=self.viewport.x_min, y=target_y),
            end=Point(x=self.viewport.x_max, y=target_y),
        )
        vertical = LineSegment(
            start=Point(x=target_x, y=self.viewport.y_min),
            end=Point(x=target_x, y=self.viewport.y_max),
        )

        return CrosshairGeometry(horizontal=horizontal, vertical=vertical)

    def get_render_primitives(self) -> list[LineSegment]:
        """Return draw commands/primitives for the overlay.

        Returns an empty list when hidden, disabled, or excluded by boundary policy.
        """
        if not self.visible or not self.enabled:
            return []

        geometry = self.calculate_geometry()
        if geometry is None:
            return []

        return [geometry.horizontal, geometry.vertical]


__all__ = [
    "BoundaryPolicy",
    "CrosshairGeometry",
    "HighPrecisionCrosshairOverlay",
    "LineSegment",
    "Point",
    "Viewport",
]