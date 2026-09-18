"""Orthographic Viewport Pan and Zoom Engine."""

from __future__ import annotations

import math

from src.viewport.matrix import Matrix3x3


class OrthoViewportEngine:
    """Manages pan, zoom, and coordinate projections for an orthographic 2D viewport."""

    def __init__(
        self,
        min_zoom: float = 0.001,
        max_zoom: float = 1000.0,
        initial_zoom: float = 1.0,
        initial_offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the viewport engine with zoom bounds and initial state.

        Raises:
            ValueError: If bounds or initial zoom values are invalid.
        """
        if min_zoom <= 0.0:
            raise ValueError(f"min_zoom must be strictly positive, got {min_zoom}")
        if max_zoom <= min_zoom:
            raise ValueError(
                f"max_zoom ({max_zoom}) must be strictly greater than min_zoom ({min_zoom})"
            )
        if not (min_zoom <= initial_zoom <= max_zoom):
            raise ValueError(
                f"initial_zoom ({initial_zoom}) must be within [{min_zoom}, {max_zoom}]"
            )

        self._min_zoom = float(min_zoom)
        self._max_zoom = float(max_zoom)
        self._zoom = float(initial_zoom)
        self._offset_x = float(initial_offset[0])
        self._offset_y = float(initial_offset[1])

    @property
    def zoom(self) -> float:
        """Current zoom level."""
        return self._zoom

    @property
    def offset(self) -> tuple[float, float]:
        """Current viewport pan offset (translation in screen pixels)."""
        return (self._offset_x, self._offset_y)

    @property
    def min_zoom(self) -> float:
        """Minimum allowable zoom level."""
        return self._min_zoom

    @property
    def max_zoom(self) -> float:
        """Maximum allowable zoom level."""
        return self._max_zoom

    @property
    def view_matrix(self) -> Matrix3x3:
        """Current view matrix mapping world coordinates to screen space."""
        return Matrix3x3((
            (self._zoom, 0.0, self._offset_x),
            (0.0, self._zoom, self._offset_y),
            (0.0, 0.0, 1.0),
        ))

    def pan(self, dx: float, dy: float) -> None:
        """Translate the viewport by (dx, dy) screen pixels."""
        self._offset_x += float(dx)
        self._offset_y += float(dy)

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Apply a zoom factor anchored at the screen coordinate (x, y).

        The world coordinate directly underneath (x, y) remains invariant before
        and after zooming. If the target zoom exceeds zoom bounds, it is clamped.

        Raises:
            ValueError: If factor is non-positive or NaN.
        """
        if factor <= 0.0 or math.isnan(factor):
            raise ValueError(f"Zoom factor must be strictly positive, got {factor}")

        target_zoom = self._zoom * float(factor)
        new_zoom = max(self._min_zoom, min(self._max_zoom, target_zoom))

        if new_zoom == self._zoom:
            return

        world_x, world_y = self.screen_to_world(x, y)

        self._zoom = new_zoom
        self._offset_x = float(x) - new_zoom * world_x
        self._offset_y = float(y) - new_zoom * world_y

    def world_to_screen(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Convert world coordinates (world_x, world_y) to screen coordinates."""
        return (
            self._zoom * float(world_x) + self._offset_x,
            self._zoom * float(world_y) + self._offset_y,
        )

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Convert screen coordinates (screen_x, screen_y) to world coordinates."""
        return (
            (float(screen_x) - self._offset_x) / self._zoom,
            (float(screen_y) - self._offset_y) / self._zoom,
        )