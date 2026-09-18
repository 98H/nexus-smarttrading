"""High-DPI Canvas Coordinate Normalizer.

Provides normalization of raw screen event coordinates to logical canvas coordinates,
accounting for device pixel ratio (DPR), viewport origin offsets, and optional boundary clamping.
"""

from typing import Sequence


class CanvasCoordinateNormalizer:
    """Normalizes raw screen coordinates to logical canvas units."""

    def __init__(
        self,
        canvas_width: float,
        canvas_height: float,
        dpr: float = 1.0,
        origin: Sequence[float] = (0.0, 0.0),
    ) -> None:
        """Initialize the coordinate normalizer.

        Args:
            canvas_width: Logical width of the canvas viewport.
            canvas_height: Logical height of the canvas viewport.
            dpr: Device pixel ratio, must be greater than 0.
            origin: (x, y) coordinates of the canvas top-left corner on screen.

        Raises:
            ValueError: If dpr is less than or equal to 0.
        """
        self.canvas_width = float(canvas_width)
        self.canvas_height = float(canvas_height)
        self.origin = (float(origin[0]), float(origin[1]))
        self.dpr = dpr

    @property
    def dpr(self) -> float:
        """Device pixel ratio."""
        return self._dpr

    @dpr.setter
    def dpr(self, value: float) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"Device pixel ratio (DPR) must be greater than 0, got {value}")
        self._dpr = float(value)

    def normalize(
        self,
        x: float,
        y: float,
        dpr: float | None = None,
        clamp: bool = False,
    ) -> tuple[float, float]:
        """Normalize raw screen coordinates to logical canvas coordinates.

        Coordinates are translated relative to the canvas origin and scaled by the DPR.
        Optionally clamped within [0, canvas_width] and [0, canvas_height].

        Args:
            x: Raw screen event x-coordinate.
            y: Raw screen event y-coordinate.
            dpr: Optional DPR override for this normalization call.
            clamp: If True, clamp normalized coordinates to viewport bounds.

        Returns:
            Tuple of (normalized_x, normalized_y) in logical canvas units.

        Raises:
            ValueError: If dpr override is less than or equal to 0.
        """
        if dpr is not None:
            if isinstance(dpr, bool) or dpr <= 0:
                raise ValueError(f"Device pixel ratio (DPR) must be greater than 0, got {dpr}")
            effective_dpr = float(dpr)
        else:
            effective_dpr = self._dpr

        norm_x = (float(x) - self.origin[0]) / effective_dpr
        norm_y = (float(y) - self.origin[1]) / effective_dpr

        if clamp:
            norm_x = max(0.0, min(self.canvas_width, norm_x))
            norm_y = max(0.0, min(self.canvas_height, norm_y))

        return norm_x, norm_y