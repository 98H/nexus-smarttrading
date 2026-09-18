"""Axis tracking badges for displaying price and time coordinates on chart axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BoundingBox:
    """Represents a 1D span along an axis."""

    start: float
    end: float


class AxisBadge:
    """Base tracking badge for chart axes."""

    def __init__(
        self,
        axis_min: float,
        axis_max: float,
        size: float,
        formatter: Optional[Callable[[float], str]] = None,
        dimension_name: str = "size",
    ) -> None:
        if axis_min >= axis_max:
            raise ValueError(
                f"axis_min ({axis_min}) must be strictly less than axis_max ({axis_max})"
            )
        if size <= 0:
            raise ValueError(f"{dimension_name} must be strictly positive, got {size}")
        if size > (axis_max - axis_min):
            raise ValueError(
                f"{dimension_name} ({size}) cannot exceed axis span ({axis_max - axis_min})"
            )

        self.axis_min = float(axis_min)
        self.axis_max = float(axis_max)
        self._size = float(size)
        self.formatter = formatter

        self.visible: bool = False
        self.text: str = ""
        self.coordinate: float = 0.0

    @property
    def bounding_box(self) -> BoundingBox:
        """Calculates the bounding box clamped within axis boundaries."""
        half_size = self._size / 2.0
        clamped_center = max(
            self.axis_min + half_size,
            min(self.axis_max - half_size, self.coordinate),
        )
        return BoundingBox(
            start=clamped_center - half_size,
            end=clamped_center + half_size,
        )

    def update(self, coordinate: float, value: float) -> None:
        """Updates badge coordinate, formatted text, and sets visibility to True."""
        self.coordinate = float(coordinate)
        if self.formatter is not None:
            self.text = self.formatter(value)
        else:
            self.text = str(value)
        self.visible = True

    def hide(self) -> None:
        """Hides the tracking badge."""
        self.visible = False

    def show(self) -> None:
        """Shows the tracking badge."""
        self.visible = True


class PriceBadge(AxisBadge):
    """Price tracking badge displayed on the vertical (Y) price axis."""

    def __init__(
        self,
        axis_min: float,
        axis_max: float,
        height: float,
        formatter: Optional[Callable[[float], str]] = None,
    ) -> None:
        super().__init__(
            axis_min=axis_min,
            axis_max=axis_max,
            size=height,
            formatter=formatter,
            dimension_name="height",
        )

    @property
    def height(self) -> float:
        """Height of the price badge along the price axis."""
        return self._size


class TimeBadge(AxisBadge):
    """Time tracking badge displayed on the horizontal (X) time axis."""

    def __init__(
        self,
        axis_min: float,
        axis_max: float,
        width: float,
        formatter: Optional[Callable[[float], str]] = None,
    ) -> None:
        super().__init__(
            axis_min=axis_min,
            axis_max=axis_max,
            size=width,
            formatter=formatter,
            dimension_name="width",
        )

    @property
    def width(self) -> float:
        """Width of the time badge along the time axis."""
        return self._size