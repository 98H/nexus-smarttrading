"""Paths module for recording freehand brush strokes and highlighting paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class BlendMode(str, Enum):
    """Supported rendering blend modes."""

    NORMAL = "normal"
    OPAQUE = "opaque"
    MULTIPLY = "multiply"


@dataclass(frozen=True)
class Point:
    """Represents a 2D coordinate point."""

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "x", float(self.x))
            object.__setattr__(self, "y", float(self.y))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Point coordinates must be numeric: {exc}") from exc

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError(f"Point index out of range: {index}")

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self) -> int:
        return 2

    @classmethod
    def from_raw(cls, value: Any) -> Point:
        """Parse raw coordinate data into a Point instance."""
        if isinstance(value, (str, bytes)):
            raise ValueError(f"Invalid coordinate format: {value!r}")
        if isinstance(value, cls):
            return value
        if hasattr(value, "x") and hasattr(value, "y"):
            if hasattr(value, "z"):
                raise ValueError(f"3D coordinates are not supported: {value!r}")
            return cls(float(value.x), float(value.y))
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"Coordinate must be 2D, got {len(value)} dimensions: {value!r}")
            return cls(float(value[0]), float(value[1]))
        raise ValueError(f"Cannot convert {type(value).__name__} to Point: {value!r}")


@dataclass
class Path:
    """Represents a recorded vector path with visual and compositing attributes."""

    points: list[Point]
    color: str
    stroke_width: float
    blend_mode: BlendMode = BlendMode.NORMAL
    alpha: float = 1.0

    def __post_init__(self) -> None:
        try:
            self.stroke_width = float(self.stroke_width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"stroke_width must be numeric: {exc}") from exc
        if self.stroke_width <= 0:
            raise ValueError(f"stroke_width must be positive, got {self.stroke_width}")

        try:
            self.alpha = float(self.alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"alpha must be numeric: {exc}") from exc
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {self.alpha}")

        if not self.points:
            raise ValueError("Path must contain at least one point")

        self.points = [Point.from_raw(pt) for pt in self.points]

        if isinstance(self.blend_mode, str) and not isinstance(self.blend_mode, BlendMode):
            try:
                self.blend_mode = BlendMode(self.blend_mode.lower())
            except ValueError:
                pass


def record_freehand_brush(
    points: Sequence[Point | tuple[float, float]],
    color: str,
    stroke_width: float,
) -> Path:
    """Record an opaque freehand brush stroke along an ordered sequence of points."""
    if not points:
        raise ValueError("Points sequence cannot be empty")
    if stroke_width <= 0:
        raise ValueError(f"stroke_width must be positive, got {stroke_width}")

    parsed_points = [Point.from_raw(pt) for pt in points]
    return Path(
        points=parsed_points,
        color=color,
        stroke_width=stroke_width,
        blend_mode=BlendMode.NORMAL,
        alpha=1.0,
    )


def record_highlight_path(
    points: Sequence[Point | tuple[float, float]],
    color: str,
    stroke_width: float,
    alpha: float = 0.5,
) -> Path:
    """Record a highlighting path with semi-transparent alpha and multiply blend mode."""
    if not points:
        raise ValueError("Points sequence cannot be empty")
    if stroke_width <= 0:
        raise ValueError(f"stroke_width must be positive, got {stroke_width}")

    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha must be numeric: {exc}") from exc

    if not (0.0 < alpha_val < 1.0):
        raise ValueError(
            f"Highlight alpha must be strictly semi-transparent (0.0 < alpha < 1.0), got {alpha}"
        )

    parsed_points = [Point.from_raw(pt) for pt in points]
    return Path(
        points=parsed_points,
        color=color,
        stroke_width=stroke_width,
        blend_mode=BlendMode.MULTIPLY,
        alpha=alpha_val,
    )