"""Vector shape core math primitives and interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """Immutable representation of a 2D point with float coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))


class VectorShape:
    """Representation of a vector shape defined by a sequence of 2D vertices."""

    def __init__(self, vertices: Sequence[Point2D] | None = None) -> None:
        self.vertices: list[Point2D] = list(vertices) if vertices is not None else []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorShape):
            return NotImplemented
        return self.vertices == other.vertices

    def __repr__(self) -> str:
        return f"VectorShape(vertices={self.vertices!r})"


def lerp(point_a: Point2D, point_b: Point2D, t: float) -> Point2D:
    """Linearly interpolate between two 2D points at factor t in [0.0, 1.0]."""
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Interpolation factor t must be in [0.0, 1.0], got {t}")

    x = point_a.x + t * (point_b.x - point_a.x)
    y = point_a.y + t * (point_b.y - point_a.y)
    return Point2D(x, y)