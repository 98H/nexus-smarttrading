"""Domain models for trendlines, rays, and infinite lines."""

from dataclasses import dataclass
import math
from typing import Any

__all__ = ["Point", "Trendline"]

# Default tolerances for coincident point detection
COINCIDENT_REL_TOL = 1e-9
COINCIDENT_ABS_TOL = 1e-9


@dataclass(frozen=True)
class Point:
    """Represents a 2D point with Cartesian coordinates (x, y)."""

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            x_val = float(self.x)
            y_val = float(self.y)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Point coordinates must be numeric, got x={self.x!r}, y={self.y!r}") from exc

        if not math.isfinite(x_val) or not math.isfinite(y_val):
            raise ValueError(f"Point coordinates must be finite real numbers, got x={x_val}, y={y_val}")

        object.__setattr__(self, "x", x_val)
        object.__setattr__(self, "y", y_val)


@dataclass(frozen=True)
class Trendline:
    """
    Domain entity representing a trendline defined by two points.

    Supports line segments, rays, and infinite lines via extension flags:
    - Standard Segment: extend_left=False, extend_right=False
    - Ray (Right): extend_left=False, extend_right=True
    - Ray (Left): extend_left=True, extend_right=False
    - Infinite Line: extend_left=True, extend_right=True
    """

    point_a: Point
    point_b: Point
    extend_left: bool = False
    extend_right: bool = False

    def __post_init__(self) -> None:
        p_a = self._validate_point(self.point_a, "point_a")
        p_b = self._validate_point(self.point_b, "point_b")
        object.__setattr__(self, "point_a", p_a)
        object.__setattr__(self, "point_b", p_b)

        object.__setattr__(self, "extend_left", bool(self.extend_left))
        object.__setattr__(self, "extend_right", bool(self.extend_right))

        if self._are_coincident(p_a, p_b):
            raise ValueError("Trendline cannot be defined by two coincident points.")

    @staticmethod
    def _validate_point(pt: Any, field_name: str) -> Point:
        if isinstance(pt, Point):
            return pt
        if isinstance(pt, (tuple, list)):
            if len(pt) != 2:
                raise ValueError(f"{field_name} sequence must contain exactly 2 coordinates, got {len(pt)}")
            return Point(pt[0], pt[1])
        raise TypeError(f"{field_name} must be an instance of Point, got {type(pt).__name__}")

    @staticmethod
    def _are_coincident(p1: Point, p2: Point) -> bool:
        """
        Check if two points are coincident using both relative and absolute tolerances.

        Relative tolerance handles large coordinates (e.g. timestamp scales ~ 1.7e12),
        while absolute tolerance handles values near zero.
        """
        return math.isclose(
            p1.x, p2.x, rel_tol=COINCIDENT_REL_TOL, abs_tol=COINCIDENT_ABS_TOL
        ) and math.isclose(
            p1.y, p2.y, rel_tol=COINCIDENT_REL_TOL, abs_tol=COINCIDENT_ABS_TOL
        )

    @property
    def is_segment(self) -> bool:
        """True if bounded between point_a and point_b without extensions."""
        return not self.extend_left and not self.extend_right

    @property
    def is_ray(self) -> bool:
        """True if extended in exactly one direction."""
        return self.extend_left != self.extend_right

    @property
    def is_infinite(self) -> bool:
        """True if extended in both directions."""
        return self.extend_left and self.extend_right