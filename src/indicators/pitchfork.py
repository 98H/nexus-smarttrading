"""
Andrews' Pitchfork, Schiff, and Modified Schiff System.

Provides data structures and calculation engines for standard Andrews' Pitchfork,
Schiff Pitchfork, and Modified Schiff Pitchfork indicators.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence, Tuple, Union


@dataclass(frozen=True)
class Point:
    """A 2D coordinate point representing (time/index, price)."""

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "x", float(self.x))
            object.__setattr__(self, "y", float(self.y))
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Point coordinates must be numeric: {exc}") from exc


def _coerce_point(pt: Any) -> Point:
    """Coerce input point representations into a validated Point instance."""
    if isinstance(pt, Point):
        return pt
    if isinstance(pt, (tuple, list)):
        if len(pt) != 2:
            raise ValueError(
                f"Point coordinate sequence must contain exactly 2 elements (x, y), got {len(pt)}"
            )
        try:
            return Point(float(pt[0]), float(pt[1]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Point coordinates must be numeric: {pt}") from exc
    raise TypeError(
        f"Point must be a Point instance or 2-element tuple/list, got {type(pt).__name__}"
    )


@dataclass(frozen=True)
class Line:
    """A 2D line defined in slope-intercept form (y = slope * x + intercept)."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "slope", float(self.slope))
            object.__setattr__(self, "intercept", float(self.intercept))
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Line parameters must be numeric: {exc}") from exc

    def evaluate(self, x: float) -> float:
        """Calculate the y value on the line corresponding to x."""
        return self.slope * float(x) + self.intercept

    def __call__(self, x: float) -> float:
        """Evaluate line equation as a callable function."""
        return self.evaluate(x)

    def passes_through(
        self,
        point: Union[Point, Tuple[float, float], Sequence[float]],
        tol: float = 1e-7,
    ) -> bool:
        """Check whether a point lies on this line within tolerance."""
        pt = _coerce_point(point)
        expected_y = self.evaluate(pt.x)
        return math.isclose(expected_y, pt.y, rel_tol=tol, abs_tol=tol)


class PitchforkType(str, Enum):
    """Enumeration of supported pitchfork calculation variants."""

    STANDARD = "STANDARD"
    SCHIFF = "SCHIFF"
    MODIFIED_SCHIFF = "MODIFIED_SCHIFF"


def _normalize_pitchfork_type(p_type: Any) -> PitchforkType:
    """Normalize and validate pitchfork variant input."""
    if isinstance(p_type, PitchforkType):
        return p_type
    if isinstance(p_type, str):
        upper = p_type.upper()
        if upper in PitchforkType.__members__:
            return PitchforkType[upper]
        try:
            return PitchforkType(p_type)
        except ValueError:
            pass
    raise ValueError(f"Invalid pitchfork type: {p_type}")


class Pitchfork:
    """
    Representation of a Pitchfork indicator comprising origin, midpoint,
    and three parallel lines (median, upper, lower).
    """

    def __init__(
        self,
        p1: Union[Point, Tuple[float, float], Sequence[float]],
        p2: Union[Point, Tuple[float, float], Sequence[float]],
        p3: Union[Point, Tuple[float, float], Sequence[float]],
        pitchfork_type: Union[PitchforkType, str] = PitchforkType.STANDARD,
    ) -> None:
        self.p1: Point = _coerce_point(p1)
        self.p2: Point = _coerce_point(p2)
        self.p3: Point = _coerce_point(p3)
        self.pitchfork_type: PitchforkType = _normalize_pitchfork_type(pitchfork_type)

        self.midpoint: Point = Point(
            (self.p2.x + self.p3.x) / 2.0,
            (self.p2.y + self.p3.y) / 2.0,
        )

        if self.pitchfork_type == PitchforkType.STANDARD:
            self.origin = Point(self.p1.x, self.p1.y)
        elif self.pitchfork_type == PitchforkType.SCHIFF:
            self.origin = Point(self.p1.x, (self.p1.y + self.p2.y) / 2.0)
        elif self.pitchfork_type == PitchforkType.MODIFIED_SCHIFF:
            self.origin = Point(
                (self.p1.x + self.p2.x) / 2.0,
                (self.p1.y + self.p2.y) / 2.0,
            )
        else:
            raise ValueError(f"Unsupported pitchfork type: {self.pitchfork_type}")

        dx = self.midpoint.x - self.origin.x
        if math.isclose(dx, 0.0, abs_tol=1e-12):
            raise ValueError("Median line is vertical (undefined slope)")

        dy = self.midpoint.y - self.origin.y
        slope = dy / dx
        intercept = self.origin.y - (slope * self.origin.x)

        self.median_line = Line(slope=slope, intercept=intercept)
        self.upper_line = Line(slope=slope, intercept=self.p2.y - (slope * self.p2.x))
        self.lower_line = Line(slope=slope, intercept=self.p3.y - (slope * self.p3.x))

    def __repr__(self) -> str:
        return (
            f"Pitchfork(type={self.pitchfork_type.value}, "
            f"origin={self.origin}, midpoint={self.midpoint}, "
            f"median_line={self.median_line})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pitchfork):
            return False
        return (
            self.pitchfork_type == other.pitchfork_type
            and self.origin == other.origin
            and self.midpoint == other.midpoint
            and self.median_line == other.median_line
            and self.upper_line == other.upper_line
            and self.lower_line == other.lower_line
        )


def calculate_pitchfork(
    p1: Union[Point, Tuple[float, float], Sequence[float]],
    p2: Union[Point, Tuple[float, float], Sequence[float]],
    p3: Union[Point, Tuple[float, float], Sequence[float]],
    pitchfork_type: Union[PitchforkType, str] = PitchforkType.STANDARD,
) -> Pitchfork:
    """Calculate pitchfork given three anchor points and variant type."""
    return Pitchfork(p1, p2, p3, pitchfork_type=pitchfork_type)


def andrews_pitchfork(
    p1: Union[Point, Tuple[float, float], Sequence[float]],
    p2: Union[Point, Tuple[float, float], Sequence[float]],
    p3: Union[Point, Tuple[float, float], Sequence[float]],
) -> Pitchfork:
    """Calculate standard Andrews' Pitchfork."""
    return calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.STANDARD)


def schiff_pitchfork(
    p1: Union[Point, Tuple[float, float], Sequence[float]],
    p2: Union[Point, Tuple[float, float], Sequence[float]],
    p3: Union[Point, Tuple[float, float], Sequence[float]],
) -> Pitchfork:
    """Calculate Schiff Pitchfork with origin adjusted to (P1.x, (P1.y + P2.y) / 2)."""
    return calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.SCHIFF)


def modified_schiff_pitchfork(
    p1: Union[Point, Tuple[float, float], Sequence[float]],
    p2: Union[Point, Tuple[float, float], Sequence[float]],
    p3: Union[Point, Tuple[float, float], Sequence[float]],
) -> Pitchfork:
    """Calculate Modified Schiff Pitchfork with origin adjusted to midpoint of P1 and P2."""
    return calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.MODIFIED_SCHIFF)