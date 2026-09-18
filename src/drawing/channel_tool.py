"""Parallel Channel and Flat Top/Bottom Drawing Tool implementation."""

import math
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union


class ChannelMode(Enum):
    """Drawing mode for channel construction."""

    PARALLEL = "PARALLEL"
    FLAT_TOP = "FLAT_TOP"
    FLAT_BOTTOM = "FLAT_BOTTOM"


class Point(namedtuple("Point", ["x", "y"])):
    """2D coordinate representation supporting attribute and index access."""

    __slots__ = ()
    x: float
    y: float

    def __new__(cls, x: float, y: float) -> "Point":
        return super().__new__(cls, float(x), float(y))


@dataclass(frozen=True)
class Line:
    """Mathematical 2D line defined by y = slope * x + intercept."""

    slope: float
    intercept: float

    @property
    def is_horizontal(self) -> bool:
        """Return True if the line slope is zero within numerical tolerance."""
        return math.isclose(self.slope, 0.0, abs_tol=1e-9)

    def get_y(self, x: float) -> float:
        """Evaluate y-coordinate at specified x."""
        return self.slope * x + self.intercept

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, Line):
            return False
        return (
            math.isclose(self.slope, other.slope, abs_tol=1e-9)
            and math.isclose(self.intercept, other.intercept, abs_tol=1e-9)
        )

    def __hash__(self) -> int:
        return hash((round(self.slope, 9), round(self.intercept, 9)))


@dataclass
class Channel:
    """Channel entity composed of boundary lines and geometric properties."""

    mode: ChannelMode
    p1: Point
    p2: Point
    base_line: Line
    top_line: Line
    bottom_line: Line
    p3: Optional[Point] = None
    parallel_line: Optional[Line] = None
    median_line: Optional[Line] = None
    slope: float = 0.0
    width: Optional[float] = None
    vertical_width: Optional[float] = None

    def get_y_bounds(self, x: float) -> Tuple[float, float]:
        """Return (lower_y, upper_y) bounds of the channel at specified x."""
        y_bottom = self.bottom_line.get_y(x)
        y_top = self.top_line.get_y(x)
        return min(y_bottom, y_top), max(y_bottom, y_top)

    def contains(self, point: Union[Point, Tuple[float, float]]) -> bool:
        """Check whether point lies inside or on the boundaries of the channel."""
        if isinstance(point, Point):
            px, py = point.x, point.y
        elif isinstance(point, (tuple, list)) and len(point) == 2:
            px, py = float(point[0]), float(point[1])
        else:
            raise ValueError(f"Invalid point for containment check: {point}")

        low_y, high_y = self.get_y_bounds(px)
        tol = max(1e-9, 1e-9 * max(abs(low_y), abs(high_y)))
        return (low_y - tol) <= py <= (high_y + tol)


class ChannelTool:
    """Drawing tool for creating channels from anchor points and modes."""

    @staticmethod
    def _parse_point(pt: Any, name: str) -> Point:
        if pt is None:
            raise ValueError(f"{name} cannot be None")
        if isinstance(pt, Point):
            return pt
        if isinstance(pt, (tuple, list)):
            if len(pt) != 2:
                raise ValueError(f"{name} must have 2 coordinates, got {len(pt)}")
            try:
                return Point(float(pt[0]), float(pt[1]))
            except (ValueError, TypeError) as err:
                raise ValueError(f"{name} coordinates must be numbers: {err}") from err
        raise ValueError(f"Invalid point format for {name}: {pt}")

    @staticmethod
    def _parse_mode(mode: Any) -> ChannelMode:
        if isinstance(mode, ChannelMode):
            return mode
        if isinstance(mode, str):
            try:
                return ChannelMode[mode]
            except KeyError:
                try:
                    return ChannelMode(mode)
                except ValueError:
                    raise ValueError(f"Invalid channel mode: {mode}")
        raise ValueError(f"Invalid channel mode: {mode}")

    @classmethod
    def create_channel(
        cls,
        p1: Union[Point, Tuple[float, float]],
        p2: Union[Point, Tuple[float, float]],
        p3: Optional[Union[Point, Tuple[float, float]]] = None,
        mode: Union[ChannelMode, str] = ChannelMode.PARALLEL,
    ) -> Channel:
        """Create a Channel instance according to anchor points and mode."""
        validated_mode = cls._parse_mode(mode)
        parsed_p1 = cls._parse_point(p1, "p1")
        parsed_p2 = cls._parse_point(p2, "p2")

        if parsed_p1 == parsed_p2:
            raise ValueError("p1 and p2 cannot be identical points")

        if math.isclose(parsed_p1.x, parsed_p2.x, abs_tol=1e-12):
            raise ValueError("Vertical base trendline (infinite slope) is not supported")

        slope = (parsed_p2.y - parsed_p1.y) / (parsed_p2.x - parsed_p1.x)
        base_intercept = parsed_p1.y - slope * parsed_p1.x
        base_line = Line(slope=slope, intercept=base_intercept)

        parsed_p3: Optional[Point] = None
        if p3 is not None:
            parsed_p3 = cls._parse_point(p3, "p3")

        if validated_mode == ChannelMode.PARALLEL:
            if parsed_p3 is None:
                raise ValueError("p3 anchor point is required for PARALLEL channel mode")

            parallel_intercept = parsed_p3.y - slope * parsed_p3.x
            parallel_line = Line(slope=slope, intercept=parallel_intercept)

            if parallel_intercept >= base_intercept:
                top_line = parallel_line
                bottom_line = base_line
            else:
                top_line = base_line
                bottom_line = parallel_line

            median_intercept = (base_intercept + parallel_intercept) / 2.0
            median_line = Line(slope=slope, intercept=median_intercept)

            vert_width = abs(parallel_intercept - base_intercept)
            perp_width = vert_width / math.sqrt(1.0 + slope * slope)

            return Channel(
                mode=ChannelMode.PARALLEL,
                p1=parsed_p1,
                p2=parsed_p2,
                p3=parsed_p3,
                base_line=base_line,
                top_line=top_line,
                bottom_line=bottom_line,
                parallel_line=parallel_line,
                median_line=median_line,
                slope=slope,
                width=perp_width,
                vertical_width=vert_width,
            )

        if validated_mode == ChannelMode.FLAT_TOP:
            top_y = max(parsed_p1.y, parsed_p2.y)
            top_line = Line(slope=0.0, intercept=top_y)
            bottom_line = base_line

            return Channel(
                mode=ChannelMode.FLAT_TOP,
                p1=parsed_p1,
                p2=parsed_p2,
                p3=parsed_p3,
                base_line=base_line,
                top_line=top_line,
                bottom_line=bottom_line,
                parallel_line=None,
                median_line=None,
                slope=slope,
                width=None,
                vertical_width=None,
            )

        if validated_mode == ChannelMode.FLAT_BOTTOM:
            bottom_y = min(parsed_p1.y, parsed_p2.y)
            bottom_line = Line(slope=0.0, intercept=bottom_y)
            top_line = base_line

            return Channel(
                mode=ChannelMode.FLAT_BOTTOM,
                p1=parsed_p1,
                p2=parsed_p2,
                p3=parsed_p3,
                base_line=base_line,
                top_line=top_line,
                bottom_line=bottom_line,
                parallel_line=None,
                median_line=None,
                slope=slope,
                width=None,
                vertical_width=None,
            )

        raise ValueError(f"Unsupported channel mode: {validated_mode}")