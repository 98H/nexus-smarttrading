"""Geometric calculation and viewport projection functions for trendlines."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from src.charting.models.trendline import Point, Trendline

__all__ = [
    "BoundingBox",
    "calculate_slope",
    "calculate_x",
    "calculate_y",
    "is_point_on_trendline",
    "project_trendline",
]


@dataclass(frozen=True)
class BoundingBox:
    """Defines a 2D rectangular bounding box for viewport clipping."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        try:
            min_x_val = float(self.min_x)
            max_x_val = float(self.max_x)
            min_y_val = float(self.min_y)
            max_y_val = float(self.max_y)
        except (TypeError, ValueError) as exc:
            raise TypeError("BoundingBox coordinates must be numeric values.") from exc

        if not all(
            math.isfinite(val)
            for val in (min_x_val, max_x_val, min_y_val, max_y_val)
        ):
            raise ValueError("BoundingBox coordinates must be finite real numbers.")

        if min_x_val > max_x_val or min_y_val > max_y_val:
            raise ValueError("BoundingBox min coordinates must not exceed max coordinates.")

        object.__setattr__(self, "min_x", min_x_val)
        object.__setattr__(self, "max_x", max_x_val)
        object.__setattr__(self, "min_y", min_y_val)
        object.__setattr__(self, "max_y", max_y_val)


def calculate_slope(line: Trendline) -> float:
    """
    Calculate the slope (dy / dx) of the trendline.

    Raises:
        ValueError: If the line is vertical.
    """
    if math.isclose(line.point_a.x, line.point_b.x, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Slope of a vertical line is undefined.")
    dy = line.point_b.y - line.point_a.y
    dx = line.point_b.x - line.point_a.x
    return dy / dx


def calculate_y(line: Trendline, x: float) -> Optional[float]:
    """
    Calculate y given x considering segment, ray, and infinite line limits.

    Returns:
        float if x is within the valid domain, or None if out of bounds.

    Raises:
        ValueError: If the line is vertical.
    """
    if math.isclose(line.point_a.x, line.point_b.x, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Cannot calculate y for a vertical line.")

    min_x = min(line.point_a.x, line.point_b.x)
    max_x = max(line.point_a.x, line.point_b.x)

    if not line.extend_left and (x < min_x and not math.isclose(x, min_x, rel_tol=1e-9, abs_tol=1e-12)):
        return None
    if not line.extend_right and (x > max_x and not math.isclose(x, max_x, rel_tol=1e-9, abs_tol=1e-12)):
        return None

    dx = line.point_b.x - line.point_a.x
    slope = (line.point_b.y - line.point_a.y) / dx
    return line.point_a.y + slope * (x - line.point_a.x)


def calculate_x(line: Trendline, y: float) -> Optional[float]:
    """
    Calculate x given y considering segment, ray, and infinite line limits.

    Returns:
        float if y is within the valid domain, or None if out of bounds.

    Raises:
        ValueError: If the line is horizontal.
    """
    if math.isclose(line.point_a.y, line.point_b.y, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Cannot calculate x for a horizontal line.")

    if math.isclose(line.point_a.x, line.point_b.x, rel_tol=1e-9, abs_tol=1e-12):
        min_y = min(line.point_a.y, line.point_b.y)
        max_y = max(line.point_a.y, line.point_b.y)
        if not line.extend_left and (y < min_y and not math.isclose(y, min_y, rel_tol=1e-9, abs_tol=1e-12)):
            return None
        if not line.extend_right and (y > max_y and not math.isclose(y, max_y, rel_tol=1e-9, abs_tol=1e-12)):
            return None
        return line.point_a.x

    dx = line.point_b.x - line.point_a.x
    dy = line.point_b.y - line.point_a.y
    slope = dy / dx
    x = line.point_a.x + (y - line.point_a.y) / slope

    min_x = min(line.point_a.x, line.point_b.x)
    max_x = max(line.point_a.x, line.point_b.x)

    if not line.extend_left and (x < min_x and not math.isclose(x, min_x, rel_tol=1e-9, abs_tol=1e-12)):
        return None
    if not line.extend_right and (x > max_x and not math.isclose(x, max_x, rel_tol=1e-9, abs_tol=1e-12)):
        return None

    return x


def is_point_on_trendline(line: Trendline, point: Point, tolerance: float = 1e-5) -> bool:
    """Check if a point lies on the trendline within a Euclidean distance tolerance."""
    dx = line.point_b.x - line.point_a.x
    dy = line.point_b.y - line.point_a.y
    length_sq = dx * dx + dy * dy

    u_x = point.x - line.point_a.x
    u_y = point.y - line.point_a.y
    t = (u_x * dx + u_y * dy) / length_sq

    proj_x = line.point_a.x + t * dx
    proj_y = line.point_a.y + t * dy

    if not math.isclose(dx, 0.0, rel_tol=1e-9, abs_tol=1e-12):
        min_x = min(line.point_a.x, line.point_b.x)
        max_x = max(line.point_a.x, line.point_b.x)

        x_start = -math.inf if line.extend_left else min_x
        x_end = math.inf if line.extend_right else max_x

        clamped_x = max(x_start, min(x_end, proj_x))
        clamped_t = (clamped_x - line.point_a.x) / dx
        clamped_y = line.point_a.y + clamped_t * dy
    else:
        min_y = min(line.point_a.y, line.point_b.y)
        max_y = max(line.point_a.y, line.point_b.y)

        y_start = -math.inf if line.extend_left else min_y
        y_end = math.inf if line.extend_right else max_y

        clamped_y = max(y_start, min(y_end, proj_y))
        clamped_x = line.point_a.x

    dist = math.hypot(point.x - clamped_x, point.y - clamped_y)
    return dist <= tolerance


def _canonicalize_endpoints(line: Trendline) -> Tuple[Point, Point]:
    """Return endpoints ordered left-to-right (or bottom-to-top if vertical)."""
    if line.point_a.x < line.point_b.x or (
        math.isclose(line.point_a.x, line.point_b.x, rel_tol=1e-9, abs_tol=1e-12)
        and line.point_a.y < line.point_b.y
    ):
        return line.point_a, line.point_b
    return line.point_b, line.point_a


def project_trendline(
    line: Trendline,
    bounds: BoundingBox,
) -> Optional[Tuple[Point, Point]]:
    """
    Clip and project a trendline (segment, ray, or infinite line) to a bounding box.

    Returns:
        A tuple of two projected Points defining the visible line segment,
        or None if the line does not intersect the bounding box.
    """
    p1, p2 = _canonicalize_endpoints(line)
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    t_0 = -math.inf if line.extend_left else 0.0
    t_1 = math.inf if line.extend_right else 1.0

    p_values = [-dx, dx, -dy, dy]
    q_values = [
        p1.x - bounds.min_x,
        bounds.max_x - p1.x,
        p1.y - bounds.min_y,
        bounds.max_y - p1.y,
    ]

    for p, q in zip(p_values, q_values):
        if math.isclose(p, 0.0, abs_tol=1e-12):
            if q < -1e-9:
                return None
        elif p < 0.0:
            r = q / p
            if r > t_1 + 1e-9:
                return None
            if r > t_0:
                t_0 = r
        else:
            r = q / p
            if r < t_0 - 1e-9:
                return None
            if r < t_1:
                t_1 = r

    if t_0 > t_1 + 1e-9:
        return None

    if math.isclose(t_0, t_1, abs_tol=1e-9):
        return None

    x0 = p1.x + t_0 * dx
    y0 = p1.y + t_0 * dy
    x1 = p1.x + t_1 * dx
    y1 = p1.y + t_1 * dy

    x0 = max(bounds.min_x, min(bounds.max_x, x0))
    y0 = max(bounds.min_y, min(bounds.max_y, y0))
    x1 = max(bounds.min_x, min(bounds.max_x, x1))
    y1 = max(bounds.min_y, min(bounds.max_y, y1))

    if math.isclose(x0, bounds.min_x, abs_tol=1e-12):
        x0 = bounds.min_x
    if math.isclose(x0, bounds.max_x, abs_tol=1e-12):
        x0 = bounds.max_x
    if math.isclose(y0, bounds.min_y, abs_tol=1e-12):
        y0 = bounds.min_y
    if math.isclose(y0, bounds.max_y, abs_tol=1e-12):
        y0 = bounds.max_y

    if math.isclose(x1, bounds.min_x, abs_tol=1e-12):
        x1 = bounds.min_x
    if math.isclose(x1, bounds.max_x, abs_tol=1e-12):
        x1 = bounds.max_x
    if math.isclose(y1, bounds.min_y, abs_tol=1e-12):
        y1 = bounds.min_y
    if math.isclose(y1, bounds.max_y, abs_tol=1e-12):
        y1 = bounds.max_y

    if math.isclose(x0, 0.0, abs_tol=1e-12):
        x0 = 0.0
    if math.isclose(y0, 0.0, abs_tol=1e-12):
        y0 = 0.0
    if math.isclose(x1, 0.0, abs_tol=1e-12):
        x1 = 0.0
    if math.isclose(y1, 0.0, abs_tol=1e-12):
        y1 = 0.0

    if math.isclose(x0, x1, abs_tol=1e-9) and math.isclose(y0, y1, abs_tol=1e-9):
        return None

    return Point(x0, y0), Point(x1, y1)