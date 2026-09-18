"""
Unit tests for Story 5.2.1: Extended Trendlines, Rays, and Infinitely Extended Lines.

Tests verify the domain model behavior and geometric projections for:
- Standard Line Segments (extend_left=False, extend_right=False)
- Rays (extend_left=False, extend_right=True or extend_left=True, extend_right=False)
- Infinitely Extended Lines (extend_left=True, extend_right=True)
"""

import math
from typing import Optional, Tuple

import pytest

from src.charting.geometry.line_projection import (
    BoundingBox,
    calculate_slope,
    calculate_x,
    calculate_y,
    is_point_on_trendline,
    project_trendline,
)
from src.charting.models.trendline import Point, Trendline


# ---------------------------------------------------------------------------
# Helpers for geometric assertions
# ---------------------------------------------------------------------------

def assert_point_approx_equal(actual: Point, expected: Point, abs_tol: float = 1e-5) -> None:
    """Assert two points have matching coordinates within an absolute tolerance."""
    assert actual.x == pytest.approx(expected.x, abs=abs_tol)
    assert actual.y == pytest.approx(expected.y, abs=abs_tol)


def assert_segment_approx_equal(
    actual: Optional[Tuple[Point, Point]],
    expected: Optional[Tuple[Point, Point]],
    abs_tol: float = 1e-5,
) -> None:
    """
    Assert two line segments match regardless of endpoint ordering.
    """
    if expected is None:
        assert actual is None
        return

    assert actual is not None, "Expected a projected segment, got None"
    act_p1, act_p2 = actual
    exp_p1, exp_p2 = expected

    direct_match = (
        act_p1.x == pytest.approx(exp_p1.x, abs=abs_tol)
        and act_p1.y == pytest.approx(exp_p1.y, abs=abs_tol)
        and act_p2.x == pytest.approx(exp_p2.x, abs=abs_tol)
        and act_p2.y == pytest.approx(exp_p2.y, abs=abs_tol)
    )
    reversed_match = (
        act_p1.x == pytest.approx(exp_p2.x, abs=abs_tol)
        and act_p1.y == pytest.approx(exp_p2.y, abs=abs_tol)
        and act_p2.x == pytest.approx(exp_p1.x, abs=abs_tol)
        and act_p2.y == pytest.approx(exp_p1.y, abs=abs_tol)
    )
    assert direct_match or reversed_match, (
        f"Segment endpoints {actual} do not match expected {expected}"
    )


# ===========================================================================
# Model Tests: src/charting/models/trendline.py
# ===========================================================================

class TestTrendlineModel:
    """Tests for the Trendline domain entity and Point structure."""

    def test_point_creation_and_coordinates(self):
        pt = Point(12.5, 45.0)
        assert pt.x == 12.5
        assert pt.y == 45.0

    def test_trendline_default_initialization_is_standard_segment(self):
        """Acceptance Criteria: Standard segment with extend_right=False, extend_left=False."""
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        line = Trendline(point_a=p1, point_b=p2)

        assert line.point_a == p1
        assert line.point_b == p2
        assert line.extend_left is False
        assert line.extend_right is False
        assert line.is_segment is True
        assert line.is_ray is False
        assert line.is_infinite is False

    def test_trendline_ray_initialization(self):
        """Acceptance Criteria: Ray with extend_right=True and extend_left=False."""
        p1 = Point(5.0, 10.0)
        p2 = Point(15.0, 20.0)
        line = Trendline(point_a=p1, point_b=p2, extend_left=False, extend_right=True)

        assert line.extend_left is False
        assert line.extend_right is True
        assert line.is_segment is False
        assert line.is_ray is True
        assert line.is_infinite is False

    def test_trendline_infinite_line_initialization(self):
        """Acceptance Criteria: Infinite line with extend_right=True and extend_left=True."""
        p1 = Point(2.0, 3.0)
        p2 = Point(8.0, 12.0)
        line = Trendline(point_a=p1, point_b=p2, extend_left=True, extend_right=True)

        assert line.extend_left is True
        assert line.extend_right is True
        assert line.is_segment is False
        assert line.is_ray is False
        assert line.is_infinite is True

    def test_trendline_left_ray_initialization(self):
        """Ray extending leftward only."""
        line = Trendline(
            point_a=Point(10.0, 10.0),
            point_b=Point(20.0, 20.0),
            extend_left=True,
            extend_right=False,
        )
        assert line.is_ray is True
        assert line.is_segment is False
        assert line.is_infinite is False

    def test_trendline_coincident_points_raises_value_error(self):
        """Trendline cannot be defined by two identical points."""
        identical_point = Point(4.0, 5.0)
        with pytest.raises(ValueError):
            Trendline(point_a=identical_point, point_b=identical_point)

    def test_trendline_immutability(self):
        """Trendline attributes should be immutable once created."""
        line = Trendline(Point(0.0, 0.0), Point(1.0, 1.0))
        with pytest.raises(Exception):
            line.extend_right = True  # type: ignore


# ===========================================================================
# Geometry Tests: src/charting/geometry/line_projection.py
# ===========================================================================

class TestSlopeAndInterceptCalculation:
    """Tests slope and intercept calculations across various orientations."""

    def test_positive_slope(self):
        line = Trendline(Point(0.0, 0.0), Point(10.0, 20.0))
        assert calculate_slope(line) == pytest.approx(2.0)

    def test_negative_slope(self):
        line = Trendline(Point(0.0, 10.0), Point(5.0, 0.0))
        assert calculate_slope(line) == pytest.approx(-2.0)

    def test_horizontal_line_slope(self):
        line = Trendline(Point(2.0, 5.0), Point(8.0, 5.0))
        assert calculate_slope(line) == pytest.approx(0.0)

    def test_vertical_line_slope_raises_value_error(self):
        line = Trendline(Point(5.0, 2.0), Point(5.0, 10.0))
        with pytest.raises(ValueError):
            calculate_slope(line)


class TestCalculateY:
    """Tests evaluating y given x considering segment, ray, and infinite line limits."""

    def test_standard_segment_calculate_y_within_bounds(self):
        """Standard segment: evaluates valid y only between point_a.x and point_b.x."""
        line = Trendline(Point(10.0, 10.0), Point(20.0, 30.0), extend_left=False, extend_right=False)
        # slope = 2.0; y = 2.0 * (x - 10) + 10
        assert calculate_y(line, 10.0) == pytest.approx(10.0)
        assert calculate_y(line, 15.0) == pytest.approx(20.0)
        assert calculate_y(line, 20.0) == pytest.approx(30.0)

    def test_standard_segment_calculate_y_out_of_bounds_returns_none(self):
        """Standard segment: points outside the x-interval return None."""
        line = Trendline(Point(10.0, 10.0), Point(20.0, 30.0), extend_left=False, extend_right=False)
        assert calculate_y(line, 9.99) is None
        assert calculate_y(line, 20.01) is None
        assert calculate_y(line, -50.0) is None
        assert calculate_y(line, 100.0) is None

    def test_ray_calculate_y_extending_right(self):
        """Ray (extend_right=True, extend_left=False): valid for x >= min(x_a, x_b)."""
        line = Trendline(Point(10.0, 10.0), Point(20.0, 30.0), extend_left=False, extend_right=True)
        # Below left bound -> None
        assert calculate_y(line, 9.99) is None
        assert calculate_y(line, 0.0) is None

        # At and above left bound -> projected y
        assert calculate_y(line, 10.0) == pytest.approx(10.0)
        assert calculate_y(line, 20.0) == pytest.approx(30.0)
        assert calculate_y(line, 30.0) == pytest.approx(50.0)
        assert calculate_y(line, 100.0) == pytest.approx(190.0)

    def test_ray_calculate_y_extending_left(self):
        """Ray (extend_left=True, extend_right=False): valid for x <= max(x_a, x_b)."""
        line = Trendline(Point(10.0, 10.0), Point(20.0, 30.0), extend_left=True, extend_right=False)
        # Above right bound -> None
        assert calculate_y(line, 20.01) is None
        assert calculate_y(line, 50.0) is None

        # At and below right bound -> projected y
        assert calculate_y(line, 20.0) == pytest.approx(30.0)
        assert calculate_y(line, 10.0) == pytest.approx(10.0)
        assert calculate_y(line, 0.0) == pytest.approx(-10.0)
        assert calculate_y(line, -10.0) == pytest.approx(-30.0)

    def test_infinite_line_calculate_y_in_both_directions(self):
        """Infinite line: valid across all x in (-inf, +inf)."""
        line = Trendline(Point(10.0, 10.0), Point(20.0, 30.0), extend_left=True, extend_right=True)
        assert calculate_y(line, -100.0) == pytest.approx(-210.0)
        assert calculate_y(line, 0.0) == pytest.approx(-10.0)
        assert calculate_y(line, 15.0) == pytest.approx(20.0)
        assert calculate_y(line, 500.0) == pytest.approx(990.0)

    def test_calculate_y_point_order_invariance(self):
        """Endpoints provided in descending x-order should still establish the same domain."""
        line_reversed = Trendline(Point(20.0, 30.0), Point(10.0, 10.0), extend_left=False, extend_right=True)
        assert calculate_y(line_reversed, 5.0) is None
        assert calculate_y(line_reversed, 10.0) == pytest.approx(10.0)
        assert calculate_y(line_reversed, 30.0) == pytest.approx(50.0)

    def test_vertical_line_calculate_y_raises_value_error(self):
        """Vertical line has no single y value for x."""
        line = Trendline(Point(5.0, 1.0), Point(5.0, 10.0), extend_left=True, extend_right=True)
        with pytest.raises(ValueError):
            calculate_y(line, 5.0)


class TestCalculateX:
    """Tests evaluating x given y considering segment, ray, and infinite line limits."""

    def test_standard_segment_calculate_x_within_bounds(self):
        line = Trendline(Point(0.0, 0.0), Point(10.0, 20.0), extend_left=False, extend_right=False)
        assert calculate_x(line, 10.0) == pytest.approx(5.0)
        assert calculate_x(line, -5.0) is None
        assert calculate_x(line, 25.0) is None

    def test_infinite_line_calculate_x(self):
        line = Trendline(Point(0.0, 0.0), Point(10.0, 20.0), extend_left=True, extend_right=True)
        assert calculate_x(line, -20.0) == pytest.approx(-10.0)
        assert calculate_x(line, 40.0) == pytest.approx(20.0)

    def test_horizontal_line_calculate_x_raises_value_error(self):
        line = Trendline(Point(0.0, 5.0), Point(10.0, 5.0))
        with pytest.raises(ValueError):
            calculate_x(line, 5.0)


class TestPointOnTrendlineMembership:
    """Tests is_point_on_trendline with tolerance and extension settings."""

    @pytest.fixture
    def standard_segment(self) -> Trendline:
        return Trendline(Point(0.0, 0.0), Point(10.0, 10.0), extend_left=False, extend_right=False)

    @pytest.fixture
    def ray_right(self) -> Trendline:
        return Trendline(Point(0.0, 0.0), Point(10.0, 10.0), extend_left=False, extend_right=True)

    @pytest.fixture
    def infinite_line(self) -> Trendline:
        return Trendline(Point(0.0, 0.0), Point(10.0, 10.0), extend_left=True, extend_right=True)

    def test_segment_membership_internal_and_boundary(self, standard_segment):
        assert is_point_on_trendline(standard_segment, Point(0.0, 0.0)) is True
        assert is_point_on_trendline(standard_segment, Point(5.0, 5.0)) is True
        assert is_point_on_trendline(standard_segment, Point(10.0, 10.0)) is True

    def test_segment_membership_collinear_outside_returns_false(self, standard_segment):
        assert is_point_on_trendline(standard_segment, Point(-1.0, -1.0)) is False
        assert is_point_on_trendline(standard_segment, Point(11.0, 11.0)) is False

    def test_ray_right_membership(self, ray_right):
        # Collinear to the left of origin -> False
        assert is_point_on_trendline(ray_right, Point(-5.0, -5.0)) is False
        # Within initial segment -> True
        assert is_point_on_trendline(ray_right, Point(5.0, 5.0)) is True
        # Beyond point B to the right -> True
        assert is_point_on_trendline(ray_right, Point(50.0, 50.0)) is True

    def test_infinite_line_membership(self, infinite_line):
        assert is_point_on_trendline(infinite_line, Point(-100.0, -100.0)) is True
        assert is_point_on_trendline(infinite_line, Point(5.0, 5.0)) is True
        assert is_point_on_trendline(infinite_line, Point(200.0, 200.0)) is True

    def test_non_collinear_points_return_false_for_all(self, standard_segment, ray_right, infinite_line):
        off_point = Point(5.0, 6.0)
        assert is_point_on_trendline(standard_segment, off_point) is False
        assert is_point_on_trendline(ray_right, off_point) is False
        assert is_point_on_trendline(infinite_line, off_point) is False

    def test_membership_tolerance_thresholds(self, standard_segment):
        tolerance = 0.01
        # Slightly off-line within tolerance
        close_point = Point(5.0, 5.005)
        assert is_point_on_trendline(standard_segment, close_point, tolerance=tolerance) is True

        # Slightly off-line beyond tolerance
        far_point = Point(5.0, 5.02)
        assert is_point_on_trendline(standard_segment, far_point, tolerance=tolerance) is False

    def test_vertical_segment_membership(self):
        vert_seg = Trendline(Point(5.0, 2.0), Point(5.0, 8.0), extend_left=False, extend_right=False)
        assert is_point_on_trendline(vert_seg, Point(5.0, 5.0)) is True
        assert is_point_on_trendline(vert_seg, Point(5.0, 1.0)) is False
        assert is_point_on_trendline(vert_seg, Point(5.0, 9.0)) is False

    def test_vertical_infinite_line_membership(self):
        vert_inf = Trendline(Point(5.0, 2.0), Point(5.0, 8.0), extend_left=True, extend_right=True)
        assert is_point_on_trendline(vert_inf, Point(5.0, -100.0)) is True
        assert is_point_on_trendline(vert_inf, Point(5.0, 500.0)) is True
        assert is_point_on_trendline(vert_inf, Point(5.1, 10.0)) is False


class TestBoundingBoxProjection:
    """
    Tests clipping and projecting segments, rays, and infinite lines to a 2D bounding box (viewport).
    """

    @pytest.fixture
    def bounds(self) -> BoundingBox:
        return BoundingBox(min_x=0.0, max_x=100.0, min_y=0.0, max_y=100.0)

    # --- Standard Segment Projection ---

    def test_segment_entirely_inside_bounds_remains_unchanged(self, bounds):
        line = Trendline(Point(10.0, 10.0), Point(50.0, 50.0), extend_left=False, extend_right=False)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(10.0, 10.0), Point(50.0, 50.0)))

    def test_segment_partially_outside_is_clipped(self, bounds):
        # Starts at x=-20, y=-20, ends at x=50, y=50 (slope = 1.0)
        line = Trendline(Point(-20.0, -20.0), Point(50.0, 50.0), extend_left=False, extend_right=False)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 0.0), Point(50.0, 50.0)))

    def test_segment_entirely_outside_returns_none(self, bounds):
        line = Trendline(Point(-50.0, 10.0), Point(-10.0, 10.0), extend_left=False, extend_right=False)
        result = project_trendline(line, bounds)
        assert result is None

    # --- Ray Projection (extend_right=True, extend_left=False) ---

    def test_ray_right_originating_inside_bounds_projects_to_boundary(self, bounds):
        # A=(10, 10), B=(20, 20) -> intersects max_x=100 at (100, 100)
        line = Trendline(Point(10.0, 10.0), Point(20.0, 20.0), extend_left=False, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(10.0, 10.0), Point(100.0, 100.0)))

    def test_ray_right_steep_slope_hits_top_boundary(self, bounds):
        # Line from (10, 10) to (20, 50): slope = 4.0; y = 4x - 30
        # Hits y=100 at 100 = 4x - 30 => x = 32.5
        line = Trendline(Point(10.0, 10.0), Point(20.0, 50.0), extend_left=False, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(10.0, 10.0), Point(32.5, 100.0)))

    def test_ray_right_originating_outside_bounds_traverses_into_viewport(self, bounds):
        # Originates at (-20, -20), points towards (10, 10). Extends right through the viewport.
        line = Trendline(Point(-20.0, -20.0), Point(10.0, 10.0), extend_left=False, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 0.0), Point(100.0, 100.0)))

    def test_ray_right_pointing_away_returns_none(self, bounds):
        # Starts at (120, 120), points rightward to (150, 150)
        line = Trendline(Point(120.0, 120.0), Point(150.0, 150.0), extend_left=False, extend_right=True)
        result = project_trendline(line, bounds)
        assert result is None

    # --- Ray Projection (extend_left=True, extend_right=False) ---

    def test_ray_left_projects_to_min_boundary(self, bounds):
        # Points (50, 50) and (80, 80). Extend left should project from (0, 0) to (80, 80).
        line = Trendline(Point(50.0, 50.0), Point(80.0, 80.0), extend_left=True, extend_right=False)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 0.0), Point(80.0, 80.0)))

    # --- Infinite Line Projection (extend_left=True, extend_right=True) ---

    def test_infinite_line_spans_entire_bounding_box(self, bounds):
        # A=(20, 40), B=(40, 60): slope = 1.0; y = x + 20
        # Intersects left boundary at x=0 => y=20
        # Intersects top boundary at y=100 => x=80
        line = Trendline(Point(20.0, 40.0), Point(40.0, 60.0), extend_left=True, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 20.0), Point(80.0, 100.0)))

    def test_infinite_line_parallel_and_outside_bounds_returns_none(self, bounds):
        # y = 150 (above viewport max_y=100)
        line = Trendline(Point(10.0, 150.0), Point(20.0, 150.0), extend_left=True, extend_right=True)
        result = project_trendline(line, bounds)
        assert result is None

    def test_infinite_horizontal_line_across_bounds(self, bounds):
        # y = 50 across viewport
        line = Trendline(Point(20.0, 50.0), Point(40.0, 50.0), extend_left=True, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 50.0), Point(100.0, 50.0)))

    def test_infinite_vertical_line_across_bounds(self, bounds):
        # x = 30 across viewport
        line = Trendline(Point(30.0, 10.0), Point(30.0, 80.0), extend_left=True, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(30.0, 0.0), Point(30.0, 100.0)))

    def test_vertical_ray_right_orientation_upwards(self, bounds):
        # For vertical lines, extend_right extends towards higher y (positive direction)
        line = Trendline(Point(50.0, 20.0), Point(50.0, 40.0), extend_left=False, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(50.0, 20.0), Point(50.0, 100.0)))

    def test_negative_slope_infinite_line_crosses_viewport(self, bounds):
        # Line through (0, 100) and (100, 0): slope = -1.0; y = -x + 100
        line = Trendline(Point(25.0, 75.0), Point(75.0, 25.0), extend_left=True, extend_right=True)
        result = project_trendline(line, bounds)
        assert_segment_approx_equal(result, (Point(0.0, 100.0), Point(100.0, 0.0)))