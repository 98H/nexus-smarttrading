import math
from typing import Tuple
import pytest

from src.drawing import Channel, ChannelMode, ChannelTool, Line, Point
from src.drawing.channel_tool import (
    Channel as DirectChannel,
    ChannelMode as DirectChannelMode,
    ChannelTool as DirectChannelTool,
    Line as DirectLine,
    Point as DirectPoint,
)


# ============================================================================
# Module Export and Structure Tests
# ============================================================================


def test_package_exports():
    """Verify that src.drawing correctly exports the channel tool entities."""
    assert ChannelTool is DirectChannelTool
    assert ChannelMode is DirectChannelMode
    assert Point is DirectPoint
    assert Channel is DirectChannel
    assert Line is DirectLine


def test_channel_mode_enum_values():
    """Verify ChannelMode provides the expected enum variants."""
    assert hasattr(ChannelMode, "PARALLEL")
    assert hasattr(ChannelMode, "FLAT_TOP")
    assert hasattr(ChannelMode, "FLAT_BOTTOM")
    assert len({ChannelMode.PARALLEL, ChannelMode.FLAT_TOP, ChannelMode.FLAT_BOTTOM}) == 3


def test_point_creation_and_interface():
    """Verify Point coordinates and duck-typing as sequence/attributes."""
    p = Point(10.5, -20.25)
    assert p.x == pytest.approx(10.5)
    assert p.y == pytest.approx(-20.25)
    assert p[0] == pytest.approx(10.5)
    assert p[1] == pytest.approx(-20.25)
    x, y = p
    assert x == pytest.approx(10.5)
    assert y == pytest.approx(-20.25)
    assert Point(10.5, -20.25) == p
    assert Point(0.0, 0.0) != p


# ============================================================================
# Acceptance Criteria 1: Parallel Channel with Three Anchor Points (P1, P2, P3)
# ============================================================================


class TestParallelChannelWithThreeAnchorPoints:
    """Tests for standard parallel channel defined by P1, P2 (base trendline)

    and P3 (width/offset).
    """

    def test_upward_parallel_channel_creation(self):
        """P1 and P2 define an upward trendline; P3 sets upper parallel boundary."""
        tool = ChannelTool()
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        p3 = Point(0.0, 5.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert isinstance(channel, Channel)
        assert channel.mode == ChannelMode.PARALLEL
        assert channel.p1 == p1
        assert channel.p2 == p2
        assert channel.p3 == p3
        assert channel.slope == pytest.approx(1.0)

        # Base line passes through (0,0) and (10,10)
        assert channel.base_line.slope == pytest.approx(1.0)
        assert channel.base_line.intercept == pytest.approx(0.0)
        assert channel.base_line.get_y(5.0) == pytest.approx(5.0)

        # Parallel line passes through (0,5) with slope 1.0
        assert channel.parallel_line.slope == pytest.approx(1.0)
        assert channel.parallel_line.intercept == pytest.approx(5.0)
        assert channel.parallel_line.get_y(5.0) == pytest.approx(10.0)

        # Top line is parallel_line, bottom line is base_line
        assert channel.top_line.intercept == pytest.approx(5.0)
        assert channel.bottom_line.intercept == pytest.approx(0.0)

    def test_downward_parallel_channel_p3_below_base_line(self):
        """P1 and P2 define a downward trendline; P3 sets lower parallel boundary."""
        tool = ChannelTool()
        p1 = Point(0.0, 10.0)
        p2 = Point(10.0, 0.0)
        p3 = Point(0.0, 4.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.slope == pytest.approx(-1.0)
        assert channel.base_line.slope == pytest.approx(-1.0)
        assert channel.base_line.intercept == pytest.approx(10.0)

        assert channel.parallel_line.slope == pytest.approx(-1.0)
        assert channel.parallel_line.intercept == pytest.approx(4.0)

        # Base line is higher than parallel line: base is top, parallel is bottom
        assert channel.top_line == channel.base_line
        assert channel.bottom_line == channel.parallel_line

    def test_parallel_channel_median_line(self):
        """Verify the median line is halfway between the base and parallel lines."""
        tool = ChannelTool()
        p1 = Point(0.0, 2.0)
        p2 = Point(4.0, 6.0)
        p3 = Point(0.0, 8.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.median_line is not None
        assert channel.median_line.slope == pytest.approx(1.0)
        # Base intercept = 2.0, parallel intercept = 8.0, median intercept = 5.0
        assert channel.median_line.intercept == pytest.approx(5.0)

        # Evaluated at x = 2: base y = 4.0, parallel y = 10.0, median y = 7.0
        assert channel.base_line.get_y(2.0) == pytest.approx(4.0)
        assert channel.parallel_line.get_y(2.0) == pytest.approx(10.0)
        assert channel.median_line.get_y(2.0) == pytest.approx(7.0)

    def test_parallel_channel_width_and_vertical_offset(self):
        """Verify perpendicular distance (width) and vertical distance calculations.

        Base line: y = 0.75x + 0 (m = 3/4 = 0.75)
        Parallel line through (0, 5): y = 0.75x + 5
        Vertical distance = |5 - 0| = 5.0
        Perpendicular distance = |5 - 0| / sqrt(1 + 0.75^2) = 5 / 1.25 = 4.0
        """
        tool = ChannelTool()
        p1 = Point(0.0, 0.0)
        p2 = Point(4.0, 3.0)
        p3 = Point(0.0, 5.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.vertical_width == pytest.approx(5.0)
        assert channel.width == pytest.approx(4.0)

    def test_parallel_channel_y_bounds_at_x(self):
        """Verify get_y_bounds returns (lower_y, upper_y) at specified x."""
        tool = ChannelTool()
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        p3 = Point(0.0, 4.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        low_y, high_y = channel.get_y_bounds(5.0)
        assert low_y == pytest.approx(5.0)
        assert high_y == pytest.approx(9.0)

    def test_parallel_channel_point_containment(self):
        """Verify containment checking for points inside, on boundary, and outside."""
        tool = ChannelTool()
        channel = tool.create_channel(
            Point(0.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 4.0),
            mode=ChannelMode.PARALLEL,
        )

        # On boundaries at x=5
        assert channel.contains(Point(5.0, 5.0)) is True
        assert channel.contains(Point(5.0, 9.0)) is True

        # Strictly inside
        assert channel.contains(Point(5.0, 7.0)) is True
        assert channel.contains((5.0, 7.0)) is True

        # Strictly outside
        assert channel.contains(Point(5.0, 4.9)) is False
        assert channel.contains(Point(5.0, 9.1)) is False

    def test_parallel_channel_collinear_p3_zero_width(self):
        """When P3 lies directly on the base line, width and offset are zero."""
        tool = ChannelTool()
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        p3 = Point(5.0, 5.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.width == pytest.approx(0.0)
        assert channel.vertical_width == pytest.approx(0.0)
        assert channel.top_line.intercept == pytest.approx(channel.bottom_line.intercept)

    def test_parallel_channel_horizontal_base(self):
        """Base trendline with slope 0 produces horizontal channel lines."""
        tool = ChannelTool()
        p1 = Point(0.0, 5.0)
        p2 = Point(10.0, 5.0)
        p3 = Point(0.0, 12.0)

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.slope == pytest.approx(0.0)
        assert channel.base_line.is_horizontal is True
        assert channel.parallel_line.is_horizontal is True
        assert channel.width == pytest.approx(7.0)
        assert channel.vertical_width == pytest.approx(7.0)


# ============================================================================
# Acceptance Criteria 2: Flat Top Drawing Tool with Two Anchor Points
# ============================================================================


class TestFlatTopChannelWithTwoAnchorPoints:
    """Tests for Flat Top channel where mode is FLAT_TOP and two anchor points

    define the sloping base trendline and horizontal resistance boundary.
    """

    def test_flat_top_ascending_base_trendline(self):
        """Ascending trendline between P1=(0, 2) and P2=(10, 8).

        Flat top resistance must sit at max(y1, y2) = 8.0 with slope 0.
        """
        tool = ChannelTool()
        p1 = Point(0.0, 2.0)
        p2 = Point(10.0, 8.0)

        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_TOP)

        assert channel.mode == ChannelMode.FLAT_TOP
        assert channel.p1 == p1
        assert channel.p2 == p2
        assert channel.p3 is None

        # Base trendline connects P1 and P2
        assert channel.base_line.slope == pytest.approx(0.6)
        assert channel.base_line.intercept == pytest.approx(2.0)

        # Top boundary is flat (slope = 0) at max(y1, y2) = 8.0
        assert channel.top_line.is_horizontal is True
        assert channel.top_line.slope == pytest.approx(0.0)
        assert channel.top_line.intercept == pytest.approx(8.0)
        assert channel.top_line.get_y(0.0) == pytest.approx(8.0)
        assert channel.top_line.get_y(10.0) == pytest.approx(8.0)

        # Bottom boundary is the base trendline
        assert channel.bottom_line == channel.base_line

    def test_flat_top_descending_base_trendline(self):
        """Descending trendline between P1=(0, 8) and P2=(10, 2).

        Flat top resistance must sit at max(y1, y2) = 8.0 with slope 0.
        """
        tool = ChannelTool()
        p1 = Point(0.0, 8.0)
        p2 = Point(10.0, 2.0)

        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_TOP)

        assert channel.base_line.slope == pytest.approx(-0.6)
        assert channel.base_line.intercept == pytest.approx(8.0)

        # Top boundary is horizontal at 8.0
        assert channel.top_line.is_horizontal is True
        assert channel.top_line.intercept == pytest.approx(8.0)
        assert channel.top_line.get_y(5.0) == pytest.approx(8.0)

        # At x=5: bottom = -0.6 * 5 + 8 = 5.0, top = 8.0
        low, high = channel.get_y_bounds(5.0)
        assert low == pytest.approx(5.0)
        assert high == pytest.approx(8.0)

    def test_flat_top_median_line_is_none(self):
        """Median line is not defined for non-parallel Flat Top channels."""
        tool = ChannelTool()
        channel = tool.create_channel(
            Point(0.0, 0.0), Point(10.0, 10.0), mode=ChannelMode.FLAT_TOP
        )
        assert channel.median_line is None

    def test_flat_top_point_containment(self):
        """Verify containment logic for Flat Top wedge/channel."""
        tool = ChannelTool()
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_TOP)

        # At x=5: bottom is 5.0, top is 10.0
        assert channel.contains(Point(5.0, 7.5)) is True
        assert channel.contains(Point(5.0, 5.0)) is True
        assert channel.contains(Point(5.0, 10.0)) is True
        assert channel.contains(Point(5.0, 4.5)) is False
        assert channel.contains(Point(5.0, 10.5)) is False


# ============================================================================
# Acceptance Criteria 2: Flat Bottom Drawing Tool with Two Anchor Points
# ============================================================================


class TestFlatBottomChannelWithTwoAnchorPoints:
    """Tests for Flat Bottom channel where mode is FLAT_BOTTOM and two anchor points

    define the sloping base trendline and horizontal support boundary.
    """

    def test_flat_bottom_descending_base_trendline(self):
        """Descending trendline between P1=(0, 10) and P2=(10, 4).

        Flat bottom support must sit at min(y1, y2) = 4.0 with slope 0.
        """
        tool = ChannelTool()
        p1 = Point(0.0, 10.0)
        p2 = Point(10.0, 4.0)

        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_BOTTOM)

        assert channel.mode == ChannelMode.FLAT_BOTTOM
        assert channel.p1 == p1
        assert channel.p2 == p2
        assert channel.p3 is None

        # Base trendline connects P1 and P2
        assert channel.base_line.slope == pytest.approx(-0.6)
        assert channel.base_line.intercept == pytest.approx(10.0)

        # Bottom boundary is flat (slope = 0) at min(y1, y2) = 4.0
        assert channel.bottom_line.is_horizontal is True
        assert channel.bottom_line.slope == pytest.approx(0.0)
        assert channel.bottom_line.intercept == pytest.approx(4.0)
        assert channel.bottom_line.get_y(0.0) == pytest.approx(4.0)
        assert channel.bottom_line.get_y(10.0) == pytest.approx(4.0)

        # Top boundary is the base trendline
        assert channel.top_line == channel.base_line

    def test_flat_bottom_ascending_base_trendline(self):
        """Ascending trendline between P1=(0, 2) and P2=(10, 8).

        Flat bottom support must sit at min(y1, y2) = 2.0 with slope 0.
        """
        tool = ChannelTool()
        p1 = Point(0.0, 2.0)
        p2 = Point(10.0, 8.0)

        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_BOTTOM)

        assert channel.base_line.slope == pytest.approx(0.6)
        assert channel.base_line.intercept == pytest.approx(2.0)

        # Bottom boundary is horizontal at 2.0
        assert channel.bottom_line.is_horizontal is True
        assert channel.bottom_line.intercept == pytest.approx(2.0)
        assert channel.bottom_line.get_y(5.0) == pytest.approx(2.0)

        # At x=5: bottom = 2.0, top = 0.6 * 5 + 2 = 5.0
        low, high = channel.get_y_bounds(5.0)
        assert low == pytest.approx(2.0)
        assert high == pytest.approx(5.0)

    def test_flat_bottom_median_line_is_none(self):
        """Median line is not defined for non-parallel Flat Bottom channels."""
        tool = ChannelTool()
        channel = tool.create_channel(
            Point(0.0, 10.0), Point(10.0, 0.0), mode=ChannelMode.FLAT_BOTTOM
        )
        assert channel.median_line is None

    def test_flat_bottom_point_containment(self):
        """Verify containment logic for Flat Bottom wedge/channel."""
        tool = ChannelTool()
        p1 = Point(0.0, 10.0)
        p2 = Point(10.0, 0.0)
        channel = tool.create_channel(p1, p2, mode=ChannelMode.FLAT_BOTTOM)

        # At x=5: bottom is 0.0, top is 5.0
        assert channel.contains(Point(5.0, 2.5)) is True
        assert channel.contains(Point(5.0, 0.0)) is True
        assert channel.contains(Point(5.0, 5.0)) is True
        assert channel.contains(Point(5.0, -0.5)) is False
        assert channel.contains(Point(5.0, 5.5)) is False


# ============================================================================
# Validation, Preconditions and Exception Handling
# ============================================================================


class TestChannelValidationAndExceptions:
    """Tests for input validation, missing arguments, and geometric edge cases."""

    def test_parallel_mode_missing_p3_raises_value_error(self):
        """Parallel mode strictly requires three anchor points."""
        tool = ChannelTool()
        with pytest.raises(ValueError):
            tool.create_channel(Point(0.0, 0.0), Point(10.0, 10.0), p3=None, mode=ChannelMode.PARALLEL)

    def test_flat_modes_allow_optional_p3(self):
        """Flat modes work with 2 points, but gracefully handle P3 if provided."""
        tool = ChannelTool()
        p1 = Point(0.0, 2.0)
        p2 = Point(10.0, 8.0)
        p3 = Point(0.0, 12.0)

        # Providing p3 should not break flat mode creation
        channel = tool.create_channel(p1, p2, p3=p3, mode=ChannelMode.FLAT_TOP)
        assert channel.mode == ChannelMode.FLAT_TOP

    def test_identical_p1_and_p2_raises_value_error(self):
        """P1 == P2 cannot establish a base trendline slope."""
        tool = ChannelTool()
        same_point = Point(5.0, 5.0)

        with pytest.raises(ValueError):
            tool.create_channel(same_point, same_point, Point(0.0, 10.0), mode=ChannelMode.PARALLEL)

        with pytest.raises(ValueError):
            tool.create_channel(same_point, same_point, mode=ChannelMode.FLAT_TOP)

        with pytest.raises(ValueError):
            tool.create_channel(same_point, same_point, mode=ChannelMode.FLAT_BOTTOM)

    def test_vertical_base_trendline_raises_value_error(self):
        """P1.x == P2.x produces an undefined/infinite slope, which is invalid."""
        tool = ChannelTool()
        p1 = Point(5.0, 0.0)
        p2 = Point(5.0, 10.0)

        with pytest.raises(ValueError):
            tool.create_channel(p1, p2, Point(0.0, 0.0), mode=ChannelMode.PARALLEL)

        with pytest.raises(ValueError):
            tool.create_channel(p1, p2, mode=ChannelMode.FLAT_TOP)

        with pytest.raises(ValueError):
            tool.create_channel(p1, p2, mode=ChannelMode.FLAT_BOTTOM)

    def test_invalid_channel_mode_raises_value_error(self):
        """Unrecognized mode values must raise ValueError."""
        tool = ChannelTool()
        with pytest.raises(ValueError):
            tool.create_channel(Point(0.0, 0.0), Point(10.0, 10.0), mode="INVALID_MODE")

    def test_none_base_points_raise_value_error(self):
        """Missing P1 or P2 must raise ValueError."""
        tool = ChannelTool()
        with pytest.raises(ValueError):
            tool.create_channel(None, Point(10.0, 10.0), mode=ChannelMode.FLAT_TOP)

        with pytest.raises(ValueError):
            tool.create_channel(Point(0.0, 0.0), None, mode=ChannelMode.FLAT_BOTTOM)


# ============================================================================
# API Usability and Numerical Robustness Tests
# ============================================================================


class TestAPIUsabilityAndNumericalRobustness:
    """Tests for raw tuple inputs, classmethods, and numerical scale variations."""

    def test_tuple_inputs_accepted(self):
        """Tool accepts standard (x, y) tuples interchangeably with Point objects."""
        tool = ChannelTool()
        channel = tool.create_channel((0.0, 0.0), (10.0, 10.0), (0.0, 5.0), mode=ChannelMode.PARALLEL)

        assert isinstance(channel.p1, Point)
        assert isinstance(channel.p2, Point)
        assert isinstance(channel.p3, Point)
        assert channel.p1.x == pytest.approx(0.0)
        assert channel.p2.y == pytest.approx(10.0)
        assert channel.parallel_line.intercept == pytest.approx(5.0)

    def test_classmethod_or_static_create_channel_dispatch(self):
        """ChannelTool.create_channel can be invoked without manual instantiation."""
        channel = ChannelTool.create_channel(
            Point(0.0, 10.0), Point(10.0, 20.0), Point(0.0, 0.0), mode=ChannelMode.PARALLEL
        )
        assert isinstance(channel, Channel)
        assert channel.slope == pytest.approx(1.0)

    def test_large_financial_scale_coordinates(self):
        """Ensure numerical precision with large coordinates (e.g. timestamps and prices)."""
        tool = ChannelTool()
        # Simulated unix epoch timestamps (X) and large index prices (Y)
        p1 = Point(1_700_000_000.0, 45_000.0)
        p2 = Point(1_700_086_400.0, 45_500.0)  # 1 day later, +500 points
        p3 = Point(1_700_000_000.0, 46_000.0)  # 1000 points vertical offset

        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        expected_slope = 500.0 / 86_400.0
        assert channel.slope == pytest.approx(expected_slope, rel=1e-9)
        assert channel.vertical_width == pytest.approx(1000.0, rel=1e-9)

        # Midpoint x after 43200 seconds (half day)
        mid_x = 1_700_043_200.0
        low_y, high_y = channel.get_y_bounds(mid_x)
        assert low_y == pytest.approx(45_250.0, rel=1e-7)
        assert high_y == pytest.approx(46_250.0, rel=1e-7)

    def test_negative_coordinates_handling(self):
        """Correct calculation when coordinates are negative."""
        tool = ChannelTool()
        p1 = Point(-50.0, -100.0)
        p2 = Point(-10.0, -20.0)
        p3 = Point(-50.0, -50.0)

        # Slope: (-20 - (-100)) / (-10 - (-50)) = 80 / 40 = 2.0
        channel = tool.create_channel(p1, p2, p3, mode=ChannelMode.PARALLEL)

        assert channel.slope == pytest.approx(2.0)
        # Base line: y - (-100) = 2(x - (-50)) => y = 2x
        assert channel.base_line.intercept == pytest.approx(0.0)
        # Parallel line through (-50, -50): -50 = 2(-50) + b => b = 50
        assert channel.parallel_line.intercept == pytest.approx(50.0)
        assert channel.vertical_width == pytest.approx(50.0)