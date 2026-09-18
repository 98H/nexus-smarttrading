"""
Unit tests for Andrews' Pitchfork, Schiff, and Modified Schiff System.
Specification: Story 5.3.2: Build Andrews' Pitchfork, Schiff, and Modified Schiff System.
Target Modules:
    - src/indicators/__init__.py
    - src/indicators/pitchfork.py
"""

import math
import pytest

from src.indicators import (
    Point as IndicatorsPoint,
    Line as IndicatorsLine,
    PitchforkType as IndicatorsPitchforkType,
    Pitchfork as IndicatorsPitchfork,
    calculate_pitchfork as indicators_calculate_pitchfork,
    andrews_pitchfork as indicators_andrews_pitchfork,
    schiff_pitchfork as indicators_schiff_pitchfork,
    modified_schiff_pitchfork as indicators_modified_schiff_pitchfork,
)
from src.indicators.pitchfork import (
    Point,
    Line,
    PitchforkType,
    Pitchfork,
    calculate_pitchfork,
    andrews_pitchfork,
    schiff_pitchfork,
    modified_schiff_pitchfork,
)


# ============================================================================
# Package Export & Interface Alignment Tests
# ============================================================================


class TestPackageExports:
    """Verify indicators package correctly re-exports Pitchfork system components."""

    def test_indicators_package_re_exports(self):
        assert IndicatorsPoint is Point
        assert IndicatorsLine is Line
        assert IndicatorsPitchforkType is PitchforkType
        assert IndicatorsPitchfork is Pitchfork
        assert indicators_calculate_pitchfork is calculate_pitchfork
        assert indicators_andrews_pitchfork is andrews_pitchfork
        assert indicators_schiff_pitchfork is schiff_pitchfork
        assert indicators_modified_schiff_pitchfork is modified_schiff_pitchfork

    def test_pitchfork_type_enum_members(self):
        assert hasattr(PitchforkType, "STANDARD")
        assert hasattr(PitchforkType, "SCHIFF")
        assert hasattr(PitchforkType, "MODIFIED_SCHIFF")


# ============================================================================
# Line & Point Data Structures Tests
# ============================================================================


class TestGeometryStructures:
    """Verify Point and Line geometric evaluation and properties."""

    def test_point_creation_and_fields(self):
        pt = Point(10.5, 250.75)
        assert pt.x == 10.5
        assert pt.y == 250.75

    def test_line_evaluation_and_slope(self):
        # Line: y = 2x + 10
        line = Line(slope=2.0, intercept=10.0)
        assert line.slope == 2.0
        assert line.intercept == 10.0
        assert line.evaluate(0.0) == pytest.approx(10.0)
        assert line.evaluate(5.0) == pytest.approx(20.0)
        assert line(5.0) == pytest.approx(20.0)  # Callable interface

    def test_line_passes_through(self):
        line = Line(slope=-1.5, intercept=30.0)
        pt_on_line = Point(10.0, 15.0)
        pt_off_line = Point(10.0, 16.0)

        assert line.passes_through(pt_on_line)
        assert not line.passes_through(pt_off_line)


# ============================================================================
# Standard Andrews' Pitchfork Tests
# ============================================================================


class TestStandardAndrewsPitchfork:
    """
    Acceptance Criteria:
    Given three anchor points (P1, P2, P3) with coordinates (time/index, price),
    When calculating standard Andrews' Pitchfork,
    Then the median line is defined by anchor P1 and the midpoint of P2-P3,
    with upper and lower parallel lines passing through P2 and P3 respectively.
    """

    @pytest.fixture
    def setup_points(self):
        p1 = Point(0.0, 100.0)
        p2 = Point(10.0, 150.0)
        p3 = Point(20.0, 110.0)
        return p1, p2, p3

    def test_standard_origin_is_p1(self, setup_points):
        p1, p2, p3 = setup_points
        pf = andrews_pitchfork(p1, p2, p3)

        assert pf.origin.x == pytest.approx(p1.x)
        assert pf.origin.y == pytest.approx(p1.y)

    def test_standard_midpoint_p2_p3(self, setup_points):
        p1, p2, p3 = setup_points
        pf = andrews_pitchfork(p1, p2, p3)

        expected_midpoint_x = (p2.x + p3.x) / 2.0  # 15.0
        expected_midpoint_y = (p2.y + p3.y) / 2.0  # 130.0

        assert pf.midpoint.x == pytest.approx(expected_midpoint_x)
        assert pf.midpoint.y == pytest.approx(expected_midpoint_y)

    def test_standard_median_line_definition(self, setup_points):
        p1, p2, p3 = setup_points
        pf = andrews_pitchfork(p1, p2, p3)

        # Expected slope = (130 - 100) / (15 - 0) = 30 / 15 = 2.0
        expected_slope = 2.0
        expected_intercept = 100.0  # y = 2.0 * 0 + 100

        assert pf.median_line.slope == pytest.approx(expected_slope)
        assert pf.median_line.intercept == pytest.approx(expected_intercept)

        # Median line must pass through P1 and Midpoint(P2, P3)
        assert pf.median_line.evaluate(p1.x) == pytest.approx(p1.y)
        assert pf.median_line.evaluate(pf.midpoint.x) == pytest.approx(pf.midpoint.y)

    def test_standard_parallel_lines_through_p2_and_p3(self, setup_points):
        p1, p2, p3 = setup_points
        pf = andrews_pitchfork(p1, p2, p3)

        # All lines must have identical slopes
        assert pf.upper_line.slope == pytest.approx(pf.median_line.slope)
        assert pf.lower_line.slope == pytest.approx(pf.median_line.slope)

        # Upper line passes through P2
        assert pf.upper_line.evaluate(p2.x) == pytest.approx(p2.y)
        expected_upper_intercept = p2.y - (pf.median_line.slope * p2.x)  # 150 - 20 = 130
        assert pf.upper_line.intercept == pytest.approx(expected_upper_intercept)

        # Lower line passes through P3
        assert pf.lower_line.evaluate(p3.x) == pytest.approx(p3.y)
        expected_lower_intercept = p3.y - (pf.median_line.slope * p3.x)  # 110 - 40 = 70
        assert pf.lower_line.intercept == pytest.approx(expected_lower_intercept)

    def test_standard_pitchfork_with_tuple_inputs(self):
        p1 = (1.0, 5.0)
        p2 = (3.0, 15.0)
        p3 = (5.0, 9.0)

        pf = andrews_pitchfork(p1, p2, p3)

        # Midpoint = ((3+5)/2, (15+9)/2) = (4.0, 12.0)
        # Slope = (12 - 5) / (4 - 1) = 7 / 3
        expected_slope = 7.0 / 3.0
        assert pf.median_line.slope == pytest.approx(expected_slope)
        assert pf.upper_line.slope == pytest.approx(expected_slope)
        assert pf.lower_line.slope == pytest.approx(expected_slope)

        assert pf.upper_line.evaluate(3.0) == pytest.approx(15.0)
        assert pf.lower_line.evaluate(5.0) == pytest.approx(9.0)

    def test_calculate_pitchfork_interface_for_standard(self, setup_points):
        p1, p2, p3 = setup_points
        pf = calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.STANDARD)

        assert pf.pitchfork_type == PitchforkType.STANDARD
        assert pf.origin.x == pytest.approx(p1.x)
        assert pf.origin.y == pytest.approx(p1.y)


# ============================================================================
# Schiff Pitchfork Tests
# ============================================================================


class TestSchiffPitchfork:
    """
    Acceptance Criteria:
    Given three anchor points (P1, P2, P3),
    When calculating Schiff Pitchfork,
    Then the origin anchor is adjusted to (P1.x, (P1.y + P2.y) / 2)
    with parallel lines through P2 and P3.
    """

    @pytest.fixture
    def setup_points(self):
        p1 = Point(0.0, 100.0)
        p2 = Point(10.0, 150.0)
        p3 = Point(20.0, 110.0)
        return p1, p2, p3

    def test_schiff_origin_adjustment(self, setup_points):
        p1, p2, p3 = setup_points
        pf = schiff_pitchfork(p1, p2, p3)

        expected_origin_x = p1.x  # 0.0
        expected_origin_y = (p1.y + p2.y) / 2.0  # (100 + 150) / 2 = 125.0

        assert pf.origin.x == pytest.approx(expected_origin_x)
        assert pf.origin.y == pytest.approx(expected_origin_y)

    def test_schiff_median_line_definition(self, setup_points):
        p1, p2, p3 = setup_points
        pf = schiff_pitchfork(p1, p2, p3)

        # Midpoint of P2-P3: (15.0, 130.0)
        # Origin: (0.0, 125.0)
        # Expected slope = (130 - 125) / (15 - 0) = 5 / 15 = 1 / 3
        expected_slope = 5.0 / 15.0
        expected_intercept = 125.0

        assert pf.median_line.slope == pytest.approx(expected_slope)
        assert pf.median_line.intercept == pytest.approx(expected_intercept)

        # Passes through Schiff origin and Midpoint(P2, P3)
        assert pf.median_line.evaluate(pf.origin.x) == pytest.approx(pf.origin.y)
        assert pf.median_line.evaluate(pf.midpoint.x) == pytest.approx(pf.midpoint.y)

    def test_schiff_parallel_lines_through_p2_and_p3(self, setup_points):
        p1, p2, p3 = setup_points
        pf = schiff_pitchfork(p1, p2, p3)

        assert pf.upper_line.slope == pytest.approx(pf.median_line.slope)
        assert pf.lower_line.slope == pytest.approx(pf.median_line.slope)

        # Upper line passes through P2 (10, 150)
        assert pf.upper_line.evaluate(p2.x) == pytest.approx(p2.y)
        expected_upper_intercept = 150.0 - (pf.median_line.slope * 10.0)
        assert pf.upper_line.intercept == pytest.approx(expected_upper_intercept)

        # Lower line passes through P3 (20, 110)
        assert pf.lower_line.evaluate(p3.x) == pytest.approx(p3.y)
        expected_lower_intercept = 110.0 - (pf.median_line.slope * 20.0)
        assert pf.lower_line.intercept == pytest.approx(expected_lower_intercept)

    def test_schiff_zero_slope_scenario(self):
        # Scenario where Schiff median line has horizontal slope (slope = 0)
        p1 = Point(0.0, 10.0)
        p2 = Point(5.0, 30.0)
        p3 = Point(10.0, 10.0)

        pf = schiff_pitchfork(p1, p2, p3)

        # Origin = (0.0, (10 + 30) / 2) = (0.0, 20.0)
        # Midpoint = ((5 + 10) / 2, (30 + 10) / 2) = (7.5, 20.0)
        # Slope = (20.0 - 20.0) / (7.5 - 0.0) = 0.0
        assert pf.origin.x == pytest.approx(0.0)
        assert pf.origin.y == pytest.approx(20.0)
        assert pf.median_line.slope == pytest.approx(0.0)
        assert pf.median_line.intercept == pytest.approx(20.0)

        # Upper line at y = 30.0, lower at y = 10.0
        assert pf.upper_line.evaluate(5.0) == pytest.approx(30.0)
        assert pf.upper_line.intercept == pytest.approx(30.0)
        assert pf.lower_line.evaluate(10.0) == pytest.approx(10.0)
        assert pf.lower_line.intercept == pytest.approx(10.0)

    def test_calculate_pitchfork_interface_for_schiff(self, setup_points):
        p1, p2, p3 = setup_points
        pf = calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.SCHIFF)

        assert pf.pitchfork_type == PitchforkType.SCHIFF
        assert pf.origin.x == pytest.approx(p1.x)
        assert pf.origin.y == pytest.approx((p1.y + p2.y) / 2.0)


# ============================================================================
# Modified Schiff Pitchfork Tests
# ============================================================================


class TestModifiedSchiffPitchfork:
    """
    Acceptance Criteria:
    Given three anchor points (P1, P2, P3),
    When calculating Modified Schiff Pitchfork,
    Then the origin anchor is adjusted to the midpoint of P1 and P2
    ((P1.x + P2.x) / 2, (P1.y + P2.y) / 2) with parallel lines through P2 and P3.
    """

    @pytest.fixture
    def setup_points(self):
        p1 = Point(0.0, 100.0)
        p2 = Point(10.0, 150.0)
        p3 = Point(20.0, 110.0)
        return p1, p2, p3

    def test_modified_schiff_origin_adjustment(self, setup_points):
        p1, p2, p3 = setup_points
        pf = modified_schiff_pitchfork(p1, p2, p3)

        expected_origin_x = (p1.x + p2.x) / 2.0  # (0 + 10) / 2 = 5.0
        expected_origin_y = (p1.y + p2.y) / 2.0  # (100 + 150) / 2 = 125.0

        assert pf.origin.x == pytest.approx(expected_origin_x)
        assert pf.origin.y == pytest.approx(expected_origin_y)

    def test_modified_schiff_median_line_definition(self, setup_points):
        p1, p2, p3 = setup_points
        pf = modified_schiff_pitchfork(p1, p2, p3)

        # Midpoint of P2-P3: (15.0, 130.0)
        # Origin: (5.0, 125.0)
        # Expected slope = (130 - 125) / (15 - 5) = 5 / 10 = 0.5
        expected_slope = 0.5
        expected_intercept = 125.0 - (0.5 * 5.0)  # 122.5

        assert pf.median_line.slope == pytest.approx(expected_slope)
        assert pf.median_line.intercept == pytest.approx(expected_intercept)

        # Passes through Modified Schiff origin and Midpoint(P2, P3)
        assert pf.median_line.evaluate(pf.origin.x) == pytest.approx(pf.origin.y)
        assert pf.median_line.evaluate(pf.midpoint.x) == pytest.approx(pf.midpoint.y)

    def test_modified_schiff_parallel_lines_through_p2_and_p3(self, setup_points):
        p1, p2, p3 = setup_points
        pf = modified_schiff_pitchfork(p1, p2, p3)

        assert pf.upper_line.slope == pytest.approx(pf.median_line.slope)
        assert pf.lower_line.slope == pytest.approx(pf.median_line.slope)

        # Upper line passes through P2 (10, 150)
        assert pf.upper_line.evaluate(p2.x) == pytest.approx(p2.y)
        expected_upper_intercept = 150.0 - (0.5 * 10.0)  # 145.0
        assert pf.upper_line.intercept == pytest.approx(expected_upper_intercept)

        # Lower line passes through P3 (20, 110)
        assert pf.lower_line.evaluate(p3.x) == pytest.approx(p3.y)
        expected_lower_intercept = 110.0 - (0.5 * 20.0)  # 100.0
        assert pf.lower_line.intercept == pytest.approx(expected_lower_intercept)

    def test_modified_schiff_numerical_example(self):
        p1 = Point(1.0, 5.0)
        p2 = Point(3.0, 15.0)
        p3 = Point(5.0, 9.0)

        pf = modified_schiff_pitchfork(p1, p2, p3)

        # Origin = ((1 + 3) / 2, (5 + 15) / 2) = (2.0, 10.0)
        # Midpoint P2-P3 = ((3 + 5) / 2, (15 + 9) / 2) = (4.0, 12.0)
        # Slope = (12 - 10) / (4 - 2) = 2 / 2 = 1.0
        assert pf.origin.x == pytest.approx(2.0)
        assert pf.origin.y == pytest.approx(10.0)
        assert pf.median_line.slope == pytest.approx(1.0)
        assert pf.median_line.intercept == pytest.approx(8.0)

        # Upper line through (3, 15): y = 1.0 * x + 12
        assert pf.upper_line.evaluate(3.0) == pytest.approx(15.0)
        assert pf.upper_line.intercept == pytest.approx(12.0)

        # Lower line through (5, 9): y = 1.0 * x + 4
        assert pf.lower_line.evaluate(5.0) == pytest.approx(9.0)
        assert pf.lower_line.intercept == pytest.approx(4.0)

    def test_calculate_pitchfork_interface_for_modified_schiff(self, setup_points):
        p1, p2, p3 = setup_points
        pf = calculate_pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.MODIFIED_SCHIFF)

        assert pf.pitchfork_type == PitchforkType.MODIFIED_SCHIFF
        assert pf.origin.x == pytest.approx((p1.x + p2.x) / 2.0)
        assert pf.origin.y == pytest.approx((p1.y + p2.y) / 2.0)


# ============================================================================
# Parametric & Edge Case Tests
# ============================================================================


class TestPitchforkEdgeCasesAndValidation:
    """Validate robustness against vertical lines, invalid types, and edge inputs."""

    def test_vertical_median_line_standard_raises_error(self):
        # Origin P1.x == Midpoint(P2, P3).x
        # P1 = (10, 50), P2 = (5, 100), P3 = (15, 80)
        # Midpoint x = (5 + 15) / 2 = 10.0 -> vertical median line
        p1 = Point(10.0, 50.0)
        p2 = Point(5.0, 100.0)
        p3 = Point(15.0, 80.0)

        with pytest.raises(ValueError):
            andrews_pitchfork(p1, p2, p3)

    def test_vertical_median_line_schiff_raises_error(self):
        # Schiff Origin.x == P1.x == 10.0
        # Midpoint x = (5 + 15) / 2 = 10.0
        p1 = Point(10.0, 50.0)
        p2 = Point(5.0, 100.0)
        p3 = Point(15.0, 80.0)

        with pytest.raises(ValueError):
            schiff_pitchfork(p1, p2, p3)

    def test_vertical_median_line_modified_schiff_raises_error(self):
        # Mod Schiff Origin.x == (P1.x + P2.x) / 2
        # Midpoint x == (P2.x + P3.x) / 2
        # When P1.x == P3.x, Origin.x == Midpoint.x -> vertical line
        p1 = Point(10.0, 50.0)
        p2 = Point(20.0, 100.0)
        p3 = Point(10.0, 80.0)

        with pytest.raises(ValueError):
            modified_schiff_pitchfork(p1, p2, p3)

    def test_invalid_pitchfork_type_raises_error(self):
        p1 = (0, 10)
        p2 = (5, 30)
        p3 = (10, 20)

        with pytest.raises(ValueError):
            calculate_pitchfork(p1, p2, p3, pitchfork_type="INVALID_TYPE")

    @pytest.mark.parametrize(
        "invalid_point",
        [
            (1.0,),  # Missing y
            (1.0, 2.0, 3.0),  # Too many elements
            "invalid_coordinate",
            None,
        ],
    )
    def test_malformed_points_raise_value_or_type_error(self, invalid_point):
        valid_pt = Point(5.0, 10.0)
        with pytest.raises((ValueError, TypeError)):
            andrews_pitchfork(invalid_point, valid_pt, valid_pt)

    def test_pitchfork_with_negative_and_floating_coordinates(self):
        p1 = Point(-10.5, -20.25)
        p2 = Point(-5.25, 45.5)
        p3 = Point(2.75, -15.0)

        pf = andrews_pitchfork(p1, p2, p3)

        # Midpoint = ((-5.25 + 2.75)/2, (45.5 - 15.0)/2) = (-1.25, 15.25)
        expected_mid_x = -1.25
        expected_mid_y = 15.25
        assert pf.midpoint.x == pytest.approx(expected_mid_x)
        assert pf.midpoint.y == pytest.approx(expected_mid_y)

        # Slope = (15.25 - (-20.25)) / (-1.25 - (-10.5)) = 35.5 / 9.25
        expected_slope = 35.5 / 9.25
        assert pf.median_line.slope == pytest.approx(expected_slope)
        assert pf.upper_line.passes_through(p2)
        assert pf.lower_line.passes_through(p3)

    def test_pitchfork_class_direct_instantiation(self):
        p1 = Point(0, 0)
        p2 = Point(2, 4)
        p3 = Point(4, 2)

        pf = Pitchfork(p1, p2, p3, pitchfork_type=PitchforkType.STANDARD)
        assert isinstance(pf.median_line, Line)
        assert isinstance(pf.upper_line, Line)
        assert isinstance(pf.lower_line, Line)
        assert pf.pitchfork_type == PitchforkType.STANDARD