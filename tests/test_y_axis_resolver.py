"""
Unit tests for the Dynamic Auto-Scaling (Fit-to-Screen) Y-Axis Resolver.

Specifications covered:
Story 1.2.3: Build Dynamic Auto-Scaling (Fit-to-Screen) Y-Axis Resolver
Target modules:
    - src/charting/y_axis_resolver.py
    - src/charting/__init__.py
"""

from decimal import Decimal
import math
from typing import Generator, Sequence

import pytest

from src.charting import YAxisRange, YAxisResolver
from src.charting.y_axis_resolver import (
    YAxisRange as ModuleYAxisRange,
    YAxisResolver as ModuleYAxisResolver,
)


# ============================================================================
# Module Exports & Packaging Tests
# ============================================================================


class TestModuleExports:
    """Verifies module-level public APIs and exports from src.charting."""

    def test_y_axis_resolver_exported_from_package_root(self):
        """YAxisResolver must be exported from src.charting root."""
        assert YAxisResolver is ModuleYAxisResolver

    def test_y_axis_range_exported_from_package_root(self):
        """YAxisRange must be exported from src.charting root."""
        assert YAxisRange is ModuleYAxisRange


# ============================================================================
# YAxisRange Data Structure Tests
# ============================================================================


class TestYAxisRangeDataStructure:
    """Verifies YAxisRange behavior, attributes, and immutability."""

    def test_range_attributes_and_access(self):
        """YAxisRange must expose min, max, and span properties."""
        y_range = YAxisRange(min=10.0, max=50.0)
        assert y_range.min == pytest.approx(10.0)
        assert y_range.max == pytest.approx(50.0)
        assert y_range.span == pytest.approx(40.0)

    def test_range_unpacking(self):
        """YAxisRange must support tuple-like unpacking (min, max)."""
        y_range = YAxisRange(min=-5.0, max=15.0)
        y_min, y_max = y_range
        assert y_min == pytest.approx(-5.0)
        assert y_max == pytest.approx(15.0)

    def test_range_immutability(self):
        """YAxisRange should be immutable to protect resolved coordinates."""
        y_range = YAxisRange(min=0.0, max=100.0)
        with pytest.raises((AttributeError, TypeError)):
            y_range.min = 10.0  # type: ignore[misc]


# ============================================================================
# AC 1: Visible Numeric Data Points with Margin Padding Percentage
# ============================================================================


class TestAutoScalingWithVarianceAndPadding:
    """
    Acceptance Criteria 1:
    Given a set of visible numeric data points and a margin padding percentage,
    accurately calculate the padded fit-to-screen min and max bounds.
    """

    @pytest.fixture
    def resolver(self) -> YAxisResolver:
        """Default resolver with 5% default margin padding."""
        return YAxisResolver(margin_padding_percentage=0.05)

    def test_standard_positive_series(self, resolver: YAxisResolver):
        """
        Points: [100.0, 150.0, 200.0], padding: 0.10 (10%)
        Span = 100.0 -> Margin = 10.0
        Expected range: [90.0, 210.0]
        """
        data = [100.0, 150.0, 200.0]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        assert result.min == pytest.approx(90.0)
        assert result.max == pytest.approx(210.0)
        assert result.span == pytest.approx(120.0)

    def test_negative_numeric_series(self, resolver: YAxisResolver):
        """
        Points: [-200.0, -150.0, -100.0], padding: 0.05 (5%)
        Span = 100.0 -> Margin = 5.0
        Expected range: [-205.0, -95.0]
        """
        data = [-200.0, -150.0, -100.0]
        result = resolver.resolve(data, margin_padding_percentage=0.05)

        assert result.min == pytest.approx(-205.0)
        assert result.max == pytest.approx(-95.0)
        assert result.span == pytest.approx(110.0)

    def test_mixed_zero_crossing_series(self, resolver: YAxisResolver):
        """
        Points: [-50.0, 0.0, 50.0], padding: 0.20 (20%)
        Span = 100.0 -> Margin = 20.0
        Expected range: [-70.0, 70.0]
        """
        data = [-50.0, 0.0, 50.0]
        result = resolver.resolve(data, margin_padding_percentage=0.20)

        assert result.min == pytest.approx(-70.0)
        assert result.max == pytest.approx(70.0)
        assert result.span == pytest.approx(140.0)

    def test_zero_padding_percentage(self, resolver: YAxisResolver):
        """
        With 0% padding, resolved range must tightly hug the min and max data points.
        """
        data = [12.5, 45.2, 98.7, 3.1]
        result = resolver.resolve(data, margin_padding_percentage=0.0)

        assert result.min == pytest.approx(3.1)
        assert result.max == pytest.approx(98.7)
        assert result.span == pytest.approx(98.7 - 3.1)

    def test_uses_instance_default_padding_when_unspecified(self):
        """
        Resolver should fall back to its configured instance default padding
        if not overridden in resolve call.
        """
        custom_resolver = YAxisResolver(margin_padding_percentage=0.15)
        data = [10.0, 30.0]
        # Span = 20.0, 15% margin = 3.0 -> [7.0, 33.0]
        result = custom_resolver.resolve(data)

        assert result.min == pytest.approx(7.0)
        assert result.max == pytest.approx(33.0)

    def test_unordered_data_series(self, resolver: YAxisResolver):
        """Resolver must properly scan extrema regardless of point ordering."""
        unsorted_data = [45.0, 120.0, -10.0, 80.0, 0.0, 95.0]
        result = resolver.resolve(unsorted_data, margin_padding_percentage=0.10)

        # Span = 120 - (-10) = 130.0, margin = 13.0
        assert result.min == pytest.approx(-23.0)
        assert result.max == pytest.approx(133.0)

    @pytest.mark.parametrize(
        "sequence_factory",
        [
            list,
            tuple,
            lambda pts: (p for p in pts),  # Generator
        ],
    )
    def test_various_iterable_containers(
        self, resolver: YAxisResolver, sequence_factory
    ):
        """Resolver must accept lists, tuples, and general iterators."""
        raw_points = [10.0, 20.0, 30.0]
        container = sequence_factory(raw_points)
        result = resolver.resolve(container, margin_padding_percentage=0.10)

        # Span = 20.0, margin = 2.0 -> [8.0, 32.0]
        assert result.min == pytest.approx(8.0)
        assert result.max == pytest.approx(32.0)

    def test_integer_and_decimal_inputs(self, resolver: YAxisResolver):
        """Resolver must gracefully handle integer and Decimal numeric types."""
        data = [Decimal("100.50"), 200, Decimal("300.50")]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        # Span = 200.0, margin = 20.0 -> [80.5, 320.5]
        assert result.min == pytest.approx(80.50)
        assert result.max == pytest.approx(320.50)

    def test_micro_scale_fractional_values(self, resolver: YAxisResolver):
        """Resolver must handle high-precision micro-scale values accurately."""
        data = [0.00010, 0.00015, 0.00020]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        # Span = 0.00010, margin = 0.00001 -> [0.00009, 0.00021]
        assert result.min == pytest.approx(0.00009)
        assert result.max == pytest.approx(0.00021)

    def test_macro_scale_large_values(self, resolver: YAxisResolver):
        """Resolver must handle very large financial/scientific numbers without overflow."""
        data = [1e9, 2e9, 3e9]
        result = resolver.resolve(data, margin_padding_percentage=0.05)

        # Span = 2e9, margin = 1e8 -> [0.9e9, 3.1e9]
        assert result.min == pytest.approx(9e8)
        assert result.max == pytest.approx(3.1e9)


# ============================================================================
# AC 2: Identical Visible Data Points (Zero Variance)
# ============================================================================


class TestAutoScalingZeroVariance:
    """
    Acceptance Criteria 2:
    Given a set of identical visible data points (zero variance),
    the resolver must prevent division-by-zero projection collapses and generate
    a valid, non-zero span surrounding the value.
    """

    @pytest.fixture
    def resolver(self) -> YAxisResolver:
        return YAxisResolver(margin_padding_percentage=0.05)

    def test_multiple_identical_positive_points(self, resolver: YAxisResolver):
        """
        Multiple identical positive values must resolve to bounds strictly
        surrounding the constant value.
        """
        constant_val = 100.0
        data = [constant_val, constant_val, constant_val]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        assert result.min < constant_val < result.max
        assert result.span > 0.0
        # When variance is zero, padding applies symmetrically to the absolute magnitude
        # 100.0 +/- (100.0 * 0.10) -> [90.0, 110.0]
        assert result.min == pytest.approx(90.0)
        assert result.max == pytest.approx(110.0)

    def test_multiple_identical_negative_points(self, resolver: YAxisResolver):
        """
        Multiple identical negative values must resolve to symmetric bounds
        enclosing the value.
        """
        constant_val = -50.0
        data = [-50.0, -50.0]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        assert result.min < constant_val < result.max
        assert result.span > 0.0
        # -50.0 +/- (50.0 * 0.10) -> [-55.0, -45.0]
        assert result.min == pytest.approx(-55.0)
        assert result.max == pytest.approx(-45.0)

    def test_multiple_identical_zero_points(self, resolver: YAxisResolver):
        """
        All zeros have both 0 variance and 0 magnitude. The resolver must
        fall back to an absolute non-zero range centered around 0.0.
        """
        data = [0.0, 0.0, 0.0, 0.0]
        result = resolver.resolve(data, margin_padding_percentage=0.10)

        assert result.min < 0.0 < result.max
        assert result.span > 0.0
        # Symmetric around zero
        assert math.isclose(result.min, -result.max, rel_tol=1e-7, abs_tol=1e-7)

    def test_single_data_point_zero_variance(self, resolver: YAxisResolver):
        """
        A single data point is an edge case of zero variance and must produce
        a renderable non-degenerate span.
        """
        data = [42.0]
        result = resolver.resolve(data, margin_padding_percentage=0.05)

        assert result.min < 42.0 < result.max
        assert result.span > 0.0

    def test_identical_points_with_zero_padding_still_guarantees_span(
        self, resolver: YAxisResolver
    ):
        """
        Even when requested padding is 0.0 on zero-variance data, fit-to-screen
        cannot collapse to min == max (which causes division by zero in UI renderers).
        """
        data = [25.0, 25.0]
        result = resolver.resolve(data, margin_padding_percentage=0.0)

        assert result.min < 25.0 < result.max
        assert result.span > 0.0


# ============================================================================
# AC 3: Empty Sequence of Data Points
# ============================================================================


class TestAutoScalingEmptySequence:
    """
    Acceptance Criteria 3:
    Given an empty sequence of data points, the resolver must reject the input
    and raise a ValueError.
    """

    @pytest.fixture
    def resolver(self) -> YAxisResolver:
        return YAxisResolver()

    def test_empty_list_raises_value_error(self, resolver: YAxisResolver):
        """Resolving empty list must raise ValueError directly."""
        with pytest.raises(ValueError):
            resolver.resolve([])

    def test_empty_tuple_raises_value_error(self, resolver: YAxisResolver):
        """Resolving empty tuple must raise ValueError directly."""
        with pytest.raises(ValueError):
            resolver.resolve(())

    def test_empty_generator_raises_value_error(self, resolver: YAxisResolver):
        """Resolving exhausted/empty generator must raise ValueError directly."""
        def empty_gen() -> Generator[float, None, None]:
            return
            yield  # unreachable

        with pytest.raises(ValueError):
            resolver.resolve(empty_gen())


# ============================================================================
# Robustness, Edge Cases, and Parameter Validation
# ============================================================================


class TestResolverParameterValidation:
    """Ensures improper parameters and non-numeric inputs are safely handled."""

    @pytest.fixture
    def resolver(self) -> YAxisResolver:
        return YAxisResolver()

    @pytest.mark.parametrize("invalid_padding", [-0.01, -0.5, -10.0])
    def test_negative_margin_padding_in_init_raises_value_error(
        self, invalid_padding: float
    ):
        """Negative margin padding percentage in constructor must raise ValueError."""
        with pytest.raises(ValueError):
            YAxisResolver(margin_padding_percentage=invalid_padding)

    @pytest.mark.parametrize("invalid_padding", [-0.01, -0.5, -10.0])
    def test_negative_margin_padding_in_resolve_raises_value_error(
        self, resolver: YAxisResolver, invalid_padding: float
    ):
        """Negative margin padding override in resolve must raise ValueError."""
        with pytest.raises(ValueError):
            resolver.resolve([10.0, 20.0], margin_padding_percentage=invalid_padding)

    def test_non_numeric_elements_raise_type_or_value_error(
        self, resolver: YAxisResolver
    ):
        """Series containing strings or None must be rejected."""
        with pytest.raises((TypeError, ValueError)):
            resolver.resolve([10.0, "invalid", 30.0])  # type: ignore[list-item]

        with pytest.raises((TypeError, ValueError)):
            resolver.resolve([10.0, None, 30.0])  # type: ignore[list-item]

    def test_nan_values_raise_value_error(self, resolver: YAxisResolver):
        """NaN values cannot form a deterministic bounding box."""
        with pytest.raises(ValueError):
            resolver.resolve([10.0, float("nan"), 30.0])

    def test_infinite_values_raise_value_error(self, resolver: YAxisResolver):
        """Infinite values cannot be bounded for screen rendering."""
        with pytest.raises(ValueError):
            resolver.resolve([10.0, float("inf"), 30.0])

        with pytest.raises(ValueError):
            resolver.resolve([float("-inf"), 0.0, 10.0])