"""
Unit tests for Linear, Logarithmic, and Percentage Coordinate Converters.

Story: 1.2.1: Construct Linear, Logarithmic, and Percentage Coordinate Converters
Target Modules:
    - src/chart/converters.py
    - src/chart/__init__.py
"""

import math
import pytest

from src.chart.converters import (
    LinearCoordinateConverter,
    LogarithmicCoordinateConverter,
    PercentageCoordinateConverter,
)
import src.chart


# ==============================================================================
# Module Export & Packaging Tests
# ==============================================================================

class TestPackageExports:
    """Verifies converters are exposed through both converters module and top-level chart package."""

    def test_top_level_package_exports_converters(self):
        """Converters must be re-exported at the top-level src.chart namespace."""
        assert hasattr(src.chart, "LinearCoordinateConverter")
        assert hasattr(src.chart, "LogarithmicCoordinateConverter")
        assert hasattr(src.chart, "PercentageCoordinateConverter")

    def test_top_level_all_contains_converters(self):
        """The __all__ definition in src.chart must declare all converter classes."""
        assert hasattr(src.chart, "__all__")
        expected_converters = {
            "LinearCoordinateConverter",
            "LogarithmicCoordinateConverter",
            "PercentageCoordinateConverter",
        }
        assert expected_converters.issubset(set(src.chart.__all__))


# ==============================================================================
# LinearCoordinateConverter Tests
# ==============================================================================

class TestLinearCoordinateConverter:
    """Tests for LinearCoordinateConverter adhering to AC1, AC5, and boundary constraints."""

    def test_linear_transformation_acceptance_criteria(self):
        """
        AC: Given a linear coordinate converter configured for input domain [0, 100]
        and output range [0, 500], When a value of 50 is transformed,
        Then the output is 250.0.
        """
        converter = LinearCoordinateConverter(domain=(0, 100), range=(0, 500))
        result = converter.transform(50)
        assert result == pytest.approx(250.0)

    @pytest.mark.parametrize(
        ("input_val", "expected_output"),
        [
            (0, 0.0),
            (25, 125.0),
            (50, 250.0),
            (75, 375.0),
            (100, 500.0),
        ],
    )
    def test_linear_transform_within_domain(self, input_val, expected_output):
        converter = LinearCoordinateConverter(domain=(0, 100), range=(0, 500))
        assert converter.transform(input_val) == pytest.approx(expected_output)

    def test_linear_inverse_transformation_acceptance_criteria(self):
        """AC: Inverse transformation reconstructs the original input value."""
        converter = LinearCoordinateConverter(domain=(0, 100), range=(0, 500))
        assert converter.inverse_transform(250.0) == pytest.approx(50.0)
        assert converter.inverse_transform(0.0) == pytest.approx(0.0)
        assert converter.inverse_transform(500.0) == pytest.approx(100.0)

    def test_linear_inverted_output_range(self):
        """Screen coordinate systems often invert Y (top=0, bottom=height)."""
        converter = LinearCoordinateConverter(domain=(0, 100), range=(500, 0))
        assert converter.transform(0) == pytest.approx(500.0)
        assert converter.transform(50) == pytest.approx(250.0)
        assert converter.transform(100) == pytest.approx(0.0)
        assert converter.inverse_transform(250.0) == pytest.approx(50.0)

    def test_linear_negative_domain(self):
        """Converter should correctly handle domains spanning negative to positive."""
        converter = LinearCoordinateConverter(domain=(-50, 50), range=(0, 200))
        assert converter.transform(-50) == pytest.approx(0.0)
        assert converter.transform(0) == pytest.approx(100.0)
        assert converter.transform(50) == pytest.approx(200.0)
        assert converter.inverse_transform(100.0) == pytest.approx(0.0)

    def test_linear_degenerate_domain_raises_error(self):
        """A domain with identical start and end values cannot be interpolated."""
        with pytest.raises(ValueError):
            LinearCoordinateConverter(domain=(50, 50), range=(0, 100))


# ==============================================================================
# LogarithmicCoordinateConverter Tests
# ==============================================================================

class TestLogarithmicCoordinateConverter:
    """Tests for LogarithmicCoordinateConverter adhering to AC2, AC3, AC5, and boundaries."""

    def test_logarithmic_transformation_acceptance_criteria(self):
        """
        AC: Given a logarithmic coordinate converter configured for positive domain [1, 1000]
        and output range [0, 300], When a value of 10 is transformed,
        Then the output evaluates to 100.0.
        """
        converter = LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300))
        result = converter.transform(10)
        assert result == pytest.approx(100.0)

    @pytest.mark.parametrize(
        ("input_val", "expected_output"),
        [
            (1, 0.0),
            (10, 100.0),
            (100, 200.0),
            (1000, 300.0),
        ],
    )
    def test_logarithmic_transform_decade_steps(self, input_val, expected_output):
        converter = LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300))
        assert converter.transform(input_val) == pytest.approx(expected_output)

    @pytest.mark.parametrize("invalid_input", [0, 0.0, -0.0001, -1.0, -100.0])
    def test_logarithmic_non_positive_input_raises_value_error(self, invalid_input):
        """
        AC: Given a logarithmic coordinate converter, When an input value less than
        or equal to zero is provided, Then a ValueError is raised.
        """
        converter = LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300))
        with pytest.raises(ValueError):
            converter.transform(invalid_input)

    @pytest.mark.parametrize(
        "invalid_domain",
        [
            (0, 100),
            (-10, 100),
            (-100, -10),
            (0, 0),
        ],
    )
    def test_logarithmic_non_positive_domain_configuration_raises_value_error(
        self, invalid_domain
    ):
        """Domain configuration containing values <= 0 must fail construction."""
        with pytest.raises(ValueError):
            LogarithmicCoordinateConverter(domain=invalid_domain, range=(0, 300))

    def test_logarithmic_degenerate_domain_raises_value_error(self):
        """Positive domain where min == max must raise ValueError."""
        with pytest.raises(ValueError):
            LogarithmicCoordinateConverter(domain=(10, 10), range=(0, 300))

    def test_logarithmic_inverse_transformation_acceptance_criteria(self):
        """AC: Inverse transformation accurately reconstructs original input values."""
        converter = LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300))
        assert converter.inverse_transform(0.0) == pytest.approx(1.0)
        assert converter.inverse_transform(100.0) == pytest.approx(10.0)
        assert converter.inverse_transform(200.0) == pytest.approx(100.0)
        assert converter.inverse_transform(300.0) == pytest.approx(1000.0)

    def test_logarithmic_inverted_output_range(self):
        """Logarithmic scale mapped to inverted display range."""
        converter = LogarithmicCoordinateConverter(domain=(1, 1000), range=(300, 0))
        assert converter.transform(1) == pytest.approx(300.0)
        assert converter.transform(10) == pytest.approx(200.0)
        assert converter.transform(1000) == pytest.approx(0.0)
        assert converter.inverse_transform(200.0) == pytest.approx(10.0)


# ==============================================================================
# PercentageCoordinateConverter Tests
# ==============================================================================

class TestPercentageCoordinateConverter:
    """Tests for PercentageCoordinateConverter adhering to AC4, AC5, and boundaries."""

    def test_percentage_transformation_acceptance_criteria(self):
        """
        AC: Given a percentage coordinate converter configured with a reference base value of 200
        and domain range [-50%, +50%] mapped to [0, 100], When an input value of 250 (+25%)
        is transformed, Then the output is 75.0.
        """
        converter = PercentageCoordinateConverter(
            base_value=200, domain=(-50, 50), range=(0, 100)
        )
        result = converter.transform(250)
        assert result == pytest.approx(75.0)

    @pytest.mark.parametrize(
        ("input_val", "expected_output"),
        [
            (100, 0.0),    # -50% change from 200 -> output min 0.0
            (150, 25.0),   # -25% change from 200
            (200, 50.0),   # 0% change from 200 -> output midpoint 50.0
            (250, 75.0),   # +25% change from 200
            (300, 100.0),  # +50% change from 200 -> output max 100.0
        ],
    )
    def test_percentage_transform_key_values(self, input_val, expected_output):
        converter = PercentageCoordinateConverter(
            base_value=200, domain=(-50, 50), range=(0, 100)
        )
        assert converter.transform(input_val) == pytest.approx(expected_output)

    def test_percentage_inverse_transformation_acceptance_criteria(self):
        """AC: Inverse transformation accurately reconstructs original input values."""
        converter = PercentageCoordinateConverter(
            base_value=200, domain=(-50, 50), range=(0, 100)
        )
        assert converter.inverse_transform(75.0) == pytest.approx(250.0)
        assert converter.inverse_transform(50.0) == pytest.approx(200.0)
        assert converter.inverse_transform(0.0) == pytest.approx(100.0)
        assert converter.inverse_transform(100.0) == pytest.approx(300.0)

    @pytest.mark.parametrize("invalid_base", [0, 0.0, -10.0, -200])
    def test_percentage_zero_or_negative_base_raises_value_error(self, invalid_base):
        """Base value cannot be zero (division by zero) or negative."""
        with pytest.raises(ValueError):
            PercentageCoordinateConverter(
                base_value=invalid_base, domain=(-50, 50), range=(0, 100)
            )

    def test_percentage_degenerate_domain_raises_value_error(self):
        """Domain with equal percentage limits must raise ValueError."""
        with pytest.raises(ValueError):
            PercentageCoordinateConverter(
                base_value=200, domain=(50, 50), range=(0, 100)
            )

    def test_percentage_inverted_output_range(self):
        """Percentage converter mapped to screen-inverted coordinates."""
        converter = PercentageCoordinateConverter(
            base_value=200, domain=(-50, 50), range=(100, 0)
        )
        assert converter.transform(100) == pytest.approx(100.0)
        assert converter.transform(200) == pytest.approx(50.0)
        assert converter.transform(250) == pytest.approx(25.0)
        assert converter.transform(300) == pytest.approx(0.0)
        assert converter.inverse_transform(25.0) == pytest.approx(250.0)


# ==============================================================================
# Universal Invertibility & Round-Trip Tests (AC5)
# ==============================================================================

class TestConverterRoundTripInvertibility:
    """
    AC5: Given any instantiated converter (linear, logarithmic, or percentage),
    When the inverse transformation is called on a valid output coordinate,
    Then it accurately reconstructs the original input value.
    """

    @pytest.mark.parametrize(
        ("converter", "test_values"),
        [
            (
                LinearCoordinateConverter(domain=(0, 100), range=(0, 500)),
                [0.0, 1.25, 10.0, 50.0, 78.4, 100.0],
            ),
            (
                LinearCoordinateConverter(domain=(-1000, 1000), range=(800, -800)),
                [-1000.0, -250.0, 0.0, 12.34, 500.0, 1000.0],
            ),
            (
                LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300)),
                [1.0, 2.5, 10.0, 45.6, 100.0, 500.0, 1000.0],
            ),
            (
                LogarithmicCoordinateConverter(domain=(0.01, 100.0), range=(50, 450)),
                [0.01, 0.1, 1.0, 15.0, 75.25, 100.0],
            ),
            (
                PercentageCoordinateConverter(
                    base_value=200, domain=(-50, 50), range=(0, 100)
                ),
                [100.0, 120.0, 180.0, 200.0, 250.0, 299.9, 300.0],
            ),
            (
                PercentageCoordinateConverter(
                    base_value=50, domain=(-100, 200), range=(600, 0)
                ),
                [1.0, 25.0, 50.0, 75.0, 100.0, 150.0],
            ),
        ],
    )
    def test_forward_then_inverse_reconstructs_original_input(
        self, converter, test_values
    ):
        """x -> transform(x) -> inverse_transform(y) must equal x."""
        for orig_val in test_values:
            coord = converter.transform(orig_val)
            reconstructed = converter.inverse_transform(coord)
            assert reconstructed == pytest.approx(orig_val, rel=1e-7, abs=1e-7)

    @pytest.mark.parametrize(
        ("converter", "output_coords"),
        [
            (
                LinearCoordinateConverter(domain=(0, 100), range=(0, 500)),
                [0.0, 125.0, 250.0, 375.0, 500.0],
            ),
            (
                LogarithmicCoordinateConverter(domain=(1, 1000), range=(0, 300)),
                [0.0, 50.0, 100.0, 200.0, 300.0],
            ),
            (
                PercentageCoordinateConverter(
                    base_value=200, domain=(-50, 50), range=(0, 100)
                ),
                [0.0, 25.0, 50.0, 75.0, 100.0],
            ),
        ],
    )
    def test_inverse_then_forward_reconstructs_output_coord(
        self, converter, output_coords
    ):
        """y -> inverse_transform(y) -> transform(x) must equal y."""
        for orig_coord in output_coords:
            val = converter.inverse_transform(orig_coord)
            reconstructed_coord = converter.transform(val)
            assert reconstructed_coord == pytest.approx(
                orig_coord, rel=1e-7, abs=1e-7
            )