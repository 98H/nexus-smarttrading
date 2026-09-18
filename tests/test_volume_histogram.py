"""
Unit tests for the Dynamic Color-Graded Volume Histogram Primitive.

Specification:
Feature: Implement Dynamic Color-Graded Volume Histogram Primitive
Story: 1.3.3: Implement Dynamic Color-Graded Volume Histogram Primitive
Target Modules:
    - src/visualization/primitives/volume_histogram.py
    - src/visualization/primitives/__init__.py
"""

from typing import Any, List
import math
import pytest

from src.visualization.primitives import (
    ColorThreshold,
    ColorThresholdMap,
    VolumeDataPoint,
    VolumeHistogramBar,
    VolumeHistogramPrimitive,
    VolumeHistogramRepresentation,
)
import src.visualization.primitives as primitives_module


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_threshold_map() -> ColorThresholdMap:
    """Provides a standard default color threshold configuration."""
    return ColorThresholdMap(
        bull_color="#26A69A",
        bear_color="#EF5350",
        neutral_color="#787B86",
    )


@pytest.fixture
def intensity_threshold_map() -> ColorThresholdMap:
    """
    Provides a tiered color-grading threshold map:
    - High Bull (delta >= 5.0) -> #00E676
    - Moderate Bull (0.0 < delta < 5.0) -> #26A69A
    - Neutral (delta == 0.0) -> #787B86
    - Moderate Bear (-5.0 < delta < 0.0) -> #EF5350
    - High Bear (delta <= -5.0) -> #D50000
    """
    return ColorThresholdMap(
        thresholds=[
            ColorThreshold(label="high_bull", min_delta=5.0, max_delta=float("inf"), color="#00E676"),
            ColorThreshold(label="bull", min_delta=0.0001, max_delta=5.0, color="#26A69A"),
            ColorThreshold(label="neutral", min_delta=0.0, max_delta=0.0, color="#787B86"),
            ColorThreshold(label="bear", min_delta=-5.0, max_delta=-0.0001, color="#EF5350"),
            ColorThreshold(label="high_bear", min_delta=-float("inf"), max_delta=-5.0, color="#D50000"),
        ],
        default_color="#787B86",
    )


@pytest.fixture
def primitive(default_threshold_map: ColorThresholdMap) -> VolumeHistogramPrimitive:
    """Provides a configured VolumeHistogramPrimitive instance."""
    return VolumeHistogramPrimitive(color_map=default_threshold_map)


@pytest.fixture
def standard_data_series() -> List[VolumeDataPoint]:
    """Provides a deterministic series of mixed bull, bear, and neutral points."""
    return [
        VolumeDataPoint(timestamp=1700000000, volume=1500.0, price_delta=2.5),    # Bull
        VolumeDataPoint(timestamp=1700000060, volume=800.0, price_delta=-1.2),    # Bear
        VolumeDataPoint(timestamp=1700000120, volume=1200.0, price_delta=0.0),    # Neutral
        VolumeDataPoint(timestamp=1700000180, volume=3400.0, price_delta=10.0),   # Bull
        VolumeDataPoint(timestamp=1700000240, volume=2100.0, price_delta=-7.5),   # Bear
    ]


# =====================================================================
# Test Suite 1: Module Packaging and Exports
# =====================================================================

class TestPrimitivesExports:
    """Verifies that all required primitive interfaces are cleanly exported."""

    def test_primitives_init_exports(self) -> None:
        """Ensure public API symbols are exposed at package root."""
        expected_symbols = [
            "VolumeHistogramPrimitive",
            "VolumeDataPoint",
            "VolumeHistogramBar",
            "VolumeHistogramRepresentation",
            "ColorThreshold",
            "ColorThresholdMap",
        ]
        for symbol in expected_symbols:
            assert hasattr(primitives_module, symbol), f"Symbol '{symbol}' missing from primitives __init__"

    def test_all_declaration_contains_symbols(self) -> None:
        """Verify __all__ accurately lists public interface primitives."""
        assert hasattr(primitives_module, "__all__")
        for symbol in [
            "VolumeHistogramPrimitive",
            "VolumeDataPoint",
            "VolumeHistogramBar",
            "VolumeHistogramRepresentation",
            "ColorThreshold",
            "ColorThresholdMap",
        ]:
            assert symbol in primitives_module.__all__


# =====================================================================
# Test Suite 2: AC 1 - Dynamic Color-Graded Bar Generation
# =====================================================================

class TestDynamicColorGradedHistogramGeneration:
    """
    Acceptance Criteria:
    - Given a series of volume data points with associated price delta values
    - When the volume histogram primitive generates bar elements
    - Then each bar's color is dynamically computed based on the configured
      color-grading threshold map (e.g., bull, bear, neutral intensity)
    """

    def test_generates_correct_bar_count(
        self, primitive: VolumeHistogramPrimitive, standard_data_series: List[VolumeDataPoint]
    ) -> None:
        """Output representation must preserve the input count."""
        result = primitive.process(standard_data_series)
        assert isinstance(result, VolumeHistogramRepresentation)
        assert len(result) == len(standard_data_series)
        assert not result.is_empty

    def test_bar_fields_mapping(
        self, primitive: VolumeHistogramPrimitive, standard_data_series: List[VolumeDataPoint]
    ) -> None:
        """Output bars must maintain coordinates, volume values, and computed colors."""
        result = primitive.process(standard_data_series)
        for original, bar in zip(standard_data_series, result.bars):
            assert isinstance(bar, VolumeHistogramBar)
            assert bar.timestamp == original.timestamp
            assert math.isclose(bar.volume, original.volume, rel_tol=1e-9)
            assert bar.color in ["#26A69A", "#EF5350", "#787B86"]

    def test_basic_bull_bear_neutral_color_assignment(
        self, primitive: VolumeHistogramPrimitive, default_threshold_map: ColorThresholdMap
    ) -> None:
        """Verify delta signs accurately map to standard bull, bear, and neutral hex colors."""
        series = [
            VolumeDataPoint(timestamp=1, volume=100.0, price_delta=0.5),   # Bull
            VolumeDataPoint(timestamp=2, volume=200.0, price_delta=-0.5),  # Bear
            VolumeDataPoint(timestamp=3, volume=300.0, price_delta=0.0),   # Neutral
        ]
        result = primitive.process(series)

        assert result.bars[0].color == default_threshold_map.bull_color
        assert result.bars[1].color == default_threshold_map.bear_color
        assert result.bars[2].color == default_threshold_map.neutral_color

    def test_intensity_tier_color_assignment(
        self, intensity_threshold_map: ColorThresholdMap
    ) -> None:
        """Dynamic color grading must reflect delta magnitude intensity."""
        primitive_intensity = VolumeHistogramPrimitive(color_map=intensity_threshold_map)
        series = [
            VolumeDataPoint(timestamp=101, volume=1000.0, price_delta=8.0),   # High Bull
            VolumeDataPoint(timestamp=102, volume=1000.0, price_delta=1.5),   # Moderate Bull
            VolumeDataPoint(timestamp=103, volume=1000.0, price_delta=0.0),   # Neutral
            VolumeDataPoint(timestamp=104, volume=1000.0, price_delta=-2.5),  # Moderate Bear
            VolumeDataPoint(timestamp=105, volume=1000.0, price_delta=-6.2),  # High Bear
        ]

        result = primitive_intensity.process(series)
        assert len(result) == 5

        assert result.bars[0].color == "#00E676"  # high_bull
        assert result.bars[1].color == "#26A69A"  # bull
        assert result.bars[2].color == "#787B86"  # neutral
        assert result.bars[3].color == "#EF5350"  # bear
        assert result.bars[4].color == "#D50000"  # high_bear

    @pytest.mark.parametrize(
        "delta,expected_color",
        [
            (5.0, "#00E676"),     # Boundary: exactly hits min_delta of high_bull
            (4.9999, "#26A69A"),  # Just under high_bull -> moderate bull
            (0.0001, "#26A69A"),  # Boundary: lower threshold of bull
            (0.0, "#787B86"),     # Neutral exact
            (-0.0001, "#EF5350"), # Boundary: upper threshold of bear
            (-4.9999, "#EF5350"), # Just above high_bear -> moderate bear
            (-5.0, "#D50000"),    # Boundary: hits max_delta of high_bear
        ],
    )
    def test_intensity_boundary_values(
        self,
        intensity_threshold_map: ColorThresholdMap,
        delta: float,
        expected_color: str,
    ) -> None:
        """Threshold edge conditions must resolve deterministically."""
        primitive_intensity = VolumeHistogramPrimitive(color_map=intensity_threshold_map)
        single_point = [VolumeDataPoint(timestamp=1000, volume=500.0, price_delta=delta)]
        result = primitive_intensity.process(single_point)

        assert len(result) == 1
        assert result.bars[0].color == expected_color

    def test_volume_intensity_grading(self) -> None:
        """Threshold rules incorporating both volume magnitude and price delta."""
        custom_map = ColorThresholdMap(
            thresholds=[
                ColorThreshold(
                    label="climax_bull",
                    min_delta=0.001,
                    max_delta=float("inf"),
                    min_volume=10000.0,
                    color="#FFD700",  # Gold for climax volume
                ),
                ColorThreshold(
                    label="standard_bull",
                    min_delta=0.001,
                    max_delta=float("inf"),
                    max_volume=9999.99,
                    color="#26A69A",
                ),
                ColorThreshold(
                    label="standard_bear",
                    min_delta=-float("inf"),
                    max_delta=-0.001,
                    color="#EF5350",
                ),
            ],
            default_color="#9E9E9E",
        )

        primitive = VolumeHistogramPrimitive(color_map=custom_map)
        series = [
            VolumeDataPoint(timestamp=1, volume=15000.0, price_delta=1.0),  # Climax bull
            VolumeDataPoint(timestamp=2, volume=5000.0, price_delta=1.0),   # Standard bull
            VolumeDataPoint(timestamp=3, volume=15000.0, price_delta=-1.0), # Standard bear
        ]

        result = primitive.process(series)
        assert result.bars[0].color == "#FFD700"
        assert result.bars[1].color == "#26A69A"
        assert result.bars[2].color == "#EF5350"

    def test_representation_sequence_protocol(
        self, primitive: VolumeHistogramPrimitive, standard_data_series: List[VolumeDataPoint]
    ) -> None:
        """VolumeHistogramRepresentation must conform to standard sequence behaviors."""
        result = primitive.process(standard_data_series)

        # Indexing and iteration
        assert len(result) == len(standard_data_series)
        assert result[0] == result.bars[0]
        assert result[-1] == result.bars[-1]

        # Iterable protocol
        collected = [bar for bar in result]
        assert len(collected) == len(standard_data_series)

    def test_input_data_series_immutability(
        self, primitive: VolumeHistogramPrimitive, standard_data_series: List[VolumeDataPoint]
    ) -> None:
        """Processing must not mutate original data points or their sequence."""
        snapshot = [(p.timestamp, p.volume, p.price_delta) for p in standard_data_series]
        _ = primitive.process(standard_data_series)

        post_snapshot = [(p.timestamp, p.volume, p.price_delta) for p in standard_data_series]
        assert snapshot == post_snapshot


# =====================================================================
# Test Suite 3: AC 2 - Empty or Invalid Input Handling
# =====================================================================

class TestEmptyOrInvalidVolumeDataHandling:
    """
    Acceptance Criteria:
    - Given an empty or invalid volume data input
    - When the primitive processes the series
    - Then it returns an empty primitive representation without raising unhandled exceptions
    """

    def test_empty_list_input(self, primitive: VolumeHistogramPrimitive) -> None:
        """Empty list input yields an empty representation without errors."""
        result = primitive.process([])
        assert isinstance(result, VolumeHistogramRepresentation)
        assert result.is_empty is True
        assert len(result) == 0
        assert list(result.bars) == []

    def test_none_input(self, primitive: VolumeHistogramPrimitive) -> None:
        """None input yields an empty representation without throwing an exception."""
        result = primitive.process(None)
        assert isinstance(result, VolumeHistogramRepresentation)
        assert result.is_empty is True
        assert len(result) == 0

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "invalid_string",
            12345,
            3.14159,
            {"not": "a series"},
            True,
            False,
            object(),
        ],
    )
    def test_non_iterable_or_unsupported_type_input(
        self, primitive: VolumeHistogramPrimitive, invalid_input: Any
    ) -> None:
        """Invalid non-iterable structures return empty representation gracefully."""
        result = primitive.process(invalid_input)
        assert isinstance(result, VolumeHistogramRepresentation)
        assert result.is_empty is True
        assert len(result) == 0

    @pytest.mark.parametrize(
        "corrupted_series",
        [
            [None],
            [None, None],
            ["string_point"],
            [123],
            [{"volume": 100}],  # missing timestamp and delta
            [VolumeDataPoint(timestamp=100, volume=-10.0, price_delta=1.0)],  # negative volume
            [VolumeDataPoint(timestamp=101, volume=float("nan"), price_delta=1.0)],  # NaN volume
            [VolumeDataPoint(timestamp=102, volume=float("inf"), price_delta=1.0)],  # Infinite volume
            [VolumeDataPoint(timestamp=103, volume=100.0, price_delta=float("nan"))],  # NaN delta
        ],
    )
    def test_fully_invalid_elements_yield_empty_representation(
        self, primitive: VolumeHistogramPrimitive, corrupted_series: Any
    ) -> None:
        """When all input elements are corrupted or invalid, returns empty representation."""
        result = primitive.process(corrupted_series)
        assert isinstance(result, VolumeHistogramRepresentation)
        assert result.is_empty is True
        assert len(result) == 0

    def test_partially_corrupted_series_filters_invalid_points(
        self, primitive: VolumeHistogramPrimitive
    ) -> None:
        """
        When mixed valid and invalid points are supplied, invalid points are sanitized/skipped,
        allowing valid points to be rendered rather than crashing the primitive.
        """
        mixed_series: List[Any] = [
            VolumeDataPoint(timestamp=1, volume=500.0, price_delta=1.5),  # Valid
            None,                                                         # Invalid
            "invalid_row",                                                # Invalid
            VolumeDataPoint(timestamp=2, volume=-50.0, price_delta=2.0),  # Invalid negative volume
            VolumeDataPoint(timestamp=3, volume=750.0, price_delta=-0.8), # Valid
            VolumeDataPoint(timestamp=4, volume=float("nan"), price_delta=0.0), # Invalid NaN
        ]

        result = primitive.process(mixed_series)
        assert isinstance(result, VolumeHistogramRepresentation)
        assert len(result) == 2
        assert not result.is_empty
        assert result.bars[0].timestamp == 1
        assert result.bars[1].timestamp == 3


# =====================================================================
# Test Suite 4: Configuration Fallbacks & Edge Cases
# =====================================================================

class TestPrimitiveConfigurationAndFallbacks:
    """Verifies edge conditions in primitive instantiation and thresholds."""

    def test_initialization_with_default_configuration(self) -> None:
        """Primitive initialized without explicit color_map uses sensible internal defaults."""
        primitive_default = VolumeHistogramPrimitive()
        assert primitive_default.color_map is not None
        assert isinstance(primitive_default.color_map, ColorThresholdMap)

        series = [VolumeDataPoint(timestamp=1, volume=100.0, price_delta=1.0)]
        result = primitive_default.process(series)
        assert len(result) == 1
        assert isinstance(result.bars[0].color, str)
        assert len(result.bars[0].color) > 0

    def test_fallback_color_when_no_threshold_matches(self) -> None:
        """Unmatched delta/volume combinations safely resolve to the configured fallback color."""
        sparse_threshold_map = ColorThresholdMap(
            thresholds=[
                ColorThreshold(label="bull_only", min_delta=10.0, max_delta=20.0, color="#00FF00"),
            ],
            default_color="#B0BEC5",
        )
        sparse_primitive = VolumeHistogramPrimitive(color_map=sparse_threshold_map)

        # Delta 1.0 is outside [10.0, 20.0], so must resolve to default_color
        series = [VolumeDataPoint(timestamp=1, volume=100.0, price_delta=1.0)]
        result = sparse_primitive.process(series)

        assert len(result) == 1
        assert result.bars[0].color == "#B0BEC5"

    def test_zero_volume_bars_processed_without_error(
        self, primitive: VolumeHistogramPrimitive
    ) -> None:
        """Zero volume is valid in market structures (e.g. illiquid intervals) and shouldn't fail."""
        zero_vol_series = [
            VolumeDataPoint(timestamp=100, volume=0.0, price_delta=0.0),
            VolumeDataPoint(timestamp=101, volume=0.0, price_delta=0.5),
        ]
        result = primitive.process(zero_vol_series)
        assert len(result) == 2
        assert result.bars[0].volume == 0.0
        assert result.bars[1].volume == 0.0

    def test_deterministic_output_on_repeated_invocations(
        self, primitive: VolumeHistogramPrimitive, standard_data_series: List[VolumeDataPoint]
    ) -> None:
        """Idempotency test: invoking process multiple times yields identical results."""
        first_run = primitive.process(standard_data_series)
        second_run = primitive.process(standard_data_series)

        assert len(first_run) == len(second_run)
        for b1, b2 in zip(first_run.bars, second_run.bars):
            assert b1.timestamp == b2.timestamp
            assert math.isclose(b1.volume, b2.volume, rel_tol=1e-9)
            assert b1.color == b2.color