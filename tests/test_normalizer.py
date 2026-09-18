"""
Unit tests for High-DPI Canvas Coordinate Normalizer.

Specifications Tested:
- Story 1.1.1: Implement High-DPI Canvas Coordinate Normalizer
- Acceptance Criteria:
  1. DPR > 1.0 and raw screen event coordinates (x, y) divided by DPR and mapped
     relative to the canvas origin.
  2. Pointer coordinates outside canvas viewport bounds clamped to
     [0, canvas_width] and [0, canvas_height] when clamping is enabled.
  3. Invalid DPR values (e.g., <= 0) raise ValueError during initialization
     or normalization execution.
- Target Modules:
  - src/canvas/normalizer.py
  - src/canvas/__init__.py
"""

import pytest

import src.canvas
from src.canvas import CanvasCoordinateNormalizer
from src.canvas.normalizer import CanvasCoordinateNormalizer as DirectNormalizer


# ============================================================================
# Package & Module Exports
# ============================================================================


class TestCanvasModuleExports:
    """Verifies module-level exposure and packaging of the normalizer."""

    def test_canvas_package_exposes_normalizer(self):
        """Ensure CanvasCoordinateNormalizer is exported at package root."""
        assert hasattr(src.canvas, "CanvasCoordinateNormalizer")
        assert CanvasCoordinateNormalizer is DirectNormalizer

    def test_canvas_dunder_all_contains_normalizer(self):
        """Ensure __all__ lists CanvasCoordinateNormalizer if defined."""
        if hasattr(src.canvas, "__all__"):
            assert "CanvasCoordinateNormalizer" in src.canvas.__all__


# ============================================================================
# Acceptance Criterion 3: Invalid DPR Validation
# ============================================================================


class TestDPRValidation:
    """Tests validation of device pixel ratio (DPR) values."""

    @pytest.mark.parametrize("invalid_dpr", [0, 0.0, -0.001, -1.0, -2.5, -100])
    def test_init_raises_value_error_for_non_positive_dpr(self, invalid_dpr: float):
        """Given invalid DPR values (<= 0), initialization raises ValueError."""
        with pytest.raises(ValueError):
            CanvasCoordinateNormalizer(
                800.0, 600.0, dpr=invalid_dpr, origin=(0.0, 0.0)
            )

    @pytest.mark.parametrize("invalid_dpr", [0, 0.0, -0.5, -2.0])
    def test_normalize_execution_raises_value_error_for_invalid_dpr_override(
        self, invalid_dpr: float
    ):
        """Given invalid DPR during execution, normalize raises ValueError."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=2.0, origin=(0.0, 0.0)
        )
        with pytest.raises(ValueError):
            normalizer.normalize(100.0, 100.0, dpr=invalid_dpr)

    @pytest.mark.parametrize("invalid_dpr", [0, -1.0])
    def test_dpr_setter_raises_value_error_when_updated_to_non_positive(
        self, invalid_dpr: float
    ):
        """Given normalizer instance, mutating dpr to <= 0 raises ValueError."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=1.0, origin=(0.0, 0.0)
        )
        if hasattr(normalizer, "dpr"):
            with pytest.raises(ValueError):
                normalizer.dpr = invalid_dpr


# ============================================================================
# Acceptance Criterion 1: Coordinate Normalization with DPR > 1.0 & Origin
# ============================================================================


class TestCoordinateNormalization:
    """Tests coordinate normalization with DPR scaling and origin mapping."""

    @pytest.mark.parametrize(
        ("raw_x", "raw_y", "dpr", "expected_x", "expected_y"),
        [
            (200.0, 100.0, 2.0, 100.0, 50.0),
            (300.0, 450.0, 1.5, 200.0, 300.0),
            (300.0, 600.0, 3.0, 100.0, 200.0),
            (500.0, 250.0, 2.5, 200.0, 100.0),
        ],
    )
    def test_normalize_divides_by_dpr_at_origin_zero(
        self,
        raw_x: float,
        raw_y: float,
        dpr: float,
        expected_x: float,
        expected_y: float,
    ):
        """Given DPR > 1.0 and origin (0, 0), coords are scaled by DPR."""
        normalizer = CanvasCoordinateNormalizer(
            1000.0, 1000.0, dpr=dpr, origin=(0.0, 0.0)
        )
        result_x, result_y = normalizer.normalize(raw_x, raw_y)

        assert result_x == pytest.approx(expected_x)
        assert result_y == pytest.approx(expected_y)

    @pytest.mark.parametrize(
        (
            "raw_x",
            "raw_y",
            "origin_x",
            "origin_y",
            "dpr",
            "expected_x",
            "expected_y",
        ),
        [
            (250.0, 300.0, 50.0, 100.0, 2.0, 100.0, 100.0),
            (170.0, 245.0, 20.0, 35.0, 1.5, 100.0, 140.0),
            (650.0, 950.0, 200.0, 50.0, 3.0, 150.0, 300.0),
        ],
    )
    def test_normalize_maps_relative_to_non_zero_origin_and_dpr(
        self,
        raw_x: float,
        raw_y: float,
        origin_x: float,
        origin_y: float,
        dpr: float,
        expected_x: float,
        expected_y: float,
    ):
        """Given non-zero origin, coords map relative to origin and scale by DPR."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=dpr, origin=(origin_x, origin_y)
        )
        result_x, result_y = normalizer.normalize(raw_x, raw_y)

        assert result_x == pytest.approx(expected_x)
        assert result_y == pytest.approx(expected_y)

    def test_pointer_exactly_at_canvas_origin_yields_zero(self):
        """When event coordinate matches origin, normalized coordinates are (0, 0)."""
        origin_x, origin_y = 120.0, 85.0
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=2.0, origin=(origin_x, origin_y)
        )
        result_x, result_y = normalizer.normalize(origin_x, origin_y)

        assert result_x == pytest.approx(0.0)
        assert result_y == pytest.approx(0.0)

    def test_normalize_with_standard_dpr_baseline(self):
        """Given DPR = 1.0, normalized coordinates reflect 1:1 screen mapping."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=1.0, origin=(50.0, 25.0)
        )
        result_x, result_y = normalizer.normalize(150.0, 125.0)

        assert result_x == pytest.approx(100.0)
        assert result_y == pytest.approx(100.0)

    def test_subpixel_floating_point_precision(self):
        """Ensure float precision is preserved during high-DPI normalization."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=2.0, origin=(10.5, 20.25)
        )
        # Raw coords: (31.5, 61.25) -> relative: (21.0, 41.0) -> / 2: (10.5, 20.5)
        result_x, result_y = normalizer.normalize(31.5, 61.25)

        assert result_x == pytest.approx(10.5)
        assert result_y == pytest.approx(20.5)

    def test_normalize_idempotency_and_no_state_leak(self):
        """Normalization calls must be purely functional and idempotent."""
        normalizer = CanvasCoordinateNormalizer(
            800.0, 600.0, dpr=2.0, origin=(50.0, 50.0)
        )
        first_call = normalizer.normalize(250.0, 150.0)
        second_call = normalizer.normalize(250.0, 150.0)

        assert first_call == second_call
        assert first_call == (pytest.approx(100.0), pytest.approx(50.0))


# ============================================================================
# Acceptance Criterion 2: Viewport Bounds Clamping
# ============================================================================


class TestCoordinateClamping:
    """Tests clamping behavior when pointer is outside canvas viewport bounds."""

    @pytest.fixture
    def normalizer(self) -> CanvasCoordinateNormalizer:
        """Fixture providing a standard canvas normalizer (800x600, DPR=2)."""
        return CanvasCoordinateNormalizer(
            canvas_width=800.0,
            canvas_height=600.0,
            dpr=2.0,
            origin=(100.0, 100.0),
        )

    def test_clamping_enabled_clamps_negative_coordinates_to_zero(
        self, normalizer: CanvasCoordinateNormalizer
    ):
        """Pointer beyond top/left origin clamps to (0, 0)."""
        # Event at (0, 0) screen coords -> relative: (-100, -100) -> / 2: (-50, -50)
        clamped_x, clamped_y = normalizer.normalize(0.0, 0.0, clamp=True)

        assert clamped_x == pytest.approx(0.0)
        assert clamped_y == pytest.approx(0.0)

    def test_clamping_enabled_clamps_overflow_coordinates_to_width_and_height(
        self, normalizer: CanvasCoordinateNormalizer
    ):
        """Pointer beyond bottom/right bounds clamps to (width, height)."""
        # Max screen: origin (100, 100) + (800*2, 600*2) = (1700, 1300)
        # Event far beyond: (3000, 2500)
        clamped_x, clamped_y = normalizer.normalize(3000.0, 2500.0, clamp=True)

        assert clamped_x == pytest.approx(800.0)
        assert clamped_y == pytest.approx(600.0)

    def test_clamping_enabled_clamps_single_axis_out_of_bounds(
        self, normalizer: CanvasCoordinateNormalizer
    ):
        """Clamp applies independently to axes (X in bounds, Y out of bounds)."""
        # X: screen 500 -> rel 400 -> norm 200 (in [0, 800])
        # Y: screen 2000 -> rel 1900 -> norm 950 (exceeds 600)
        clamped_x, clamped_y = normalizer.normalize(500.0, 2000.0, clamp=True)

        assert clamped_x == pytest.approx(200.0)
        assert clamped_y == pytest.approx(600.0)

        # X: screen -50 -> rel -150 -> norm -75 (below 0)
        # Y: screen 700 -> rel 600 -> norm 300 (in [0, 600])
        clamped_x_neg, clamped_y_in = normalizer.normalize(-50.0, 700.0, clamp=True)

        assert clamped_x_neg == pytest.approx(0.0)
        assert clamped_y_in == pytest.approx(300.0)

    @pytest.mark.parametrize(
        ("screen_x", "screen_y", "expected_x", "expected_y"),
        [
            (100.0, 100.0, 0.0, 0.0),  # Top-Left boundary
            (1700.0, 100.0, 800.0, 0.0),  # Top-Right boundary
            (100.0, 1300.0, 0.0, 600.0),  # Bottom-Left boundary
            (1700.0, 1300.0, 800.0, 600.0),  # Bottom-Right boundary
        ],
    )
    def test_clamping_boundary_values_exact(
        self,
        normalizer: CanvasCoordinateNormalizer,
        screen_x: float,
        screen_y: float,
        expected_x: float,
        expected_y: float,
    ):
        """Pointer directly on canvas boundary remains at boundaries when clamped."""
        norm_x, norm_y = normalizer.normalize(screen_x, screen_y, clamp=True)

        assert norm_x == pytest.approx(expected_x)
        assert norm_y == pytest.approx(expected_y)

    def test_clamping_disabled_preserves_negative_and_overflow_coordinates(
        self, normalizer: CanvasCoordinateNormalizer
    ):
        """When clamping is False, returned coordinates can be negative or exceed canvas size."""
        # Top-left out of bounds
        out_x, out_y = normalizer.normalize(0.0, 0.0, clamp=False)
        assert out_x == pytest.approx(-50.0)
        assert out_y == pytest.approx(-50.0)

        # Bottom-right out of bounds: screen (2100, 1500) -> rel (2000, 1400) -> / 2: (1000, 700)
        over_x, over_y = normalizer.normalize(2100.0, 1500.0, clamp=False)
        assert over_x == pytest.approx(1000.0)
        assert over_y == pytest.approx(700.0)

    def test_clamp_parameter_defaults_to_false(
        self, normalizer: CanvasCoordinateNormalizer
    ):
        """Default behavior without specifying clamp argument must not clamp."""
        default_x, default_y = normalizer.normalize(0.0, 0.0)
        explicit_x, explicit_y = normalizer.normalize(0.0, 0.0, clamp=False)

        assert default_x == explicit_x == pytest.approx(-50.0)
        assert default_y == explicit_y == pytest.approx(-50.0)