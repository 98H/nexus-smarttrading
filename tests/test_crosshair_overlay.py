"""
Unit tests for High-Precision Crosshair Overlay Layer.

Specification / Stories Tested:
- Story 1.4.1: Develop High-Precision Crosshair Overlay Layer
  - AC 1: Sub-pixel coordinate precision for horizontal and vertical guides matching canvas boundaries.
  - AC 2: Empty draw commands / primitives when hidden or disabled.
  - AC 3: Out-of-bounds coordinate handling according to boundary policy (CLIP vs EXCLUDE).
"""

import math
import pytest

from src.ui.overlays import (
    BoundaryPolicy,
    HighPrecisionCrosshairOverlay,
    LineSegment,
    Point,
    Viewport,
)
from src.ui.overlays.crosshair import (
    BoundaryPolicy as DirectBoundaryPolicy,
    CrosshairGeometry,
    HighPrecisionCrosshairOverlay as DirectHighPrecisionCrosshairOverlay,
    LineSegment as DirectLineSegment,
    Point as DirectPoint,
    Viewport as DirectViewport,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_viewport() -> Viewport:
    """Standard 1920x1080 canvas viewport starting at origin (0, 0)."""
    return Viewport(x_min=0.0, y_min=0.0, x_max=1920.0, y_max=1080.0)


@pytest.fixture
def offset_viewport() -> Viewport:
    """Non-zero origin viewport with fractional sub-pixel extents."""
    return Viewport(x_min=100.25, y_min=50.75, x_max=800.5, y_max=600.125)


@pytest.fixture
def clip_overlay(default_viewport: Viewport) -> HighPrecisionCrosshairOverlay:
    """Overlay configured with BoundaryPolicy.CLIP."""
    return HighPrecisionCrosshairOverlay(
        viewport=default_viewport,
        boundary_policy=BoundaryPolicy.CLIP,
        visible=True,
        enabled=True,
    )


@pytest.fixture
def exclude_overlay(default_viewport: Viewport) -> HighPrecisionCrosshairOverlay:
    """Overlay configured with BoundaryPolicy.EXCLUDE."""
    return HighPrecisionCrosshairOverlay(
        viewport=default_viewport,
        boundary_policy=BoundaryPolicy.EXCLUDE,
        visible=True,
        enabled=True,
    )


# ============================================================================
# Module Export & Package Structure Tests
# ============================================================================

class TestOverlayExports:
    """Verify clean public API exposure through __init__.py and crosshair.py."""

    def test_direct_and_package_imports_match(self):
        assert HighPrecisionCrosshairOverlay is DirectHighPrecisionCrosshairOverlay
        assert BoundaryPolicy is DirectBoundaryPolicy
        assert Viewport is DirectViewport
        assert Point is DirectPoint
        assert LineSegment is DirectLineSegment


# ============================================================================
# Viewport & Configuration Validation Tests
# ============================================================================

class TestOverlayInitialization:
    """Test overlay initialization and viewport boundary constraints."""

    def test_valid_initialization(self, default_viewport: Viewport):
        overlay = HighPrecisionCrosshairOverlay(
            viewport=default_viewport,
            boundary_policy=BoundaryPolicy.CLIP,
            visible=True,
            enabled=True,
        )
        assert overlay.viewport == default_viewport
        assert overlay.boundary_policy == BoundaryPolicy.CLIP
        assert overlay.visible is True
        assert overlay.enabled is True

    def test_invalid_viewport_inverted_horizontal_raises_value_error(self):
        with pytest.raises(ValueError):
            Viewport(x_min=500.0, y_min=0.0, x_max=200.0, y_max=1000.0)

    def test_invalid_viewport_zero_width_raises_value_error(self):
        with pytest.raises(ValueError):
            Viewport(x_min=100.0, y_min=0.0, x_max=100.0, y_max=1000.0)

    def test_invalid_viewport_inverted_vertical_raises_value_error(self):
        with pytest.raises(ValueError):
            Viewport(x_min=0.0, y_min=800.0, x_max=1000.0, y_max=200.0)

    def test_invalid_viewport_zero_height_raises_value_error(self):
        with pytest.raises(ValueError):
            Viewport(x_min=0.0, y_min=500.0, x_max=1000.0, y_max=500.0)


# ============================================================================
# AC 1: High-Precision Sub-Pixel Guide Coordinates
# ============================================================================

class TestSubPixelPrecision:
    """
    AC 1: Given an active canvas viewport and a HighPrecisionCrosshairOverlay instance,
    When sub-pixel coordinates (x, y) are supplied with sub-pixel precision,
    Then horizontal and vertical guide coordinates are generated with floating-point
    accuracy matching the canvas boundaries.
    """

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (100.123456789, 200.987654321),
            (0.000000001, 1079.999999999),
            (960.5, 540.25),
            (1919.999999999, 0.000000001),
        ],
    )
    def test_subpixel_guide_coordinates_with_origin_viewport(
        self,
        clip_overlay: HighPrecisionCrosshairOverlay,
        default_viewport: Viewport,
        x: float,
        y: float,
    ):
        clip_overlay.set_position(x, y)
        geometry = clip_overlay.calculate_geometry()

        assert isinstance(geometry, CrosshairGeometry)

        # Horizontal guide line checks
        h_line = geometry.horizontal
        assert isinstance(h_line, LineSegment)
        assert math.isclose(h_line.start.x, default_viewport.x_min, abs_tol=1e-12)
        assert math.isclose(h_line.start.y, y, abs_tol=1e-12)
        assert math.isclose(h_line.end.x, default_viewport.x_max, abs_tol=1e-12)
        assert math.isclose(h_line.end.y, y, abs_tol=1e-12)

        # Vertical guide line checks
        v_line = geometry.vertical
        assert isinstance(v_line, LineSegment)
        assert math.isclose(v_line.start.x, x, abs_tol=1e-12)
        assert math.isclose(v_line.start.y, default_viewport.y_min, abs_tol=1e-12)
        assert math.isclose(v_line.end.x, x, abs_tol=1e-12)
        assert math.isclose(v_line.end.y, default_viewport.y_max, abs_tol=1e-12)

    def test_subpixel_precision_with_offset_viewport(self, offset_viewport: Viewport):
        overlay = HighPrecisionCrosshairOverlay(viewport=offset_viewport)
        target_x = 350.333333333333
        target_y = 275.666666666667

        overlay.set_position(target_x, target_y)
        geometry = overlay.calculate_geometry()

        assert geometry is not None
        # Horizontal guide spans strictly from viewport.x_min to viewport.x_max at target_y
        assert math.isclose(geometry.horizontal.start.x, offset_viewport.x_min, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.start.y, target_y, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.end.x, offset_viewport.x_max, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.end.y, target_y, abs_tol=1e-12)

        # Vertical guide spans strictly from viewport.y_min to viewport.y_max at target_x
        assert math.isclose(geometry.vertical.start.x, target_x, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.start.y, offset_viewport.y_min, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.end.x, target_x, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.end.y, offset_viewport.y_max, abs_tol=1e-12)

    def test_calculate_geometry_before_setting_position_raises_value_error(
        self, clip_overlay: HighPrecisionCrosshairOverlay
    ):
        with pytest.raises(ValueError):
            clip_overlay.calculate_geometry()


# ============================================================================
# AC 2: Visibility and Enabled State Controls
# ============================================================================

class TestVisibilityAndEnabledStates:
    """
    AC 2: Given the crosshair overlay is marked as hidden or disabled,
    When rendering primitives are requested,
    Then an empty set of draw commands is returned.
    """

    def test_render_primitives_non_empty_when_active(
        self, clip_overlay: HighPrecisionCrosshairOverlay
    ):
        clip_overlay.set_position(500.0, 500.0)
        primitives = clip_overlay.get_render_primitives()
        assert len(primitives) > 0

    def test_render_primitives_empty_when_hidden(
        self, default_viewport: Viewport
    ):
        overlay = HighPrecisionCrosshairOverlay(
            viewport=default_viewport,
            visible=False,
            enabled=True,
        )
        overlay.set_position(500.0, 500.0)
        commands = overlay.get_render_primitives()
        assert len(commands) == 0

    def test_render_primitives_empty_when_disabled(
        self, default_viewport: Viewport
    ):
        overlay = HighPrecisionCrosshairOverlay(
            viewport=default_viewport,
            visible=True,
            enabled=False,
        )
        overlay.set_position(500.0, 500.0)
        commands = overlay.get_render_primitives()
        assert len(commands) == 0

    def test_render_primitives_empty_when_hidden_and_disabled(
        self, default_viewport: Viewport
    ):
        overlay = HighPrecisionCrosshairOverlay(
            viewport=default_viewport,
            visible=False,
            enabled=False,
        )
        overlay.set_position(500.0, 500.0)
        commands = overlay.get_render_primitives()
        assert len(commands) == 0

    def test_dynamic_visibility_toggle(self, clip_overlay: HighPrecisionCrosshairOverlay):
        clip_overlay.set_position(250.125, 450.875)

        # Initially active
        assert len(clip_overlay.get_render_primitives()) > 0

        # Hide overlay
        clip_overlay.visible = False
        assert len(clip_overlay.get_render_primitives()) == 0

        # Restore visibility
        clip_overlay.visible = True
        assert len(clip_overlay.get_render_primitives()) > 0

    def test_dynamic_enabled_toggle(self, clip_overlay: HighPrecisionCrosshairOverlay):
        clip_overlay.set_position(250.125, 450.875)

        # Initially active
        assert len(clip_overlay.get_render_primitives()) > 0

        # Disable overlay
        clip_overlay.enabled = False
        assert len(clip_overlay.get_render_primitives()) == 0

        # Re-enable overlay
        clip_overlay.enabled = True
        assert len(clip_overlay.get_render_primitives()) > 0


# ============================================================================
# AC 3: Out-of-Bounds Handling and Boundary Policies
# ============================================================================

class TestBoundaryPolicies:
    """
    AC 3: Given target coordinates outside the viewport boundaries,
    When calculating overlay geometry,
    Then coordinates are either clipped to the viewport extents or excluded
    based on the configured boundary policy.
    """

    # --- BoundaryPolicy.CLIP tests ---

    @pytest.mark.parametrize(
        ("input_x", "input_y", "expected_clipped_x", "expected_clipped_y"),
        [
            (-100.5, 500.0, 0.0, 500.0),                  # Left of viewport
            (2500.0, 500.0, 1920.0, 500.0),               # Right of viewport
            (500.0, -50.25, 500.0, 0.0),                  # Above viewport
            (500.0, 1500.75, 500.0, 1080.0),              # Below viewport
            (-200.0, -300.0, 0.0, 0.0),                   # Top-Left outside
            (3000.0, 2000.0, 1920.0, 1080.0),             # Bottom-Right outside
        ],
    )
    def test_boundary_policy_clip_clamps_coordinates(
        self,
        clip_overlay: HighPrecisionCrosshairOverlay,
        default_viewport: Viewport,
        input_x: float,
        input_y: float,
        expected_clipped_x: float,
        expected_clipped_y: float,
    ):
        clip_overlay.set_position(input_x, input_y)
        geometry = clip_overlay.calculate_geometry()

        assert geometry is not None

        # Horizontal guide line clamped to viewport y-extents
        assert math.isclose(geometry.horizontal.start.x, default_viewport.x_min, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.start.y, expected_clipped_y, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.end.x, default_viewport.x_max, abs_tol=1e-12)
        assert math.isclose(geometry.horizontal.end.y, expected_clipped_y, abs_tol=1e-12)

        # Vertical guide line clamped to viewport x-extents
        assert math.isclose(geometry.vertical.start.x, expected_clipped_x, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.start.y, default_viewport.y_min, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.end.x, expected_clipped_x, abs_tol=1e-12)
        assert math.isclose(geometry.vertical.end.y, default_viewport.y_max, abs_tol=1e-12)

        # Draw commands must still be produced for clipped coordinates
        assert len(clip_overlay.get_render_primitives()) > 0

    # --- BoundaryPolicy.EXCLUDE tests ---

    @pytest.mark.parametrize(
        ("outside_x", "outside_y"),
        [
            (-0.000001, 500.0),
            (1920.000001, 500.0),
            (500.0, -0.000001),
            (500.0, 1080.000001),
            (-50.0, -50.0),
            (2000.0, 1200.0),
        ],
    )
    def test_boundary_policy_exclude_returns_none_geometry(
        self,
        exclude_overlay: HighPrecisionCrosshairOverlay,
        outside_x: float,
        outside_y: float,
    ):
        exclude_overlay.set_position(outside_x, outside_y)
        geometry = exclude_overlay.calculate_geometry()
        assert geometry is None

    @pytest.mark.parametrize(
        ("outside_x", "outside_y"),
        [
            (-10.0, 500.0),
            (1920.01, 500.0),
            (500.0, -0.1),
            (500.0, 1081.0),
        ],
    )
    def test_boundary_policy_exclude_returns_empty_render_primitives(
        self,
        exclude_overlay: HighPrecisionCrosshairOverlay,
        outside_x: float,
        outside_y: float,
    ):
        exclude_overlay.set_position(outside_x, outside_y)
        commands = exclude_overlay.get_render_primitives()
        assert len(commands) == 0

    @pytest.mark.parametrize(
        ("boundary_x", "boundary_y"),
        [
            (0.0, 0.0),
            (1920.0, 1080.0),
            (0.0, 1080.0),
            (1920.0, 0.0),
        ],
    )
    def test_boundary_policy_exclude_includes_exact_viewport_edges(
        self,
        exclude_overlay: HighPrecisionCrosshairOverlay,
        boundary_x: float,
        boundary_y: float,
    ):
        exclude_overlay.set_position(boundary_x, boundary_y)
        geometry = exclude_overlay.calculate_geometry()
        assert geometry is not None
        assert len(exclude_overlay.get_render_primitives()) > 0

    def test_policy_switch_at_runtime(self, default_viewport: Viewport):
        overlay = HighPrecisionCrosshairOverlay(
            viewport=default_viewport,
            boundary_policy=BoundaryPolicy.EXCLUDE,
        )
        overlay.set_position(-10.0, 500.0)

        # Under EXCLUDE: out of bounds produces no geometry/primitives
        assert overlay.calculate_geometry() is None
        assert len(overlay.get_render_primitives()) == 0

        # Switch to CLIP: out of bounds produces clamped geometry/primitives
        overlay.boundary_policy = BoundaryPolicy.CLIP
        geometry = overlay.calculate_geometry()
        assert geometry is not None
        assert math.isclose(geometry.vertical.start.x, default_viewport.x_min, abs_tol=1e-12)
        assert len(overlay.get_render_primitives()) > 0