"""Unit tests for Orthographic Matrix Viewport Pan and Zoom Engine.

Covers:
- Story 1.1.2: Implement Orthographic Matrix Viewport Pan and Zoom Engine
Target modules:
- src/viewport/matrix.py
- src/viewport/ortho_engine.py
"""

import math
import pytest

from src.viewport.matrix import Matrix3x3
from src.viewport.ortho_engine import OrthoViewportEngine

EPSILON = 1e-7


# ============================================================================
# Matrix3x3 Unit Tests (src/viewport/matrix.py)
# ============================================================================


class TestMatrix3x3:
    """Deterministic unit tests for 3x3 affine matrix mathematics."""

    def test_identity_matrix(self):
        """Identity matrix has 1.0 on diagonal and 0.0 elsewhere."""
        ident = Matrix3x3.identity()

        for r in range(3):
            for c in range(3):
                expected = 1.0 if r == c else 0.0
                assert ident[r, c] == pytest.approx(expected, abs=EPSILON)

        px, py = ident.transform_point(12.34, -56.78)
        assert px == pytest.approx(12.34, abs=EPSILON)
        assert py == pytest.approx(-56.78, abs=EPSILON)

    def test_translation_matrix(self):
        """Translation matrix translates coordinates by (tx, ty)."""
        tx, ty = 45.0, -30.0
        mat = Matrix3x3.translation(tx, ty)

        assert mat[0, 2] == pytest.approx(tx, abs=EPSILON)
        assert mat[1, 2] == pytest.approx(ty, abs=EPSILON)

        out_x, out_y = mat.transform_point(10.0, 20.0)
        assert out_x == pytest.approx(55.0, abs=EPSILON)
        assert out_y == pytest.approx(-10.0, abs=EPSILON)

    def test_scale_matrix(self):
        """Scale matrix scales coordinates by (sx, sy)."""
        sx, sy = 2.5, 0.5
        mat = Matrix3x3.scale(sx, sy)

        assert mat[0, 0] == pytest.approx(sx, abs=EPSILON)
        assert mat[1, 1] == pytest.approx(sy, abs=EPSILON)

        out_x, out_y = mat.transform_point(4.0, 10.0)
        assert out_x == pytest.approx(10.0, abs=EPSILON)
        assert out_y == pytest.approx(5.0, abs=EPSILON)

    def test_matrix_multiplication(self):
        """Multiplying translation and scale matrices correctly chains transforms."""
        trans = Matrix3x3.translation(100.0, 50.0)
        scale = Matrix3x3.scale(2.0, 2.0)

        # Combined transform: Translation * Scale -> Scale first, then Translate
        combined = trans @ scale
        out_x, out_y = combined.transform_point(10.0, 20.0)
        # 10 * 2 + 100 = 120, 20 * 2 + 50 = 90
        assert out_x == pytest.approx(120.0, abs=EPSILON)
        assert out_y == pytest.approx(90.0, abs=EPSILON)

    def test_matrix_inverse_valid(self):
        """Inverse matrix undoes transformation and recovers the identity matrix."""
        mat = Matrix3x3.translation(15.0, -25.0) @ Matrix3x3.scale(3.0, 3.0)
        inv = mat.inverse()

        identity_candidate = mat @ inv
        ident = Matrix3x3.identity()

        for r in range(3):
            for c in range(3):
                assert identity_candidate[r, c] == pytest.approx(ident[r, c], abs=EPSILON)

        orig_x, orig_y = 42.0, -17.5
        transformed_x, transformed_y = mat.transform_point(orig_x, orig_y)
        roundtrip_x, roundtrip_y = inv.transform_point(transformed_x, transformed_y)

        assert roundtrip_x == pytest.approx(orig_x, abs=EPSILON)
        assert roundtrip_y == pytest.approx(orig_y, abs=EPSILON)

    def test_singular_matrix_inverse_raises(self):
        """Attempting to invert a singular non-invertible matrix raises ValueError."""
        singular_mat = Matrix3x3.scale(0.0, 1.0)
        with pytest.raises(ValueError):
            singular_mat.inverse()


# ============================================================================
# OrthoViewportEngine Unit Tests (src/viewport/ortho_engine.py)
# ============================================================================


class TestOrthoViewportEngineInitialization:
    """Tests for initial state and configuration validation."""

    def test_default_initial_state(self):
        """Engine defaults to zoom 1.0, offset (0, 0), and an identity view matrix."""
        engine = OrthoViewportEngine()
        assert engine.zoom == pytest.approx(1.0, abs=EPSILON)
        assert engine.offset == pytest.approx((0.0, 0.0), abs=EPSILON)

        view_mat = engine.view_matrix
        assert view_mat[0, 0] == pytest.approx(1.0, abs=EPSILON)
        assert view_mat[1, 1] == pytest.approx(1.0, abs=EPSILON)
        assert view_mat[0, 2] == pytest.approx(0.0, abs=EPSILON)
        assert view_mat[1, 2] == pytest.approx(0.0, abs=EPSILON)

    def test_custom_initial_state(self):
        """Engine correctly initializes with user-specified zoom, offset, and bounds."""
        engine = OrthoViewportEngine(
            min_zoom=0.2,
            max_zoom=5.0,
            initial_zoom=2.0,
            initial_offset=(100.0, -50.0),
        )
        assert engine.zoom == pytest.approx(2.0, abs=EPSILON)
        assert engine.offset == pytest.approx((100.0, -50.0), abs=EPSILON)
        assert engine.min_zoom == pytest.approx(0.2, abs=EPSILON)
        assert engine.max_zoom == pytest.approx(5.0, abs=EPSILON)

    @pytest.mark.parametrize(
        "min_zoom, max_zoom, initial_zoom",
        [
            (0.0, 5.0, 1.0),      # min_zoom <= 0
            (-1.0, 5.0, 1.0),     # negative min_zoom
            (5.0, 2.0, 3.0),      # min_zoom > max_zoom
            (2.0, 2.0, 2.0),      # min_zoom == max_zoom (degenerate)
            (1.0, 5.0, 0.5),      # initial_zoom < min_zoom
            (1.0, 5.0, 6.0),      # initial_zoom > max_zoom
        ],
    )
    def test_invalid_bounds_raise_value_error(self, min_zoom, max_zoom, initial_zoom):
        """Invalid bounds or initial zoom levels outside bounds raise ValueError."""
        with pytest.raises(ValueError):
            OrthoViewportEngine(
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                initial_zoom=initial_zoom,
            )


class TestOrthoViewportEnginePan:
    """Acceptance Criteria:

    Given an orthographic viewport at initial zoom 1.0 and offset (0, 0),
    When applying a pan translation of (dx, dy),
    Then the viewport offset updates to (dx, dy) and the view matrix reflects the translation.
    """

    def test_single_pan_updates_offset_and_matrix(self):
        """Pan translates the offset and updates translation elements of the view matrix."""
        engine = OrthoViewportEngine()
        dx, dy = 150.0, -75.0

        engine.pan(dx, dy)

        assert engine.offset == pytest.approx((dx, dy), abs=EPSILON)

        # View matrix reflects translation in columns (0, 2) and (1, 2)
        view_mat = engine.view_matrix
        assert view_mat[0, 2] == pytest.approx(dx, abs=EPSILON)
        assert view_mat[1, 2] == pytest.approx(dy, abs=EPSILON)

        # World origin (0, 0) should now map to screen coordinate (dx, dy)
        screen_origin = engine.world_to_screen(0.0, 0.0)
        assert screen_origin == pytest.approx((dx, dy), abs=EPSILON)

    def test_cumulative_pans_accumulate_offset(self):
        """Multiple sequential pan calls accumulate linearly."""
        engine = OrthoViewportEngine()

        engine.pan(50.0, 25.0)
        engine.pan(-20.0, 15.0)
        engine.pan(10.5, -40.5)

        expected_offset = (40.5, -0.5)
        assert engine.offset == pytest.approx(expected_offset, abs=EPSILON)

        view_mat = engine.view_matrix
        assert view_mat[0, 2] == pytest.approx(expected_offset[0], abs=EPSILON)
        assert view_mat[1, 2] == pytest.approx(expected_offset[1], abs=EPSILON)

    def test_zero_pan_leaves_state_unchanged(self):
        """Panning by (0, 0) does not alter offset or matrix."""
        engine = OrthoViewportEngine(initial_offset=(10.0, 20.0))
        engine.pan(0.0, 0.0)

        assert engine.offset == pytest.approx((10.0, 20.0), abs=EPSILON)
        assert engine.view_matrix[0, 2] == pytest.approx(10.0, abs=EPSILON)
        assert engine.view_matrix[1, 2] == pytest.approx(20.0, abs=EPSILON)


class TestOrthoViewportEngineZoomAnchor:
    """Acceptance Criteria:

    Given a screen coordinate anchor point (x, y),
    When applying a zoom factor,
    Then the zoom level scales proportionally and the pan offset adjusts
    so the world coordinate under (x, y) remains invariant.
    """

    @pytest.mark.parametrize("zoom_factor", [1.5, 2.0, 0.5, 0.75])
    def test_world_coordinate_under_anchor_remains_invariant(self, zoom_factor):
        """World point under the screen anchor must be identical before and after zoom."""
        engine = OrthoViewportEngine(min_zoom=0.1, max_zoom=10.0)
        anchor_x, anchor_y = 300.0, 200.0

        world_before = engine.screen_to_world(anchor_x, anchor_y)
        engine.zoom_at(anchor_x, anchor_y, zoom_factor)
        world_after = engine.screen_to_world(anchor_x, anchor_y)

        assert world_after[0] == pytest.approx(world_before[0], abs=EPSILON)
        assert world_after[1] == pytest.approx(world_before[1], abs=EPSILON)

        # And reverse: the invariant world point maps back to the exact same screen anchor
        screen_mapped = engine.world_to_screen(world_before[0], world_before[1])
        assert screen_mapped[0] == pytest.approx(anchor_x, abs=EPSILON)
        assert screen_mapped[1] == pytest.approx(anchor_y, abs=EPSILON)

    def test_zoom_scales_proportionally(self):
        """Zoom level scales by factor: zoom_new = zoom_old * factor."""
        engine = OrthoViewportEngine(initial_zoom=1.5, min_zoom=0.1, max_zoom=10.0)
        factor = 2.0

        engine.zoom_at(100.0, 100.0, factor)

        assert engine.zoom == pytest.approx(3.0, abs=EPSILON)

    def test_zoom_anchor_at_screen_origin(self):
        """When zooming anchored at screen (0, 0), offset scales directly with zoom factor."""
        initial_offset = (50.0, -80.0)
        engine = OrthoViewportEngine(
            initial_zoom=1.0,
            initial_offset=initial_offset,
            min_zoom=0.1,
            max_zoom=10.0,
        )

        factor = 2.0
        engine.zoom_at(0.0, 0.0, factor)

        # For anchor (0, 0): offset_new = offset_old * factor
        expected_offset = (initial_offset[0] * factor, initial_offset[1] * factor)
        assert engine.offset == pytest.approx(expected_offset, abs=EPSILON)

    def test_multiple_consecutive_anchored_zooms(self):
        """Successive zooms at varying anchor points consistently preserve world invariants."""
        engine = OrthoViewportEngine(min_zoom=0.01, max_zoom=100.0)

        anchors = [
            (100.0, 150.0, 1.25),
            (400.0, 300.0, 1.5),
            (50.0, 80.0, 0.8),
            (250.0, 250.0, 2.0),
        ]

        for ax, ay, factor in anchors:
            w_pre = engine.screen_to_world(ax, ay)
            engine.zoom_at(ax, ay, factor)
            w_post = engine.screen_to_world(ax, ay)

            assert w_post[0] == pytest.approx(w_pre[0], abs=EPSILON)
            assert w_post[1] == pytest.approx(w_pre[1], abs=EPSILON)

    def test_pan_combined_with_zoom_at_anchor(self):
        """World invariant is preserved after arbitrary pan translations followed by zoom."""
        engine = OrthoViewportEngine(min_zoom=0.1, max_zoom=10.0)

        engine.pan(234.5, -678.9)
        anchor_x, anchor_y = 512.0, 384.0

        world_before = engine.screen_to_world(anchor_x, anchor_y)
        engine.zoom_at(anchor_x, anchor_y, 1.75)
        world_after = engine.screen_to_world(anchor_x, anchor_y)

        assert world_after[0] == pytest.approx(world_before[0], abs=EPSILON)
        assert world_after[1] == pytest.approx(world_before[1], abs=EPSILON)

    @pytest.mark.parametrize("invalid_factor", [0.0, -1.0, -10.5])
    def test_non_positive_zoom_factor_raises(self, invalid_factor):
        """Applying non-positive zoom factor raises ValueError."""
        engine = OrthoViewportEngine()
        with pytest.raises(ValueError):
            engine.zoom_at(100.0, 100.0, invalid_factor)


class TestOrthoViewportEngineZoomClamping:
    """Acceptance Criteria:

    Given zoom bounds [min_zoom, max_zoom],
    When a zoom operation exceeds these boundaries,
    Then the zoom factor is clamped within the allowed range.
    """

    def test_zoom_clamped_to_max_zoom(self):
        """Zooming past max_zoom clamps the zoom level to max_zoom."""
        engine = OrthoViewportEngine(min_zoom=0.5, max_zoom=4.0, initial_zoom=2.0)
        anchor_x, anchor_y = 200.0, 150.0

        world_before = engine.screen_to_world(anchor_x, anchor_y)

        # 2.0 * 5.0 = 10.0 -> Clamped to 4.0
        engine.zoom_at(anchor_x, anchor_y, 5.0)

        assert engine.zoom == pytest.approx(4.0, abs=EPSILON)

        # Even with clamping, the world coordinate invariant must hold for effective zoom
        world_after = engine.screen_to_world(anchor_x, anchor_y)
        assert world_after[0] == pytest.approx(world_before[0], abs=EPSILON)
        assert world_after[1] == pytest.approx(world_before[1], abs=EPSILON)

    def test_zoom_clamped_to_min_zoom(self):
        """Zooming below min_zoom clamps the zoom level to min_zoom."""
        engine = OrthoViewportEngine(min_zoom=0.5, max_zoom=4.0, initial_zoom=1.0)
        anchor_x, anchor_y = 120.0, 80.0

        world_before = engine.screen_to_world(anchor_x, anchor_y)

        # 1.0 * 0.1 = 0.1 -> Clamped to 0.5
        engine.zoom_at(anchor_x, anchor_y, 0.1)

        assert engine.zoom == pytest.approx(0.5, abs=EPSILON)

        # Invariant maintained for the clamped scale
        world_after = engine.screen_to_world(anchor_x, anchor_y)
        assert world_after[0] == pytest.approx(world_before[0], abs=EPSILON)
        assert world_after[1] == pytest.approx(world_before[1], abs=EPSILON)

    def test_zoom_when_already_at_boundary_is_noop(self):
        """Zooming further outwards when already at min_zoom causes no change in zoom or offset."""
        engine = OrthoViewportEngine(min_zoom=0.5, max_zoom=4.0, initial_zoom=0.5)
        engine.pan(40.0, -30.0)

        initial_offset = engine.offset
        initial_zoom = engine.zoom

        engine.zoom_at(100.0, 100.0, 0.5)

        assert engine.zoom == pytest.approx(initial_zoom, abs=EPSILON)
        assert engine.offset == pytest.approx(initial_offset, abs=EPSILON)

    def test_zoom_exactly_at_boundary_values(self):
        """Zooming exactly to bounds works without overshoot or numeric drift."""
        engine = OrthoViewportEngine(min_zoom=0.25, max_zoom=4.0, initial_zoom=1.0)

        # Exactly reach max
        engine.zoom_at(10.0, 10.0, 4.0)
        assert engine.zoom == pytest.approx(4.0, abs=EPSILON)

        # Exactly reach min: 4.0 * (1/16) = 0.25
        engine.zoom_at(10.0, 10.0, 0.0625)
        assert engine.zoom == pytest.approx(0.25, abs=EPSILON)


class TestOrthoViewportEngineCoordinateRoundtrip:
    """Acceptance Criteria:

    Given arbitrary pan and zoom transformations,
    When converting coordinates from screen to world space and back to screen space,
    Then the output matches the original input within floating-point epsilon.
    """

    @pytest.fixture
    def transformed_engine(self):
        """Create an engine with complex pan and zoom state."""
        engine = OrthoViewportEngine(min_zoom=0.05, max_zoom=50.0)
        # Apply sequential pans and anchor zooms
        engine.pan(350.2, -180.7)
        engine.zoom_at(400.0, 300.0, 2.7)
        engine.pan(-45.3, 99.1)
        engine.zoom_at(120.0, 85.0, 0.65)
        return engine

    @pytest.mark.parametrize(
        "screen_x, screen_y",
        [
            (0.0, 0.0),
            (1920.0, 1080.0),
            (-500.0, 250.0),
            (320.5, -450.25),
            (1e6, -1e6),
            (-1e6, 1e6),
        ],
    )
    def test_screen_to_world_to_screen_roundtrip(self, transformed_engine, screen_x, screen_y):
        """Converting screen -> world -> screen recovers original screen coordinate."""
        world_x, world_y = transformed_engine.screen_to_world(screen_x, screen_y)
        roundtrip_sx, roundtrip_sy = transformed_engine.world_to_screen(world_x, world_y)

        assert roundtrip_sx == pytest.approx(screen_x, abs=EPSILON)
        assert roundtrip_sy == pytest.approx(screen_y, abs=EPSILON)

    @pytest.mark.parametrize(
        "world_x, world_y",
        [
            (0.0, 0.0),
            (100.0, -200.0),
            (-1500.75, 3450.125),
            (1e5, -1e5),
            (-5e5, 5e5),
        ],
    )
    def test_world_to_screen_to_world_roundtrip(self, transformed_engine, world_x, world_y):
        """Converting world -> screen -> world recovers original world coordinate."""
        screen_x, screen_y = transformed_engine.world_to_screen(world_x, world_y)
        roundtrip_wx, roundtrip_wy = transformed_engine.screen_to_world(screen_x, screen_y)

        assert roundtrip_wx == pytest.approx(world_x, abs=EPSILON)
        assert roundtrip_wy == pytest.approx(world_y, abs=EPSILON)

    def test_view_matrix_consistency_with_direct_transformations(self, transformed_engine):
        """world_to_screen transformation matches direct view_matrix.transform_point."""
        test_points = [
            (0.0, 0.0),
            (45.0, -80.0),
            (-123.4, 567.8),
        ]
        view_mat = transformed_engine.view_matrix

        for wx, wy in test_points:
            sx_engine, sy_engine = transformed_engine.world_to_screen(wx, wy)
            sx_matrix, sy_matrix = view_mat.transform_point(wx, wy)

            assert sx_engine == pytest.approx(sx_matrix, abs=EPSILON)
            assert sy_engine == pytest.approx(sy_matrix, abs=EPSILON)