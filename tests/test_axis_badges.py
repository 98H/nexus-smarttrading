import pytest
from typing import Callable

from src.ui.chart.axis_badges import BoundingBox, PriceBadge, TimeBadge
from src.ui.chart.crosshair_controller import CrosshairController, Viewport


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def price_formatter() -> Callable[[float], str]:
    """Default price formatting fixture ($X.XX)."""
    return lambda price: f"${price:.2f}"


@pytest.fixture
def time_formatter() -> Callable[[float], str]:
    """Default time formatting fixture (e.g. timestamp formatted as integer + 's')."""
    return lambda timestamp: f"{int(timestamp)}s"


@pytest.fixture
def default_price_badge(price_formatter: Callable[[float], str]) -> PriceBadge:
    """Creates a default PriceBadge with range [0.0, 600.0] and height 24.0."""
    return PriceBadge(
        axis_min=0.0,
        axis_max=600.0,
        height=24.0,
        formatter=price_formatter,
    )


@pytest.fixture
def default_time_badge(time_formatter: Callable[[float], str]) -> TimeBadge:
    """Creates a default TimeBadge with range [0.0, 800.0] and width 80.0."""
    return TimeBadge(
        axis_min=0.0,
        axis_max=800.0,
        width=80.0,
        formatter=time_formatter,
    )


@pytest.fixture
def default_viewport() -> Viewport:
    """Creates a plot area viewport with bounds: x in [0.0, 800.0], y in [0.0, 600.0]."""
    return Viewport(x_min=0.0, y_min=0.0, x_max=800.0, y_max=600.0)


@pytest.fixture
def price_transform() -> Callable[[float], float]:
    """Linear scale transforming Y pixel coordinate [0.0, 600.0] to price [200.0, 100.0]."""
    # y=0 -> 200.0, y=600 -> 100.0
    return lambda y: 200.0 - (y / 600.0) * 100.0


@pytest.fixture
def time_transform() -> Callable[[float], float]:
    """Linear scale transforming X pixel coordinate [0.0, 800.0] to epoch timestamp [1000.0, 1800.0]."""
    # x=0 -> 1000.0, x=800 -> 1800.0
    return lambda x: 1000.0 + x


@pytest.fixture
def crosshair_controller(
    default_viewport: Viewport,
    default_price_badge: PriceBadge,
    default_time_badge: TimeBadge,
    price_transform: Callable[[float], float],
    time_transform: Callable[[float], float],
) -> CrosshairController:
    """Instantiates a CrosshairController coordinating viewport, badges, and coordinate transforms."""
    return CrosshairController(
        viewport=default_viewport,
        price_badge=default_price_badge,
        time_badge=default_time_badge,
        price_transform=price_transform,
        time_transform=time_transform,
    )


# ============================================================================
# Unit Tests: Axis Badges Initialization & Validation
# ============================================================================

class TestBadgeInitialization:
    """Validates boundary rules and argument validations during badge instantiation."""

    def test_price_badge_invalid_axis_bounds_raises_error(self) -> None:
        """axis_min >= axis_max must raise ValueError."""
        with pytest.raises(ValueError):
            PriceBadge(axis_min=500.0, axis_max=100.0, height=20.0)

        with pytest.raises(ValueError):
            PriceBadge(axis_min=200.0, axis_max=200.0, height=20.0)

    def test_price_badge_invalid_height_raises_error(self) -> None:
        """height <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            PriceBadge(axis_min=0.0, axis_max=500.0, height=0.0)

        with pytest.raises(ValueError):
            PriceBadge(axis_min=0.0, axis_max=500.0, height=-15.0)

    def test_price_badge_height_exceeding_axis_length_raises_error(self) -> None:
        """Badge size larger than total axis span must raise ValueError."""
        with pytest.raises(ValueError):
            PriceBadge(axis_min=0.0, axis_max=50.0, height=60.0)

    def test_time_badge_invalid_axis_bounds_raises_error(self) -> None:
        """axis_min >= axis_max must raise ValueError."""
        with pytest.raises(ValueError):
            TimeBadge(axis_min=1000.0, axis_max=500.0, width=50.0)

    def test_time_badge_invalid_width_raises_error(self) -> None:
        """width <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            TimeBadge(axis_min=0.0, axis_max=1000.0, width=0.0)

        with pytest.raises(ValueError):
            TimeBadge(axis_min=0.0, axis_max=1000.0, width=-5.0)

    def test_time_badge_width_exceeding_axis_length_raises_error(self) -> None:
        """Badge size larger than axis span must raise ValueError."""
        with pytest.raises(ValueError):
            TimeBadge(axis_min=0.0, axis_max=40.0, width=50.0)

    def test_badge_default_initial_state(self, default_price_badge: PriceBadge) -> None:
        """Badges must initialize in a hidden state prior to any crosshair activity."""
        assert default_price_badge.visible is False
        assert default_price_badge.text == ""


# ============================================================================
# Unit Tests: Acceptance Criteria 1 - Coordinates Change and Badge Updates
# ============================================================================

class TestBadgeUpdatesOnCrosshairMovement:
    """Story 1.4.3 - AC 1:

    When the crosshair coordinates change, Then the price badge updates to display
    the formatted price at the cursor's Y-position and the time badge updates to display
    the formatted timestamp at the cursor's X-position.
    """

    def test_crosshair_movement_updates_badges_text_and_position(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        # Move cursor to X=400.0, Y=300.0
        # price_transform(300.0) = 200 - (300/600)*100 = 150.0 -> "$150.00"
        # time_transform(400.0) = 1000.0 + 400.0 = 1400.0 -> "1400s"
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)

        assert default_price_badge.visible is True
        assert default_price_badge.text == "$150.00"
        assert default_price_badge.coordinate == pytest.approx(300.0)

        assert default_time_badge.visible is True
        assert default_time_badge.text == "1400s"
        assert default_time_badge.coordinate == pytest.approx(400.0)

    def test_consecutive_movements_update_state_reactively(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=200.0, y=150.0)
        assert default_price_badge.text == "$175.00"
        assert default_time_badge.text == "1200s"

        # Subsequent movement
        crosshair_controller.on_mouse_move(x=600.0, y=450.0)
        assert default_price_badge.text == "$125.00"
        assert default_time_badge.text == "1600s"

    def test_badge_centered_bounding_box_unclamped(
        self,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        """When not constrained by axis boundaries, badge bounding boxes should be centered."""
        default_price_badge.update(coordinate=300.0, value=150.0)
        # Height is 24.0 -> [300 - 12, 300 + 12] = [288.0, 312.0]
        assert default_price_badge.bounding_box == BoundingBox(start=288.0, end=312.0)

        default_time_badge.update(coordinate=400.0, value=1400.0)
        # Width is 80.0 -> [400 - 40, 400 + 40] = [360.0, 440.0]
        assert default_time_badge.bounding_box == BoundingBox(start=360.0, end=440.0)


# ============================================================================
# Unit Tests: Acceptance Criteria 2 - Cursor Leaves Plot Area Boundary
# ============================================================================

class TestBadgesHiddenWhenCursorLeavesPlotArea:
    """Story 1.4.3 - AC 2:

    Given visible axis tracking badges, When the cursor leaves the plot area boundary,
    Then both the price and time tracking badges are set to hidden.
    """

    def test_cursor_leaves_viewport_left_hides_badges(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        # Move inside first to make them visible
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        assert default_price_badge.visible is True
        assert default_time_badge.visible is True

        # Move outside beyond left boundary (x < 0.0)
        crosshair_controller.on_mouse_move(x=-0.1, y=300.0)
        assert default_price_badge.visible is False
        assert default_time_badge.visible is False

    def test_cursor_leaves_viewport_right_hides_badges(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        # Move outside beyond right boundary (x > 800.0)
        crosshair_controller.on_mouse_move(x=800.1, y=300.0)
        assert default_price_badge.visible is False
        assert default_time_badge.visible is False

    def test_cursor_leaves_viewport_top_hides_badges(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        # Move outside beyond top boundary (y < 0.0)
        crosshair_controller.on_mouse_move(x=400.0, y=-0.1)
        assert default_price_badge.visible is False
        assert default_time_badge.visible is False

    def test_cursor_leaves_viewport_bottom_hides_badges(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        # Move outside beyond bottom boundary (y > 600.0)
        crosshair_controller.on_mouse_move(x=400.0, y=600.1)
        assert default_price_badge.visible is False
        assert default_time_badge.visible is False

    def test_explicit_on_mouse_leave_event_hides_badges(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        assert default_price_badge.visible is True
        assert default_time_badge.visible is True

        crosshair_controller.on_mouse_leave()
        assert default_price_badge.visible is False
        assert default_time_badge.visible is False

    def test_reentering_viewport_restores_badge_visibility(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        crosshair_controller.on_mouse_move(x=400.0, y=300.0)
        crosshair_controller.on_mouse_leave()
        assert default_price_badge.visible is False

        # Re-entry
        crosshair_controller.on_mouse_move(x=350.0, y=250.0)
        assert default_price_badge.visible is True
        assert default_time_badge.visible is True


# ============================================================================
# Unit Tests: Acceptance Criteria 3 - Clamping Near Viewport Edges
# ============================================================================

class TestBadgeBoundaryClamping:
    """Story 1.4.3 - AC 3:

    Given coordinates near the viewport edges, When badge positions are calculated,
    Then badge bounding boxes are clamped within the visible boundaries of their respective axes.
    """

    def test_price_badge_clamped_at_min_boundary(
        self,
        default_price_badge: PriceBadge,
    ) -> None:
        """Price badge positioned near axis_min (0.0).

        Height = 24.0.
        If center is 5.0, unclamped box = [-7.0, 17.0].
        Clamped box must be [0.0, 24.0].
        """
        default_price_badge.update(coordinate=5.0, value=195.0)

        box = default_price_badge.bounding_box
        assert box.start == pytest.approx(0.0)
        assert box.end == pytest.approx(24.0)

    def test_price_badge_clamped_at_exact_min_boundary(
        self,
        default_price_badge: PriceBadge,
    ) -> None:
        """Price badge positioned exactly at axis_min (0.0).

        Clamped box must be [0.0, 24.0].
        """
        default_price_badge.update(coordinate=0.0, value=200.0)

        box = default_price_badge.bounding_box
        assert box.start == pytest.approx(0.0)
        assert box.end == pytest.approx(24.0)

    def test_price_badge_clamped_at_max_boundary(
        self,
        default_price_badge: PriceBadge,
    ) -> None:
        """Price badge positioned near axis_max (600.0).

        Height = 24.0.
        If center is 595.0, unclamped box = [583.0, 607.0].
        Clamped box must be [576.0, 600.0].
        """
        default_price_badge.update(coordinate=595.0, value=102.0)

        box = default_price_badge.bounding_box
        assert box.start == pytest.approx(576.0)
        assert box.end == pytest.approx(600.0)

    def test_price_badge_clamped_at_exact_max_boundary(
        self,
        default_price_badge: PriceBadge,
    ) -> None:
        """Price badge positioned exactly at axis_max (600.0).

        Clamped box must be [576.0, 600.0].
        """
        default_price_badge.update(coordinate=600.0, value=100.0)

        box = default_price_badge.bounding_box
        assert box.start == pytest.approx(576.0)
        assert box.end == pytest.approx(600.0)

    def test_time_badge_clamped_at_min_boundary(
        self,
        default_time_badge: TimeBadge,
    ) -> None:
        """Time badge positioned near axis_min (0.0).

        Width = 80.0.
        If center is 20.0, unclamped box = [-20.0, 60.0].
        Clamped box must be [0.0, 80.0].
        """
        default_time_badge.update(coordinate=20.0, value=1020.0)

        box = default_time_badge.bounding_box
        assert box.start == pytest.approx(0.0)
        assert box.end == pytest.approx(80.0)

    def test_time_badge_clamped_at_exact_min_boundary(
        self,
        default_time_badge: TimeBadge,
    ) -> None:
        """Time badge positioned exactly at axis_min (0.0).

        Clamped box must be [0.0, 80.0].
        """
        default_time_badge.update(coordinate=0.0, value=1000.0)

        box = default_time_badge.bounding_box
        assert box.start == pytest.approx(0.0)
        assert box.end == pytest.approx(80.0)

    def test_time_badge_clamped_at_max_boundary(
        self,
        default_time_badge: TimeBadge,
    ) -> None:
        """Time badge positioned near axis_max (800.0).

        Width = 80.0.
        If center is 780.0, unclamped box = [740.0, 820.0].
        Clamped box must be [720.0, 800.0].
        """
        default_time_badge.update(coordinate=780.0, value=1780.0)

        box = default_time_badge.bounding_box
        assert box.start == pytest.approx(720.0)
        assert box.end == pytest.approx(800.0)

    def test_time_badge_clamped_at_exact_max_boundary(
        self,
        default_time_badge: TimeBadge,
    ) -> None:
        """Time badge positioned exactly at axis_max (800.0).

        Clamped box must be [720.0, 800.0].
        """
        default_time_badge.update(coordinate=800.0, value=1800.0)

        box = default_time_badge.bounding_box
        assert box.start == pytest.approx(720.0)
        assert box.end == pytest.approx(800.0)

    def test_crosshair_controller_edge_clamping_at_corners(
        self,
        crosshair_controller: CrosshairController,
        default_price_badge: PriceBadge,
        default_time_badge: TimeBadge,
    ) -> None:
        """Crosshair placed at top-left corner (0.0, 0.0) and bottom-right (800.0, 600.0)."""
        # Top-left corner
        crosshair_controller.on_mouse_move(x=0.0, y=0.0)
        assert default_price_badge.visible is True
        assert default_time_badge.visible is True
        assert default_price_badge.bounding_box == BoundingBox(start=0.0, end=24.0)
        assert default_time_badge.bounding_box == BoundingBox(start=0.0, end=80.0)

        # Bottom-right corner
        crosshair_controller.on_mouse_move(x=800.0, y=600.0)
        assert default_price_badge.visible is True
        assert default_time_badge.visible is True
        assert default_price_badge.bounding_box == BoundingBox(start=576.0, end=600.0)
        assert default_time_badge.bounding_box == BoundingBox(start=720.0, end=800.0)


# ============================================================================
# Unit Tests: Viewport Validation & Robustness
# ============================================================================

class TestViewportAndEdgeCases:
    """Validates Viewport constraints and arbitrary formatting behaviors."""

    def test_viewport_invalid_dimensions_raises_error(self) -> None:
        """Viewport requires x_min < x_max and y_min < y_max."""
        with pytest.raises(ValueError):
            Viewport(x_min=100.0, y_min=0.0, x_max=50.0, y_max=600.0)

        with pytest.raises(ValueError):
            Viewport(x_min=0.0, y_min=600.0, x_max=800.0, y_max=100.0)

    def test_viewport_contains_boundary_values(self, default_viewport: Viewport) -> None:
        """Viewport inclusion includes inclusive outer boundaries."""
        assert default_viewport.contains(0.0, 0.0) is True
        assert default_viewport.contains(800.0, 600.0) is True
        assert default_viewport.contains(400.0, 300.0) is True

        # Outside boundary
        assert default_viewport.contains(-0.001, 300.0) is False
        assert default_viewport.contains(800.001, 300.0) is False
        assert default_viewport.contains(400.0, -0.001) is False
        assert default_viewport.contains(400.0, 600.001) is False

    def test_custom_formatting_function_used(self) -> None:
        """Ensures custom formatters are invoked and formatted string matches expectations."""
        badge = PriceBadge(
            axis_min=0.0,
            axis_max=500.0,
            height=20.0,
            formatter=lambda p: f"PRICE: {p:.4f}",
        )
        badge.update(coordinate=250.0, value=12.345678)
        assert badge.text == "PRICE: 12.3457"

    def test_default_formatter_fallback(self) -> None:
        """When formatter is not provided, defaults to str(val)."""
        badge = PriceBadge(
            axis_min=0.0,
            axis_max=500.0,
            height=20.0,
            formatter=None,
        )
        badge.update(coordinate=250.0, value=123.45)
        assert badge.text == "123.45"