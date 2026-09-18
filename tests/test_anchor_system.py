import math
import pytest

from src.canvas.models import (
    Anchor,
    AnchorState,
    CanvasObject,
    Point,
    Transform,
)
from src.canvas.anchor_system import (
    AnchorSystem,
    AnchorSystemError,
    InvalidDragOperationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def initial_point() -> Point:
    return Point(x=100.0, y=100.0)


@pytest.fixture
def initial_transform(initial_point: Point) -> Transform:
    return Transform(
        position=Point(x=initial_point.x, y=initial_point.y),
        scale=Point(x=1.0, y=1.0),
        rotation=0.0,
    )


@pytest.fixture
def default_anchor(initial_point: Point) -> Anchor:
    return Anchor(
        id="anchor-1",
        position=Point(x=initial_point.x, y=initial_point.y),
        hit_radius=10.0,
    )


@pytest.fixture
def canvas_object(initial_transform: Transform, default_anchor: Anchor) -> CanvasObject:
    return CanvasObject(
        id="object-1",
        transform=initial_transform,
        anchors=[default_anchor],
    )


@pytest.fixture
def anchor_system(canvas_object: CanvasObject) -> AnchorSystem:
    system = AnchorSystem()
    system.register_object(canvas_object)
    return system


# ============================================================================
# Acceptance Criteria 1: Cursor Hover & Hit Radius
# ============================================================================

class TestAnchorHover:
    """Tests verifying hit radius detection and hover state transitions."""

    def test_cursor_inside_hit_radius_transitions_to_hovered(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        # Anchor at (100, 100) with radius 10. Point at (105, 100) is distance 5.
        cursor = Point(x=105.0, y=100.0)

        anchor_system.handle_pointer_move(cursor)

        assert default_anchor.state == AnchorState.HOVERED

    def test_cursor_exactly_on_boundary_transitions_to_hovered(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        # Anchor at (100, 100) with radius 10. Point on boundary (110, 100)
        cursor = Point(x=110.0, y=100.0)

        anchor_system.handle_pointer_move(cursor)

        assert default_anchor.state == AnchorState.HOVERED

    def test_cursor_on_diagonal_boundary_transitions_to_hovered(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        # Distance sqrt(dx^2 + dy^2) == 10
        offset = 10.0 / math.sqrt(2)
        cursor = Point(x=100.0 + offset, y=100.0 + offset)

        anchor_system.handle_pointer_move(cursor)

        assert default_anchor.state == AnchorState.HOVERED

    def test_cursor_outside_hit_radius_remains_idle(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        # Anchor at (100, 100) with radius 10. Point at (110.01, 100)
        cursor = Point(x=110.01, y=100.0)

        anchor_system.handle_pointer_move(cursor)

        assert default_anchor.state == AnchorState.IDLE

    def test_cursor_exiting_hit_radius_transitions_from_hovered_to_idle(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        inside_cursor = Point(x=102.0, y=100.0)
        outside_cursor = Point(x=150.0, y=100.0)

        anchor_system.handle_pointer_move(inside_cursor)
        assert default_anchor.state == AnchorState.HOVERED

        anchor_system.handle_pointer_move(outside_cursor)
        assert default_anchor.state == AnchorState.IDLE

    def test_closest_anchor_hovered_when_multiple_in_range(self):
        system = AnchorSystem()
        anchor_a = Anchor(id="a", position=Point(100.0, 100.0), hit_radius=15.0)
        anchor_b = Anchor(id="b", position=Point(105.0, 100.0), hit_radius=15.0)
        obj = CanvasObject(
            id="multi-obj",
            transform=Transform(Point(100.0, 100.0)),
            anchors=[anchor_a, anchor_b],
        )
        system.register_object(obj)

        # Cursor at (104.0, 100.0) -> distance to A is 4.0, distance to B is 1.0
        system.handle_pointer_move(Point(104.0, 100.0))

        assert anchor_b.state == AnchorState.HOVERED
        assert anchor_a.state == AnchorState.IDLE


# ============================================================================
# Acceptance Criteria 2: Drag Event & Coordinate Update
# ============================================================================

class TestAnchorDrag:
    """Tests verifying drag initiation and continuous coordinate tracking."""

    def test_drag_starts_from_hovered_anchor_transitions_to_dragging(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        hover_cursor = Point(x=102.0, y=100.0)
        anchor_system.handle_pointer_move(hover_cursor)
        assert default_anchor.state == AnchorState.HOVERED

        anchor_system.start_drag(hover_cursor)

        assert default_anchor.state == AnchorState.DRAGGING
        assert anchor_system.active_anchor == default_anchor

    def test_start_drag_without_hovered_anchor_raises_exception(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        outside_cursor = Point(x=200.0, y=200.0)
        anchor_system.handle_pointer_move(outside_cursor)

        with pytest.raises(InvalidDragOperationError):
            anchor_system.start_drag(outside_cursor)

        assert default_anchor.state == AnchorState.IDLE
        assert anchor_system.active_anchor is None

    def test_drag_move_updates_anchor_coordinates_to_target(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        hover_cursor = Point(x=100.0, y=100.0)
        anchor_system.handle_pointer_move(hover_cursor)
        anchor_system.start_drag(hover_cursor)

        target_position = Point(x=175.5, y=210.25)
        anchor_system.drag_to(target_position)

        assert default_anchor.position.x == pytest.approx(175.5)
        assert default_anchor.position.y == pytest.approx(210.25)
        assert default_anchor.state == AnchorState.DRAGGING

    def test_drag_move_continuous_updates(
        self, anchor_system: AnchorSystem, default_anchor: Anchor
    ):
        anchor_system.handle_pointer_move(Point(100.0, 100.0))
        anchor_system.start_drag(Point(100.0, 100.0))

        waypoints = [
            Point(x=120.0, y=110.0),
            Point(x=145.0, y=130.0),
            Point(x=90.0, y=85.0),
        ]

        for pt in waypoints:
            anchor_system.drag_to(pt)
            assert default_anchor.position.x == pytest.approx(pt.x)
            assert default_anchor.position.y == pytest.approx(pt.y)

    def test_drag_to_without_active_drag_raises_exception(
        self, anchor_system: AnchorSystem
    ):
        with pytest.raises(InvalidDragOperationError):
            anchor_system.drag_to(Point(x=150.0, y=150.0))

    def test_cannot_start_secondary_drag_while_already_dragging(
        self, anchor_system: AnchorSystem
    ):
        anchor_system.handle_pointer_move(Point(100.0, 100.0))
        anchor_system.start_drag(Point(100.0, 100.0))

        with pytest.raises(InvalidDragOperationError):
            anchor_system.start_drag(Point(100.0, 100.0))


# ============================================================================
# Acceptance Criteria 3: Commit Drag & Target Transform Recalculation
# ============================================================================

class TestAnchorTransformRecalculation:
    """Tests verifying target object transform recalculation upon committing drag."""

    def test_commit_drag_recalculates_target_transform_position(
        self,
        anchor_system: AnchorSystem,
        canvas_object: CanvasObject,
        default_anchor: Anchor,
    ):
        initial_obj_x = canvas_object.transform.position.x
        initial_obj_y = canvas_object.transform.position.y

        # Hover and start drag
        anchor_system.handle_pointer_move(Point(100.0, 100.0))
        anchor_system.start_drag(Point(100.0, 100.0))

        # Displace anchor by dx = +35.0, dy = -20.0
        target_point = Point(x=135.0, y=80.0)
        anchor_system.drag_to(target_point)

        # Commit changes
        anchor_system.commit_drag()

        expected_x = initial_obj_x + 35.0
        expected_y = initial_obj_y - 20.0

        assert canvas_object.transform.position.x == pytest.approx(expected_x)
        assert canvas_object.transform.position.y == pytest.approx(expected_y)
        assert default_anchor.position.x == pytest.approx(135.0)
        assert default_anchor.position.y == pytest.approx(80.0)

    def test_commit_drag_transitions_state_from_dragging_to_hovered_if_under_cursor(
        self,
        anchor_system: AnchorSystem,
        default_anchor: Anchor,
    ):
        anchor_system.handle_pointer_move(Point(100.0, 100.0))
        anchor_system.start_drag(Point(100.0, 100.0))
        anchor_system.drag_to(Point(120.0, 120.0))

        anchor_system.commit_drag()

        assert anchor_system.active_anchor is None
        assert default_anchor.state == AnchorState.HOVERED

    def test_commit_drag_with_zero_displacement_leaves_transform_invariant(
        self,
        anchor_system: AnchorSystem,
        canvas_object: CanvasObject,
        default_anchor: Anchor,
    ):
        orig_pos = Point(canvas_object.transform.position.x, canvas_object.transform.position.y)

        anchor_system.handle_pointer_move(Point(100.0, 100.0))
        anchor_system.start_drag(Point(100.0, 100.0))
        anchor_system.drag_to(Point(150.0, 150.0))
        # Move back to start
        anchor_system.drag_to(Point(100.0, 100.0))
        anchor_system.commit_drag()

        assert canvas_object.transform.position.x == pytest.approx(orig_pos.x)
        assert canvas_object.transform.position.y == pytest.approx(orig_pos.y)

    def test_commit_drag_without_prior_drag_raises_exception(
        self, anchor_system: AnchorSystem
    ):
        with pytest.raises(InvalidDragOperationError):
            anchor_system.commit_drag()

    def test_cancel_drag_reverts_anchor_and_preserves_original_transform(
        self,
        anchor_system: AnchorSystem,
        canvas_object: CanvasObject,
        default_anchor: Anchor,
    ):
        orig_anchor_x = default_anchor.position.x
        orig_anchor_y = default_anchor.position.y
        orig_transform_x = canvas_object.transform.position.x
        orig_transform_y = canvas_object.transform.position.y

        anchor_system.handle_pointer_move(Point(orig_anchor_x, orig_anchor_y))
        anchor_system.start_drag(Point(orig_anchor_x, orig_anchor_y))
        anchor_system.drag_to(Point(250.0, 300.0))

        anchor_system.cancel_drag()

        assert default_anchor.position.x == pytest.approx(orig_anchor_x)
        assert default_anchor.position.y == pytest.approx(orig_anchor_y)
        assert canvas_object.transform.position.x == pytest.approx(orig_transform_x)
        assert canvas_object.transform.position.y == pytest.approx(orig_transform_y)
        assert default_anchor.state != AnchorState.DRAGGING
        assert anchor_system.active_anchor is None

    def test_multiple_sequential_drags_accumulate_transform(
        self,
        anchor_system: AnchorSystem,
        canvas_object: CanvasObject,
        default_anchor: Anchor,
    ):
        initial_x = canvas_object.transform.position.x
        initial_y = canvas_object.transform.position.y

        # Drag 1: +10, +10
        anchor_system.handle_pointer_move(default_anchor.position)
        anchor_system.start_drag(default_anchor.position)
        anchor_system.drag_to(Point(default_anchor.position.x + 10.0, default_anchor.position.y + 10.0))
        anchor_system.commit_drag()

        # Drag 2: +20, -5
        anchor_system.handle_pointer_move(default_anchor.position)
        anchor_system.start_drag(default_anchor.position)
        anchor_system.drag_to(Point(default_anchor.position.x + 20.0, default_anchor.position.y - 5.0))
        anchor_system.commit_drag()

        assert canvas_object.transform.position.x == pytest.approx(initial_x + 30.0)
        assert canvas_object.transform.position.y == pytest.approx(initial_y + 5.0)


# ============================================================================
# Validation & Edge Cases
# ============================================================================

class TestAnchorValidation:
    """Tests covering domain constraints and boundary validations."""

    def test_anchor_hit_radius_must_be_positive(self):
        with pytest.raises(ValueError):
            Anchor(id="invalid-radius", position=Point(0.0, 0.0), hit_radius=-1.0)

    def test_anchor_hit_radius_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            Anchor(id="zero-radius", position=Point(0.0, 0.0), hit_radius=0.0)

    def test_unregistered_object_does_not_respond_to_events(self):
        unregistered_anchor = Anchor(id="ghost", position=Point(50.0, 50.0), hit_radius=10.0)
        unregistered_obj = CanvasObject(
            id="ghost-obj",
            transform=Transform(Point(50.0, 50.0)),
            anchors=[unregistered_anchor],
        )
        system = AnchorSystem()

        system.handle_pointer_move(Point(50.0, 50.0))

        assert unregistered_anchor.state == AnchorState.IDLE
        assert system.active_anchor is None