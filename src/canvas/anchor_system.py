"""Interactive anchor drag, hover detection, and transform recalculation system."""

from __future__ import annotations

import math
from typing import Optional

from src.canvas.models import (
    Anchor,
    AnchorState,
    CanvasObject,
    Point,
)


class AnchorSystemError(Exception):
    """Base exception for anchor system failures."""


class InvalidDragOperationError(AnchorSystemError):
    """Raised when drag lifecycle operations occur in invalid states."""


class AnchorSystem:
    """Manages pointer interactions, hover detection, and drag transforms for anchors."""

    def __init__(self) -> None:
        self._objects: list[CanvasObject] = []
        self._active_anchor: Optional[Anchor] = None
        self._active_object: Optional[CanvasObject] = None
        self._drag_start_anchor_position: Optional[Point] = None
        self._last_cursor_position: Optional[Point] = None

    @property
    def active_anchor(self) -> Optional[Anchor]:
        """Currently active anchor during a drag operation, if any."""
        return self._active_anchor

    def register_object(self, canvas_object: CanvasObject) -> None:
        """Register a canvas object for interaction tracking."""
        if not any(o is canvas_object for o in self._objects):
            self._objects.append(canvas_object)

    def unregister_object(self, canvas_object: CanvasObject) -> None:
        """Unregister a canvas object from interaction tracking."""
        self._objects = [o for o in self._objects if o is not canvas_object]

    def _all_anchors(self) -> list[Anchor]:
        """Return all anchors across registered objects."""
        anchors: list[Anchor] = []
        for obj in self._objects:
            anchors.extend(obj.anchors)
        return anchors

    def _find_anchor_owner(self, anchor: Anchor) -> Optional[CanvasObject]:
        """Locate the registered CanvasObject owning the specified anchor."""
        for obj in self._objects:
            if any(a is anchor for a in obj.anchors):
                return obj
        return None

    def _update_hover_states(self, cursor: Point) -> None:
        """Update hover state across all anchors based on hit radius and distance."""
        candidates: list[tuple[float, Anchor]] = []
        for anchor in self._all_anchors():
            dist = cursor.distance_to(anchor.position)
            if dist <= anchor.hit_radius or math.isclose(
                dist, anchor.hit_radius, rel_tol=1e-9, abs_tol=1e-9
            ):
                candidates.append((dist, anchor))

        if not candidates:
            for anchor in self._all_anchors():
                anchor.state = AnchorState.IDLE
            return

        closest_anchor = min(candidates, key=lambda item: item[0])[1]
        for anchor in self._all_anchors():
            anchor.state = (
                AnchorState.HOVERED if anchor is closest_anchor else AnchorState.IDLE
            )

    def handle_pointer_move(self, cursor: Point) -> None:
        """Handle pointer movements to update hover states when not actively dragging."""
        self._last_cursor_position = Point(cursor.x, cursor.y)
        if self._active_anchor is not None:
            return
        self._update_hover_states(cursor)

    def start_drag(self, cursor: Point) -> None:
        """Initiate drag on the currently hovered anchor."""
        if self._active_anchor is not None:
            raise InvalidDragOperationError("Cannot start drag while already dragging.")

        hovered_anchor: Optional[Anchor] = None
        for anchor in self._all_anchors():
            if anchor.state == AnchorState.HOVERED:
                hovered_anchor = anchor
                break

        if hovered_anchor is None:
            raise InvalidDragOperationError("Cannot start drag without a hovered anchor.")

        target_object = self._find_anchor_owner(hovered_anchor)
        if target_object is None:
            raise InvalidDragOperationError("Hovered anchor has no registered owner object.")

        self._active_anchor = hovered_anchor
        self._active_object = target_object
        self._active_anchor.state = AnchorState.DRAGGING
        self._drag_start_anchor_position = Point(
            hovered_anchor.position.x, hovered_anchor.position.y
        )
        self._last_cursor_position = Point(cursor.x, cursor.y)

    def drag_to(self, target_position: Point) -> None:
        """Update active anchor coordinates continuously during drag."""
        if self._active_anchor is None:
            raise InvalidDragOperationError("Cannot perform drag_to without active drag.")

        self._active_anchor.position = Point(target_position.x, target_position.y)
        self._last_cursor_position = Point(target_position.x, target_position.y)

    def commit_drag(self) -> None:
        """Commit active drag changes and recalculate target object transform."""
        if (
            self._active_anchor is None
            or self._active_object is None
            or self._drag_start_anchor_position is None
        ):
            raise InvalidDragOperationError("Cannot commit drag without active drag.")

        displacement_x = (
            self._active_anchor.position.x - self._drag_start_anchor_position.x
        )
        displacement_y = (
            self._active_anchor.position.y - self._drag_start_anchor_position.y
        )

        self._active_object.transform.position = Point(
            x=self._active_object.transform.position.x + displacement_x,
            y=self._active_object.transform.position.y + displacement_y,
        )

        committed_anchor = self._active_anchor
        self._active_anchor = None
        self._active_object = None
        self._drag_start_anchor_position = None

        if self._last_cursor_position is not None:
            self._update_hover_states(self._last_cursor_position)
        else:
            committed_anchor.state = AnchorState.IDLE

    def cancel_drag(self) -> None:
        """Cancel active drag, reverting coordinates and preserving original transform."""
        if self._active_anchor is None or self._drag_start_anchor_position is None:
            raise InvalidDragOperationError("Cannot cancel drag without active drag.")

        self._active_anchor.position = Point(
            x=self._drag_start_anchor_position.x,
            y=self._drag_start_anchor_position.y,
        )

        cancelled_anchor = self._active_anchor
        self._active_anchor = None
        self._active_object = None
        self._drag_start_anchor_position = None

        if self._last_cursor_position is not None:
            self._update_hover_states(self._last_cursor_position)
        else:
            cancelled_anchor.state = AnchorState.IDLE