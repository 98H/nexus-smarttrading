"""Domain models for canvas elements, transforms, and interaction anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Point:
    """Represents a 2D coordinate on the canvas."""

    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: Point) -> float:
        """Calculate the Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Transform:
    """Spatial transformation representing position, scale, and rotation."""

    position: Point
    scale: Point = field(default_factory=lambda: Point(1.0, 1.0))
    rotation: float = 0.0


class AnchorState(str, Enum):
    """Interaction lifecycle states for canvas anchors."""

    IDLE = "idle"
    HOVERED = "hovered"
    DRAGGING = "dragging"


@dataclass
class Anchor:
    """Interactive control point attached to canvas objects."""

    id: str
    position: Point
    hit_radius: float
    state: AnchorState = AnchorState.IDLE

    def __post_init__(self) -> None:
        if self.hit_radius <= 0:
            raise ValueError(f"hit_radius must be positive, got {self.hit_radius}")


@dataclass
class CanvasObject:
    """Entity positioned on the canvas containing transforms and control anchors."""

    id: str
    transform: Transform
    anchors: list[Anchor] = field(default_factory=list)