"""Polygon module for validated, auto-closed 2D polygons."""

from __future__ import annotations

from typing import Sequence

from src.canvas.paths import BlendMode, Path, Point


class Polygon:
    """Represents a closed 2D polygon with at least three distinct vertices."""

    def __init__(
        self,
        vertices: Sequence[Point | tuple[float, float]],
        color: str = "#000000",
        stroke_width: float = 1.0,
    ) -> None:
        if vertices is None:
            raise ValueError("Vertices sequence cannot be None")

        parsed = [Point.from_raw(v) for v in vertices]

        if len(parsed) >= 2 and parsed[0] == parsed[-1]:
            if len(parsed) - 1 < 3:
                raise ValueError("A polygon must have at least 3 vertices")
            closed = list(parsed)
        else:
            if len(parsed) < 3:
                raise ValueError("A polygon must have at least 3 vertices")
            closed = list(parsed) + [parsed[0]]

        self.vertices: list[Point] = closed
        self.points: list[Point] = self.vertices
        self.origin: Point = self.vertices[0]
        self.color: str = color
        self.stroke_width: float = stroke_width
        self.path: Path = Path(
            points=list(self.vertices),
            color=color,
            stroke_width=stroke_width,
            blend_mode=BlendMode.NORMAL,
            alpha=1.0,
        )

    def __repr__(self) -> str:
        return f"Polygon(vertices={self.vertices!r})"