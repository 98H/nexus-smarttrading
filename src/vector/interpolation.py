"""Pipeline for vector shape interpolation."""

from __future__ import annotations

from src.vector.math import VectorShape, lerp


def interpolate_shapes(
    shape_a: VectorShape, shape_b: VectorShape, t: float
) -> VectorShape:
    """Pairwise interpolate between two VectorShapes at factor t in [0.0, 1.0]."""
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Interpolation factor t must be in [0.0, 1.0], got {t}")

    if len(shape_a.vertices) != len(shape_b.vertices):
        raise ValueError(
            f"Shape vertex counts do not match: {len(shape_a.vertices)} != {len(shape_b.vertices)}"
        )

    interpolated_vertices = [
        lerp(vertex_a, vertex_b, t)
        for vertex_a, vertex_b in zip(shape_a.vertices, shape_b.vertices)
    ]
    return VectorShape(interpolated_vertices)