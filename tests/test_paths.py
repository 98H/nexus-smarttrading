import pytest
from src.canvas.paths import (
    BlendMode,
    Path,
    Point,
    record_freehand_brush,
    record_highlight_path,
)
from src.canvas.polygon import Polygon


def _extract_coordinates(points):
    """Helper to extract (x, y) coordinate tuples from Point objects or coordinate tuples."""
    coords = []
    for p in points:
        if hasattr(p, "x") and hasattr(p, "y"):
            coords.append((float(p.x), float(p.y)))
        else:
            coords.append((float(p[0]), float(p[1])))
    return coords


# ============================================================================
# Feature: Freehand Brush Stroke Tests
# ============================================================================


class TestFreehandBrush:
    """Unit tests for recording freehand brush strokes (Story 5.2.3)."""

    def test_record_freehand_brush_creates_path_instance(self):
        points = [(0.0, 0.0), (5.0, 10.0), (10.0, 20.0)]
        color = "#FF0000"
        stroke_width = 2.5

        path = record_freehand_brush(points=points, color=color, stroke_width=stroke_width)

        assert isinstance(path, Path)

    def test_record_freehand_brush_preserves_ordered_coordinates(self):
        input_coords = [(10.5, 20.2), (30.1, 40.8), (15.0, 25.0), (5.0, 8.0)]
        color = "#00FF00"
        stroke_width = 3.0

        path = record_freehand_brush(points=input_coords, color=color, stroke_width=stroke_width)

        extracted = _extract_coordinates(path.points)
        assert extracted == input_coords

    def test_record_freehand_brush_accepts_point_objects(self):
        point_objs = [Point(1.0, 2.0), Point(3.0, 4.0), Point(5.0, 6.0)]
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

        path = record_freehand_brush(points=point_objs, color="#123456", stroke_width=1.0)

        assert _extract_coordinates(path.points) == expected

    def test_record_freehand_brush_opaque_blend_mode_and_alpha(self):
        points = [(0.0, 0.0), (10.0, 10.0)]
        path = record_freehand_brush(points=points, color="#000000", stroke_width=1.0)

        # Blend mode must represent opaque rendering
        assert path.blend_mode in {
            BlendMode.NORMAL,
            getattr(BlendMode, "OPAQUE", BlendMode.NORMAL),
            "normal",
            "opaque",
        }
        # Alpha must be fully opaque
        assert path.alpha == 1.0

    def test_record_freehand_brush_attributes_preserved(self):
        points = [(0.0, 0.0), (1.0, 1.0)]
        color = "rgba(255, 0, 0, 1.0)"
        stroke_width = 4.75

        path = record_freehand_brush(points=points, color=color, stroke_width=stroke_width)

        assert path.color == color
        assert path.stroke_width == stroke_width

    def test_record_freehand_brush_input_points_immutability(self):
        input_coords = [(1.0, 1.0), (2.0, 2.0)]
        path = record_freehand_brush(points=input_coords, color="#000000", stroke_width=1.0)

        # Mutating the input sequence after recording must not alter the path coordinates
        input_coords.append((99.0, 99.0))
        assert len(path.points) == 2
        assert _extract_coordinates(path.points) == [(1.0, 1.0), (2.0, 2.0)]

    def test_record_freehand_brush_empty_points_raises_value_error(self):
        with pytest.raises(ValueError):
            record_freehand_brush(points=[], color="#000000", stroke_width=1.0)

    @pytest.mark.parametrize("invalid_width", [0.0, -1.0, -0.01])
    def test_record_freehand_brush_non_positive_stroke_width_raises_value_error(
        self, invalid_width
    ):
        with pytest.raises(ValueError):
            record_freehand_brush(
                points=[(0.0, 0.0), (1.0, 1.0)], color="#000000", stroke_width=invalid_width
            )


# ============================================================================
# Feature: Highlighting Path Tests
# ============================================================================


class TestHighlightingPath:
    """Unit tests for recording highlighting paths (Story 5.2.3)."""

    def test_record_highlight_path_creates_path_instance(self):
        points = [(10.0, 10.0), (50.0, 10.0)]
        color = "#FFFF00"
        stroke_width = 12.0

        path = record_highlight_path(points=points, color=color, stroke_width=stroke_width)

        assert isinstance(path, Path)

    def test_record_highlight_path_preserves_ordered_coordinates(self):
        input_coords = [(0.0, 5.0), (20.0, 15.0), (40.0, 10.0), (80.0, 25.0)]
        path = record_highlight_path(points=input_coords, color="#FFFF00", stroke_width=10.0)

        assert _extract_coordinates(path.points) == input_coords

    def test_record_highlight_path_multiply_blend_mode(self):
        points = [(0.0, 0.0), (10.0, 20.0)]
        path = record_highlight_path(points=points, color="#FFFF00", stroke_width=8.0)

        assert path.blend_mode in {BlendMode.MULTIPLY, "multiply"}

    def test_record_highlight_path_semi_transparent_alpha(self):
        points = [(0.0, 0.0), (10.0, 20.0)]
        path = record_highlight_path(points=points, color="#FFFF00", stroke_width=8.0)

        # Alpha must be strictly semi-transparent
        assert 0.0 < path.alpha < 1.0

    def test_record_highlight_path_custom_semi_transparent_alpha(self):
        points = [(0.0, 0.0), (10.0, 20.0)]
        custom_alpha = 0.35
        path = record_highlight_path(
            points=points, color="#FFFF00", stroke_width=8.0, alpha=custom_alpha
        )

        assert path.alpha == pytest.approx(custom_alpha)

    @pytest.mark.parametrize("invalid_alpha", [0.0, 1.0, 1.5, -0.1])
    def test_record_highlight_path_non_semi_transparent_alpha_raises_value_error(
        self, invalid_alpha
    ):
        with pytest.raises(ValueError):
            record_highlight_path(
                points=[(0.0, 0.0), (10.0, 20.0)],
                color="#FFFF00",
                stroke_width=8.0,
                alpha=invalid_alpha,
            )

    def test_record_highlight_path_attributes_preserved(self):
        points = [(5.0, 5.0), (15.0, 25.0)]
        color = "#00FFFF"
        stroke_width = 14.5

        path = record_highlight_path(points=points, color=color, stroke_width=stroke_width)

        assert path.color == color
        assert path.stroke_width == stroke_width

    def test_record_highlight_path_input_points_immutability(self):
        input_coords = [(0.0, 0.0), (10.0, 0.0)]
        path = record_highlight_path(points=input_coords, color="#FFFF00", stroke_width=5.0)

        input_coords.append((20.0, 0.0))
        assert len(path.points) == 2
        assert _extract_coordinates(path.points) == [(0.0, 0.0), (10.0, 0.0)]

    def test_record_highlight_path_empty_points_raises_value_error(self):
        with pytest.raises(ValueError):
            record_highlight_path(points=[], color="#FFFF00", stroke_width=5.0)

    @pytest.mark.parametrize("invalid_width", [0.0, -5.0])
    def test_record_highlight_path_invalid_stroke_width_raises_value_error(self, invalid_width):
        with pytest.raises(ValueError):
            record_highlight_path(
                points=[(0.0, 0.0), (1.0, 1.0)], color="#FFFF00", stroke_width=invalid_width
            )


# ============================================================================
# Feature: Polygon Tests
# ============================================================================


class TestPolygon:
    """Unit tests for Polygon validation and auto-closing (Story 5.2.3)."""

    @pytest.mark.parametrize(
        "invalid_vertices",
        [
            [],
            [(0.0, 0.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ],
    )
    def test_polygon_fewer_than_three_vertices_raises_value_error(self, invalid_vertices):
        with pytest.raises(ValueError):
            Polygon(vertices=invalid_vertices)

    @pytest.mark.parametrize(
        "malformed_vertex",
        [
            [(0.0,), (1.0, 1.0), (2.0, 2.0)],  # 1D vertex
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],  # 3D vertex
            ["invalid", (1.0, 1.0), (2.0, 2.0)],  # non-numeric
        ],
    )
    def test_polygon_invalid_vertex_dimensions_raises_value_error(self, malformed_vertex):
        with pytest.raises(ValueError):
            Polygon(vertices=malformed_vertex)

    def test_polygon_triangle_auto_closes_to_origin(self):
        vertices = [(0.0, 0.0), (5.0, 0.0), (2.5, 4.0)]
        polygon = Polygon(vertices=vertices)

        # Polygon must provide closed coordinates connecting the final vertex to the origin
        closed_coords = _extract_coordinates(
            polygon.vertices if hasattr(polygon, "vertices") else polygon.points
        )

        assert len(closed_coords) == 4
        assert closed_coords[0] == (0.0, 0.0)
        assert closed_coords[1] == (5.0, 0.0)
        assert closed_coords[2] == (2.5, 4.0)
        assert closed_coords[3] == closed_coords[0]

    def test_polygon_quadrilateral_auto_closes_to_origin(self):
        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        polygon = Polygon(vertices=vertices)

        closed_coords = _extract_coordinates(
            polygon.vertices if hasattr(polygon, "vertices") else polygon.points
        )

        assert len(closed_coords) == 5
        assert closed_coords[:4] == vertices
        assert closed_coords[-1] == vertices[0]

    def test_polygon_already_closed_input_does_not_duplicate_closing_segment(self):
        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]
        polygon = Polygon(vertices=vertices)

        closed_coords = _extract_coordinates(
            polygon.vertices if hasattr(polygon, "vertices") else polygon.points
        )

        assert len(closed_coords) == 4
        assert closed_coords[0] == closed_coords[-1]

    def test_polygon_origin_vertex_property(self):
        vertices = [(12.0, 34.0), (56.0, 78.0), (90.0, 12.0)]
        polygon = Polygon(vertices=vertices)

        origin = polygon.origin if hasattr(polygon, "origin") else polygon.vertices[0]
        origin_coord = (
            (float(origin.x), float(origin.y))
            if hasattr(origin, "x")
            else (float(origin[0]), float(origin[1]))
        )

        assert origin_coord == (12.0, 34.0)

    def test_polygon_path_representation_is_closed(self):
        vertices = [(1.0, 1.0), (5.0, 1.0), (3.0, 4.0)]
        polygon = Polygon(vertices=vertices)

        if hasattr(polygon, "path") and isinstance(polygon.path, Path):
            path_coords = _extract_coordinates(polygon.path.points)
            assert len(path_coords) == 4
            assert path_coords[0] == path_coords[-1]
            assert path_coords[0] == (1.0, 1.0)

    def test_polygon_accepts_point_objects(self):
        point_objs = [Point(0.0, 0.0), Point(4.0, 0.0), Point(4.0, 3.0)]
        polygon = Polygon(vertices=point_objs)

        closed_coords = _extract_coordinates(
            polygon.vertices if hasattr(polygon, "vertices") else polygon.points
        )
        assert closed_coords == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 0.0)]

    def test_polygon_input_vertices_immutability(self):
        input_vertices = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
        polygon = Polygon(vertices=input_vertices)

        input_vertices.append((100.0, 100.0))
        closed_coords = _extract_coordinates(
            polygon.vertices if hasattr(polygon, "vertices") else polygon.points
        )

        assert len(closed_coords) == 4
        assert closed_coords[-1] == (0.0, 0.0)