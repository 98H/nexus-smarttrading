import pytest
from src.vector.math import Point2D, VectorShape, lerp
from src.vector.interpolation import interpolate_shapes


# ============================================================================
# Helper Functions
# ============================================================================

def assert_point_approx_equal(
    actual: Point2D, expected: Point2D, rel: float = 1e-7, abs_tol: float = 1e-7
) -> None:
    """Assert that two Point2D instances have approximately equal coordinates."""
    assert actual.x == pytest.approx(expected.x, rel=rel, abs=abs_tol)
    assert actual.y == pytest.approx(expected.y, rel=rel, abs=abs_tol)


def assert_shape_approx_equal(
    actual: VectorShape, expected: VectorShape, rel: float = 1e-7, abs_tol: float = 1e-7
) -> None:
    """Assert that two VectorShape instances have identical vertex counts and pairwise matching coordinates."""
    assert len(actual.vertices) == len(expected.vertices)
    for act_pt, exp_pt in zip(actual.vertices, expected.vertices):
        assert_point_approx_equal(act_pt, exp_pt, rel=rel, abs_tol=abs_tol)


# ============================================================================
# AC 1: Point Lerp Tests (src/vector/math.py)
# ============================================================================

class TestPoint2DMath:
    """Unit tests for Point2D representation and linear interpolation (lerp)."""

    def test_point2d_initialization_and_attributes(self) -> None:
        """Verify Point2D correctly stores x and y coordinates."""
        point = Point2D(3.5, -7.2)
        assert point.x == 3.5
        assert point.y == -7.2

    def test_point2d_equality(self) -> None:
        """Verify Point2D supports value-based equality comparison."""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0, 2.0)
        p3 = Point2D(2.0, 1.0)
        assert p1 == p2
        assert p1 != p3

    @pytest.mark.parametrize(
        ("t", "point_a", "point_b", "expected"),
        [
            # t = 0.0 returns point A exactly
            (0.0, Point2D(0.0, 0.0), Point2D(10.0, 20.0), Point2D(0.0, 0.0)),
            (0.0, Point2D(-5.0, 15.0), Point2D(5.0, -15.0), Point2D(-5.0, 15.0)),
            # t = 1.0 returns point B exactly
            (1.0, Point2D(0.0, 0.0), Point2D(10.0, 20.0), Point2D(10.0, 20.0)),
            (1.0, Point2D(-5.0, 15.0), Point2D(5.0, -15.0), Point2D(5.0, -15.0)),
            # t = 0.5 returns the midpoint
            (0.5, Point2D(0.0, 0.0), Point2D(10.0, 20.0), Point2D(5.0, 10.0)),
            (0.5, Point2D(-10.0, -20.0), Point2D(10.0, 20.0), Point2D(0.0, 0.0)),
            # Arbitrary factors in [0.0, 1.0]
            (0.25, Point2D(0.0, 0.0), Point2D(8.0, 16.0), Point2D(2.0, 4.0)),
            (0.75, Point2D(4.0, 8.0), Point2D(12.0, 16.0), Point2D(10.0, 14.0)),
            (0.2, Point2D(1.0, 3.0), Point2D(6.0, 8.0), Point2D(2.0, 4.0)),
            # Non-trivial floating point coordinates
            (0.33, Point2D(1.5, 2.5), Point2D(4.5, 8.5), Point2D(2.49, 4.48)),
        ],
    )
    def test_lerp_evaluated_correctly(
        self, t: float, point_a: Point2D, point_b: Point2D, expected: Point2D
    ) -> None:
        """Given points A and B and t in [0.0, 1.0], verify lerp computes linear interpolation."""
        result = lerp(point_a, point_b, t)
        assert isinstance(result, Point2D)
        assert_point_approx_equal(result, expected)

    def test_lerp_identical_points(self) -> None:
        """Verify lerping between identical points returns the same point for any t in [0.0, 1.0]."""
        pt = Point2D(42.0, -13.5)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = lerp(pt, pt, t)
            assert_point_approx_equal(result, pt)

    def test_lerp_negative_and_zero_coordinates(self) -> None:
        """Verify lerp handles negative coordinates and crossing zero."""
        pt_a = Point2D(-100.0, -50.0)
        pt_b = Point2D(100.0, 50.0)
        midpoint = lerp(pt_a, pt_b, 0.5)
        assert_point_approx_equal(midpoint, Point2D(0.0, 0.0))


# ============================================================================
# AC 2: Shape Pairwise Interpolation Tests (src/vector/interpolation.py)
# ============================================================================

class TestVectorShapeInterpolation:
    """Unit tests for vector shape representation and pairwise interpolation."""

    def test_vector_shape_initialization_and_vertices(self) -> None:
        """Verify VectorShape correctly stores vertices as a sequence."""
        vertices = [Point2D(0.0, 0.0), Point2D(1.0, 2.0), Point2D(3.0, 4.0)]
        shape = VectorShape(vertices)
        assert len(shape.vertices) == 3
        for actual, expected in zip(shape.vertices, vertices):
            assert actual == expected

    def test_interpolate_shapes_at_factor_zero_returns_shape_a(self) -> None:
        """Given factor t = 0.0, produce a shape identical to shape A."""
        shape_a = VectorShape([Point2D(0.0, 0.0), Point2D(10.0, 5.0), Point2D(20.0, 30.0)])
        shape_b = VectorShape([Point2D(10.0, 20.0), Point2D(30.0, 15.0), Point2D(40.0, 50.0)])

        result = interpolate_shapes(shape_a, shape_b, 0.0)

        assert isinstance(result, VectorShape)
        assert_shape_approx_equal(result, shape_a)

    def test_interpolate_shapes_at_factor_one_returns_shape_b(self) -> None:
        """Given factor t = 1.0, produce a shape identical to shape B."""
        shape_a = VectorShape([Point2D(0.0, 0.0), Point2D(10.0, 5.0), Point2D(20.0, 30.0)])
        shape_b = VectorShape([Point2D(10.0, 20.0), Point2D(30.0, 15.0), Point2D(40.0, 50.0)])

        result = interpolate_shapes(shape_a, shape_b, 1.0)

        assert isinstance(result, VectorShape)
        assert_shape_approx_equal(result, shape_b)

    def test_interpolate_shapes_midpoint_pairwise(self) -> None:
        """Given factor t = 0.5, produce a shape whose vertices are pairwise midpoints."""
        shape_a = VectorShape([Point2D(0.0, 0.0), Point2D(10.0, -20.0), Point2D(4.0, 8.0)])
        shape_b = VectorShape([Point2D(10.0, 20.0), Point2D(30.0, 20.0), Point2D(12.0, 16.0)])
        expected = VectorShape([Point2D(5.0, 10.0), Point2D(20.0, 0.0), Point2D(8.0, 12.0)])

        result = interpolate_shapes(shape_a, shape_b, 0.5)

        assert isinstance(result, VectorShape)
        assert_shape_approx_equal(result, expected)

    def test_interpolate_shapes_arbitrary_factor(self) -> None:
        """Given factor t = 0.25, pairwise interpolate each vertex coordinate."""
        shape_a = VectorShape([Point2D(0.0, 100.0), Point2D(40.0, 0.0)])
        shape_b = VectorShape([Point2D(100.0, 0.0), Point2D(0.0, 80.0)])
        expected = VectorShape([Point2D(25.0, 75.0), Point2D(30.0, 20.0)])

        result = interpolate_shapes(shape_a, shape_b, 0.25)

        assert isinstance(result, VectorShape)
        assert_shape_approx_equal(result, expected)

    def test_interpolate_shapes_with_single_vertex(self) -> None:
        """Verify interpolation succeeds on equal-length shapes with a single vertex."""
        shape_a = VectorShape([Point2D(2.0, 4.0)])
        shape_b = VectorShape([Point2D(6.0, 8.0)])
        expected = VectorShape([Point2D(4.0, 6.0)])

        result = interpolate_shapes(shape_a, shape_b, 0.5)

        assert len(result.vertices) == 1
        assert_shape_approx_equal(result, expected)

    def test_interpolate_shapes_with_empty_vertex_sequences(self) -> None:
        """Verify interpolation succeeds on two empty shapes (0 == 0 vertex count)."""
        shape_a = VectorShape([])
        shape_b = VectorShape([])

        result = interpolate_shapes(shape_a, shape_b, 0.5)

        assert isinstance(result, VectorShape)
        assert len(result.vertices) == 0

    def test_interpolate_shapes_immutability(self) -> None:
        """Verify interpolation does not modify the vertices of input shapes."""
        pts_a = [Point2D(1.0, 2.0), Point2D(3.0, 4.0)]
        pts_b = [Point2D(5.0, 6.0), Point2D(7.0, 8.0)]
        shape_a = VectorShape(pts_a)
        shape_b = VectorShape(pts_b)

        _ = interpolate_shapes(shape_a, shape_b, 0.5)

        assert shape_a.vertices == pts_a
        assert shape_b.vertices == pts_b

    def test_interpolate_shapes_produces_new_instances(self) -> None:
        """Verify the resulting shape and vertices are newly instantiated objects."""
        shape_a = VectorShape([Point2D(1.0, 1.0)])
        shape_b = VectorShape([Point2D(1.0, 1.0)])

        result = interpolate_shapes(shape_a, shape_b, 0.0)

        assert result is not shape_a
        assert result.vertices[0] is not shape_a.vertices[0]


# ============================================================================
# AC 3: Validation and Error Handling Tests (src/vector/interpolation.py)
# ============================================================================

class TestInterpolationPipelineValidation:
    """Unit tests for input validation on interpolation factor and vertex count alignment."""

    @pytest.mark.parametrize("invalid_t", [-0.0001, -0.1, -1.0, -50.0])
    def test_raises_value_error_for_factor_less_than_zero(self, invalid_t: float) -> None:
        """Given t < 0.0, the interpolation pipeline must raise a ValueError."""
        shape_a = VectorShape([Point2D(0.0, 0.0), Point2D(1.0, 1.0)])
        shape_b = VectorShape([Point2D(2.0, 2.0), Point2D(3.0, 3.0)])

        with pytest.raises(ValueError):
            interpolate_shapes(shape_a, shape_b, invalid_t)

    @pytest.mark.parametrize("invalid_t", [1.0001, 1.1, 2.0, 100.0])
    def test_raises_value_error_for_factor_greater_than_one(self, invalid_t: float) -> None:
        """Given t > 1.0, the interpolation pipeline must raise a ValueError."""
        shape_a = VectorShape([Point2D(0.0, 0.0), Point2D(1.0, 1.0)])
        shape_b = VectorShape([Point2D(2.0, 2.0), Point2D(3.0, 3.0)])

        with pytest.raises(ValueError):
            interpolate_shapes(shape_a, shape_b, invalid_t)

    @pytest.mark.parametrize(
        ("count_a", "count_b"),
        [
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (4, 1),
            (10, 5),
        ],
    )
    def test_raises_value_error_for_mismatched_vertex_counts(
        self, count_a: int, count_b: int
    ) -> None:
        """Given shape sequences with mismatched vertex counts, raise a ValueError."""
        shape_a = VectorShape([Point2D(float(i), float(i)) for i in range(count_a)])
        shape_b = VectorShape([Point2D(float(i), float(i)) for i in range(count_b)])

        with pytest.raises(ValueError):
            interpolate_shapes(shape_a, shape_b, 0.5)

    def test_raises_value_error_when_both_factor_and_counts_invalid(self) -> None:
        """Given both factor t out of range and mismatched vertex counts, raise a ValueError."""
        shape_a = VectorShape([Point2D(0.0, 0.0)])
        shape_b = VectorShape([Point2D(1.0, 1.0), Point2D(2.0, 2.0)])

        with pytest.raises(ValueError):
            interpolate_shapes(shape_a, shape_b, -0.5)