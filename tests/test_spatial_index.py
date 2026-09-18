import math
import pytest

from src.interaction import (
    SpatialIndex as RootSpatialIndex,
    BoundingBox as RootBoundingBox,
    HitResult as RootHitResult,
)
from src.interaction.spatial_index import BoundingBox, HitResult, SpatialIndex


# ============================================================================
# Package and Export Verification
# ============================================================================


def test_package_exports():
    """Verify that interaction package exposes the primary spatial index classes."""
    assert RootSpatialIndex is SpatialIndex
    assert RootBoundingBox is BoundingBox
    assert RootHitResult is HitResult


# ============================================================================
# Acceptance Criteria: Empty Spatial Index & Outside-of-Bounds Queries
# ============================================================================


class TestEmptySpatialIndexAndMissQueries:
    """Tests covering AC: Given an empty spatial index or a query coordinate outside all bounding boxes."""

    def test_empty_index_initial_state(self):
        """Empty spatial index must have length 0 and empty bounds."""
        index = SpatialIndex()
        assert len(index) == 0
        assert index.bounds is None

    def test_empty_index_hit_test_returns_empty_list(self):
        """Hit testing an empty spatial index must return an empty list."""
        index = SpatialIndex()
        results = index.hit_test(x=10.0, y=20.0)
        assert results == []

    def test_empty_index_hit_test_with_tolerance_returns_empty_list(self):
        """Hit testing an empty spatial index even with large tolerance returns empty list."""
        index = SpatialIndex()
        results = index.hit_test(x=0.0, y=0.0, tolerance=1000.0)
        assert results == []

    def test_empty_index_query_rect_returns_empty_list(self):
        """Querying a rectangle on an empty index returns an empty list."""
        index = SpatialIndex()
        results = index.query_rect(min_x=-100.0, min_y=-100.0, max_x=100.0, max_y=100.0)
        assert results == []

    def test_query_coordinate_completely_outside_all_bounding_boxes(self):
        """Hit testing far outside all inserted bounding boxes returns an empty list."""
        index = SpatialIndex()
        index.insert_bar(item_id="bar_1", min_x=10.0, min_y=10.0, max_x=20.0, max_y=50.0)
        index.insert_point(item_id="point_1", x=5.0, y=5.0, radius=1.0)

        # Coordinate in a completely different quadrant
        results = index.hit_test(x=-50.0, y=-50.0)
        assert results == []

    def test_query_coordinate_just_outside_bar_bounding_box(self):
        """Coordinate slightly outside a bar bbox by epsilon must miss without tolerance."""
        index = SpatialIndex()
        index.insert_bar(item_id="bar_1", min_x=10.0, min_y=10.0, max_x=20.0, max_y=50.0)

        epsilon = 1e-5
        # Just to the left
        assert index.hit_test(x=10.0 - epsilon, y=25.0) == []
        # Just to the right
        assert index.hit_test(x=20.0 + epsilon, y=25.0) == []
        # Just below
        assert index.hit_test(x=15.0, y=10.0 - epsilon) == []
        # Just above
        assert index.hit_test(x=15.0, y=50.0 + epsilon) == []

    def test_query_coordinate_just_outside_point_radius(self):
        """Coordinate outside point radius must miss when queried without additional tolerance."""
        index = SpatialIndex()
        index.insert_point(item_id="pt_1", x=100.0, y=100.0, radius=5.0)

        # Distance is 5.01 > radius 5.0
        assert index.hit_test(x=105.01, y=100.0) == []
        assert index.hit_test(x=100.0, y=94.99) == []


# ============================================================================
# Acceptance Criteria: Bar Bounding Boxes Hit-Testing
# ============================================================================


class TestBarBoundingBoxHitTesting:
    """Tests covering AC: Collection of bar bounding boxes inserted into spatial index."""

    def test_single_bar_interior_hit(self):
        """Querying strictly inside a bar bounding box hits the bar."""
        index = SpatialIndex()
        payload = {"series": "sales", "value": 120}
        index.insert_bar(
            item_id="bar_q1",
            min_x=10.0,
            min_y=0.0,
            max_x=30.0,
            max_y=120.0,
            data=payload,
        )

        results = index.hit_test(x=20.0, y=60.0)
        assert len(results) == 1
        hit = results[0]
        assert isinstance(hit, HitResult)
        assert hit.item_id == "bar_q1"
        assert hit.data == payload
        assert hit.bounds == BoundingBox(10.0, 0.0, 30.0, 120.0)

    @pytest.mark.parametrize(
        ("query_x", "query_y"),
        [
            (10.0, 0.0),  # Bottom-left corner
            (30.0, 0.0),  # Bottom-right corner
            (10.0, 100.0),  # Top-left corner
            (30.0, 100.0),  # Top-right corner
            (20.0, 0.0),  # Bottom edge midpoint
            (20.0, 100.0),  # Top edge midpoint
            (10.0, 50.0),  # Left edge midpoint
            (30.0, 50.0),  # Right edge midpoint
        ],
    )
    def test_bar_boundary_inclusive_hit(self, query_x, query_y):
        """Hit test must be inclusive of bar edges and corners."""
        index = SpatialIndex()
        index.insert_bar("bar_boundary", min_x=10.0, min_y=0.0, max_x=30.0, max_y=100.0)

        results = index.hit_test(x=query_x, y=query_y)
        assert len(results) == 1
        assert results[0].item_id == "bar_boundary"

    def test_multiple_non_overlapping_bars_hit_discrimination(self):
        """In an index with multiple non-overlapping bars, query matches only the targeted bar."""
        index = SpatialIndex()
        # Bar 1: [0, 10] x [0, 50]
        # Bar 2: [20, 30] x [0, 80]
        # Bar 3: [40, 50] x [0, 40]
        index.insert_bar("bar_1", 0.0, 0.0, 10.0, 50.0)
        index.insert_bar("bar_2", 20.0, 0.0, 30.0, 80.0)
        index.insert_bar("bar_3", 40.0, 0.0, 50.0, 40.0)

        assert len(index) == 3

        hit_b1 = index.hit_test(x=5.0, y=25.0)
        assert [h.item_id for h in hit_b1] == ["bar_1"]

        hit_b2 = index.hit_test(x=25.0, y=70.0)
        assert [h.item_id for h in hit_b2] == ["bar_2"]

        hit_b3 = index.hit_test(x=45.0, y=10.0)
        assert [h.item_id for h in hit_b3] == ["bar_3"]

        # In-between gap
        assert index.hit_test(x=15.0, y=25.0) == []

    def test_overlapping_bars_return_all_matching_items(self):
        """When query falls in intersection of multiple bars, all containing bars are returned."""
        index = SpatialIndex()
        # Bar A: [0, 50] x [0, 50]
        # Bar B: [25, 75] x [25, 75]
        index.insert_bar("bar_A", 0.0, 0.0, 50.0, 50.0)
        index.insert_bar("bar_B", 25.0, 25.0, 75.0, 75.0)

        # Query in intersection [25, 50] x [25, 50]
        results = index.hit_test(x=30.0, y=30.0)
        hit_ids = {h.item_id for h in results}
        assert hit_ids == {"bar_A", "bar_B"}

        # Query in Bar A only
        assert {h.item_id for h in index.hit_test(x=10.0, y=10.0)} == {"bar_A"}

        # Query in Bar B only
        assert {h.item_id for h in index.hit_test(x=60.0, y=60.0)} == {"bar_B"}

    def test_bar_hit_test_with_tolerance(self):
        """Tolerance extends the effective hit-test area of bars symmetrically."""
        index = SpatialIndex()
        index.insert_bar("bar_1", min_x=10.0, min_y=10.0, max_x=20.0, max_y=20.0)

        # Coordinate is 3.0 units away to the left (x=7.0)
        assert index.hit_test(x=7.0, y=15.0, tolerance=2.0) == []
        assert len(index.hit_test(x=7.0, y=15.0, tolerance=3.0)) == 1
        assert len(index.hit_test(x=7.0, y=15.0, tolerance=5.0)) == 1


# ============================================================================
# Acceptance Criteria: Scatter Points Hit-Testing
# ============================================================================


class TestScatterPointHitTesting:
    """Tests covering AC: Collection of scatter points inserted into spatial index."""

    def test_zero_radius_point_exact_hit(self):
        """Zero-radius point can be hit exactly at its coordinate."""
        index = SpatialIndex()
        index.insert_point("pt_exact", x=12.5, y=-45.0, radius=0.0)

        results = index.hit_test(x=12.5, y=-45.0)
        assert len(results) == 1
        assert results[0].item_id == "pt_exact"

    def test_zero_radius_point_with_hit_tolerance(self):
        """Zero-radius point can be hit within a specified query tolerance distance."""
        index = SpatialIndex()
        index.insert_point("pt_1", x=0.0, y=0.0, radius=0.0)

        # Euclidean distance = 5.0
        query_x, query_y = 3.0, 4.0

        # Tolerance 4.99 misses
        assert index.hit_test(x=query_x, y=query_y, tolerance=4.99) == []
        # Tolerance 5.00 hits
        hit = index.hit_test(x=query_x, y=query_y, tolerance=5.0)
        assert len(hit) == 1
        assert hit[0].item_id == "pt_1"

    def test_point_with_intrinsic_radius_hit(self):
        """Point inserted with intrinsic radius expands its hit bounding area."""
        index = SpatialIndex()
        index.insert_point("scatter_circle", x=50.0, y=50.0, radius=10.0)

        # Expected bounding box: [40.0, 40.0, 60.0, 60.0]
        assert index.bounds == BoundingBox(40.0, 40.0, 60.0, 60.0)

        # Hit inside the circle
        results = index.hit_test(x=56.0, y=58.0)  # dist = sqrt(6^2 + 8^2) = 10.0
        assert len(results) == 1
        assert results[0].item_id == "scatter_circle"

    def test_point_intrinsic_radius_plus_query_tolerance(self):
        """Query tolerance stacks with point intrinsic radius."""
        index = SpatialIndex()
        index.insert_point("pt_stacked", x=0.0, y=0.0, radius=5.0)

        # Distance from center = 8.0. Radius = 5.0. Needs tolerance >= 3.0
        assert index.hit_test(x=8.0, y=0.0, tolerance=2.9) == []
        results = index.hit_test(x=8.0, y=0.0, tolerance=3.0)
        assert len(results) == 1
        assert results[0].item_id == "pt_stacked"

    def test_dense_scatter_points_spatial_filtering(self):
        """Multiple nearby scatter points are accurately and independently resolved."""
        index = SpatialIndex()
        # Arrange 5 points along a line: x = 0, 10, 20, 30, 40
        for i in range(5):
            index.insert_point(f"pt_{i}", x=i * 10.0, y=0.0, radius=2.0)

        assert len(index) == 5

        # Hit test at x=10 hits only pt_1
        hit = index.hit_test(x=10.0, y=0.0)
        assert len(hit) == 1
        assert hit[0].item_id == "pt_1"

        # Hit test between pt_1 and pt_2 with large tolerance hits both
        multi_hit = index.hit_test(x=15.0, y=0.0, tolerance=6.0)
        multi_ids = {h.item_id for h in multi_hit}
        assert multi_ids == {"pt_1", "pt_2"}


# ============================================================================
# Mixed Collections: Scatter Points and Bar Bounding Boxes
# ============================================================================


class TestMixedScatterAndBarHitTesting:
    """Tests covering AC: Combined collection of scatter points and bar bounding boxes."""

    def test_mixed_collection_discrete_hits(self):
        """Index handles simultaneous presence of bars and scatter points."""
        index = SpatialIndex()
        index.insert_bar("bar_background", min_x=0.0, min_y=0.0, max_x=100.0, max_y=50.0)
        index.insert_point("scatter_foreground", x=50.0, y=25.0, radius=2.0)
        index.insert_point("scatter_isolated", x=200.0, y=200.0, radius=5.0)

        assert len(index) == 3

        # Query isolated point
        hit_iso = index.hit_test(x=200.0, y=200.0)
        assert [h.item_id for h in hit_iso] == ["scatter_isolated"]

        # Query inside bar but away from foreground scatter point
        hit_bar_only = index.hit_test(x=10.0, y=10.0)
        assert [h.item_id for h in hit_bar_only] == ["bar_background"]

        # Query at center where point and bar coincide
        hit_both = index.hit_test(x=50.0, y=25.0)
        hit_both_ids = {h.item_id for h in hit_both}
        assert hit_both_ids == {"bar_background", "scatter_foreground"}

    def test_generic_insert_with_bounding_box_instance(self):
        """Index supports inserting directly using BoundingBox dataclass."""
        index = SpatialIndex()
        bbox = BoundingBox(min_x=1.0, min_y=2.0, max_x=3.0, max_y=4.0)
        index.insert("custom_box", bbox, data={"kind": "custom"})

        assert len(index) == 1
        results = index.hit_test(x=2.0, y=3.0)
        assert len(results) == 1
        assert results[0].item_id == "custom_box"
        assert results[0].data == {"kind": "custom"}
        assert results[0].bounds == bbox

    def test_negative_quadrant_mixed_elements(self):
        """Spatial index correctly handles negative coordinate spaces."""
        index = SpatialIndex()
        index.insert_bar("neg_bar", min_x=-100.0, min_y=-100.0, max_x=-50.0, max_y=-50.0)
        index.insert_point("neg_pt", x=-75.0, y=-75.0, radius=5.0)

        # Hits both
        results = index.hit_test(x=-75.0, y=-75.0)
        assert {r.item_id for r in results} == {"neg_bar", "neg_pt"}

        # Miss
        assert index.hit_test(x=-49.0, y=-75.0) == []


# ============================================================================
# Spatial Range / Rectangle Queries
# ============================================================================


class TestSpatialIndexRangeQuery:
    """Tests covering range/box intersection queries on the spatial index."""

    def test_query_rect_encompassing_all_items(self):
        """Range query enclosing all items returns all items."""
        index = SpatialIndex()
        index.insert_bar("bar_1", 10.0, 10.0, 20.0, 20.0)
        index.insert_point("pt_1", 50.0, 50.0, radius=2.0)

        results = index.query_rect(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0)
        assert {r.item_id for r in results} == {"bar_1", "pt_1"}

    def test_query_rect_partial_overlap(self):
        """Range query returns only items that intersect query box."""
        index = SpatialIndex()
        index.insert_bar("bar_inside", 10.0, 10.0, 20.0, 20.0)
        index.insert_bar("bar_outside", 100.0, 100.0, 120.0, 120.0)
        index.insert_bar("bar_intersecting", 25.0, 25.0, 35.0, 35.0)

        # Query box [0, 30] x [0, 30] contains bar_inside and intersects bar_intersecting
        results = index.query_rect(min_x=0.0, min_y=0.0, max_x=30.0, max_y=30.0)
        assert {r.item_id for r in results} == {"bar_inside", "bar_intersecting"}

    def test_query_rect_disjoint_returns_empty(self):
        """Range query completely outside all items returns empty list."""
        index = SpatialIndex()
        index.insert_bar("bar_1", 10.0, 10.0, 20.0, 20.0)

        results = index.query_rect(min_x=30.0, min_y=30.0, max_x=40.0, max_y=40.0)
        assert results == []


# ============================================================================
# Mutation and Lifecycle: Deletion, Clearing, and Re-querying
# ============================================================================


class TestSpatialIndexMutation:
    """Tests covering dynamic index updates: item removal, clear, and re-querying."""

    def test_remove_existing_item(self):
        """Removing an inserted item decreases length and excludes it from hit testing."""
        index = SpatialIndex()
        index.insert_bar("bar_1", 0.0, 0.0, 10.0, 10.0)
        index.insert_bar("bar_2", 20.0, 20.0, 30.0, 30.0)

        assert len(index) == 2
        removed = index.remove("bar_1")
        assert removed is True
        assert len(index) == 1

        # Querying removed location returns nothing
        assert index.hit_test(x=5.0, y=5.0) == []
        # Remaining item still intact
        assert len(index.hit_test(x=25.0, y=25.0)) == 1

    def test_remove_nonexistent_item(self):
        """Removing an item ID not in index returns False and leaves index unchanged."""
        index = SpatialIndex()
        index.insert_point("pt_1", 1.0, 1.0)
        removed = index.remove("ghost_id")
        assert removed is False
        assert len(index) == 1

    def test_clear_resets_index_completely(self):
        """Calling clear resets size to zero and makes subsequent queries return empty."""
        index = SpatialIndex()
        for i in range(10):
            index.insert_point(f"pt_{i}", float(i), float(i))

        assert len(index) == 10
        index.clear()

        assert len(index) == 0
        assert index.bounds is None
        assert index.hit_test(x=0.0, y=0.0, tolerance=100.0) == []
        assert index.query_rect(-50.0, -50.0, 50.0, 50.0) == []

    def test_reinsertion_after_clear(self):
        """Index is fully reusable after clear."""
        index = SpatialIndex()
        index.insert_bar("bar_old", 0.0, 0.0, 10.0, 10.0)
        index.clear()

        index.insert_bar("bar_new", 50.0, 50.0, 60.0, 60.0)
        assert len(index) == 1
        assert index.hit_test(x=5.0, y=5.0) == []
        assert len(index.hit_test(x=55.0, y=55.0)) == 1


# ============================================================================
# Input Validation and Robustness
# ============================================================================


class TestInputValidationAndEdgeCases:
    """Tests verifying input validation and edge cases."""

    def test_invalid_bounding_box_min_greater_than_max(self):
        """Inserting a bounding box where min > max must raise ValueError."""
        index = SpatialIndex()
        with pytest.raises(ValueError):
            index.insert_bar("bad_x", min_x=50.0, min_y=0.0, max_x=10.0, max_y=10.0)

        with pytest.raises(ValueError):
            index.insert_bar("bad_y", min_x=0.0, min_y=50.0, max_x=10.0, max_y=10.0)

        with pytest.raises(ValueError):
            BoundingBox(min_x=20.0, min_y=0.0, max_x=10.0, max_y=10.0)

    def test_negative_radius_raises_value_error(self):
        """Negative radius for scatter point must raise ValueError."""
        index = SpatialIndex()
        with pytest.raises(ValueError):
            index.insert_point("neg_rad", x=0.0, y=0.0, radius=-1.0)

    def test_negative_tolerance_raises_value_error(self):
        """Negative hit test tolerance must raise ValueError."""
        index = SpatialIndex()
        index.insert_point("pt_1", x=0.0, y=0.0)
        with pytest.raises(ValueError):
            index.hit_test(x=0.0, y=0.0, tolerance=-0.5)

    def test_nan_or_inf_coordinates_raise_value_error(self):
        """NaN or Inf coordinates during insertion or query must raise ValueError."""
        index = SpatialIndex()
        with pytest.raises(ValueError):
            index.insert_point("nan_pt", x=float("nan"), y=0.0)

        with pytest.raises(ValueError):
            index.insert_bar("inf_bar", 0.0, 0.0, float("inf"), 10.0)

        index.insert_point("valid_pt", x=1.0, y=1.0)
        with pytest.raises(ValueError):
            index.hit_test(x=float("nan"), y=1.0)

        with pytest.raises(ValueError):
            index.query_rect(0.0, 0.0, float("inf"), 10.0)

    def test_zero_area_bar_acts_as_line_or_point(self):
        """Zero-width or zero-height bar (e.g. min_x == max_x) is valid and hittable with tolerance."""
        index = SpatialIndex()
        # Vertical guideline / line: x=5, y in [0, 10]
        index.insert_bar("line_segment", min_x=5.0, min_y=0.0, max_x=5.0, max_y=10.0)

        # Exact hit
        assert len(index.hit_test(x=5.0, y=5.0)) == 1
        # Off by 0.5 without tolerance misses
        assert index.hit_test(x=5.5, y=5.0, tolerance=0.0) == []
        # Off by 0.5 with tolerance 1.0 hits
        assert len(index.hit_test(x=5.5, y=5.0, tolerance=1.0)) == 1

    def test_duplicate_item_id_replaces_or_raises_cleanly(self):
        """Re-inserting the same item ID updates its position."""
        index = SpatialIndex()
        index.insert_point("unique_node", x=10.0, y=10.0)
        index.insert_point("unique_node", x=50.0, y=50.0)

        assert len(index) == 1
        assert index.hit_test(x=10.0, y=10.0) == []
        assert len(index.hit_test(x=50.0, y=50.0)) == 1


# ============================================================================
# Scale and Performance Characteristics
# ============================================================================


class TestSpatialIndexScale:
    """Verify hit-testing correctness across larger datasets (R-tree indexing)."""

    def test_grid_of_bars_and_points_correctness(self):
        """100 bars and 100 points arranged in a grid can all be hit correctly."""
        index = SpatialIndex()
        grid_size = 10
        spacing = 50.0

        for r in range(grid_size):
            for c in range(grid_size):
                base_x = c * spacing
                base_y = r * spacing
                # Bar occupying [base_x, base_x + 20] x [base_y, base_y + 20]
                index.insert_bar(
                    item_id=f"bar_{r}_{c}",
                    min_x=base_x,
                    min_y=base_y,
                    max_x=base_x + 20.0,
                    max_y=base_y + 20.0,
                )
                # Scatter point at center of the remaining quadrant
                index.insert_point(
                    item_id=f"pt_{r}_{c}",
                    x=base_x + 35.0,
                    y=base_y + 35.0,
                    radius=3.0,
                )

        assert len(index) == 200

        # Query all bars at their center
        for r in range(grid_size):
            for c in range(grid_size):
                base_x = c * spacing
                base_y = r * spacing
                hits = index.hit_test(x=base_x + 10.0, y=base_y + 10.0)
                assert len(hits) == 1
                assert hits[0].item_id == f"bar_{r}_{c}"

                pt_hits = index.hit_test(x=base_x + 35.0, y=base_y + 35.0)
                assert len(pt_hits) == 1
                assert pt_hits[0].item_id == f"pt_{r}_{c}"

        # Hit test in empty space between items returns empty
        assert index.hit_test(x=22.0, y=22.0) == []