"""
Unit tests for Magnetic Snap-to-OHLC Algorithm (Story 5.1.3).

Covers:
- Snap to closest OHLC component within configured radius (threshold).
- Return original cursor coordinates unchanged when outside radius.
- Deterministic tie-breaking favoring extreme levels (High/Low) over intermediate levels (Open/Close).
- Custom coordinate projections (e.g. canvas price-to-Y mapping).
- Edge cases, parameter validation, immutability, and module exports.
"""

import math
from typing import Callable

import pytest

from src.charting import (
    Candlestick,
    MagneticSnap,
    Point,
    SnapResult,
    snap_to_ohlc,
)
from src.charting.snap import (
    Candlestick as SnapCandlestick,
    MagneticSnap as SnapMagneticSnap,
    Point as SnapPoint,
    SnapResult as SnapResultDirect,
    snap_to_ohlc as snap_to_ohlc_direct,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def standard_candle() -> Candlestick:
    """
    Standard test candlestick:
    x: 100.0, Open: 105.0, High: 120.0, Low: 85.0, Close: 95.0.
    """
    return Candlestick(open=105.0, high=120.0, low=85.0, close=95.0, x=100.0)


@pytest.fixture
def linear_screen_projection() -> Callable[[float], float]:
    """
    Screen pixel projection where y decreases as price increases (canvas coords).
    price 100 -> y = 500, price 120 -> y = 460, price 80 -> y = 540.
    formula: y = 500.0 - (price - 100.0) * 2.0
    """
    return lambda price: 500.0 - (price - 100.0) * 2.0


# ============================================================================
# Acceptance Criteria 1: Snap within configured radius
# ============================================================================


class TestSnapWithinRadius:
    """
    Tests verifying that cursor within snap radius snaps to closest OHLC component
    and returns its price and coordinate.
    """

    @pytest.mark.parametrize(
        ("cursor", "expected_component", "expected_price", "expected_coord"),
        [
            (
                Point(100.0, 106.0),
                "O",
                105.0,
                Point(100.0, 105.0),
            ),
            (
                Point(100.0, 118.5),
                "H",
                120.0,
                Point(100.0, 120.0),
            ),
            (
                Point(100.0, 86.2),
                "L",
                85.0,
                Point(100.0, 85.0),
            ),
            (
                Point(100.0, 94.1),
                "C",
                95.0,
                Point(100.0, 95.0),
            ),
        ],
    )
    def test_snap_to_each_ohlc_component_within_radius(
        self,
        standard_candle: Candlestick,
        cursor: Point,
        expected_component: str,
        expected_price: float,
        expected_coord: Point,
    ) -> None:
        """Cursor positioned closest to each component snaps correctly."""
        result: SnapResult = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=cursor,
            snap_radius=5.0,
        )

        assert result.snapped is True
        assert result.component == expected_component
        assert result.price == pytest.approx(expected_price)
        assert result.coordinate == expected_coord

    def test_snap_with_horizontal_offset_within_euclidean_radius(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """
        Cursor offset on X axis (e.g. candle at x=100, cursor at x=103, y=120).
        Distance to High is 3.0, within radius of 5.0.
        """
        cursor = Point(103.0, 120.0)
        result = snap_to_ohlc(standard_candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(100.0, 120.0)

    def test_snap_at_exact_component_coordinate(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Distance of 0.0 must snap cleanly to that component."""
        cursor = Point(100.0, 85.0)  # Exactly at Low
        result = snap_to_ohlc(standard_candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "L"
        assert result.price == pytest.approx(85.0)
        assert result.coordinate == Point(100.0, 85.0)

    def test_snap_on_exact_radius_boundary(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Distance exactly equal to snap_radius must trigger snap."""
        radius = 4.0
        cursor = Point(100.0, 120.0 + radius)  # exactly 4.0 units above High
        result = snap_to_ohlc(standard_candle, cursor, snap_radius=radius)

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(100.0, 120.0)

    def test_snap_with_screen_coordinate_projection(
        self,
        standard_candle: Candlestick,
        linear_screen_projection: Callable[[float], float],
    ) -> None:
        """
        Using custom price_to_y projection:
        High (120) -> screen Y = 460.0
        Cursor at (100.0, 462.0) -> distance to High screen coordinate is 2.0px.
        """
        cursor = Point(100.0, 462.0)
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=cursor,
            snap_radius=5.0,
            price_to_y=linear_screen_projection,
        )

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(100.0, 460.0)


# ============================================================================
# Acceptance Criteria 2: Cursor outside snap radius
# ============================================================================


class TestSnapOutsideRadius:
    """
    Tests verifying that cursor outside snap radius returns original cursor
    coordinates without snapping.
    """

    def test_cursor_far_above_highest_level(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Cursor placed far above High level returns original coordinates."""
        original_cursor = Point(100.0, 300.0)
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=original_cursor,
            snap_radius=10.0,
        )

        assert result.snapped is False
        assert result.component is None
        assert result.price is None
        assert result.coordinate == original_cursor

    def test_cursor_far_below_lowest_level(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Cursor placed far below Low level returns original coordinates."""
        original_cursor = Point(100.0, 20.0)
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=original_cursor,
            snap_radius=10.0,
        )

        assert result.snapped is False
        assert result.component is None
        assert result.price is None
        assert result.coordinate == original_cursor

    def test_cursor_far_horizontally(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Cursor at matching Y level but X distance exceeds snap radius."""
        original_cursor = Point(200.0, 120.0)  # 100px away horizontally
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=original_cursor,
            snap_radius=10.0,
        )

        assert result.snapped is False
        assert result.component is None
        assert result.price is None
        assert result.coordinate == original_cursor

    def test_cursor_just_beyond_radius_boundary(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Cursor at snap_radius + epsilon must not snap."""
        radius = 5.0
        epsilon = 0.001
        original_cursor = Point(100.0, 120.0 + radius + epsilon)
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=original_cursor,
            snap_radius=radius,
        )

        assert result.snapped is False
        assert result.component is None
        assert result.price is None
        assert result.coordinate == original_cursor

    def test_cursor_outside_with_screen_projection(
        self,
        standard_candle: Candlestick,
        linear_screen_projection: Callable[[float], float],
    ) -> None:
        """Screen coordinate cursor beyond threshold returns original coordinates."""
        original_cursor = Point(100.0, 800.0)
        result = snap_to_ohlc(
            candlestick=standard_candle,
            cursor=original_cursor,
            snap_radius=10.0,
            price_to_y=linear_screen_projection,
        )

        assert result.snapped is False
        assert result.component is None
        assert result.price is None
        assert result.coordinate == original_cursor


# ============================================================================
# Acceptance Criteria 3: Deterministic tie-breaking (Extreme over Intermediate)
# ============================================================================


class TestEquidistantTieBreaking:
    """
    Tests verifying deterministic tie resolution favoring extremes (High/Low)
    over intermediate levels (Open/Close).
    """

    def test_tie_high_vs_open_at_same_price(self) -> None:
        """
        Candle opens at high: Open == High == 120.0.
        Cursor directly at 120.0 -> distance to both is 0.0.
        Must deterministically select extreme 'H' over intermediate 'O'.
        """
        candle = Candlestick(open=120.0, high=120.0, low=90.0, close=100.0, x=50.0)
        cursor = Point(50.0, 120.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(50.0, 120.0)

    def test_tie_high_vs_close_at_same_price(self) -> None:
        """
        Candle closes at high: Close == High == 120.0.
        Cursor directly at 120.0.
        Must deterministically select extreme 'H' over intermediate 'C'.
        """
        candle = Candlestick(open=100.0, high=120.0, low=90.0, close=120.0, x=50.0)
        cursor = Point(50.0, 120.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(50.0, 120.0)

    def test_tie_low_vs_open_at_same_price(self) -> None:
        """
        Candle opens at low: Open == Low == 90.0.
        Cursor directly at 90.0.
        Must deterministically select extreme 'L' over intermediate 'O'.
        """
        candle = Candlestick(open=90.0, high=120.0, low=90.0, close=110.0, x=50.0)
        cursor = Point(50.0, 90.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "L"
        assert result.price == pytest.approx(90.0)
        assert result.coordinate == Point(50.0, 90.0)

    def test_tie_low_vs_close_at_same_price(self) -> None:
        """
        Candle closes at low: Close == Low == 90.0.
        Cursor directly at 90.0.
        Must deterministically select extreme 'L' over intermediate 'C'.
        """
        candle = Candlestick(open=110.0, high=120.0, low=90.0, close=90.0, x=50.0)
        cursor = Point(50.0, 90.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component == "L"
        assert result.price == pytest.approx(90.0)
        assert result.coordinate == Point(50.0, 90.0)

    def test_equidistant_cursor_between_high_and_open(self) -> None:
        """
        Open = 100.0, High = 120.0, Low = 70.0, Close = 80.0.
        Cursor at y = 110.0 (exact midpoint between High and Open).
        Distance to High = 10.0, Distance to Open = 10.0.
        Radius = 15.0.
        Must select extreme 'H' over intermediate 'O'.
        """
        candle = Candlestick(open=100.0, high=120.0, low=70.0, close=80.0, x=50.0)
        cursor = Point(50.0, 110.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=15.0)

        assert result.snapped is True
        assert result.component == "H"
        assert result.price == pytest.approx(120.0)
        assert result.coordinate == Point(50.0, 120.0)

    def test_equidistant_cursor_between_low_and_close(self) -> None:
        """
        Open = 110.0, High = 130.0, Close = 100.0, Low = 80.0.
        Cursor at y = 90.0 (exact midpoint between Close and Low).
        Distance to Close = 10.0, Distance to Low = 10.0.
        Radius = 15.0.
        Must select extreme 'L' over intermediate 'C'.
        """
        candle = Candlestick(open=110.0, high=130.0, low=80.0, close=100.0, x=50.0)
        cursor = Point(50.0, 90.0)

        result = snap_to_ohlc(candle, cursor, snap_radius=15.0)

        assert result.snapped is True
        assert result.component == "L"
        assert result.price == pytest.approx(80.0)
        assert result.coordinate == Point(50.0, 80.0)

    def test_flat_candlestick_all_ohlc_identical(self) -> None:
        """
        Flat candle where Open == High == Low == Close = 100.0.
        All 4 components coincide.
        Algorithm must deterministically pick an extreme level ('H' or 'L'), never 'O' or 'C'.
        """
        flat_candle = Candlestick(open=100.0, high=100.0, low=100.0, close=100.0, x=25.0)
        cursor = Point(25.0, 100.0)

        result = snap_to_ohlc(flat_candle, cursor, snap_radius=5.0)

        assert result.snapped is True
        assert result.component in {"H", "L"}
        assert result.component not in {"O", "C"}
        assert result.price == pytest.approx(100.0)
        assert result.coordinate == Point(25.0, 100.0)

    def test_tie_breaking_determinism_across_multiple_invocations(self) -> None:
        """
        Algorithm must be strictly deterministic across repeated executions,
        never showing non-deterministic behavior (e.g. from set/dict iteration order).
        """
        candle = Candlestick(open=100.0, high=120.0, low=80.0, close=100.0, x=0.0)
        cursor = Point(0.0, 110.0)  # Midpoint between High and Open/Close

        first_run = snap_to_ohlc(candle, cursor, snap_radius=15.0)
        assert first_run.component == "H"

        for _ in range(50):
            repeated_run = snap_to_ohlc(candle, cursor, snap_radius=15.0)
            assert repeated_run.component == first_run.component
            assert repeated_run.price == first_run.price
            assert repeated_run.coordinate == first_run.coordinate
            assert repeated_run.snapped == first_run.snapped


# ============================================================================
# MagneticSnap Class / OO Interface Tests
# ============================================================================


class TestMagneticSnapClassAPI:
    """Tests evaluating the MagneticSnap class interface."""

    def test_magnetic_snap_instantiation_and_evaluation(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """MagneticSnap instance can be configured with radius and evaluated."""
        algorithm = MagneticSnap(snap_radius=5.0)
        cursor = Point(100.0, 105.5)  # closest to Open (105.0)

        result = algorithm.snap(standard_candle, cursor)

        assert isinstance(result, SnapResult)
        assert result.snapped is True
        assert result.component == "O"
        assert result.price == pytest.approx(105.0)

    def test_magnetic_snap_with_projection(
        self,
        standard_candle: Candlestick,
        linear_screen_projection: Callable[[float], float],
    ) -> None:
        """MagneticSnap configured with projection function evaluates properly."""
        algorithm = MagneticSnap(
            snap_radius=5.0,
            price_to_y=linear_screen_projection,
        )
        # Low price = 85.0 -> Screen Y = 500 - (85 - 100) * 2 = 530.0
        cursor = Point(100.0, 529.0)

        result = algorithm.snap(standard_candle, cursor)

        assert result.snapped is True
        assert result.component == "L"
        assert result.price == pytest.approx(85.0)
        assert result.coordinate == Point(100.0, 530.0)


# ============================================================================
# Input Validation & Domain Integrity Tests
# ============================================================================


class TestInputValidationAndIntegrity:
    """Validates domain constraints, invalid parameters, and immutability."""

    def test_invalid_candlestick_high_lower_than_open_raises_value_error(self) -> None:
        """High must be >= Open."""
        with pytest.raises(ValueError):
            Candlestick(open=100.0, high=95.0, low=90.0, close=92.0)

    def test_invalid_candlestick_high_lower_than_close_raises_value_error(self) -> None:
        """High must be >= Close."""
        with pytest.raises(ValueError):
            Candlestick(open=90.0, high=95.0, low=85.0, close=100.0)

    def test_invalid_candlestick_low_higher_than_open_raises_value_error(self) -> None:
        """Low must be <= Open."""
        with pytest.raises(ValueError):
            Candlestick(open=100.0, high=110.0, low=105.0, close=108.0)

    def test_invalid_candlestick_low_higher_than_close_raises_value_error(self) -> None:
        """Low must be <= Close."""
        with pytest.raises(ValueError):
            Candlestick(open=100.0, high=110.0, low=95.0, close=90.0)

    def test_invalid_candlestick_high_lower_than_low_raises_value_error(self) -> None:
        """High must be >= Low."""
        with pytest.raises(ValueError):
            Candlestick(open=100.0, high=80.0, low=90.0, close=85.0)

    @pytest.mark.parametrize("invalid_radius", [0.0, -1.0, -0.001])
    def test_non_positive_snap_radius_raises_value_error(
        self,
        standard_candle: Candlestick,
        invalid_radius: float,
    ) -> None:
        """Snap radius must be strictly positive (> 0)."""
        cursor = Point(100.0, 100.0)
        with pytest.raises(ValueError):
            snap_to_ohlc(standard_candle, cursor, snap_radius=invalid_radius)

        with pytest.raises(ValueError):
            MagneticSnap(snap_radius=invalid_radius)

    def test_nan_or_infinite_coordinates_raise_value_error(
        self,
        standard_candle: Candlestick,
    ) -> None:
        """Non-finite cursor coordinates must raise ValueError."""
        with pytest.raises(ValueError):
            snap_to_ohlc(standard_candle, Point(float("nan"), 100.0), snap_radius=5.0)

        with pytest.raises(ValueError):
            snap_to_ohlc(standard_candle, Point(100.0, float("inf")), snap_radius=5.0)

    def test_point_immutability(self) -> None:
        """Point instances must be immutable."""
        pt = Point(10.0, 20.0)
        with pytest.raises((AttributeError, TypeError)):
            pt.x = 30.0  # type: ignore[misc]

    def test_snap_result_immutability(self) -> None:
        """SnapResult instances must be immutable."""
        result = SnapResult(
            snapped=True,
            price=100.0,
            coordinate=Point(10.0, 100.0),
            component="O",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.snapped = False  # type: ignore[misc]


# ============================================================================
# Module Exports & Package Interface
# ============================================================================


class TestModuleExports:
    """Verifies module integrity and identical exports between package and module."""

    def test_package_exports_identical_to_snap_module(self) -> None:
        """`src.charting` public exports must map identically to `src.charting.snap`."""
        assert Candlestick is SnapCandlestick
        assert Point is SnapPoint
        assert SnapResult is SnapResultDirect
        assert MagneticSnap is SnapMagneticSnap
        assert snap_to_ohlc is snap_to_ohlc_direct

    def test_point_equality_and_coordinates(self) -> None:
        """Point equality check."""
        p1 = Point(10.0, 20.0)
        p2 = Point(10.0, 20.0)
        p3 = Point(10.0, 25.0)

        assert p1 == p2
        assert p1 != p3
        assert p1.x == 10.0
        assert p1.y == 20.0