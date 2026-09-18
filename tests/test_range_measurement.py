"""
Unit tests for Date and Price Range Measurement Primitive.

Covers:
- Story 5.4.2: Build Date and Price Range Measurement Primitive
- Acceptance Criteria verification:
  1. Start timestamp 2023-01-01T00:00:00 at 100.0 and end timestamp 2023-01-10T00:00:00
     at 150.0 produces price delta 50.0, percentage change 50.0%, and duration 9 days.
  2. Start price 200.0 and end price 150.0 over 4 days produces price delta -50.0
     and percentage change -25.0%.
  3. Start price 0.0 raises ValueError on percentage change calculation due to division by zero.
- Edge cases: fractional prices, intra-day durations, reverse chronological ranges, zero delta.
- Target modules: src/charting/primitives/range_measurement.py, src/charting/primitives/__init__.py
"""

from datetime import datetime, timedelta
import pytest

from src.charting.primitives import (
    CoordinatePoint,
    DatePriceRangeMeasurement,
    RangeMeasurement,
)
import src.charting.primitives as primitives_pkg
import src.charting.primitives.range_measurement as range_measurement_module


# =====================================================================
# Acceptance Criteria Tests
# =====================================================================


def test_ac1_positive_price_delta_and_duration():
    """
    AC 1:
    Given two coordinate points with start timestamp 2023-01-01T00:00:00 at price 100.0
    and end timestamp 2023-01-10T00:00:00 at price 150.0,
    When the date and price range measurement primitive is computed,
    Then the price delta is 50.0, the percentage change is 50.0%, and the duration is 9 days.
    """
    start_point = CoordinatePoint(
        timestamp=datetime.fromisoformat("2023-01-01T00:00:00"),
        price=100.0,
    )
    end_point = CoordinatePoint(
        timestamp=datetime.fromisoformat("2023-01-10T00:00:00"),
        price=150.0,
    )

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == 50.0
    assert measurement.percentage_change == 50.0
    assert measurement.duration == timedelta(days=9)


def test_ac2_negative_price_delta_and_percentage_change():
    """
    AC 2:
    Given two coordinate points with start price 200.0 and end price 150.0
    over a duration of 4 days,
    When the measurement primitive is computed,
    Then the price delta is -50.0 and the percentage change is -25.0%.
    """
    start_time = datetime(2023, 5, 1, 0, 0, 0)
    end_time = start_time + timedelta(days=4)

    start_point = CoordinatePoint(timestamp=start_time, price=200.0)
    end_point = CoordinatePoint(timestamp=end_time, price=150.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == -50.0
    assert measurement.percentage_change == -25.0
    assert measurement.duration == timedelta(days=4)


def test_ac3_zero_start_price_raises_value_error_on_percentage_change():
    """
    AC 3:
    Given a starting point with price 0.0,
    When the measurement primitive calculates percentage change,
    Then a ValueError is raised indicating division by zero is invalid.
    """
    start_point = CoordinatePoint(
        timestamp=datetime(2023, 1, 1, 0, 0, 0),
        price=0.0,
    )
    end_point = CoordinatePoint(
        timestamp=datetime(2023, 1, 5, 0, 0, 0),
        price=100.0,
    )

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    # Price delta and duration remain mathematically valid
    assert measurement.price_delta == 100.0
    assert measurement.duration == timedelta(days=4)

    # Percentage change calculation must raise ValueError
    with pytest.raises(ValueError):
        _ = measurement.percentage_change


# =====================================================================
# Package and Module Export Tests
# =====================================================================


def test_package_exports():
    """Verify classes are exported from src.charting.primitives and range_measurement module."""
    assert hasattr(primitives_pkg, "CoordinatePoint")
    assert hasattr(primitives_pkg, "DatePriceRangeMeasurement")
    assert hasattr(primitives_pkg, "RangeMeasurement")
    assert primitives_pkg.RangeMeasurement is primitives_pkg.DatePriceRangeMeasurement

    assert hasattr(range_measurement_module, "CoordinatePoint")
    assert hasattr(range_measurement_module, "DatePriceRangeMeasurement")
    assert hasattr(range_measurement_module, "RangeMeasurement")

    assert "CoordinatePoint" in primitives_pkg.__all__
    assert "DatePriceRangeMeasurement" in primitives_pkg.__all__
    assert "RangeMeasurement" in primitives_pkg.__all__


# =====================================================================
# Boundary and Edge Case Tests
# =====================================================================


def test_zero_price_change():
    """Verify behavior when start price equals end price (flat market)."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 3, 1, 12, 0, 0), price=125.50)
    end_point = CoordinatePoint(timestamp=datetime(2023, 3, 3, 12, 0, 0), price=125.50)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == 0.0
    assert measurement.percentage_change == 0.0
    assert measurement.duration == timedelta(days=2)


def test_zero_duration():
    """Verify behavior when start and end timestamps are identical."""
    timestamp = datetime(2023, 4, 15, 9, 30, 0)
    start_point = CoordinatePoint(timestamp=timestamp, price=50.0)
    end_point = CoordinatePoint(timestamp=timestamp, price=75.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == 25.0
    assert measurement.percentage_change == 50.0
    assert measurement.duration == timedelta(0)


def test_sub_day_duration_resolution():
    """Verify precision with intra-day hours, minutes, and seconds."""
    start_time = datetime(2023, 6, 15, 9, 30, 0)
    end_time = datetime(2023, 6, 15, 16, 0, 30)

    start_point = CoordinatePoint(timestamp=start_time, price=100.0)
    end_point = CoordinatePoint(timestamp=end_time, price=110.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    expected_duration = timedelta(hours=6, minutes=30, seconds=30)
    assert measurement.duration == expected_duration
    assert measurement.price_delta == 10.0
    assert measurement.percentage_change == 10.0


def test_floating_point_precision():
    """Verify precision with fractional price values."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=100.25)
    end_point = CoordinatePoint(timestamp=datetime(2023, 1, 2), price=100.75)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == pytest.approx(0.5)
    expected_percentage = (0.5 / 100.25) * 100.0
    assert measurement.percentage_change == pytest.approx(expected_percentage)


def test_end_price_zero():
    """Verify percentage change calculation when end price falls to 0.0 (-100%)."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=80.0)
    end_point = CoordinatePoint(timestamp=datetime(2023, 1, 2), price=0.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == -80.0
    assert measurement.percentage_change == -100.0


def test_reverse_chronological_order():
    """Verify duration and price deltas when end point is earlier in time than start point."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 1, 10, 0, 0), price=150.0)
    end_point = CoordinatePoint(timestamp=datetime(2023, 1, 1, 0, 0), price=100.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.duration == timedelta(days=-9)
    assert measurement.price_delta == -50.0
    assert measurement.percentage_change == pytest.approx((-50.0 / 150.0) * 100.0)


def test_negative_start_price_percentage_change():
    """Verify percentage change from a negative base price (e.g., spread or oil futures)."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=-50.0)
    end_point = CoordinatePoint(timestamp=datetime(2023, 1, 2), price=-25.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.price_delta == 25.0
    # Standard financial formulation: (end - start) / abs(start) * 100
    assert measurement.percentage_change == pytest.approx(50.0)


# =====================================================================
# CoordinatePoint Validation and Properties
# =====================================================================


def test_coordinate_point_attributes_and_equality():
    """Verify CoordinatePoint stores timestamp and price and supports value equality."""
    ts = datetime(2023, 1, 1, 12, 0, 0)
    pt1 = CoordinatePoint(timestamp=ts, price=100.0)
    pt2 = CoordinatePoint(timestamp=ts, price=100.0)
    pt3 = CoordinatePoint(timestamp=ts, price=105.0)

    assert pt1.timestamp == ts
    assert pt1.price == 100.0
    assert pt1 == pt2
    assert pt1 != pt3


@pytest.mark.parametrize("invalid_timestamp", ["2023-01-01", 1672531199, None])
def test_coordinate_point_type_error_on_invalid_timestamp(invalid_timestamp):
    """Verify CoordinatePoint raises TypeError when timestamp is not a datetime instance."""
    with pytest.raises(TypeError):
        CoordinatePoint(timestamp=invalid_timestamp, price=100.0)


@pytest.mark.parametrize("invalid_price", ["100.0", None, [100.0], complex(1, 2)])
def test_coordinate_point_type_error_on_invalid_price(invalid_price):
    """Verify CoordinatePoint raises TypeError when price is not numeric (float or int)."""
    with pytest.raises(TypeError):
        CoordinatePoint(timestamp=datetime(2023, 1, 1), price=invalid_price)


def test_measurement_retains_start_and_end_point_references():
    """Verify measurement stores references to start and end CoordinatePoints."""
    start_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=10.0)
    end_point = CoordinatePoint(timestamp=datetime(2023, 1, 2), price=20.0)

    measurement = DatePriceRangeMeasurement(start=start_point, end=end_point)

    assert measurement.start == start_point
    assert measurement.end == end_point


@pytest.mark.parametrize("invalid_start", [None, "invalid", 123])
def test_measurement_invalid_start_point_type(invalid_start):
    """Verify TypeError is raised when start is not a CoordinatePoint."""
    valid_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=10.0)
    with pytest.raises(TypeError):
        DatePriceRangeMeasurement(start=invalid_start, end=valid_point)


@pytest.mark.parametrize("invalid_end", [None, "invalid", 123])
def test_measurement_invalid_end_point_type(invalid_end):
    """Verify TypeError is raised when end is not a CoordinatePoint."""
    valid_point = CoordinatePoint(timestamp=datetime(2023, 1, 1), price=10.0)
    with pytest.raises(TypeError):
        DatePriceRangeMeasurement(start=valid_point, end=invalid_end)