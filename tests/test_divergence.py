from datetime import datetime, timezone
import pytest

from src.analysis.types import (
    DivergenceResult,
    DivergenceType,
    SwingPoint,
    SwingType,
)
from src.analysis.divergence import DivergenceEngine, detect_divergence


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> DivergenceEngine:
    """Fixture providing a fresh DivergenceEngine instance."""
    return DivergenceEngine()


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixture providing a base UTC timestamp."""
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# Types and Data Structures Tests
# ============================================================================


class TestTypes:
    """Tests for types, enums, and data models."""

    def test_divergence_type_enum_members(self):
        assert DivergenceType.REGULAR_BULLISH.value == "REGULAR_BULLISH"
        assert DivergenceType.REGULAR_BEARISH.value == "REGULAR_BEARISH"
        assert DivergenceType.HIDDEN_BULLISH.value == "HIDDEN_BULLISH"
        assert DivergenceType.HIDDEN_BEARISH.value == "HIDDEN_BEARISH"

    def test_swing_type_enum_members(self):
        assert SwingType.HIGH.value == "HIGH"
        assert SwingType.LOW.value == "LOW"

    def test_swing_point_creation_and_attributes(self, base_timestamp):
        point = SwingPoint(
            index=10,
            price=150.50,
            oscillator=35.20,
            swing_type=SwingType.LOW,
            timestamp=base_timestamp,
        )
        assert point.index == 10
        assert point.price == 150.50
        assert point.oscillator == 35.20
        assert point.swing_type == SwingType.LOW
        assert point.timestamp == base_timestamp

    def test_swing_point_is_frozen_dataclass(self):
        point = SwingPoint(
            index=1,
            price=100.0,
            oscillator=50.0,
            swing_type=SwingType.HIGH,
        )
        with pytest.raises((AttributeError, TypeError)):
            point.price = 105.0  # type: ignore[misc]

    def test_divergence_result_attributes(self):
        prev = SwingPoint(index=1, price=100.0, oscillator=30.0, swing_type=SwingType.LOW)
        curr = SwingPoint(index=5, price=90.0, oscillator=35.0, swing_type=SwingType.LOW)
        result = DivergenceResult(
            divergence_type=DivergenceType.REGULAR_BULLISH,
            swing_type=SwingType.LOW,
            prev_point=prev,
            curr_point=curr,
        )
        assert result.divergence_type == DivergenceType.REGULAR_BULLISH
        assert result.swing_type == SwingType.LOW
        assert result.prev_point == prev
        assert result.curr_point == curr


# ============================================================================
# Regular Bullish Divergence Tests
# Price: Lower Low, Oscillator: Higher Low
# ============================================================================


class TestRegularBullishDivergence:
    """
    Acceptance Criteria 1:
    Given a series of price swing lows and corresponding oscillator swing lows,
    When the latest price swing low is lower than the previous low but the
    oscillator swing low is higher,
    Then the engine must detect and classify a Regular Bullish Divergence.
    """

    @pytest.mark.parametrize(
        ("price_prev", "price_curr", "osc_prev", "osc_curr"),
        [
            (100.0, 95.0, 20.0, 25.0),
            (1050.25, 1049.75, 15.10, 15.20),
            (0.0050, 0.0040, -10.0, -5.0),
            (50.0, 40.0, -50.0, 10.0),
        ],
    )
    def test_regular_bullish_detected_via_function(
        self, price_prev: float, price_curr: float, osc_prev: float, osc_curr: float
    ):
        prev_swing = SwingPoint(
            index=10, price=price_prev, oscillator=osc_prev, swing_type=SwingType.LOW
        )
        curr_swing = SwingPoint(
            index=20, price=price_curr, oscillator=osc_curr, swing_type=SwingType.LOW
        )

        result = detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert isinstance(result, DivergenceResult)
        assert result.divergence_type == DivergenceType.REGULAR_BULLISH
        assert result.swing_type == SwingType.LOW
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing

    def test_regular_bullish_detected_via_engine_method(self, engine: DivergenceEngine):
        prev_swing = SwingPoint(index=1, price=120.0, oscillator=22.0, swing_type=SwingType.LOW)
        curr_swing = SwingPoint(index=7, price=110.0, oscillator=28.0, swing_type=SwingType.LOW)

        result = engine.detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert result.divergence_type == DivergenceType.REGULAR_BULLISH
        assert result.swing_type == SwingType.LOW
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing


# ============================================================================
# Regular Bearish Divergence Tests
# Price: Higher High, Oscillator: Lower High
# ============================================================================


class TestRegularBearishDivergence:
    """
    Acceptance Criteria 2:
    Given a series of price swing highs and corresponding oscillator swing highs,
    When the latest price swing high is higher than the previous high but the
    oscillator swing high is lower,
    Then the engine must detect and classify a Regular Bearish Divergence.
    """

    @pytest.mark.parametrize(
        ("price_prev", "price_curr", "osc_prev", "osc_curr"),
        [
            (100.0, 105.0, 80.0, 75.0),
            (2500.0, 2550.5, 78.4, 71.2),
            (1.50, 1.55, 60.0, 50.0),
            (10.0, 20.0, 0.0, -10.0),
        ],
    )
    def test_regular_bearish_detected_via_function(
        self, price_prev: float, price_curr: float, osc_prev: float, osc_curr: float
    ):
        prev_swing = SwingPoint(
            index=10, price=price_prev, oscillator=osc_prev, swing_type=SwingType.HIGH
        )
        curr_swing = SwingPoint(
            index=20, price=price_curr, oscillator=osc_curr, swing_type=SwingType.HIGH
        )

        result = detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert isinstance(result, DivergenceResult)
        assert result.divergence_type == DivergenceType.REGULAR_BEARISH
        assert result.swing_type == SwingType.HIGH
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing

    def test_regular_bearish_detected_via_engine_method(self, engine: DivergenceEngine):
        prev_swing = SwingPoint(index=5, price=200.0, oscillator=85.0, swing_type=SwingType.HIGH)
        curr_swing = SwingPoint(index=15, price=215.0, oscillator=79.0, swing_type=SwingType.HIGH)

        result = engine.detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert result.divergence_type == DivergenceType.REGULAR_BEARISH
        assert result.swing_type == SwingType.HIGH
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing


# ============================================================================
# Hidden Bullish Divergence Tests
# Price: Higher Low, Oscillator: Lower Low
# ============================================================================


class TestHiddenBullishDivergence:
    """
    Acceptance Criteria 3:
    Given a series of price swing lows and corresponding oscillator swing lows,
    When the latest price swing low is higher than the previous low but the
    oscillator swing low is lower,
    Then the engine must detect and classify a Hidden Bullish Divergence.
    """

    @pytest.mark.parametrize(
        ("price_prev", "price_curr", "osc_prev", "osc_curr"),
        [
            (100.0, 105.0, 30.0, 25.0),
            (500.0, 501.0, 40.0, 32.0),
            (0.010, 0.012, -15.0, -25.0),
            (75.5, 80.0, 20.0, 15.0),
        ],
    )
    def test_hidden_bullish_detected_via_function(
        self, price_prev: float, price_curr: float, osc_prev: float, osc_curr: float
    ):
        prev_swing = SwingPoint(
            index=10, price=price_prev, oscillator=osc_prev, swing_type=SwingType.LOW
        )
        curr_swing = SwingPoint(
            index=20, price=price_curr, oscillator=osc_curr, swing_type=SwingType.LOW
        )

        result = detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert isinstance(result, DivergenceResult)
        assert result.divergence_type == DivergenceType.HIDDEN_BULLISH
        assert result.swing_type == SwingType.LOW
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing

    def test_hidden_bullish_detected_via_engine_method(self, engine: DivergenceEngine):
        prev_swing = SwingPoint(index=2, price=300.0, oscillator=35.0, swing_type=SwingType.LOW)
        curr_swing = SwingPoint(index=9, price=310.0, oscillator=28.0, swing_type=SwingType.LOW)

        result = engine.detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert result.divergence_type == DivergenceType.HIDDEN_BULLISH
        assert result.swing_type == SwingType.LOW
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing


# ============================================================================
# Hidden Bearish Divergence Tests
# Price: Lower High, Oscillator: Higher High
# ============================================================================


class TestHiddenBearishDivergence:
    """
    Acceptance Criteria 4:
    Given a series of price swing highs and corresponding oscillator swing highs,
    When the latest price swing high is lower than the previous high but the
    oscillator swing high is higher,
    Then the engine must detect and classify a Hidden Bearish Divergence.
    """

    @pytest.mark.parametrize(
        ("price_prev", "price_curr", "osc_prev", "osc_curr"),
        [
            (100.0, 95.0, 70.0, 75.0),
            (1500.0, 1490.0, 65.0, 72.0),
            (2.50, 2.45, 55.0, 65.0),
            (45.0, 42.0, -5.0, 5.0),
        ],
    )
    def test_hidden_bearish_detected_via_function(
        self, price_prev: float, price_curr: float, osc_prev: float, osc_curr: float
    ):
        prev_swing = SwingPoint(
            index=10, price=price_prev, oscillator=osc_prev, swing_type=SwingType.HIGH
        )
        curr_swing = SwingPoint(
            index=20, price=price_curr, oscillator=osc_curr, swing_type=SwingType.HIGH
        )

        result = detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert isinstance(result, DivergenceResult)
        assert result.divergence_type == DivergenceType.HIDDEN_BEARISH
        assert result.swing_type == SwingType.HIGH
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing

    def test_hidden_bearish_detected_via_engine_method(self, engine: DivergenceEngine):
        prev_swing = SwingPoint(index=3, price=450.0, oscillator=60.0, swing_type=SwingType.HIGH)
        curr_swing = SwingPoint(index=8, price=440.0, oscillator=68.0, swing_type=SwingType.HIGH)

        result = engine.detect_divergence(prev_swing, curr_swing)

        assert result is not None
        assert result.divergence_type == DivergenceType.HIDDEN_BEARISH
        assert result.swing_type == SwingType.HIGH
        assert result.prev_point == prev_swing
        assert result.curr_point == curr_swing


# ============================================================================
# Non-Divergence / Trend Continuation Scenarios
# ============================================================================


class TestNoDivergenceScenarios:
    """Ensure engine returns None when price and oscillator confirm each other."""

    @pytest.mark.parametrize(
        ("price_prev", "price_curr", "osc_prev", "osc_curr", "swing_type"),
        [
            # Price LL, Osc LL (Low confirmation in downtrend)
            (100.0, 90.0, 30.0, 20.0, SwingType.LOW),
            # Price HH, Osc HH (High confirmation in uptrend)
            (100.0, 110.0, 70.0, 80.0, SwingType.HIGH),
            # Price HL, Osc HL (Low confirmation in uptrend)
            (100.0, 110.0, 30.0, 40.0, SwingType.LOW),
            # Price LH, Osc LH (High confirmation in downtrend)
            (100.0, 90.0, 70.0, 60.0, SwingType.HIGH),
            # Equal Price
            (100.0, 100.0, 30.0, 35.0, SwingType.LOW),
            (100.0, 100.0, 70.0, 65.0, SwingType.HIGH),
            # Equal Oscillator
            (90.0, 80.0, 30.0, 30.0, SwingType.LOW),
            (80.0, 90.0, 70.0, 70.0, SwingType.HIGH),
            # Identical points
            (100.0, 100.0, 50.0, 50.0, SwingType.LOW),
            (100.0, 100.0, 50.0, 50.0, SwingType.HIGH),
        ],
    )
    def test_no_divergence_returns_none(
        self,
        price_prev: float,
        price_curr: float,
        osc_prev: float,
        osc_curr: float,
        swing_type: SwingType,
    ):
        prev_swing = SwingPoint(
            index=1, price=price_prev, oscillator=osc_prev, swing_type=swing_type
        )
        curr_swing = SwingPoint(
            index=5, price=price_curr, oscillator=osc_curr, swing_type=swing_type
        )

        assert detect_divergence(prev_swing, curr_swing) is None


# ============================================================================
# Validation and Edge Cases
# ============================================================================


class TestValidationAndEdgeCases:
    """Tests for invalid inputs, temporal consistency, and edge conditions."""

    def test_mismatched_swing_types_raises_value_error(self):
        high_swing = SwingPoint(index=1, price=100.0, oscillator=70.0, swing_type=SwingType.HIGH)
        low_swing = SwingPoint(index=5, price=80.0, oscillator=30.0, swing_type=SwingType.LOW)

        with pytest.raises(ValueError):
            detect_divergence(high_swing, low_swing)

        with pytest.raises(ValueError):
            detect_divergence(low_swing, high_swing)

    def test_non_chronological_indices_raises_value_error(self):
        prev_swing = SwingPoint(index=10, price=100.0, oscillator=30.0, swing_type=SwingType.LOW)
        curr_swing = SwingPoint(index=5, price=90.0, oscillator=35.0, swing_type=SwingType.LOW)

        with pytest.raises(ValueError):
            detect_divergence(prev_swing, curr_swing)

    def test_identical_indices_raises_value_error(self):
        prev_swing = SwingPoint(index=10, price=100.0, oscillator=30.0, swing_type=SwingType.LOW)
        curr_swing = SwingPoint(index=10, price=90.0, oscillator=35.0, swing_type=SwingType.LOW)

        with pytest.raises(ValueError):
            detect_divergence(prev_swing, curr_swing)

    def test_non_chronological_timestamps_raises_value_error(self):
        t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        prev_swing = SwingPoint(
            index=1, price=100.0, oscillator=30.0, swing_type=SwingType.LOW, timestamp=t1
        )
        curr_swing = SwingPoint(
            index=2, price=90.0, oscillator=35.0, swing_type=SwingType.LOW, timestamp=t2
        )

        with pytest.raises(ValueError):
            detect_divergence(prev_swing, curr_swing)


# ============================================================================
# Real-Time Streaming State Engine Tests
# ============================================================================


class TestRealTimeStreamingDivergenceEngine:
    """Tests stateful, sequential real-time swing ingestion by DivergenceEngine."""

    def test_initial_swings_return_none_until_sufficient_history(self, engine: DivergenceEngine):
        first_low = SwingPoint(index=1, price=100.0, oscillator=25.0, swing_type=SwingType.LOW)
        first_high = SwingPoint(index=3, price=110.0, oscillator=75.0, swing_type=SwingType.HIGH)

        assert engine.process_swing(first_low) is None
        assert engine.process_swing(first_high) is None

    def test_streaming_regular_bullish_divergence(self, engine: DivergenceEngine):
        low_1 = SwingPoint(index=1, price=100.0, oscillator=20.0, swing_type=SwingType.LOW)
        high_1 = SwingPoint(index=5, price=115.0, oscillator=65.0, swing_type=SwingType.HIGH)
        low_2 = SwingPoint(index=10, price=95.0, oscillator=26.0, swing_type=SwingType.LOW)

        assert engine.process_swing(low_1) is None
        assert engine.process_swing(high_1) is None

        result = engine.process_swing(low_2)
        assert result is not None
        assert result.divergence_type == DivergenceType.REGULAR_BULLISH
        assert result.prev_point == low_1
        assert result.curr_point == low_2

    def test_streaming_regular_bearish_divergence(self, engine: DivergenceEngine):
        high_1 = SwingPoint(index=2, price=150.0, oscillator=80.0, swing_type=SwingType.HIGH)
        low_1 = SwingPoint(index=6, price=130.0, oscillator=40.0, swing_type=SwingType.LOW)
        high_2 = SwingPoint(index=12, price=160.0, oscillator=72.0, swing_type=SwingType.HIGH)

        assert engine.process_swing(high_1) is None
        assert engine.process_swing(low_1) is None

        result = engine.process_swing(high_2)
        assert result is not None
        assert result.divergence_type == DivergenceType.REGULAR_BEARISH
        assert result.prev_point == high_1
        assert result.curr_point == high_2

    def test_streaming_hidden_bullish_divergence(self, engine: DivergenceEngine):
        low_1 = SwingPoint(index=1, price=100.0, oscillator=30.0, swing_type=SwingType.LOW)
        high_1 = SwingPoint(index=4, price=120.0, oscillator=65.0, swing_type=SwingType.HIGH)
        low_2 = SwingPoint(index=8, price=105.0, oscillator=22.0, swing_type=SwingType.LOW)

        assert engine.process_swing(low_1) is None
        assert engine.process_swing(high_1) is None

        result = engine.process_swing(low_2)
        assert result is not None
        assert result.divergence_type == DivergenceType.HIDDEN_BULLISH
        assert result.prev_point == low_1
        assert result.curr_point == low_2

    def test_streaming_hidden_bearish_divergence(self, engine: DivergenceEngine):
        high_1 = SwingPoint(index=1, price=200.0, oscillator=65.0, swing_type=SwingType.HIGH)
        low_1 = SwingPoint(index=5, price=170.0, oscillator=35.0, swing_type=SwingType.LOW)
        high_2 = SwingPoint(index=9, price=190.0, oscillator=75.0, swing_type=SwingType.HIGH)

        assert engine.process_swing(high_1) is None
        assert engine.process_swing(low_1) is None

        result = engine.process_swing(high_2)
        assert result is not None
        assert result.divergence_type == DivergenceType.HIDDEN_BEARISH
        assert result.prev_point == high_1
        assert result.curr_point == high_2

    def test_consecutive_same_swing_types_updates_reference_point(
        self, engine: DivergenceEngine
    ):
        # Three successive lows
        low_1 = SwingPoint(index=1, price=100.0, oscillator=20.0, swing_type=SwingType.LOW)
        low_2 = SwingPoint(index=5, price=95.0, oscillator=25.0, swing_type=SwingType.LOW)
        low_3 = SwingPoint(index=10, price=90.0, oscillator=30.0, swing_type=SwingType.LOW)

        assert engine.process_swing(low_1) is None

        res_1 = engine.process_swing(low_2)
        assert res_1 is not None
        assert res_1.divergence_type == DivergenceType.REGULAR_BULLISH
        assert res_1.prev_point == low_1
        assert res_1.curr_point == low_2

        # low_3 should compare against low_2
        res_2 = engine.process_swing(low_3)
        assert res_2 is not None
        assert res_2.divergence_type == DivergenceType.REGULAR_BULLISH
        assert res_2.prev_point == low_2
        assert res_2.curr_point == low_3

    def test_engine_reset_clears_internal_history(self, engine: DivergenceEngine):
        low_1 = SwingPoint(index=1, price=100.0, oscillator=20.0, swing_type=SwingType.LOW)
        assert engine.process_swing(low_1) is None

        engine.reset()

        # After reset, the next low has no previous point to compare with
        low_2 = SwingPoint(index=5, price=95.0, oscillator=25.0, swing_type=SwingType.LOW)
        assert engine.process_swing(low_2) is None

    def test_streaming_unordered_points_raises_value_error(self, engine: DivergenceEngine):
        p1 = SwingPoint(index=10, price=100.0, oscillator=20.0, swing_type=SwingType.LOW)
        p2 = SwingPoint(index=8, price=95.0, oscillator=25.0, swing_type=SwingType.LOW)

        engine.process_swing(p1)
        with pytest.raises(ValueError):
            engine.process_swing(p2)