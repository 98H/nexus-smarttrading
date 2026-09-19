from datetime import datetime, timezone
from typing import Callable
import pytest

from src.analysis.market_structure import (
    Candle,
    MarketStructureEngine,
    MarketStructureEvent,
    StructureEventType,
    TrendState,
)


@pytest.fixture
def candle_factory() -> Callable[..., Candle]:
    """Factory fixture for generating test candle instances."""

    def _create_candle(
        close: float,
        open_price: float | None = None,
        high: float | None = None,
        low: float | None = None,
        volume: float = 1000.0,
        timestamp: datetime | None = None,
    ) -> Candle:
        open_val = open_price if open_price is not None else close
        high_val = high if high is not None else max(open_val, close) + 1.0
        low_val = low if low is not None else min(open_val, close) - 1.0
        ts = timestamp or datetime.now(timezone.utc)

        return Candle(
            timestamp=ts,
            open=open_val,
            high=high_val,
            low=low_val,
            close=close,
            volume=volume,
        )

    return _create_candle


# ============================================================================
# Acceptance Criteria Tests
# ============================================================================


def test_bullish_trend_close_above_swing_high_emits_bos_and_maintains_trend(
    candle_factory: Callable[..., Candle],
) -> None:
    """
    Acceptance Criteria:
    Given an established bullish trend with a confirmed swing high at 100.0,
    When a new candle closes above 100.0,
    Then emit a Break of Structure (BOS) event and maintain the bullish trend state.
    """
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    breakout_candle = candle_factory(
        open_price=98.0,
        close=101.5,
        high=102.0,
        low=97.5,
    )

    event = engine.process_candle(breakout_candle)

    assert event is not None
    assert isinstance(event, MarketStructureEvent)
    assert event.event_type == StructureEventType.BOS
    assert event.broken_level == 100.0
    assert event.previous_trend == TrendState.BULLISH
    assert event.new_trend == TrendState.BULLISH
    assert event.candle == breakout_candle
    assert engine.current_trend == TrendState.BULLISH


def test_bullish_trend_close_below_swing_low_emits_choch_and_flips_to_bearish(
    candle_factory: Callable[..., Candle],
) -> None:
    """
    Acceptance Criteria:
    Given an established bullish trend with a confirmed swing low at 90.0,
    When a new candle closes below 90.0,
    Then emit a Change of Character (CHoCH) event and flip the market structure state to bearish.
    """
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    reversal_candle = candle_factory(
        open_price=92.0,
        close=88.5,
        high=92.5,
        low=88.0,
    )

    event = engine.process_candle(reversal_candle)

    assert event is not None
    assert isinstance(event, MarketStructureEvent)
    assert event.event_type == StructureEventType.CHOCH
    assert event.broken_level == 90.0
    assert event.previous_trend == TrendState.BULLISH
    assert event.new_trend == TrendState.BEARISH
    assert event.candle == reversal_candle
    assert engine.current_trend == TrendState.BEARISH


def test_bearish_trend_close_above_swing_high_emits_choch_and_flips_to_bullish(
    candle_factory: Callable[..., Candle],
) -> None:
    """
    Acceptance Criteria:
    Given an established bearish trend with a confirmed swing high at 105.0,
    When a new candle closes above 105.0,
    Then emit a Change of Character (CHoCH) event and flip the market structure state to bullish.
    """
    engine = MarketStructureEngine(
        initial_trend=TrendState.BEARISH,
        swing_high=105.0,
        swing_low=95.0,
    )

    reversal_candle = candle_factory(
        open_price=103.0,
        close=106.5,
        high=107.0,
        low=102.5,
    )

    event = engine.process_candle(reversal_candle)

    assert event is not None
    assert isinstance(event, MarketStructureEvent)
    assert event.event_type == StructureEventType.CHOCH
    assert event.broken_level == 105.0
    assert event.previous_trend == TrendState.BEARISH
    assert event.new_trend == TrendState.BULLISH
    assert event.candle == reversal_candle
    assert engine.current_trend == TrendState.BULLISH


# ============================================================================
# Complementary Structural & Edge Case Tests
# ============================================================================


def test_bearish_trend_close_below_swing_low_emits_bos_and_maintains_trend(
    candle_factory: Callable[..., Candle],
) -> None:
    """Bearish BOS: Close below swing low maintains bearish trend and emits BOS."""
    engine = MarketStructureEngine(
        initial_trend=TrendState.BEARISH,
        swing_high=105.0,
        swing_low=95.0,
    )

    continuation_candle = candle_factory(
        open_price=96.0,
        close=94.0,
        high=96.5,
        low=93.5,
    )

    event = engine.process_candle(continuation_candle)

    assert event is not None
    assert event.event_type == StructureEventType.BOS
    assert event.broken_level == 95.0
    assert event.previous_trend == TrendState.BEARISH
    assert event.new_trend == TrendState.BEARISH
    assert engine.current_trend == TrendState.BEARISH


def test_wick_penetration_without_close_beyond_level_does_not_trigger_event(
    candle_factory: Callable[..., Candle],
) -> None:
    """A wick piercing the swing level without a candle close beyond it must NOT emit any event."""
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    wick_high_candle = candle_factory(
        open_price=98.0,
        close=99.5,  # Close is <= 100.0
        high=103.0,  # Wick penetrated above 100.0
        low=97.0,
    )
    event_high = engine.process_candle(wick_high_candle)
    assert event_high is None
    assert engine.current_trend == TrendState.BULLISH

    wick_low_candle = candle_factory(
        open_price=92.0,
        close=90.5,  # Close is >= 90.0
        high=93.0,
        low=87.0,  # Wick penetrated below 90.0
    )
    event_low = engine.process_candle(wick_low_candle)
    assert event_low is None
    assert engine.current_trend == TrendState.BULLISH


def test_candle_close_exactly_on_swing_level_does_not_trigger_event(
    candle_factory: Callable[..., Candle],
) -> None:
    """Close equal to swing level (not strictly above/below) must NOT trigger BOS or CHoCH."""
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    equal_high_candle = candle_factory(
        open_price=99.0,
        close=100.0,
        high=100.5,
        low=98.5,
    )
    assert engine.process_candle(equal_high_candle) is None

    equal_low_candle = candle_factory(
        open_price=91.0,
        close=90.0,
        high=91.5,
        low=89.5,
    )
    assert engine.process_candle(equal_low_candle) is None


def test_candle_inside_range_maintains_trend_and_emits_nothing(
    candle_factory: Callable[..., Candle],
) -> None:
    """Candles trading strictly between swing high and swing low emit no events."""
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    inside_candle = candle_factory(
        open_price=95.0,
        close=96.0,
        high=97.0,
        low=94.0,
    )

    event = engine.process_candle(inside_candle)
    assert event is None
    assert engine.current_trend == TrendState.BULLISH


# ============================================================================
# Validation & Error Handling Tests
# ============================================================================


def test_invalid_swing_levels_raises_value_error() -> None:
    """Initializing engine with swing_high <= swing_low must raise ValueError."""
    with pytest.raises(ValueError):
        MarketStructureEngine(
            initial_trend=TrendState.BULLISH,
            swing_high=90.0,
            swing_low=100.0,
        )

    with pytest.raises(ValueError):
        MarketStructureEngine(
            initial_trend=TrendState.BULLISH,
            swing_high=100.0,
            swing_low=100.0,
        )


def test_invalid_candle_data_raises_value_error() -> None:
    """Candle with invalid OHLC relationship (high < low) must raise ValueError."""
    with pytest.raises(ValueError):
        Candle(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=90.0,
            low=110.0,
            close=95.0,
            volume=100.0,
        )


def test_set_swing_levels_runtime_validation() -> None:
    """Dynamic updates to swing levels must validate high > low constraint."""
    engine = MarketStructureEngine(
        initial_trend=TrendState.BULLISH,
        swing_high=100.0,
        swing_low=90.0,
    )

    with pytest.raises(ValueError):
        engine.set_swing_high(85.0)

    with pytest.raises(ValueError):
        engine.set_swing_low(105.0)