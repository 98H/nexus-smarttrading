"""
Unit tests for High-Probability Reversal Wick and Trend Reversal Zones.

Specification:
Story 4.2.3: Implement High-Probability Reversal Wick and Trend Reversal Zones
Acceptance Criteria:
- Given a candlestick with a lower wick that constitutes at least 60% of the total range
  and a small real body, When evaluated by the reversal wick detector, Then it is classified
  as a bullish high-probability reversal wick.
- Given a candlestick with an upper wick that constitutes at least 60% of the total range
  and a small real body, When evaluated by the reversal wick detector, Then it is classified
  as a bearish high-probability reversal wick.
- Given a detected reversal wick, When calculating the trend reversal zone, Then the zone
  boundaries are defined by the wick's extreme high/low and the candle's open/close body boundary.
- Given a candlestick where neither wick exceeds the 60% threshold, When evaluated by the
  reversal wick detector, Then it returns no reversal signal and no reversal zone.
"""

import pytest

from src.indicators import (
    Candle,
    ReversalResult,
    ReversalSignal,
    ReversalWickDetector,
    TrendReversalZone,
)
import src.indicators as indicators
import src.indicators.reversal_patterns as reversal_patterns


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_detector() -> ReversalWickDetector:
    """Provides a ReversalWickDetector with standard defaults (60% wick, 20% max body)."""
    return ReversalWickDetector(wick_ratio_threshold=0.60, max_body_ratio=0.20)


# -----------------------------------------------------------------------------
# Module Structure & Export Tests
# -----------------------------------------------------------------------------


class TestModuleExports:
    """Verifies that all required classes and enums are exposed in both public interfaces."""

    def test_reversal_patterns_exports(self) -> None:
        """Verify exports in src/indicators/reversal_patterns.py."""
        assert hasattr(reversal_patterns, "Candle")
        assert hasattr(reversal_patterns, "ReversalSignal")
        assert hasattr(reversal_patterns, "TrendReversalZone")
        assert hasattr(reversal_patterns, "ReversalResult")
        assert hasattr(reversal_patterns, "ReversalWickDetector")

    def test_indicators_package_exports(self) -> None:
        """Verify re-exports in src/indicators/__init__.py."""
        assert hasattr(indicators, "Candle")
        assert hasattr(indicators, "ReversalSignal")
        assert hasattr(indicators, "TrendReversalZone")
        assert hasattr(indicators, "ReversalResult")
        assert hasattr(indicators, "ReversalWickDetector")


# -----------------------------------------------------------------------------
# Acceptance Criteria 1: Bullish Reversal Wick
# -----------------------------------------------------------------------------


class TestBullishReversalWick:
    """
    AC 1: Given a candlestick with a lower wick that constitutes at least 60% of
    the total range and a small real body, When evaluated by the reversal wick detector,
    Then it is classified as a bullish high-probability reversal wick.
    """

    @pytest.mark.parametrize(
        "open_price, close_price, description",
        [
            (165.0, 175.0, "Bullish (green) real body"),
            (175.0, 165.0, "Bearish (red) real body"),
            (170.0, 170.0, "Doji (zero) real body"),
        ],
    )
    def test_bullish_reversal_detected_for_various_body_directions(
        self,
        default_detector: ReversalWickDetector,
        open_price: float,
        close_price: float,
        description: str,
    ) -> None:
        """
        Total range: 200 - 100 = 100.
        Lower body boundary is 165.0 (or 170 for doji).
        Lower wick is 65.0 (65%) or 70.0 (70%), both >= 60%.
        Body is 10.0 (10%) or 0.0 (0%), both <= 20% small body threshold.
        """
        candle = Candle(open=open_price, high=200.0, low=100.0, close=close_price)
        result = default_detector.evaluate(candle)

        assert isinstance(result, ReversalResult)
        assert result.signal == ReversalSignal.BULLISH, f"Failed for {description}"
        assert result.zone is not None

    def test_bullish_reversal_exact_threshold(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Lower wick is exactly 60% of the range."""
        # Range: 200 - 100 = 100
        # Lower wick: min(160, 170) - 100 = 60.0 (60%)
        # Body: |170 - 160| = 10.0 (10%)
        candle = Candle(open=160.0, high=200.0, low=100.0, close=170.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BULLISH
        assert result.lower_wick_ratio == pytest.approx(0.60)
        assert result.body_ratio == pytest.approx(0.10)

    def test_bullish_reversal_rejected_when_lower_wick_just_below_threshold(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Lower wick is 59.9% of range, which must not trigger bullish reversal."""
        # Range: 100
        # Lower wick: min(159.9, 170.0) - 100 = 59.9 (59.9%)
        candle = Candle(open=159.9, high=200.0, low=100.0, close=170.0)
        result = default_detector.evaluate(candle)

        assert result.signal != ReversalSignal.BULLISH


# -----------------------------------------------------------------------------
# Acceptance Criteria 2: Bearish Reversal Wick
# -----------------------------------------------------------------------------


class TestBearishReversalWick:
    """
    AC 2: Given a candlestick with an upper wick that constitutes at least 60% of
    the total range and a small real body, When evaluated by the reversal wick detector,
    Then it is classified as a bearish high-probability reversal wick.
    """

    @pytest.mark.parametrize(
        "open_price, close_price, description",
        [
            (135.0, 125.0, "Bearish (red) real body"),
            (125.0, 135.0, "Bullish (green) real body"),
            (130.0, 130.0, "Doji (zero) real body"),
        ],
    )
    def test_bearish_reversal_detected_for_various_body_directions(
        self,
        default_detector: ReversalWickDetector,
        open_price: float,
        close_price: float,
        description: str,
    ) -> None:
        """
        Total range: 200 - 100 = 100.
        Upper body boundary is 135.0 (or 130 for doji).
        Upper wick is 200 - 135 = 65.0 (65%) or 70.0 (70%), both >= 60%.
        Body is 10.0 (10%) or 0.0 (0%), both <= 20% small body threshold.
        """
        candle = Candle(open=open_price, high=200.0, low=100.0, close=close_price)
        result = default_detector.evaluate(candle)

        assert isinstance(result, ReversalResult)
        assert result.signal == ReversalSignal.BEARISH, f"Failed for {description}"
        assert result.zone is not None

    def test_bearish_reversal_exact_threshold(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Upper wick is exactly 60% of the range."""
        # Range: 200 - 100 = 100
        # Upper wick: 200 - max(130, 140) = 60.0 (60%)
        # Body: |140 - 130| = 10.0 (10%)
        candle = Candle(open=130.0, high=200.0, low=100.0, close=140.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BEARISH
        assert result.upper_wick_ratio == pytest.approx(0.60)
        assert result.body_ratio == pytest.approx(0.10)

    def test_bearish_reversal_rejected_when_upper_wick_just_below_threshold(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Upper wick is 59.9% of range, which must not trigger bearish reversal."""
        # Range: 100
        # Upper wick: 200 - 140.1 = 59.9 (59.9%)
        candle = Candle(open=130.0, high=200.0, low=100.0, close=140.1)
        result = default_detector.evaluate(candle)

        assert result.signal != ReversalSignal.BEARISH


# -----------------------------------------------------------------------------
# Acceptance Criteria 3: Trend Reversal Zone Calculations
# -----------------------------------------------------------------------------


class TestTrendReversalZone:
    """
    AC 3: Given a detected reversal wick, When calculating the trend reversal zone,
    Then the zone boundaries are defined by the wick's extreme high/low and the
    candle's open/close body boundary.
    """

    def test_bullish_zone_boundaries_with_bullish_body(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Bullish Candle (close > open):
        Lower wick extreme = low (100.0)
        Body lower boundary = min(open, close) = open (165.0)
        """
        candle = Candle(open=165.0, high=200.0, low=100.0, close=175.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BULLISH
        assert result.zone is not None
        assert isinstance(result.zone, TrendReversalZone)
        assert result.zone.lower_boundary == pytest.approx(100.0)
        assert result.zone.upper_boundary == pytest.approx(165.0)

    def test_bullish_zone_boundaries_with_bearish_body(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Bearish Candle (open > close):
        Lower wick extreme = low (100.0)
        Body lower boundary = min(open, close) = close (165.0)
        """
        candle = Candle(open=175.0, high=200.0, low=100.0, close=165.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BULLISH
        assert result.zone is not None
        assert result.zone.lower_boundary == pytest.approx(100.0)
        assert result.zone.upper_boundary == pytest.approx(165.0)

    def test_bearish_zone_boundaries_with_bearish_body(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Bearish Candle (open > close):
        Upper wick extreme = high (200.0)
        Body upper boundary = max(open, close) = open (135.0)
        """
        candle = Candle(open=135.0, high=200.0, low=100.0, close=125.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BEARISH
        assert result.zone is not None
        assert isinstance(result.zone, TrendReversalZone)
        assert result.zone.lower_boundary == pytest.approx(135.0)
        assert result.zone.upper_boundary == pytest.approx(200.0)

    def test_bearish_zone_boundaries_with_bullish_body(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Bullish Candle (close > open):
        Upper wick extreme = high (200.0)
        Body upper boundary = max(open, close) = close (135.0)
        """
        candle = Candle(open=125.0, high=200.0, low=100.0, close=135.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.BEARISH
        assert result.zone is not None
        assert result.zone.lower_boundary == pytest.approx(135.0)
        assert result.zone.upper_boundary == pytest.approx(200.0)

    def test_calculate_reversal_zone_direct_call(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Detector provides a direct method to calculate the zone given a candle and signal."""
        candle = Candle(open=165.0, high=200.0, low=100.0, close=175.0)
        zone = default_detector.calculate_trend_reversal_zone(
            candle, ReversalSignal.BULLISH
        )

        assert zone is not None
        assert zone.lower_boundary == pytest.approx(100.0)
        assert zone.upper_boundary == pytest.approx(165.0)


# -----------------------------------------------------------------------------
# Acceptance Criteria 4: Neither Wick Exceeds Threshold
# -----------------------------------------------------------------------------


class TestNoReversalSignal:
    """
    AC 4: Given a candlestick where neither wick exceeds the 60% threshold,
    When evaluated by the reversal wick detector, Then it returns no reversal
    signal and no reversal zone.
    """

    def test_spinning_top_balanced_wicks(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Lower wick = 40%, Upper wick = 40%, Body = 20%.
        Neither wick exceeds 60%.
        """
        # Low=100, Open=140, Close=160, High=200
        candle = Candle(open=140.0, high=200.0, low=100.0, close=160.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.NONE
        assert result.zone is None

    def test_large_body_marubozu(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        A strong trend candle with body = 90% and minimal wicks.
        """
        candle = Candle(open=105.0, high=200.0, low=100.0, close=195.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.NONE
        assert result.zone is None

    def test_direct_calculate_zone_for_none_signal_returns_none(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Direct calculation with ReversalSignal.NONE yields None."""
        candle = Candle(open=140.0, high=200.0, low=100.0, close=160.0)
        zone = default_detector.calculate_trend_reversal_zone(candle, ReversalSignal.NONE)
        assert zone is None


# -----------------------------------------------------------------------------
# Real Body Constraints & Configurable Thresholds
# -----------------------------------------------------------------------------


class TestBodyRatioConstraintAndConfiguration:
    """
    Ensures the 'small real body' rule is strictly enforced and that
    thresholds can be customized.
    """

    def test_rejected_when_lower_wick_is_large_but_body_is_not_small(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Lower wick = 60.0 (60%), Upper wick = 0.0 (0%), Body = 40.0 (40%).
        Default max_body_ratio is 20%. Body is too large for a reversal wick pattern.
        """
        candle = Candle(open=160.0, high=200.0, low=100.0, close=200.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.NONE
        assert result.zone is None

    def test_rejected_when_upper_wick_is_large_but_body_is_not_small(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """
        Upper wick = 60.0 (60%), Lower wick = 0.0 (0%), Body = 40.0 (40%).
        Default max_body_ratio is 20%.
        """
        candle = Candle(open=100.0, high=200.0, low=100.0, close=140.0)
        result = default_detector.evaluate(candle)

        assert result.signal == ReversalSignal.NONE
        assert result.zone is None

    def test_custom_thresholds_override(self) -> None:
        """Detector initialized with custom wick and body thresholds operates accordingly."""
        custom_detector = ReversalWickDetector(
            wick_ratio_threshold=0.70, max_body_ratio=0.10
        )
        # Lower wick: 65% (would pass 60%, fails 70%)
        candle = Candle(open=165.0, high=200.0, low=100.0, close=175.0)
        result = custom_detector.evaluate(candle)

        assert result.signal == ReversalSignal.NONE
        assert result.zone is None


# -----------------------------------------------------------------------------
# Input Validation & Edge Cases
# -----------------------------------------------------------------------------


class TestInputValidationAndEdgeCases:
    """Tests invalid candlestick configurations and edge cases."""

    def test_zero_range_candle_raises_value_error(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """A flat candle with high == low has 0 total range and cannot be evaluated."""
        candle = Candle(open=100.0, high=100.0, low=100.0, close=100.0)
        with pytest.raises(ValueError):
            default_detector.evaluate(candle)

    def test_invalid_high_low_inverted_raises_value_error(self) -> None:
        """Candle where high < low must raise ValueError during instantiation or evaluation."""
        with pytest.raises(ValueError):
            Candle(open=150.0, high=100.0, low=200.0, close=150.0)

    def test_open_price_outside_high_low_raises_value_error(self) -> None:
        """Open price exceeding High must raise ValueError."""
        with pytest.raises(ValueError):
            Candle(open=250.0, high=200.0, low=100.0, close=150.0)

    def test_close_price_outside_high_low_raises_value_error(self) -> None:
        """Close price below Low must raise ValueError."""
        with pytest.raises(ValueError):
            Candle(open=150.0, high=200.0, low=100.0, close=50.0)

    def test_invalid_wick_threshold_initialization_raises_value_error(self) -> None:
        """Wick threshold <= 0 or > 1.0 must raise ValueError."""
        with pytest.raises(ValueError):
            ReversalWickDetector(wick_ratio_threshold=1.5)

        with pytest.raises(ValueError):
            ReversalWickDetector(wick_ratio_threshold=0.0)

    def test_invalid_max_body_ratio_initialization_raises_value_error(self) -> None:
        """Max body ratio <= 0 or > 1.0 must raise ValueError."""
        with pytest.raises(ValueError):
            ReversalWickDetector(max_body_ratio=-0.1)

        with pytest.raises(ValueError):
            ReversalWickDetector(max_body_ratio=1.1)

    def test_immutable_candle_and_zone(
        self, default_detector: ReversalWickDetector
    ) -> None:
        """Ensure core dataclasses are frozen to prevent accidental state mutation."""
        candle = Candle(open=165.0, high=200.0, low=100.0, close=175.0)
        with pytest.raises(Exception):
            candle.open = 180.0  # type: ignore[misc]

        result = default_detector.evaluate(candle)
        assert result.zone is not None
        with pytest.raises(Exception):
            result.zone.lower_boundary = 90.0  # type: ignore[misc]