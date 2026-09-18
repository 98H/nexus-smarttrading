"""
Unit tests for Fair Value Gap (FVG) and Imbalance Zone Detector.

Story: 4.1.3: Implement Fair Value Gap (FVG) and Imbalance Zone Detector
Modules under test:
- src.indicators.fvg_detector
- src.indicators
"""

import math
from typing import List

import pandas as pd
import pytest

from src.indicators import Candle, FairValueGap, FVGDetector, FVGType
import src.indicators as indicators_module
import src.indicators.fvg_detector as fvg_module


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def detector() -> FVGDetector:
    """Provides a fresh instance of FVGDetector for each test."""
    return FVGDetector()


@pytest.fixture
def bullish_candle_sequence() -> List[Candle]:
    """
    Standard 3-candle bullish sequence:
    Candle 1: High = 100.0, Low = 90.0
    Candle 2: Large bullish impulse candle
    Candle 3: High = 120.0, Low = 105.0
    Condition: Candle 3 Low (105.0) > Candle 1 High (100.0) -> Bullish FVG [100.0, 105.0]
    """
    return [
        Candle(open=92.0, high=100.0, low=90.0, close=98.0, index=0),
        Candle(open=98.0, high=115.0, low=97.0, close=114.0, index=1),
        Candle(open=114.0, high=120.0, low=105.0, close=118.0, index=2),
    ]


@pytest.fixture
def bearish_candle_sequence() -> List[Candle]:
    """
    Standard 3-candle bearish sequence:
    Candle 1: High = 110.0, Low = 100.0
    Candle 2: Large bearish impulse candle
    Candle 3: High = 95.0, Low = 80.0
    Condition: Candle 3 High (95.0) < Candle 1 Low (100.0) -> Bearish FVG [95.0, 100.0]
    """
    return [
        Candle(open=108.0, high=110.0, low=100.0, close=102.0, index=0),
        Candle(open=102.0, high=103.0, low=85.0, close=86.0, index=1),
        Candle(open=86.0, high=95.0, low=80.0, close=82.0, index=2),
    ]


@pytest.fixture
def overlapping_candle_sequence() -> List[Candle]:
    """
    Standard 3-candle sequence without imbalance:
    Candle 1: High = 100.0, Low = 90.0
    Candle 2: Normal range
    Candle 3: High = 102.0, Low = 95.0
    Condition: Candle 3 Low (95.0) <= Candle 1 High (100.0) AND Candle 3 High (102.0) >= Candle 1 Low (90.0)
    """
    return [
        Candle(open=92.0, high=100.0, low=90.0, close=95.0, index=0),
        Candle(open=95.0, high=104.0, low=93.0, close=101.0, index=1),
        Candle(open=101.0, high=102.0, low=95.0, close=97.0, index=2),
    ]


# =====================================================================
# Test Module Interface and Exports
# =====================================================================


class TestModuleExports:
    """Verifies that the required classes and enums are properly exported."""

    def test_indicators_package_exports_expected_symbols(self):
        """All public detector primitives must be exported at the package root."""
        expected_symbols = {"FVGDetector", "FairValueGap", "FVGType", "Candle"}
        assert expected_symbols.issubset(set(dir(indicators_module)))

        if hasattr(indicators_module, "__all__"):
            assert expected_symbols.issubset(set(indicators_module.__all__))

    def test_fvg_detector_module_exports_expected_symbols(self):
        """All public detector primitives must be exported from the submodule."""
        expected_symbols = {"FVGDetector", "FairValueGap", "FVGType", "Candle"}
        assert expected_symbols.issubset(set(dir(fvg_module)))

        if hasattr(fvg_module, "__all__"):
            assert expected_symbols.issubset(set(fvg_module.__all__))

    def test_fvg_type_enum_members(self):
        """FVGType must strictly support BULLISH and BEARISH members."""
        assert hasattr(FVGType, "BULLISH")
        assert hasattr(FVGType, "BEARISH")
        assert FVGType.BULLISH != FVGType.BEARISH


# =====================================================================
# Test Acceptance Criterion 1: Bullish FVG Detection
# =====================================================================


class TestBullishFairValueGap:
    """
    Acceptance Criterion:
    Given a series of 3 consecutive candles where Candle 3's low is greater
    than Candle 1's high, When the FVG detector evaluates the sequence,
    Then it registers a Bullish Fair Value Gap with the gap boundaries
    defined by [Candle 1 High, Candle 3 Low].
    """

    def test_bullish_fvg_detected_when_candle3_low_exceeds_candle1_high(
        self, detector: FVGDetector, bullish_candle_sequence: List[Candle]
    ):
        c1, c2, c3 = bullish_candle_sequence
        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        assert isinstance(gap, FairValueGap)
        assert gap.gap_type == FVGType.BULLISH
        assert gap.lower_boundary == pytest.approx(c1.high)
        assert gap.upper_boundary == pytest.approx(c3.low)
        assert list(gap.boundaries) == [pytest.approx(c1.high), pytest.approx(c3.low)]
        assert gap.bar_index == c3.index

    @pytest.mark.parametrize(
        "c1_high, c3_low",
        [
            (10.0, 10.05),
            (1500.50, 1500.51),
            (0.00012, 0.00015),
            (99999.0, 100000.0),
        ],
    )
    def test_bullish_fvg_with_various_price_scales(
        self, detector: FVGDetector, c1_high: float, c3_low: float
    ):
        c1 = Candle(open=c1_high - 2.0, high=c1_high, low=c1_high - 5.0, close=c1_high - 1.0, index=0)
        c2 = Candle(open=c1.close, high=c3_low + 5.0, low=c1.close - 1.0, close=c3_low + 4.0, index=1)
        c3 = Candle(open=c2.close, high=c3_low + 10.0, low=c3_low, close=c3_low + 2.0, index=2)

        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        assert gap.gap_type == FVGType.BULLISH
        assert gap.lower_boundary == pytest.approx(c1_high)
        assert gap.upper_boundary == pytest.approx(c3_low)
        assert list(gap.boundaries) == [pytest.approx(c1_high), pytest.approx(c3_low)]

    def test_bullish_fvg_gap_size_calculation(
        self, detector: FVGDetector, bullish_candle_sequence: List[Candle]
    ):
        c1, c2, c3 = bullish_candle_sequence
        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        expected_size = c3.low - c1.high
        if hasattr(gap, "size"):
            assert gap.size == pytest.approx(expected_size)


# =====================================================================
# Test Acceptance Criterion 2: Bearish FVG Detection
# =====================================================================


class TestBearishFairValueGap:
    """
    Acceptance Criterion:
    Given a series of 3 consecutive candles where Candle 3's high is lower
    than Candle 1's low, When the FVG detector evaluates the sequence,
    Then it registers a Bearish Fair Value Gap with the gap boundaries
    defined by [Candle 3 High, Candle 1 Low].
    """

    def test_bearish_fvg_detected_when_candle3_high_below_candle1_low(
        self, detector: FVGDetector, bearish_candle_sequence: List[Candle]
    ):
        c1, c2, c3 = bearish_candle_sequence
        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        assert isinstance(gap, FairValueGap)
        assert gap.gap_type == FVGType.BEARISH
        assert gap.lower_boundary == pytest.approx(c3.high)
        assert gap.upper_boundary == pytest.approx(c1.low)
        assert list(gap.boundaries) == [pytest.approx(c3.high), pytest.approx(c1.low)]
        assert gap.bar_index == c3.index

    @pytest.mark.parametrize(
        "c1_low, c3_high",
        [
            (10.05, 10.0),
            (1500.51, 1500.50),
            (0.00015, 0.00012),
            (100000.0, 99999.0),
        ],
    )
    def test_bearish_fvg_with_various_price_scales(
        self, detector: FVGDetector, c1_low: float, c3_high: float
    ):
        c1 = Candle(open=c1_low + 2.0, high=c1_low + 5.0, low=c1_low, close=c1_low + 1.0, index=10)
        c2 = Candle(open=c1.close, high=c1.close + 1.0, low=c3_high - 5.0, close=c3_high - 4.0, index=11)
        c3 = Candle(open=c2.close, high=c3_high, low=c3_high - 10.0, close=c3_high - 2.0, index=12)

        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        assert gap.gap_type == FVGType.BEARISH
        assert gap.lower_boundary == pytest.approx(c3_high)
        assert gap.upper_boundary == pytest.approx(c1.low)
        assert list(gap.boundaries) == [pytest.approx(c3_high), pytest.approx(c1.low)]

    def test_bearish_fvg_gap_size_calculation(
        self, detector: FVGDetector, bearish_candle_sequence: List[Candle]
    ):
        c1, c2, c3 = bearish_candle_sequence
        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is not None
        expected_size = c1.low - c3.high
        if hasattr(gap, "size"):
            assert gap.size == pytest.approx(expected_size)


# =====================================================================
# Test Acceptance Criterion 3: Overlapping Ranges (No Gap)
# =====================================================================


class TestOverlappingRangesWithoutDisplacement:
    """
    Acceptance Criterion:
    Given a series of 3 consecutive candles where price ranges overlap
    without a displacement gap, When the FVG detector evaluates the sequence,
    Then no gap entry is returned for that bar index.
    """

    def test_no_gap_returned_when_ranges_overlap(
        self, detector: FVGDetector, overlapping_candle_sequence: List[Candle]
    ):
        c1, c2, c3 = overlapping_candle_sequence
        gap = detector.evaluate_sequence(c1, c2, c3)

        assert gap is None

    def test_no_gap_when_candle3_low_equals_candle1_high_touching_boundary(
        self, detector: FVGDetector
    ):
        """Exact touch where Candle 3 Low == Candle 1 High does not form an open gap."""
        c1 = Candle(open=95.0, high=100.0, low=90.0, close=98.0, index=0)
        c2 = Candle(open=98.0, high=110.0, low=97.0, close=108.0, index=1)
        c3 = Candle(open=108.0, high=115.0, low=100.0, close=112.0, index=2)

        gap = detector.evaluate_sequence(c1, c2, c3)
        assert gap is None

    def test_no_gap_when_candle3_high_equals_candle1_low_touching_boundary(
        self, detector: FVGDetector
    ):
        """Exact touch where Candle 3 High == Candle 1 Low does not form an open gap."""
        c1 = Candle(open=105.0, high=110.0, low=100.0, close=102.0, index=0)
        c2 = Candle(open=102.0, high=103.0, low=90.0, close=92.0, index=1)
        c3 = Candle(open=92.0, high=100.0, low=85.0, close=88.0, index=2)

        gap = detector.evaluate_sequence(c1, c2, c3)
        assert gap is None

    def test_no_gap_when_candle3_is_fully_engulfed_by_candle1(
        self, detector: FVGDetector
    ):
        """Candle 3 is completely within the range of Candle 1."""
        c1 = Candle(open=90.0, high=120.0, low=80.0, close=110.0, index=0)
        c2 = Candle(open=110.0, high=115.0, low=95.0, close=100.0, index=1)
        c3 = Candle(open=100.0, high=105.0, low=95.0, close=102.0, index=2)

        gap = detector.evaluate_sequence(c1, c2, c3)
        assert gap is None

    def test_no_gap_when_candle1_is_fully_engulfed_by_candle3(
        self, detector: FVGDetector
    ):
        """Candle 3 engulfs the entire range of Candle 1."""
        c1 = Candle(open=98.0, high=102.0, low=97.0, close=100.0, index=0)
        c2 = Candle(open=100.0, high=105.0, low=95.0, close=102.0, index=1)
        c3 = Candle(open=102.0, high=120.0, low=90.0, close=110.0, index=2)

        gap = detector.evaluate_sequence(c1, c2, c3)
        assert gap is None


# =====================================================================
# Test Multi-bar Detection and DataFrame Support
# =====================================================================


class TestMultiBarAndDataFrameDetection:
    """Verifies scanning over series of candles and DataFrames."""

    def test_detect_from_candle_list_returns_only_valid_gaps(
        self,
        detector: FVGDetector,
        bullish_candle_sequence: List[Candle],
        bearish_candle_sequence: List[Candle],
        overlapping_candle_sequence: List[Candle],
    ):
        """Ensure full series evaluation produces correct gaps with matching bar indices."""
        # Shift bar indices to form a contiguous sequence of 9 candles
        candles = []
        for i, c in enumerate(
            bullish_candle_sequence
            + overlapping_candle_sequence
            + bearish_candle_sequence
        ):
            candles.append(
                Candle(
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    index=i,
                )
            )

        gaps = detector.detect(candles)

        # Candle 2 should trigger Bullish FVG
        # Candle 8 should trigger Bearish FVG
        bullish_gaps = [g for g in gaps if g.gap_type == FVGType.BULLISH]
        bearish_gaps = [g for g in gaps if g.gap_type == FVGType.BEARISH]

        assert len(bullish_gaps) >= 1
        assert bullish_gaps[0].bar_index == 2
        assert bullish_gaps[0].lower_boundary == pytest.approx(candles[0].high)
        assert bullish_gaps[0].upper_boundary == pytest.approx(candles[2].low)

        assert len(bearish_gaps) >= 1
        assert any(g.bar_index == 8 for g in bearish_gaps)

    def test_detect_from_dataframe(self, detector: FVGDetector):
        """Detector should process standard OHLC pandas DataFrames correctly."""
        df = pd.DataFrame(
            {
                "open": [92.0, 98.0, 114.0, 118.0, 105.0, 86.0],
                "high": [100.0, 115.0, 120.0, 120.0, 110.0, 95.0],
                "low": [90.0, 97.0, 105.0, 112.0, 85.0, 80.0],
                "close": [98.0, 114.0, 118.0, 115.0, 86.0, 82.0],
            }
        )

        gaps = detector.detect(df)

        assert isinstance(gaps, list)
        assert len(gaps) >= 2

        # Index 2: Bullish FVG (c3 low 105.0 > c1 high 100.0)
        gap_2 = next((g for g in gaps if g.bar_index == 2), None)
        assert gap_2 is not None
        assert gap_2.gap_type == FVGType.BULLISH
        assert gap_2.lower_boundary == pytest.approx(100.0)
        assert gap_2.upper_boundary == pytest.approx(105.0)

        # Index 5: Bearish FVG (c3 high 95.0 < c1 low 112.0)
        gap_5 = next((g for g in gaps if g.bar_index == 5), None)
        assert gap_5 is not None
        assert gap_5.gap_type == FVGType.BEARISH
        assert gap_5.lower_boundary == pytest.approx(95.0)
        assert gap_5.upper_boundary == pytest.approx(112.0)

    def test_detect_returns_empty_when_fewer_than_3_candles(
        self, detector: FVGDetector
    ):
        """At least 3 candles are required to form an FVG."""
        assert detector.detect([]) == []

        one_candle = [Candle(open=10, high=12, low=9, close=11, index=0)]
        assert detector.detect(one_candle) == []

        two_candles = [
            Candle(open=10, high=12, low=9, close=11, index=0),
            Candle(open=11, high=15, low=10, close=14, index=1),
        ]
        assert detector.detect(two_candles) == []

        df_two_rows = pd.DataFrame(
            {"open": [10, 11], "high": [12, 15], "low": [9, 10], "close": [11, 14]}
        )
        assert detector.detect(df_two_rows) == []


# =====================================================================
# Test Validation and Defensive Edge Cases
# =====================================================================


class TestValidationAndEdgeCases:
    """Verifies detector robustness against invalid candle geometry and data types."""

    def test_invalid_candle_high_less_than_low_raises_error(
        self, detector: FVGDetector
    ):
        """Candle where High < Low violates basic OHLC constraints."""
        with pytest.raises(ValueError):
            Candle(open=100.0, high=90.0, low=110.0, close=95.0, index=0)

    def test_dataframe_missing_required_columns_raises_error(
        self, detector: FVGDetector
    ):
        """DataFrame missing 'high' or 'low' cannot be evaluated."""
        df_invalid = pd.DataFrame({"open": [10.0, 11.0, 12.0], "close": [11.0, 12.0, 13.0]})
        with pytest.raises((ValueError, KeyError)):
            detector.detect(df_invalid)

    def test_dataframe_with_nan_in_ohlc_raises_error(
        self, detector: FVGDetector
    ):
        """NaN values within price series must be rejected."""
        df_nan = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [12.0, float("nan"), 14.0],
                "low": [9.0, 10.0, 11.0],
                "close": [11.0, 12.0, 13.0],
            }
        )
        with pytest.raises(ValueError):
            detector.detect(df_nan)

    def test_dataframe_with_non_numeric_prices_raises_error(
        self, detector: FVGDetector
    ):
        """Non-numeric string values must raise an error."""
        df_str = pd.DataFrame(
            {
                "open": ["10", "11", "12"],
                "high": ["12", "invalid", "14"],
                "low": ["9", "10", "11"],
                "close": ["11", "12", "13"],
            }
        )
        with pytest.raises((ValueError, TypeError)):
            detector.detect(df_str)

    def test_evaluate_sequence_with_none_candles_raises_error(
        self, detector: FVGDetector, bullish_candle_sequence: List[Candle]
    ):
        """Evaluating with None values must raise an error."""
        c1, c2, _ = bullish_candle_sequence
        with pytest.raises((ValueError, TypeError)):
            detector.evaluate_sequence(c1, c2, None)  # type: ignore

    def test_candle_indices_are_preserved_in_gap_output(
        self, detector: FVGDetector
    ):
        """The bar_index on FairValueGap must reflect candle 3's index accurately."""
        c1 = Candle(open=100.0, high=105.0, low=98.0, close=104.0, index=42)
        c2 = Candle(open=104.0, high=120.0, low=103.0, close=119.0, index=43)
        c3 = Candle(open=119.0, high=125.0, low=110.0, close=122.0, index=44)

        gap = detector.evaluate_sequence(c1, c2, c3)
        assert gap is not None
        assert gap.bar_index == 44