"""
Unit tests for Heikin-Ashi, Renko, and Kagi transform renderers.

Specification:
Feature: Implement Heikin-Ashi, Renko, and Kagi Transform Renderers
Story 1.3.2: Implement Heikin-Ashi, Renko, and Kagi Transform Renderers
Target Modules: src/renderers/transforms.py, src/renderers/__init__.py
"""

from typing import List
import pytest

from src.renderers.transforms import (
    OHLCBar,
    HeikinAshiBar,
    HeikinAshiRenderer,
    RenkoBrick,
    RenkoDirection,
    RenkoRenderer,
    KagiSegment,
    KagiState,
    KagiRenderer,
    transform_heikin_ashi,
    transform_renko,
    transform_kagi,
)
import src.renderers as renderers


# ============================================================================
# Package Export & Integration Tests
# ============================================================================


class TestRendererPackageExports:
    """Verify that src.renderers correctly exposes transform renderers and data models."""

    def test_renderers_module_exports_all_required_symbols(self):
        expected_symbols = [
            "OHLCBar",
            "HeikinAshiBar",
            "HeikinAshiRenderer",
            "RenkoBrick",
            "RenkoDirection",
            "RenkoRenderer",
            "KagiSegment",
            "KagiState",
            "KagiRenderer",
            "transform_heikin_ashi",
            "transform_renko",
            "transform_kagi",
        ]
        for symbol in expected_symbols:
            assert hasattr(renderers, symbol), f"src.renderers must export {symbol}"


# ============================================================================
# Heikin-Ashi Transform Renderer Tests
# ============================================================================


class TestHeikinAshiRenderer:
    """
    Acceptance Criteria:
    - Given a time series of standard OHLC bars,
      When the Heikin-Ashi transform renderer is executed,
      Then it returns transformed bars where:
        `ha_open = (prev_ha_open + prev_ha_close) / 2`
        `ha_close = (open + high + low + close) / 4`
    """

    def test_empty_bars_returns_empty_list(self):
        renderer = HeikinAshiRenderer()
        assert renderer.render([]) == []
        assert transform_heikin_ashi([]) == []

    def test_initial_bar_calculation(self):
        first_bar = OHLCBar(open=100.0, high=110.0, low=90.0, close=100.0)
        renderer = HeikinAshiRenderer()
        result = renderer.render([first_bar])

        assert len(result) == 1
        transformed = result[0]
        assert isinstance(transformed, HeikinAshiBar)

        # First bar ha_open is (open + close) / 2
        expected_ha_open = (100.0 + 100.0) / 2.0
        # ha_close = (open + high + low + close) / 4
        expected_ha_close = (100.0 + 110.0 + 90.0 + 100.0) / 4.0
        expected_ha_high = max(110.0, expected_ha_open, expected_ha_close)
        expected_ha_low = min(90.0, expected_ha_open, expected_ha_close)

        assert transformed.ha_open == pytest.approx(expected_ha_open)
        assert transformed.ha_close == pytest.approx(expected_ha_close)
        assert transformed.ha_high == pytest.approx(expected_ha_high)
        assert transformed.ha_low == pytest.approx(expected_ha_low)

    def test_recursive_formula_across_multiple_bars(self):
        bars = [
            OHLCBar(open=100.0, high=110.0, low=90.0, close=100.0),
            OHLCBar(open=105.0, high=115.0, low=95.0, close=110.0),
            OHLCBar(open=108.0, high=120.0, low=102.0, close=118.0),
        ]
        result = transform_heikin_ashi(bars)

        assert len(result) == 3

        # Bar 0
        ha_open_0 = (100.0 + 100.0) / 2.0  # 100.0
        ha_close_0 = (100.0 + 110.0 + 90.0 + 100.0) / 4.0  # 100.0
        assert result[0].ha_open == pytest.approx(ha_open_0)
        assert result[0].ha_close == pytest.approx(ha_close_0)

        # Bar 1: ha_open = (prev_ha_open + prev_ha_close) / 2
        ha_open_1 = (ha_open_0 + ha_close_0) / 2.0  # 100.0
        ha_close_1 = (105.0 + 115.0 + 95.0 + 110.0) / 4.0  # 106.25
        ha_high_1 = max(115.0, ha_open_1, ha_close_1)  # 115.0
        ha_low_1 = min(95.0, ha_open_1, ha_close_1)  # 95.0

        assert result[1].ha_open == pytest.approx(ha_open_1)
        assert result[1].ha_close == pytest.approx(ha_close_1)
        assert result[1].ha_high == pytest.approx(ha_high_1)
        assert result[1].ha_low == pytest.approx(ha_low_1)

        # Bar 2: ha_open = (prev_ha_open + prev_ha_close) / 2
        ha_open_2 = (ha_open_1 + ha_close_1) / 2.0  # 103.125
        ha_close_2 = (108.0 + 120.0 + 102.0 + 118.0) / 4.0  # 112.0
        ha_high_2 = max(120.0, ha_open_2, ha_close_2)  # 120.0
        ha_low_2 = min(102.0, ha_open_2, ha_close_2)  # 102.0

        assert result[2].ha_open == pytest.approx(ha_open_2)
        assert result[2].ha_close == pytest.approx(ha_close_2)
        assert result[2].ha_high == pytest.approx(ha_high_2)
        assert result[2].ha_low == pytest.approx(ha_low_2)

    def test_high_low_envelope_integrity(self):
        # Scenario where ha_open or ha_close exceeds standard high or low
        bar_0 = OHLCBar(open=10.0, high=10.5, low=9.5, close=10.0)
        # Gap up open
        bar_1 = OHLCBar(open=50.0, high=50.5, low=49.5, close=50.0)

        renderer = HeikinAshiRenderer()
        result = renderer.render([bar_0, bar_1])

        ha_1 = result[1]
        # ha_open_1 = (10.0 + 10.0) / 2 = 10.0
        # ha_close_1 = (50.0 + 50.5 + 49.5 + 50.0) / 4 = 50.0
        # ha_high must be max(50.5, 10.0, 50.0) = 50.5
        # ha_low must be min(49.5, 10.0, 50.0) = 10.0 (enveloped by ha_open)
        assert ha_1.ha_low == pytest.approx(10.0)
        assert ha_1.ha_high == pytest.approx(50.5)

    def test_invalid_bar_data_raises_value_error(self):
        # High strictly less than low is physically invalid for OHLC
        invalid_bar = OHLCBar(open=10.0, high=8.0, low=12.0, close=9.0)
        renderer = HeikinAshiRenderer()
        with pytest.raises(ValueError):
            renderer.render([invalid_bar])

    def test_negative_price_values_raise_value_error(self):
        negative_bar = OHLCBar(open=-1.0, high=10.0, low=-5.0, close=5.0)
        with pytest.raises(ValueError):
            transform_heikin_ashi([negative_bar])


# ============================================================================
# Renko Transform Renderer Tests
# ============================================================================


class TestRenkoRenderer:
    """
    Acceptance Criteria:
    - Given a series of close prices and a fixed brick size parameter,
      When the Renko transform renderer is executed,
      Then it produces discrete brick entities filtered by price movements
      that equal or exceed the brick threshold.
    """

    def test_invalid_brick_size_raises_value_error(self):
        with pytest.raises(ValueError):
            RenkoRenderer(brick_size=0.0)

        with pytest.raises(ValueError):
            RenkoRenderer(brick_size=-5.0)

        with pytest.raises(ValueError):
            transform_renko(prices=[100.0, 105.0], brick_size=0.0)

    def test_empty_or_single_price_produces_no_bricks(self):
        renderer = RenkoRenderer(brick_size=10.0)
        assert renderer.render([]) == []
        assert renderer.render([100.0]) == []

    def test_movements_below_threshold_are_filtered_out(self):
        brick_size = 10.0
        # Price starts at 100, moves to 109.9 (< 10), then 101.0, then 108.0
        prices = [100.0, 105.0, 109.9, 101.0, 108.0]
        result = transform_renko(prices=prices, brick_size=brick_size)
        assert result == []

    def test_exact_brick_threshold_movement_produces_up_brick(self):
        brick_size = 10.0
        prices = [100.0, 110.0]
        renderer = RenkoRenderer(brick_size=brick_size)
        result = renderer.render(prices)

        assert len(result) == 1
        brick = result[0]
        assert isinstance(brick, RenkoBrick)
        assert brick.open_price == pytest.approx(100.0)
        assert brick.close_price == pytest.approx(110.0)
        assert brick.direction == RenkoDirection.UP

    def test_multiple_bricks_generated_from_large_movement(self):
        brick_size = 5.0
        # Price moves from 100 to 117 -> should produce 3 UP bricks (100-105, 105-110, 110-115)
        # Remainder 2 is retained / ignored until next threshold
        prices = [100.0, 117.0]
        result = transform_renko(prices=prices, brick_size=brick_size)

        assert len(result) == 3
        assert result[0].open_price == pytest.approx(100.0)
        assert result[0].close_price == pytest.approx(105.0)
        assert result[0].direction == RenkoDirection.UP

        assert result[1].open_price == pytest.approx(105.0)
        assert result[1].close_price == pytest.approx(110.0)
        assert result[1].direction == RenkoDirection.UP

        assert result[2].open_price == pytest.approx(110.0)
        assert result[2].close_price == pytest.approx(115.0)
        assert result[2].direction == RenkoDirection.UP

    def test_downward_reversal_requires_exceeding_prior_brick_bottom(self):
        brick_size = 10.0
        # First: UP brick 100 -> 110
        # Reversal requires dropping below bottom of previous brick (100) by at least brick_size (<= 90)
        prices = [100.0, 110.0, 105.0, 95.0, 90.0]
        renderer = RenkoRenderer(brick_size=brick_size)
        result = renderer.render(prices)

        assert len(result) == 2
        # First brick: UP (100 -> 110)
        assert result[0].direction == RenkoDirection.UP
        assert result[0].open_price == pytest.approx(100.0)
        assert result[0].close_price == pytest.approx(110.0)

        # Second brick: Reversal DOWN (100 -> 90)
        assert result[1].direction == RenkoDirection.DOWN
        assert result[1].open_price == pytest.approx(100.0)
        assert result[1].close_price == pytest.approx(90.0)

    def test_upward_reversal_requires_exceeding_prior_brick_top(self):
        brick_size = 10.0
        # Start at 100, drop to 90 (DOWN brick: 100 -> 90).
        # Prior brick top was 100. Reversal requires price >= 100 + 10 = 110.
        prices = [100.0, 90.0, 105.0, 110.0]
        result = transform_renko(prices=prices, brick_size=brick_size)

        assert len(result) == 2
        # DOWN brick
        assert result[0].direction == RenkoDirection.DOWN
        assert result[0].open_price == pytest.approx(100.0)
        assert result[0].close_price == pytest.approx(90.0)

        # Reversal UP brick
        assert result[1].direction == RenkoDirection.UP
        assert result[1].open_price == pytest.approx(100.0)
        assert result[1].close_price == pytest.approx(110.0)

    def test_renko_brick_continuity(self):
        brick_size = 2.0
        prices = [10.0, 16.0, 12.0, 8.0]
        renderer = RenkoRenderer(brick_size=brick_size)
        result = renderer.render(prices)

        # 10 -> 16 gives 3 UP bricks: [10,12], [12,14], [14,16]
        # From [14,16], drop to 12 is at brick bottom (14), reversal requires <= 12 (1 DOWN brick: [14,12])
        # Drop to 8 continues DOWN: [12,10], [10,8]
        assert len(result) == 6
        for brick in result:
            assert abs(brick.close_price - brick.open_price) == pytest.approx(brick_size)


# ============================================================================
# Kagi Transform Renderer Tests
# ============================================================================


class TestKagiRenderer:
    """
    Acceptance Criteria:
    - Given a series of close prices and a reversal percentage threshold,
      When the Kagi transform renderer is executed,
      Then it produces trend-reversal line segments switching state between
      yang (bullish) and yin (bearish) upon surpassing prior breakout levels.
    """

    def test_invalid_reversal_percentage_raises_value_error(self):
        with pytest.raises(ValueError):
            KagiRenderer(reversal_pct=0.0)

        with pytest.raises(ValueError):
            KagiRenderer(reversal_pct=-0.05)

        with pytest.raises(ValueError):
            transform_kagi(prices=[100.0, 110.0], reversal_pct=-0.1)

    def test_empty_or_single_price_produces_no_segments(self):
        renderer = KagiRenderer(reversal_pct=0.04)
        assert renderer.render([]) == []
        assert renderer.render([100.0]) == []

    def test_trend_continuation_without_reversal(self):
        # 4% reversal threshold
        reversal_pct = 0.04
        # Prices move generally up, pullbacks are < 4%
        prices = [100.0, 105.0, 103.0, 110.0]
        # Pullback from 105 to 103 is 1.9% < 4%, does not trigger reversal
        segments = transform_kagi(prices=prices, reversal_pct=reversal_pct)

        assert len(segments) >= 1
        # The main trend continued upward to 110
        assert segments[-1].end_price == pytest.approx(110.0)
        assert segments[0].start_price == pytest.approx(100.0)

    def test_reversal_occurs_when_threshold_exceeded(self):
        reversal_pct = 0.05  # 5%
        # 100 -> 120 (UP trend)
        # Pullback: 120 -> 113 is (120 - 113) / 120 = 5.83% >= 5% -> Reversal!
        prices = [100.0, 120.0, 113.0]
        renderer = KagiRenderer(reversal_pct=reversal_pct)
        segments = renderer.render(prices)

        assert len(segments) == 2
        # First segment: UP 100 -> 120
        assert segments[0].start_price == pytest.approx(100.0)
        assert segments[0].end_price == pytest.approx(120.0)

        # Second segment: Reversal DOWN 120 -> 113
        assert segments[1].start_price == pytest.approx(120.0)
        assert segments[1].end_price == pytest.approx(113.0)

    def test_breakout_above_prior_swing_high_switches_state_to_yang(self):
        """
        Kagi Rule: When price rises above the previous swing high (shoulder),
        the line turns thick (YANG / bullish).
        """
        reversal_pct = 0.10  # 10%
        prices = [
            100.0,  # Start
            120.0,  # Peak 1 (Swing High = 120.0)
            105.0,  # Reversal down by 12.5% >= 10% (Swing Low = 105.0), enters YIN
            125.0,  # Reversal up by 19.0% >= 10%, breaks above prior swing high (120.0) -> switches to YANG
        ]
        renderer = KagiRenderer(reversal_pct=reversal_pct)
        segments = renderer.render(prices)

        assert len(segments) >= 2

        # The segment ending at 125.0 broke above prior high (120.0), so final state must be YANG
        final_segment = segments[-1]
        assert final_segment.end_price == pytest.approx(125.0)
        assert final_segment.state == KagiState.YANG

    def test_breakdown_below_prior_swing_low_switches_state_to_yin(self):
        """
        Kagi Rule: When price falls below the previous swing low (waist),
        the line turns thin (YIN / bearish).
        """
        reversal_pct = 0.10  # 10%
        prices = [
            100.0,  # Start
            130.0,  # Peak 1 (State: YANG)
            110.0,  # Trough 1 (Swing Low = 110.0), reversal down
            125.0,  # Peak 2 (Swing High = 125.0), reversal up but doesn't break 130
            95.0,   # Reversal down, falls below Trough 1 (110.0) -> Breakdown to YIN
        ]
        result = transform_kagi(prices=prices, reversal_pct=reversal_pct)

        assert len(result) >= 3

        # Final segment ends at 95.0, having broken below swing low (110.0)
        final_segment = result[-1]
        assert final_segment.end_price == pytest.approx(95.0)
        assert final_segment.state == KagiState.YIN

    def test_kagi_segment_attributes_and_typing(self):
        reversal_pct = 0.05
        prices = [50.0, 60.0, 55.0]
        renderer = KagiRenderer(reversal_pct=reversal_pct)
        segments: List[KagiSegment] = renderer.render(prices)

        for segment in segments:
            assert isinstance(segment, KagiSegment)
            assert hasattr(segment, "start_price")
            assert hasattr(segment, "end_price")
            assert hasattr(segment, "state")
            assert segment.state in (KagiState.YANG, KagiState.YIN)