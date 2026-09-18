"""
Unit tests for Institutional Volume Profiler and Liquidity Sweeps.
Covers Story 4.3.3 Acceptance Criteria:
1. Volume Profile: POC, VAH, VAL covering 70% of total volume for a given session.
2. Liquidity Sweeps: Piercing of key levels with institutional volume (e.g. >= 2x avg)
   closing back inside the range, triggering sweep events with level, direction, and volume.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.analysis.volume_profile import (
    VolumeProfiler,
    VolumeProfileResult,
    compute_volume_profile,
)
from src.analysis.liquidity_sweep import (
    LiquidityLevel,
    LiquiditySweepDetector,
    LiquiditySweepEvent,
    SweepDirection,
    detect_liquidity_sweeps,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def discrete_price_volume_df() -> pd.DataFrame:
    """
    Creates a deterministic discrete price and volume distribution.
    Total Volume = 100
    Price:   100.0, 101.0, 102.0, 103.0, 104.0
    Volume:   10.0,  20.0,  40.0,  20.0,  10.0
    Expected POC: 102.0 (vol 40)
    70% Value Area Volume target = 70.0
    Cumulative with 101.0 (20) & 103.0 (20) = 80.0 >= 70.0
    Expected VAL: 101.0, VAH: 103.0
    """
    return pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10.0, 20.0, 40.0, 20.0, 10.0],
        }
    )


@pytest.fixture
def session_ohlcv_df() -> pd.DataFrame:
    """
    Generates time-indexed OHLCV session data.
    20 baseline candles followed by test candles.
    """
    base_time = datetime(2023, 10, 25, 9, 30)
    rows = []

    # 20 background baseline candles with uniform volume = 1000.0
    for i in range(20):
        rows.append(
            {
                "timestamp": base_time + timedelta(minutes=i),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Module: Volume Profile Tests (src/analysis/volume_profile.py)
# ============================================================================


class TestVolumeProfiler:
    """Tests for Volume Profile calculation (POC, VAH, VAL)."""

    def test_compute_volume_profile_poc_vah_val_exact(
        self, discrete_price_volume_df: pd.DataFrame
    ) -> None:
        """
        Acceptance Criteria: Return POC, VAH, and VAL covering 70% of total volume.
        """
        result = compute_volume_profile(
            df=discrete_price_volume_df,
            price_col="price",
            volume_col="volume",
            value_area_pct=0.70,
        )

        assert isinstance(result, VolumeProfileResult)
        assert result.total_volume == pytest.approx(100.0, rel=1e-5)
        assert result.poc == pytest.approx(102.0, abs=1e-4)
        assert result.val == pytest.approx(101.0, abs=1e-4)
        assert result.vah == pytest.approx(103.0, abs=1e-4)

        # Verify value area volume is at least 70% of total volume
        assert result.value_area_volume >= 0.70 * result.total_volume

    def test_compute_volume_profile_asymmetric_distribution(self) -> None:
        """
        Tests skewed volume distribution where value area expands towards
        the side with higher volume.
        """
        df = pd.DataFrame(
            {
                "price": [50.0, 51.0, 52.0, 53.0, 54.0],
                "volume": [5.0, 10.0, 50.0, 30.0, 5.0],
            }
        )
        # Total vol: 100, 70% = 70
        # POC: 52.0 (vol 50)
        # Compare 53.0 (vol 30) vs 51.0 (vol 10): 53.0 is higher
        # 50 + 30 = 80 >= 70, so VAH=53.0, VAL=52.0
        result = compute_volume_profile(
            df=df, price_col="price", volume_col="volume", value_area_pct=0.70
        )

        assert result.poc == pytest.approx(52.0, abs=1e-4)
        assert result.vah == pytest.approx(53.0, abs=1e-4)
        assert result.val == pytest.approx(52.0, abs=1e-4)
        assert result.value_area_volume == pytest.approx(80.0, rel=1e-5)

    def test_compute_volume_profile_session_filter(self) -> None:
        """
        Ensures volume profile calculation strictly respects session range parameters.
        """
        t1 = datetime(2023, 10, 25, 9, 30)
        t2 = datetime(2023, 10, 25, 10, 0)
        t3 = datetime(2023, 10, 25, 10, 30)

        df = pd.DataFrame(
            {
                "timestamp": [t1, t2, t3],
                "price": [10.0, 20.0, 30.0],
                "volume": [500.0, 100.0, 500.0],
            }
        )

        # Restrict session to [t2, t2]
        profiler = VolumeProfiler(
            session_start=t2,
            session_end=t2,
            price_col="price",
            volume_col="volume",
            timestamp_col="timestamp",
        )
        result = profiler.compute(df)

        assert result.total_volume == pytest.approx(100.0, rel=1e-5)
        assert result.poc == pytest.approx(20.0, abs=1e-4)
        assert result.val == pytest.approx(20.0, abs=1e-4)
        assert result.vah == pytest.approx(20.0, abs=1e-4)

    def test_volume_profile_single_price_level(self) -> None:
        """Edge case: Single price level should result in POC == VAH == VAL."""
        df = pd.DataFrame({"price": [150.0], "volume": [1000.0]})
        result = compute_volume_profile(df, price_col="price", volume_col="volume")

        assert result.poc == pytest.approx(150.0)
        assert result.vah == pytest.approx(150.0)
        assert result.val == pytest.approx(150.0)
        assert result.total_volume == pytest.approx(1000.0)

    def test_volume_profile_custom_value_area_pct(
        self, discrete_price_volume_df: pd.DataFrame
    ) -> None:
        """Test with different value area percentages (e.g. 50% and 90%)."""
        res_50 = compute_volume_profile(
            discrete_price_volume_df,
            price_col="price",
            volume_col="volume",
            value_area_pct=0.40,
        )
        # POC alone covers 40% (40/100)
        assert res_50.poc == 102.0
        assert res_50.val == 102.0
        assert res_50.vah == 102.0

        res_90 = compute_volume_profile(
            discrete_price_volume_df,
            price_col="price",
            volume_col="volume",
            value_area_pct=0.90,
        )
        assert res_90.val <= 101.0
        assert res_90.vah >= 103.0
        assert res_90.value_area_volume >= 90.0

    def test_volume_profile_empty_dataframe_raises(self) -> None:
        """Empty input data must raise a ValueError."""
        df_empty = pd.DataFrame(columns=["price", "volume"])
        with pytest.raises(ValueError):
            compute_volume_profile(df_empty, price_col="price", volume_col="volume")

    def test_volume_profile_invalid_value_area_pct_raises(
        self, discrete_price_volume_df: pd.DataFrame
    ) -> None:
        """Percentage outside (0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError):
            compute_volume_profile(discrete_price_volume_df, value_area_pct=0.0)
        with pytest.raises(ValueError):
            compute_volume_profile(discrete_price_volume_df, value_area_pct=1.5)

    def test_volume_profile_negative_or_zero_total_volume_raises(self) -> None:
        """Zero or negative total volume data must raise ValueError."""
        df_zero = pd.DataFrame({"price": [10.0, 20.0], "volume": [0.0, 0.0]})
        with pytest.raises(ValueError):
            compute_volume_profile(df_zero, price_col="price", volume_col="volume")

        df_neg = pd.DataFrame({"price": [10.0, 20.0], "volume": [-10.0, 5.0]})
        with pytest.raises(ValueError):
            compute_volume_profile(df_neg, price_col="price", volume_col="volume")


# ============================================================================
# Module: Liquidity Sweep Tests (src/analysis/liquidity_sweep.py)
# ============================================================================


class TestLiquiditySweepDetector:
    """Tests for institutional liquidity sweep detection."""

    def test_bearish_sweep_of_swing_high(self, session_ohlcv_df: pd.DataFrame) -> None:
        """
        Acceptance Criteria:
        When a candle pierces the level (Swing High) but closes back inside the range
        with volume exceeding institutional threshold (2x average volume),
        trigger a liquidity sweep event (bearish).
        """
        key_level = LiquidityLevel(price=105.0, level_type="SWING_HIGH")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        # Candle pierces 105.0 (High=106.5) but closes inside at 104.0 (< 105.0)
        # Baseline average volume is 1000.0. Sweep volume is 2500.0 (2.5x > 2.0x threshold)
        sweep_candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 103.0,
                    "high": 106.5,
                    "low": 102.5,
                    "close": 104.0,
                    "volume": 2500.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, sweep_candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[key_level],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, LiquiditySweepEvent)
        assert event.swept_level == pytest.approx(105.0)
        assert event.direction in (SweepDirection.BEARISH, "bearish")
        assert event.sweep_volume == pytest.approx(2500.0)
        assert event.pierce_price == pytest.approx(106.5)
        assert event.close_price == pytest.approx(104.0)

    def test_bullish_sweep_of_val(self, session_ohlcv_df: pd.DataFrame) -> None:
        """
        Acceptance Criteria:
        When a candle pierces VAL from above, goes lower, and closes back above VAL
        with volume exceeding 2x average volume, trigger a bullish sweep event.
        """
        val_level = LiquidityLevel(price=95.0, level_type="VAL")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        # Candle dips below 95.0 (Low=93.0) and closes at 96.0 (> 95.0)
        # Volume = 2100.0 (> 2x baseline avg of 1000.0)
        sweep_candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 97.0,
                    "high": 97.5,
                    "low": 93.0,
                    "close": 96.0,
                    "volume": 2100.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, sweep_candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[val_level],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 1
        event = events[0]
        assert event.swept_level == pytest.approx(95.0)
        assert event.direction in (SweepDirection.BULLISH, "bullish")
        assert event.sweep_volume == pytest.approx(2100.0)
        assert event.pierce_price == pytest.approx(93.0)
        assert event.close_price == pytest.approx(96.0)

    def test_no_sweep_when_volume_below_threshold(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """
        A candle pierces the level and closes back inside, but volume is insufficient
        (< 2.0x avg volume). No sweep event should be triggered.
        """
        swing_high = LiquidityLevel(price=105.0, level_type="SWING_HIGH")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        # Volume is 1500.0 (only 1.5x average volume of 1000.0)
        candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 103.0,
                    "high": 107.0,
                    "low": 103.0,
                    "close": 104.0,
                    "volume": 1500.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[swing_high],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 0

    def test_no_sweep_on_genuine_breakout(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """
        Candle pierces level and CLOSES OUTSIDE (breakout / acceptance, not a sweep).
        No sweep event should be triggered.
        """
        swing_high = LiquidityLevel(price=105.0, level_type="SWING_HIGH")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        # Close is 108.0 (> 105.0), volume is high (3000.0)
        breakout_candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 104.0,
                    "high": 109.0,
                    "low": 103.5,
                    "close": 108.0,
                    "volume": 3000.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, breakout_candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[swing_high],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 0

    def test_no_sweep_without_piercing_level(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """Candle volume is extreme, but High does not pierce the level."""
        swing_high = LiquidityLevel(price=105.0, level_type="SWING_HIGH")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 103.0,
                    "high": 104.99,  # Just below 105.0
                    "low": 102.0,
                    "close": 103.5,
                    "volume": 5000.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[swing_high],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 0

    def test_multiple_levels_swept_simultaneously(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """
        A large institutional candle pierces two key levels (e.g. VAH and a Swing High)
        and closes back below both. Should record events for both swept levels.
        """
        level_vah = LiquidityLevel(price=104.0, level_type="VAH")
        level_swing = LiquidityLevel(price=105.0, level_type="SWING_HIGH")
        last_time = session_ohlcv_df["timestamp"].iloc[-1]

        candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 103.0,
                    "high": 106.0,  # Pierces both 104 and 105
                    "low": 102.0,
                    "close": 103.5,  # Closes below both
                    "volume": 3500.0,
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, candle], ignore_index=True)

        events = detect_liquidity_sweeps(
            candles=candles,
            key_levels=[level_vah, level_swing],
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 2
        swept_prices = {e.swept_level for e in events}
        assert swept_prices == {104.0, 105.0}
        for e in events:
            assert e.direction in (SweepDirection.BEARISH, "bearish")
            assert e.sweep_volume == pytest.approx(3500.0)

    def test_detector_class_interface(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test stateful / class-based interface of LiquiditySweepDetector."""
        detector = LiquiditySweepDetector(
            volume_threshold_multiplier=2.5,
            volume_ma_period=20,
        )
        key_level = LiquidityLevel(price=110.0, level_type="SWING_HIGH")

        last_time = session_ohlcv_df["timestamp"].iloc[-1]
        candle = pd.DataFrame(
            [
                {
                    "timestamp": last_time + timedelta(minutes=1),
                    "open": 108.0,
                    "high": 112.0,
                    "low": 107.0,
                    "close": 109.0,
                    "volume": 2600.0,  # 2.6x > 2.5x threshold
                }
            ]
        )
        candles = pd.concat([session_ohlcv_df, candle], ignore_index=True)

        events = detector.detect(candles=candles, key_levels=[key_level])
        assert len(events) == 1
        assert events[0].swept_level == pytest.approx(110.0)

    def test_liquidity_sweep_empty_candles_raises(self) -> None:
        """Empty candle dataframe must raise ValueError."""
        with pytest.raises(ValueError):
            detect_liquidity_sweeps(
                candles=pd.DataFrame(),
                key_levels=[LiquidityLevel(price=100.0, level_type="SWING_HIGH")],
            )

    def test_liquidity_sweep_invalid_threshold_raises(
        self, session_ohlcv_df: pd.DataFrame
    ) -> None:
        """Volume multiplier <= 0 or invalid must raise ValueError."""
        level = LiquidityLevel(price=100.0, level_type="SWING_HIGH")
        with pytest.raises(ValueError):
            detect_liquidity_sweeps(
                candles=session_ohlcv_df,
                key_levels=[level],
                volume_threshold_multiplier=0.0,
            )
        with pytest.raises(ValueError):
            detect_liquidity_sweeps(
                candles=session_ohlcv_df,
                key_levels=[level],
                volume_threshold_multiplier=-1.5,
            )


# ============================================================================
# Integration: Volume Profile -> Liquidity Sweeps
# ============================================================================


class TestVolumeProfileToSweepIntegration:
    """End-to-end integration: extract VAH/VAL from volume profile and detect sweeps."""

    def test_volume_profile_levels_piped_to_sweep_detector(self) -> None:
        """
        Verify that VAH and VAL derived from compute_volume_profile can be directly
        used as LiquidityLevels to detect sweeps in subsequent session candles.
        """
        # Session 1: Baseline trading establishing POC, VAH, VAL
        s1_data = pd.DataFrame(
            {
                "price": [100.0, 105.0, 110.0, 115.0, 120.0],
                "volume": [100.0, 300.0, 500.0, 300.0, 100.0],
            }
        )
        vp_result = compute_volume_profile(
            s1_data, price_col="price", volume_col="volume", value_area_pct=0.70
        )

        assert vp_result.poc == 110.0
        assert vp_result.val == 105.0
        assert vp_result.vah == 115.0

        levels = [
            LiquidityLevel(price=vp_result.vah, level_type="VAH"),
            LiquidityLevel(price=vp_result.val, level_type="VAL"),
        ]

        # Session 2: Candles piercing Session 1 VAH with high institutional volume
        base_time = datetime(2023, 10, 26, 9, 30)
        s2_candles = []
        for i in range(20):
            s2_candles.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "open": 112.0,
                    "high": 114.0,
                    "low": 110.0,
                    "close": 112.0,
                    "volume": 1000.0,
                }
            )

        # Sweep candle piercing VAH (115.0) up to 117.0, closing at 114.5 (< 115.0)
        sweep_time = base_time + timedelta(minutes=20)
        s2_candles.append(
            {
                "timestamp": sweep_time,
                "open": 113.0,
                "high": 117.0,
                "low": 112.5,
                "close": 114.5,
                "volume": 3200.0,  # 3.2x baseline
            }
        )

        s2_df = pd.DataFrame(s2_candles)
        events = detect_liquidity_sweeps(
            candles=s2_df,
            key_levels=levels,
            volume_threshold_multiplier=2.0,
            volume_ma_period=20,
        )

        assert len(events) == 1
        sweep = events[0]
        assert sweep.swept_level == pytest.approx(vp_result.vah)
        assert sweep.direction in (SweepDirection.BEARISH, "bearish")
        assert sweep.sweep_volume == pytest.approx(3200.0)
        assert sweep.pierce_price == pytest.approx(117.0)
        assert sweep.close_price == pytest.approx(114.5)