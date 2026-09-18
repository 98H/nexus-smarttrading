"""
Unit tests for Smart Dynamic Support and Resistance Zones.
Story 4.2.2: Implement Smart Dynamic Support and Resistance Zones
Target modules:
- src/indicators/dynamic_sr.py
- src/indicators/__init__.py
"""

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.indicators import calculate_dynamic_sr_zones as exported_calc_sr
from src.indicators.dynamic_sr import (
    DynamicSRResult,
    SRZone,
    calculate_dynamic_sr_zones,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_zone_boundaries(zone: SRZone) -> Tuple[float, float]:
    """Extract (lower, upper) boundaries supporting both property naming styles."""
    lower = getattr(zone, "lower_boundary", getattr(zone, "lower", None))
    upper = getattr(zone, "upper_boundary", getattr(zone, "upper", None))
    assert lower is not None and upper is not None, "Zone must expose upper and lower boundaries"
    return float(lower), float(upper)


def _build_ohlcv(
    highs: np.ndarray,
    lows: np.ndarray,
    start_date: str = "2023-01-01",
    columns_case: str = "lower",
) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame from high and low arrays."""
    n = len(highs)
    assert len(lows) == n, "Highs and lows must be equal length"

    # Mid-prices for open and close
    closes = (highs + lows) / 2.0
    opens = closes.copy()
    volumes = np.full(n, 1000.0)

    dates = pd.date_range(start=start_date, periods=n, freq="1D")

    if columns_case == "lower":
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    elif columns_case == "upper":
        data = {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    else:
        raise ValueError(f"Unknown columns_case: {columns_case}")

    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_ohlcv_df() -> pd.DataFrame:
    """Returns an empty OHLCV DataFrame."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def short_ohlcv_df() -> pd.DataFrame:
    """Returns an OHLCV DataFrame with 10 periods (too short for window=20)."""
    highs = np.array([102, 103, 101, 104, 102, 105, 103, 106, 104, 105], dtype=float)
    lows = highs - 4.0
    return _build_ohlcv(highs, lows)


@pytest.fixture
def synthetic_swing_ohlcv_df() -> pd.DataFrame:
    """
    Returns 50 candles with deterministic swing highs and lows:
    - Peak 1 at idx 10: High = 150.0
    - Peak 2 at idx 25: High = 150.4 (close to Peak 1)
    - Peak 3 at idx 40: High = 180.0 (distant peak)
    - Trough 1 at idx 15: Low = 100.0
    - Trough 2 at idx 30: Low = 99.6 (close to Trough 1)
    - Trough 3 at idx 45: Low = 70.0 (distant trough)
    """
    n = 50
    highs = np.full(n, 120.0, dtype=float)
    lows = np.full(n, 110.0, dtype=float)

    # Intersperse background noise
    for i in range(n):
        highs[i] += (i % 3) * 0.5
        lows[i] -= (i % 3) * 0.5

    # Known Swing Highs
    highs[10] = 150.0
    highs[25] = 150.4
    highs[40] = 180.0

    # Ensure surrounding bars are lower than peaks (order 3)
    for offset in [-2, -1, 1, 2]:
        highs[10 + offset] = 130.0
        highs[25 + offset] = 130.0
        highs[40 + offset] = 130.0

    # Known Swing Lows
    lows[15] = 100.0
    lows[30] = 99.6
    lows[45] = 70.0

    # Ensure surrounding bars are higher than troughs (order 3)
    for offset in [-2, -1, 1, 2]:
        lows[15 + offset] = 108.0
        lows[30 + offset] = 108.0
        lows[45 + offset] = 85.0

    return _build_ohlcv(highs, lows)


# ---------------------------------------------------------------------------
# Module Exports Tests
# ---------------------------------------------------------------------------

def test_module_exports_from_indicators_init():
    """Verify that calculate_dynamic_sr_zones is exported from src.indicators."""
    assert exported_calc_sr is calculate_dynamic_sr_zones
    assert callable(exported_calc_sr)


# ---------------------------------------------------------------------------
# Insufficient Data Acceptance Criteria Tests
# ---------------------------------------------------------------------------

def test_insufficient_data_empty_series_returns_empty_zones(empty_ohlcv_df):
    """
    Given an empty OHLCV series
    When dynamic support and resistance calculation is requested
    Then return empty zone lists without raising an unhandled exception.
    """
    result = calculate_dynamic_sr_zones(empty_ohlcv_df, window=20, tolerance=0.01)

    assert isinstance(result, DynamicSRResult)
    assert isinstance(result.support_zones, list)
    assert isinstance(result.resistance_zones, list)
    assert len(result.support_zones) == 0
    assert len(result.resistance_zones) == 0


def test_insufficient_data_fewer_than_window_returns_empty_zones(short_ohlcv_df):
    """
    Given an OHLCV series with fewer data points (10) than the required window size (20)
    When dynamic support and resistance calculation is requested
    Then return empty zone lists without raising an unhandled exception.
    """
    result = calculate_dynamic_sr_zones(short_ohlcv_df, window=20, tolerance=0.02)

    assert isinstance(result, DynamicSRResult)
    assert result.support_zones == []
    assert result.resistance_zones == []


def test_exact_window_size_does_not_crash(short_ohlcv_df):
    """
    Given an OHLCV series with exactly window length data points (10 == 10)
    When dynamic S&R calculation is requested
    Then calculation runs and returns empty or valid zones without raising.
    """
    result = calculate_dynamic_sr_zones(short_ohlcv_df, window=10, tolerance=0.02)
    assert isinstance(result, DynamicSRResult)
    assert isinstance(result.support_zones, list)
    assert isinstance(result.resistance_zones, list)


# ---------------------------------------------------------------------------
# Dynamic Swing Detection & Clustering Acceptance Criteria Tests
# ---------------------------------------------------------------------------

def test_swing_highs_clustered_within_tolerance(synthetic_swing_ohlcv_df):
    """
    Given an OHLCV series with swing highs within tolerance band
    When dynamic resistance zones are calculated
    Then swing highs close together cluster into a single zone with upper and lower boundaries.
    """
    # Peaks at 150.0 and 150.4 are within ~0.3% of each other. Peak at 180.0 is ~20% away.
    # tolerance=0.01 (1%) should cluster 150.0 and 150.4, leaving 180.0 separate.
    result = calculate_dynamic_sr_zones(
        synthetic_swing_ohlcv_df,
        window=50,
        tolerance=0.01,
        swing_order=2,
    )

    assert len(result.resistance_zones) >= 2

    # Find the cluster around 150
    clustered_zone = None
    isolated_zone = None
    for zone in result.resistance_zones:
        lower, upper = _get_zone_boundaries(zone)
        if 148.0 <= lower <= 152.0 or 148.0 <= upper <= 152.0:
            clustered_zone = zone
        elif upper >= 175.0:
            isolated_zone = zone

    assert clustered_zone is not None, "Expected clustered resistance zone near 150.0"
    c_lower, c_upper = _get_zone_boundaries(clustered_zone)
    assert c_lower <= 150.4
    assert c_upper >= 150.0
    assert c_upper >= c_lower

    assert isolated_zone is not None, "Expected isolated resistance zone near 180.0"
    i_lower, i_upper = _get_zone_boundaries(isolated_zone)
    assert i_upper >= 180.0
    assert i_upper >= i_lower


def test_swing_lows_clustered_within_tolerance(synthetic_swing_ohlcv_df):
    """
    Given an OHLCV series with swing lows within tolerance band
    When dynamic support zones are calculated
    Then swing lows close together cluster into a single zone with upper and lower boundaries.
    """
    # Troughs at 100.0 and 99.6 are within 0.4% of each other. Trough at 70.0 is ~30% away.
    # tolerance=0.01 (1%) should cluster 100.0 and 99.6, leaving 70.0 separate.
    result = calculate_dynamic_sr_zones(
        synthetic_swing_ohlcv_df,
        window=50,
        tolerance=0.01,
        swing_order=2,
    )

    assert len(result.support_zones) >= 2

    clustered_zone = None
    isolated_zone = None
    for zone in result.support_zones:
        lower, upper = _get_zone_boundaries(zone)
        if 98.0 <= lower <= 102.0 or 98.0 <= upper <= 102.0:
            clustered_zone = zone
        elif lower <= 75.0:
            isolated_zone = zone

    assert clustered_zone is not None, "Expected clustered support zone near 100.0"
    c_lower, c_upper = _get_zone_boundaries(clustered_zone)
    assert c_lower <= 100.0
    assert c_upper >= 99.6
    assert c_upper >= c_lower

    assert isolated_zone is not None, "Expected isolated support zone near 70.0"
    i_lower, i_upper = _get_zone_boundaries(isolated_zone)
    assert i_lower <= 70.0
    assert i_upper >= i_lower


def test_zone_boundary_validity_and_ordering(synthetic_swing_ohlcv_df):
    """Every detected support and resistance zone must satisfy upper_boundary >= lower_boundary."""
    result = calculate_dynamic_sr_zones(
        synthetic_swing_ohlcv_df,
        window=40,
        tolerance=0.02,
        swing_order=2,
    )

    all_zones: List[SRZone] = result.support_zones + result.resistance_zones
    assert len(all_zones) > 0

    for zone in all_zones:
        lower, upper = _get_zone_boundaries(zone)
        assert not math.isnan(lower)
        assert not math.isnan(upper)
        assert not math.isinf(lower)
        assert not math.isinf(upper)
        assert upper >= lower, f"Upper boundary {upper} must be >= lower boundary {lower}"


def test_rolling_window_excludes_old_swings(synthetic_swing_ohlcv_df):
    """
    Given a swing high at idx 10 and window=20 on a 50-bar series
    When dynamic zones are calculated for the latest window [30..49]
    Then the swing high at idx 10 should NOT be included in the active zones.
    """
    window_size = 20
    # synthetic_swing_ohlcv_df has peak at idx 10 (150.0), peak at idx 25 (150.4), peak at idx 40 (180.0).
    # For window=20 (indices 30 to 49), only peak at idx 40 (180.0) falls within the window.
    result = calculate_dynamic_sr_zones(
        synthetic_swing_ohlcv_df,
        window=window_size,
        tolerance=0.02,
        swing_order=2,
    )

    # 150.0 should not be present in resistance zones
    for zone in result.resistance_zones:
        lower, upper = _get_zone_boundaries(zone)
        assert not (149.0 <= lower <= 151.0 or 149.0 <= upper <= 151.0), (
            f"Zone ({lower}, {upper}) from idx 10 should have expired outside rolling window of {window_size}"
        )


def test_tolerance_sensitivity_merging():
    """
    Given swing points with fixed relative distance
    When tolerance is widened
    Then separate zones merge into fewer, broader zones.
    """
    n = 40
    highs = np.full(n, 100.0, dtype=float)
    lows = np.full(n, 90.0, dtype=float)

    # Two peaks 2.5% apart: 100.0 and 102.5
    highs[10] = 100.0
    highs[20] = 102.5
    for offset in [-1, 1]:
        highs[10 + offset] = 95.0
        highs[20 + offset] = 95.0

    df = _build_ohlcv(highs, lows)

    # Tight tolerance (1%): should produce 2 distinct resistance zones
    res_tight = calculate_dynamic_sr_zones(df, window=35, tolerance=0.01, swing_order=1)
    # Loose tolerance (5%): should merge into 1 resistance zone
    res_loose = calculate_dynamic_sr_zones(df, window=35, tolerance=0.05, swing_order=1)

    assert len(res_tight.resistance_zones) == 2
    assert len(res_loose.resistance_zones) == 1

    loose_lower, loose_upper = _get_zone_boundaries(res_loose.resistance_zones[0])
    assert loose_lower <= 100.0
    assert loose_upper >= 102.5


def test_case_insensitive_column_handling(synthetic_swing_ohlcv_df):
    """Verify that uppercase column names (Open, High, Low, Close, Volume) are supported."""
    df_upper = synthetic_swing_ohlcv_df.rename(columns=str.capitalize)
    result = calculate_dynamic_sr_zones(df_upper, window=30, tolerance=0.02, swing_order=2)

    assert isinstance(result, DynamicSRResult)
    assert isinstance(result.support_zones, list)
    assert isinstance(result.resistance_zones, list)


def test_flat_market_produces_no_or_graceful_zones():
    """Given completely flat price series, dynamic S&R calculation handles it gracefully without error."""
    n = 30
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    df = _build_ohlcv(highs, lows)

    result = calculate_dynamic_sr_zones(df, window=20, tolerance=0.01)
    assert isinstance(result, DynamicSRResult)
    assert isinstance(result.support_zones, list)
    assert isinstance(result.resistance_zones, list)


# ---------------------------------------------------------------------------
# Validation & Error Handling Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_window", [0, -1, -50])
def test_invalid_window_raises_value_error(synthetic_swing_ohlcv_df, invalid_window):
    """Window size <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_dynamic_sr_zones(synthetic_swing_ohlcv_df, window=invalid_window, tolerance=0.01)


@pytest.mark.parametrize("invalid_tolerance", [0.0, -0.01, -1.0])
def test_invalid_tolerance_raises_value_error(synthetic_swing_ohlcv_df, invalid_tolerance):
    """Tolerance band <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_dynamic_sr_zones(synthetic_swing_ohlcv_df, window=20, tolerance=invalid_tolerance)


def test_missing_required_columns_raises_error():
    """DataFrame missing 'high' or 'low' columns must raise KeyError or ValueError."""
    bad_df = pd.DataFrame({"close": [10.0, 11.0, 12.0], "volume": [100, 200, 300]})
    with pytest.raises((KeyError, ValueError)):
        calculate_dynamic_sr_zones(bad_df, window=2, tolerance=0.01)


def test_non_dataframe_input_raises_type_error():
    """Passing a non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError):
        calculate_dynamic_sr_zones([1, 2, 3], window=2, tolerance=0.01)  # type: ignore