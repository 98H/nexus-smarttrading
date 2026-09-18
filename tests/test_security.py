"""
Unit tests for Multi-Timeframe Series Pull (request.security) Engine.

Specification: Story 3.3.3: Implement Multi-Timeframe Series Pull (request.security) Engine
Target Modules:
    - src/engine/security.py
    - src/engine/__init__.py
"""

import numpy as np
import pandas as pd
import pytest

from src.engine import resolve_security_series as resolve_from_engine_init
from src.engine.security import resolve_security_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def primary_1m_index() -> pd.DatetimeIndex:
    """Primary lower-timeframe index (1-minute bars) spanning 09:00 to 09:15 UTC."""
    return pd.date_range(
        start="2023-01-01 09:00:00",
        end="2023-01-01 09:15:00",
        freq="1min",
        tz="UTC",
        name="timestamp",
    )


@pytest.fixture
def primary_1m_series(primary_1m_index: pd.DatetimeIndex) -> pd.Series:
    """Primary lower-timeframe close series (1-minute bars)."""
    values = np.linspace(100.0, 115.0, len(primary_1m_index))
    return pd.Series(values, index=primary_1m_index, name="primary_close")


@pytest.fixture
def secondary_5m_series() -> pd.Series:
    """
    Secondary higher-timeframe series (5-minute bars) with bar open times:
    08:55:00 -> value 10.0 (closes at 09:00:00)
    09:00:00 -> value 20.0 (closes at 09:05:00)
    09:05:00 -> value 30.0 (closes at 09:10:00)
    09:10:00 -> value 40.0 (closes at 09:15:00)
    """
    index = pd.date_range(
        start="2023-01-01 08:55:00",
        end="2023-01-01 09:10:00",
        freq="5min",
        tz="UTC",
        name="timestamp",
    )
    return pd.Series([10.0, 20.0, 30.0, 40.0], index=index, name="secondary_close")


# ---------------------------------------------------------------------------
# Module and Export Integrity
# ---------------------------------------------------------------------------


def test_engine_package_exports_resolve_security_series():
    """Verify resolve_security_series is properly exposed in src.engine namespace."""
    assert resolve_from_engine_init is resolve_security_series
    assert callable(resolve_security_series)


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: Closed-Bar Alignment Without Lookahead Bias
# ---------------------------------------------------------------------------


def test_resolve_security_series_lookahead_false_alignment(
    primary_1m_series: pd.Series, secondary_5m_series: pd.Series
):
    """
    Given a primary 1m series (09:00 - 09:15) and secondary 5m series (08:55 - 09:10),
    When resolve_security_series is called with lookahead=False,
    Then data is aligned using closed-bar forward filling without lookahead bias:
        - 09:00 to 09:04 receives 10.0 (from 08:55 bar closed at 09:00)
        - 09:05 to 09:09 receives 20.0 (from 09:00 bar closed at 09:05)
        - 09:10 to 09:14 receives 30.0 (from 09:05 bar closed at 09:10)
        - 09:15 receives 40.0 (from 09:10 bar closed at 09:15)
    """
    aligned = resolve_security_series(
        primary_series=primary_1m_series,
        secondary_series=secondary_5m_series,
        lookahead=False,
    )

    # Validate alignment for 09:00 through 09:04 (08:55 bar closed at 09:00:00)
    for minute in range(0, 5):
        ts = pd.Timestamp(f"2023-01-01 09:0{minute}:00", tz="UTC")
        assert aligned.loc[ts] == pytest.approx(10.0), f"Lookahead leak at {ts}"

    # Validate alignment for 09:05 through 09:09 (09:00 bar closed at 09:05:00)
    for minute in range(5, 10):
        ts = pd.Timestamp(f"2023-01-01 09:0{minute}:00", tz="UTC")
        assert aligned.loc[ts] == pytest.approx(20.0), f"Lookahead leak at {ts}"

    # Validate alignment for 09:10 through 09:14 (09:05 bar closed at 09:10:00)
    for minute in range(10, 15):
        ts = pd.Timestamp(f"2023-01-01 09:{minute}:00", tz="UTC")
        assert aligned.loc[ts] == pytest.approx(30.0), f"Lookahead leak at {ts}"

    # Validate alignment for 09:15 (09:10 bar closed at 09:15:00)
    ts_final = pd.Timestamp("2023-01-01 09:15:00", tz="UTC")
    assert aligned.loc[ts_final] == pytest.approx(40.0)


def test_resolve_security_series_default_lookahead_is_false(
    primary_1m_series: pd.Series, secondary_5m_series: pd.Series
):
    """
    Given a primary series and a secondary series,
    When resolve_security_series is called without specifying lookahead,
    Then the default behavior must be lookahead=False (no future leakage).
    """
    aligned_default = resolve_security_series(
        primary_series=primary_1m_series,
        secondary_series=secondary_5m_series,
    )
    aligned_explicit_false = resolve_security_series(
        primary_series=primary_1m_series,
        secondary_series=secondary_5m_series,
        lookahead=False,
    )

    pd.testing.assert_series_equal(aligned_default, aligned_explicit_false)


def test_resolve_security_series_preserves_primary_index(
    primary_1m_series: pd.Series, secondary_5m_series: pd.Series
):
    """
    Given primary and secondary series,
    When resolve_security_series is executed,
    Then the resulting series must strictly preserve the primary series index.
    """
    aligned = resolve_security_series(
        primary_series=primary_1m_series,
        secondary_series=secondary_5m_series,
        lookahead=False,
    )

    assert len(aligned) == len(primary_1m_series)
    assert aligned.index.equals(primary_1m_series.index)


def test_resolve_security_series_initial_unclosed_bars_yield_nan(
    secondary_5m_series: pd.Series,
):
    """
    Given primary series starting before any secondary bar has completed and closed,
    When resolve_security_series is called with lookahead=False,
    Then timestamps preceding the first completed secondary close must be NaN.
    """
    # Primary starting at 08:50 (first secondary bar is 08:55, which closes at 09:00)
    pre_primary_index = pd.date_range(
        start="2023-01-01 08:50:00",
        end="2023-01-01 09:05:00",
        freq="1min",
        tz="UTC",
    )
    pre_primary_series = pd.Series(100.0, index=pre_primary_index)

    aligned = resolve_security_series(
        primary_series=pre_primary_series,
        secondary_series=secondary_5m_series,
        lookahead=False,
    )

    # Bars from 08:50 to 08:59 have no closed 5m bar available
    for minute in range(50, 60):
        ts = pd.Timestamp(f"2023-01-01 08:{minute}:00", tz="UTC")
        assert np.isnan(aligned.loc[ts]), f"Expected NaN at {ts} prior to bar close"

    # Bar at 09:00 sees the closed 08:55 bar (value 10.0)
    assert aligned.loc[pd.Timestamp("2023-01-01 09:00:00", tz="UTC")] == pytest.approx(10.0)


def test_resolve_security_series_dataframe_support(
    primary_1m_series: pd.Series, secondary_5m_series: pd.Series
):
    """
    Given a multi-column DataFrame secondary series (e.g. OHLCV),
    When resolve_security_series is called with lookahead=False,
    Then each column is correctly aligned to the primary index without lookahead bias.
    """
    sec_df = pd.DataFrame(
        {
            "open": secondary_5m_series,
            "high": secondary_5m_series + 2.0,
            "low": secondary_5m_series - 2.0,
            "close": secondary_5m_series + 1.0,
            "volume": [1000, 2000, 3000, 4000],
        },
        index=secondary_5m_series.index,
    )

    aligned_df = resolve_security_series(
        primary_series=primary_1m_series,
        secondary_series=sec_df,
        lookahead=False,
    )

    assert isinstance(aligned_df, pd.DataFrame)
    assert list(aligned_df.columns) == ["open", "high", "low", "close", "volume"]
    assert aligned_df.index.equals(primary_1m_series.index)

    # Verify lookahead prevention on multiple columns at 09:04
    ts_0904 = pd.Timestamp("2023-01-01 09:04:00", tz="UTC")
    assert aligned_df.loc[ts_0904, "open"] == pytest.approx(10.0)
    assert aligned_df.loc[ts_0904, "high"] == pytest.approx(12.0)
    assert aligned_df.loc[ts_0904, "close"] == pytest.approx(11.0)

    # Verify shift at 09:05
    ts_0905 = pd.Timestamp("2023-01-01 09:05:00", tz="UTC")
    assert aligned_df.loc[ts_0905, "open"] == pytest.approx(20.0)
    assert aligned_df.loc[ts_0905, "high"] == pytest.approx(22.0)
    assert aligned_df.loc[ts_0905, "close"] == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: Unaligned or Non-Chronological Secondary Series
# ---------------------------------------------------------------------------


def test_raises_value_error_on_non_chronological_secondary(
    primary_1m_series: pd.Series,
):
    """
    Given a secondary series with timestamps in non-chronological order,
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    out_of_order_index = pd.to_datetime(
        [
            "2023-01-01 09:05:00",
            "2023-01-01 09:00:00",
            "2023-01-01 09:10:00",
        ],
        utc=True,
    )
    non_chrono_secondary = pd.Series([20.0, 10.0, 30.0], index=out_of_order_index)

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=primary_1m_series,
            secondary_series=non_chrono_secondary,
            lookahead=False,
        )


def test_raises_value_error_on_unaligned_secondary_timestamps(
    primary_1m_series: pd.Series,
):
    """
    Given a secondary series with irregular or unaligned timestamps that do not match
    a valid timeframe grid,
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    unaligned_index = pd.to_datetime(
        [
            "2023-01-01 08:55:00",
            "2023-01-01 09:01:23",  # Unaligned timestamp
            "2023-01-01 09:06:45",
        ],
        utc=True,
    )
    unaligned_secondary = pd.Series([10.0, 20.0, 30.0], index=unaligned_index)

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=primary_1m_series,
            secondary_series=unaligned_secondary,
            lookahead=False,
        )


def test_raises_value_error_on_duplicate_secondary_timestamps(
    primary_1m_series: pd.Series,
):
    """
    Given a secondary series containing duplicate timestamps,
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    duplicate_index = pd.to_datetime(
        [
            "2023-01-01 08:55:00",
            "2023-01-01 09:00:00",
            "2023-01-01 09:00:00",  # Duplicate timestamp
            "2023-01-01 09:05:00",
        ],
        utc=True,
    )
    dup_secondary = pd.Series([10.0, 20.0, 25.0, 30.0], index=duplicate_index)

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=primary_1m_series,
            secondary_series=dup_secondary,
            lookahead=False,
        )


def test_raises_value_error_on_empty_secondary_series(
    primary_1m_series: pd.Series,
):
    """
    Given an empty secondary series,
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    empty_secondary = pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([], tz="UTC"),
    )

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=primary_1m_series,
            secondary_series=empty_secondary,
            lookahead=False,
        )


def test_raises_value_error_on_non_chronological_primary(
    secondary_5m_series: pd.Series,
):
    """
    Given a primary series with non-chronological index,
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    out_of_order_primary_index = pd.to_datetime(
        [
            "2023-01-01 09:02:00",
            "2023-01-01 09:00:00",
            "2023-01-01 09:01:00",
        ],
        utc=True,
    )
    non_chrono_primary = pd.Series([102.0, 100.0, 101.0], index=out_of_order_primary_index)

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=non_chrono_primary,
            secondary_series=secondary_5m_series,
            lookahead=False,
        )


def test_raises_value_error_on_timezone_mismatch(
    primary_1m_series: pd.Series,
):
    """
    Given primary and secondary series with mismatched timezones (UTC vs naive/localized),
    When resolve_security_series is invoked,
    Then a ValueError is raised before resolution.
    """
    naive_secondary_index = pd.date_range(
        start="2023-01-01 08:55:00",
        end="2023-01-01 09:10:00",
        freq="5min",
        tz=None,  # Naive timezone
    )
    naive_secondary = pd.Series([10.0, 20.0, 30.0, 40.0], index=naive_secondary_index)

    with pytest.raises(ValueError):
        resolve_security_series(
            primary_series=primary_1m_series,
            secondary_series=naive_secondary,
            lookahead=False,
        )