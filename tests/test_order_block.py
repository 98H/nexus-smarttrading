"""
Unit tests for Dynamic Order Block (OB) Detection and Mitigation Tracker.

Story 4.1.2: Implement Dynamic Order Block (OB) Detection and Mitigation Tracker
Acceptance Criteria:
- Given a validated swing breakout candle sequence in an OHLCV series,
  When detect_order_blocks is evaluated,
  Then an unmitigated OrderBlock is created using the high and low bounds
  of the final opposing candle before the displacement.
- Given an active (unmitigated) bullish or bearish OrderBlock,
  When subsequent price action touches or crosses within the block's boundaries,
  Then the tracker marks the block as mitigated and records the mitigation timestamp.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

import src.indicators as indicators
from src.indicators.order_block import (
    OrderBlock,
    OrderBlockTracker,
    OrderBlockType,
    detect_order_blocks,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_timestamp() -> pd.Timestamp:
    """Provides a deterministic base timestamp."""
    return pd.Timestamp("2024-01-01 09:30:00")


@pytest.fixture
def bullish_breakout_df(base_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Creates an OHLCV sequence with a clear bullish breakout:
    - Candle 0: Range establishing a swing high at 100.0.
    - Candle 1: Pullback down.
    - Candle 2: Final opposing candle (bearish: close < open) before displacement.
                High = 95.0, Low = 90.0, Open = 93.0, Close = 91.0.
    - Candle 3: Bullish displacement breakout candle closing at 104.0 (breaks swing high 100.0).
    - Candle 4: Continuation higher, stays well above the order block.
    """
    timestamps = [base_timestamp + timedelta(minutes=i) for i in range(5)]
    data = {
        "timestamp": timestamps,
        "open": [92.0, 96.0, 93.0, 91.0, 104.0],
        "high": [100.0, 97.0, 95.0, 105.0, 108.0],
        "low": [90.0, 92.0, 90.0, 91.0, 102.0],
        "close": [98.0, 93.0, 91.0, 104.0, 107.0],
        "volume": [1000.0, 800.0, 1200.0, 5000.0, 2500.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def bearish_breakout_df(base_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Creates an OHLCV sequence with a clear bearish breakout:
    - Candle 0: Range establishing a swing low at 100.0.
    - Candle 1: Pullback up.
    - Candle 2: Final opposing candle (bullish: close > open) before displacement.
                High = 110.0, Low = 105.0, Open = 106.0, Close = 109.0.
    - Candle 3: Bearish displacement breakout candle closing at 96.0 (breaks swing low 100.0).
    - Candle 4: Continuation lower, stays well below the order block.
    """
    timestamps = [base_timestamp + timedelta(minutes=i) for i in range(5)]
    data = {
        "timestamp": timestamps,
        "open": [108.0, 103.0, 106.0, 109.0, 96.0],
        "high": [110.0, 107.0, 110.0, 109.0, 98.0],
        "low": [100.0, 103.0, 105.0, 95.0, 92.0],
        "close": [102.0, 106.0, 109.0, 96.0, 93.0],
        "volume": [1000.0, 800.0, 1200.0, 5000.0, 2500.0],
    }
    return pd.DataFrame(data)


# ============================================================================
# Package & Interface Tests
# ============================================================================


def test_package_exports():
    """Verify indicators package exposes key Order Block components."""
    assert hasattr(indicators, "detect_order_blocks")
    assert hasattr(indicators, "OrderBlock")
    assert hasattr(indicators, "OrderBlockTracker")
    assert hasattr(indicators, "OrderBlockType")


def test_order_block_dataclass_initialization_defaults(base_timestamp: pd.Timestamp):
    """Verify OrderBlock initializes with expected fields and unmitigated defaults."""
    ob = OrderBlock(
        timestamp=base_timestamp,
        block_type=OrderBlockType.BULLISH,
        high=105.0,
        low=100.0,
    )
    assert ob.timestamp == base_timestamp
    assert ob.block_type == OrderBlockType.BULLISH
    assert ob.high == 105.0
    assert ob.low == 100.0
    assert ob.mitigated is False
    assert ob.mitigated_at is None


# ============================================================================
# AC 1: Order Block Detection Tests
# ============================================================================


def test_detect_bullish_order_block_unmitigated(bullish_breakout_df: pd.DataFrame):
    """
    Given a validated swing breakout candle sequence in an OHLCV series,
    When detect_order_blocks is evaluated,
    Then an unmitigated Bullish OrderBlock is created using the bounds of the final opposing candle.
    """
    blocks = detect_order_blocks(bullish_breakout_df)

    assert len(blocks) == 1
    ob = blocks[0]

    # Opposing candle was index 2: high = 95.0, low = 90.0, timestamp = index 2 timestamp
    expected_timestamp = bullish_breakout_df.loc[2, "timestamp"]
    assert ob.block_type == OrderBlockType.BULLISH
    assert ob.high == 95.0
    assert ob.low == 90.0
    assert ob.timestamp == expected_timestamp
    assert ob.mitigated is False
    assert ob.mitigated_at is None


def test_detect_bearish_order_block_unmitigated(bearish_breakout_df: pd.DataFrame):
    """
    Given a validated bearish swing breakout candle sequence in an OHLCV series,
    When detect_order_blocks is evaluated,
    Then an unmitigated Bearish OrderBlock is created using the bounds of the final opposing candle.
    """
    blocks = detect_order_blocks(bearish_breakout_df)

    assert len(blocks) == 1
    ob = blocks[0]

    # Opposing candle was index 2: high = 110.0, low = 105.0, timestamp = index 2 timestamp
    expected_timestamp = bearish_breakout_df.loc[2, "timestamp"]
    assert ob.block_type == OrderBlockType.BEARISH
    assert ob.high == 110.0
    assert ob.low == 105.0
    assert ob.timestamp == expected_timestamp
    assert ob.mitigated is False
    assert ob.mitigated_at is None


# ============================================================================
# AC 2: Mitigation Tracking Tests
# ============================================================================


def test_bullish_order_block_mitigation_on_penetration(
    bullish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bullish OrderBlock [90.0, 95.0],
    When subsequent price action crosses inside the block's boundaries,
    Then the block is marked mitigated with the mitigation timestamp recorded.
    """
    # Candle 5 enters zone [90.0, 95.0] with low = 93.0
    candle_5_time = base_timestamp + timedelta(minutes=5)
    mitigating_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 105.0,
                "high": 106.0,
                "low": 93.0,
                "close": 96.0,
                "volume": 1500.0,
            }
        ]
    )
    df = pd.concat([bullish_breakout_df, mitigating_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == candle_5_time


def test_bullish_order_block_mitigation_on_exact_boundary_touch(
    bullish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bullish OrderBlock [90.0, 95.0],
    When subsequent price touches the exact top boundary (low == 95.0),
    Then the block is marked mitigated.
    """
    candle_5_time = base_timestamp + timedelta(minutes=5)
    touch_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 102.0,
                "high": 103.0,
                "low": 95.0,
                "close": 99.0,
                "volume": 1200.0,
            }
        ]
    )
    df = pd.concat([bullish_breakout_df, touch_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == candle_5_time


def test_bullish_order_block_remains_unmitigated_on_near_miss(
    bullish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bullish OrderBlock [90.0, 95.0],
    When subsequent price misses the boundary (e.g. low == 95.01),
    Then the block remains unmitigated.
    """
    candle_5_time = base_timestamp + timedelta(minutes=5)
    near_miss_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 102.0,
                "high": 103.0,
                "low": 95.01,
                "close": 99.0,
                "volume": 1200.0,
            }
        ]
    )
    df = pd.concat([bullish_breakout_df, near_miss_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is False
    assert ob.mitigated_at is None


def test_bearish_order_block_mitigation_on_penetration(
    bearish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bearish OrderBlock [105.0, 110.0],
    When subsequent price action crosses inside the block's boundaries,
    Then the block is marked mitigated with the mitigation timestamp recorded.
    """
    # Candle 5 enters zone [105.0, 110.0] with high = 107.0
    candle_5_time = base_timestamp + timedelta(minutes=5)
    mitigating_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 94.0,
                "high": 107.0,
                "low": 94.0,
                "close": 103.0,
                "volume": 1500.0,
            }
        ]
    )
    df = pd.concat([bearish_breakout_df, mitigating_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == candle_5_time


def test_bearish_order_block_mitigation_on_exact_boundary_touch(
    bearish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bearish OrderBlock [105.0, 110.0],
    When subsequent price touches the exact bottom boundary (high == 105.0),
    Then the block is marked mitigated.
    """
    candle_5_time = base_timestamp + timedelta(minutes=5)
    touch_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 95.0,
                "high": 105.0,
                "low": 94.0,
                "close": 98.0,
                "volume": 1100.0,
            }
        ]
    )
    df = pd.concat([bearish_breakout_df, touch_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == candle_5_time


def test_bearish_order_block_remains_unmitigated_on_near_miss(
    bearish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    Given an active bearish OrderBlock [105.0, 110.0],
    When subsequent price misses the boundary (e.g. high == 104.99),
    Then the block remains unmitigated.
    """
    candle_5_time = base_timestamp + timedelta(minutes=5)
    near_miss_candle = pd.DataFrame(
        [
            {
                "timestamp": candle_5_time,
                "open": 95.0,
                "high": 104.99,
                "low": 94.0,
                "close": 98.0,
                "volume": 1100.0,
            }
        ]
    )
    df = pd.concat([bearish_breakout_df, near_miss_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is False
    assert ob.mitigated_at is None


def test_mitigation_timestamp_records_first_touch_only(
    bullish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    When multiple subsequent candles interact with an OrderBlock,
    Then the tracker preserves the timestamp of the first mitigating candle.
    """
    first_touch_time = base_timestamp + timedelta(minutes=5)
    second_touch_time = base_timestamp + timedelta(minutes=6)

    subsequent_candles = pd.DataFrame(
        [
            {
                "timestamp": first_touch_time,
                "open": 104.0,
                "high": 104.0,
                "low": 94.0,  # Enters [90.0, 95.0]
                "close": 98.0,
                "volume": 1000.0,
            },
            {
                "timestamp": second_touch_time,
                "open": 98.0,
                "high": 99.0,
                "low": 91.0,  # Deep penetration inside [90.0, 95.0]
                "close": 95.0,
                "volume": 1400.0,
            },
        ]
    )
    df = pd.concat([bullish_breakout_df, subsequent_candles], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == first_touch_time


def test_gap_through_block_triggers_mitigation(
    bullish_breakout_df: pd.DataFrame, base_timestamp: pd.Timestamp
):
    """
    When price action gaps completely through the OrderBlock bounds,
    Then the block is marked mitigated.
    """
    # Bullish OB is [90.0, 95.0]. Price gaps below to [80.0, 85.0].
    gap_candle_time = base_timestamp + timedelta(minutes=5)
    gap_candle = pd.DataFrame(
        [
            {
                "timestamp": gap_candle_time,
                "open": 85.0,
                "high": 88.0,
                "low": 80.0,
                "close": 82.0,
                "volume": 3000.0,
            }
        ]
    )
    df = pd.concat([bullish_breakout_df, gap_candle], ignore_index=True)

    blocks = detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    assert ob.mitigated is True
    assert ob.mitigated_at == gap_candle_time


# ============================================================================
# OrderBlockTracker Standalone / State-Management Tests
# ============================================================================


def test_tracker_dynamic_candle_update_stream(base_timestamp: pd.Timestamp):
    """
    Verify OrderBlockTracker correctly updates mitigation state bar-by-bar.
    """
    block = OrderBlock(
        timestamp=base_timestamp,
        block_type=OrderBlockType.BULLISH,
        high=100.0,
        low=95.0,
    )
    tracker = OrderBlockTracker()
    tracker.add_block(block)

    assert len(tracker.active_blocks) == 1
    assert len(tracker.mitigated_blocks) == 0

    # Candle 1: Stays outside zone -> no mitigation
    candle_1 = {
        "timestamp": base_timestamp + timedelta(minutes=1),
        "open": 105.0,
        "high": 108.0,
        "low": 102.0,
        "close": 106.0,
    }
    tracker.update(candle_1)
    assert len(tracker.active_blocks) == 1
    assert len(tracker.mitigated_blocks) == 0
    assert block.mitigated is False

    # Candle 2: Touches zone at 100.0 -> marks mitigated
    mitigating_time = base_timestamp + timedelta(minutes=2)
    candle_2 = {
        "timestamp": mitigating_time,
        "open": 106.0,
        "high": 106.0,
        "low": 100.0,
        "close": 103.0,
    }
    tracker.update(candle_2)
    assert len(tracker.active_blocks) == 0
    assert len(tracker.mitigated_blocks) == 1
    assert block.mitigated is True
    assert block.mitigated_at == mitigating_time


def test_multiple_order_blocks_with_mixed_mitigation_states(
    base_timestamp: pd.Timestamp,
):
    """
    Verify tracker handles multiple active blocks concurrently and transitions
    only the touched blocks.
    """
    ob1 = OrderBlock(
        timestamp=base_timestamp,
        block_type=OrderBlockType.BULLISH,
        high=90.0,
        low=85.0,
    )
    ob2 = OrderBlock(
        timestamp=base_timestamp + timedelta(minutes=2),
        block_type=OrderBlockType.BULLISH,
        high=105.0,
        low=100.0,
    )

    tracker = OrderBlockTracker(blocks=[ob1, ob2])

    # Candle enters ob2 ([100.0, 105.0]) with low=102.0, but stays above ob1 ([85.0, 90.0])
    test_time = base_timestamp + timedelta(minutes=5)
    candle = {
        "timestamp": test_time,
        "open": 110.0,
        "high": 110.0,
        "low": 102.0,
        "close": 108.0,
    }
    tracker.update(candle)

    assert ob1.mitigated is False
    assert ob1.mitigated_at is None
    assert ob2.mitigated is True
    assert ob2.mitigated_at == test_time

    assert ob1 in tracker.active_blocks
    assert ob2 in tracker.mitigated_blocks


# ============================================================================
# Validation and Edge Cases
# ============================================================================


def test_detect_order_blocks_empty_dataframe():
    """Verify empty DataFrame input returns an empty list without error."""
    empty_df = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    blocks = detect_order_blocks(empty_df)
    assert blocks == []


def test_detect_order_blocks_missing_columns_raises_error():
    """Verify missing OHLC columns raises ValueError."""
    invalid_df = pd.DataFrame(
        {
            "timestamp": [datetime.now()],
            "open": [100.0],
            # 'high', 'low', 'close' missing
        }
    )
    with pytest.raises(ValueError):
        detect_order_blocks(invalid_df)


def test_detect_order_blocks_invalid_type_raises_error():
    """Verify passing non-DataFrame raises TypeError or ValueError."""
    with pytest.raises((TypeError, ValueError)):
        detect_order_blocks([1, 2, 3])  # type: ignore


def test_no_breakout_sequence_yields_no_order_blocks(base_timestamp: pd.Timestamp):
    """
    Given an OHLCV series with flat, oscillating price and no swing breakout,
    When detect_order_blocks is evaluated,
    Then no order blocks are detected.
    """
    timestamps = [base_timestamp + timedelta(minutes=i) for i in range(10)]
    flat_data = {
        "timestamp": timestamps,
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [100.0] * 10,
    }
    df = pd.DataFrame(flat_data)
    blocks = detect_order_blocks(df)
    assert blocks == []