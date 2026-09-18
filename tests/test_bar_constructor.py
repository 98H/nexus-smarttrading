from datetime import datetime, timedelta, timezone
import pytest

from src.market_data.bar_constructor import DynamicBarConstructor
from src.market_data.models import Bar, Trade


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_time() -> datetime:
    """Provides a fixed, UTC-aligned base timestamp for deterministic testing."""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# 1. Constructor Initialization & Parameter Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "resolution",
    [
        "500ms",
        "1s",
        "3s",
        "15s",
        "1m",
        "7m",
        "15m",
        "1h",
        "2h",
        "4h",
        "1d",
    ],
)
def test_constructor_accepts_valid_resolutions(resolution: str):
    """Constructor must accept standard and custom resolution formats."""
    constructor = DynamicBarConstructor(resolution=resolution)
    assert constructor.resolution == resolution


@pytest.mark.parametrize(
    "invalid_resolution",
    [
        "",
        "0s",
        "-5s",
        "0m",
        "-2h",
        "invalid",
        "10x",
        "s",
        "m",
        "h",
        "3.5s",
    ],
)
def test_constructor_rejects_invalid_resolutions(invalid_resolution: str):
    """Constructor must raise ValueError on malformed or non-positive resolutions."""
    with pytest.raises(ValueError):
        DynamicBarConstructor(resolution=invalid_resolution)


def test_initial_state_empty():
    """Before any trades are ingested, current bar query returns None."""
    constructor = DynamicBarConstructor(resolution="3s")
    assert constructor.get_current_bar() is None


# ============================================================================
# 2. In-Progress Bar Queries (Acceptance Criteria 2)
# ============================================================================


def test_running_bar_state_single_trade(base_time: datetime):
    """After one trade, in-progress bar OHLC must equal trade price, and volume match."""
    constructor = DynamicBarConstructor(resolution="3s")
    trade = Trade(
        timestamp=base_time + timedelta(milliseconds=100),
        price=100.0,
        volume=10.0,
    )

    completed_bars = constructor.process_trade(trade)

    assert completed_bars == []
    current_bar = constructor.get_current_bar()
    assert current_bar is not None
    assert current_bar.open == 100.0
    assert current_bar.high == 100.0
    assert current_bar.low == 100.0
    assert current_bar.close == 100.0
    assert current_bar.volume == 10.0


def test_running_bar_state_cumulative_updates(base_time: datetime):
    """Running bar must accurately maintain high, low, close, and cumulative volume."""
    constructor = DynamicBarConstructor(resolution="3s")

    # Trade 1: initial
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(milliseconds=100), price=100.0, volume=5.0)
    )
    # Trade 2: higher price
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(milliseconds=500), price=105.0, volume=10.0)
    )
    # Trade 3: lower price
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=1, milliseconds=200), price=95.0, volume=15.0)
    )
    # Trade 4: intermediate close
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=2, milliseconds=800), price=98.0, volume=20.0)
    )

    current_bar = constructor.get_current_bar()
    assert current_bar is not None
    assert current_bar.open == 100.0
    assert current_bar.high == 105.0
    assert current_bar.low == 95.0
    assert current_bar.close == 98.0
    assert current_bar.volume == 50.0  # 5 + 10 + 15 + 20


# ============================================================================
# 3. Bar Completion, Boundary Alignment & Aggregation (Acceptance Criteria 1)
# ============================================================================


def test_bar_emission_at_boundary_3s(base_time: datetime):
    """Trades within [00:00:00, 00:00:03) close upon trade at/after 00:00:03."""
    constructor = DynamicBarConstructor(resolution="3s")

    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(milliseconds=200), price=100.0, volume=10.0)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=1, milliseconds=500), price=110.0, volume=5.0)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=2, milliseconds=900), price=90.0, volume=15.0)
    )

    # Boundary trade exactly at 00:00:03.000 triggers bar completion
    boundary_trade = Trade(
        timestamp=base_time + timedelta(seconds=3),
        price=95.0,
        volume=25.0,
    )
    completed_bars = constructor.process_trade(boundary_trade)

    assert len(completed_bars) == 1
    bar = completed_bars[0]
    assert bar.open == 100.0
    assert bar.high == 110.0
    assert bar.low == 90.0
    assert bar.close == 90.0
    assert bar.volume == 30.0  # 10 + 5 + 15
    assert bar.close_timestamp == base_time + timedelta(seconds=3)


def test_no_lookahead_bias(base_time: datetime):
    """The trade triggering the boundary close must NOT bleed into the closed bar."""
    constructor = DynamicBarConstructor(resolution="3s")

    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=1), price=100.0, volume=10.0)
    )

    # Trigger trade arrives after the boundary with a drastic new high and volume
    trigger_trade = Trade(
        timestamp=base_time + timedelta(seconds=3, milliseconds=100),
        price=200.0,
        volume=50.0,
    )
    completed_bars = constructor.process_trade(trigger_trade)

    assert len(completed_bars) == 1
    closed_bar = completed_bars[0]

    # Closed bar must strictly reflect data prior to boundary
    assert closed_bar.high == 100.0
    assert closed_bar.close == 100.0
    assert closed_bar.volume == 10.0
    assert closed_bar.close_timestamp == base_time + timedelta(seconds=3)

    # In-progress bar must contain the triggering trade
    current_bar = constructor.get_current_bar()
    assert current_bar is not None
    assert current_bar.open == 200.0
    assert current_bar.high == 200.0
    assert current_bar.low == 200.0
    assert current_bar.close == 200.0
    assert current_bar.volume == 50.0


def test_custom_resolution_7m_boundary_alignment(base_time: datetime):
    """Bars must align to custom non-standard minute boundaries (e.g., 7m)."""
    constructor = DynamicBarConstructor(resolution="7m")

    # Base time is 12:00:00 -> First bar interval is [12:00:00, 12:07:00)
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(minutes=1), price=50.0, volume=100.0)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(minutes=4), price=55.0, volume=200.0)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(minutes=6, seconds=59), price=48.0, volume=150.0)
    )

    # Trade at 12:07:00 crosses into next 7m bar [12:07:00, 12:14:00)
    trade_next_window = Trade(
        timestamp=base_time + timedelta(minutes=7),
        price=52.0,
        volume=75.0,
    )
    completed_bars = constructor.process_trade(trade_next_window)

    assert len(completed_bars) == 1
    bar = completed_bars[0]
    assert bar.open == 50.0
    assert bar.high == 55.0
    assert bar.low == 48.0
    assert bar.close == 48.0
    assert bar.volume == 450.0
    assert bar.close_timestamp == base_time + timedelta(minutes=7)

    # Verify state of the new unfinalized bar
    current_bar = constructor.get_current_bar()
    assert current_bar is not None
    assert current_bar.open == 52.0
    assert current_bar.close == 52.0
    assert current_bar.volume == 75.0


def test_custom_resolution_2h_boundary_alignment(base_time: datetime):
    """Bars must align to multi-hour boundary intervals (e.g., 2h)."""
    # base_time: 12:00:00 UTC -> bar [12:00:00, 14:00:00)
    constructor = DynamicBarConstructor(resolution="2h")

    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(minutes=30), price=1000.0, volume=2.5)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(hours=1, minutes=45), price=1020.0, volume=3.5)
    )

    # Trade crossing boundary at 14:01:00
    trade_crossing = Trade(
        timestamp=base_time + timedelta(hours=2, minutes=1),
        price=1010.0,
        volume=1.0,
    )
    completed_bars = constructor.process_trade(trade_crossing)

    assert len(completed_bars) == 1
    bar = completed_bars[0]
    assert bar.open == 1000.0
    assert bar.high == 1020.0
    assert bar.low == 1000.0
    assert bar.close == 1020.0
    assert bar.volume == 6.0
    assert bar.close_timestamp == base_time + timedelta(hours=2)


# ============================================================================
# 4. Multi-Interval Skips and Continuous Streams
# ============================================================================


def test_large_time_gap_between_trades(base_time: datetime):
    """When a trade arrives after multiple skipped intervals, active bar closes without phantom bars."""
    constructor = DynamicBarConstructor(resolution="3s")

    # Trade in [00:00:00, 00:00:03)
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=1), price=10.0, volume=100.0)
    )

    # Next trade arrives at 00:00:15 (skipping 00:03-00:06, 00:06-00:09, 00:09-00:12)
    completed = constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=15, milliseconds=500), price=12.0, volume=50.0)
    )

    # Exactly one bar emitted for the window that had data
    assert len(completed) == 1
    assert completed[0].close_timestamp == base_time + timedelta(seconds=3)
    assert completed[0].open == 10.0
    assert completed[0].close == 10.0
    assert completed[0].volume == 100.0

    # Current bar is now tracking the 00:00:15 - 00:00:18 interval
    current = constructor.get_current_bar()
    assert current is not None
    assert current.open == 12.0
    assert current.volume == 50.0


def test_sequential_multi_bar_generation(base_time: datetime):
    """Stream of ticks spanning consecutive intervals produces sequential completed bars."""
    constructor = DynamicBarConstructor(resolution="5s")
    all_emitted_bars: list[Bar] = []

    # Window 0: [00:00:00, 00:00:05)
    all_emitted_bars.extend(
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=1), price=100.0, volume=10.0)
        )
    )
    all_emitted_bars.extend(
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=4), price=102.0, volume=20.0)
        )
    )
    assert len(all_emitted_bars) == 0

    # Window 1: [00:00:05, 00:00:10)
    all_emitted_bars.extend(
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=5), price=103.0, volume=15.0)
        )
    )
    assert len(all_emitted_bars) == 1
    assert all_emitted_bars[0].close_timestamp == base_time + timedelta(seconds=5)
    assert all_emitted_bars[0].close == 102.0

    all_emitted_bars.extend(
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=7), price=99.0, volume=25.0)
        )
    )
    assert len(all_emitted_bars) == 1

    # Window 2: [00:00:10, 00:00:15)
    all_emitted_bars.extend(
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=10), price=101.0, volume=30.0)
        )
    )
    assert len(all_emitted_bars) == 2
    assert all_emitted_bars[1].close_timestamp == base_time + timedelta(seconds=10)
    assert all_emitted_bars[1].open == 103.0
    assert all_emitted_bars[1].low == 99.0
    assert all_emitted_bars[1].close == 99.0
    assert all_emitted_bars[1].volume == 40.0


# ============================================================================
# 5. Data Integrity & Validation Rejections
# ============================================================================


def test_reject_out_of_order_trade_timestamps(base_time: datetime):
    """Constructor must reject ticks arriving with non-monotonic timestamps."""
    constructor = DynamicBarConstructor(resolution="1m")

    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=30), price=100.0, volume=10.0)
    )

    # Previous timestamp was +30s, arrival of +20s is out-of-order
    with pytest.raises(ValueError):
        constructor.process_trade(
            Trade(timestamp=base_time + timedelta(seconds=20), price=101.0, volume=5.0)
        )


@pytest.mark.parametrize("invalid_price", [-10.0, 0.0, float("nan"), float("inf")])
def test_reject_invalid_trade_prices(base_time: datetime, invalid_price: float):
    """Trades with non-positive or non-finite prices must be rejected."""
    with pytest.raises(ValueError):
        Trade(timestamp=base_time, price=invalid_price, volume=10.0)


@pytest.mark.parametrize("invalid_volume", [-1.0, float("nan"), float("inf")])
def test_reject_invalid_trade_volumes(base_time: datetime, invalid_volume: float):
    """Trades with negative or non-finite volumes must be rejected."""
    with pytest.raises(ValueError):
        Trade(timestamp=base_time, price=100.0, volume=invalid_volume)


def test_zero_volume_trade_handling(base_time: datetime):
    """Trades with valid zero volume (e.g. quote updates / indicative trades) update price without volume distortion."""
    constructor = DynamicBarConstructor(resolution="3s")

    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=1), price=100.0, volume=0.0)
    )
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(seconds=2), price=105.0, volume=0.0)
    )

    current = constructor.get_current_bar()
    assert current is not None
    assert current.open == 100.0
    assert current.high == 105.0
    assert current.close == 105.0
    assert current.volume == 0.0


def test_subsecond_microsecond_precision(base_time: datetime):
    """Boundary calculations must preserve microsecond precision without float drift."""
    constructor = DynamicBarConstructor(resolution="500ms")

    # Bar 0: [12:00:00.000000, 12:00:00.500000)
    constructor.process_trade(
        Trade(timestamp=base_time + timedelta(microseconds=499999), price=10.0, volume=1.0)
    )

    # Bar 1 trigger at exactly 12:00:00.500000
    completed = constructor.process_trade(
        Trade(timestamp=base_time + timedelta(microseconds=500000), price=11.0, volume=2.0)
    )

    assert len(completed) == 1
    assert completed[0].close_timestamp == base_time + timedelta(milliseconds=500)
    assert completed[0].close == 10.0
    assert completed[0].volume == 1.0

    current = constructor.get_current_bar()
    assert current is not None
    assert current.open == 11.0
    assert current.volume == 2.0