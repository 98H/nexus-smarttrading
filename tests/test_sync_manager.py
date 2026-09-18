"""
Unit tests for Crosshair, Symbol, and Interval Multi-Chart Synchronizer.

Feature: Build Crosshair, Symbol, and Interval Multi-Chart Synchronizer
Requirement: Story 9.1.2: Build Crosshair, Symbol, and Interval Multi-Chart Synchronizer
Target Modules:
    - src/charting/sync_manager.py
    - src/charting/__init__.py
"""

from typing import List, Optional
import pytest

from src.charting import (
    CrosshairPosition as ExportedCrosshairPosition,
    MultiChartSynchronizer as ExportedMultiChartSynchronizer,
    SyncManager as ExportedSyncManager,
)
from src.charting.sync_manager import (
    CrosshairPosition,
    MultiChartSynchronizer,
    SyncManager,
)


class ChartSpy:
    """Deterministic spy representing a chart subscriber for synchronization events."""

    def __init__(self, chart_id: str):
        self.chart_id = chart_id
        self.crosshair_events: List[CrosshairPosition] = []
        self.symbol_events: List[str] = []
        self.interval_events: List[str] = []

    def on_crosshair(self, position: CrosshairPosition) -> None:
        self.crosshair_events.append(position)

    def on_symbol(self, symbol: str) -> None:
        self.symbol_events.append(symbol)

    def on_interval(self, interval: str) -> None:
        self.interval_events.append(interval)

    def clear(self) -> None:
        self.crosshair_events.clear()
        self.symbol_events.clear()
        self.interval_events.clear()


@pytest.fixture
def synchronizer() -> SyncManager:
    """Provides a clean SyncManager instance for each test."""
    return SyncManager()


@pytest.fixture
def chart_a() -> ChartSpy:
    return ChartSpy(chart_id="chart-A")


@pytest.fixture
def chart_b() -> ChartSpy:
    return ChartSpy(chart_id="chart-B")


@pytest.fixture
def chart_c() -> ChartSpy:
    return ChartSpy(chart_id="chart-C")


# ==============================================================================
# 1. Package Exports & API Consistency Tests
# ==============================================================================


def test_package_exports():
    """Verify primary classes are exported from both module and package root."""
    assert SyncManager is ExportedSyncManager
    assert MultiChartSynchronizer is ExportedMultiChartSynchronizer
    assert CrosshairPosition is ExportedCrosshairPosition
    assert MultiChartSynchronizer is SyncManager


def test_crosshair_position_dataclass():
    """Verify CrosshairPosition value semantics and properties."""
    pos1 = CrosshairPosition(timestamp=1700000000.0, price=150.25)
    pos2 = CrosshairPosition(timestamp=1700000000.0, price=150.25)
    pos3 = CrosshairPosition(timestamp=1700000000.0, price=None)

    assert pos1 == pos2
    assert pos1.timestamp == 1700000000.0
    assert pos1.price == 150.25
    assert pos3.price is None


# ==============================================================================
# 2. Crosshair Synchronization Tests (AC 1)
# ==============================================================================


def test_crosshair_broadcasts_to_peers_without_echo(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """
    Given multiple charts registered in the same synchronization group with crosshair sync enabled,
    When one chart updates its crosshair position,
    Then the synchronizer broadcasts to all other charts in the group without echoing to the source chart.
    """
    group_id = "group-1"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
        sync_crosshair=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
        sync_crosshair=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_crosshair=chart_c.on_crosshair,
        sync_crosshair=True,
    )

    expected_position = CrosshairPosition(timestamp=1680000000, price=105.50)
    synchronizer.update_crosshair(
        group_id=group_id,
        source_chart_id=chart_a.chart_id,
        position=expected_position,
    )

    # Source chart MUST NOT receive an echo update
    assert len(chart_a.crosshair_events) == 0

    # Peer charts MUST receive the exact crosshair update
    assert len(chart_b.crosshair_events) == 1
    assert chart_b.crosshair_events[0] == expected_position

    assert len(chart_c.crosshair_events) == 1
    assert chart_c.crosshair_events[0] == expected_position


def test_crosshair_sync_disabled_on_target_chart_suppresses_receipt(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """
    Given Chart B has sync_crosshair=False,
    When Chart A updates crosshair,
    Then Chart C receives update, but Chart B does not.
    """
    group_id = "group-alpha"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
        sync_crosshair=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
        sync_crosshair=False,  # Disabled
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_crosshair=chart_c.on_crosshair,
        sync_crosshair=True,
    )

    pos = CrosshairPosition(timestamp=1690000000, price=200.0)
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos)

    assert len(chart_a.crosshair_events) == 0
    assert len(chart_b.crosshair_events) == 0
    assert len(chart_c.crosshair_events) == 1
    assert chart_c.crosshair_events[0] == pos


def test_crosshair_sync_isolated_between_groups(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """Verify crosshair updates are strictly isolated within their registered group."""
    # Chart A & B in group-1; Chart C in group-2
    synchronizer.register_chart(
        group_id="group-1",
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
    )
    synchronizer.register_chart(
        group_id="group-1",
        chart_id=chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
    )
    synchronizer.register_chart(
        group_id="group-2",
        chart_id=chart_c.chart_id,
        on_crosshair=chart_c.on_crosshair,
    )

    pos = CrosshairPosition(timestamp=1680000000, price=50.0)
    synchronizer.update_crosshair("group-1", chart_a.chart_id, pos)

    assert len(chart_b.crosshair_events) == 1
    assert len(chart_c.crosshair_events) == 0


def test_crosshair_sync_allows_none_price_for_time_only_marker(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy
):
    """Verify that a crosshair event with price=None is safely broadcast."""
    group_id = "group-time-only"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
    )

    pos = CrosshairPosition(timestamp=1680000000, price=None)
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos)

    assert len(chart_b.crosshair_events) == 1
    assert chart_b.crosshair_events[0].price is None
    assert chart_b.crosshair_events[0].timestamp == 1680000000


# ==============================================================================
# 3. Symbol Synchronization Tests (AC 2)
# ==============================================================================


def test_symbol_sync_broadcasts_to_peers_without_echo(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """
    Given multiple charts registered in the same synchronization group with symbol sync enabled,
    When a chart's symbol changes,
    Then the synchronizer notifies all peer charts in the group to switch to the new symbol without echoing.
    """
    group_id = "symbols-group"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_symbol=chart_a.on_symbol,
        sync_symbol=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_symbol=chart_b.on_symbol,
        sync_symbol=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_symbol=chart_c.on_symbol,
        sync_symbol=True,
    )

    new_symbol = "BTC/USDT"
    synchronizer.update_symbol(
        group_id=group_id,
        source_chart_id=chart_a.chart_id,
        symbol=new_symbol,
    )

    assert len(chart_a.symbol_events) == 0
    assert chart_b.symbol_events == ["BTC/USDT"]
    assert chart_c.symbol_events == ["BTC/USDT"]


def test_symbol_sync_disabled_on_chart_suppresses_notification(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """Verify that a chart with sync_symbol=False does not receive symbol updates."""
    group_id = "symbols-group"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_symbol=chart_a.on_symbol,
        sync_symbol=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_symbol=chart_b.on_symbol,
        sync_symbol=False,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_symbol=chart_c.on_symbol,
        sync_symbol=True,
    )

    synchronizer.update_symbol(group_id, chart_a.chart_id, "ETH/USD")

    assert len(chart_a.symbol_events) == 0
    assert len(chart_b.symbol_events) == 0
    assert chart_c.symbol_events == ["ETH/USD"]


def test_symbol_sync_group_isolation(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy
):
    """Verify symbol updates in group-1 do not leak to group-2."""
    synchronizer.register_chart(
        group_id="group-1", chart_id=chart_a.chart_id, on_symbol=chart_a.on_symbol
    )
    synchronizer.register_chart(
        group_id="group-2", chart_id=chart_b.chart_id, on_symbol=chart_b.on_symbol
    )

    synchronizer.update_symbol("group-1", chart_a.chart_id, "AAPL")

    assert len(chart_a.symbol_events) == 0
    assert len(chart_b.symbol_events) == 0


def test_symbol_sync_rejects_empty_symbol(
    synchronizer: SyncManager, chart_a: ChartSpy
):
    """Verify symbol update validates the symbol string."""
    group_id = "group-1"
    synchronizer.register_chart(group_id, chart_a.chart_id, on_symbol=chart_a.on_symbol)

    with pytest.raises(ValueError):
        synchronizer.update_symbol(group_id, chart_a.chart_id, "")

    with pytest.raises(ValueError):
        synchronizer.update_symbol(group_id, chart_a.chart_id, "   ")


# ==============================================================================
# 4. Interval Synchronization Tests (AC 3)
# ==============================================================================


def test_interval_sync_propagates_to_peers_without_echo(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """
    Given multiple charts registered in the same synchronization group with interval sync enabled,
    When a chart's interval changes,
    Then the synchronizer propagates the new interval to all peer charts in the group.
    """
    group_id = "intervals-group"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_interval=chart_a.on_interval,
        sync_interval=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_interval=chart_b.on_interval,
        sync_interval=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_interval=chart_c.on_interval,
        sync_interval=True,
    )

    new_interval = "15m"
    synchronizer.update_interval(
        group_id=group_id,
        source_chart_id=chart_a.chart_id,
        interval=new_interval,
    )

    assert len(chart_a.interval_events) == 0
    assert chart_b.interval_events == ["15m"]
    assert chart_c.interval_events == ["15m"]


def test_interval_sync_disabled_on_chart_suppresses_propagation(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """Verify that a chart with sync_interval=False does not receive interval updates."""
    group_id = "intervals-group"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_interval=chart_a.on_interval,
        sync_interval=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_interval=chart_b.on_interval,
        sync_interval=False,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_interval=chart_c.on_interval,
        sync_interval=True,
    )

    synchronizer.update_interval(group_id, chart_a.chart_id, "1h")

    assert len(chart_a.interval_events) == 0
    assert len(chart_b.interval_events) == 0
    assert chart_c.interval_events == ["1h"]


def test_interval_sync_group_isolation(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy
):
    """Verify interval updates in group-1 do not leak to group-2."""
    synchronizer.register_chart(
        group_id="group-1", chart_id=chart_a.chart_id, on_interval=chart_a.on_interval
    )
    synchronizer.register_chart(
        group_id="group-2", chart_id=chart_b.chart_id, on_interval=chart_b.on_interval
    )

    synchronizer.update_interval("group-1", chart_a.chart_id, "4h")

    assert len(chart_a.interval_events) == 0
    assert len(chart_b.interval_events) == 0


def test_interval_sync_rejects_empty_interval(
    synchronizer: SyncManager, chart_a: ChartSpy
):
    """Verify interval update validates the interval string."""
    group_id = "group-1"
    synchronizer.register_chart(group_id, chart_a.chart_id, on_interval=chart_a.on_interval)

    with pytest.raises(ValueError):
        synchronizer.update_interval(group_id, chart_a.chart_id, "")

    with pytest.raises(ValueError):
        synchronizer.update_interval(group_id, chart_a.chart_id, "   ")


# ==============================================================================
# 5. Registration, Unregistration, and Lifecycle Management Tests
# ==============================================================================


def test_duplicate_chart_registration_raises_error(
    synchronizer: SyncManager, chart_a: ChartSpy
):
    """Registering the same chart_id in the same group twice must raise ValueError."""
    group_id = "group-dup"
    synchronizer.register_chart(group_id, chart_a.chart_id)

    with pytest.raises(ValueError):
        synchronizer.register_chart(group_id, chart_a.chart_id)


def test_unregistered_chart_stops_receiving_any_synchronization(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """Verify that unregistering a chart removes it from all subsequent broadcasts."""
    group_id = "group-lifecycle"
    synchronizer.register_chart(
        group_id, chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
        on_symbol=chart_a.on_symbol,
        on_interval=chart_a.on_interval,
    )
    synchronizer.register_chart(
        group_id, chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
        on_symbol=chart_b.on_symbol,
        on_interval=chart_b.on_interval,
    )
    synchronizer.register_chart(
        group_id, chart_c.chart_id,
        on_crosshair=chart_c.on_crosshair,
        on_symbol=chart_c.on_symbol,
        on_interval=chart_c.on_interval,
    )

    # Unregister chart B
    synchronizer.unregister_chart(group_id=group_id, chart_id=chart_b.chart_id)

    # Perform updates from chart A
    synchronizer.update_crosshair(group_id, chart_a.chart_id, CrosshairPosition(100, 20.0))
    synchronizer.update_symbol(group_id, chart_a.chart_id, "SOL/USDT")
    synchronizer.update_interval(group_id, chart_a.chart_id, "5m")

    # Chart B must not have received anything
    assert len(chart_b.crosshair_events) == 0
    assert len(chart_b.symbol_events) == 0
    assert len(chart_b.interval_events) == 0

    # Chart C must receive all events
    assert len(chart_c.crosshair_events) == 1
    assert chart_c.symbol_events == ["SOL/USDT"]
    assert chart_c.interval_events == ["5m"]


def test_unregister_nonexistent_chart_raises_error(synchronizer: SyncManager):
    """Unregistering a chart that does not exist in the group raises KeyError or ValueError."""
    with pytest.raises((ValueError, KeyError)):
        synchronizer.unregister_chart(group_id="nonexistent-group", chart_id="unknown-chart")


def test_update_from_unregistered_chart_raises_error(
    synchronizer: SyncManager, chart_a: ChartSpy
):
    """An update initiated by a chart not registered in the group raises KeyError or ValueError."""
    group_id = "group-1"
    synchronizer.register_chart(group_id, chart_a.chart_id)

    pos = CrosshairPosition(timestamp=100, price=10.0)

    with pytest.raises((ValueError, KeyError)):
        synchronizer.update_crosshair(group_id, "unknown-source", pos)

    with pytest.raises((ValueError, KeyError)):
        synchronizer.update_symbol(group_id, "unknown-source", "BTC")

    with pytest.raises((ValueError, KeyError)):
        synchronizer.update_interval(group_id, "unknown-source", "1d")


def test_dynamic_sync_flag_modification(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy
):
    """Verify sync flags can be altered dynamically for a registered chart."""
    group_id = "group-dynamic"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
        sync_crosshair=True,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_b.chart_id,
        on_crosshair=chart_b.on_crosshair,
        sync_crosshair=True,
    )

    pos1 = CrosshairPosition(timestamp=100, price=10.0)
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos1)
    assert len(chart_b.crosshair_events) == 1

    # Disable crosshair sync dynamically for chart B
    synchronizer.set_sync_flags(group_id, chart_b.chart_id, sync_crosshair=False)

    pos2 = CrosshairPosition(timestamp=200, price=20.0)
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos2)
    # chart B event count must remain 1
    assert len(chart_b.crosshair_events) == 1

    # Re-enable crosshair sync
    synchronizer.set_sync_flags(group_id, chart_b.chart_id, sync_crosshair=True)

    pos3 = CrosshairPosition(timestamp=300, price=30.0)
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos3)
    assert len(chart_b.crosshair_events) == 2
    assert chart_b.crosshair_events[1] == pos3


def test_get_charts_in_group(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy
):
    """Verify group querying reflects registrations and unregistrations."""
    group_id = "inspection-group"
    synchronizer.register_chart(group_id, chart_a.chart_id)
    synchronizer.register_chart(group_id, chart_b.chart_id)

    charts = synchronizer.get_charts_in_group(group_id)
    assert set(charts) == {chart_a.chart_id, chart_b.chart_id}

    synchronizer.unregister_chart(group_id, chart_a.chart_id)
    charts_after = synchronizer.get_charts_in_group(group_id)
    assert set(charts_after) == {chart_b.chart_id}


def test_chart_participating_in_multiple_groups(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_b: ChartSpy, chart_c: ChartSpy
):
    """
    Verify a single chart can participate in two different groups,
    acting as a bridge only when explicitly updating.
    """
    # Chart B is in both group-1 and group-2
    synchronizer.register_chart("group-1", chart_a.chart_id, on_symbol=chart_a.on_symbol)
    synchronizer.register_chart("group-1", chart_b.chart_id, on_symbol=chart_b.on_symbol)

    synchronizer.register_chart("group-2", chart_b.chart_id, on_symbol=chart_b.on_symbol)
    synchronizer.register_chart("group-2", chart_c.chart_id, on_symbol=chart_c.on_symbol)

    # Chart A updates group-1
    synchronizer.update_symbol("group-1", chart_a.chart_id, "ETH/USDT")
    assert chart_b.symbol_events == ["ETH/USDT"]
    assert len(chart_c.symbol_events) == 0

    # Chart C updates group-2
    synchronizer.update_symbol("group-2", chart_c.chart_id, "ADA/USDT")
    assert chart_b.symbol_events == ["ETH/USDT", "ADA/USDT"]
    assert chart_a.symbol_events == []


# ==============================================================================
# 6. Resilience and Edge Cases
# ==============================================================================


def test_callback_exception_does_not_halt_broadcast_to_remaining_peers(
    synchronizer: SyncManager, chart_a: ChartSpy, chart_c: ChartSpy
):
    """
    If one chart's callback raises an unexpected exception,
    other peer charts in the group should still receive the broadcast.
    """
    group_id = "fault-tolerant-group"

    def faulty_crosshair_callback(pos: CrosshairPosition) -> None:
        raise RuntimeError("Unexpected chart rendering error")

    synchronizer.register_chart(
        group_id=group_id,
        chart_id="faulty-chart",
        on_crosshair=faulty_crosshair_callback,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=chart_a.on_crosshair,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_c.chart_id,
        on_crosshair=chart_c.on_crosshair,
    )

    pos = CrosshairPosition(timestamp=5000, price=350.0)

    # Update from chart_a. Faulty chart will throw, but chart_c must receive it.
    synchronizer.update_crosshair(group_id, chart_a.chart_id, pos)

    assert len(chart_c.crosshair_events) == 1
    assert chart_c.crosshair_events[0] == pos


def test_none_callbacks_handled_gracefully(
    synchronizer: SyncManager, chart_a: ChartSpy
):
    """Registering a chart without providing callbacks should not cause errors on sync events."""
    group_id = "group-no-callbacks"
    synchronizer.register_chart(
        group_id=group_id,
        chart_id=chart_a.chart_id,
        on_crosshair=None,
        on_symbol=None,
        on_interval=None,
    )
    synchronizer.register_chart(
        group_id=group_id,
        chart_id="chart-passive",
        on_crosshair=None,
        on_symbol=None,
        on_interval=None,
    )

    # Should execute cleanly without raising AttributeError/TypeError
    synchronizer.update_crosshair(group_id, chart_a.chart_id, CrosshairPosition(1, 1.0))
    synchronizer.update_symbol(group_id, chart_a.chart_id, "BTC")
    synchronizer.update_interval(group_id, chart_a.chart_id, "1m")