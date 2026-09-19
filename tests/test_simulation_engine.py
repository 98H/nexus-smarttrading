"""
Unit tests for deterministic event-driven bar and tick simulator.

Specification:
Story 6.1.1: Build Deterministic Event-Driven Bar and Tick Simulator
- AC 1: Dispatched strictly in deterministic, non-decreasing timestamp order.
- AC 2: Identical timestamps processed deterministically based on monotonic sequence identifier.
- AC 3: Registered bar and tick handlers invoke corresponding callbacks without dropping or altering payloads.
"""

from typing import List, Union
import pytest

from src.simulation.events import BarEvent, Event, TickEvent
from src.simulation.engine import SimulationEngine


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def engine() -> SimulationEngine:
    """Provides a fresh SimulationEngine instance for each test."""
    return SimulationEngine()


# -----------------------------------------------------------------------------
# Acceptance Criteria 1: Deterministic Non-Decreasing Timestamp Order
# -----------------------------------------------------------------------------


def test_events_dispatched_in_strictly_non_decreasing_timestamp_order(
    engine: SimulationEngine,
) -> None:
    """AC 1: Events loaded out-of-order are dispatched in non-decreasing timestamp order."""
    dispatched_timestamps: List[int] = []

    engine.register_tick_handler(lambda t: dispatched_timestamps.append(t.timestamp))
    engine.register_bar_handler(lambda b: dispatched_timestamps.append(b.timestamp))

    # Load unordered timestamps
    raw_events = [
        TickEvent(timestamp=300, symbol="AAPL", price=150.5, volume=100),
        BarEvent(
            timestamp=100,
            symbol="AAPL",
            open=149.0,
            high=150.0,
            low=148.5,
            close=149.5,
            volume=1000,
        ),
        TickEvent(timestamp=500, symbol="AAPL", price=151.0, volume=200),
        BarEvent(
            timestamp=200,
            symbol="AAPL",
            open=149.5,
            high=150.5,
            low=149.0,
            close=150.2,
            volume=1500,
        ),
        TickEvent(timestamp=400, symbol="AAPL", price=150.8, volume=50),
    ]

    for event in raw_events:
        engine.load_event(event)

    engine.run()

    assert dispatched_timestamps == [100, 200, 300, 400, 500]


def test_batch_load_events_sorted_chronologically(engine: SimulationEngine) -> None:
    """AC 1: Batch loading events dispatches strictly in ascending timestamp order."""
    dispatched_events: List[Event] = []

    def record_event(e: Event) -> None:
        dispatched_events.append(e)

    engine.register_tick_handler(record_event)
    engine.register_bar_handler(record_event)

    events: List[Union[TickEvent, BarEvent]] = [
        TickEvent(timestamp=1050, symbol="MSFT", price=310.0, volume=10),
        BarEvent(
            timestamp=1010,
            symbol="MSFT",
            open=308.0,
            high=309.5,
            low=307.5,
            close=309.0,
            volume=500,
        ),
        TickEvent(timestamp=1000, symbol="MSFT", price=308.5, volume=15),
        BarEvent(
            timestamp=1040,
            symbol="MSFT",
            open=309.0,
            high=310.2,
            low=308.8,
            close=310.0,
            volume=800,
        ),
    ]

    engine.load_events(events)
    engine.run()

    timestamps = [e.timestamp for e in dispatched_events]
    assert timestamps == sorted(timestamps)
    assert timestamps == [1000, 1010, 1040, 1050]


# -----------------------------------------------------------------------------
# Acceptance Criteria 2: Identical Timestamps and Monotonic Sequence Identifiers
# -----------------------------------------------------------------------------


def test_identical_timestamps_resolved_by_explicit_sequence_id(
    engine: SimulationEngine,
) -> None:
    """AC 2: Events with identical timestamps are ordered by monotonic sequence identifier."""
    dispatched_seq_ids: List[int] = []

    engine.register_tick_handler(lambda t: dispatched_seq_ids.append(t.sequence_id))
    engine.register_bar_handler(lambda b: dispatched_seq_ids.append(b.sequence_id))

    # All events have the same timestamp but out-of-order sequence IDs
    same_ts = 1_000_000
    events = [
        TickEvent(
            timestamp=same_ts,
            sequence_id=3,
            symbol="SPY",
            price=450.3,
            volume=100,
        ),
        BarEvent(
            timestamp=same_ts,
            sequence_id=1,
            symbol="SPY",
            open=450.0,
            high=450.5,
            low=449.8,
            close=450.2,
            volume=5000,
        ),
        TickEvent(
            timestamp=same_ts,
            sequence_id=4,
            symbol="SPY",
            price=450.4,
            volume=200,
        ),
        BarEvent(
            timestamp=same_ts,
            sequence_id=2,
            symbol="SPY",
            open=450.1,
            high=450.3,
            low=449.9,
            close=450.1,
            volume=3000,
        ),
    ]

    for event in events:
        engine.load_event(event)

    engine.run()

    assert dispatched_seq_ids == [1, 2, 3, 4]


def test_engine_assigns_monotonic_sequence_ids_on_insertion_when_omitted(
    engine: SimulationEngine,
) -> None:
    """AC 2: When sequence IDs are equal or default, the engine maintains insertion order deterministically."""
    dispatched_symbols: List[str] = []

    engine.register_tick_handler(lambda t: dispatched_symbols.append(t.symbol))

    # Inserting multiple ticks with identical timestamps and no explicit differing sequence_id
    engine.load_event(
        TickEvent(timestamp=100, sequence_id=0, symbol="TICK_A", price=10.0, volume=1)
    )
    engine.load_event(
        TickEvent(timestamp=100, sequence_id=0, symbol="TICK_B", price=20.0, volume=2)
    )
    engine.load_event(
        TickEvent(timestamp=100, sequence_id=0, symbol="TICK_C", price=30.0, volume=3)
    )

    engine.run()

    assert dispatched_symbols == ["TICK_A", "TICK_B", "TICK_C"]


def test_composite_ordering_timestamp_then_sequence_id(
    engine: SimulationEngine,
) -> None:
    """AC 2: Verification that timestamp takes precedence, sequence identifier breaks ties."""
    execution_order: List[str] = []

    engine.register_tick_handler(lambda t: execution_order.append(t.symbol))
    engine.register_bar_handler(lambda b: execution_order.append(b.symbol))

    events = [
        TickEvent(timestamp=200, sequence_id=1, symbol="T200_S1", price=1.0, volume=1),
        TickEvent(timestamp=100, sequence_id=2, symbol="T100_S2", price=1.0, volume=1),
        BarEvent(
            timestamp=100,
            sequence_id=1,
            symbol="T100_S1",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
        ),
        BarEvent(
            timestamp=200,
            sequence_id=2,
            symbol="T200_S2",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
        ),
    ]

    engine.load_events(events)
    engine.run()

    assert execution_order == ["T100_S1", "T100_S2", "T200_S1", "T200_S2"]


# -----------------------------------------------------------------------------
# Acceptance Criteria 3: Route to Respective Handlers and Preserve Payload
# -----------------------------------------------------------------------------


def test_tick_and_bar_handlers_called_for_exact_event_type_only(
    engine: SimulationEngine,
) -> None:
    """AC 3: Tick events invoke tick callbacks; Bar events invoke bar callbacks without cross-dispatch."""
    received_ticks: List[TickEvent] = []
    received_bars: List[BarEvent] = []

    engine.register_tick_handler(lambda t: received_ticks.append(t))
    engine.register_bar_handler(lambda b: received_bars.append(b))

    tick = TickEvent(
        timestamp=10, sequence_id=1, symbol="NVDA", price=450.0, volume=25
    )
    bar = BarEvent(
        timestamp=20,
        sequence_id=2,
        symbol="NVDA",
        open=449.0,
        high=451.0,
        low=448.5,
        close=450.5,
        volume=1000,
    )

    engine.load_events([tick, bar])
    engine.run()

    assert len(received_ticks) == 1
    assert received_ticks[0] is tick
    assert isinstance(received_ticks[0], TickEvent)

    assert len(received_bars) == 1
    assert received_bars[0] is bar
    assert isinstance(received_bars[0], BarEvent)


def test_payload_integrity_is_strictly_preserved(engine: SimulationEngine) -> None:
    """AC 3: Ensure event payload attributes are not dropped, mutated, or truncated."""
    captured_ticks: List[TickEvent] = []
    captured_bars: List[BarEvent] = []

    engine.register_tick_handler(lambda t: captured_ticks.append(t))
    engine.register_bar_handler(lambda b: captured_bars.append(b))

    orig_tick = TickEvent(
        timestamp=1690000000123,
        sequence_id=42,
        symbol="TSLA",
        price=260.123456,
        volume=150.75,
    )

    orig_bar = BarEvent(
        timestamp=1690000000456,
        sequence_id=43,
        symbol="GOOGL",
        open=130.10,
        high=132.45,
        low=129.90,
        close=131.80,
        volume=250000.0,
    )

    engine.load_event(orig_tick)
    engine.load_event(orig_bar)
    engine.run()

    # Verify Tick fields
    dispatched_tick = captured_ticks[0]
    assert dispatched_tick.timestamp == 1690000000123
    assert dispatched_tick.sequence_id == 42
    assert dispatched_tick.symbol == "TSLA"
    assert dispatched_tick.price == 260.123456
    assert dispatched_tick.volume == 150.75

    # Verify Bar fields
    dispatched_bar = captured_bars[0]
    assert dispatched_bar.timestamp == 1690000000456
    assert dispatched_bar.sequence_id == 43
    assert dispatched_bar.symbol == "GOOGL"
    assert dispatched_bar.open == 130.10
    assert dispatched_bar.high == 132.45
    assert dispatched_bar.low == 129.90
    assert dispatched_bar.close == 131.80
    assert dispatched_bar.volume == 250000.0


def test_no_events_are_dropped_with_multiple_handlers(
    engine: SimulationEngine,
) -> None:
    """AC 3: Multiple handlers receive all events without any event drop."""
    handler_1_events: List[Event] = []
    handler_2_events: List[Event] = []

    engine.register_tick_handler(lambda t: handler_1_events.append(t))
    engine.register_tick_handler(lambda t: handler_2_events.append(t))

    total_events = 1000
    for i in range(total_events):
        engine.load_event(
            TickEvent(
                timestamp=i,
                sequence_id=i,
                symbol="BTC/USD",
                price=30000.0 + i,
                volume=1.5,
            )
        )

    engine.run()

    assert len(handler_1_events) == total_events
    assert len(handler_2_events) == total_events
    assert [e.sequence_id for e in handler_1_events] == list(range(total_events))
    assert [e.sequence_id for e in handler_2_events] == list(range(total_events))


# -----------------------------------------------------------------------------
# Edge Cases and Defensive Boundaries
# -----------------------------------------------------------------------------


def test_empty_engine_run_does_not_fail(engine: SimulationEngine) -> None:
    """Empty queue run should complete gracefully without invoking handlers."""
    invoked = False

    def dummy_handler(_: Event) -> None:
        nonlocal invoked
        invoked = True

    engine.register_tick_handler(dummy_handler)
    engine.register_bar_handler(dummy_handler)

    engine.run()
    assert not invoked


def test_queue_drained_after_run(engine: SimulationEngine) -> None:
    """Running the engine consumes events; subsequent run should not redispatch old events."""
    counter = 0

    def increment(_: TickEvent) -> None:
        nonlocal counter
        counter += 1

    engine.register_tick_handler(increment)
    engine.load_event(
        TickEvent(timestamp=100, sequence_id=1, symbol="X", price=1.0, volume=1)
    )

    engine.run()
    assert counter == 1

    # Second run without new events
    engine.run()
    assert counter == 1


def test_invalid_handler_registration_raises_exception(
    engine: SimulationEngine,
) -> None:
    """Registering a non-callable handler raises a TypeError."""
    with pytest.raises(TypeError):
        engine.register_tick_handler("not_callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        engine.register_bar_handler(12345)  # type: ignore[arg-type]


def test_loading_unsupported_event_type_raises_exception(
    engine: SimulationEngine,
) -> None:
    """Loading invalid non-event objects into the engine queue raises a TypeError."""
    with pytest.raises(TypeError):
        engine.load_event({"timestamp": 100, "price": 10})  # type: ignore[arg-type]


def test_simulation_handles_interleaved_large_scale_determinism(
    engine: SimulationEngine,
) -> None:
    """Stress test: 500 interleaved ticks and bars with duplicate timestamps dispatch deterministically."""
    results: List[int] = []

    engine.register_tick_handler(lambda t: results.append(t.sequence_id))
    engine.register_bar_handler(lambda b: results.append(b.sequence_id))

    expected_order: List[int] = []
    events: List[Union[TickEvent, BarEvent]] = []

    # 10 unique timestamps, each containing 10 events (5 ticks, 5 bars) with sequence_ids 0..9
    global_seq = 0
    for ts in range(10, 20):
        for seq in range(10):
            global_seq += 1
            expected_order.append(global_seq)
            if seq % 2 == 0:
                events.append(
                    TickEvent(
                        timestamp=ts,
                        sequence_id=global_seq,
                        symbol="TEST",
                        price=100.0,
                        volume=10,
                    )
                )
            else:
                events.append(
                    BarEvent(
                        timestamp=ts,
                        sequence_id=global_seq,
                        symbol="TEST",
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=100,
                    )
                )

    # Shuffle loading order to test priority sorting
    import random

    rng = random.Random(42)
    shuffled_events = events.copy()
    rng.shuffle(shuffled_events)

    engine.load_events(shuffled_events)
    engine.run()

    assert results == expected_order