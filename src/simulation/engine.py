"""Deterministic event-driven simulation engine."""

import heapq
from typing import Callable, Iterable, List, Tuple

from src.simulation.events import BarEvent, Event, TickEvent

TickHandler = Callable[[TickEvent], None]
BarHandler = Callable[[BarEvent], None]


class SimulationEngine:
    """Simulation engine ensuring deterministic ordering and dispatch of events."""

    def __init__(self) -> None:
        self._queue: List[Tuple[int, int, int, Event]] = []
        self._insertion_counter: int = 0
        self._tick_handlers: List[TickHandler] = []
        self._bar_handlers: List[BarHandler] = []

    def register_tick_handler(self, handler: TickHandler) -> None:
        """Register a callback handler for tick events."""
        if not callable(handler):
            raise TypeError(
                f"Tick handler must be callable, got {type(handler).__name__}"
            )
        self._tick_handlers.append(handler)

    def register_bar_handler(self, handler: BarHandler) -> None:
        """Register a callback handler for bar events."""
        if not callable(handler):
            raise TypeError(
                f"Bar handler must be callable, got {type(handler).__name__}"
            )
        self._bar_handlers.append(handler)

    def load_event(self, event: Event) -> None:
        """Load a single event into the simulator queue."""
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event instance, got {type(event).__name__}")
        self._insertion_counter += 1
        seq_id = getattr(event, "sequence_id", 0)
        if seq_id is None:
            seq_id = 0
        heapq.heappush(
            self._queue,
            (event.timestamp, seq_id, self._insertion_counter, event),
        )

    def load_events(self, events: Iterable[Event]) -> None:
        """Load multiple events into the simulator queue."""
        for event in events:
            self.load_event(event)

    def run(self) -> None:
        """Execute simulation, dispatching queued events in deterministic order."""
        while self._queue:
            _, _, _, event = heapq.heappop(self._queue)
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Route event to registered handlers corresponding to its event type."""
        if isinstance(event, TickEvent):
            for handler in self._tick_handlers:
                handler(event)
        elif isinstance(event, BarEvent):
            for handler in self._bar_handlers:
                handler(event)