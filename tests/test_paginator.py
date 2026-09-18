"""
Unit tests for the Backward-Seeking Historical Data Paginator.

Specification:
Story 2.3.1: Build Backward-Seeking Historical Data Paginator
Acceptance Criteria:
- Given a target start_time, end_time, and a fetch callback returning descending timestamped records,
  When the backward-seeking paginator is iterated,
  Then it queries records starting from end_time backward, updating the cursor to the oldest record's
  timestamp from each batch until reaching start_time or until no further records are returned.
- Given a returned batch that crosses or reaches start_time,
  When the paginator processes the batch,
  Then it truncates records older than start_time and terminates subsequent queries.
- Given an empty response from the underlying fetch function,
  When seeking backwards,
  Then the paginator stops pagination gracefully without raising an exception.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import pytest

from src.data.paginator import BackwardSeekingPaginator


class FetchCallbackSpy:
    """
    Deterministic test spy simulating an API returning descending timestamped records.
    Records cursor history and returns configured batches per cursor.
    """

    def __init__(self, responses: Dict[Any, List[Any]]):
        self.responses = responses
        self.queried_cursors: List[Any] = []

    def __call__(self, cursor: Optional[Any] = None, *args: Any, **kwargs: Any) -> List[Any]:
        # Handle positional cursor or keyword-based cursor/end_time passing
        resolved_cursor = cursor
        if resolved_cursor is None:
            if "cursor" in kwargs:
                resolved_cursor = kwargs["cursor"]
            elif "end_time" in kwargs:
                resolved_cursor = kwargs["end_time"]
            elif args:
                resolved_cursor = args[0]

        self.queried_cursors.append(resolved_cursor)
        return self.responses.get(resolved_cursor, [])


@dataclass
class CandleRecord:
    timestamp: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Acceptance Criteria 1 Tests: Cursor traversal from end_time backward
# ---------------------------------------------------------------------------


def test_queries_start_from_end_time_and_update_cursor_to_oldest_record():
    """
    AC 1: Queries start at end_time, then update cursor to the oldest record's timestamp
    from each batch across multiple pages until start_time is reached.
    """
    start_time = 100
    end_time = 500

    # Batch 1 oldest is 380, Batch 2 oldest is 240, Batch 3 reaches start_time 100
    responses = {
        500: [
            {"timestamp": 480, "price": 101.5},
            {"timestamp": 440, "price": 101.0},
            {"timestamp": 380, "price": 100.5},
        ],
        380: [
            {"timestamp": 320, "price": 100.0},
            {"timestamp": 290, "price": 99.5},
            {"timestamp": 240, "price": 99.0},
        ],
        240: [
            {"timestamp": 180, "price": 98.5},
            {"timestamp": 130, "price": 98.0},
            {"timestamp": 100, "price": 97.5},
        ],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    # Must query starting from end_time (500), then oldest of batch 1 (380), then oldest of batch 2 (240)
    assert spy.queried_cursors == [500, 380, 240]

    # Verify all records are yielded in order
    expected_timestamps = [480, 440, 380, 320, 290, 240, 180, 130, 100]
    assert [r["timestamp"] for r in records] == expected_timestamps


def test_terminates_when_no_further_records_returned_before_reaching_start_time():
    """
    AC 1: When fewer records exist than the requested time window, pagination terminates
    cleanly once no further records are returned by the fetch callback.
    """
    start_time = 100
    end_time = 400

    responses = {
        400: [
            {"timestamp": 350, "val": 1},
            {"timestamp": 300, "val": 2},
        ],
        300: [],  # Upstream has no older records available
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert spy.queried_cursors == [400, 300]
    assert [r["timestamp"] for r in records] == [350, 300]


# ---------------------------------------------------------------------------
# Acceptance Criteria 2 Tests: Truncation at start_time and termination
# ---------------------------------------------------------------------------


def test_truncates_records_older_than_start_time_and_terminates():
    """
    AC 2: When a batch crosses start_time, records older than start_time are truncated,
    and subsequent queries are terminated.
    """
    start_time = 200
    end_time = 500

    responses = {
        500: [
            {"timestamp": 450, "val": 1},
            {"timestamp": 350, "val": 2},
        ],
        350: [
            {"timestamp": 280, "val": 3},
            {"timestamp": 210, "val": 4},
            {"timestamp": 190, "val": 5},  # Older than start_time (200) -> must truncate
            {"timestamp": 120, "val": 6},  # Older than start_time (200) -> must truncate
        ],
        # If paginator made an extra call, it would be for cursor 120 or 190, which must NOT happen
        120: [{"timestamp": 90, "val": 7}],
        190: [{"timestamp": 90, "val": 7}],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    # Verify truncated records (190, 120) are excluded
    assert [r["timestamp"] for r in records] == [450, 350, 280, 210]

    # Verify no query was made after crossing start_time
    assert spy.queried_cursors == [500, 350]


def test_batch_reaching_start_time_exactly_is_not_truncated_and_terminates():
    """
    AC 2: When the oldest record in a batch exactly reaches start_time, that record is kept,
    and subsequent queries are terminated without issuing extra fetches.
    """
    start_time = 150
    end_time = 300

    responses = {
        300: [
            {"timestamp": 250, "val": 1},
            {"timestamp": 200, "val": 2},
            {"timestamp": 150, "val": 3},  # Exactly reaches start_time
        ],
        150: [
            {"timestamp": 100, "val": 4},  # Must NOT be queried
        ],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    # 150 is retained, exactly 1 query is made
    assert [r["timestamp"] for r in records] == [250, 200, 150]
    assert spy.queried_cursors == [300]


def test_batch_where_all_records_are_older_than_start_time_is_fully_truncated():
    """
    AC 2: If an entire batch contains records strictly older than start_time, all records
    are truncated and iteration terminates immediately.
    """
    start_time = 300
    end_time = 400

    responses = {
        400: [
            {"timestamp": 250, "val": 1},
            {"timestamp": 200, "val": 2},
        ]
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert records == []
    assert spy.queried_cursors == [400]


# ---------------------------------------------------------------------------
# Acceptance Criteria 3 Tests: Graceful handling of empty responses
# ---------------------------------------------------------------------------


def test_empty_initial_response_stops_gracefully_without_exception():
    """
    AC 3: Given an empty response on the initial query, the paginator stops pagination
    gracefully yielding an empty sequence without raising any exceptions.
    """
    spy = FetchCallbackSpy(responses={1000: []})

    paginator = BackwardSeekingPaginator(
        start_time=500,
        end_time=1000,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert records == []
    assert spy.queried_cursors == [1000]


def test_empty_response_on_subsequent_batch_stops_gracefully():
    """
    AC 3: Given an empty response on a subsequent query, the paginator terminates
    gracefully without exception, returning only records fetched so far.
    """
    responses = {
        1000: [{"timestamp": 900, "data": "A"}],
        900: [],
    }
    spy = FetchCallbackSpy(responses=responses)

    paginator = BackwardSeekingPaginator(
        start_time=100,
        end_time=1000,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert [r["timestamp"] for r in records] == [900]
    assert spy.queried_cursors == [1000, 900]


# ---------------------------------------------------------------------------
# Robustness, Edge Cases, and Data Type Support Tests
# ---------------------------------------------------------------------------


def test_lazy_iteration_does_not_fetch_all_batches_upfront():
    """
    Verify streaming/lazy behavior: iterating stops querying once the consumer
    breaks out of iteration, avoiding unnecessary network/fetch calls.
    """
    responses = {
        1000: [{"timestamp": 900}, {"timestamp": 800}],
        800: [{"timestamp": 700}, {"timestamp": 600}],
        600: [{"timestamp": 500}, {"timestamp": 400}],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=100,
        end_time=1000,
        fetch_callback=spy,
    )

    iterator = iter(paginator)
    first_record = next(iterator)

    assert first_record["timestamp"] == 900
    # Only the first batch should have been fetched to yield the first item
    assert spy.queried_cursors == [1000]


def test_supports_object_records_with_timestamp_attribute():
    """
    Verify support for records represented as dataclasses/objects having a .timestamp attribute.
    """
    start_time = 100
    end_time = 300

    responses = {
        300: [
            CandleRecord(timestamp=250, open=1.0, high=2.0, low=0.5, close=1.5),
            CandleRecord(timestamp=180, open=1.5, high=2.5, low=1.0, close=2.0),
        ],
        180: [
            CandleRecord(timestamp=120, open=2.0, high=3.0, low=1.5, close=2.5),
            CandleRecord(timestamp=80, open=2.5, high=3.5, low=2.0, close=3.0),  # Truncate (< 100)
        ],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert [r.timestamp for r in records] == [250, 180, 120]
    assert spy.queried_cursors == [300, 180]


def test_supports_datetime_timestamps():
    """
    Verify support for datetime objects for start_time, end_time, and record timestamps.
    """
    t_start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    t_mid = datetime(2023, 1, 1, 6, 0, tzinfo=timezone.utc)
    t_end = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    t_below = datetime(2022, 12, 31, 23, 0, tzinfo=timezone.utc)

    responses = {
        t_end: [{"timestamp": t_mid}],
        t_mid: [{"timestamp": t_start}, {"timestamp": t_below}],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=t_start,
        end_time=t_end,
        fetch_callback=spy,
    )

    records = list(paginator)

    assert [r["timestamp"] for r in records] == [t_mid, t_start]
    assert spy.queried_cursors == [t_end, t_mid]


def test_custom_timestamp_key():
    """
    Verify paginator functions correctly when records use an alternative timestamp key (e.g. 'created_at').
    """
    start_time = 100
    end_time = 300

    responses = {
        300: [
            {"created_at": 250, "payload": "batch1"},
            {"created_at": 200, "payload": "batch1"},
        ],
        200: [
            {"created_at": 150, "payload": "batch2"},
            {"created_at": 90, "payload": "batch2"},  # Older than start_time
        ],
    }
    spy = FetchCallbackSpy(responses)

    paginator = BackwardSeekingPaginator(
        start_time=start_time,
        end_time=end_time,
        fetch_callback=spy,
        timestamp_key="created_at",
    )

    records = list(paginator)

    assert [r["created_at"] for r in records] == [250, 200, 150]
    assert spy.queried_cursors == [300, 200]


def test_invalid_time_range_raises_value_error():
    """
    Verify ValueError is raised if start_time is greater than end_time.
    Asserts exception type directly without strict regex matching.
    """
    dummy_callback: Callable[[Any], List[Any]] = lambda cursor: []

    with pytest.raises(ValueError):
        BackwardSeekingPaginator(
            start_time=500,
            end_time=100,
            fetch_callback=dummy_callback,
        )