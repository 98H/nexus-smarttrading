"""
Backward-Seeking Historical Data Paginator.

Provides pagination support for seeking historical records backward in time
from end_time towards start_time using descending timestamp batches.
"""

from collections.abc import Iterator, Mapping
from typing import Any, Callable


class BackwardSeekingPaginator:
    """
    Paginator that traverses historical data backward in time.

    Queries records starting from end_time backward, updating the cursor
    to the oldest record's timestamp from each batch until reaching start_time
    or until no further records are returned.
    """

    def __init__(
        self,
        start_time: Any,
        end_time: Any,
        fetch_callback: Callable[..., Any],
        timestamp_key: str = "timestamp",
    ) -> None:
        if start_time > end_time:
            raise ValueError(
                f"start_time ({start_time!r}) cannot be greater than end_time ({end_time!r})"
            )
        self.start_time = start_time
        self.end_time = end_time
        self.fetch_callback = fetch_callback
        self.timestamp_key = timestamp_key

    def _extract_timestamp(self, record: Any) -> Any:
        """Extract timestamp value from a dictionary or object record."""
        if isinstance(record, Mapping):
            return record[self.timestamp_key]
        try:
            return getattr(record, self.timestamp_key)
        except AttributeError:
            return record[self.timestamp_key]

    def __iter__(self) -> Iterator[Any]:
        """
        Lazily iterate over records from end_time backward down to start_time.
        """
        current_cursor = self.end_time

        while True:
            batch = self.fetch_callback(current_cursor)
            if not batch:
                break

            oldest_timestamp = None
            for record in batch:
                ts = self._extract_timestamp(record)
                if ts < self.start_time:
                    # Truncate records older than start_time and terminate
                    return
                oldest_timestamp = ts
                yield record

            # Stop if the batch reached or crossed start_time
            if oldest_timestamp is None or oldest_timestamp <= self.start_time:
                return

            # Avoid infinite loops if cursor fails to move backward
            if oldest_timestamp >= current_cursor:
                break

            current_cursor = oldest_timestamp