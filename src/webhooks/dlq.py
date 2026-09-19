from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class DLQRecord:
    """Record enqueued into the Dead-Letter Queue upon exhausted deliveries."""

    delivery_id: str
    payload: Any
    error_reason: str
    timestamp: float = field(default_factory=time.time)

    @property
    def reason(self) -> str:
        """Alias for error_reason."""
        return self.error_reason


class DeadLetterQueue:
    """FIFO queue storing delivery payloads that exhausted retries."""

    def __init__(self) -> None:
        self._records: list[DLQRecord] = []

    def enqueue(
        self,
        delivery_id: Union[str, DLQRecord],
        payload: Any = None,
        reason: Optional[str] = None,
        error_reason: Optional[str] = None,
    ) -> DLQRecord:
        """Enqueue a failed delivery into the DLQ."""
        if isinstance(delivery_id, DLQRecord):
            self._records.append(delivery_id)
            return delivery_id

        final_reason = reason if reason is not None else (error_reason or "")
        record = DLQRecord(
            delivery_id=str(delivery_id),
            payload=payload,
            error_reason=final_reason,
        )
        self._records.append(record)
        return record

    def peek(self) -> DLQRecord:
        """Inspect the oldest record without removing it."""
        if not self._records:
            raise IndexError("DLQ is empty")
        return self._records[0]

    def pop(self) -> DLQRecord:
        """Remove and return the oldest record from the queue."""
        if not self._records:
            raise IndexError("DLQ is empty")
        return self._records.pop(0)

    def size(self) -> int:
        """Return the current number of records in the queue."""
        return len(self._records)

    def get_records(self) -> list[DLQRecord]:
        """Return a copy of all records currently in the queue."""
        return list(self._records)

    def clear(self) -> None:
        """Remove all records from the queue."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)