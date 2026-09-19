from __future__ import annotations

from typing import Any, Generic, Iterator, TypeVar, overload

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """High-throughput fixed-capacity circular ring buffer.

    Maintains FIFO ordering of elements up to a fixed capacity. When capacity
    is reached, subsequent appends overwrite the oldest elements in O(1) time.
    """

    __slots__ = ("_capacity", "_buffer", "_start", "_size")

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError(f"Capacity must be an integer, got {type(capacity).__name__}")
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity: int = capacity
        self._buffer: list[Any] = [None] * capacity
        self._start: int = 0
        self._size: int = 0

    @property
    def capacity(self) -> int:
        """Maximum number of elements the buffer can hold."""
        return self._capacity

    @property
    def is_empty(self) -> bool:
        """True if the buffer contains no elements."""
        return self._size == 0

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached its capacity."""
        return self._size == self._capacity

    @property
    def oldest(self) -> T:
        """Retrieve the oldest element in O(1) time without removing it."""
        if self._size == 0:
            raise IndexError("RingBuffer is empty")
        return self._buffer[self._start]

    @property
    def newest(self) -> T:
        """Retrieve the newest element in O(1) time without removing it."""
        if self._size == 0:
            raise IndexError("RingBuffer is empty")
        return self._buffer[(self._start + self._size - 1) % self._capacity]

    def append(self, item: T) -> None:
        """Append an item to the buffer in O(1) time, overwriting oldest if full."""
        if self._size < self._capacity:
            write_index = (self._start + self._size) % self._capacity
            self._buffer[write_index] = item
            self._size += 1
        else:
            self._buffer[self._start] = item
            self._start = (self._start + 1) % self._capacity

    def clear(self) -> None:
        """Reset the buffer to an empty state."""
        self._buffer = [None] * self._capacity
        self._start = 0
        self._size = 0

    def to_list(self) -> list[T]:
        """Return contents as a list in chronological order (oldest to newest)."""
        if self._size == 0:
            return []
        if self._start + self._size <= self._capacity:
            return self._buffer[self._start : self._start + self._size]
        return (
            self._buffer[self._start : self._capacity]
            + self._buffer[: (self._start + self._size) % self._capacity]
        )

    def __len__(self) -> int:
        """Return the current number of elements in the buffer."""
        return self._size

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Retrieve element or slice in chronological order without mutating state."""
        if isinstance(index, slice):
            return self.to_list()[index]

        if isinstance(index, int):
            resolved_index = index if index >= 0 else index + self._size
            if resolved_index < 0 or resolved_index >= self._size:
                raise IndexError(f"RingBuffer index out of range: {index}")
            return self._buffer[(self._start + resolved_index) % self._capacity]

        raise TypeError(
            f"RingBuffer indices must be integers or slices, not {type(index).__name__}"
        )

    def __iter__(self) -> Iterator[T]:
        """Iterate over elements in chronological order (oldest to newest)."""
        start = self._start
        capacity = self._capacity
        buffer = self._buffer
        for i in range(self._size):
            yield buffer[(start + i) % capacity]

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, size={self._size})"