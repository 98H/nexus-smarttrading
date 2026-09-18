from __future__ import annotations

import math
from typing import Any, Iterable


class Series:
    """A series representing historical values where offset 0 is the current value."""

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        self._values: list[Any] = list(values) if values is not None else []

    def append(self, value: Any) -> None:
        """Append a new current value to the series."""
        self._values.append(value)

    def __getitem__(self, offset: int) -> Any:
        """Retrieve historical value at the given offset.

        Offset 0 corresponds to the current (most recent) value.
        Offset k corresponds to k bars ago.
        """
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError(f"Historic reference offset must be an integer, got {type(offset).__name__}")

        if offset < 0:
            raise ValueError(f"Historic reference offset must be non-negative, got {offset}")

        if offset >= len(self._values):
            return math.nan

        return self._values[len(self._values) - 1 - offset]

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Series({self._values!r})"