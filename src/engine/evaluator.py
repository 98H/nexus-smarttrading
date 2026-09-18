from __future__ import annotations

from typing import Any

from src.engine.series import Series


class Evaluator:
    """Engine evaluator responsible for resolving series operations and expressions."""

    def evaluate_historic_reference(self, series: Series, offset: int) -> Any:
        """Evaluate the historic reference operator `series[offset]`."""
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError(f"Historic reference offset must be an integer, got {type(offset).__name__}")

        return series[offset]