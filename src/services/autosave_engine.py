import asyncio
from datetime import datetime, timezone
import inspect
from typing import Any, Dict, Optional

from src.models.chart_state import ChartDelta, ChartState


class ConcurrencyConflictError(Exception):
    """Raised when an auto-save delta's expected version conflicts with the persisted version."""

    def __init__(
        self,
        chart_id: str,
        expected_version: int,
        current_version: int,
        message: Optional[str] = None,
    ) -> None:
        self.chart_id = chart_id
        self.expected_version = expected_version
        self.current_version = current_version
        if message is None:
            message = (
                f"Concurrency conflict for chart '{chart_id}': "
                f"expected version {expected_version}, but current version is {current_version}."
            )
        super().__init__(message)


class ChartNotFoundError(Exception):
    """Raised when an auto-save delta targets a chart that does not exist."""

    def __init__(self, chart_id: str, message: Optional[str] = None) -> None:
        self.chart_id = chart_id
        if message is None:
            message = f"Chart '{chart_id}' was not found."
        super().__init__(message)


class AutoSaveEngine:
    """Engine responsible for auto-saving chart states with optimistic concurrency control."""

    def __init__(self, storage: Any) -> None:
        self.storage = storage
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, chart_id: str) -> asyncio.Lock:
        if chart_id not in self._locks:
            self._locks[chart_id] = asyncio.Lock()
        return self._locks[chart_id]

    async def process_delta(self, delta: ChartDelta) -> ChartState:
        """
        Processes an incoming chart delta.

        If the expected version matches the current version, the delta changes are merged,
        the version increments to N + 1, and the state is persisted.
        If the version does not match, a ConcurrencyConflictError is raised and the
        existing state remains unaltered.
        """
        lock = self._get_lock(delta.chart_id)
        async with lock:
            chart_raw = self.storage.get_chart(delta.chart_id)
            chart: Optional[ChartState] = (
                await chart_raw if inspect.isawaitable(chart_raw) else chart_raw
            )

            if chart is None:
                raise ChartNotFoundError(chart_id=delta.chart_id)

            if chart.version != delta.expected_version:
                raise ConcurrencyConflictError(
                    chart_id=delta.chart_id,
                    expected_version=delta.expected_version,
                    current_version=chart.version,
                )

            merged_data = dict(chart.data) if chart.data else {}
            merged_data.update(delta.changes)

            updated_state = ChartState(
                chart_id=chart.chart_id,
                version=chart.version + 1,
                data=merged_data,
                user_id=chart.user_id,
                updated_at=datetime.now(timezone.utc),
            )

            save_raw = self.storage.save_chart(updated_state)
            persisted = await save_raw if inspect.isawaitable(save_raw) else save_raw

            return persisted if isinstance(persisted, ChartState) else updated_state