"""
Multi-chart synchronizer module for crosshair, symbol, and interval events.

Provides a centralized SyncManager to coordinate interactive state updates across
independent chart subscribers organized by synchronization groups.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrosshairPosition:
    """Represents a crosshair coordinate in time and price dimensions."""

    timestamp: float
    price: Optional[float] = None


@dataclass
class _ChartSubscription:
    """Internal registration model holding subscription configuration and callbacks."""

    chart_id: str
    on_crosshair: Optional[Callable[[CrosshairPosition], None]] = None
    on_symbol: Optional[Callable[[str], None]] = None
    on_interval: Optional[Callable[[str], None]] = None
    sync_crosshair: bool = True
    sync_symbol: bool = True
    sync_interval: bool = True


class SyncManager:
    """
    Coordinates state synchronization (crosshair, symbol, interval) among charts
    partitioned into logical synchronization groups.
    """

    def __init__(self) -> None:
        # Structure: {group_id: {chart_id: _ChartSubscription}}
        self._groups: Dict[str, Dict[str, _ChartSubscription]] = {}

    def register_chart(
        self,
        group_id: str,
        chart_id: str,
        on_crosshair: Optional[Callable[[CrosshairPosition], None]] = None,
        on_symbol: Optional[Callable[[str], None]] = None,
        on_interval: Optional[Callable[[str], None]] = None,
        sync_crosshair: bool = True,
        sync_symbol: bool = True,
        sync_interval: bool = True,
    ) -> None:
        """
        Registers a chart subscriber in a synchronization group.

        Raises:
            ValueError: If the chart_id is already registered within the group.
        """
        group = self._groups.setdefault(group_id, {})
        if chart_id in group:
            raise ValueError(f"Chart '{chart_id}' is already registered in group '{group_id}'.")

        group[chart_id] = _ChartSubscription(
            chart_id=chart_id,
            on_crosshair=on_crosshair,
            on_symbol=on_symbol,
            on_interval=on_interval,
            sync_crosshair=sync_crosshair,
            sync_symbol=sync_symbol,
            sync_interval=sync_interval,
        )

    def unregister_chart(self, group_id: str, chart_id: str) -> None:
        """
        Removes a chart subscriber from a synchronization group.

        Raises:
            ValueError: If the group or chart_id is not registered.
        """
        if group_id not in self._groups or chart_id not in self._groups[group_id]:
            raise ValueError(f"Chart '{chart_id}' is not registered in group '{group_id}'.")

        del self._groups[group_id][chart_id]
        if not self._groups[group_id]:
            del self._groups[group_id]

    def set_sync_flags(
        self,
        group_id: str,
        chart_id: str,
        sync_crosshair: Optional[bool] = None,
        sync_symbol: Optional[bool] = None,
        sync_interval: Optional[bool] = None,
    ) -> None:
        """
        Dynamically updates synchronization flags for a registered chart.

        Raises:
            ValueError: If the chart is not registered in the group.
        """
        if group_id not in self._groups or chart_id not in self._groups[group_id]:
            raise ValueError(f"Chart '{chart_id}' is not registered in group '{group_id}'.")

        sub = self._groups[group_id][chart_id]
        if sync_crosshair is not None:
            sub.sync_crosshair = sync_crosshair
        if sync_symbol is not None:
            sub.sync_symbol = sync_symbol
        if sync_interval is not None:
            sub.sync_interval = sync_interval

    def get_charts_in_group(self, group_id: str) -> List[str]:
        """Returns the list of chart identifiers registered in the group."""
        return list(self._groups.get(group_id, {}).keys())

    def update_crosshair(
        self,
        group_id: str,
        source_chart_id: str,
        position: CrosshairPosition,
    ) -> None:
        """
        Broadcasts crosshair update to all peer charts in the group.

        Raises:
            ValueError: If source_chart_id is not registered in group_id.
        """
        self._validate_source(group_id, source_chart_id)

        group = self._groups[group_id]
        for chart_id, sub in list(group.items()):
            if chart_id == source_chart_id:
                continue
            if sub.sync_crosshair and sub.on_crosshair is not None:
                try:
                    sub.on_crosshair(position)
                except Exception:
                    logger.exception(
                        "Error propagating crosshair to chart '%s' in group '%s'",
                        chart_id,
                        group_id,
                    )

    def update_symbol(
        self,
        group_id: str,
        source_chart_id: str,
        symbol: str,
    ) -> None:
        """
        Broadcasts symbol change to all peer charts in the group.

        Raises:
            ValueError: If symbol is empty or source_chart_id is not registered.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string.")

        self._validate_source(group_id, source_chart_id)

        group = self._groups[group_id]
        for chart_id, sub in list(group.items()):
            if chart_id == source_chart_id:
                continue
            if sub.sync_symbol and sub.on_symbol is not None:
                try:
                    sub.on_symbol(symbol)
                except Exception:
                    logger.exception(
                        "Error propagating symbol to chart '%s' in group '%s'",
                        chart_id,
                        group_id,
                    )

    def update_interval(
        self,
        group_id: str,
        source_chart_id: str,
        interval: str,
    ) -> None:
        """
        Broadcasts interval change to all peer charts in the group.

        Raises:
            ValueError: If interval is empty or source_chart_id is not registered.
        """
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("Interval must be a non-empty string.")

        self._validate_source(group_id, source_chart_id)

        group = self._groups[group_id]
        for chart_id, sub in list(group.items()):
            if chart_id == source_chart_id:
                continue
            if sub.sync_interval and sub.on_interval is not None:
                try:
                    sub.on_interval(interval)
                except Exception:
                    logger.exception(
                        "Error propagating interval to chart '%s' in group '%s'",
                        chart_id,
                        group_id,
                    )

    def _validate_source(self, group_id: str, source_chart_id: str) -> None:
        """Validates that group exists and source chart is a registered member."""
        if group_id not in self._groups or source_chart_id not in self._groups[group_id]:
            raise ValueError(
                f"Source chart '{source_chart_id}' is not registered in group '{group_id}'."
            )


# Class alias to fulfill API naming requirements
MultiChartSynchronizer = SyncManager

__all__ = ["CrosshairPosition", "MultiChartSynchronizer", "SyncManager"]