"""Charting module providing multi-chart synchronization, visualization, and snapping components."""

from src.charting.snap import (
    Candlestick,
    MagneticSnap,
    Point,
    SnapResult,
    snap_to_ohlc,
)
from src.charting.sync_manager import (
    CrosshairPosition,
    MultiChartSynchronizer,
    SyncManager,
)
from src.charting.y_axis_resolver import (
    YAxisRange,
    YAxisResolver,
)

__all__ = [
    "Candlestick",
    "CrosshairPosition",
    "MagneticSnap",
    "MultiChartSynchronizer",
    "Point",
    "SnapResult",
    "SyncManager",
    "YAxisRange",
    "YAxisResolver",
    "snap_to_ohlc",
]