"""Data models representing chart layout configurations, panels, indicators, and viewport states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PanelConfig:
    """Configuration for an individual chart panel."""

    panel_id: str
    height_ratio: float
    is_visible: bool = True


@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator associated with a panel."""

    indicator_id: str
    indicator_type: str
    panel_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_visible: bool = True


@dataclass
class ViewportParameters:
    """Parameters defining chart viewport boundaries and zoom state."""

    start_index: int
    end_index: int
    auto_scale: bool = True
    zoom_level: float = 1.0


@dataclass
class ChartLayoutState:
    """Root state encapsulating layout identity, panels, indicators, and viewport."""

    layout_id: str
    panels: List[PanelConfig] = field(default_factory=list)
    indicator_configs: List[IndicatorConfig] = field(default_factory=list)
    viewport_parameters: ViewportParameters = field(
        default_factory=lambda: ViewportParameters(start_index=0, end_index=0)
    )