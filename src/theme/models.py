"""Domain models for theme configurations, canvas parameters, and styling tokens."""

from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class GridLineSettings:
    """Settings defining grid line styling on the canvas."""

    color: str
    width: Union[int, float] = 1.0
    style: str = "solid"


@dataclass
class ScaleSettings:
    """Settings defining scale tick and font styling on the canvas."""

    tick_color: str
    font_color: str
    font_size: Union[int, float] = 12.0


@dataclass
class CanvasSettings:
    """Aggregate settings for canvas rendering including background, grid, and scales."""

    background_color: str
    grid_lines: GridLineSettings
    scales: ScaleSettings


@dataclass
class ThemeConfig:
    """Full theme configuration containing identity, canvas settings, and CSS variables."""

    identifier: str
    canvas: CanvasSettings
    css_variables: Dict[str, str]