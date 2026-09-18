"""Visualization package providing chart snapshot and embed generation tools."""

from src.visualization.html_embed import (
    ChartEmbedResult,
    generate_chart_snapshot_and_embed,
)

__all__ = [
    "ChartEmbedResult",
    "generate_chart_snapshot_and_embed",
]