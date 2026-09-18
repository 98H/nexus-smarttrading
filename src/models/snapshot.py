"""Domain models and enumerations for high-resolution snapshot exports."""

from dataclasses import dataclass
from enum import Enum
from typing import Union


class ExportFormat(str, Enum):
    """Supported export formats for chart snapshots and dataset records."""

    PNG = "PNG"
    SVG = "SVG"
    CSV = "CSV"


@dataclass
class SnapshotRequest:
    """Request payload specifying export format and rendering configuration."""

    format: Union[ExportFormat, str]
    dpi: int = 300