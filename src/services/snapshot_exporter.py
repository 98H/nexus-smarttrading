"""Snapshot exporter service supporting raster (PNG), vector (SVG), and tabular (CSV) data."""

import csv
import io
from typing import Any, Dict, List, Union

import matplotlib
from matplotlib.figure import Figure

from src.models.snapshot import ExportFormat, SnapshotRequest


class UnsupportedFormatError(Exception):
    """Raised when an unsupported export format is requested."""


class SnapshotExporter:
    """Exports chart figures and tabular dataset records into specified formats."""

    @classmethod
    def _normalize_format(cls, format_value: Union[ExportFormat, str]) -> str:
        """Normalizes format values to uppercase string representation."""
        if isinstance(format_value, ExportFormat):
            return format_value.value.upper()
        if isinstance(format_value, str):
            return format_value.strip().upper()
        return str(format_value).strip().upper()

    @classmethod
    def export(
        cls,
        figure: Figure,
        request: SnapshotRequest,
    ) -> bytes:
        """Exports a matplotlib figure to binary PNG or UTF-8 vector SVG bytes.

        Raises:
            UnsupportedFormatError: If the requested format is unsupported or non-visual.
        """
        normalized_format = cls._normalize_format(request.format)

        if normalized_format == ExportFormat.PNG.value:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=request.dpi)
            return buffer.getvalue()

        if normalized_format == ExportFormat.SVG.value:
            buffer = io.BytesIO()
            with matplotlib.rc_context({"svg.fonttype": "none"}):
                figure.savefig(buffer, format="svg", dpi=request.dpi)
            return buffer.getvalue()

        raise UnsupportedFormatError(
            f"Unsupported format '{request.format}' for chart figure export. "
            f"Expected one of: {[ExportFormat.PNG.value, ExportFormat.SVG.value]}"
        )

    @classmethod
    def export_data(
        cls,
        records: List[Dict[str, Any]],
        request: SnapshotRequest,
    ) -> bytes:
        """Exports tabular dataset records to UTF-8 encoded CSV bytes.

        Raises:
            UnsupportedFormatError: If the requested format is not CSV.
        """
        normalized_format = cls._normalize_format(request.format)

        if normalized_format != ExportFormat.CSV.value:
            raise UnsupportedFormatError(
                f"Unsupported format '{request.format}' for tabular data export. "
                f"Expected: {ExportFormat.CSV.value}"
            )

        if not records:
            return b""

        fieldnames = list({key: None for record in records for key in record.keys()}.keys())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        return buffer.getvalue().encode("utf-8")