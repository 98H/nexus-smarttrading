import csv
import io
import struct
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.snapshot import ExportFormat, SnapshotRequest
from src.services.snapshot_exporter import SnapshotExporter, UnsupportedFormatError

# Standard 8-byte PNG file signature
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def exporter() -> SnapshotExporter:
    """Provides a fresh instance of SnapshotExporter."""
    return SnapshotExporter()


@pytest.fixture
def sample_figure() -> plt.Figure:
    """Provides a clean matplotlib chart figure for snapshot export tests."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label="Active Users")
    ax.set_title("User Growth Trend")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Count")
    ax.legend()
    yield fig
    plt.close(fig)


@pytest.fixture
def sample_tabular_records() -> List[Dict[str, Any]]:
    """Provides sample tabular dataset records for CSV export tests."""
    return [
        {"timestamp": "2026-01-01T00:00:00Z", "metric": "cpu_usage", "value": 45.2, "status": "OK"},
        {"timestamp": "2026-01-01T01:00:00Z", "metric": "memory_usage", "value": 78.9, "status": "WARNING"},
        {"timestamp": "2026-01-01T02:00:00Z", "metric": "disk_io", "value": 12.0, "status": "OK"},
    ]


# ==============================================================================
# AC 1: High-Resolution Binary PNG Export
# ==============================================================================


class TestSnapshotExporterPNG:
    def test_export_png_returns_binary_png_bytes(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Given a chart figure and request with format PNG and DPI 300,

        returns binary PNG bytes matching the PNG magic signature.
        """
        request = SnapshotRequest(format=ExportFormat.PNG, dpi=300)

        result = exporter.export(figure=sample_figure, request=request)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result.startswith(PNG_SIGNATURE)

    def test_export_png_with_string_format_returns_png_bytes(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Ensures that string literal 'PNG' is accepted interchangeably with ExportFormat.PNG."""
        request = SnapshotRequest(format="PNG", dpi=300)

        result = exporter.export(figure=sample_figure, request=request)

        assert isinstance(result, bytes)
        assert result.startswith(PNG_SIGNATURE)

    def test_export_png_dpi_300_produces_high_resolution_scaling(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Verifies that DPI 300 scaling produces higher pixel dimensions and byte size

        than standard screen DPI (72).
        """
        request_high_dpi = SnapshotRequest(format=ExportFormat.PNG, dpi=300)
        request_low_dpi = SnapshotRequest(format=ExportFormat.PNG, dpi=72)

        png_high = exporter.export(figure=sample_figure, request=request_high_dpi)
        png_low = exporter.export(figure=sample_figure, request=request_low_dpi)

        # In standard PNG, bytes 16:24 store big-endian 32-bit width and height (IHDR chunk)
        width_high, height_high = struct.unpack(">II", png_high[16:24])
        width_low, height_low = struct.unpack(">II", png_low[16:24])

        assert width_high > width_low
        assert height_high > height_low
        assert len(png_high) > len(png_low)

    def test_export_png_case_insensitive_format(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Ensures lower-case 'png' is accepted and handled consistently."""
        request = SnapshotRequest(format="png", dpi=300)

        result = exporter.export(figure=sample_figure, request=request)

        assert isinstance(result, bytes)
        assert result.startswith(PNG_SIGNATURE)


# ==============================================================================
# AC 2: Standard UTF-8 Encoded Vector SVG Export
# ==============================================================================


class TestSnapshotExporterSVG:
    def test_export_svg_returns_utf8_encoded_vector_svg_bytes(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Given a chart figure and request with format SVG,

        returns standard UTF-8 encoded vector SVG data.
        """
        request = SnapshotRequest(format=ExportFormat.SVG)

        result = exporter.export(figure=sample_figure, request=request)

        assert isinstance(result, bytes)
        # Verify valid UTF-8 encoding
        decoded_svg = result.decode("utf-8")
        assert "<svg" in decoded_svg.lower()
        assert "</svg>" in decoded_svg.lower()

    def test_export_svg_parses_as_valid_xml(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Verifies that returned SVG data is syntactically valid XML containing vector elements."""
        request = SnapshotRequest(format="SVG")

        result = exporter.export(figure=sample_figure, request=request)

        root = ET.fromstring(result)
        # Root tag should be the SVG namespace or svg element
        assert root.tag.endswith("svg")
        # Ensure graphic elements exist in the vector tree
        assert len(list(root)) > 0

    def test_export_svg_contains_rendered_text_elements(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Verifies that vector SVG includes rendered chart content such as titles and labels."""
        request = SnapshotRequest(format=ExportFormat.SVG)

        result = exporter.export(figure=sample_figure, request=request)
        decoded_svg = result.decode("utf-8")

        assert "User Growth Trend" in decoded_svg
        assert "Quarter" in decoded_svg
        assert "Active Users" in decoded_svg


# ==============================================================================
# AC 3: Tabular Dataset Records CSV Export
# ==============================================================================


class TestSnapshotExporterCSVData:
    def test_export_data_returns_formatted_csv_text_bytes(
        self, exporter: SnapshotExporter, sample_tabular_records: List[Dict[str, Any]]
    ):
        """Given tabular dataset records and request format CSV,

        When export_data is invoked, formatted CSV text bytes are returned.
        """
        request = SnapshotRequest(format=ExportFormat.CSV)

        result = exporter.export_data(records=sample_tabular_records, request=request)

        assert isinstance(result, bytes)
        csv_text = result.decode("utf-8")

        # Parse CSV to ensure it is correctly formatted RFC 4180 tabular text
        reader = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(reader) == 3
        assert reader[0]["metric"] == "cpu_usage"
        assert float(reader[0]["value"]) == 45.2
        assert reader[1]["status"] == "WARNING"

    def test_export_data_preserves_columns_order_and_headers(
        self, exporter: SnapshotExporter, sample_tabular_records: List[Dict[str, Any]]
    ):
        """Verifies CSV header matches dataset record keys."""
        request = SnapshotRequest(format="CSV")

        result = exporter.export_data(records=sample_tabular_records, request=request)
        csv_text = result.decode("utf-8")

        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        expected_header = ["timestamp", "metric", "value", "status"]
        assert header == expected_header

    def test_export_data_handles_special_characters_escaping(
        self, exporter: SnapshotExporter
    ):
        """Verifies CSV exporter properly escapes commas, quotes, and newlines."""
        records = [
            {"id": 1, "description": 'Value with, comma and "quotes"'},
            {"id": 2, "description": "Line 1\nLine 2"},
        ]
        request = SnapshotRequest(format=ExportFormat.CSV)

        result = exporter.export_data(records=records, request=request)
        csv_text = result.decode("utf-8")

        reader = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(reader) == 2
        assert reader[0]["description"] == 'Value with, comma and "quotes"'
        assert reader[1]["description"] == "Line 1\nLine 2"

    def test_export_data_handles_utf8_multibyte_characters(
        self, exporter: SnapshotExporter
    ):
        """Verifies UTF-8 encoding handles international characters and symbols."""
        records = [
            {"region": "Zürich", "currency": "€", "symbol": "📈"},
            {"region": "Tokyo", "currency": "¥", "symbol": "📊"},
        ]
        request = SnapshotRequest(format=ExportFormat.CSV)

        result = exporter.export_data(records=records, request=request)
        csv_text = result.decode("utf-8")

        reader = list(csv.DictReader(io.StringIO(csv_text)))
        assert reader[0]["region"] == "Zürich"
        assert reader[0]["currency"] == "€"
        assert reader[0]["symbol"] == "📈"

    def test_export_data_with_empty_records_returns_empty_or_header_bytes(
        self, exporter: SnapshotExporter
    ):
        """Verifies export_data on empty record set returns valid bytes without crashing."""
        request = SnapshotRequest(format=ExportFormat.CSV)

        result = exporter.export_data(records=[], request=request)

        assert isinstance(result, bytes)


# ==============================================================================
# AC 4: Unsupported Export Format Handling
# ==============================================================================


class TestSnapshotExporterUnsupportedFormats:
    @pytest.mark.parametrize(
        "invalid_format",
        ["PDF", "TIFF", "EXE", "JSON", "DOCX", "UNSUPPORTED"],
    )
    def test_export_unsupported_format_raises_unsupported_format_error(
        self,
        exporter: SnapshotExporter,
        sample_figure: plt.Figure,
        invalid_format: str,
    ):
        """Given an unsupported export format requested, When export is invoked,

        Then UnsupportedFormatError is raised.
        """
        request = SnapshotRequest(format=invalid_format)

        with pytest.raises(UnsupportedFormatError):
            exporter.export(figure=sample_figure, request=request)

    def test_export_csv_format_on_chart_figure_raises_unsupported_format_error(
        self, exporter: SnapshotExporter, sample_figure: plt.Figure
    ):
        """Verifies that attempting to export a visual figure as CSV raises UnsupportedFormatError."""
        request = SnapshotRequest(format=ExportFormat.CSV)

        with pytest.raises(UnsupportedFormatError):
            exporter.export(figure=sample_figure, request=request)

    @pytest.mark.parametrize("invalid_data_format", ["PNG", "SVG", "BMP", "HTML"])
    def test_export_data_unsupported_format_raises_unsupported_format_error(
        self,
        exporter: SnapshotExporter,
        sample_tabular_records: List[Dict[str, Any]],
        invalid_data_format: str,
    ):
        """Verifies that requesting non-tabular formats in export_data raises UnsupportedFormatError."""
        request = SnapshotRequest(format=invalid_data_format)

        with pytest.raises(UnsupportedFormatError):
            exporter.export_data(records=sample_tabular_records, request=request)


# ==============================================================================
# Models and Direct Class Invocation Tests
# ==============================================================================


class TestSnapshotModels:
    def test_snapshot_request_default_dpi(self):
        """Verifies SnapshotRequest defaults DPI to 300 if not specified."""
        req = SnapshotRequest(format=ExportFormat.PNG)
        assert req.dpi == 300

    def test_snapshot_request_custom_dpi(self):
        """Verifies SnapshotRequest retains custom DPI configuration."""
        req = SnapshotRequest(format=ExportFormat.PNG, dpi=600)
        assert req.dpi == 600

    def test_export_format_enum_values(self):
        """Verifies that ExportFormat enum exposes required format constants."""
        assert ExportFormat.PNG.value.upper() == "PNG"
        assert ExportFormat.SVG.value.upper() == "SVG"
        assert ExportFormat.CSV.value.upper() == "CSV"


class TestSnapshotExporterInvocationPatterns:
    def test_export_callable_as_class_method(
        self, sample_figure: plt.Figure
    ):
        """Verifies SnapshotExporter.export can be invoked directly as specified in AC."""
        request = SnapshotRequest(format=ExportFormat.PNG, dpi=300)

        result = SnapshotExporter.export(figure=sample_figure, request=request)

        assert isinstance(result, bytes)
        assert result.startswith(PNG_SIGNATURE)

    def test_export_data_callable_as_class_method(
        self, sample_tabular_records: List[Dict[str, Any]]
    ):
        """Verifies SnapshotExporter.export_data can be invoked directly as specified in AC."""
        request = SnapshotRequest(format=ExportFormat.CSV)

        result = SnapshotExporter.export_data(records=sample_tabular_records, request=request)

        assert isinstance(result, bytes)
        assert len(result) > 0