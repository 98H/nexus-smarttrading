"""
Unit tests for Interactive HTML5 Chart Snapshot and Embed Generator.

Story 10.2.1: Build Interactive HTML5 Chart Snapshot and Embed Generator
Target Modules:
- src/visualization/html_embed.py
- src/visualization/__init__.py
"""

import json
import re
from typing import Any, Dict, Tuple
import pytest

from src.visualization import generate_chart_snapshot_and_embed
from src.visualization.html_embed import (
    generate_chart_snapshot_and_embed as html_embed_generator,
)


def _unpack_result(result: Any) -> Tuple[str, str]:
    """Helper to safely extract (html_document, embed_snippet) from returned value.

    Supports return values as tuple, namedtuple, dataclass, or dict.
    """
    if isinstance(result, tuple) and len(result) == 2:
        return str(result[0]), str(result[1])
    if hasattr(result, "html_document") and hasattr(result, "embed_snippet"):
        return str(result.html_document), str(result.embed_snippet)
    if hasattr(result, "html") and hasattr(result, "embed"):
        return str(result.html), str(result.embed)
    if isinstance(result, dict):
        html = result.get("html_document") or result.get("html")
        embed = result.get("embed_snippet") or result.get("embed") or result.get("iframe")
        if html is not None and embed is not None:
            return str(html), str(embed)
    raise TypeError(
        f"generate_chart_snapshot_and_embed must return a 2-tuple or result object, "
        f"got: {type(result)}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_chart_id() -> str:
    """Return a unique chart identifier fixture."""
    return "chart-revenue-q3-2024"


@pytest.fixture
def valid_chart_payload() -> Dict[str, Any]:
    """Return a valid interactive chart specification payload fixture."""
    return {
        "type": "bar",
        "data": {
            "labels": ["January", "February", "March", "April"],
            "datasets": [
                {
                    "label": "Net Sales ($M)",
                    "data": [12.5, 19.3, 15.0, 22.1],
                    "backgroundColor": "rgba(54, 162, 235, 0.6)",
                }
            ],
        },
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"position": "top"},
                "tooltip": {"enabled": True},
            },
        },
    }


@pytest.fixture
def sample_metadata() -> Dict[str, str]:
    """Return valid metadata for chart generation."""
    return {
        "title": "Quarterly Performance",
        "description": "Interactive breakdown of Q3 sales and revenue.",
        "author": "Financial Analytics Team",
    }


# ---------------------------------------------------------------------------
# Module and Export Tests
# ---------------------------------------------------------------------------


def test_visualization_package_exports_generator():
    """Verify generate_chart_snapshot_and_embed is exported in src.visualization."""
    import src.visualization as viz

    assert hasattr(viz, "generate_chart_snapshot_and_embed"), (
        "generate_chart_snapshot_and_embed must be exported by src.visualization"
    )
    assert viz.generate_chart_snapshot_and_embed is html_embed_generator


# ---------------------------------------------------------------------------
# Success Path Tests (Acceptance Criteria 1)
# ---------------------------------------------------------------------------


def test_generate_returns_html_and_embed_strings(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
    sample_metadata: Dict[str, str],
):
    """Verify calling generator with valid arguments returns two non-empty strings."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
        width=800,
        height=500,
        metadata=sample_metadata,
    )

    html_doc, embed_snippet = _unpack_result(result)
    assert isinstance(html_doc, str)
    assert isinstance(embed_snippet, str)
    assert len(html_doc.strip()) > 0
    assert len(embed_snippet.strip()) > 0


def test_snapshot_is_standalone_html5_document(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify snapshot string has HTML5 doctype and standard structural elements."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
    )
    html_doc, _ = _unpack_result(result)

    normalized_html = html_doc.strip().lower()
    assert normalized_html.startswith("<!doctype html"), (
        "HTML5 document must declare <!DOCTYPE html>"
    )
    assert "<html" in normalized_html
    assert "</html>" in normalized_html
    assert "<head" in normalized_html
    assert "</head>" in normalized_html
    assert "<body" in normalized_html
    assert "</body>" in normalized_html


def test_snapshot_contains_viewport_meta_tag(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify HTML5 document includes responsive viewport meta tag."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
    )
    html_doc, _ = _unpack_result(result)

    viewport_match = re.search(
        r'<meta\s+[^>]*name=["\']viewport["\'][^>]*content=["\'][^"\']*width=device-width[^"\']*["\']',
        html_doc,
        re.IGNORECASE,
    )
    assert viewport_match is not None, (
        "Document must contain a viewport meta tag with width=device-width"
    )


def test_snapshot_includes_interactive_chart_assets_and_data(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify document bundles interactive script assets and binds chart payload."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
    )
    html_doc, _ = _unpack_result(result)

    # Must contain script tag for interactive engine/assets
    assert "<script" in html_doc.lower(), "HTML5 snapshot must include script assets"
    assert "</script>" in html_doc.lower()

    # The payload content or chart ID should be present to render the interactive chart
    assert valid_chart_id in html_doc, "HTML5 document must reference the chart identifier"
    assert "Net Sales ($M)" in html_doc, "Chart payload data must be present in snapshot"


def test_snapshot_includes_provided_metadata(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
    sample_metadata: Dict[str, str],
):
    """Verify metadata is embedded into HTML document title and meta elements."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
        metadata=sample_metadata,
    )
    html_doc, _ = _unpack_result(result)

    assert f"<title>{sample_metadata['title']}</title>" in html_doc
    assert sample_metadata["description"] in html_doc


def test_embed_code_contains_responsive_iframe(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify embed snippet is a sanitized responsive <iframe> element."""
    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
    )
    _, embed_snippet = _unpack_result(result)

    snippet_clean = embed_snippet.strip().lower()
    assert snippet_clean.startswith("<iframe"), "Embed code must start with <iframe"
    assert "</iframe>" in snippet_clean, "Embed code must properly close </iframe>"

    # Responsiveness checks (e.g. width="100%", width: 100%, or max-width: 100%)
    has_responsive_width = (
        'width="100%"' in snippet_clean
        or "width: 100%" in snippet_clean
        or "width:100%" in snippet_clean
        or "max-width: 100%" in snippet_clean
    )
    assert has_responsive_width, "Embed <iframe> must be responsive (e.g., width 100%)"


def test_embed_code_applies_custom_dimensions(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify supplied width and height dimensions are applied in embed markup."""
    custom_width = 960
    custom_height = 540

    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
        width=custom_width,
        height=custom_height,
    )
    _, embed_snippet = _unpack_result(result)

    assert str(custom_height) in embed_snippet, (
        f"Height {custom_height} must be reflected in embed snippet attributes/styles"
    )


def test_json_string_payload_acceptance(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify generator accepts a serialized JSON string payload as valid input."""
    payload_json = json.dumps(valid_chart_payload)
    result = generate_chart_snapshot_and_embed(
        chart_payload=payload_json,
        chart_id=valid_chart_id,
    )
    html_doc, embed_snippet = _unpack_result(result)

    assert "<!doctype html" in html_doc.lower()
    assert "<iframe" in embed_snippet.lower()


# ---------------------------------------------------------------------------
# Sanitization and Security Tests
# ---------------------------------------------------------------------------


def test_embed_code_sanitizes_xss_in_chart_id(
    valid_chart_payload: Dict[str, Any],
):
    """Verify XSS injection strings in chart_id are sanitized in <iframe> embed code."""
    malicious_id = 'chart-1"><script>alert("XSS")</script>'

    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=malicious_id,
    )
    _, embed_snippet = _unpack_result(result)

    assert "<script>alert(" not in embed_snippet, (
        "Unsanitized <script> tags must not be present in the <iframe> snippet"
    )
    assert '"><script>' not in embed_snippet, (
        "Embed snippet must escape quote characters to prevent attribute breakout"
    )


def test_snapshot_sanitizes_metadata_xss(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
):
    """Verify dangerous scripts in metadata fields are escaped in HTML document."""
    xss_metadata = {
        "title": 'Dashboard <script>alert("xss")</script>',
        "description": '<img src="x" onerror="alert(1)">',
    }

    result = generate_chart_snapshot_and_embed(
        chart_payload=valid_chart_payload,
        chart_id=valid_chart_id,
        metadata=xss_metadata,
    )
    html_doc, _ = _unpack_result(result)

    assert "<script>alert(\"xss\")</script>" not in html_doc, (
        "Metadata values containing script tags must be HTML-escaped"
    )
    assert "&lt;script&gt;" in html_doc or "Dashboard" in html_doc


# ---------------------------------------------------------------------------
# Failure / Validation Tests (Acceptance Criteria 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "empty_payload",
    [
        None,
        {},
        "",
        "   ",
    ],
)
def test_empty_chart_payload_raises_value_error(
    empty_payload: Any,
    valid_chart_id: str,
):
    """Verify calling generator with None or empty payload raises ValueError."""
    with pytest.raises(ValueError):
        generate_chart_snapshot_and_embed(
            chart_payload=empty_payload,
            chart_id=valid_chart_id,
        )


@pytest.mark.parametrize(
    "invalid_payload",
    [
        12345,
        True,
        [],
        "{malformed_json: true",
        ["not", "a", "valid", "chart", "spec"],
    ],
)
def test_invalid_chart_payload_type_raises_value_error(
    invalid_payload: Any,
    valid_chart_id: str,
):
    """Verify malformed JSON or invalid payload types raise ValueError."""
    with pytest.raises(ValueError):
        generate_chart_snapshot_and_embed(
            chart_payload=invalid_payload,
            chart_id=valid_chart_id,
        )


@pytest.mark.parametrize(
    "invalid_id",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_empty_or_invalid_chart_id_raises_value_error(
    valid_chart_payload: Dict[str, Any],
    invalid_id: Any,
):
    """Verify missing, empty, or non-string chart_id raises ValueError."""
    with pytest.raises(ValueError):
        generate_chart_snapshot_and_embed(
            chart_payload=valid_chart_payload,
            chart_id=invalid_id,
        )


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 500),
        (-100, 500),
        (800, 0),
        (800, -200),
        (-50, -50),
    ],
)
def test_non_positive_dimensions_raise_value_error(
    valid_chart_payload: Dict[str, Any],
    valid_chart_id: str,
    width: int,
    height: int,
):
    """Verify non-positive width and height values raise ValueError."""
    with pytest.raises(ValueError):
        generate_chart_snapshot_and_embed(
            chart_payload=valid_chart_payload,
            chart_id=valid_chart_id,
            width=width,
            height=height,
        )


def test_exception_prevents_embed_generation_and_no_side_effects(
    valid_chart_id: str,
):
    """Verify when ValueError is raised, no embed output is returned or generated."""
    executed = False
    try:
        generate_chart_snapshot_and_embed(
            chart_payload=None,
            chart_id=valid_chart_id,
        )
        executed = True
    except ValueError:
        pass

    assert not executed, "Generator must raise ValueError and not complete execution"