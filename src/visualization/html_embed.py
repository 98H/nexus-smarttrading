"""Interactive HTML5 Chart Snapshot and Embed Generator.

Provides capabilities to compile interactive chart specifications into
standalone HTML5 documents and responsive, sanitized iframe embed snippets.
"""

import html
import json
import math
from typing import Any, Dict, NamedTuple, Optional, Union


class ChartEmbedResult(NamedTuple):
    """Container holding the standalone HTML5 document and iframe embed snippet."""

    html_document: str
    embed_snippet: str


def _validate_chart_payload(chart_payload: Any) -> Dict[str, Any]:
    """Validate and normalize the chart specification payload.

    Args:
        chart_payload: Dictionary or JSON string representing the chart specification.

    Returns:
        Dict[str, Any]: Validated non-empty chart payload dictionary.

    Raises:
        ValueError: If payload is missing, invalid type, malformed JSON, or empty.
    """
    if chart_payload is None:
        raise ValueError("Chart payload cannot be None.")

    if isinstance(chart_payload, bool):
        raise ValueError("Chart payload cannot be a boolean.")

    if isinstance(chart_payload, dict):
        if not chart_payload:
            raise ValueError("Chart payload dictionary cannot be empty.")
        return chart_payload

    if isinstance(chart_payload, str):
        stripped = chart_payload.strip()
        if not stripped:
            raise ValueError("Chart payload string cannot be empty or whitespace.")
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"Chart payload JSON could not be decoded: {err}") from err

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Parsed chart payload must be a non-empty dictionary.")
        return parsed

    raise ValueError(
        f"Chart payload must be a non-empty dict or JSON string, got: {type(chart_payload).__name__}"
    )


def _validate_chart_id(chart_id: Any) -> str:
    """Validate the unique chart identifier.

    Args:
        chart_id: Identifier for the chart.

    Returns:
        str: Validated chart identifier string.

    Raises:
        ValueError: If chart_id is None, non-string, or empty/whitespace.
    """
    if chart_id is None:
        raise ValueError("chart_id cannot be None.")
    if not isinstance(chart_id, str):
        raise ValueError(f"chart_id must be a string, got: {type(chart_id).__name__}")
    if not chart_id.strip():
        raise ValueError("chart_id cannot be empty or whitespace.")
    return chart_id


def _validate_dimensions(width: Any, height: Any) -> None:
    """Validate that width and height dimensions are positive numbers.

    Args:
        width: Width dimension.
        height: Height dimension.

    Raises:
        ValueError: If either dimension is non-numeric, boolean, NaN, or non-positive.
    """
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or math.isnan(width)
        or width <= 0
    ):
        raise ValueError(f"Width must be a positive number, got: {width}")

    if (
        isinstance(height, bool)
        or not isinstance(height, (int, float))
        or math.isnan(height)
        or height <= 0
    ):
        raise ValueError(f"Height must be a positive number, got: {height}")


def _build_html_document(
    chart_id: str,
    payload: Dict[str, Any],
    width: Union[int, float],
    height: Union[int, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a standalone HTML5 snapshot document containing chart assets and data.

    Args:
        chart_id: Unique chart identifier.
        payload: Chart configuration dictionary.
        width: Target chart display width.
        height: Target chart display height.
        metadata: Optional metadata dictionary (title, description, author, etc.).

    Returns:
        str: Standalone HTML5 document.
    """
    meta_tags = []
    meta_title: Optional[str] = None

    if metadata and isinstance(metadata, dict):
        if "title" in metadata and metadata["title"] is not None:
            meta_title = html.escape(str(metadata["title"]))
        if "description" in metadata and metadata["description"] is not None:
            escaped_desc = html.escape(str(metadata["description"]), quote=True)
            meta_tags.append(f'<meta name="description" content="{escaped_desc}">')
        if "author" in metadata and metadata["author"] is not None:
            escaped_author = html.escape(str(metadata["author"]), quote=True)
            meta_tags.append(f'<meta name="author" content="{escaped_author}">')
        for k, v in metadata.items():
            if k not in ("title", "description", "author") and v is not None:
                escaped_k = html.escape(str(k), quote=True)
                escaped_v = html.escape(str(v), quote=True)
                meta_tags.append(f'<meta name="{escaped_k}" content="{escaped_v}">')

    doc_title = meta_title if meta_title is not None else html.escape(chart_id)
    extra_meta_html = ("\n  " + "\n  ".join(meta_tags)) if meta_tags else ""

    # Prevent premature script tag breakout by escaping closing tag sequences
    safe_payload_json = json.dumps(payload, indent=2).replace("</", "<\\/")
    safe_chart_id_js = (
        json.dumps(chart_id).replace("<", "\\u003c").replace(">", "\\u003e")
    )
    escaped_chart_id = html.escape(chart_id, quote=True)

    w_val = int(width) if int(width) == width else width
    h_val = int(height) if int(height) == height else height

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{doc_title}</title>{extra_meta_html}
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .chart-container {{
      position: relative;
      width: 100%;
      max-width: {w_val}px;
      height: {h_val}px;
      margin: 0 auto;
      padding: 12px;
    }}
  </style>
</head>
<body>
  <div class="chart-container">
    <canvas id="{escaped_chart_id}"></canvas>
  </div>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      const chartId = {safe_chart_id_js};
      const chartConfig = {safe_payload_json};
      const canvas = document.getElementById(chartId);
      if (canvas && typeof Chart !== "undefined") {{
        new Chart(canvas, chartConfig);
      }}
    }});
  </script>
</body>
</html>"""


def _build_embed_snippet(
    chart_id: str,
    width: Union[int, float],
    height: Union[int, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a responsive and sanitized <iframe> embed snippet.

    Args:
        chart_id: Unique chart identifier.
        width: Responsive max width in pixels.
        height: Fixed embed height in pixels.
        metadata: Optional metadata dictionary for iframe title.

    Returns:
        str: Responsive iframe embed snippet.
    """
    escaped_id = html.escape(chart_id, quote=True)
    raw_title = (
        metadata.get("title", f"Chart - {chart_id}")
        if metadata and "title" in metadata
        else f"Chart - {chart_id}"
    )
    escaped_title = html.escape(str(raw_title), quote=True)

    w_val = int(width) if int(width) == width else width
    h_val = int(height) if int(height) == height else height

    return (
        f'<iframe id="chart-embed-{escaped_id}" '
        f'src="{escaped_id}.html" '
        f'title="{escaped_title}" '
        f'width="100%" '
        f'height="{h_val}" '
        f'style="width: 100%; max-width: {w_val}px; height: {h_val}px; border: 0;" '
        f'loading="lazy" '
        f'allowfullscreen></iframe>'
    )


def generate_chart_snapshot_and_embed(
    chart_payload: Union[Dict[str, Any], str],
    chart_id: str,
    width: int = 800,
    height: int = 600,
    metadata: Optional[Dict[str, Any]] = None,
) -> ChartEmbedResult:
    """Generate a standalone interactive HTML5 snapshot and a responsive iframe embed snippet.

    Args:
        chart_payload: Interactive chart specification as a dict or JSON string.
        chart_id: Unique string identifier for the chart.
        width: Target width dimension in pixels (default: 800).
        height: Target height dimension in pixels (default: 600).
        metadata: Optional metadata dictionary (title, description, author, etc.).

    Returns:
        ChartEmbedResult: NamedTuple containing:
            - html_document: Standalone HTML5 document string.
            - embed_snippet: Sanitized responsive <iframe> embed snippet.

    Raises:
        ValueError: If chart_payload, chart_id, or dimensions are invalid or empty.
    """
    validated_payload = _validate_chart_payload(chart_payload)
    validated_chart_id = _validate_chart_id(chart_id)
    _validate_dimensions(width, height)

    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata must be a dictionary if provided, got: {type(metadata).__name__}"
        )

    html_doc = _build_html_document(
        chart_id=validated_chart_id,
        payload=validated_payload,
        width=width,
        height=height,
        metadata=metadata,
    )

    embed_snippet = _build_embed_snippet(
        chart_id=validated_chart_id,
        width=width,
        height=height,
        metadata=metadata,
    )

    return ChartEmbedResult(html_document=html_doc, embed_snippet=embed_snippet)