"""JSON serialization and deserialization for ChartLayoutState models."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.charting.layout_models import (
    ChartLayoutState,
    IndicatorConfig,
    PanelConfig,
    ViewportParameters,
)


class ChartLayoutSerializationError(Exception):
    """Raised when chart layout serialization or deserialization fails."""


def serialize_chart_layout(state: ChartLayoutState) -> str:
    """Serialize a ChartLayoutState instance into a JSON string.

    Args:
        state: The ChartLayoutState instance to serialize.

    Returns:
        A JSON string encoding the layout state.

    Raises:
        ChartLayoutSerializationError: If state is invalid or serialization fails.
    """
    if not isinstance(state, ChartLayoutState):
        raise ChartLayoutSerializationError(
            f"Expected ChartLayoutState instance, got {type(state).__name__}"
        )

    if not isinstance(state.panels, list) or not all(
        isinstance(p, PanelConfig) for p in state.panels
    ):
        raise ChartLayoutSerializationError(
            "panels must be a list of PanelConfig instances"
        )

    if not isinstance(state.indicator_configs, list) or not all(
        isinstance(i, IndicatorConfig) for i in state.indicator_configs
    ):
        raise ChartLayoutSerializationError(
            "indicator_configs must be a list of IndicatorConfig instances"
        )

    if not isinstance(state.viewport_parameters, ViewportParameters):
        raise ChartLayoutSerializationError(
            "viewport_parameters must be a ViewportParameters instance"
        )

    try:
        data = asdict(state)
        return json.dumps(data)
    except Exception as exc:
        raise ChartLayoutSerializationError(
            f"Failed to serialize ChartLayoutState to JSON: {exc}"
        ) from exc


def deserialize_chart_layout(json_str: Any) -> ChartLayoutState:
    """Deserialize a JSON string into a ChartLayoutState instance.

    Args:
        json_str: A valid JSON string representing chart layout state.

    Returns:
        A reconstructed ChartLayoutState instance.

    Raises:
        ChartLayoutSerializationError: If the JSON is invalid, malformed, or missing required fields.
    """
    if not isinstance(json_str, str):
        raise ChartLayoutSerializationError(
            f"Input must be a string, got {type(json_str).__name__}"
        )

    try:
        payload = json.loads(json_str)
    except Exception as exc:
        raise ChartLayoutSerializationError(
            f"Malformed JSON string: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ChartLayoutSerializationError(
            f"Expected JSON object at root, got {type(payload).__name__}"
        )

    required_root_keys = ("layout_id", "panels", "indicator_configs", "viewport_parameters")
    for key in required_root_keys:
        if key not in payload:
            raise ChartLayoutSerializationError(f"Missing required root field: '{key}'")

    layout_id = payload["layout_id"]
    if not isinstance(layout_id, str):
        raise ChartLayoutSerializationError(
            f"Field 'layout_id' must be a string, got {type(layout_id).__name__}"
        )

    panels_raw = payload["panels"]
    if not isinstance(panels_raw, list):
        raise ChartLayoutSerializationError(
            f"Field 'panels' must be a list, got {type(panels_raw).__name__}"
        )

    indicator_configs_raw = payload["indicator_configs"]
    if not isinstance(indicator_configs_raw, list):
        raise ChartLayoutSerializationError(
            f"Field 'indicator_configs' must be a list, got {type(indicator_configs_raw).__name__}"
        )

    viewport_raw = payload["viewport_parameters"]
    if not isinstance(viewport_raw, dict):
        raise ChartLayoutSerializationError(
            f"Field 'viewport_parameters' must be a dict, got {type(viewport_raw).__name__}"
        )

    panels = _deserialize_panels(panels_raw)
    indicators = _deserialize_indicators(indicator_configs_raw)
    viewport = _deserialize_viewport(viewport_raw)

    return ChartLayoutState(
        layout_id=layout_id,
        panels=panels,
        indicator_configs=indicators,
        viewport_parameters=viewport,
    )


def _deserialize_panels(panels_raw: list[Any]) -> list[PanelConfig]:
    """Parse and validate panel configuration entries."""
    panels: list[PanelConfig] = []
    required_keys = ("panel_id", "height_ratio", "is_visible")

    for idx, item in enumerate(panels_raw):
        if not isinstance(item, dict):
            raise ChartLayoutSerializationError(
                f"Panel entry at index {idx} must be a dict, got {type(item).__name__}"
            )

        for key in required_keys:
            if key not in item:
                raise ChartLayoutSerializationError(
                    f"Panel entry at index {idx} missing required field '{key}'"
                )

        panel_id = item["panel_id"]
        if not isinstance(panel_id, str):
            raise ChartLayoutSerializationError(
                f"Panel 'panel_id' must be a string, got {type(panel_id).__name__}"
            )

        height_ratio = item["height_ratio"]
        if not isinstance(height_ratio, (int, float)) or isinstance(height_ratio, bool):
            raise ChartLayoutSerializationError(
                f"Panel 'height_ratio' must be numeric, got {type(height_ratio).__name__}"
            )

        is_visible = item["is_visible"]
        if not isinstance(is_visible, bool):
            raise ChartLayoutSerializationError(
                f"Panel 'is_visible' must be a bool, got {type(is_visible).__name__}"
            )

        panels.append(
            PanelConfig(
                panel_id=panel_id,
                height_ratio=float(height_ratio),
                is_visible=is_visible,
            )
        )

    return panels


def _deserialize_indicators(indicators_raw: list[Any]) -> list[IndicatorConfig]:
    """Parse and validate indicator configuration entries."""
    indicators: list[IndicatorConfig] = []
    required_keys = ("indicator_id", "indicator_type", "panel_id", "parameters", "is_visible")

    for idx, item in enumerate(indicators_raw):
        if not isinstance(item, dict):
            raise ChartLayoutSerializationError(
                f"Indicator entry at index {idx} must be a dict, got {type(item).__name__}"
            )

        for key in required_keys:
            if key not in item:
                raise ChartLayoutSerializationError(
                    f"Indicator entry at index {idx} missing required field '{key}'"
                )

        indicator_id = item["indicator_id"]
        if not isinstance(indicator_id, str):
            raise ChartLayoutSerializationError(
                f"Indicator 'indicator_id' must be a string, got {type(indicator_id).__name__}"
            )

        indicator_type = item["indicator_type"]
        if not isinstance(indicator_type, str):
            raise ChartLayoutSerializationError(
                f"Indicator 'indicator_type' must be a string, got {type(indicator_type).__name__}"
            )

        panel_id = item["panel_id"]
        if not isinstance(panel_id, str):
            raise ChartLayoutSerializationError(
                f"Indicator 'panel_id' must be a string, got {type(panel_id).__name__}"
            )

        parameters = item["parameters"]
        if not isinstance(parameters, dict):
            raise ChartLayoutSerializationError(
                f"Indicator 'parameters' must be a dict, got {type(parameters).__name__}"
            )

        is_visible = item["is_visible"]
        if not isinstance(is_visible, bool):
            raise ChartLayoutSerializationError(
                f"Indicator 'is_visible' must be a bool, got {type(is_visible).__name__}"
            )

        indicators.append(
            IndicatorConfig(
                indicator_id=indicator_id,
                indicator_type=indicator_type,
                panel_id=panel_id,
                parameters=parameters,
                is_visible=is_visible,
            )
        )

    return indicators


def _deserialize_viewport(viewport_raw: dict[str, Any]) -> ViewportParameters:
    """Parse and validate viewport parameters."""
    required_keys = ("start_index", "end_index", "auto_scale", "zoom_level")

    for key in required_keys:
        if key not in viewport_raw:
            raise ChartLayoutSerializationError(
                f"Viewport parameters missing required field '{key}'"
            )

    start_index = viewport_raw["start_index"]
    if not isinstance(start_index, int) or isinstance(start_index, bool):
        raise ChartLayoutSerializationError(
            f"Viewport 'start_index' must be an int, got {type(start_index).__name__}"
        )

    end_index = viewport_raw["end_index"]
    if not isinstance(end_index, int) or isinstance(end_index, bool):
        raise ChartLayoutSerializationError(
            f"Viewport 'end_index' must be an int, got {type(end_index).__name__}"
        )

    auto_scale = viewport_raw["auto_scale"]
    if not isinstance(auto_scale, bool):
        raise ChartLayoutSerializationError(
            f"Viewport 'auto_scale' must be a bool, got {type(auto_scale).__name__}"
        )

    zoom_level = viewport_raw["zoom_level"]
    if not isinstance(zoom_level, (int, float)) or isinstance(zoom_level, bool):
        raise ChartLayoutSerializationError(
            f"Viewport 'zoom_level' must be numeric, got {type(zoom_level).__name__}"
        )

    return ViewportParameters(
        start_index=start_index,
        end_index=end_index,
        auto_scale=auto_scale,
        zoom_level=float(zoom_level),
    )