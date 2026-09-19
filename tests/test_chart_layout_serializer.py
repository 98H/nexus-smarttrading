"""
Unit tests for Chart Layout State Serializer and Deserializer.

Specification:
Story 9.2.1: Build JSON Chart Layout State Serializer and Deserializer
- AC 1: Given a valid ChartLayoutState instance containing panels, indicator configs,
        and viewport parameters, When serialize_chart_layout(state) is invoked,
        Then it returns a valid JSON string encoding all state attributes.
- AC 2: Given a valid JSON string representing chart layout state,
        When deserialize_chart_layout(json_str) is invoked,
        Then it returns a matching ChartLayoutState instance.
- AC 3: Given an invalid JSON string or a payload with missing required fields,
        When deserialize_chart_layout(json_str) is invoked,
        Then it raises ChartLayoutSerializationError.

Target Modules:
- src/charting/layout_models.py
- src/charting/layout_serializer.py
"""

import json
from typing import Any, Dict

import pytest

from src.charting.layout_models import (
    ChartLayoutState,
    IndicatorConfig,
    PanelConfig,
    ViewportParameters,
)
from src.charting.layout_serializer import (
    ChartLayoutSerializationError,
    deserialize_chart_layout,
    serialize_chart_layout,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_viewport() -> ViewportParameters:
    """Fixture providing a standard ViewportParameters instance."""
    return ViewportParameters(
        start_index=0,
        end_index=150,
        auto_scale=True,
        zoom_level=1.25,
    )


@pytest.fixture
def sample_panels() -> list[PanelConfig]:
    """Fixture providing a list of PanelConfig instances."""
    return [
        PanelConfig(panel_id="panel-main", height_ratio=0.75, is_visible=True),
        PanelConfig(panel_id="panel-sub1", height_ratio=0.25, is_visible=True),
    ]


@pytest.fixture
def sample_indicators() -> list[IndicatorConfig]:
    """Fixture providing a list of IndicatorConfig instances with nested parameters."""
    return [
        IndicatorConfig(
            indicator_id="ind-sma-1",
            indicator_type="SMA",
            panel_id="panel-main",
            parameters={"period": 20, "source": "close", "color": "#00FF00"},
            is_visible=True,
        ),
        IndicatorConfig(
            indicator_id="ind-rsi-1",
            indicator_type="RSI",
            panel_id="panel-sub1",
            parameters={
                "period": 14,
                "overbought": 70.0,
                "oversold": 30.0,
                "nested_style": {"linewidth": 1.5, "line_dash": "solid"},
            },
            is_visible=False,
        ),
    ]


@pytest.fixture
def sample_chart_layout_state(
    sample_panels: list[PanelConfig],
    sample_indicators: list[IndicatorConfig],
    sample_viewport: ViewportParameters,
) -> ChartLayoutState:
    """Fixture providing a complete ChartLayoutState instance."""
    return ChartLayoutState(
        layout_id="layout-standard-001",
        panels=sample_panels,
        indicator_configs=sample_indicators,
        viewport_parameters=sample_viewport,
    )


@pytest.fixture
def valid_layout_dict() -> Dict[str, Any]:
    """Fixture providing a dictionary representation of a valid chart layout."""
    return {
        "layout_id": "layout-test-dict",
        "panels": [
            {"panel_id": "p1", "height_ratio": 0.8, "is_visible": True},
            {"panel_id": "p2", "height_ratio": 0.2, "is_visible": False},
        ],
        "indicator_configs": [
            {
                "indicator_id": "ind-ema",
                "indicator_type": "EMA",
                "panel_id": "p1",
                "parameters": {"length": 50, "color": "#FF0000"},
                "is_visible": True,
            }
        ],
        "viewport_parameters": {
            "start_index": 10,
            "end_index": 200,
            "auto_scale": False,
            "zoom_level": 2.0,
        },
    }


# ============================================================================
# AC 1: Serialization Tests
# ============================================================================


class TestSerializeChartLayout:
    """Tests verifying serialization of ChartLayoutState to JSON string."""

    def test_serialize_valid_full_layout_returns_valid_json_string(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1: Given a valid ChartLayoutState, serialize returns a valid JSON string."""
        serialized = serialize_chart_layout(sample_chart_layout_state)

        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_serialize_encodes_all_root_attributes(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1: Serialized JSON includes layout_id, panels, indicators, and viewport."""
        serialized = serialize_chart_layout(sample_chart_layout_state)
        parsed = json.loads(serialized)

        assert parsed["layout_id"] == "layout-standard-001"
        assert "panels" in parsed
        assert "indicator_configs" in parsed
        assert "viewport_parameters" in parsed

    def test_serialize_encodes_panel_attributes_correctly(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1: Verify panel properties are preserved in the serialized payload."""
        serialized = serialize_chart_layout(sample_chart_layout_state)
        parsed = json.loads(serialized)
        panels = parsed["panels"]

        assert len(panels) == 2
        assert panels[0] == {
            "panel_id": "panel-main",
            "height_ratio": 0.75,
            "is_visible": True,
        }
        assert panels[1] == {
            "panel_id": "panel-sub1",
            "height_ratio": 0.25,
            "is_visible": True,
        }

    def test_serialize_encodes_indicator_configs_with_nested_parameters(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1: Indicator configurations with nested parameters are accurately serialized."""
        serialized = serialize_chart_layout(sample_chart_layout_state)
        parsed = json.loads(serialized)
        indicators = parsed["indicator_configs"]

        assert len(indicators) == 2
        rsi_ind = next(i for i in indicators if i["indicator_id"] == "ind-rsi-1")
        assert rsi_ind["indicator_type"] == "RSI"
        assert rsi_ind["panel_id"] == "panel-sub1"
        assert rsi_ind["is_visible"] is False
        assert rsi_ind["parameters"]["period"] == 14
        assert rsi_ind["parameters"]["overbought"] == 70.0
        assert rsi_ind["parameters"]["nested_style"]["linewidth"] == 1.5

    def test_serialize_encodes_viewport_parameters_correctly(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1: Viewport parameters are preserved in the serialized payload."""
        serialized = serialize_chart_layout(sample_chart_layout_state)
        parsed = json.loads(serialized)
        viewport = parsed["viewport_parameters"]

        assert viewport == {
            "start_index": 0,
            "end_index": 150,
            "auto_scale": True,
            "zoom_level": 1.25,
        }

    def test_serialize_empty_panels_and_indicators(
        self, sample_viewport: ViewportParameters
    ) -> None:
        """AC 1: An empty layout with no panels or indicators serializes cleanly."""
        state = ChartLayoutState(
            layout_id="layout-empty",
            panels=[],
            indicator_configs=[],
            viewport_parameters=sample_viewport,
        )

        serialized = serialize_chart_layout(state)
        parsed = json.loads(serialized)

        assert parsed["panels"] == []
        assert parsed["indicator_configs"] == []

    @pytest.mark.parametrize(
        "invalid_state",
        [
            None,
            "not-a-state-instance",
            123,
            {"layout_id": "dict_instead_of_state"},
            [1, 2, 3],
        ],
    )
    def test_serialize_invalid_instance_raises_serialization_error(
        self, invalid_state: Any
    ) -> None:
        """AC 1/3: Invoking serialize with an invalid instance raises ChartLayoutSerializationError."""
        with pytest.raises(ChartLayoutSerializationError):
            serialize_chart_layout(invalid_state)

    def test_serialize_non_json_serializable_content_raises_error(
        self, sample_viewport: ViewportParameters
    ) -> None:
        """AC 1/3: Non-serializable parameters within config trigger serialization error."""
        non_serializable_indicator = IndicatorConfig(
            indicator_id="ind-unserializable",
            indicator_type="CUSTOM",
            panel_id="panel-main",
            parameters={"unserializable_object": object()},
            is_visible=True,
        )
        state = ChartLayoutState(
            layout_id="layout-err",
            panels=[],
            indicator_configs=[non_serializable_indicator],
            viewport_parameters=sample_viewport,
        )

        with pytest.raises(ChartLayoutSerializationError):
            serialize_chart_layout(state)


# ============================================================================
# AC 2: Deserialization Tests
# ============================================================================


class TestDeserializeChartLayout:
    """Tests verifying deserialization of JSON strings into ChartLayoutState instances."""

    def test_deserialize_valid_json_returns_chart_layout_state(
        self, valid_layout_dict: Dict[str, Any]
    ) -> None:
        """AC 2: Given a valid JSON string, deserialize returns matching ChartLayoutState."""
        json_str = json.dumps(valid_layout_dict)
        state = deserialize_chart_layout(json_str)

        assert isinstance(state, ChartLayoutState)
        assert state.layout_id == valid_layout_dict["layout_id"]

    def test_deserialize_instantiates_panel_configs(
        self, valid_layout_dict: Dict[str, Any]
    ) -> None:
        """AC 2: Deserialized panels are proper PanelConfig model instances."""
        json_str = json.dumps(valid_layout_dict)
        state = deserialize_chart_layout(json_str)

        assert len(state.panels) == 2
        for panel in state.panels:
            assert isinstance(panel, PanelConfig)

        p1 = state.panels[0]
        assert p1.panel_id == "p1"
        assert p1.height_ratio == 0.8
        assert p1.is_visible is True

        p2 = state.panels[1]
        assert p2.panel_id == "p2"
        assert p2.height_ratio == 0.2
        assert p2.is_visible is False

    def test_deserialize_instantiates_indicator_configs(
        self, valid_layout_dict: Dict[str, Any]
    ) -> None:
        """AC 2: Deserialized indicators are proper IndicatorConfig model instances."""
        json_str = json.dumps(valid_layout_dict)
        state = deserialize_chart_layout(json_str)

        assert len(state.indicator_configs) == 1
        ind = state.indicator_configs[0]
        assert isinstance(ind, IndicatorConfig)
        assert ind.indicator_id == "ind-ema"
        assert ind.indicator_type == "EMA"
        assert ind.panel_id == "p1"
        assert ind.parameters == {"length": 50, "color": "#FF0000"}
        assert ind.is_visible is True

    def test_deserialize_instantiates_viewport_parameters(
        self, valid_layout_dict: Dict[str, Any]
    ) -> None:
        """AC 2: Deserialized viewport is a proper ViewportParameters instance."""
        json_str = json.dumps(valid_layout_dict)
        state = deserialize_chart_layout(json_str)

        assert isinstance(state.viewport_parameters, ViewportParameters)
        assert state.viewport_parameters.start_index == 10
        assert state.viewport_parameters.end_index == 200
        assert state.viewport_parameters.auto_scale is False
        assert state.viewport_parameters.zoom_level == 2.0

    def test_deserialize_preserves_panel_and_indicator_order(self) -> None:
        """AC 2: Sequential order of panels and indicators is preserved during deserialization."""
        payload = {
            "layout_id": "layout-ordered",
            "panels": [
                {"panel_id": "panel-z", "height_ratio": 0.3, "is_visible": True},
                {"panel_id": "panel-a", "height_ratio": 0.4, "is_visible": True},
                {"panel_id": "panel-m", "height_ratio": 0.3, "is_visible": True},
            ],
            "indicator_configs": [
                {
                    "indicator_id": f"ind-{idx}",
                    "indicator_type": "TYPE",
                    "panel_id": "panel-z",
                    "parameters": {},
                    "is_visible": True,
                }
                for idx in range(5)
            ],
            "viewport_parameters": {
                "start_index": 0,
                "end_index": 100,
                "auto_scale": True,
                "zoom_level": 1.0,
            },
        }
        state = deserialize_chart_layout(json.dumps(payload))

        assert [p.panel_id for p in state.panels] == ["panel-z", "panel-a", "panel-m"]
        assert [i.indicator_id for i in state.indicator_configs] == [
            f"ind-{idx}" for idx in range(5)
        ]


# ============================================================================
# Round-Trip Serialization & Deserialization
# ============================================================================


class TestRoundTripSerialization:
    """Tests ensuring full round-trip fidelity between model and JSON string."""

    def test_round_trip_equality(
        self, sample_chart_layout_state: ChartLayoutState
    ) -> None:
        """AC 1 & AC 2: Serializing and then deserializing returns an equivalent instance."""
        serialized_json = serialize_chart_layout(sample_chart_layout_state)
        deserialized_state = deserialize_chart_layout(serialized_json)

        assert deserialized_state == sample_chart_layout_state

    def test_round_trip_with_empty_collections(
        self, sample_viewport: ViewportParameters
    ) -> None:
        """AC 1 & AC 2: Round-trip preserves empty collections."""
        initial_state = ChartLayoutState(
            layout_id="layout-round-trip-empty",
            panels=[],
            indicator_configs=[],
            viewport_parameters=sample_viewport,
        )

        serialized = serialize_chart_layout(initial_state)
        restored = deserialize_chart_layout(serialized)

        assert restored == initial_state
        assert restored.panels == []
        assert restored.indicator_configs == []

    def test_round_trip_with_special_characters_and_types(self) -> None:
        """AC 1 & AC 2: Unicode, booleans, negative numbers, and floats survive round-trip."""
        state = ChartLayoutState(
            layout_id="layout-αβγ-123_#@!",
            panels=[
                PanelConfig(
                    panel_id="pane-メイン-0", height_ratio=0.55555, is_visible=True
                )
            ],
            indicator_configs=[
                IndicatorConfig(
                    indicator_id="ind-1",
                    indicator_type="MACD",
                    panel_id="pane-メイン-0",
                    parameters={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                        "offset": -2.5,
                        "enabled_flags": [True, False, True],
                        "metadata": {"title": "指数平滑移動平均", "threshold": None},
                    },
                    is_visible=False,
                )
            ],
            viewport_parameters=ViewportParameters(
                start_index=0,
                end_index=5000,
                auto_scale=False,
                zoom_level=0.0001,
            ),
        )

        serialized = serialize_chart_layout(state)
        restored = deserialize_chart_layout(serialized)

        assert restored == state


# ============================================================================
# AC 3: Error Handling and Validation Tests
# ============================================================================


class TestDeserializationErrors:
    """Tests verifying ChartLayoutSerializationError is raised on invalid inputs."""

    @pytest.mark.parametrize(
        "malformed_json",
        [
            "",
            "   ",
            "{not a valid json}",
            '{"layout_id": "incomplete"',
            '{"layout_id": "val", "panels": [}',
            "<div>XML is not JSON</div>",
            "None",
            "undefined",
        ],
    )
    def test_deserialize_malformed_syntax_raises_error(
        self, malformed_json: str
    ) -> None:
        """AC 3: Invalid JSON syntax raises ChartLayoutSerializationError."""
        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(malformed_json)

    @pytest.mark.parametrize(
        "non_string_input",
        [
            None,
            123,
            45.67,
            True,
            ["json", "string"],
            {"layout_id": "direct_dict"},
            b'{"valid": "bytes"}',
        ],
    )
    def test_deserialize_non_string_type_raises_error(
        self, non_string_input: Any
    ) -> None:
        """AC 3: Passing non-string inputs to deserialize raises ChartLayoutSerializationError."""
        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(non_string_input)

    @pytest.mark.parametrize(
        "json_primitive_literal",
        [
            "123",
            "45.67",
            '"just a json string"',
            "true",
            "false",
            "null",
            "[1, 2, 3]",
        ],
    )
    def test_deserialize_non_object_json_raises_error(
        self, json_primitive_literal: str
    ) -> None:
        """AC 3: Valid JSON that is not an object raises ChartLayoutSerializationError."""
        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json_primitive_literal)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "layout_id",
            "panels",
            "indicator_configs",
            "viewport_parameters",
        ],
    )
    def test_deserialize_missing_root_required_fields_raises_error(
        self, valid_layout_dict: Dict[str, Any], missing_key: str
    ) -> None:
        """AC 3: Missing any required root field raises ChartLayoutSerializationError."""
        invalid_payload = dict(valid_layout_dict)
        del invalid_payload[missing_key]

        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json.dumps(invalid_payload))

    @pytest.mark.parametrize(
        "field_name,invalid_value",
        [
            ("panels", "not-a-list"),
            ("panels", 123),
            ("panels", None),
            ("panels", {"panel_id": "p1"}),
            ("indicator_configs", "not-a-list"),
            ("indicator_configs", 456),
            ("indicator_configs", None),
            ("indicator_configs", {"indicator_id": "i1"}),
            ("viewport_parameters", "not-a-dict"),
            ("viewport_parameters", [1, 2, 3]),
            ("viewport_parameters", None),
            ("viewport_parameters", 789),
            ("layout_id", 12345),
            ("layout_id", None),
            ("layout_id", []),
        ],
    )
    def test_deserialize_invalid_root_field_types_raises_error(
        self,
        valid_layout_dict: Dict[str, Any],
        field_name: str,
        invalid_value: Any,
    ) -> None:
        """AC 3: Root fields having invalid data types raise ChartLayoutSerializationError."""
        invalid_payload = dict(valid_layout_dict)
        invalid_payload[field_name] = invalid_value

        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json.dumps(invalid_payload))

    @pytest.mark.parametrize(
        "invalid_panel",
        [
            "string_instead_of_dict",
            123,
            {"height_ratio": 0.5, "is_visible": True},  # Missing panel_id
            {"panel_id": 123, "height_ratio": 0.5, "is_visible": True},  # Invalid panel_id type
            {"panel_id": "p1", "height_ratio": "high", "is_visible": True},  # Invalid height_ratio
        ],
    )
    def test_deserialize_invalid_panel_entry_raises_error(
        self, valid_layout_dict: Dict[str, Any], invalid_panel: Any
    ) -> None:
        """AC 3: Invalid panel items in the panels list raise ChartLayoutSerializationError."""
        payload = dict(valid_layout_dict)
        payload["panels"] = [invalid_panel]

        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json.dumps(payload))

    @pytest.mark.parametrize(
        "invalid_indicator",
        [
            "string_instead_of_dict",
            # Missing indicator_type
            {
                "indicator_id": "i1",
                "panel_id": "p1",
                "parameters": {},
                "is_visible": True,
            },
            # Missing panel_id
            {
                "indicator_id": "i1",
                "indicator_type": "SMA",
                "parameters": {},
                "is_visible": True,
            },
            # Missing indicator_id
            {
                "indicator_type": "SMA",
                "panel_id": "p1",
                "parameters": {},
                "is_visible": True,
            },
            # Parameters is not a dict
            {
                "indicator_id": "i1",
                "indicator_type": "SMA",
                "panel_id": "p1",
                "parameters": "invalid_params",
                "is_visible": True,
            },
        ],
    )
    def test_deserialize_invalid_indicator_entry_raises_error(
        self, valid_layout_dict: Dict[str, Any], invalid_indicator: Any
    ) -> None:
        """AC 3: Invalid indicator items raise ChartLayoutSerializationError."""
        payload = dict(valid_layout_dict)
        payload["indicator_configs"] = [invalid_indicator]

        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json.dumps(payload))

    @pytest.mark.parametrize(
        "invalid_viewport",
        [
            # Missing end_index
            {"start_index": 0, "auto_scale": True, "zoom_level": 1.0},
            # Missing start_index
            {"end_index": 100, "auto_scale": True, "zoom_level": 1.0},
            # start_index is not an integer
            {
                "start_index": "zero",
                "end_index": 100,
                "auto_scale": True,
                "zoom_level": 1.0,
            },
            # zoom_level is not a number
            {
                "start_index": 0,
                "end_index": 100,
                "auto_scale": True,
                "zoom_level": "x2",
            },
        ],
    )
    def test_deserialize_invalid_viewport_parameters_raises_error(
        self, valid_layout_dict: Dict[str, Any], invalid_viewport: Any
    ) -> None:
        """AC 3: Missing required viewport fields or wrong types raise ChartLayoutSerializationError."""
        payload = dict(valid_layout_dict)
        payload["viewport_parameters"] = invalid_viewport

        with pytest.raises(ChartLayoutSerializationError):
            deserialize_chart_layout(json.dumps(payload))


# ============================================================================
# Model Integrity and Equality Tests
# ============================================================================


class TestLayoutModels:
    """Tests verifying layout model initialization, attribute accessibility, and equality."""

    def test_panel_config_equality(self) -> None:
        """Verify PanelConfig value equality."""
        p1 = PanelConfig(panel_id="panel_1", height_ratio=0.5, is_visible=True)
        p2 = PanelConfig(panel_id="panel_1", height_ratio=0.5, is_visible=True)
        p3 = PanelConfig(panel_id="panel_2", height_ratio=0.5, is_visible=True)

        assert p1 == p2
        assert p1 != p3

    def test_indicator_config_equality(self) -> None:
        """Verify IndicatorConfig value equality."""
        i1 = IndicatorConfig(
            indicator_id="ind_1",
            indicator_type="EMA",
            panel_id="panel_1",
            parameters={"period": 10},
            is_visible=True,
        )
        i2 = IndicatorConfig(
            indicator_id="ind_1",
            indicator_type="EMA",
            panel_id="panel_1",
            parameters={"period": 10},
            is_visible=True,
        )
        i3 = IndicatorConfig(
            indicator_id="ind_1",
            indicator_type="EMA",
            panel_id="panel_1",
            parameters={"period": 20},
            is_visible=True,
        )

        assert i1 == i2
        assert i1 != i3

    def test_viewport_parameters_equality(self) -> None:
        """Verify ViewportParameters value equality."""
        v1 = ViewportParameters(
            start_index=0, end_index=100, auto_scale=True, zoom_level=1.0
        )
        v2 = ViewportParameters(
            start_index=0, end_index=100, auto_scale=True, zoom_level=1.0
        )
        v3 = ViewportParameters(
            start_index=5, end_index=100, auto_scale=True, zoom_level=1.0
        )

        assert v1 == v2
        assert v1 != v3

    def test_chart_layout_state_equality(
        self,
        sample_panels: list[PanelConfig],
        sample_indicators: list[IndicatorConfig],
        sample_viewport: ViewportParameters,
    ) -> None:
        """Verify ChartLayoutState deep value equality."""
        s1 = ChartLayoutState(
            layout_id="layout-eq",
            panels=sample_panels,
            indicator_configs=sample_indicators,
            viewport_parameters=sample_viewport,
        )
        s2 = ChartLayoutState(
            layout_id="layout-eq",
            panels=sample_panels,
            indicator_configs=sample_indicators,
            viewport_parameters=sample_viewport,
        )
        s3 = ChartLayoutState(
            layout_id="layout-different-id",
            panels=sample_panels,
            indicator_configs=sample_indicators,
            viewport_parameters=sample_viewport,
        )

        assert s1 == s2
        assert s1 != s3