"""
Unit tests for the Global Theme Configuration Engine.
Covers Story 9.3.1: Dark, Light, Institutional Custom theme resolution, fallback behavior,
and custom institutional token overrides with mandatory canvas/grid preservation.
"""

import copy
from typing import Any, Dict
import pytest

from src.theme.models import (
    CanvasSettings,
    GridLineSettings,
    ScaleSettings,
    ThemeConfig,
)
from src.theme.engine import ThemeEngine


@pytest.fixture
def engine() -> ThemeEngine:
    """Fixture providing a freshly instantiated ThemeEngine."""
    return ThemeEngine()


@pytest.fixture
def institutional_tokens() -> Dict[str, Any]:
    """Sample institutional custom tokens overriding brand styling."""
    return {
        "css_variables": {
            "--color-primary": "#0A2540",
            "--color-accent": "#635BFF",
            "--color-text-main": "#F8FAFC",
        },
        "canvas": {
            "background_color": "#051329",
            "grid_lines": {
                "color": "rgba(99, 91, 255, 0.2)",
            },
        },
    }


# ============================================================================
# Acceptance Criteria 1: Valid Theme Resolution ('dark', 'light', 'institutional_custom')
# ============================================================================


class TestValidThemeResolution:
    """Tests resolving configurations for standard and pre-defined valid themes."""

    @pytest.mark.parametrize(
        "theme_id",
        ["dark", "light", "institutional_custom"],
    )
    def test_resolve_valid_theme_returns_theme_config_instance(
        self, engine: ThemeEngine, theme_id: str
    ):
        config = engine.resolve(theme_id)
        assert isinstance(config, ThemeConfig)
        assert config.identifier == theme_id

    @pytest.mark.parametrize(
        "theme_id",
        ["dark", "light", "institutional_custom"],
    )
    def test_resolve_valid_theme_returns_complete_canvas_settings(
        self, engine: ThemeEngine, theme_id: str
    ):
        config = engine.resolve(theme_id)
        canvas = config.canvas

        # Verify CanvasSettings structure
        assert isinstance(canvas, CanvasSettings)
        assert isinstance(canvas.background_color, str)
        assert len(canvas.background_color.strip()) > 0

        # Verify GridLineSettings structure
        assert isinstance(canvas.grid_lines, GridLineSettings)
        assert isinstance(canvas.grid_lines.color, str)
        assert isinstance(canvas.grid_lines.width, (int, float))
        assert canvas.grid_lines.width > 0
        assert isinstance(canvas.grid_lines.style, str)

        # Verify ScaleSettings structure
        assert isinstance(canvas.scales, ScaleSettings)
        assert isinstance(canvas.scales.tick_color, str)
        assert isinstance(canvas.scales.font_color, str)
        assert isinstance(canvas.scales.font_size, (int, float))
        assert canvas.scales.font_size > 0

    @pytest.mark.parametrize(
        "theme_id",
        ["dark", "light", "institutional_custom"],
    )
    def test_resolve_valid_theme_returns_css_variables_map(
        self, engine: ThemeEngine, theme_id: str
    ):
        config = engine.resolve(theme_id)
        css_vars = config.css_variables

        assert isinstance(css_vars, dict)
        assert len(css_vars) > 0

        # Enforce that all keys are valid CSS custom property names
        for key, value in css_vars.items():
            assert key.startswith("--")
            assert isinstance(value, str)
            assert len(value.strip()) > 0

    def test_dark_and_light_themes_have_distinct_parameters(
        self, engine: ThemeEngine
    ):
        dark_config = engine.resolve("dark")
        light_config = engine.resolve("light")

        # Distinct background canvas colors
        assert dark_config.canvas.background_color != light_config.canvas.background_color

        # Distinct grid line colors
        assert dark_config.canvas.grid_lines.color != light_config.canvas.grid_lines.color

        # Distinct CSS variables (e.g., primary background / surface)
        assert dark_config.css_variables != light_config.css_variables


# ============================================================================
# Acceptance Criteria 2: Fallback to Default 'dark' Theme
# ============================================================================


class TestThemeResolutionFallback:
    """Tests fallback to the default 'dark' theme configuration on invalid/undefined inputs."""

    @pytest.mark.parametrize(
        "invalid_id",
        [
            None,
            "",
            "   ",
            "unknown_theme",
            "solarized",
            "neon_synthwave",
            12345,
        ],
    )
    def test_invalid_or_undefined_identifier_falls_back_to_dark_theme(
        self, engine: ThemeEngine, invalid_id: Any
    ):
        resolved_config = engine.resolve(invalid_id)
        dark_config = engine.resolve("dark")

        assert isinstance(resolved_config, ThemeConfig)
        # Fallback configuration must match default dark configuration
        assert resolved_config.canvas.background_color == dark_config.canvas.background_color
        assert resolved_config.canvas.grid_lines.color == dark_config.canvas.grid_lines.color
        assert resolved_config.canvas.grid_lines.width == dark_config.canvas.grid_lines.width
        assert resolved_config.canvas.scales.tick_color == dark_config.canvas.scales.tick_color
        assert resolved_config.css_variables == dark_config.css_variables

    def test_fallback_maintains_default_dark_identifier(self, engine: ThemeEngine):
        resolved_config = engine.resolve("non_existent_identifier")
        assert resolved_config.identifier == "dark"


# ============================================================================
# Acceptance Criteria 3: Institutional Custom Theme Registration & Override
# ============================================================================


class TestInstitutionalCustomThemeRegistration:
    """Tests registering custom institutional styling tokens with base parameter overrides."""

    def test_register_custom_theme_overrides_css_variables(
        self, engine: ThemeEngine, institutional_tokens: Dict[str, Any]
    ):
        custom_id = "institutional_custom"
        registered_config = engine.register_custom_theme(
            identifier=custom_id,
            tokens=institutional_tokens,
            base_theme="dark",
        )

        for var_name, expected_val in institutional_tokens["css_variables"].items():
            assert registered_config.css_variables[var_name] == expected_val

    def test_register_custom_theme_overrides_canvas_background(
        self, engine: ThemeEngine, institutional_tokens: Dict[str, Any]
    ):
        custom_id = "institutional_custom"
        registered_config = engine.register_custom_theme(
            identifier=custom_id,
            tokens=institutional_tokens,
            base_theme="dark",
        )

        expected_bg = institutional_tokens["canvas"]["background_color"]
        assert registered_config.canvas.background_color == expected_bg

    def test_register_custom_theme_preserves_mandatory_grid_definitions(
        self, engine: ThemeEngine
    ):
        # Tokens omit grid width and style, but override grid color
        tokens: Dict[str, Any] = {
            "canvas": {
                "grid_lines": {
                    "color": "#FFD700",
                }
            }
        }
        base_dark = engine.resolve("dark")

        registered_config = engine.register_custom_theme(
            identifier="hedge_fund_custom",
            tokens=tokens,
            base_theme="dark",
        )

        # Overridden field
        assert registered_config.canvas.grid_lines.color == "#FFD700"
        # Preserved mandatory base definitions
        assert registered_config.canvas.grid_lines.width == base_dark.canvas.grid_lines.width
        assert registered_config.canvas.grid_lines.style == base_dark.canvas.grid_lines.style

    def test_register_custom_theme_preserves_scales_when_omitted(
        self, engine: ThemeEngine
    ):
        # Tokens do not define any scale settings
        tokens: Dict[str, Any] = {
            "css_variables": {
                "--color-brand": "#123456",
            }
        }
        base_dark = engine.resolve("dark")

        registered_config = engine.register_custom_theme(
            identifier="corp_custom",
            tokens=tokens,
            base_theme="dark",
        )

        # Scales must be fully retained from base
        assert registered_config.canvas.scales.tick_color == base_dark.canvas.scales.tick_color
        assert registered_config.canvas.scales.font_color == base_dark.canvas.scales.font_color
        assert registered_config.canvas.scales.font_size == base_dark.canvas.scales.font_size

    def test_register_custom_theme_with_light_base(self, engine: ThemeEngine):
        tokens: Dict[str, Any] = {
            "canvas": {
                "background_color": "#FAFAFA",
            }
        }
        base_light = engine.resolve("light")

        registered_config = engine.register_custom_theme(
            identifier="light_institutional",
            tokens=tokens,
            base_theme="light",
        )

        assert registered_config.canvas.background_color == "#FAFAFA"
        # Scales and grid lines preserved from 'light'
        assert registered_config.canvas.grid_lines.color == base_light.canvas.grid_lines.color
        assert registered_config.canvas.scales.tick_color == base_light.canvas.scales.tick_color

    def test_registered_custom_theme_is_subsequently_resolvable(
        self, engine: ThemeEngine, institutional_tokens: Dict[str, Any]
    ):
        custom_id = "institutional_custom"
        engine.register_custom_theme(
            identifier=custom_id,
            tokens=institutional_tokens,
            base_theme="dark",
        )

        resolved_config = engine.resolve(custom_id)
        assert resolved_config.identifier == custom_id
        assert (
            resolved_config.canvas.background_color
            == institutional_tokens["canvas"]["background_color"]
        )
        assert (
            resolved_config.css_variables["--color-primary"]
            == institutional_tokens["css_variables"]["--color-primary"]
        )


# ============================================================================
# Robustness, Validation, and Immutability Tests
# ============================================================================


class TestThemeEngineValidationAndImmutability:
    """Tests validation rules, error handling, and state immutability."""

    def test_register_with_non_dict_tokens_raises_type_error(
        self, engine: ThemeEngine
    ):
        with pytest.raises(TypeError):
            engine.register_custom_theme(
                identifier="invalid_tokens",
                tokens="not-a-dict",  # type: ignore[arg-type]
                base_theme="dark",
            )

    def test_register_with_empty_identifier_raises_value_error(
        self, engine: ThemeEngine
    ):
        with pytest.raises(ValueError):
            engine.register_custom_theme(
                identifier="",
                tokens={"canvas": {"background_color": "#FFFFFF"}},
                base_theme="dark",
            )

    def test_register_with_unsupported_base_theme_raises_value_error(
        self, engine: ThemeEngine
    ):
        with pytest.raises(ValueError):
            engine.register_custom_theme(
                identifier="custom_broken_base",
                tokens={},
                base_theme="non_existent_base",
            )

    def test_resolving_theme_does_not_mutate_internal_state(
        self, engine: ThemeEngine
    ):
        dark_1 = engine.resolve("dark")
        # Mutate the returned config copy
        dark_1.css_variables["--color-primary"] = "#MUTATED"
        dark_1.canvas.background_color = "#MUTATED"

        dark_2 = engine.resolve("dark")
        assert dark_2.css_variables["--color-primary"] != "#MUTATED"
        assert dark_2.canvas.background_color != "#MUTATED"

    def test_registering_custom_theme_does_not_mutate_base_theme(
        self, engine: ThemeEngine
    ):
        dark_before = copy.deepcopy(engine.resolve("dark"))

        engine.register_custom_theme(
            identifier="institutional_variant",
            tokens={
                "canvas": {
                    "background_color": "#990000",
                    "grid_lines": {"color": "#FF0000"},
                },
                "css_variables": {
                    "--canvas-bg": "#990000",
                },
            },
            base_theme="dark",
        )

        dark_after = engine.resolve("dark")
        assert dark_after.canvas.background_color == dark_before.canvas.background_color
        assert dark_after.canvas.grid_lines.color == dark_before.canvas.grid_lines.color
        assert dark_after.css_variables == dark_before.css_variables