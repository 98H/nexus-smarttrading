"""Global Theme Configuration Engine for dark, light, and institutional custom themes."""

import copy
from typing import Any, Dict

from src.theme.models import (
    CanvasSettings,
    GridLineSettings,
    ScaleSettings,
    ThemeConfig,
)


def _build_default_themes() -> Dict[str, ThemeConfig]:
    """Constructs the standard pre-configured themes: dark, light, and institutional custom."""
    dark_theme = ThemeConfig(
        identifier="dark",
        canvas=CanvasSettings(
            background_color="#0D1117",
            grid_lines=GridLineSettings(
                color="#30363D",
                width=1.0,
                style="solid",
            ),
            scales=ScaleSettings(
                tick_color="#8B949E",
                font_color="#C9D1D9",
                font_size=12.0,
            ),
        ),
        css_variables={
            "--color-primary": "#58A6FF",
            "--color-background": "#0D1117",
            "--color-surface": "#161B22",
            "--color-text": "#F0F6FC",
            "--color-accent": "#1F6FEB",
            "--color-grid": "#30363D",
        },
    )

    light_theme = ThemeConfig(
        identifier="light",
        canvas=CanvasSettings(
            background_color="#FFFFFF",
            grid_lines=GridLineSettings(
                color="#E1E4E8",
                width=1.0,
                style="solid",
            ),
            scales=ScaleSettings(
                tick_color="#6E7781",
                font_color="#24292F",
                font_size=12.0,
            ),
        ),
        css_variables={
            "--color-primary": "#0969DA",
            "--color-background": "#FFFFFF",
            "--color-surface": "#F6F8FA",
            "--color-text": "#24292F",
            "--color-accent": "#218BFF",
            "--color-grid": "#E1E4E8",
        },
    )

    institutional_theme = ThemeConfig(
        identifier="institutional_custom",
        canvas=CanvasSettings(
            background_color="#051329",
            grid_lines=GridLineSettings(
                color="rgba(99, 91, 255, 0.2)",
                width=1.0,
                style="solid",
            ),
            scales=ScaleSettings(
                tick_color="#635BFF",
                font_color="#F8FAFC",
                font_size=12.0,
            ),
        ),
        css_variables={
            "--color-primary": "#0A2540",
            "--color-accent": "#635BFF",
            "--color-text-main": "#F8FAFC",
            "--color-background": "#051329",
            "--color-surface": "#0E1E38",
            "--color-grid": "rgba(99, 91, 255, 0.2)",
        },
    )

    return {
        "dark": dark_theme,
        "light": light_theme,
        "institutional_custom": institutional_theme,
    }


class ThemeEngine:
    """Manages theme resolution, fallback defaults, and custom token registration."""

    DEFAULT_THEME = "dark"

    def __init__(self) -> None:
        self._themes: Dict[str, ThemeConfig] = _build_default_themes()

    def resolve(self, identifier: Any) -> ThemeConfig:
        """Resolves theme configuration for an identifier, falling back to 'dark' if invalid."""
        if not isinstance(identifier, str) or not identifier.strip() or identifier not in self._themes:
            return copy.deepcopy(self._themes[self.DEFAULT_THEME])

        return copy.deepcopy(self._themes[identifier])

    def register_custom_theme(
        self,
        identifier: str,
        tokens: Dict[str, Any],
        base_theme: str = "dark",
    ) -> ThemeConfig:
        """Registers a custom theme by overriding base theme parameters with custom tokens."""
        if not isinstance(tokens, dict):
            raise TypeError(f"Tokens must be a dict, got {type(tokens).__name__}")

        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Theme identifier must be a non-empty string.")

        if not isinstance(base_theme, str) or base_theme not in self._themes:
            raise ValueError(f"Base theme '{base_theme}' is not supported.")

        base = self._themes[base_theme]

        # Merge CSS variables on top of base
        css_variables = dict(base.css_variables)
        custom_css = tokens.get("css_variables")
        if custom_css is not None:
            if not isinstance(custom_css, dict):
                raise TypeError("css_variables must be a dict.")
            css_variables.update(custom_css)

        # Merge canvas settings while preserving mandatory base parameters
        bg_color = base.canvas.background_color
        gl_color = base.canvas.grid_lines.color
        gl_width = base.canvas.grid_lines.width
        gl_style = base.canvas.grid_lines.style
        scale_tick = base.canvas.scales.tick_color
        scale_font = base.canvas.scales.font_color
        scale_size = base.canvas.scales.font_size

        canvas_tokens = tokens.get("canvas")
        if canvas_tokens is not None:
            if not isinstance(canvas_tokens, dict):
                raise TypeError("canvas must be a dict.")

            if "background_color" in canvas_tokens:
                bg_color = canvas_tokens["background_color"]

            grid_tokens = canvas_tokens.get("grid_lines")
            if grid_tokens is not None:
                if not isinstance(grid_tokens, dict):
                    raise TypeError("grid_lines must be a dict.")
                if "color" in grid_tokens:
                    gl_color = grid_tokens["color"]
                if "width" in grid_tokens:
                    gl_width = grid_tokens["width"]
                if "style" in grid_tokens:
                    gl_style = grid_tokens["style"]

            scale_tokens = canvas_tokens.get("scales")
            if scale_tokens is not None:
                if not isinstance(scale_tokens, dict):
                    raise TypeError("scales must be a dict.")
                if "tick_color" in scale_tokens:
                    scale_tick = scale_tokens["tick_color"]
                if "font_color" in scale_tokens:
                    scale_font = scale_tokens["font_color"]
                if "font_size" in scale_tokens:
                    scale_size = scale_tokens["font_size"]

        new_config = ThemeConfig(
            identifier=identifier,
            canvas=CanvasSettings(
                background_color=bg_color,
                grid_lines=GridLineSettings(
                    color=gl_color,
                    width=gl_width,
                    style=gl_style,
                ),
                scales=ScaleSettings(
                    tick_color=scale_tick,
                    font_color=scale_font,
                    font_size=scale_size,
                ),
            ),
            css_variables=css_variables,
        )

        self._themes[identifier] = new_config
        return copy.deepcopy(new_config)