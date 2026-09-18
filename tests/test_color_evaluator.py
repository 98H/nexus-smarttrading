"""
Unit tests for dynamic color and fill expression evaluation.

Target modules:
- src.rendering.expressions.color_evaluator
- src.rendering.styles.fill_style

Acceptance Criteria:
- Given an element definition with a dynamic expression string for fill or color
  (e.g., conditional statement or context variable)
- When the expression evaluator resolves the color properties against the runtime data context
- Then the expression evaluates to a valid hex, rgb, or named color string and updates
  the element's style attribute
- Given an invalid or unresolvable color expression
- When evaluation is attempted
- Then the evaluator gracefully falls back to a configured default fill color and logs a warning
"""

import logging
import pytest

from src.rendering.expressions.color_evaluator import ColorEvaluator
from src.rendering.styles.fill_style import Element, FillStyle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_evaluator() -> ColorEvaluator:
    """Provides a ColorEvaluator with default settings."""
    return ColorEvaluator()


@pytest.fixture
def custom_fallback_evaluator() -> ColorEvaluator:
    """Provides a ColorEvaluator with a distinct configured fallback color."""
    return ColorEvaluator(default_fill_color="#FF00FF")


@pytest.fixture
def runtime_context() -> dict:
    """Provides a realistic runtime data context for expression evaluation."""
    return {
        "status": "active",
        "severity": "high",
        "theme": {
            "primary": "#3498DB",
            "danger": "#E74C3C",
            "success": "#2ECC71",
            "muted_rgba": "rgba(0, 0, 0, 0.4)",
        },
        "flags": {
            "is_highlighted": True,
            "is_disabled": False,
        },
        "palette": {
            "accent": "rgb(255, 165, 0)",
            "neutral": "whitesmoke",
        },
    }


# ---------------------------------------------------------------------------
# Tests: Valid Dynamic Color Expressions & Formats
# ---------------------------------------------------------------------------

class TestValidColorExpressions:
    """Tests expression evaluation resolving to valid hex, rgb, rgba, and named colors."""

    @pytest.mark.parametrize(
        "expression, expected_color",
        [
            ("theme.primary", "#3498DB"),
            ("theme.danger", "#E74C3C"),
            ("'#FFF'", "#FFF"),
            ("'#12345678'", "#12345678"),
            ("'#abcdef'", "#abcdef"),
        ],
    )
    def test_evaluates_hex_colors(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        expression: str,
        expected_color: str,
    ):
        """Verifies resolution of 3, 6, and 8-character hex color codes."""
        resolved = default_evaluator.evaluate(expression, runtime_context)
        assert resolved == expected_color

    @pytest.mark.parametrize(
        "expression, expected_color",
        [
            ("palette.accent", "rgb(255, 165, 0)"),
            ("theme.muted_rgba", "rgba(0, 0, 0, 0.4)"),
            ("'rgb(0, 128, 255)'", "rgb(0, 128, 255)"),
            ("'rgba(100, 100, 100, 0.75)'", "rgba(100, 100, 100, 0.75)"),
        ],
    )
    def test_evaluates_rgb_and_rgba_colors(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        expression: str,
        expected_color: str,
    ):
        """Verifies resolution of functional rgb(...) and rgba(...) color strings."""
        resolved = default_evaluator.evaluate(expression, runtime_context)
        assert resolved == expected_color

    @pytest.mark.parametrize(
        "expression, expected_color",
        [
            ("palette.neutral", "whitesmoke"),
            ("'transparent'", "transparent"),
            ("'red'", "red"),
            ("'blue'", "blue"),
            ("'none'", "none"),
        ],
    )
    def test_evaluates_named_colors(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        expression: str,
        expected_color: str,
    ):
        """Verifies resolution of standard CSS and SVG named color identifiers."""
        resolved = default_evaluator.evaluate(expression, runtime_context)
        assert resolved == expected_color

    @pytest.mark.parametrize(
        "expression, expected_color",
        [
            (
                "theme.success if status == 'active' else theme.danger",
                "#2ECC71",
            ),
            (
                "theme.danger if status == 'inactive' else theme.primary",
                "#3498DB",
            ),
            (
                "'#00FF00' if flags.is_highlighted and not flags.is_disabled else '#555555'",
                "#00FF00",
            ),
            (
                "'#FF0000' if severity == 'critical' else ('#FFA500' if severity == 'high' else '#00FF00')",
                "#FFA500",
            ),
        ],
    )
    def test_evaluates_conditional_expressions(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        expression: str,
        expected_color: str,
    ):
        """Verifies that dynamic conditional logic correctly resolves colors based on context state."""
        resolved = default_evaluator.evaluate(expression, runtime_context)
        assert resolved == expected_color


# ---------------------------------------------------------------------------
# Tests: Element Style Updating
# ---------------------------------------------------------------------------

class TestElementStyleUpdates:
    """Tests updating the element's style attribute via expression evaluator."""

    def test_resolves_and_updates_element_fill_attribute(
        self, default_evaluator: ColorEvaluator, runtime_context: dict
    ):
        """Verifies element.style.fill is updated from fill_expression."""
        element = Element(
            id="node-1",
            style=FillStyle(
                fill=None,
                fill_expression="theme.primary",
            ),
        )

        default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == "#3498DB"

    def test_resolves_and_updates_element_color_attribute(
        self, default_evaluator: ColorEvaluator, runtime_context: dict
    ):
        """Verifies element.style.color is updated from color_expression."""
        element = Element(
            id="text-1",
            style=FillStyle(
                color=None,
                color_expression="theme.danger",
            ),
        )

        default_evaluator.resolve_element(element, runtime_context)

        assert element.style.color == "#E74C3C"

    def test_resolves_both_fill_and_color_expressions_simultaneously(
        self, default_evaluator: ColorEvaluator, runtime_context: dict
    ):
        """Verifies that fill and color expressions can resolve independently on the same element."""
        element = Element(
            id="composite-1",
            style=FillStyle(
                fill=None,
                color=None,
                fill_expression="theme.primary",
                color_expression="palette.neutral",
            ),
        )

        default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == "#3498DB"
        assert element.style.color == "whitesmoke"

    def test_preserves_unrelated_style_attributes(
        self, default_evaluator: ColorEvaluator, runtime_context: dict
    ):
        """Ensures non-color style attributes (e.g., opacity, stroke) remain intact after resolution."""
        element = Element(
            id="box-1",
            style=FillStyle(
                fill_expression="theme.success",
                opacity=0.75,
                stroke="#000000",
                stroke_width=2.0,
            ),
        )

        default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == "#2ECC71"
        assert element.style.opacity == 0.75
        assert element.style.stroke == "#000000"
        assert element.style.stroke_width == 2.0

    def test_element_without_expressions_remains_unmodified(
        self, default_evaluator: ColorEvaluator, runtime_context: dict
    ):
        """Static styles without expressions must not be altered during evaluation passes."""
        element = Element(
            id="static-1",
            style=FillStyle(
                fill="#123456",
                color="#654321",
            ),
        )

        default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == "#123456"
        assert element.style.color == "#654321"

    def test_no_warning_logged_on_successful_resolution(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        caplog: pytest.LogCaptureFixture,
    ):
        """Confirms that successful evaluation emits zero warning logs."""
        element = Element(
            id="success-el",
            style=FillStyle(fill_expression="theme.primary"),
        )

        with caplog.at_level(logging.WARNING):
            default_evaluator.resolve_element(element, runtime_context)

        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Tests: Error Handling, Graceful Fallback, and Logging
# ---------------------------------------------------------------------------

class TestFallbackAndErrorHandling:
    """Tests fallback to configured default fill color and warning logging on invalid expressions."""

    def test_falls_back_on_undefined_context_variable_and_logs_warning(
        self,
        default_evaluator: ColorEvaluator,
        caplog: pytest.LogCaptureFixture,
    ):
        """When an expression references a nonexistent variable, fallback to default color and log warning."""
        element = Element(
            id="error-node-1",
            style=FillStyle(fill_expression="non_existent_variable.color"),
        )
        empty_context = {}

        with caplog.at_level(logging.WARNING):
            default_evaluator.resolve_element(element, empty_context)

        assert element.style.fill == default_evaluator.default_fill_color
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_falls_back_on_syntax_error_in_expression_and_logs_warning(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        caplog: pytest.LogCaptureFixture,
    ):
        """Malformed syntax in expressions triggers graceful fallback and logs a warning."""
        element = Element(
            id="syntax-error-node",
            style=FillStyle(fill_expression="status === 'active' ? ? invalid"),
        )

        with caplog.at_level(logging.WARNING):
            default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == default_evaluator.default_fill_color
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    @pytest.mark.parametrize(
        "invalid_result_expression",
        [
            ("'not_a_real_color'",),
            ("'#12'",),
            ("'#GGGGGG'",),
            ("'rgb(999, 999, 999)'",),
            ("'rgb(255, 255)'",),
            ("''",),
            ("'   '",),
        ],
    )
    def test_falls_back_when_expression_evaluates_to_invalid_color_string(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        caplog: pytest.LogCaptureFixture,
        invalid_result_expression: tuple,
    ):
        """Valid syntax that produces an invalid color representation must trigger fallback and log a warning."""
        element = Element(
            id="bad-color-result",
            style=FillStyle(fill_expression=invalid_result_expression[0]),
        )

        with caplog.at_level(logging.WARNING):
            default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == default_evaluator.default_fill_color
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    @pytest.mark.parametrize(
        "non_string_expression",
        [
            ("42",),
            ("True",),
            ("None",),
            ("[1, 2, 3]",),
            ("{'key': 'value'}",),
        ],
    )
    def test_falls_back_when_expression_evaluates_to_non_string_type(
        self,
        default_evaluator: ColorEvaluator,
        runtime_context: dict,
        caplog: pytest.LogCaptureFixture,
        non_string_expression: tuple,
    ):
        """Expressions returning non-string types must fall back gracefully and log a warning."""
        element = Element(
            id="non-string-result",
            style=FillStyle(fill_expression=non_string_expression[0]),
        )

        with caplog.at_level(logging.WARNING):
            default_evaluator.resolve_element(element, runtime_context)

        assert element.style.fill == default_evaluator.default_fill_color
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_custom_configured_default_color_used_on_failure(
        self,
        custom_fallback_evaluator: ColorEvaluator,
        caplog: pytest.LogCaptureFixture,
    ):
        """Verifies evaluator uses instance-configured fallback color rather than hardcoded global."""
        element = Element(
            id="custom-fallback-node",
            style=FillStyle(fill_expression="missing.field"),
        )

        with caplog.at_level(logging.WARNING):
            custom_fallback_evaluator.resolve_element(element, {})

        assert element.style.fill == "#FF00FF"
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_color_expression_fallback_uses_default_fill_color(
        self,
        custom_fallback_evaluator: ColorEvaluator,
        caplog: pytest.LogCaptureFixture,
    ):
        """Verifies color_expression also falls back to default_fill_color on evaluation failure."""
        element = Element(
            id="custom-color-fallback-node",
            style=FillStyle(color_expression="invalid_fn()"),
        )

        with caplog.at_level(logging.WARNING):
            custom_fallback_evaluator.resolve_element(element, {})

        assert element.style.color == "#FF00FF"
        assert any(record.levelno == logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests: Direct Evaluator Helper and Validation
# ---------------------------------------------------------------------------

class TestColorEvaluatorValidation:
    """Tests the color validation utility exposed by the ColorEvaluator module."""

    @pytest.mark.parametrize(
        "valid_color",
        [
            "#000",
            "#FFF",
            "#1a2b3c",
            "#AABBCC",
            "#12345678",
            "#AABBCCDD",
            "rgb(0, 0, 0)",
            "rgb(255, 255, 255)",
            "rgba(0, 0, 0, 0)",
            "rgba(255, 255, 255, 1)",
            "rgba(100, 150, 200, 0.5)",
            "red",
            "green",
            "blue",
            "transparent",
            "none",
        ],
    )
    def test_validates_correct_color_strings(
        self, default_evaluator: ColorEvaluator, valid_color: str
    ):
        """Ensures valid CSS/SVG color formats are recognized as valid."""
        assert default_evaluator.is_valid_color(valid_color) is True

    @pytest.mark.parametrize(
        "invalid_color",
        [
            "",
            "   ",
            "#1",
            "#12",
            "#12345",
            "#1234567",
            "#123456789",
            "#ZZZ",
            "rgb()",
            "rgb(256, 0, 0)",
            "rgb(-1, 0, 0)",
            "rgba(0, 0, 0, 2.5)",
            "rgba(0, 0, 0, -0.5)",
            "invalid_name",
            None,
            12345,
        ],
    )
    def test_rejects_invalid_color_strings(
        self, default_evaluator: ColorEvaluator, invalid_color: str
    ):
        """Ensures malformed or out-of-range color representations are rejected."""
        assert default_evaluator.is_valid_color(invalid_color) is False

    def test_evaluator_raises_value_error_for_invalid_configured_default(self):
        """Configuring an evaluator with an invalid default fill color raises ValueError."""
        with pytest.raises(ValueError):
            ColorEvaluator(default_fill_color="not_a_color")