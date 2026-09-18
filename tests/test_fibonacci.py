"""
Unit tests for Multi-Ratio Fibonacci Retracement and Extension Tool.

Story 5.3.1: Implement Multi-Ratio Fibonacci Retracement and Extension Tool
Target Modules:
- src/analysis/fibonacci.py
- src/analysis/__init__.py
"""

import inspect
from typing import Any, Sequence

import pytest

# Target imports - will fail with ModuleNotFoundError / ImportError until implemented
from src.analysis import (
    calculate_extension_levels as analysis_calculate_extension_levels,
    calculate_retracement_levels as analysis_calculate_retracement_levels,
)
from src.analysis.fibonacci import (
    DEFAULT_EXTENSION_RATIOS,
    DEFAULT_RETRACEMENT_RATIOS,
    calculate_extension_levels,
    calculate_retracement_levels,
)


def _invoke_extension(
    high: float,
    low: float,
    pullback_anchor: float | None = None,
    trend: str = "uptrend",
    ratios: Sequence[float] | None = None,
) -> dict[float, float]:
    """
    Helper to call calculate_extension_levels, adapting transparently to either
    'pullback_anchor' or 'anchor' parameter naming.
    """
    sig = inspect.signature(calculate_extension_levels)
    kwargs: dict[str, Any] = {}

    if "pullback_anchor" in sig.parameters:
        kwargs["pullback_anchor"] = pullback_anchor
    elif "anchor" in sig.parameters:
        kwargs["anchor"] = pullback_anchor
    else:
        # Fallback to positional invocation if 3rd parameter exists
        params = list(sig.parameters.values())
        if len(params) >= 3 and pullback_anchor is not None:
            return calculate_extension_levels(
                high, low, pullback_anchor, trend=trend, ratios=ratios
            )

    kwargs["trend"] = trend
    if ratios is not None:
        kwargs["ratios"] = ratios

    return calculate_extension_levels(high=high, low=low, **kwargs)


class TestModuleExportsAndConstants:
    """Verify package architecture and default constant definitions."""

    def test_analysis_package_exports_fibonacci_functions(self) -> None:
        """Verify __init__.py exposes the Fibonacci calculation functions."""
        assert callable(analysis_calculate_retracement_levels)
        assert callable(analysis_calculate_extension_levels)
        assert (
            analysis_calculate_retracement_levels is calculate_retracement_levels
        )
        assert analysis_calculate_extension_levels is calculate_extension_levels

    def test_default_ratios_constants(self) -> None:
        """Verify default retracement and extension ratios conform to specification."""
        expected_retracement = (0.236, 0.382, 0.5, 0.618, 0.786)
        expected_extension = (1.0, 1.272, 1.618, 2.618)

        assert tuple(DEFAULT_RETRACEMENT_RATIOS) == expected_retracement
        assert tuple(DEFAULT_EXTENSION_RATIOS) == expected_extension


class TestCalculateRetracementLevels:
    """Test calculate_retracement_levels across uptrends, downtrends, and ratios."""

    def test_uptrend_retracement_default_ratios(self) -> None:
        """
        In an uptrend (swing low to swing high), retracements measure pullback
        downward from the high: Level = High - (High - Low) * ratio.
        """
        high = 200.0
        low = 100.0
        # range = 100.0

        levels = calculate_retracement_levels(high=high, low=low, trend="uptrend")

        assert isinstance(levels, dict)
        assert len(levels) == 5

        assert levels[0.236] == pytest.approx(176.4, rel=1e-5)
        assert levels[0.382] == pytest.approx(161.8, rel=1e-5)
        assert levels[0.5] == pytest.approx(150.0, rel=1e-5)
        assert levels[0.618] == pytest.approx(138.2, rel=1e-5)
        assert levels[0.786] == pytest.approx(121.4, rel=1e-5)

    def test_retracement_defaults_to_uptrend_when_trend_unspecified(self) -> None:
        """Verify function defaults to 'uptrend' when trend argument is omitted."""
        high = 200.0
        low = 100.0

        levels_implicit = calculate_retracement_levels(high=high, low=low)
        levels_explicit = calculate_retracement_levels(
            high=high, low=low, trend="uptrend"
        )

        assert levels_implicit == levels_explicit

    def test_downtrend_retracement_default_ratios(self) -> None:
        """
        In a downtrend (swing high to swing low), retracements measure bounce
        upward from the low: Level = Low + (High - Low) * ratio.
        """
        high = 200.0
        low = 100.0
        # range = 100.0

        levels = calculate_retracement_levels(high=high, low=low, trend="downtrend")

        assert isinstance(levels, dict)
        assert len(levels) == 5

        assert levels[0.236] == pytest.approx(123.6, rel=1e-5)
        assert levels[0.382] == pytest.approx(138.2, rel=1e-5)
        assert levels[0.5] == pytest.approx(150.0, rel=1e-5)
        assert levels[0.618] == pytest.approx(161.8, rel=1e-5)
        assert levels[0.786] == pytest.approx(178.6, rel=1e-5)

    def test_retracement_custom_ratios(self) -> None:
        """Verify custom ratio sequences override defaults accurately."""
        high = 300.0
        low = 100.0
        custom_ratios = [0.382, 0.618]

        levels = calculate_retracement_levels(
            high=high, low=low, trend="uptrend", ratios=custom_ratios
        )

        assert list(levels.keys()) == custom_ratios
        assert levels[0.382] == pytest.approx(300.0 - 200.0 * 0.382, rel=1e-5)
        assert levels[0.618] == pytest.approx(300.0 - 200.0 * 0.618, rel=1e-5)

    @pytest.mark.parametrize("trend_str", ["uptrend", "UPTREND", "UpTrend", "up"])
    def test_retracement_trend_case_insensitivity_uptrend(
        self, trend_str: str
    ) -> None:
        """Verify trend string matching is case-insensitive for uptrends."""
        levels = calculate_retracement_levels(high=150.0, low=50.0, trend=trend_str)
        assert levels[0.5] == pytest.approx(100.0, rel=1e-5)

    @pytest.mark.parametrize(
        "trend_str", ["downtrend", "DOWNTREND", "DownTrend", "down"]
    )
    def test_retracement_trend_case_insensitivity_downtrend(
        self, trend_str: str
    ) -> None:
        """Verify trend string matching is case-insensitive for downtrends."""
        levels = calculate_retracement_levels(high=150.0, low=50.0, trend=trend_str)
        assert levels[0.5] == pytest.approx(100.0, rel=1e-5)

    def test_retracement_boundary_ratios(self) -> None:
        """Verify 0.0 (no retracement) and 1.0 (full 100% retracement)."""
        high = 250.0
        low = 150.0

        levels_up = calculate_retracement_levels(
            high=high, low=low, trend="uptrend", ratios=[0.0, 1.0]
        )
        assert levels_up[0.0] == pytest.approx(high, rel=1e-5)
        assert levels_up[1.0] == pytest.approx(low, rel=1e-5)

        levels_down = calculate_retracement_levels(
            high=high, low=low, trend="downtrend", ratios=[0.0, 1.0]
        )
        assert levels_down[0.0] == pytest.approx(low, rel=1e-5)
        assert levels_down[1.0] == pytest.approx(high, rel=1e-5)

    def test_retracement_decimal_precision(self) -> None:
        """Verify calculation accuracy with fractional price values."""
        high = 154.375
        low = 102.125
        # range = 52.25

        levels = calculate_retracement_levels(
            high=high, low=low, trend="uptrend", ratios=[0.5]
        )
        assert levels[0.5] == pytest.approx(128.25, rel=1e-5)


class TestCalculateExtensionLevels:
    """Test calculate_extension_levels with and without pullback anchor."""

    def test_extension_uptrend_with_pullback_anchor(self) -> None:
        """
        3-point Fibonacci extension in uptrend:
        Swing: Low -> High. Pullback to Anchor.
        Projection: Level = Anchor + (High - Low) * ratio.
        """
        high = 150.0
        low = 100.0
        anchor = 120.0  # Pulled back from 150 to 120
        # swing range = 50.0

        levels = _invoke_extension(
            high=high,
            low=low,
            pullback_anchor=anchor,
            trend="uptrend",
            ratios=[1.0, 1.272, 1.618, 2.618],
        )

        assert isinstance(levels, dict)
        assert levels[1.0] == pytest.approx(120.0 + 50.0 * 1.0, rel=1e-5)  # 170.0
        assert levels[1.272] == pytest.approx(120.0 + 50.0 * 1.272, rel=1e-5)  # 183.6
        assert levels[1.618] == pytest.approx(120.0 + 50.0 * 1.618, rel=1e-5)  # 200.9
        assert levels[2.618] == pytest.approx(120.0 + 50.0 * 2.618, rel=1e-5)  # 250.9

    def test_extension_downtrend_with_pullback_anchor(self) -> None:
        """
        3-point Fibonacci extension in downtrend:
        Swing: High -> Low. Pullback bounce to Anchor.
        Projection: Level = Anchor - (High - Low) * ratio.
        """
        high = 150.0
        low = 100.0
        anchor = 130.0  # Bounced from 100 up to 130
        # swing range = 50.0

        levels = _invoke_extension(
            high=high,
            low=low,
            pullback_anchor=anchor,
            trend="downtrend",
            ratios=[1.0, 1.272, 1.618, 2.618],
        )

        assert isinstance(levels, dict)
        assert levels[1.0] == pytest.approx(130.0 - 50.0 * 1.0, rel=1e-5)  # 80.0
        assert levels[1.272] == pytest.approx(130.0 - 50.0 * 1.272, rel=1e-5)  # 66.4
        assert levels[1.618] == pytest.approx(130.0 - 50.0 * 1.618, rel=1e-5)  # 49.1
        assert levels[2.618] == pytest.approx(130.0 - 50.0 * 2.618, rel=1e-5)  # -0.9

    def test_extension_uptrend_without_pullback_anchor(self) -> None:
        """
        When pullback anchor is omitted in uptrend, projection extends
        directly from the trend swing high: Level = High + (High - Low) * ratio.
        """
        high = 200.0
        low = 100.0
        # swing range = 100.0

        levels = _invoke_extension(
            high=high, low=low, pullback_anchor=None, trend="uptrend"
        )

        assert isinstance(levels, dict)
        assert levels[1.0] == pytest.approx(300.0, rel=1e-5)
        assert levels[1.272] == pytest.approx(327.2, rel=1e-5)
        assert levels[1.618] == pytest.approx(361.8, rel=1e-5)
        assert levels[2.618] == pytest.approx(461.8, rel=1e-5)

    def test_extension_downtrend_without_pullback_anchor(self) -> None:
        """
        When pullback anchor is omitted in downtrend, projection extends
        directly downward from the trend swing low: Level = Low - (High - Low) * ratio.
        """
        high = 200.0
        low = 100.0
        # swing range = 100.0

        levels = _invoke_extension(
            high=high, low=low, pullback_anchor=None, trend="downtrend"
        )

        assert isinstance(levels, dict)
        assert levels[1.0] == pytest.approx(0.0, rel=1e-5)
        assert levels[1.272] == pytest.approx(-27.2, rel=1e-5)
        assert levels[1.618] == pytest.approx(-61.8, rel=1e-5)
        assert levels[2.618] == pytest.approx(-161.8, rel=1e-5)

    def test_extension_custom_ratios(self) -> None:
        """Verify extension levels honor custom target ratios."""
        high = 100.0
        low = 50.0
        custom_ratios = [0.5, 1.0, 2.0]

        levels = _invoke_extension(
            high=high,
            low=low,
            pullback_anchor=60.0,
            trend="uptrend",
            ratios=custom_ratios,
        )

        assert list(levels.keys()) == custom_ratios
        assert levels[0.5] == pytest.approx(60.0 + 50.0 * 0.5, rel=1e-5)
        assert levels[1.0] == pytest.approx(60.0 + 50.0 * 1.0, rel=1e-5)
        assert levels[2.0] == pytest.approx(60.0 + 50.0 * 2.0, rel=1e-5)

    def test_extension_positional_arguments(self) -> None:
        """Verify calculate_extension_levels functions correctly with positional arguments."""
        # high, low, pullback_anchor
        levels = calculate_extension_levels(150.0, 100.0, 120.0)
        assert 1.0 in levels
        assert levels[1.0] == pytest.approx(170.0, rel=1e-5)


class TestFibonacciValidationAndExceptions:
    """Verify input validation and error raising per Acceptance Criteria 3."""

    def test_retracement_raises_value_error_when_high_less_than_low(self) -> None:
        """AC 3: High less than low must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_retracement_levels(high=50.0, low=100.0)
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_extension_raises_value_error_when_high_less_than_low(self) -> None:
        """AC 3: High less than low must raise ValueError in extension calculation."""
        with pytest.raises(ValueError) as exc_info:
            _invoke_extension(high=80.0, low=120.0)
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_extension_with_anchor_raises_value_error_when_high_less_than_low(
        self,
    ) -> None:
        """AC 3: High less than low must raise ValueError even when anchor is provided."""
        with pytest.raises(ValueError) as exc_info:
            _invoke_extension(high=40.0, low=60.0, pullback_anchor=50.0)
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_retracement_raises_value_error_on_empty_ratios(self) -> None:
        """AC 3: Empty ratio list must raise ValueError in retracement calculation."""
        with pytest.raises(ValueError) as exc_info:
            calculate_retracement_levels(high=100.0, low=50.0, ratios=[])
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_extension_raises_value_error_on_empty_ratios(self) -> None:
        """AC 3: Empty ratio list must raise ValueError in extension calculation."""
        with pytest.raises(ValueError) as exc_info:
            _invoke_extension(high=100.0, low=50.0, ratios=[])
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_retracement_raises_value_error_on_empty_tuple_ratios(self) -> None:
        """Verify empty tuple ratios also trigger ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_retracement_levels(high=100.0, low=50.0, ratios=())
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_retracement_raises_value_error_invalid_trend_string(self) -> None:
        """Invalid trend name must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_retracement_levels(
                high=100.0, low=50.0, trend="sideways_trend"
            )
        assert str(exc_info.value), "Exception message should be descriptive"

    def test_extension_raises_value_error_invalid_trend_string(self) -> None:
        """Invalid trend name must raise ValueError in extension calculation."""
        with pytest.raises(ValueError) as exc_info:
            _invoke_extension(high=100.0, low=50.0, trend="non_existent")
        assert str(exc_info.value), "Exception message should be descriptive"