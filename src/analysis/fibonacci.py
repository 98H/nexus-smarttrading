"""Fibonacci retracement and extension calculations for technical analysis."""

from collections.abc import Sequence

DEFAULT_RETRACEMENT_RATIOS: tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)
DEFAULT_EXTENSION_RATIOS: tuple[float, ...] = (1.0, 1.272, 1.618, 2.618)

_UPTREND_IDENTIFIERS: frozenset[str] = frozenset({"uptrend", "up"})
_DOWNTREND_IDENTIFIERS: frozenset[str] = frozenset({"downtrend", "down"})


def _validate_price_range(high: float, low: float) -> None:
    """Validate that swing high is greater than or equal to swing low."""
    if high < low:
        raise ValueError(
            f"Swing high ({high}) cannot be less than swing low ({low})."
        )


def _validate_ratios(ratios: Sequence[float] | None) -> None:
    """Validate that ratio collection is non-empty when supplied."""
    if ratios is not None and len(ratios) == 0:
        raise ValueError("Fibonacci ratios sequence cannot be empty.")


def _normalize_trend(trend: str) -> bool:
    """
    Validate and determine if the trend is an uptrend.

    Returns:
        bool: True if uptrend, False if downtrend.

    Raises:
        ValueError: If trend string is not recognized.
    """
    cleaned = trend.strip().lower()
    if cleaned in _UPTREND_IDENTIFIERS:
        return True
    if cleaned in _DOWNTREND_IDENTIFIERS:
        return False
    raise ValueError(
        f"Invalid trend '{trend}'. Expected 'uptrend' ('up') or 'downtrend' ('down')."
    )


def calculate_retracement_levels(
    high: float,
    low: float,
    trend: str = "uptrend",
    ratios: Sequence[float] | None = None,
) -> dict[float, float]:
    """
    Calculate Fibonacci retracement levels for a given price swing.

    Args:
        high: Swing high price point.
        low: Swing low price point.
        trend: Trend direction ('uptrend'/'up' or 'downtrend'/'down').
        ratios: Custom sequence of retracement ratios. Defaults to standard ratios.

    Returns:
        Dictionary mapping each ratio to its corresponding price level.

    Raises:
        ValueError: If high is less than low, ratios is empty, or trend is invalid.
    """
    _validate_price_range(high, low)
    _validate_ratios(ratios)
    is_uptrend = _normalize_trend(trend)

    active_ratios = DEFAULT_RETRACEMENT_RATIOS if ratios is None else ratios
    price_range = high - low

    if is_uptrend:
        return {ratio: high - price_range * ratio for ratio in active_ratios}
    return {ratio: low + price_range * ratio for ratio in active_ratios}


def calculate_extension_levels(
    high: float,
    low: float,
    pullback_anchor: float | None = None,
    trend: str = "uptrend",
    ratios: Sequence[float] | None = None,
) -> dict[float, float]:
    """
    Calculate Fibonacci extension/expansion levels for target projection.

    Args:
        high: Swing high price point.
        low: Swing low price point.
        pullback_anchor: Optional 3-point anchor price after the initial swing.
            If omitted, projects directly from swing high (uptrend) or low (downtrend).
        trend: Trend direction ('uptrend'/'up' or 'downtrend'/'down').
        ratios: Custom sequence of extension ratios. Defaults to standard ratios.

    Returns:
        Dictionary mapping each ratio to its projected price target.

    Raises:
        ValueError: If high is less than low, ratios is empty, or trend is invalid.
    """
    _validate_price_range(high, low)
    _validate_ratios(ratios)
    is_uptrend = _normalize_trend(trend)

    active_ratios = DEFAULT_EXTENSION_RATIOS if ratios is None else ratios
    price_range = high - low

    if is_uptrend:
        base = high if pullback_anchor is None else pullback_anchor
        return {ratio: base + price_range * ratio for ratio in active_ratios}

    base = low if pullback_anchor is None else pullback_anchor
    return {ratio: base - price_range * ratio for ratio in active_ratios}