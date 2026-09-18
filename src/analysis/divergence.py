from __future__ import annotations

from typing import Optional

from src.analysis.types import (
    DivergenceResult,
    DivergenceType,
    SwingPoint,
    SwingType,
)


def detect_divergence(
    prev_point: SwingPoint, curr_point: SwingPoint
) -> Optional[DivergenceResult]:
    """Detect regular or hidden divergence between two consecutive swing points of the same type.

    Args:
        prev_point: Chronologically preceding swing point.
        curr_point: Chronologically succeeding swing point.

    Returns:
        DivergenceResult if divergence pattern matches, None otherwise.

    Raises:
        ValueError: If swing types do not match or points are non-chronological.
    """
    if prev_point.swing_type != curr_point.swing_type:
        raise ValueError(
            f"Mismatched swing types: prev={prev_point.swing_type}, curr={curr_point.swing_type}"
        )

    if curr_point.index <= prev_point.index:
        raise ValueError(
            f"Non-chronological or identical indices: prev={prev_point.index}, curr={curr_point.index}"
        )

    if (
        prev_point.timestamp is not None
        and curr_point.timestamp is not None
        and curr_point.timestamp < prev_point.timestamp
    ):
        raise ValueError(
            f"Non-chronological timestamps: prev={prev_point.timestamp}, curr={curr_point.timestamp}"
        )

    div_type: Optional[DivergenceType] = None

    if prev_point.swing_type == SwingType.LOW:
        # Regular Bullish: Lower Low in Price, Higher Low in Oscillator
        if curr_point.price < prev_point.price and curr_point.oscillator > prev_point.oscillator:
            div_type = DivergenceType.REGULAR_BULLISH
        # Hidden Bullish: Higher Low in Price, Lower Low in Oscillator
        elif curr_point.price > prev_point.price and curr_point.oscillator < prev_point.oscillator:
            div_type = DivergenceType.HIDDEN_BULLISH
    else:
        # Regular Bearish: Higher High in Price, Lower High in Oscillator
        if curr_point.price > prev_point.price and curr_point.oscillator < prev_point.oscillator:
            div_type = DivergenceType.REGULAR_BEARISH
        # Hidden Bearish: Lower High in Price, Higher High in Oscillator
        elif curr_point.price < prev_point.price and curr_point.oscillator > prev_point.oscillator:
            div_type = DivergenceType.HIDDEN_BEARISH

    if div_type is None:
        return None

    return DivergenceResult(
        divergence_type=div_type,
        swing_type=curr_point.swing_type,
        prev_point=prev_point,
        curr_point=curr_point,
    )


class DivergenceEngine:
    """Stateful engine for sequential, real-time swing ingestion and divergence detection."""

    def __init__(self) -> None:
        self._last_low: Optional[SwingPoint] = None
        self._last_high: Optional[SwingPoint] = None
        self._last_point: Optional[SwingPoint] = None

    def reset(self) -> None:
        """Reset internal history and state."""
        self._last_low = None
        self._last_high = None
        self._last_point = None

    def detect_divergence(
        self, prev_point: SwingPoint, curr_point: SwingPoint
    ) -> Optional[DivergenceResult]:
        """Stateless evaluation between two explicit swing points."""
        return detect_divergence(prev_point, curr_point)

    def process_swing(self, point: SwingPoint) -> Optional[DivergenceResult]:
        """Ingest a swing point in streaming order and evaluate divergence.

        Args:
            point: The latest swing point received.

        Returns:
            DivergenceResult if divergence is confirmed against previous swing of same type, else None.

        Raises:
            ValueError: If incoming swing point arrives out of chronological sequence.
        """
        if self._last_point is not None:
            if point.index <= self._last_point.index:
                raise ValueError(
                    f"Incoming swing point index ({point.index}) must be greater than "
                    f"last ingested point index ({self._last_point.index})"
                )
            if (
                self._last_point.timestamp is not None
                and point.timestamp is not None
                and point.timestamp < self._last_point.timestamp
            ):
                raise ValueError(
                    f"Incoming swing point timestamp ({point.timestamp}) must be chronological with "
                    f"last ingested point timestamp ({self._last_point.timestamp})"
                )

        prev_reference = self._last_high if point.swing_type == SwingType.HIGH else self._last_low

        result: Optional[DivergenceResult] = None
        if prev_reference is not None:
            result = detect_divergence(prev_reference, point)

        if point.swing_type == SwingType.HIGH:
            self._last_high = point
        else:
            self._last_low = point
        self._last_point = point

        return result