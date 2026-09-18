"""High-Probability Reversal Wick and Trend Reversal Zone detector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.indicators.fvg_detector import Candle as BaseCandle


class Candle(BaseCandle):
    """Immutable candlestick representation satisfying range and boundary invariants."""

    def __init__(
        self,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        timestamp: Any = None,
    ) -> None:
        if high < low:
            raise ValueError(f"high ({high}) cannot be less than low ({low})")
        if not (low <= open <= high):
            raise ValueError(
                f"open ({open}) must be within [low ({low}), high ({high})]"
            )
        if not (low <= close <= high):
            raise ValueError(
                f"close ({close}) must be within [low ({low}), high ({high})]"
            )

        try:
            super().__init__(
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=timestamp,
            )
        except TypeError:
            super().__init__(
                open=open,
                high=high,
                low=low,
                close=close,
            )

        object.__setattr__(self, "_is_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Candle is immutable; cannot modify attribute '{name}'"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Candle is immutable; cannot delete attribute '{name}'"
            )
        super().__delattr__(name)

    def __hash__(self) -> int:
        return hash(
            (
                self.open,
                self.high,
                self.low,
                self.close,
                getattr(self, "volume", 0.0),
                getattr(self, "timestamp", None),
            )
        )


class ReversalSignal(str, Enum):
    """Classification of reversal signals."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"


@dataclass(frozen=True)
class TrendReversalZone:
    """Immutable price range defining a trend reversal zone."""

    lower_boundary: float
    upper_boundary: float

    def __post_init__(self) -> None:
        if self.lower_boundary > self.upper_boundary:
            raise ValueError(
                f"lower_boundary ({self.lower_boundary}) cannot exceed upper_boundary ({self.upper_boundary})"
            )


@dataclass(frozen=True)
class ReversalResult:
    """Result of candlestick reversal wick evaluation."""

    signal: ReversalSignal
    zone: Optional[TrendReversalZone] = None
    lower_wick_ratio: float = 0.0
    upper_wick_ratio: float = 0.0
    body_ratio: float = 0.0
    candle: Optional[Candle] = None


class ReversalWickDetector:
    """Evaluates candlesticks for high-probability reversal wicks and trend reversal zones."""

    def __init__(
        self,
        wick_ratio_threshold: float = 0.60,
        max_body_ratio: float = 0.20,
    ) -> None:
        if not (0.0 < wick_ratio_threshold <= 1.0):
            raise ValueError(
                f"wick_ratio_threshold must be between 0.0 (exclusive) and 1.0 (inclusive), got {wick_ratio_threshold}"
            )
        if not (0.0 < max_body_ratio <= 1.0):
            raise ValueError(
                f"max_body_ratio must be between 0.0 (exclusive) and 1.0 (inclusive), got {max_body_ratio}"
            )

        self.wick_ratio_threshold = float(wick_ratio_threshold)
        self.max_body_ratio = float(max_body_ratio)

    def calculate_trend_reversal_zone(
        self, candle: Candle, signal: ReversalSignal
    ) -> Optional[TrendReversalZone]:
        """Calculate the trend reversal zone boundaries for a given candle and signal."""
        if signal == ReversalSignal.BULLISH:
            return TrendReversalZone(
                lower_boundary=float(candle.low),
                upper_boundary=float(min(candle.open, candle.close)),
            )
        if signal == ReversalSignal.BEARISH:
            return TrendReversalZone(
                lower_boundary=float(max(candle.open, candle.close)),
                upper_boundary=float(candle.high),
            )
        return None

    def evaluate(self, candle: Candle) -> ReversalResult:
        """Evaluate a candlestick for high-probability reversal patterns."""
        total_range = candle.high - candle.low
        if total_range <= 0.0:
            raise ValueError(
                f"Candle total range must be greater than zero, got {total_range}"
            )

        body_bottom = min(candle.open, candle.close)
        body_top = max(candle.open, candle.close)
        body_size = body_top - body_bottom

        lower_wick = body_bottom - candle.low
        upper_wick = candle.high - body_top

        lower_wick_ratio = lower_wick / total_range
        upper_wick_ratio = upper_wick / total_range
        body_ratio = body_size / total_range

        is_bullish = (
            lower_wick_ratio >= self.wick_ratio_threshold
            and body_ratio <= self.max_body_ratio
        )
        is_bearish = (
            upper_wick_ratio >= self.wick_ratio_threshold
            and body_ratio <= self.max_body_ratio
        )

        signal: ReversalSignal
        if is_bullish and is_bearish:
            if lower_wick_ratio > upper_wick_ratio:
                signal = ReversalSignal.BULLISH
            elif upper_wick_ratio > lower_wick_ratio:
                signal = ReversalSignal.BEARISH
            else:
                signal = ReversalSignal.NONE
        elif is_bullish:
            signal = ReversalSignal.BULLISH
        elif is_bearish:
            signal = ReversalSignal.BEARISH
        else:
            signal = ReversalSignal.NONE

        zone = self.calculate_trend_reversal_zone(candle, signal)

        return ReversalResult(
            signal=signal,
            zone=zone,
            lower_wick_ratio=lower_wick_ratio,
            upper_wick_ratio=upper_wick_ratio,
            body_ratio=body_ratio,
            candle=candle,
        )


__all__ = [
    "Candle",
    "ReversalResult",
    "ReversalSignal",
    "ReversalWickDetector",
    "TrendReversalZone",
]