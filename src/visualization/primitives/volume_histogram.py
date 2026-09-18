"""
Dynamic Color-Graded Volume Histogram Primitive.

This module provides data structures and primitives to transform raw volume
and price delta series into color-graded histogram bars based on configurable
threshold maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ColorThreshold:
    """Represents a single threshold rule for color assignment."""

    label: str = ""
    min_delta: float = -float("inf")
    max_delta: float = float("inf")
    min_volume: float = 0.0
    max_volume: float = float("inf")
    color: str = "#787B86"

    def matches(self, price_delta: float, volume: float) -> bool:
        """Determines if the provided delta and volume satisfy this threshold."""
        delta_in_range = self.min_delta <= price_delta <= self.max_delta
        volume_in_range = self.min_volume <= volume <= self.max_volume
        return delta_in_range and volume_in_range


class ColorThresholdMap:
    """
    Manages dynamic color assignment based on tiered thresholds or
    standard bull/bear/neutral configurations.
    """

    def __init__(
        self,
        thresholds: Optional[List[ColorThreshold]] = None,
        default_color: str = "#787B86",
        bull_color: str = "#26A69A",
        bear_color: str = "#EF5350",
        neutral_color: str = "#787B86",
    ) -> None:
        self.thresholds = thresholds
        self.default_color = default_color
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.neutral_color = neutral_color

        if self.thresholds is not None:
            # Sort thresholds by intensity distance from 0 and volume constraint
            # descending so higher-intensity edge conditions resolve deterministically.
            self._sorted_thresholds: List[ColorThreshold] = sorted(
                self.thresholds,
                key=self._compute_threshold_priority,
                reverse=True,
            )
        else:
            self._sorted_thresholds = []

    @staticmethod
    def _compute_threshold_priority(threshold: ColorThreshold) -> tuple[float, float]:
        """
        Computes prioritization key based on threshold intensity distance from zero
        and volume requirements.
        """
        if threshold.min_delta > 0:
            delta_distance = threshold.min_delta
        elif threshold.max_delta < 0:
            delta_distance = abs(threshold.max_delta)
        else:
            delta_distance = 0.0

        return (delta_distance, threshold.min_volume)

    def get_color(self, price_delta: float, volume: float = 0.0) -> str:
        """Resolves the appropriate color for the given price delta and volume."""
        if self.thresholds is not None:
            for threshold in self._sorted_thresholds:
                if threshold.matches(price_delta, volume):
                    return threshold.color
            return self.default_color

        if price_delta > 0.0:
            return self.bull_color
        if price_delta < 0.0:
            return self.bear_color
        return self.neutral_color


@dataclass(frozen=True)
class VolumeDataPoint:
    """Represents an input volume data observation with an associated price delta."""

    timestamp: Any
    volume: float
    price_delta: float


@dataclass(frozen=True)
class VolumeHistogramBar:
    """Represents a rendered histogram bar element with resolved color."""

    timestamp: Any
    volume: float
    color: str
    price_delta: Optional[float] = None


class VolumeHistogramRepresentation:
    """
    Encapsulates the rendered collection of volume histogram bars, conforming to
    standard sequence behaviors.
    """

    def __init__(self, bars: Optional[Sequence[VolumeHistogramBar]] = None) -> None:
        self._bars: tuple[VolumeHistogramBar, ...] = tuple(bars) if bars else ()

    @property
    def bars(self) -> tuple[VolumeHistogramBar, ...]:
        """Returns the immutable tuple of bars."""
        return self._bars

    @property
    def is_empty(self) -> bool:
        """Indicates whether this representation contains any bars."""
        return len(self._bars) == 0

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index: int | slice) -> Any:
        return self._bars[index]

    def __iter__(self) -> Iterator[VolumeHistogramBar]:
        return iter(self._bars)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VolumeHistogramRepresentation):
            return False
        return self._bars == other._bars

    def __repr__(self) -> str:
        return f"VolumeHistogramRepresentation(bars={list(self._bars)})"


class VolumeHistogramPrimitive:
    """Primitive responsible for transforming volume data series into histogram bars."""

    def __init__(self, color_map: Optional[ColorThresholdMap] = None) -> None:
        self.color_map = color_map if color_map is not None else ColorThresholdMap()

    @staticmethod
    def _sanitize_point(element: Any) -> Optional[tuple[Any, float, float]]:
        """
        Validates and extracts timestamp, volume, and price delta from a data element.
        Returns None if the element is malformed or invalid.
        """
        if element is None or isinstance(element, (str, bytes, int, float, bool)):
            return None

        if isinstance(element, VolumeDataPoint):
            timestamp = element.timestamp
            volume = element.volume
            delta = element.price_delta
        elif isinstance(element, Mapping):
            if "timestamp" not in element or "volume" not in element or "price_delta" not in element:
                return None
            timestamp = element["timestamp"]
            volume = element["volume"]
            delta = element["price_delta"]
        elif hasattr(element, "timestamp") and hasattr(element, "volume") and hasattr(element, "price_delta"):
            timestamp = getattr(element, "timestamp")
            volume = getattr(element, "volume")
            delta = getattr(element, "price_delta")
        else:
            return None

        if timestamp is None:
            return None

        if not isinstance(volume, (int, float)) or isinstance(volume, bool):
            return None
        if not math.isfinite(volume) or volume < 0.0:
            return None

        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            return None
        if not math.isfinite(delta):
            return None

        return timestamp, float(volume), float(delta)

    def process(self, series: Any) -> VolumeHistogramRepresentation:
        """
        Processes an input series of volume data points into a histogram representation.
        Invalid inputs and corrupt elements are filtered gracefully without unhandled exceptions.
        """
        if series is None or isinstance(series, (str, bytes, bytearray, Mapping)):
            return VolumeHistogramRepresentation()

        if not isinstance(series, Iterable):
            return VolumeHistogramRepresentation()

        bars: List[VolumeHistogramBar] = []
        try:
            for item in series:
                sanitized = self._sanitize_point(item)
                if sanitized is None:
                    continue

                timestamp, volume, delta = sanitized
                color = self.color_map.get_color(price_delta=delta, volume=volume)
                bars.append(
                    VolumeHistogramBar(
                        timestamp=timestamp,
                        volume=volume,
                        color=color,
                        price_delta=delta,
                    )
                )
        except Exception:
            return VolumeHistogramRepresentation()

        return VolumeHistogramRepresentation(bars)