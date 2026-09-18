"""Elliott Wave (Impulse 1-5 and Correction A-B-C) Vector Visualizer.

Provides domain models and visualizer rendering sequential vector segments
for Elliott Wave impulse and corrective patterns with styling metadata.
"""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Sequence

IMPULSE_LABELS: Sequence[str] = ("1", "2", "3", "4", "5")
CORRECTIVE_LABELS: Sequence[str] = ("A", "B", "C")

DEFAULT_IMPULSE_STYLING: Dict[str, Any] = {
    "pattern_type": "impulse",
    "wave_type": "impulse",
    "color": "#2962FF",
    "line_style": "solid",
    "line_width": 2,
}

DEFAULT_CORRECTIVE_STYLING: Dict[str, Any] = {
    "pattern_type": "corrective",
    "wave_type": "corrective",
    "color": "#FF6D00",
    "line_style": "dashed",
    "line_width": 2,
}


@dataclass(frozen=True)
class WavePoint:
    """Represents an Elliott Wave coordinate pivot point."""

    label: str
    timestamp: datetime
    price: float


@dataclass
class VectorSegment:
    """Represents a rendered vector segment connecting adjacent Elliott Wave points."""

    start_point: WavePoint
    end_point: WavePoint
    label: str
    styling: Dict[str, Any]


class ElliottWaveVisualizer:
    """Visualizer for Elliott Wave impulse and corrective vector patterns."""

    def __init__(
        self,
        impulse_styling: Optional[Dict[str, Any]] = None,
        corrective_styling: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._impulse_styling = (
            dict(impulse_styling) if impulse_styling is not None else DEFAULT_IMPULSE_STYLING
        )
        self._corrective_styling = (
            dict(corrective_styling) if corrective_styling is not None else DEFAULT_CORRECTIVE_STYLING
        )

    def render_impulse_wave(self, points: Sequence[WavePoint]) -> List[VectorSegment]:
        """Render standard 5-point Elliott Impulse Wave coordinates (1-5) into 4 vector segments."""
        self._validate_points(points, IMPULSE_LABELS, "impulse")
        return self._build_segments(points, self._impulse_styling)

    def render_corrective_wave(self, points: Sequence[WavePoint]) -> List[VectorSegment]:
        """Render standard 3-point Elliott Corrective Wave coordinates (A-C) into 2 vector segments."""
        self._validate_points(points, CORRECTIVE_LABELS, "corrective")
        return self._build_segments(points, self._corrective_styling)

    @staticmethod
    def _validate_points(
        points: Sequence[WavePoint],
        expected_labels: Sequence[str],
        pattern_name: str,
    ) -> None:
        """Validate count, labels, finite prices, and strictly chronological timestamps."""
        if points is None or len(points) != len(expected_labels):
            point_count = len(points) if points is not None else 0
            raise ValueError(
                f"Invalid point count for Elliott {pattern_name} wave: "
                f"expected {len(expected_labels)} points, got {point_count}."
            )

        for i, point in enumerate(points):
            if not isinstance(point, WavePoint):
                raise ValueError(
                    f"Point at index {i} must be a WavePoint instance, got {type(point).__name__}."
                )

            if not math.isfinite(point.price):
                raise ValueError(
                    f"Point '{point.label}' has invalid price {point.price}; price must be finite."
                )

            if point.label != expected_labels[i]:
                raise ValueError(
                    f"Invalid label sequence for Elliott {pattern_name} wave: "
                    f"expected '{expected_labels[i]}' at index {i}, got '{point.label}'."
                )

            if i > 0 and point.timestamp <= points[i - 1].timestamp:
                raise ValueError(
                    f"Non-chronological timestamp at point '{point.label}' ({point.timestamp}) "
                    f"preceded by '{points[i - 1].label}' ({points[i - 1].timestamp}). "
                    f"Wave points must have strictly ascending timestamps."
                )

    @staticmethod
    def _build_segments(
        points: Sequence[WavePoint],
        styling: Dict[str, Any],
    ) -> List[VectorSegment]:
        """Generate sequential vector segments from validated wave points."""
        segments: List[VectorSegment] = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            segments.append(
                VectorSegment(
                    start_point=start,
                    end_point=end,
                    label=f"{start.label}-{end.label}",
                    styling=dict(styling),
                )
            )
        return segments