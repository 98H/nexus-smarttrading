from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class DivergenceType(str, Enum):
    """Classification of technical divergence patterns."""

    REGULAR_BULLISH = "REGULAR_BULLISH"
    REGULAR_BEARISH = "REGULAR_BEARISH"
    HIDDEN_BULLISH = "HIDDEN_BULLISH"
    HIDDEN_BEARISH = "HIDDEN_BEARISH"


class SwingType(str, Enum):
    """Type of swing pivot point."""

    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class SwingPoint:
    """Immutable data model representing a price and oscillator swing pivot."""

    index: int
    price: float
    oscillator: float
    swing_type: SwingType
    timestamp: Optional[datetime] = None


@dataclass(frozen=True)
class DivergenceResult:
    """Detection result capturing a confirmed divergence between two swing points."""

    divergence_type: DivergenceType
    swing_type: SwingType
    prev_point: SwingPoint
    curr_point: SwingPoint