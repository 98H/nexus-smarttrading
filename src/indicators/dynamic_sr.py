"""Smart Dynamic Support and Resistance Zones indicator module."""

from dataclasses import dataclass, field
import math
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SRZone:
    """Represents a support or resistance zone with upper and lower boundaries."""

    lower_boundary: float
    upper_boundary: float

    def __post_init__(self) -> None:
        self.lower_boundary = float(self.lower_boundary)
        self.upper_boundary = float(self.upper_boundary)
        if self.upper_boundary < self.lower_boundary:
            raise ValueError(
                f"upper_boundary ({self.upper_boundary}) must be >= lower_boundary ({self.lower_boundary})"
            )

    @property
    def lower(self) -> float:
        """Alias for lower_boundary."""
        return self.lower_boundary

    @property
    def upper(self) -> float:
        """Alias for upper_boundary."""
        return self.upper_boundary


@dataclass
class DynamicSRResult:
    """Container for active support and resistance zones."""

    support_zones: List[SRZone] = field(default_factory=list)
    resistance_zones: List[SRZone] = field(default_factory=list)


def _find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    swing_order: int,
) -> Tuple[List[float], List[float]]:
    """Identify swing high and swing low price points within the given price arrays."""
    n = len(highs)
    swing_highs: List[float] = []
    swing_lows: List[float] = []

    if n < 2 * swing_order + 1:
        return swing_highs, swing_lows

    for i in range(swing_order, n - swing_order):
        current_high = highs[i]
        current_low = lows[i]

        if not math.isnan(current_high):
            is_swing_high = True
            for offset in range(-swing_order, swing_order + 1):
                if offset == 0:
                    continue
                neighbor_high = highs[i + offset]
                if math.isnan(neighbor_high) or neighbor_high >= current_high:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append(float(current_high))

        if not math.isnan(current_low):
            is_swing_low = True
            for offset in range(-swing_order, swing_order + 1):
                if offset == 0:
                    continue
                neighbor_low = lows[i + offset]
                if math.isnan(neighbor_low) or neighbor_low <= current_low:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append(float(current_low))

    return swing_highs, swing_lows


def _cluster_prices(prices: List[float], tolerance: float) -> List[SRZone]:
    """Cluster 1D price points within a defined relative tolerance band into SRZone objects."""
    if not prices:
        return []

    sorted_p = sorted(prices)
    clusters: List[List[float]] = []
    current_cluster = [sorted_p[0]]

    for p in sorted_p[1:]:
        cluster_min = current_cluster[0]
        ref = abs(cluster_min) if abs(cluster_min) > 1e-9 else 1.0
        if (p - cluster_min) / ref <= tolerance:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]

    clusters.append(current_cluster)

    return [
        SRZone(lower_boundary=float(min(c)), upper_boundary=float(max(c)))
        for c in clusters
    ]


def calculate_dynamic_sr_zones(
    df: pd.DataFrame,
    window: int = 20,
    tolerance: float = 0.02,
    swing_order: int = 2,
) -> DynamicSRResult:
    """
    Calculate dynamic support and resistance zones from an OHLCV price series.

    Parameters:
        df: pd.DataFrame containing OHLCV price series with 'high' and 'low' columns.
        window: Rolling window lookback size. Must be > 0.
        tolerance: Relative price tolerance band for clustering swing points. Must be > 0.
        swing_order: Number of surrounding bars on each side to qualify a swing point. Must be > 0.

    Returns:
        DynamicSRResult containing lists of active support and resistance zones.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")

    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    if swing_order <= 0:
        raise ValueError(f"swing_order must be a positive integer, got {swing_order}")

    col_map = {str(col).lower(): col for col in df.columns}
    if "high" not in col_map or "low" not in col_map:
        raise ValueError("DataFrame must contain 'high' and 'low' columns (case-insensitive)")

    if len(df) < window:
        return DynamicSRResult(support_zones=[], resistance_zones=[])

    sub_df = df.iloc[-window:]
    highs = sub_df[col_map["high"]].to_numpy(dtype=float)
    lows = sub_df[col_map["low"]].to_numpy(dtype=float)

    swing_highs, swing_lows = _find_swing_points(highs, lows, swing_order)

    resistance_zones = _cluster_prices(swing_highs, tolerance)
    support_zones = _cluster_prices(swing_lows, tolerance)

    return DynamicSRResult(
        support_zones=support_zones,
        resistance_zones=resistance_zones,
    )