"""
Institutional Volume Profile Analysis.

Computes Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)
covering a configurable percentage (typically 70%) of total session volume.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfileResult:
    """Represents the output of a Volume Profile calculation."""

    poc: float
    vah: float
    val: float
    total_volume: float
    value_area_volume: float


class VolumeProfiler:
    """Computes session-filtered Volume Profile levels (POC, VAH, VAL)."""

    def __init__(
        self,
        session_start: Optional[Any] = None,
        session_end: Optional[Any] = None,
        price_col: str = "price",
        volume_col: str = "volume",
        timestamp_col: str = "timestamp",
        value_area_pct: float = 0.70,
    ) -> None:
        if not (0.0 < value_area_pct <= 1.0):
            raise ValueError(
                f"value_area_pct must be in range (0.0, 1.0], got {value_area_pct}"
            )

        self.session_start = session_start
        self.session_end = session_end
        self.price_col = price_col
        self.volume_col = volume_col
        self.timestamp_col = timestamp_col
        self.value_area_pct = value_area_pct

    def compute(self, df: pd.DataFrame) -> VolumeProfileResult:
        """Compute POC, VAH, and VAL from price and volume data."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        if self.price_col not in df.columns or self.volume_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{self.price_col}' and '{self.volume_col}' columns."
            )

        filtered_df = df
        if self.session_start is not None or self.session_end is not None:
            if self.timestamp_col not in df.columns:
                raise ValueError(
                    f"Timestamp column '{self.timestamp_col}' required for session filtering."
                )
            if self.session_start is not None:
                filtered_df = filtered_df[
                    filtered_df[self.timestamp_col] >= self.session_start
                ]
            if self.session_end is not None:
                filtered_df = filtered_df[
                    filtered_df[self.timestamp_col] <= self.session_end
                ]

        if filtered_df.empty:
            raise ValueError("DataFrame is empty after applying session filter.")

        if (filtered_df[self.volume_col] < 0).any():
            raise ValueError("Volume values cannot be negative.")

        total_volume = float(filtered_df[self.volume_col].sum())
        if total_volume <= 0.0:
            raise ValueError(f"Total volume must be positive, got {total_volume}")

        # Aggregate volumes per discrete price level, sorted ascending
        aggregated = (
            filtered_df.groupby(self.price_col)[self.volume_col].sum().sort_index()
        )
        prices = aggregated.index.to_numpy(dtype=float)
        volumes = aggregated.to_numpy(dtype=float)

        target_volume = total_volume * self.value_area_pct

        # Identify Point of Control (POC)
        poc_idx = int(np.argmax(volumes))
        poc_price = float(prices[poc_idx])

        # Single price level edge case
        if len(prices) == 1:
            return VolumeProfileResult(
                poc=poc_price,
                vah=poc_price,
                val=poc_price,
                total_volume=total_volume,
                value_area_volume=total_volume,
            )

        # Expand Value Area outwards from POC
        current_volume = float(volumes[poc_idx])
        upper_idx = poc_idx
        lower_idx = poc_idx

        while current_volume < target_volume and (
            upper_idx < len(prices) - 1 or lower_idx > 0
        ):
            has_upper = upper_idx < len(prices) - 1
            has_lower = lower_idx > 0

            vol_above = volumes[upper_idx + 1] if has_upper else -1.0
            vol_below = volumes[lower_idx - 1] if has_lower else -1.0

            if has_upper and not has_lower:
                upper_idx += 1
                current_volume += vol_above
            elif has_lower and not has_upper:
                lower_idx -= 1
                current_volume += vol_below
            else:
                if vol_above > vol_below:
                    upper_idx += 1
                    current_volume += vol_above
                elif vol_below > vol_above:
                    lower_idx -= 1
                    current_volume += vol_below
                else:
                    # Symmetric expansion when volumes above and below are identical
                    upper_idx += 1
                    lower_idx -= 1
                    current_volume += vol_above + vol_below

        val = float(prices[lower_idx])
        vah = float(prices[upper_idx])

        return VolumeProfileResult(
            poc=poc_price,
            vah=vah,
            val=val,
            total_volume=total_volume,
            value_area_volume=current_volume,
        )


def compute_volume_profile(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    value_area_pct: float = 0.70,
    timestamp_col: Optional[str] = "timestamp",
    session_start: Optional[Any] = None,
    session_end: Optional[Any] = None,
) -> VolumeProfileResult:
    """Convenience function to calculate Volume Profile POC, VAH, and VAL."""
    profiler = VolumeProfiler(
        session_start=session_start,
        session_end=session_end,
        price_col=price_col,
        volume_col=volume_col,
        timestamp_col=timestamp_col or "timestamp",
        value_area_pct=value_area_pct,
    )
    return profiler.compute(df)