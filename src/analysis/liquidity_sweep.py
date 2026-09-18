"""
Institutional Liquidity Sweep Detection.

Detects piercing of key levels (e.g., Swing High/Low, VAH/VAL) by candles with
institutional volume that close back inside the range, triggering sweep events.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence
import pandas as pd


class SweepDirection(str, Enum):
    """Direction of the liquidity sweep reversal."""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass(frozen=True)
class LiquidityLevel:
    """Key price level representing an institutional liquidity pool."""

    price: float
    level_type: str = "UNKNOWN"


@dataclass(frozen=True)
class LiquiditySweepEvent:
    """Represents a validated institutional liquidity sweep."""

    swept_level: float
    direction: SweepDirection
    sweep_volume: float
    pierce_price: float
    close_price: float
    timestamp: Optional[Any] = None
    level_type: Optional[str] = None


class LiquiditySweepDetector:
    """Detects institutional liquidity sweeps across key price levels."""

    def __init__(
        self,
        volume_threshold_multiplier: float = 2.0,
        volume_ma_period: int = 20,
    ) -> None:
        if volume_threshold_multiplier is None or volume_threshold_multiplier <= 0:
            raise ValueError(
                f"volume_threshold_multiplier must be greater than 0, got {volume_threshold_multiplier}"
            )
        if volume_ma_period is None or volume_ma_period <= 0:
            raise ValueError(
                f"volume_ma_period must be a positive integer, got {volume_ma_period}"
            )

        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.volume_ma_period = volume_ma_period

    def detect(
        self,
        candles: pd.DataFrame,
        key_levels: Sequence[LiquidityLevel],
    ) -> List[LiquiditySweepEvent]:
        """Detect liquidity sweep events for given candles against key levels."""
        if candles is None or candles.empty:
            raise ValueError("candles DataFrame cannot be empty.")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in candles.columns]
        if missing:
            raise ValueError(f"Missing required columns in candles DataFrame: {missing}")

        if not key_levels:
            return []

        # Calculate baseline moving average of volume from prior candles
        vol_series = candles["volume"].astype(float)
        rolling_ma = vol_series.shift(1).rolling(
            window=self.volume_ma_period,
            min_periods=self.volume_ma_period,
        ).mean()

        events: List[LiquiditySweepEvent] = []
        has_timestamp = "timestamp" in candles.columns

        for i in range(len(candles)):
            avg_vol = rolling_ma.iloc[i]
            if pd.isna(avg_vol) or avg_vol <= 0:
                continue

            vol = float(vol_series.iloc[i])
            threshold = avg_vol * self.volume_threshold_multiplier
            if vol < threshold - 1e-9:
                continue

            open_p = float(candles["open"].iloc[i])
            high_p = float(candles["high"].iloc[i])
            low_p = float(candles["low"].iloc[i])
            close_p = float(candles["close"].iloc[i])
            ts = candles["timestamp"].iloc[i] if has_timestamp else None

            for level in key_levels:
                lt = (level.level_type or "").upper()
                is_high_level = any(
                    k in lt for k in ["HIGH", "VAH", "RESISTANCE", "BSL", "TOP"]
                )
                is_low_level = any(
                    k in lt for k in ["LOW", "VAL", "SUPPORT", "SSL", "BOTTOM"]
                )

                # Bearish sweep: Pierces above resistance / high and closes back below
                if not is_low_level:
                    if high_p > level.price and close_p < level.price:
                        if is_high_level or open_p <= level.price:
                            events.append(
                                LiquiditySweepEvent(
                                    swept_level=float(level.price),
                                    direction=SweepDirection.BEARISH,
                                    sweep_volume=vol,
                                    pierce_price=high_p,
                                    close_price=close_p,
                                    timestamp=ts,
                                    level_type=level.level_type,
                                )
                            )

                # Bullish sweep: Pierces below support / low and closes back above
                if not is_high_level:
                    if low_p < level.price and close_p > level.price:
                        if is_low_level or open_p >= level.price:
                            events.append(
                                LiquiditySweepEvent(
                                    swept_level=float(level.price),
                                    direction=SweepDirection.BULLISH,
                                    sweep_volume=vol,
                                    pierce_price=low_p,
                                    close_price=close_p,
                                    timestamp=ts,
                                    level_type=level.level_type,
                                )
                            )

        return events


def detect_liquidity_sweeps(
    candles: pd.DataFrame,
    key_levels: Sequence[LiquidityLevel],
    volume_threshold_multiplier: float = 2.0,
    volume_ma_period: int = 20,
) -> List[LiquiditySweepEvent]:
    """Convenience function to detect liquidity sweeps against key price levels."""
    detector = LiquiditySweepDetector(
        volume_threshold_multiplier=volume_threshold_multiplier,
        volume_ma_period=volume_ma_period,
    )
    return detector.detect(candles=candles, key_levels=key_levels)