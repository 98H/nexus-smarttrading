from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TrendState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class StructureEventType(str, Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) cannot be less than low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high cannot be less than open or close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low cannot be greater than open or close")


@dataclass(frozen=True)
class MarketStructureEvent:
    event_type: StructureEventType
    broken_level: float
    previous_trend: TrendState
    new_trend: TrendState
    candle: Candle


class MarketStructureEngine:
    def __init__(
        self,
        initial_trend: TrendState,
        swing_high: float,
        swing_low: float,
    ) -> None:
        if swing_high <= swing_low:
            raise ValueError(
                f"swing_high ({swing_high}) must be strictly greater than swing_low ({swing_low})"
            )
        self.current_trend: TrendState = initial_trend
        self._swing_high: float = swing_high
        self._swing_low: float = swing_low

    @property
    def swing_high(self) -> float:
        return self._swing_high

    @swing_high.setter
    def swing_high(self, price: float) -> None:
        self.set_swing_high(price)

    @property
    def swing_low(self) -> float:
        return self._swing_low

    @swing_low.setter
    def swing_low(self, price: float) -> None:
        self.set_swing_low(price)

    def set_swing_high(self, price: float) -> None:
        if price <= self._swing_low:
            raise ValueError(
                f"swing_high ({price}) must be strictly greater than swing_low ({self._swing_low})"
            )
        self._swing_high = price

    def set_swing_low(self, price: float) -> None:
        if price >= self._swing_high:
            raise ValueError(
                f"swing_low ({price}) must be strictly less than swing_high ({self._swing_high})"
            )
        self._swing_low = price

    def process_candle(self, candle: Candle) -> MarketStructureEvent | None:
        previous_trend = self.current_trend

        if candle.close > self._swing_high:
            broken_level = self._swing_high
            new_trend = TrendState.BULLISH
            event_type = (
                StructureEventType.BOS
                if previous_trend == TrendState.BULLISH
                else StructureEventType.CHOCH
            )
            self.current_trend = new_trend
            return MarketStructureEvent(
                event_type=event_type,
                broken_level=broken_level,
                previous_trend=previous_trend,
                new_trend=new_trend,
                candle=candle,
            )

        if candle.close < self._swing_low:
            broken_level = self._swing_low
            new_trend = TrendState.BEARISH
            event_type = (
                StructureEventType.BOS
                if previous_trend == TrendState.BEARISH
                else StructureEventType.CHOCH
            )
            self.current_trend = new_trend
            return MarketStructureEvent(
                event_type=event_type,
                broken_level=broken_level,
                previous_trend=previous_trend,
                new_trend=new_trend,
                candle=candle,
            )

        return None