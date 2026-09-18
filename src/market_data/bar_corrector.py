"""Engine for re-indexing historical bar buckets and processing late-arriving ticks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import math
from typing import Callable

from src.market_data.models import OHLCVBar, Tick

__all__ = ["BarCorrectionEngine"]


class BarCorrectionEngine:
    """Engine responsible for correcting finalized historical OHLCV bars upon arrival of late ticks."""

    def __init__(self, timeframe: timedelta) -> None:
        if not isinstance(timeframe, timedelta) or timeframe <= timedelta(0):
            raise ValueError("Timeframe must be a positive timedelta")
        self._timeframe = timeframe
        self._watermark: datetime | None = None
        self._bars: dict[tuple[str, datetime], OHLCVBar] = {}
        self._latest_tick_timestamps: dict[tuple[str, datetime], datetime] = {}
        self._listeners: list[Callable[[OHLCVBar], None]] = []

    @property
    def timeframe(self) -> timedelta:
        """Return the timeframe duration of bars processed by this engine."""
        return self._timeframe

    @property
    def watermark(self) -> datetime | None:
        """Return the current watermark timestamp."""
        return self._watermark

    def set_watermark(self, watermark: datetime) -> None:
        """Set the current processing watermark."""
        if not isinstance(watermark, datetime):
            raise ValueError(f"Watermark must be a datetime instance, got {type(watermark).__name__}")
        self._watermark = watermark

    def add_bar(self, bar: OHLCVBar) -> None:
        """Register an existing OHLCV bar into the engine."""
        key = (bar.symbol, bar.timestamp)
        self._bars[key] = bar
        self._latest_tick_timestamps.setdefault(key, None)

    def get_bar(self, symbol: str, timestamp: datetime) -> OHLCVBar | None:
        """Retrieve an OHLCV bar by symbol and bucket timestamp."""
        key = (symbol, timestamp)
        if key in self._bars:
            return self._bars[key]
        bucket_time = self._align_timestamp(timestamp)
        return self._bars.get((symbol, bucket_time))

    def register_listener(self, listener: Callable[[OHLCVBar], None]) -> None:
        """Register a callback to be notified when a corrected bar is emitted."""
        self._listeners.append(listener)

    def _align_timestamp(self, dt: datetime) -> datetime:
        """Align a datetime down to the bucket start boundary [T_bucket, T_bucket + Delta t)."""
        tz = dt.tzinfo
        epoch = datetime(1970, 1, 1, tzinfo=tz)
        total_us = (dt - epoch) // timedelta(microseconds=1)
        timeframe_us = self._timeframe // timedelta(microseconds=1)
        bucket_us = (total_us // timeframe_us) * timeframe_us
        return epoch + timedelta(microseconds=bucket_us)

    def process_tick(self, tick: Tick) -> OHLCVBar | None:
        """
        Process a market tick event.

        If the tick arrives prior to the watermark, it is treated as a late tick.
        The historical bar bucket covering [T_bucket, T_bucket + Delta t) is updated:
        High, Low, Close, and Volume are recalculated, is_corrected is set to True,
        and the corrected bar is emitted to registered listeners.

        If the tick is at or after the watermark, it is an on-time tick and no corrected
        finalized bar is emitted.
        """
        try:
            price_val = float(tick.price)
            if not math.isfinite(price_val) or price_val <= 0.0:
                raise ValueError(f"Price must be positive and finite, got {tick.price}")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Price must be positive and finite, got {tick.price}") from err

        try:
            vol_val = float(tick.volume)
            if not math.isfinite(vol_val) or vol_val < 0.0:
                raise ValueError(f"Volume must be non-negative and finite, got {tick.volume}")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Volume must be non-negative and finite, got {tick.volume}") from err

        if self._watermark is not None and tick.timestamp >= self._watermark:
            return None

        bucket_time = self._align_timestamp(tick.timestamp)
        key = (tick.symbol, bucket_time)

        if key not in self._bars:
            for (sym, b_time), _ in self._bars.items():
                if sym == tick.symbol and b_time <= tick.timestamp < b_time + self._timeframe:
                    key = (sym, b_time)
                    bucket_time = b_time
                    break
            else:
                raise KeyError(
                    f"No historical bar found for symbol {tick.symbol!r} at bucket timestamp {bucket_time}"
                )

        current_bar = self._bars[key]

        new_high = max(current_bar.high, price_val)
        new_low = min(current_bar.low, price_val)
        new_volume = current_bar.volume + vol_val

        last_tick_time = self._latest_tick_timestamps.get(key)
        if last_tick_time is None or tick.timestamp >= last_tick_time:
            self._latest_tick_timestamps[key] = tick.timestamp
            new_close = price_val
        else:
            new_close = current_bar.close

        corrected_bar = replace(
            current_bar,
            high=new_high,
            low=new_low,
            close=new_close,
            volume=new_volume,
            is_corrected=True,
        )

        self._bars[key] = corrected_bar

        for listener in self._listeners:
            listener(corrected_bar)

        return corrected_bar