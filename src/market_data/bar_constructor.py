"""Dynamic OHLCV bar constructor for custom resolution boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
import re

from src.market_data.models import Bar, Trade

__all__ = ["DynamicBarConstructor"]

_RESOLUTION_PATTERN = re.compile(r"^([1-9]\d*)(ms|s|m|h|d)$")
_MULTIPLIERS_IN_US: dict[str, int] = {
    "ms": 1_000,
    "s": 1_000_000,
    "m": 60 * 1_000_000,
    "h": 3600 * 1_000_000,
    "d": 86400 * 1_000_000,
}
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_EPOCH_NAIVE = datetime(1970, 1, 1)


def _parse_resolution_to_us(resolution: str) -> int:
    """Parses resolution string into integer microseconds."""
    if not isinstance(resolution, str):
        raise ValueError(f"Resolution must be a string, got {type(resolution).__name__}")

    match = _RESOLUTION_PATTERN.match(resolution)
    if not match:
        raise ValueError(f"Invalid resolution format: {resolution!r}")

    value_str, unit = match.groups()
    return int(value_str) * _MULTIPLIERS_IN_US[unit]


def _datetime_to_us(dt: datetime) -> int:
    """Converts a datetime into integer microseconds since epoch."""
    if dt.tzinfo is not None:
        delta = dt - _EPOCH_UTC
    else:
        delta = dt - _EPOCH_NAIVE
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _us_to_datetime(total_us: int, tz: tzinfo | None) -> datetime:
    """Converts epoch microseconds back into a datetime preserving timezone info."""
    seconds, microseconds = divmod(total_us, 1_000_000)
    if tz is not None:
        dt_utc = _EPOCH_UTC + timedelta(seconds=seconds, microseconds=microseconds)
        return dt_utc if tz == timezone.utc else dt_utc.astimezone(tz)
    return _EPOCH_NAIVE + timedelta(seconds=seconds, microseconds=microseconds)


class DynamicBarConstructor:
    """Aggregates stream of trades into custom resolution OHLCV bars without lookahead bias."""

    def __init__(self, resolution: str) -> None:
        self._interval_us = _parse_resolution_to_us(resolution)
        self.resolution = resolution

        self._last_trade_timestamp: datetime | None = None
        self._current_window_end_us: int | None = None
        self._current_tz: tzinfo | None = None

        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._volume: float = 0.0

    def process_trade(self, trade: Trade) -> list[Bar]:
        """Ingests a trade, updating current bar and emitting completed bars at boundaries."""
        if not isinstance(trade, Trade):
            raise TypeError(f"Expected Trade instance, got {type(trade).__name__}")

        if self._last_trade_timestamp is not None:
            try:
                if trade.timestamp < self._last_trade_timestamp:
                    raise ValueError(
                        f"Non-monotonic trade timestamp: {trade.timestamp} < {self._last_trade_timestamp}"
                    )
            except TypeError as err:
                raise ValueError(f"Cannot compare trade timestamps: {err}") from err

        trade_us = _datetime_to_us(trade.timestamp)
        self._last_trade_timestamp = trade.timestamp
        self._current_tz = trade.timestamp.tzinfo

        completed_bars: list[Bar] = []

        # Check if trade crosses beyond the active bar window boundary
        if self._current_window_end_us is not None and trade_us >= self._current_window_end_us:
            completed_bar = Bar(
                open=self._open,  # type: ignore[arg-type]
                high=self._high,  # type: ignore[arg-type]
                low=self._low,  # type: ignore[arg-type]
                close=self._close,  # type: ignore[arg-type]
                volume=self._volume,
                close_timestamp=_us_to_datetime(self._current_window_end_us, self._current_tz),
            )
            completed_bars.append(completed_bar)
            self._reset_current_bar()

        # Initialize new bar window if not active
        if self._current_window_end_us is None:
            window_start_us = (trade_us // self._interval_us) * self._interval_us
            self._current_window_end_us = window_start_us + self._interval_us
            self._open = trade.price
            self._high = trade.price
            self._low = trade.price
            self._close = trade.price
            self._volume = trade.volume
        else:
            # Update running state of the active bar
            self._high = max(self._high, trade.price)  # type: ignore[type-var]
            self._low = min(self._low, trade.price)  # type: ignore[type-var]
            self._close = trade.price
            self._volume += trade.volume

        return completed_bars

    def get_current_bar(self) -> Bar | None:
        """Returns snapshot of the in-progress bar or None if no trades ingested."""
        if self._open is None:
            return None

        return Bar(
            open=self._open,
            high=self._high,  # type: ignore[arg-type]
            low=self._low,  # type: ignore[arg-type]
            close=self._close,  # type: ignore[arg-type]
            volume=self._volume,
            close_timestamp=None,
        )

    def _reset_current_bar(self) -> None:
        """Resets the running OHLCV state."""
        self._current_window_end_us = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume = 0.0