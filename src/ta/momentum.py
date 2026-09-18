"""Momentum indicators: RSI and MACD."""

from typing import NamedTuple

import numpy as np
import pandas as pd


class MACDResult(NamedTuple):
    """Container for MACD indicator components."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self._fields:
                return getattr(self, item)
            raise KeyError(item)
        return tuple.__getitem__(self, item)

    def __contains__(self, item: object) -> bool:
        return item in self._fields

    def keys(self):
        """Return indicator field names."""
        return self._fields


def _calc_rsi(gain: float, loss: float) -> float:
    """Calculate bounded RSI value from average gain and loss."""
    if loss == 0.0:
        return 100.0 if gain > 0.0 else 50.0
    if gain == 0.0:
        return 0.0
    val = 100.0 * gain / (gain + loss)
    if val > 100.0:
        return 100.0
    if val < 0.0:
        return 0.0
    return float(val)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Computes Wilder's Relative Strength Index over the given period.

    Args:
        series: Series of closing prices.
        period: Lookback period for RSI calculation. Default is 14.

    Returns:
        pd.Series: RSI values bounded between 0 and 100 with initial NaN padding.

    Raises:
        ValueError: If period is non-positive or series length is insufficient.
    """
    if period <= 0:
        raise ValueError(f"period must be a positive integer, got {period}")

    if len(series) <= period:
        raise ValueError(
            f"series length ({len(series)}) must be greater than period ({period})"
        )

    values = series.to_numpy(dtype=float)
    n = len(values)
    diff = np.diff(values)

    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)

    rsi_values = np.full(n, np.nan, dtype=float)

    # First average gain and loss over the first `period` changes
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    rsi_values[period] = _calc_rsi(avg_gain, avg_loss)

    # Wilder's smoothing for subsequent values
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        idx = i + 1
        rsi_values[idx] = _calc_rsi(avg_gain, avg_loss)

    return pd.Series(rsi_values, index=series.index, name="rsi")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """Moving Average Convergence Divergence (MACD).

    Args:
        series: Series of closing prices.
        fast: Lookback period for fast EMA. Default is 12.
        slow: Lookback period for slow EMA. Default is 26.
        signal: Lookback period for signal line EMA. Default is 9.

    Returns:
        MACDResult: NamedTuple containing macd, signal, and histogram series.

    Raises:
        ValueError: If parameters are non-positive, fast >= slow, or data is insufficient.
    """
    if fast <= 0:
        raise ValueError(f"fast must be a positive integer, got {fast}")
    if slow <= 0:
        raise ValueError(f"slow must be a positive integer, got {slow}")
    if signal <= 0:
        raise ValueError(f"signal must be a positive integer, got {signal}")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be less than slow ({slow})")

    min_required_len = slow + signal - 1
    if len(series) < min_required_len:
        raise ValueError(
            f"series length ({len(series)}) must be at least {min_required_len} "
            f"for slow={slow} and signal={signal}"
        )

    ema_fast = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_line.name = "macd"

    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    signal_line.name = "signal"

    hist_line = macd_line - signal_line
    hist_line.name = "histogram"

    return MACDResult(macd=macd_line, signal=signal_line, histogram=hist_line)


__all__ = ["rsi", "macd", "MACDResult"]