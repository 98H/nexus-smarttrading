"""Volatility indicators: Bollinger Bands and Average True Range (ATR)."""

from typing import NamedTuple

import numpy as np
import pandas as pd


class BollingerBandsResult(NamedTuple):
    """Container for Bollinger Bands components."""

    upper: pd.Series
    middle: pd.Series
    lower: pd.Series

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


def bb(
    series: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> BollingerBandsResult:
    """Bollinger Bands (BB).

    Computes middle band (SMA), upper band, and lower band.

    Args:
        series: Series of closing prices.
        period: Moving average lookback window. Default is 20.
        std_multiplier: Multiplier for rolling standard deviation. Default is 2.0.

    Returns:
        BollingerBandsResult: NamedTuple containing upper, middle, and lower series.

    Raises:
        ValueError: If period or std_multiplier is non-positive, or data is insufficient.
    """
    if period <= 0:
        raise ValueError(f"period must be a positive integer, got {period}")
    if std_multiplier <= 0:
        raise ValueError(f"std_multiplier must be positive, got {std_multiplier}")
    if len(series) < period:
        raise ValueError(
            f"series length ({len(series)}) must be at least period ({period})"
        )

    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std

    upper.name = "upper"
    middle.name = "middle"
    lower.name = "lower"

    return BollingerBandsResult(upper=upper, middle=middle, lower=lower)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR).

    Computes Wilder's Average True Range reflecting moving average of true ranges.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        period: Smoothing period for ATR. Default is 14.

    Returns:
        pd.Series: Average True Range values.

    Raises:
        ValueError: If period is non-positive, series lengths or indices mismatch,
                    high < low, or series length is insufficient.
    """
    if period <= 0:
        raise ValueError(f"period must be a positive integer, got {period}")

    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            f"Series lengths do not match: high ({len(high)}), "
            f"low ({len(low)}), close ({len(close)})"
        )

    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("Series indices for high, low, and close must match")

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    if (h < l).any():
        raise ValueError("High must be greater than or equal to Low for all entries")

    if len(h) < period:
        raise ValueError(
            f"series length ({len(h)}) must be at least period ({period})"
        )

    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    if n > 1:
        hl = h[1:] - l[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(l[1:] - c[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    atr_values = np.full(n, np.nan, dtype=float)

    # Initial value is simple mean of first `period` true ranges
    atr_values[period - 1] = float(np.mean(tr[:period]))

    # Wilder's smoothing
    for i in range(period, n):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    return pd.Series(atr_values, index=high.index, name="atr")


__all__ = ["bb", "atr", "BollingerBandsResult"]