"""Foundation Moving Averages: SMA, EMA, WMA, RMA."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["sma", "ema", "wma", "rma"]


def _validate_length(length: Any) -> int:
    """Validate window length to ensure it is strictly a positive integer."""
    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError(f"Window length must be an integer, got {type(length).__name__}")
    if length <= 0:
        raise ValueError(f"Window length must be a positive integer (> 0), got {length}")
    return length


def _clean_input(series: Any) -> list[float]:
    """Convert input series-like container to a list of floats."""
    if series is None:
        raise TypeError("Input series cannot be None")

    try:
        iterator = iter(series)
    except TypeError:
        raise TypeError(f"Input series must be iterable, got {type(series).__name__}")

    cleaned: list[float] = []
    for x in iterator:
        if x is None:
            cleaned.append(math.nan)
        else:
            try:
                cleaned.append(float(x))
            except (ValueError, TypeError):
                cleaned.append(math.nan)
    return cleaned


def _calc_exponential_ma(prices: list[float], length: int, alpha: float) -> list[float]:
    """Calculate recursive exponential-type moving average seeded with SMA."""
    n = len(prices)
    result = [math.nan] * n
    if n < length:
        return result

    # Seed with SMA at index length - 1
    result[length - 1] = sum(prices[0:length]) / length
    for i in range(length, n):
        result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1]

    return result


def sma(series: Any, length: int) -> list[float]:
    """Calculate the Simple Moving Average (SMA).

    Computes the arithmetic mean over a rolling window of specified length.
    Initial warmup elements where index < length - 1 evaluate to NaN.
    """
    _validate_length(length)
    prices = _clean_input(series)
    n = len(prices)
    if n == 0:
        return []

    result = [math.nan] * n
    if n < length:
        return result

    for i in range(length - 1, n):
        result[i] = sum(prices[i - length + 1 : i + 1]) / length

    return result


def wma(series: Any, length: int) -> list[float]:
    """Calculate the Linearly Weighted Moving Average (WMA).

    Applies linear weights from 1 to length over the rolling window.
    Initial warmup elements where index < length - 1 evaluate to NaN.
    """
    _validate_length(length)
    prices = _clean_input(series)
    n = len(prices)
    if n == 0:
        return []

    result = [math.nan] * n
    if n < length:
        return result

    weight_sum = (length * (length + 1)) / 2.0
    for i in range(length - 1, n):
        window = prices[i - length + 1 : i + 1]
        weighted_sum = sum(w * val for w, val in enumerate(window, start=1))
        result[i] = weighted_sum / weight_sum

    return result


def ema(series: Any, length: int) -> list[float]:
    """Calculate the Exponential Moving Average (EMA).

    Uses smoothing factor alpha = 2 / (length + 1) with an SMA seed
    at index length - 1.
    Initial warmup elements where index < length - 1 evaluate to NaN.
    """
    _validate_length(length)
    prices = _clean_input(series)
    if len(prices) == 0:
        return []

    alpha = 2.0 / (length + 1.0)
    return _calc_exponential_ma(prices, length, alpha)


def rma(series: Any, length: int) -> list[float]:
    """Calculate Wilder's Smoothing Moving Average (RMA).

    Uses smoothing factor alpha = 1 / length with an SMA seed
    at index length - 1.
    Initial warmup elements where index < length - 1 evaluate to NaN.
    """
    _validate_length(length)
    prices = _clean_input(series)
    if len(prices) == 0:
        return []

    alpha = 1.0 / float(length)
    return _calc_exponential_ma(prices, length, alpha)