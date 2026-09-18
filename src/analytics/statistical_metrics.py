"""Statistical performance metric calculators for financial return series."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    """Structured container for statistical performance metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from a sequence of periodic returns."""
    if len(returns) <= 1:
        return 0.0

    cum_wealth = np.cumprod(1.0 + returns)
    wealth = np.insert(cum_wealth, 0, 1.0)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (running_max - wealth) / running_max
    max_dd = float(np.max(drawdowns))

    if max_dd < 0.0 or np.isclose(max_dd, 0.0):
        return 0.0
    return max_dd


def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """Calculate periodic Sharpe ratio."""
    if len(returns) <= 1:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = float(np.mean(excess_returns))
    std = float(np.std(returns, ddof=1))

    if np.isclose(std, 0.0) or np.isclose(mean_excess, 0.0):
        return 0.0

    sharpe = mean_excess / std
    return 0.0 if np.isclose(sharpe, 0.0) else float(sharpe)


def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """Calculate periodic Sortino ratio penalizing only downside deviations below the risk-free rate."""
    if len(returns) <= 1:
        return 0.0

    std = float(np.std(returns, ddof=1))
    if np.isclose(std, 0.0):
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = float(np.mean(excess_returns))
    if np.isclose(mean_excess, 0.0):
        return 0.0

    downside_diff = np.minimum(excess_returns, 0.0)
    downside_variance = float(np.mean(downside_diff**2))
    downside_dev = float(np.sqrt(downside_variance))

    if np.isclose(downside_dev, 0.0):
        return 0.0

    sortino = mean_excess / downside_dev
    return 0.0 if np.isclose(sortino, 0.0) else float(sortino)


def calculate_performance_metrics(
    returns: Any,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute Sharpe ratio, Sortino ratio, and Maximum Drawdown for periodic returns.

    Args:
        returns: Sequence of periodic returns (list, tuple, np.ndarray, or pd.Series).
        risk_free_rate: Periodic risk-free benchmark rate (defaults to 0.0).

    Returns:
        PerformanceMetrics containing sharpe_ratio, sortino_ratio, and max_drawdown.

    Raises:
        TypeError: If returns is not a 1-dimensional numeric sequence or risk_free_rate is invalid.
        ValueError: If returns contains non-numeric elements, NaN, or Inf values.
    """
    if returns is None or isinstance(returns, (str, bytes, bytearray)):
        raise TypeError("Returns must be a numeric sequence, not None, string, or bytes.")

    try:
        arr = np.asarray(returns, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert returns to numeric array: {e}") from e

    if arr.ndim != 1:
        raise TypeError("Returns must be a 1-dimensional sequence.")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Returns must not contain NaN or Inf values.")

    try:
        rf = float(risk_free_rate)
    except (ValueError, TypeError) as e:
        raise TypeError(f"risk_free_rate must be a numeric value: {e}") from e

    return PerformanceMetrics(
        sharpe_ratio=_calculate_sharpe_ratio(arr, rf),
        sortino_ratio=_calculate_sortino_ratio(arr, rf),
        max_drawdown=_calculate_max_drawdown(arr),
    )