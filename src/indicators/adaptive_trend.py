"""Adaptive trend filter implementation using dynamic efficiency ratio smoothing."""

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class TrendDirection(str, Enum):
    """Directional classification of a trend."""

    UPWARD = "UPWARD"
    DOWNWARD = "DOWNWARD"
    FLAT = "FLAT"


@dataclass
class AdaptiveTrendResult:
    """Encapsulates the calculation outputs of the adaptive trend filter."""

    filter_values: pd.Series
    efficiency_ratio: pd.Series
    trend_direction: pd.Series
    current_direction: TrendDirection


class AdaptiveTrendFilter:
    """Calculates Kaufman's Adaptive Moving Average (KAMA) based dynamic filter.

    Dynamically adjusts smoothing constant based on market noise using the
    Efficiency Ratio (ER).
    """

    def __init__(
        self,
        period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
    ) -> None:
        """Initialize the adaptive trend filter parameters.

        Args:
            period: Lookback window for efficiency ratio calculation.
            fast_period: Fastest EMA smoothing period.
            slow_period: Slowest EMA smoothing period.

        Raises:
            TypeError: If any period parameter is not an integer.
            ValueError: If periods are non-positive or fast_period >= slow_period.
        """
        for param_name, param_val in [
            ("period", period),
            ("fast_period", fast_period),
            ("slow_period", slow_period),
        ]:
            if not isinstance(param_val, int) or isinstance(param_val, bool):
                raise TypeError(
                    f"{param_name} must be an integer, got {type(param_val).__name__}."
                )
            if param_val <= 0:
                raise ValueError(
                    f"{param_name} must be a positive integer, got {param_val}."
                )

        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be strictly less than "
                f"slow_period ({slow_period})."
            )

        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period

        self._fast_sc = 2.0 / (self.fast_period + 1)
        self._slow_sc = 2.0 / (self.slow_period + 1)

    def calculate(self, prices: pd.Series) -> AdaptiveTrendResult:
        """Calculate efficiency ratio, adaptive filter values, and trend directions.

        Args:
            prices: Series of prices with monotonic or volatile changes.

        Returns:
            AdaptiveTrendResult with aligned series and latest trend direction.

        Raises:
            ValueError: If prices series is empty, not a Series, length < period + 1,
                        or contains non-finite values (NaN / Inf).
        """
        if prices is None or not isinstance(prices, pd.Series):
            raise ValueError("prices must be a valid pandas Series.")
        if prices.empty:
            raise ValueError("prices series cannot be empty.")
        if len(prices) < self.period + 1:
            raise ValueError(
                f"Insufficient data points: prices length ({len(prices)}) "
                f"must be at least period + 1 ({self.period + 1})."
            )

        prices_clean = prices.astype(float)
        prices_arr = prices_clean.to_numpy()

        if not np.isfinite(prices_arr).all():
            raise ValueError("prices series contains non-finite values (NaN or Inf).")

        n_points = len(prices_arr)

        # 1. Compute Efficiency Ratio (ER) across lookback period
        change_arr = np.full(n_points, np.nan, dtype=float)
        change_arr[self.period:] = np.abs(
            prices_arr[self.period:] - prices_arr[:-self.period]
        )

        abs_diff = np.abs(np.diff(prices_arr))
        cs = np.cumsum(np.insert(abs_diff, 0, 0.0))
        vol_arr = np.full(n_points, np.nan, dtype=float)
        vol_arr[self.period:] = cs[self.period:n_points] - cs[: n_points - self.period]

        valid_vol = ~np.isnan(vol_arr)
        nonzero_vol = valid_vol & (vol_arr > 0.0)

        safe_vol = np.where(nonzero_vol, vol_arr, 1.0)
        ratio = np.clip(change_arr / safe_vol, 0.0, 1.0)

        er_arr = np.where(
            valid_vol,
            np.where(vol_arr == 0.0, 0.0, ratio),
            np.nan,
        )

        # 2. Compute dynamic smoothing constants (SC)
        sc_arr = np.full(n_points, np.nan, dtype=float)
        valid_er = ~np.isnan(er_arr)
        sc_arr[valid_er] = (
            er_arr[valid_er] * (self._fast_sc - self._slow_sc) + self._slow_sc
        ) ** 2

        # 3. Iteratively compute adaptive filter values and directions
        kama = np.full(n_points, np.nan, dtype=float)
        trend = np.full(n_points, None, dtype=object)

        # Seed KAMA at the final bar of the initial lookback window
        seed_idx = self.period - 1
        kama[seed_idx] = prices_arr[seed_idx]

        for i in range(self.period, n_points):
            current_sc = sc_arr[i]
            if np.isnan(current_sc):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + current_sc * (prices_arr[i] - kama[i - 1])

            diff = kama[i] - kama[i - 1]
            if diff > 1e-9:
                trend[i] = TrendDirection.UPWARD
            elif diff < -1e-9:
                trend[i] = TrendDirection.DOWNWARD
            else:
                trend[i] = TrendDirection.FLAT

        filter_values = pd.Series(kama, index=prices.index, name=prices.name)
        efficiency_ratio = pd.Series(er_arr, index=prices.index, name=prices.name)
        trend_direction = pd.Series(trend, index=prices.index, name=prices.name, dtype=object)
        current_direction = (
            trend[-1]
            if (n_points > 0 and trend[-1] is not None)
            else TrendDirection.FLAT
        )

        return AdaptiveTrendResult(
            filter_values=filter_values,
            efficiency_ratio=efficiency_ratio,
            trend_direction=trend_direction,
            current_direction=current_direction,
        )