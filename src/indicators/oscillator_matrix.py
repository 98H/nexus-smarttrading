from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd


class BoundaryClassification(str, Enum):
    """Classification states for oscillator boundary levels."""

    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"
    NEUTRAL = "NEUTRAL"

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class OscillatorMatrixConfig:
    """Configuration parameters for the oscillator matrix composite engine."""

    overbought_threshold: float = 60.0
    oversold_threshold: float = -60.0
    lookback_windows: Sequence[int] = field(default_factory=lambda: [7, 14, 28])
    component_types: Sequence[str] = field(default_factory=lambda: ["rsi", "stochastic"])
    weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not (-100.0 <= self.overbought_threshold <= 100.0):
            raise ValueError(
                f"overbought_threshold ({self.overbought_threshold}) must be within [-100.0, 100.0]."
            )
        if not (-100.0 <= self.oversold_threshold <= 100.0):
            raise ValueError(
                f"oversold_threshold ({self.oversold_threshold}) must be within [-100.0, 100.0]."
            )
        if self.overbought_threshold <= self.oversold_threshold:
            raise ValueError(
                f"overbought_threshold ({self.overbought_threshold}) must be strictly greater "
                f"than oversold_threshold ({self.oversold_threshold})."
            )
        if not self.lookback_windows or any(lb <= 0 for lb in self.lookback_windows):
            raise ValueError("All lookback windows must be positive integers (> 0).")
        if self.weights is not None:
            if not self.weights:
                raise ValueError("Weights cannot be empty if specified.")
            if any(w <= 0.0 for w in self.weights.values()):
                raise ValueError("All weights must be strictly positive (> 0).")


@dataclass
class OscillatorMatrixResult:
    """Output results from the oscillator matrix composite evaluation."""

    composite: pd.Series
    classification: pd.Series
    matrix: pd.DataFrame


def _compute_rsi_norm(close: pd.Series, lookback: int) -> pd.Series:
    """Compute RSI normalized between -100.0 and 100.0."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window=lookback, min_periods=lookback).mean()
    avg_loss = loss.rolling(window=lookback, min_periods=lookback).mean()

    total = avg_gain + avg_loss
    rsi_norm = pd.Series(np.nan, index=close.index, dtype=float)

    valid = total.notna()
    non_zero = valid & (total > 0)
    rsi_norm[non_zero] = ((avg_gain[non_zero] - avg_loss[non_zero]) / total[non_zero]) * 100.0
    zero_mask = valid & (total == 0)
    rsi_norm[zero_mask] = 0.0
    return rsi_norm.clip(-100.0, 100.0)


def _compute_stochastic_norm(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
) -> pd.Series:
    """Compute Fast Stochastic %K normalized between -100.0 and 100.0."""
    low_min = low.rolling(window=lookback, min_periods=lookback).min()
    high_max = high.rolling(window=lookback, min_periods=lookback).max()
    hl_range = high_max - low_min

    stoch_norm = pd.Series(np.nan, index=close.index, dtype=float)
    valid = hl_range.notna()
    non_zero = valid & (hl_range > 0)
    stoch_norm[non_zero] = (
        (close[non_zero] - low_min[non_zero]) / hl_range[non_zero]
    ) * 200.0 - 100.0
    zero_mask = valid & (hl_range == 0)
    stoch_norm[zero_mask] = 0.0
    return stoch_norm.clip(-100.0, 100.0)


def _compute_cci_norm(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
) -> pd.Series:
    """Compute CCI bounded between -100.0 and 100.0."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=lookback, min_periods=lookback).mean()
    mad = tp.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    cci_norm = pd.Series(np.nan, index=close.index, dtype=float)
    valid = mad.notna()
    non_zero = valid & (mad > 0)
    cci_norm[non_zero] = (tp[non_zero] - sma_tp[non_zero]) / (0.015 * mad[non_zero])
    zero_mask = valid & (mad == 0)
    cci_norm[zero_mask] = 0.0
    return cci_norm.clip(-100.0, 100.0)


def _compute_mfi_norm(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, lookback: int
) -> pd.Series:
    """Compute MFI normalized between -100.0 and 100.0."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    delta = tp.diff()

    pos_mf = pd.Series(np.where(delta > 0, rmf, 0.0), index=close.index)
    neg_mf = pd.Series(np.where(delta < 0, rmf, 0.0), index=close.index)

    pos_sum = pos_mf.rolling(window=lookback, min_periods=lookback).sum()
    neg_sum = neg_mf.rolling(window=lookback, min_periods=lookback).sum()

    total = pos_sum + neg_sum
    mfi_norm = pd.Series(np.nan, index=close.index, dtype=float)
    valid = total.notna()
    non_zero = valid & (total > 0)
    mfi_norm[non_zero] = ((pos_sum[non_zero] - neg_sum[non_zero]) / total[non_zero]) * 100.0
    zero_mask = valid & (total == 0)
    mfi_norm[zero_mask] = 0.0
    return mfi_norm.clip(-100.0, 100.0)


def _classify_boundary(
    val: float, overbought_threshold: float, oversold_threshold: float
) -> BoundaryClassification:
    """Classify an oscillator value against designated threshold boundaries."""
    if pd.isna(val):
        return BoundaryClassification.NEUTRAL
    if val > overbought_threshold:
        return BoundaryClassification.OVERBOUGHT
    if val < oversold_threshold:
        return BoundaryClassification.OVERSOLD
    return BoundaryClassification.NEUTRAL


class OscillatorMatrixEngine:
    """Composite matrix engine for multi-lookback oscillator aggregation and boundary classification."""

    def __init__(self, config: OscillatorMatrixConfig | None = None) -> None:
        self.config = config if config is not None else OscillatorMatrixConfig()

    def _validate_inputs(
        self,
        ohlcv: pd.DataFrame,
        components: dict[str, pd.Series] | None,
    ) -> None:
        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            raise ValueError("OHLCV DataFrame must be a non-empty pandas DataFrame.")

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(ohlcv.columns):
            raise ValueError(
                f"OHLCV data is missing required columns: {required_cols - set(ohlcv.columns)}"
            )

        if components is not None:
            if not components:
                raise ValueError("Components dictionary cannot be empty when provided.")
            for name, series in components.items():
                if len(series) != len(ohlcv):
                    raise ValueError(
                        f"Component '{name}' length ({len(series)}) does not match OHLCV length ({len(ohlcv)})."
                    )
                if not series.index.equals(ohlcv.index):
                    raise ValueError(
                        f"Component '{name}' index does not match OHLCV index."
                    )

    def _build_component_matrix(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        matrix_dict: dict[str, pd.Series] = {}
        for ctype in self.config.component_types:
            key = ctype.lower().strip()
            for lb in self.config.lookback_windows:
                col_name = f"{key}_{lb}"
                if key in ("rsi", "rsi_norm"):
                    matrix_dict[col_name] = _compute_rsi_norm(ohlcv["close"], lb)
                elif key in ("stoch", "stochastic", "stoch_norm"):
                    matrix_dict[col_name] = _compute_stochastic_norm(
                        ohlcv["high"], ohlcv["low"], ohlcv["close"], lb
                    )
                elif key in ("cci", "cci_norm"):
                    matrix_dict[col_name] = _compute_cci_norm(
                        ohlcv["high"], ohlcv["low"], ohlcv["close"], lb
                    )
                elif key in ("mfi", "mfi_norm"):
                    matrix_dict[col_name] = _compute_mfi_norm(
                        ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], lb
                    )
                else:
                    raise ValueError(f"Unsupported oscillator component type: '{ctype}'")
        return pd.DataFrame(matrix_dict, index=ohlcv.index)

    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        components: dict[str, pd.Series] | None = None,
    ) -> OscillatorMatrixResult:
        """Evaluate the composite oscillator matrix and boundary classifications."""
        self._validate_inputs(ohlcv, components)

        if components is not None:
            matrix_df = pd.DataFrame(components, index=ohlcv.index)
        else:
            matrix_df = self._build_component_matrix(ohlcv)

        matrix_df = matrix_df.clip(-100.0, 100.0)

        weights = self.config.weights
        if weights is not None:
            weights_list: list[float] = []
            for col in matrix_df.columns:
                if col in weights:
                    weights_list.append(weights[col])
                else:
                    prefix = col.split("_")[0]
                    weights_list.append(weights.get(prefix, 1.0))
            weights_series = pd.Series(weights_list, index=matrix_df.columns, dtype=float)
            total_weight = weights_series.sum()
            composite = matrix_df.dot(weights_series) / total_weight
        else:
            composite = matrix_df.mean(axis=1, skipna=False)

        composite = composite.clip(-100.0, 100.0)
        composite.name = "composite"

        classification_values = [
            _classify_boundary(
                val,
                self.config.overbought_threshold,
                self.config.oversold_threshold,
            )
            for val in composite.values
        ]
        classification = pd.Series(
            classification_values, index=ohlcv.index, dtype=object, name="classification"
        )

        return OscillatorMatrixResult(
            composite=composite,
            classification=classification,
            matrix=matrix_df,
        )