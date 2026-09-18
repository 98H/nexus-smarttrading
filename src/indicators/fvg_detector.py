"""Fair Value Gap (FVG) and Imbalance Zone Detector."""

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import pandas as pd


class FVGType(Enum):
    """Enumeration of Fair Value Gap types."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class Candle:
    """Represents a single price candlestick bar with OHLC values and bar index."""

    open: float
    high: float
    low: float
    close: float
    index: int = 0

    def __post_init__(self) -> None:
        try:
            self.open = float(self.open)
            self.high = float(self.high)
            self.low = float(self.low)
            self.close = float(self.close)
            self.index = int(self.index)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Candle values must be numeric: {exc}") from exc

        if (
            math.isnan(self.open)
            or math.isnan(self.high)
            or math.isnan(self.low)
            or math.isnan(self.close)
        ):
            raise ValueError("Candle prices cannot be NaN.")

        if self.high < self.low:
            raise ValueError(
                f"High ({self.high}) cannot be less than Low ({self.low})."
            )


@dataclass(frozen=True)
class FairValueGap:
    """Represents a detected Fair Value Gap / Imbalance Zone."""

    gap_type: FVGType
    lower_boundary: float
    upper_boundary: float
    bar_index: int

    @property
    def boundaries(self) -> tuple[float, float]:
        """Returns the lower and upper boundaries as a tuple."""
        return (self.lower_boundary, self.upper_boundary)

    @property
    def size(self) -> float:
        """Returns the gap size (height)."""
        return self.upper_boundary - self.lower_boundary


class FVGDetector:
    """Detects Fair Value Gaps (FVG) and imbalance zones across candlestick series."""

    def evaluate_sequence(
        self, c1: Candle, c2: Candle, c3: Candle
    ) -> Optional[FairValueGap]:
        """
        Evaluate a 3-candle sequence for Fair Value Gap formation.

        Bullish FVG: Candle 3 Low > Candle 1 High.
            Boundaries: [Candle 1 High, Candle 3 Low]
        Bearish FVG: Candle 3 High < Candle 1 Low.
            Boundaries: [Candle 3 High, Candle 1 Low]
        Overlap / No displacement: Returns None.
        """
        if c1 is None or c2 is None or c3 is None:
            raise ValueError("Sequence candles cannot be None.")

        if not (
            isinstance(c1, Candle)
            and isinstance(c2, Candle)
            and isinstance(c3, Candle)
        ):
            raise TypeError("All sequence items must be instances of Candle.")

        if c3.low > c1.high:
            return FairValueGap(
                gap_type=FVGType.BULLISH,
                lower_boundary=c1.high,
                upper_boundary=c3.low,
                bar_index=c3.index,
            )
        elif c3.high < c1.low:
            return FairValueGap(
                gap_type=FVGType.BEARISH,
                lower_boundary=c3.high,
                upper_boundary=c1.low,
                bar_index=c3.index,
            )

        return None

    def detect(
        self, data: Union[Sequence[Candle], pd.DataFrame]
    ) -> List[FairValueGap]:
        """
        Detect all Fair Value Gaps across a sequence of Candles or a pandas DataFrame.
        """
        if data is None:
            raise ValueError("Data cannot be None.")

        if isinstance(data, pd.DataFrame):
            if "high" not in data.columns or "low" not in data.columns:
                raise ValueError("DataFrame must contain 'high' and 'low' columns.")
            if len(data) < 3:
                return []
            candles = self._candles_from_dataframe(data)
        elif isinstance(data, (list, tuple)):
            if len(data) < 3:
                return []
            for item in data:
                if not isinstance(item, Candle):
                    raise TypeError("Elements of candle sequence must be Candle instances.")
            candles = list(data)
        else:
            raise TypeError("Data must be a Sequence[Candle] or pd.DataFrame.")

        gaps: List[FairValueGap] = []
        for i in range(2, len(candles)):
            gap = self.evaluate_sequence(candles[i - 2], candles[i - 1], candles[i])
            if gap is not None:
                gaps.append(gap)

        return gaps

    def _candles_from_dataframe(self, df: pd.DataFrame) -> List[Candle]:
        """Validate and extract Candle instances from an OHLC DataFrame."""
        ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

        for col in ohlc_cols:
            for val in df[col]:
                if pd.isna(val):
                    raise ValueError(f"NaN value found in column '{col}'.")
                if isinstance(val, str):
                    try:
                        num = float(val)
                        if math.isnan(num):
                            raise ValueError(f"NaN value found in column '{col}'.")
                    except ValueError as exc:
                        raise TypeError(f"Non-numeric value '{val}' in column '{col}'.") from exc
                elif not isinstance(val, (int, float, numbers.Number)):
                    raise TypeError(f"Non-numeric value '{val}' in column '{col}'.")
                elif isinstance(val, float) and math.isnan(val):
                    raise ValueError(f"NaN value found in column '{col}'.")

        candles: List[Candle] = []
        has_open = "open" in df.columns
        has_close = "close" in df.columns

        for i in range(len(df)):
            row = df.iloc[i]
            high_val = float(row["high"])
            low_val = float(row["low"])
            open_val = float(row["open"]) if has_open else low_val
            close_val = float(row["close"]) if has_close else high_val

            try:
                idx = int(df.index[i])
            except (ValueError, TypeError):
                idx = i

            candles.append(
                Candle(
                    open=open_val,
                    high=high_val,
                    low=low_val,
                    close=close_val,
                    index=idx,
                )
            )

        return candles


__all__ = [
    "Candle",
    "FairValueGap",
    "FVGDetector",
    "FVGType",
]