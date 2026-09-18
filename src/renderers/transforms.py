"""
Transform renderers and data structures for Heikin-Ashi, Renko, and Kagi charting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

__all__ = [
    "OHLCBar",
    "HeikinAshiBar",
    "HeikinAshiRenderer",
    "RenkoBrick",
    "RenkoDirection",
    "RenkoRenderer",
    "KagiSegment",
    "KagiState",
    "KagiRenderer",
    "transform_heikin_ashi",
    "transform_renko",
    "transform_kagi",
]


# ============================================================================
# Heikin-Ashi Transform Renderer
# ============================================================================


@dataclass(frozen=True)
class OHLCBar:
    """Standard Open-High-Low-Close bar representation."""

    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class HeikinAshiBar:
    """Heikin-Ashi transformed bar representation."""

    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float


class HeikinAshiRenderer:
    """Renderer computing Heikin-Ashi transformed bars from standard OHLC time series."""

    def render(self, bars: List[OHLCBar]) -> List[HeikinAshiBar]:
        """
        Transforms standard OHLC bars into Heikin-Ashi bars.

        Formula:
        - ha_close = (open + high + low + close) / 4
        - ha_open (first bar) = (open + close) / 2
        - ha_open (subsequent) = (prev_ha_open + prev_ha_close) / 2
        - ha_high = max(high, ha_open, ha_close)
        - ha_low = min(low, ha_open, ha_close)
        """
        if not bars:
            return []

        for bar in bars:
            if bar.open < 0 or bar.high < 0 or bar.low < 0 or bar.close < 0:
                raise ValueError("OHLC price values must be non-negative.")
            if bar.high < bar.low:
                raise ValueError("OHLC bar high cannot be strictly less than low.")

        result: List[HeikinAshiBar] = []
        first = bars[0]
        ha_open = (first.open + first.close) / 2.0
        ha_close = (first.open + first.high + first.low + first.close) / 4.0
        ha_high = max(first.high, ha_open, ha_close)
        ha_low = min(first.low, ha_open, ha_close)

        result.append(
            HeikinAshiBar(
                ha_open=ha_open,
                ha_high=ha_high,
                ha_low=ha_low,
                ha_close=ha_close,
            )
        )

        for bar in bars[1:]:
            ha_open = (ha_open + ha_close) / 2.0
            ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0
            ha_high = max(bar.high, ha_open, ha_close)
            ha_low = min(bar.low, ha_open, ha_close)
            result.append(
                HeikinAshiBar(
                    ha_open=ha_open,
                    ha_high=ha_high,
                    ha_low=ha_low,
                    ha_close=ha_close,
                )
            )

        return result


def transform_heikin_ashi(bars: List[OHLCBar]) -> List[HeikinAshiBar]:
    """Convenience function to transform standard OHLC bars to Heikin-Ashi bars."""
    return HeikinAshiRenderer().render(bars)


# ============================================================================
# Renko Transform Renderer
# ============================================================================


class RenkoDirection(Enum):
    """Directional state of a Renko brick."""

    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True)
class RenkoBrick:
    """Discrete brick entity produced by the Renko transform."""

    open_price: float
    close_price: float
    direction: RenkoDirection


class RenkoRenderer:
    """Renderer computing discrete Renko bricks based on price changes exceeding a fixed brick size."""

    def __init__(self, brick_size: float) -> None:
        if brick_size <= 0:
            raise ValueError("brick_size must be strictly positive.")
        self.brick_size = float(brick_size)

    def render(self, prices: List[float]) -> List[RenkoBrick]:
        """Produces discrete Renko bricks filtered by the configured brick threshold."""
        if len(prices) < 2:
            return []

        brick_size = self.brick_size
        bricks: List[RenkoBrick] = []
        anchor = float(prices[0])

        for p in prices[1:]:
            p = float(p)
            if not bricks:
                diff = p - anchor
                if diff >= brick_size - 1e-9:
                    num_bricks = int((diff + 1e-9) // brick_size)
                    for k in range(num_bricks):
                        b_open = anchor + k * brick_size
                        b_close = anchor + (k + 1) * brick_size
                        bricks.append(
                            RenkoBrick(
                                open_price=b_open,
                                close_price=b_close,
                                direction=RenkoDirection.UP,
                            )
                        )
                elif -diff >= brick_size - 1e-9:
                    num_bricks = int((-diff + 1e-9) // brick_size)
                    for k in range(num_bricks):
                        b_open = anchor - k * brick_size
                        b_close = anchor - (k + 1) * brick_size
                        bricks.append(
                            RenkoBrick(
                                open_price=b_open,
                                close_price=b_close,
                                direction=RenkoDirection.DOWN,
                            )
                        )
            else:
                last = bricks[-1]
                top = max(last.open_price, last.close_price)
                bottom = min(last.open_price, last.close_price)

                if last.direction == RenkoDirection.UP:
                    diff_up = p - top
                    if diff_up >= brick_size - 1e-9:
                        num_bricks = int((diff_up + 1e-9) // brick_size)
                        for k in range(num_bricks):
                            b_open = top + k * brick_size
                            b_close = top + (k + 1) * brick_size
                            bricks.append(
                                RenkoBrick(
                                    open_price=b_open,
                                    close_price=b_close,
                                    direction=RenkoDirection.UP,
                                )
                            )
                    else:
                        diff_down = bottom - p
                        if diff_down >= brick_size - 1e-9:
                            num_bricks = int((diff_down + 1e-9) // brick_size)
                            for k in range(num_bricks):
                                b_open = bottom - k * brick_size
                                b_close = bottom - (k + 1) * brick_size
                                bricks.append(
                                    RenkoBrick(
                                        open_price=b_open,
                                        close_price=b_close,
                                        direction=RenkoDirection.DOWN,
                                    )
                                )
                else:  # RenkoDirection.DOWN
                    diff_down = bottom - p
                    if diff_down >= brick_size - 1e-9:
                        num_bricks = int((diff_down + 1e-9) // brick_size)
                        for k in range(num_bricks):
                            b_open = bottom - k * brick_size
                            b_close = bottom - (k + 1) * brick_size
                            bricks.append(
                                RenkoBrick(
                                    open_price=b_open,
                                    close_price=b_close,
                                    direction=RenkoDirection.DOWN,
                                )
                            )
                    else:
                        diff_up = p - top
                        if diff_up >= brick_size - 1e-9:
                            num_bricks = int((diff_up + 1e-9) // brick_size)
                            for k in range(num_bricks):
                                b_open = top + k * brick_size
                                b_close = top + (k + 1) * brick_size
                                bricks.append(
                                    RenkoBrick(
                                        open_price=b_open,
                                        close_price=b_close,
                                        direction=RenkoDirection.UP,
                                    )
                                )

        return bricks


def transform_renko(prices: List[float], brick_size: float) -> List[RenkoBrick]:
    """Convenience function to transform a close price series into Renko bricks."""
    return RenkoRenderer(brick_size=brick_size).render(prices)


# ============================================================================
# Kagi Transform Renderer
# ============================================================================


class KagiState(Enum):
    """Trend/thickness state of a Kagi chart segment."""

    YANG = "YANG"
    YIN = "YIN"


@dataclass(frozen=True)
class KagiSegment:
    """Continuous trend line segment produced by the Kagi transform."""

    start_price: float
    end_price: float
    state: KagiState


class KagiRenderer:
    """Renderer computing Kagi trend-reversal line segments with Yang/Yin state transitions."""

    def __init__(self, reversal_pct: float) -> None:
        if reversal_pct <= 0:
            raise ValueError("reversal_pct must be strictly positive.")
        self.reversal_pct = float(reversal_pct)

    def render(self, prices: List[float]) -> List[KagiSegment]:
        """
        Produces Kagi trend-reversal line segments switching state between
        yang (bullish) and yin (bearish) upon surpassing prior breakout levels.
        """
        if any(p <= 0 for p in prices):
            raise ValueError("Price values must be positive.")
        if len(prices) < 2:
            return []

        reversal_pct = self.reversal_pct
        start_price = float(prices[0])

        init_idx = -1
        direction: Optional[str] = None
        for i in range(1, len(prices)):
            p = float(prices[i])
            if (p - start_price) / start_price >= reversal_pct - 1e-9:
                direction = "UP"
                init_idx = i
                break
            if (start_price - p) / start_price >= reversal_pct - 1e-9:
                direction = "DOWN"
                init_idx = i
                break

        if direction is None:
            return []

        segments: List[KagiSegment] = []
        current_segment_start = start_price
        extreme = float(prices[init_idx])
        current_state = KagiState.YANG if direction == "UP" else KagiState.YIN
        prior_swing_high: Optional[float] = None
        prior_swing_low: Optional[float] = None

        for i in range(init_idx + 1, len(prices)):
            p = float(prices[i])
            if direction == "UP":
                if p >= extreme:
                    extreme = p
                    if prior_swing_high is not None and extreme > prior_swing_high:
                        current_state = KagiState.YANG
                elif (extreme - p) / extreme >= reversal_pct - 1e-9:
                    prior_swing_high = extreme
                    segments.append(
                        KagiSegment(
                            start_price=current_segment_start,
                            end_price=extreme,
                            state=current_state,
                        )
                    )
                    direction = "DOWN"
                    current_segment_start = extreme
                    extreme = p
                    if prior_swing_low is not None and extreme < prior_swing_low:
                        current_state = KagiState.YIN
            else:  # direction == "DOWN"
                if p <= extreme:
                    extreme = p
                    if prior_swing_low is not None and extreme < prior_swing_low:
                        current_state = KagiState.YIN
                elif (p - extreme) / extreme >= reversal_pct - 1e-9:
                    prior_swing_low = extreme
                    segments.append(
                        KagiSegment(
                            start_price=current_segment_start,
                            end_price=extreme,
                            state=current_state,
                        )
                    )
                    direction = "UP"
                    current_segment_start = extreme
                    extreme = p
                    if prior_swing_high is not None and extreme > prior_swing_high:
                        current_state = KagiState.YANG

        segments.append(
            KagiSegment(
                start_price=current_segment_start,
                end_price=extreme,
                state=current_state,
            )
        )
        return segments


def transform_kagi(prices: List[float], reversal_pct: float) -> List[KagiSegment]:
    """Convenience function to transform a price series into Kagi segments."""
    return KagiRenderer(reversal_pct=reversal_pct).render(prices)