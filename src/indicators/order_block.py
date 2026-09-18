"""
Dynamic Order Block (OB) Detection and Mitigation Tracker.

This module provides detection of institutional Order Blocks formed by
swing breakout candle sequences in OHLCV series, and tracks their subsequent
mitigation state as price evolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class OrderBlockType(str, Enum):
    """Classification of Order Block bias."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class OrderBlock:
    """Represents an Order Block and its mitigation status."""

    timestamp: Any
    block_type: OrderBlockType
    high: float
    low: float
    mitigated: bool = False
    mitigated_at: Any = None
    confirmed_at: Any = None


class OrderBlockTracker:
    """Tracks active Order Blocks and updates mitigation state bar-by-bar."""

    def __init__(
        self,
        blocks: list[OrderBlock] | None = None,
        prev_candle: dict[str, Any] | pd.Series | None = None,
    ) -> None:
        self._blocks: list[OrderBlock] = list(blocks) if blocks is not None else []
        self._prev_candle: dict[str, Any] | None = None
        if prev_candle is not None:
            self.set_previous_candle(prev_candle)

    @property
    def active_blocks(self) -> list[OrderBlock]:
        """Return all currently unmitigated order blocks."""
        return [b for b in self._blocks if not b.mitigated]

    @property
    def mitigated_blocks(self) -> list[OrderBlock]:
        """Return all mitigated order blocks."""
        return [b for b in self._blocks if b.mitigated]

    @property
    def prev_candle(self) -> dict[str, Any] | None:
        """Return the most recent reference candle used for gap evaluation."""
        return self._prev_candle

    @prev_candle.setter
    def prev_candle(self, candle: dict[str, Any] | pd.Series | None) -> None:
        """Set the reference candle for gap mitigation detection."""
        self.set_previous_candle(candle)

    def set_previous_candle(self, candle: dict[str, Any] | pd.Series | None) -> None:
        """Set the reference candle for gap mitigation detection."""
        if candle is None:
            self._prev_candle = None
        elif isinstance(candle, dict):
            self._prev_candle = dict(candle)
        elif isinstance(candle, pd.Series):
            self._prev_candle = candle.to_dict()
        else:
            self._prev_candle = dict(candle)

    def add_block(self, block: OrderBlock, confirmed_at: Any = None) -> None:
        """Register a new order block for tracking."""
        if confirmed_at is not None and block.confirmed_at is None:
            block.confirmed_at = confirmed_at
        if not any(b is block for b in self._blocks):
            self._blocks.append(block)

    def update(self, candle: dict[str, Any] | pd.Series) -> list[OrderBlock]:
        """
        Evaluate a single incoming candle against active order blocks.

        Marks touched, penetrated, or gapped-through blocks as mitigated.
        """
        candle_dict = (
            candle.to_dict()
            if isinstance(candle, pd.Series)
            else dict(candle)
            if isinstance(candle, dict)
            else dict(candle)
        )

        candle_ts = candle_dict.get("timestamp")
        candle_high = float(candle_dict["high"])
        candle_low = float(candle_dict["low"])

        mitigated_in_this_step: list[OrderBlock] = []

        for block in self.active_blocks:
            # Mitigation must only occur from subsequent price action post-confirmation
            effective_ts = (
                block.confirmed_at
                if block.confirmed_at is not None
                else block.timestamp
            )
            if effective_ts is not None and candle_ts is not None:
                try:
                    if candle_ts <= effective_ts:
                        continue
                except TypeError:
                    pass

            # Direct boundary touch or internal penetration
            touches_or_penetrates = (
                candle_low <= block.high and candle_high >= block.low
            )

            # Check for gap completely through the block
            is_gap_through = False
            if self._prev_candle is not None:
                prev_close = float(
                    self._prev_candle.get(
                        "close", self._prev_candle.get("high", 0.0)
                    )
                )

                # Gapped down through the block (was above, now entirely below)
                gap_down = (
                    prev_close > block.high and candle_high < block.low
                )
                # Gapped up through the block (was below, now entirely above)
                gap_up = (
                    prev_close < block.low and candle_low > block.high
                )
                is_gap_through = gap_down or gap_up

            if touches_or_penetrates or is_gap_through:
                block.mitigated = True
                if block.mitigated_at is None:
                    block.mitigated_at = candle_ts
                mitigated_in_this_step.append(block)

        self._prev_candle = candle_dict
        return mitigated_in_this_step


def detect_order_blocks(
    df: pd.DataFrame,
    swing_lookback: int = 50,
) -> list[OrderBlock]:
    """
    Detect unmitigated Order Blocks from swing breakouts and track their mitigation.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with required columns: timestamp, open, high, low, close.
    swing_lookback : int, default 50
        Maximum lookback window to scan for swing extremes.

    Returns
    -------
    list[OrderBlock]
        Detected OrderBlock instances with mitigation status resolved across the series.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    required_columns = {"timestamp", "open", "high", "low", "close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty or len(df) < 3:
        return []

    df_reset = df.reset_index(drop=True)
    records = df_reset.to_dict(orient="records")
    n = len(records)

    opens = df_reset["open"].astype(float).to_numpy()
    highs = df_reset["high"].astype(float).to_numpy()
    lows = df_reset["low"].astype(float).to_numpy()
    closes = df_reset["close"].astype(float).to_numpy()

    # Precompute swing highs and lows in O(N)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for s in range(n):
        if s == 0:
            is_swing_high[s] = n > 1 and highs[0] > highs[1]
            is_swing_low[s] = n > 1 and lows[0] < lows[1]
        else:
            sh_left = highs[s] > highs[s - 1] and (
                s < 2 or highs[s] > highs[s - 2]
            )
            sh_right = (s + 1 >= n) or (highs[s] > highs[s + 1])
            is_swing_high[s] = sh_left and sh_right

            sl_left = lows[s] < lows[s - 1] and (
                s < 2 or lows[s] < lows[s - 2]
            )
            sl_right = (s + 1 >= n) or (lows[s] < lows[s + 1])
            is_swing_low[s] = sl_left and sl_right

    tracker = OrderBlockTracker()
    detected_blocks: list[OrderBlock] = []
    seen_opposing_indices: set[int] = set()

    for i in range(n):
        candle = records[i]

        # 1. Update existing active blocks bar-by-bar
        tracker.update(candle)

        # 2. Check for breakout confirmation (minimum 3 candles required)
        if i < 2:
            continue

        # Case 1: Bullish breakout displacement candle
        if closes[i] > opens[i]:
            d = i
            while d > 0 and closes[d - 1] >= opens[d - 1]:
                d -= 1

            o = d - 1
            if o >= 0 and closes[o] < opens[o] and o not in seen_opposing_indices:
                min_s = max(0, o - max(1, swing_lookback))
                for s in range(o - 1, min_s - 1, -1):
                    if not is_swing_high[s]:
                        continue

                    swing_val = highs[s]

                    # Breakout verification: displacement closes above swing high
                    if closes[i] <= swing_val:
                        continue

                    # Pullback verification: price stayed below swing high
                    if d > s + 1 and np.max(highs[s + 1 : d]) >= swing_val:
                        continue

                    # Prior breakout check: candle i must be the first breakout candle
                    if i > s + 1 and np.max(closes[s + 1 : i]) > swing_val:
                        continue

                    ob = OrderBlock(
                        timestamp=records[o]["timestamp"],
                        block_type=OrderBlockType.BULLISH,
                        high=float(highs[o]),
                        low=float(lows[o]),
                        mitigated=False,
                        mitigated_at=None,
                        confirmed_at=candle["timestamp"],
                    )
                    tracker.add_block(ob)
                    detected_blocks.append(ob)
                    seen_opposing_indices.add(o)
                    break

        # Case 2: Bearish breakout displacement candle
        elif closes[i] < opens[i]:
            d = i
            while d > 0 and closes[d - 1] <= opens[d - 1]:
                d -= 1

            o = d - 1
            if o >= 0 and closes[o] > opens[o] and o not in seen_opposing_indices:
                min_s = max(0, o - max(1, swing_lookback))
                for s in range(o - 1, min_s - 1, -1):
                    if not is_swing_low[s]:
                        continue

                    swing_val = lows[s]

                    # Breakout verification: displacement closes below swing low
                    if closes[i] >= swing_val:
                        continue

                    # Pullback verification: price stayed above swing low
                    if d > s + 1 and np.min(lows[s + 1 : d]) <= swing_val:
                        continue

                    # Prior breakout check: candle i must be the first breakout candle
                    if i > s + 1 and np.min(closes[s + 1 : i]) < swing_val:
                        continue

                    ob = OrderBlock(
                        timestamp=records[o]["timestamp"],
                        block_type=OrderBlockType.BEARISH,
                        high=float(highs[o]),
                        low=float(lows[o]),
                        mitigated=False,
                        mitigated_at=None,
                        confirmed_at=candle["timestamp"],
                    )
                    tracker.add_block(ob)
                    detected_blocks.append(ob)
                    seen_opposing_indices.add(o)
                    break

    return detected_blocks