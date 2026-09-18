"""Order book bid/ask volume imbalance analytics."""

from typing import Union


def calculate_order_imbalance(
    bid_volume: Union[int, float],
    ask_volume: Union[int, float],
) -> float:
    """Calculate normalized order imbalance score in range [-1.0, 1.0].

    The formula used is (bid_volume - ask_volume) / (bid_volume + ask_volume).
    Returns 0.0 when total volume is zero to safely prevent division by zero.

    Args:
        bid_volume: Aggregate volume on the bid side across depth.
        ask_volume: Aggregate volume on the ask side across depth.

    Returns:
        Normalized imbalance score between -1.0 and 1.0.

    Raises:
        TypeError: If bid_volume or ask_volume are not numeric.
        ValueError: If bid_volume or ask_volume are negative.
    """
    if isinstance(bid_volume, bool) or not isinstance(bid_volume, (int, float)):
        raise TypeError(f"bid_volume must be numeric, got {type(bid_volume).__name__}")
    if isinstance(ask_volume, bool) or not isinstance(ask_volume, (int, float)):
        raise TypeError(f"ask_volume must be numeric, got {type(ask_volume).__name__}")

    if bid_volume < 0:
        raise ValueError(f"bid_volume cannot be negative, got {bid_volume}")
    if ask_volume < 0:
        raise ValueError(f"ask_volume cannot be negative, got {ask_volume}")

    total_volume = bid_volume + ask_volume
    if total_volume == 0.0:
        return 0.0

    score = (bid_volume - ask_volume) / total_volume
    return max(-1.0, min(1.0, float(score)))