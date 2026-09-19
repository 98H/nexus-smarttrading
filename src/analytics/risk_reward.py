"""Risk-Reward calculation tool for trading positions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRewardResult:
    """Represents the outcome of a risk-reward calculation."""

    risk: float
    reward: float
    ratio: float

    @property
    def risk_reward_ratio(self) -> float:
        """Alias for ratio."""
        return self.ratio


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_type: str,
) -> RiskRewardResult:
    """Calculate the risk, reward, and risk-reward ratio for a position.

    Args:
        entry_price: The entry price of the position (must be positive).
        stop_loss: The stop loss price (must be positive).
        take_profit: The take profit target price (must be positive).
        position_type: Position side ('LONG' or 'SHORT', case-insensitive).

    Returns:
        RiskRewardResult containing risk, reward, and risk-reward ratio.

    Raises:
        ValueError: If prices are non-positive, position_type is unsupported,
            or prices violate position logic.
    """
    if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
        raise ValueError("Prices must be positive numbers greater than zero.")

    normalized_type = position_type.strip().upper()
    if normalized_type not in ("LONG", "SHORT"):
        raise ValueError(
            f"Unsupported position type '{position_type}'. Must be 'LONG' or 'SHORT'."
        )

    if normalized_type == "LONG":
        if stop_loss >= entry_price:
            raise ValueError(
                f"For LONG positions, stop loss ({stop_loss}) must be less than entry price ({entry_price})."
            )
        if take_profit <= entry_price:
            raise ValueError(
                f"For LONG positions, take profit ({take_profit}) must be greater than entry price ({entry_price})."
            )
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        if stop_loss <= entry_price:
            raise ValueError(
                f"For SHORT positions, stop loss ({stop_loss}) must be greater than entry price ({entry_price})."
            )
        if take_profit >= entry_price:
            raise ValueError(
                f"For SHORT positions, take profit ({take_profit}) must be less than entry price ({entry_price})."
            )
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    ratio = reward / risk
    return RiskRewardResult(risk=risk, reward=reward, ratio=ratio)