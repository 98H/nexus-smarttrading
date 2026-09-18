"""
Liquidation Engine module.

Evaluates account solvency and generates orderly liquidation events
to eliminate margin deficits when accounts fall below maintenance margin requirements.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from src.risk.margin_calculator import Account, MaintenanceMarginCalculator


class LiquidationSide(str, Enum):
    """Execution side for liquidation orders."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class LiquidationEvent:
    """Represents an order generated to reduce or close an insolvent position."""

    account_id: str
    symbol: str
    side: LiquidationSide
    mark_price: Decimal
    volume: Decimal


@dataclass
class LiquidationEvaluation:
    """Evaluation summary including solvency status and any required liquidation events."""

    account_id: str
    is_solvent: bool
    shortfall: Decimal
    liquidation_events: list[LiquidationEvent] = field(default_factory=list)


class LiquidationEngine:
    """Orchestrates solvency checks and liquidation event generation."""

    def __init__(
        self,
        margin_calculator: MaintenanceMarginCalculator | None = None,
    ) -> None:
        self.margin_calculator = margin_calculator or MaintenanceMarginCalculator()

    def evaluate_account(
        self, account: Account, mark_prices: dict[str, Decimal]
    ) -> LiquidationEvaluation:
        """
        Evaluate account solvency and determine liquidation actions if in deficit.

        Raises:
            KeyError: If a mark price is missing for any active position.
        """
        margin_eval = self.margin_calculator.evaluate_solvency(account, mark_prices)

        if margin_eval.is_solvent:
            return LiquidationEvaluation(
                account_id=account.account_id,
                is_solvent=True,
                shortfall=Decimal("0.00"),
                liquidation_events=[],
            )

        liquidation_events = self._generate_liquidation_events(
            account, mark_prices, margin_eval.shortfall
        )

        return LiquidationEvaluation(
            account_id=account.account_id,
            is_solvent=False,
            shortfall=margin_eval.shortfall,
            liquidation_events=liquidation_events,
        )

    def process_account(
        self, account: Account, mark_prices: dict[str, Decimal]
    ) -> list[LiquidationEvent]:
        """
        Process an account and return liquidation events needed to restore margin requirements.

        Raises:
            KeyError: If a mark price is missing for any active position.
        """
        evaluation = self.evaluate_account(account, mark_prices)
        return evaluation.liquidation_events

    def _generate_liquidation_events(
        self,
        account: Account,
        mark_prices: dict[str, Decimal],
        initial_shortfall: Decimal,
    ) -> list[LiquidationEvent]:
        """Calculate required liquidation volume per position until the shortfall is covered."""
        events: list[LiquidationEvent] = []
        remaining_shortfall = initial_shortfall

        for position in account.positions:
            if remaining_shortfall <= Decimal("0"):
                break

            if position.size == Decimal("0"):
                continue

            mark_price = mark_prices[position.symbol]
            side = (
                LiquidationSide.SELL
                if position.size > Decimal("0")
                else LiquidationSide.BUY
            )
            open_volume = abs(position.size)

            unit_margin_requirement = mark_price * position.maintenance_margin_rate
            if unit_margin_requirement <= Decimal("0"):
                continue

            required_volume = remaining_shortfall / unit_margin_requirement
            liquidation_volume = min(open_volume, required_volume)

            if liquidation_volume > Decimal("0"):
                events.append(
                    LiquidationEvent(
                        account_id=account.account_id,
                        symbol=position.symbol,
                        side=side,
                        mark_price=mark_price,
                        volume=liquidation_volume,
                    )
                )
                margin_relieved = liquidation_volume * unit_margin_requirement
                remaining_shortfall -= margin_relieved

        return events