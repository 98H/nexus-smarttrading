"""
Maintenance Margin Calculator module.

Provides models and calculation logic for evaluating account solvency,
unrealized profit/loss, maintenance margin requirements, and margin shortfalls.
"""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Position:
    """Represents an open financial position."""

    symbol: str
    size: Decimal
    entry_price: Decimal
    maintenance_margin_rate: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.size, Decimal):
            self.size = Decimal(str(self.size))
        if not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if not isinstance(self.maintenance_margin_rate, Decimal):
            self.maintenance_margin_rate = Decimal(str(self.maintenance_margin_rate))

        if self.maintenance_margin_rate <= Decimal("0"):
            raise ValueError(
                f"Invalid maintenance margin rate: {self.maintenance_margin_rate}. "
                "Must be greater than zero."
            )


@dataclass
class Account:
    """Represents a trading account with cash balance and open positions."""

    account_id: str
    balance: Decimal
    positions: list[Position] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.balance, Decimal):
            self.balance = Decimal(str(self.balance))


@dataclass(frozen=True)
class MarginEvaluation:
    """Result of an account solvency and margin requirement evaluation."""

    account_id: str
    is_solvent: bool
    equity: Decimal
    maintenance_margin: Decimal
    shortfall: Decimal


class MaintenanceMarginCalculator:
    """Calculates maintenance margin requirements and evaluates solvency."""

    def evaluate_solvency(
        self, account: Account, mark_prices: dict[str, Decimal]
    ) -> MarginEvaluation:
        """
        Evaluate account solvency against current mark prices.

        Raises:
            KeyError: If a mark price is missing for any active position.
        """
        total_unrealized_pnl = Decimal("0.00")
        total_maintenance_margin = Decimal("0.00")

        for position in account.positions:
            if position.size == Decimal("0"):
                continue

            if position.symbol not in mark_prices:
                raise KeyError(
                    f"Mark price not available for active position symbol: '{position.symbol}'"
                )

            mark_price = mark_prices[position.symbol]
            unrealized_pnl = (mark_price - position.entry_price) * position.size
            position_mm = (
                abs(position.size) * mark_price * position.maintenance_margin_rate
            )

            total_unrealized_pnl += unrealized_pnl
            total_maintenance_margin += position_mm

        equity = account.balance + total_unrealized_pnl

        if equity >= total_maintenance_margin:
            is_solvent = True
            shortfall = Decimal("0.00")
        else:
            is_solvent = False
            shortfall = total_maintenance_margin - equity

        return MarginEvaluation(
            account_id=account.account_id,
            is_solvent=is_solvent,
            equity=equity,
            maintenance_margin=total_maintenance_margin,
            shortfall=shortfall,
        )