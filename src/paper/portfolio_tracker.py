"""Portfolio state, balances, and real-time PnL tracker."""

from decimal import Decimal

from src.paper.models import Position, PositionSide, Trade, TradeSide


class PortfolioTracker:
    """Tracks cash balances, open positions, MTM valuation, and margin utilization."""

    def __init__(
        self,
        initial_cash: Decimal,
        maintenance_margin_ratio: Decimal,
    ) -> None:
        self.initial_cash: Decimal = (
            Decimal(str(initial_cash)) if not isinstance(initial_cash, Decimal) else initial_cash
        )
        self.maintenance_margin_ratio: Decimal = (
            Decimal(str(maintenance_margin_ratio))
            if not isinstance(maintenance_margin_ratio, Decimal)
            else maintenance_margin_ratio
        )
        self.cash_balance: Decimal = self.initial_cash
        self.total_realized_pnl: Decimal = Decimal("0.00")
        self._positions: dict[str, Position] = {}

    def get_position(self, symbol: str) -> Position | None:
        """Return the active position for a symbol, if any."""
        return self._positions.get(symbol)

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update mark-to-market price for an active symbol."""
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
        if price < Decimal("0"):
            raise ValueError("Price cannot be negative")

        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def execute_trade(self, trade: Trade) -> None:
        """Process a trade execution, updating positions, cash, and realized PnL."""
        if trade.quantity <= Decimal("0"):
            raise ValueError("Trade quantity must be positive")
        if trade.price <= Decimal("0"):
            raise ValueError("Trade price must be positive")

        symbol = trade.symbol
        existing = self._positions.get(symbol)

        if existing is None:
            side = PositionSide.LONG if trade.side == TradeSide.BUY else PositionSide.SHORT
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
            )
            return

        is_increasing = (
            (existing.side == PositionSide.LONG and trade.side == TradeSide.BUY)
            or (existing.side == PositionSide.SHORT and trade.side == TradeSide.SELL)
        )

        if is_increasing:
            new_quantity = existing.quantity + trade.quantity
            new_entry_price = (
                existing.quantity * existing.entry_price + trade.quantity * trade.price
            ) / new_quantity
            existing.quantity = new_quantity
            existing.entry_price = new_entry_price
            existing.current_price = trade.price
            return

        # Trade is in opposing direction: reducing, closing, or reversing
        if trade.quantity < existing.quantity:
            closed_quantity = trade.quantity
            realized = self._calculate_realized_pnl(existing, closed_quantity, trade.price)
            self._apply_realized_pnl(realized)
            existing.quantity -= closed_quantity
            existing.current_price = trade.price

        elif trade.quantity == existing.quantity:
            closed_quantity = trade.quantity
            realized = self._calculate_realized_pnl(existing, closed_quantity, trade.price)
            self._apply_realized_pnl(realized)
            del self._positions[symbol]

        else:
            # Position reversal
            closed_quantity = existing.quantity
            realized = self._calculate_realized_pnl(existing, closed_quantity, trade.price)
            self._apply_realized_pnl(realized)

            excess_quantity = trade.quantity - existing.quantity
            new_side = PositionSide.SHORT if trade.side == TradeSide.SELL else PositionSide.LONG
            self._positions[symbol] = Position(
                symbol=symbol,
                side=new_side,
                quantity=excess_quantity,
                entry_price=trade.price,
                current_price=trade.price,
            )

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Net unrealized profit/loss across all active positions."""
        if not self._positions:
            return Decimal("0.00")
        return sum(
            (pos.unrealized_pnl for pos in self._positions.values()),
            start=Decimal("0.00"),
        )

    @property
    def total_equity(self) -> Decimal:
        """Total account equity: cash balance + net unrealized PnL."""
        return self.cash_balance + self.total_unrealized_pnl

    @property
    def maintenance_margin(self) -> Decimal:
        """Total maintenance margin required across all positions."""
        if not self._positions:
            return Decimal("0.00")
        total_notional = sum(
            (pos.notional_value for pos in self._positions.values()),
            start=Decimal("0.00"),
        )
        return total_notional * self.maintenance_margin_ratio

    @property
    def margin_utilization(self) -> Decimal:
        """Ratio of required maintenance margin to total account equity."""
        if not self._positions:
            return Decimal("0.00")

        equity = self.total_equity
        if equity <= Decimal("0.00"):
            return Decimal("Infinity")

        return self.maintenance_margin / equity

    def _calculate_realized_pnl(
        self,
        position: Position,
        closed_quantity: Decimal,
        exit_price: Decimal,
    ) -> Decimal:
        """Calculate realized PnL for a closed position segment."""
        if position.side == PositionSide.LONG:
            return (exit_price - position.entry_price) * closed_quantity
        return (position.entry_price - exit_price) * closed_quantity

    def _apply_realized_pnl(self, realized_pnl: Decimal) -> None:
        """Lock realized PnL into cash balance and realized accumulator."""
        self.total_realized_pnl += realized_pnl
        self.cash_balance += realized_pnl