from decimal import Decimal

from src.execution.models import PositionLot, PyramidingLimitBreachError


class InventoryTracker:
    """Tracks multi-position inventory lots and enforces pyramiding constraints."""

    def __init__(self, max_pyramiding_level: int) -> None:
        if max_pyramiding_level <= 0:
            raise ValueError(
                f"max_pyramiding_level must be positive, got {max_pyramiding_level}"
            )
        self.max_pyramiding_level = max_pyramiding_level
        self._lots: list[PositionLot] = []

    @property
    def lots(self) -> list[PositionLot]:
        return list(self._lots)

    @property
    def active_layers(self) -> int:
        return len(self._lots)

    @property
    def aggregate_size(self) -> Decimal:
        return sum((lot.quantity for lot in self._lots), Decimal("0"))

    @property
    def vwap(self) -> Decimal:
        if not self._lots or self.aggregate_size == Decimal("0"):
            return Decimal("0")
        total_notional = sum((lot.quantity * lot.price for lot in self._lots), Decimal("0"))
        return total_notional / self.aggregate_size

    def add_lot(self, lot: PositionLot) -> None:
        if len(self._lots) >= self.max_pyramiding_level:
            raise PyramidingLimitBreachError(
                f"Active layers ({len(self._lots)}) reached maximum pyramiding level ({self.max_pyramiding_level})"
            )
        self._lots.append(lot)

    def execute_exit(self, quantity: Decimal) -> None:
        if quantity <= Decimal("0"):
            raise ValueError(f"Exit quantity must be positive, got {quantity}")
        if quantity > self.aggregate_size:
            raise ValueError(
                f"Exit quantity {quantity} exceeds aggregate inventory size {self.aggregate_size}"
            )

        remaining_to_exit = quantity
        new_lots: list[PositionLot] = []
        for lot in self._lots:
            if remaining_to_exit == Decimal("0"):
                new_lots.append(lot)
            elif lot.quantity <= remaining_to_exit:
                remaining_to_exit -= lot.quantity
            else:
                new_lots.append(
                    PositionLot(
                        lot_id=lot.lot_id,
                        quantity=lot.quantity - remaining_to_exit,
                        price=lot.price,
                    )
                )
                remaining_to_exit = Decimal("0")

        self._lots = new_lots