from decimal import Decimal
import pytest

from src.execution.models import PositionLot, PyramidingLimitBreachError
from src.execution.inventory_tracker import InventoryTracker


class TestInventoryTrackerPyramiding:
    """Unit tests for multi-position inventory tracking and pyramiding limits."""

    def test_add_entry_lot_increments_layer_and_updates_vwap_and_size(self):
        """
        GIVEN an inventory tracker with a configured maximum pyramiding level of 3
              and 1 open position lot
        WHEN a valid new entry lot is submitted
        THEN the lot is appended to inventory, aggregate position size and VWAP
             are updated, and active layers increment to 2.
        """
        tracker = InventoryTracker(max_pyramiding_level=3)

        first_lot = PositionLot(
            lot_id="LOT-001",
            quantity=Decimal("5"),
            price=Decimal("100.00"),
        )
        tracker.add_lot(first_lot)

        assert tracker.active_layers == 1
        assert tracker.aggregate_size == Decimal("5")
        assert tracker.vwap == Decimal("100.00")
        assert len(tracker.lots) == 1
        assert tracker.lots[0] == first_lot

        second_lot = PositionLot(
            lot_id="LOT-002",
            quantity=Decimal("5"),
            price=Decimal("110.00"),
        )
        tracker.add_lot(second_lot)

        # Expected VWAP = (5 * 100.00 + 5 * 110.00) / 10 = 105.00
        assert tracker.active_layers == 2
        assert tracker.aggregate_size == Decimal("10")
        assert tracker.vwap == Decimal("105.00")
        assert len(tracker.lots) == 2
        assert tracker.lots[1] == second_lot

    def test_add_entry_lot_calculates_weighted_vwap_correctly(self):
        """Verify VWAP calculation when lot quantities are unequal."""
        tracker = InventoryTracker(max_pyramiding_level=3)

        tracker.add_lot(
            PositionLot(lot_id="LOT-A", quantity=Decimal("2"), price=Decimal("100.00"))
        )
        tracker.add_lot(
            PositionLot(lot_id="LOT-B", quantity=Decimal("8"), price=Decimal("150.00"))
        )

        # Expected VWAP = (2 * 100.00 + 8 * 150.00) / 10 = 1400 / 10 = 140.00
        assert tracker.aggregate_size == Decimal("10")
        assert tracker.vwap == Decimal("140.00")
        assert tracker.active_layers == 2

    def test_pyramiding_limit_breach_rejects_entry_and_preserves_state(self):
        """
        GIVEN an inventory tracker where active position layers equal the maximum
              pyramiding level
        WHEN another entry lot is attempted
        THEN the addition is rejected with a pyramiding limit breach error and existing
             inventory state remains unchanged.
        """
        tracker = InventoryTracker(max_pyramiding_level=3)

        lot1 = PositionLot(lot_id="LOT-001", quantity=Decimal("2"), price=Decimal("100.00"))
        lot2 = PositionLot(lot_id="LOT-002", quantity=Decimal("3"), price=Decimal("105.00"))
        lot3 = PositionLot(lot_id="LOT-003", quantity=Decimal("5"), price=Decimal("110.00"))

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)
        tracker.add_lot(lot3)

        assert tracker.active_layers == 3
        expected_size = Decimal("10")
        expected_vwap = Decimal("106.50")  # (200 + 315 + 550) / 10 = 1065 / 10 = 106.50
        assert tracker.aggregate_size == expected_size
        assert tracker.vwap == expected_vwap
        snapshot_lots = list(tracker.lots)

        excess_lot = PositionLot(lot_id="LOT-004", quantity=Decimal("1"), price=Decimal("120.00"))

        with pytest.raises(PyramidingLimitBreachError):
            tracker.add_lot(excess_lot)

        # Verify state is completely preserved
        assert tracker.active_layers == 3
        assert tracker.aggregate_size == expected_size
        assert tracker.vwap == expected_vwap
        assert tracker.lots == snapshot_lots
        assert excess_lot not in tracker.lots

    def test_pyramiding_limit_enforced_at_configured_boundary_one(self):
        """Verify pyramiding limit enforcement when max_pyramiding_level is 1."""
        tracker = InventoryTracker(max_pyramiding_level=1)

        tracker.add_lot(
            PositionLot(lot_id="LOT-1", quantity=Decimal("10"), price=Decimal("50.00"))
        )
        assert tracker.active_layers == 1

        second_lot = PositionLot(lot_id="LOT-2", quantity=Decimal("5"), price=Decimal("55.00"))
        with pytest.raises(PyramidingLimitBreachError):
            tracker.add_lot(second_lot)

        assert tracker.active_layers == 1
        assert tracker.aggregate_size == Decimal("10")


class TestInventoryTrackerFIFOExits:
    """Unit tests for FIFO-based lot reduction and partial exits."""

    def test_fifo_partial_exit_exact_first_lot_depletion(self):
        """
        GIVEN an inventory tracker with multiple open lots totaling 10 units
        WHEN a partial exit of 4 units is executed
        THEN lots are reduced via FIFO order and remaining aggregate size reflects 6 units.
        """
        tracker = InventoryTracker(max_pyramiding_level=3)

        lot1 = PositionLot(lot_id="LOT-1", quantity=Decimal("4"), price=Decimal("100.00"))
        lot2 = PositionLot(lot_id="LOT-2", quantity=Decimal("3"), price=Decimal("110.00"))
        lot3 = PositionLot(lot_id="LOT-3", quantity=Decimal("3"), price=Decimal("120.00"))

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)
        tracker.add_lot(lot3)

        assert tracker.aggregate_size == Decimal("10")
        assert tracker.active_layers == 3

        tracker.execute_exit(Decimal("4"))

        # Lot 1 should be completely removed via FIFO
        assert tracker.aggregate_size == Decimal("6")
        assert tracker.active_layers == 2
        assert len(tracker.lots) == 2
        assert tracker.lots[0].lot_id == "LOT-2"
        assert tracker.lots[0].quantity == Decimal("3")
        assert tracker.lots[1].lot_id == "LOT-3"
        assert tracker.lots[1].quantity == Decimal("3")
        # Remaining VWAP = (3 * 110.00 + 3 * 120.00) / 6 = 115.00
        assert tracker.vwap == Decimal("115.00")

    def test_fifo_partial_exit_within_first_lot(self):
        """Partial exit that only partially consumes the oldest open lot."""
        tracker = InventoryTracker(max_pyramiding_level=3)

        lot1 = PositionLot(lot_id="LOT-1", quantity=Decimal("6"), price=Decimal("100.00"))
        lot2 = PositionLot(lot_id="LOT-2", quantity=Decimal("4"), price=Decimal("120.00"))

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)

        tracker.execute_exit(Decimal("4"))

        # Lot 1 had 6 units, 4 exited, 2 remain. Lot 2 is untouched.
        assert tracker.aggregate_size == Decimal("6")
        assert tracker.active_layers == 2
        assert len(tracker.lots) == 2
        assert tracker.lots[0].lot_id == "LOT-1"
        assert tracker.lots[0].quantity == Decimal("2")
        assert tracker.lots[1].lot_id == "LOT-2"
        assert tracker.lots[1].quantity == Decimal("4")
        # Remaining VWAP = (2 * 100.00 + 4 * 120.00) / 6 = 680 / 6 = 113.3333...
        expected_vwap = (Decimal("2") * Decimal("100.00") + Decimal("4") * Decimal("120.00")) / Decimal("6")
        assert tracker.vwap == expected_vwap

    def test_fifo_partial_exit_spanning_multiple_lots(self):
        """Partial exit spanning across multiple lots, fully depleting one and partially another."""
        tracker = InventoryTracker(max_pyramiding_level=3)

        lot1 = PositionLot(lot_id="LOT-1", quantity=Decimal("2"), price=Decimal("100.00"))
        lot2 = PositionLot(lot_id="LOT-2", quantity=Decimal("3"), price=Decimal("110.00"))
        lot3 = PositionLot(lot_id="LOT-3", quantity=Decimal("5"), price=Decimal("120.00"))

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)
        tracker.add_lot(lot3)

        # Exit 4 units: consumes Lot 1 (2 units) fully, consumes 2 units of Lot 2 (1 unit remaining)
        tracker.execute_exit(Decimal("4"))

        assert tracker.aggregate_size == Decimal("6")
        assert tracker.active_layers == 2
        assert len(tracker.lots) == 2
        assert tracker.lots[0].lot_id == "LOT-2"
        assert tracker.lots[0].quantity == Decimal("1")
        assert tracker.lots[1].lot_id == "LOT-3"
        assert tracker.lots[1].quantity == Decimal("5")
        # Remaining VWAP = (1 * 110.00 + 5 * 120.00) / 6 = 710 / 6
        expected_vwap = (Decimal("1") * Decimal("110.00") + Decimal("5") * Decimal("120.00")) / Decimal("6")
        assert tracker.vwap == expected_vwap

    def test_full_exit_clears_inventory_and_resets_layers(self):
        """Full exit of all units resets aggregate size, layers, and VWAP."""
        tracker = InventoryTracker(max_pyramiding_level=3)

        tracker.add_lot(PositionLot(lot_id="LOT-1", quantity=Decimal("5"), price=Decimal("100.00")))
        tracker.add_lot(PositionLot(lot_id="LOT-2", quantity=Decimal("5"), price=Decimal("110.00")))

        tracker.execute_exit(Decimal("10"))

        assert tracker.aggregate_size == Decimal("0")
        assert tracker.active_layers == 0
        assert tracker.lots == []
        assert tracker.vwap == Decimal("0")


class TestInventoryTrackerValidation:
    """Unit tests for invalid inputs, negative quantities, and over-allocation."""

    def test_exit_quantity_exceeding_inventory_raises_error(self):
        """Attempting to exit more units than available raises ValueError and preserves state."""
        tracker = InventoryTracker(max_pyramiding_level=3)
        lot = PositionLot(lot_id="LOT-1", quantity=Decimal("10"), price=Decimal("100.00"))
        tracker.add_lot(lot)

        with pytest.raises(ValueError):
            tracker.execute_exit(Decimal("10.0001"))

        assert tracker.aggregate_size == Decimal("10")
        assert tracker.active_layers == 1
        assert len(tracker.lots) == 1

    @pytest.mark.parametrize("invalid_qty", [Decimal("0"), Decimal("-1"), Decimal("-0.001")])
    def test_execute_exit_with_non_positive_quantity_raises_error(self, invalid_qty):
        """Exiting non-positive quantity must raise ValueError."""
        tracker = InventoryTracker(max_pyramiding_level=3)
        tracker.add_lot(PositionLot(lot_id="LOT-1", quantity=Decimal("5"), price=Decimal("100.00")))

        with pytest.raises(ValueError):
            tracker.execute_exit(invalid_qty)

    @pytest.mark.parametrize(
        "quantity,price",
        [
            (Decimal("0"), Decimal("100.00")),
            (Decimal("-5"), Decimal("100.00")),
            (Decimal("5"), Decimal("0")),
            (Decimal("5"), Decimal("-10.00")),
        ],
    )
    def test_position_lot_validation_rejects_non_positive_values(self, quantity, price):
        """PositionLot rejects zero or negative quantity or price."""
        with pytest.raises(ValueError):
            PositionLot(lot_id="INVALID", quantity=quantity, price=price)

    @pytest.mark.parametrize("invalid_level", [0, -1, -5])
    def test_tracker_initialization_rejects_non_positive_max_pyramiding_level(self, invalid_level):
        """InventoryTracker rejects zero or negative max_pyramiding_level."""
        with pytest.raises(ValueError):
            InventoryTracker(max_pyramiding_level=invalid_level)