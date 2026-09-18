from unittest.mock import Mock, call
import pytest

from src.execution.oco_manager import (
    BracketOrder,
    BracketStatus,
    ExecutionReport,
    OCOManager,
    OrderStatus,
)


@pytest.fixture
def mock_gateway() -> Mock:
    """Mock external order gateway responsible for sending I/O cancellation requests."""
    gateway = Mock()
    gateway.cancel_order.return_value = True
    return gateway


@pytest.fixture
def oco_manager(mock_gateway: Mock) -> OCOManager:
    """Returns an instance of OCOManager wired with the mocked external gateway."""
    return OCOManager(order_gateway=mock_gateway)


class TestBracketOrderOCOManager:
    """Tests for Story 7.3.1: Bracket Order (Take-Profit / Stop-Loss) OCO Manager."""

    def test_tp_filled_cancels_sl_and_closes_bracket(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Acceptance Criteria 1:

        Given an active bracket order with take-profit order ID "TP-1" and stop-loss order ID "SL-1",
        When an execution report indicates "TP-1" is filled,
        Then a cancellation request is issued for "SL-1" and the bracket is closed.
        """
        oco_manager.register_bracket(
            bracket_id="BRK-1", tp_order_id="TP-1", sl_order_id="SL-1"
        )

        report = ExecutionReport(order_id="TP-1", status=OrderStatus.FILLED)
        oco_manager.handle_execution_report(report)

        # SL leg must be cancelled via external gateway
        mock_gateway.cancel_order.assert_called_once_with("SL-1")

        # Bracket status must be closed
        bracket = oco_manager.get_bracket("BRK-1")
        assert bracket is not None
        assert bracket.status == BracketStatus.CLOSED

    def test_sl_filled_cancels_tp_and_closes_bracket(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Acceptance Criteria 2:

        Given an active bracket order with take-profit order ID "TP-2" and stop-loss order ID "SL-2",
        When an execution report indicates "SL-2" is filled,
        Then a cancellation request is issued for "TP-2" and the bracket is closed.
        """
        oco_manager.register_bracket(
            bracket_id="BRK-2", tp_order_id="TP-2", sl_order_id="SL-2"
        )

        report = ExecutionReport(order_id="SL-2", status=OrderStatus.FILLED)
        oco_manager.handle_execution_report(report)

        # TP leg must be cancelled via external gateway
        mock_gateway.cancel_order.assert_called_once_with("TP-2")

        # Bracket status must be closed
        bracket = oco_manager.get_bracket("BRK-2")
        assert bracket is not None
        assert bracket.status == BracketStatus.CLOSED

    def test_tp_cancelled_externally_cancels_sl_and_closes_bracket(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Acceptance Criteria 3:

        Given an active bracket order with "TP-3" and "SL-3",
        When "TP-3" is canceled externally,
        Then "SL-3" is automatically canceled to prevent unhedged exposure.
        """
        oco_manager.register_bracket(
            bracket_id="BRK-3", tp_order_id="TP-3", sl_order_id="SL-3"
        )

        report = ExecutionReport(order_id="TP-3", status=OrderStatus.CANCELLED)
        oco_manager.handle_execution_report(report)

        mock_gateway.cancel_order.assert_called_once_with("SL-3")

        bracket = oco_manager.get_bracket("BRK-3")
        assert bracket is not None
        assert bracket.status == BracketStatus.CLOSED

    def test_sl_cancelled_externally_cancels_tp_to_prevent_unhedged_exposure(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Symmetric safety: When SL is canceled externally, TP is cancelled to avoid unhedged exposure."""
        oco_manager.register_bracket(
            bracket_id="BRK-4", tp_order_id="TP-4", sl_order_id="SL-4"
        )

        report = ExecutionReport(order_id="SL-4", status=OrderStatus.CANCELLED)
        oco_manager.handle_execution_report(report)

        mock_gateway.cancel_order.assert_called_once_with("TP-4")

        bracket = oco_manager.get_bracket("BRK-4")
        assert bracket is not None
        assert bracket.status == BracketStatus.CLOSED

    def test_partial_fill_does_not_cancel_opposite_leg(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Partial fills should not trigger cancellation of the remaining leg."""
        oco_manager.register_bracket(
            bracket_id="BRK-5", tp_order_id="TP-5", sl_order_id="SL-5"
        )

        report = ExecutionReport(order_id="TP-5", status=OrderStatus.PARTIALLY_FILLED)
        oco_manager.handle_execution_report(report)

        mock_gateway.cancel_order.assert_not_called()

        bracket = oco_manager.get_bracket("BRK-5")
        assert bracket is not None
        assert bracket.status == BracketStatus.ACTIVE

    def test_manual_bracket_cancellation_cancels_both_active_legs(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Manual cancellation of an active bracket must cancel both legs."""
        oco_manager.register_bracket(
            bracket_id="BRK-6", tp_order_id="TP-6", sl_order_id="SL-6"
        )

        oco_manager.cancel_bracket("BRK-6")

        assert mock_gateway.cancel_order.call_count == 2
        mock_gateway.cancel_order.assert_has_calls(
            [call("TP-6"), call("SL-6")], any_order=True
        )

        bracket = oco_manager.get_bracket("BRK-6")
        assert bracket is not None
        assert bracket.status == BracketStatus.CLOSED

    def test_idempotent_handling_on_already_closed_bracket(
        self, oco_manager: OCOManager, mock_gateway: Mock
    ):
        """Subsequent execution reports for a closed bracket must not issue duplicate cancels."""
        oco_manager.register_bracket(
            bracket_id="BRK-7", tp_order_id="TP-7", sl_order_id="SL-7"
        )

        first_report = ExecutionReport(order_id="TP-7", status=OrderStatus.FILLED)
        oco_manager.handle_execution_report(first_report)
        mock_gateway.cancel_order.assert_called_once_with("SL-7")

        mock_gateway.reset_mock()

        # Follow-up cancellation report from the gateway confirming SL-7 cancelled
        follow_up_report = ExecutionReport(order_id="SL-7", status=OrderStatus.CANCELLED)
        oco_manager.handle_execution_report(follow_up_report)

        # Should not issue any further cancellation calls
        mock_gateway.cancel_order.assert_not_called()

    def test_register_bracket_with_identical_tp_and_sl_raises_error(
        self, oco_manager: OCOManager
    ):
        """A bracket cannot be created with identical order IDs for TP and SL."""
        with pytest.raises(ValueError):
            oco_manager.register_bracket(
                bracket_id="BRK-ERR-1", tp_order_id="SAME-ID", sl_order_id="SAME-ID"
            )

    def test_register_bracket_with_duplicate_bracket_id_raises_error(
        self, oco_manager: OCOManager
    ):
        """Registering a bracket with an existing bracket_id must raise ValueError."""
        oco_manager.register_bracket(
            bracket_id="BRK-DUP", tp_order_id="TP-10", sl_order_id="SL-10"
        )

        with pytest.raises(ValueError):
            oco_manager.register_bracket(
                bracket_id="BRK-DUP", tp_order_id="TP-11", sl_order_id="SL-11"
            )

    def test_register_bracket_with_order_id_already_in_use_raises_error(
        self, oco_manager: OCOManager
    ):
        """Order IDs must be unique across active bracket orders."""
        oco_manager.register_bracket(
            bracket_id="BRK-12", tp_order_id="TP-SHARED", sl_order_id="SL-12"
        )

        with pytest.raises(ValueError):
            oco_manager.register_bracket(
                bracket_id="BRK-13", tp_order_id="TP-SHARED", sl_order_id="SL-13"
            )

    def test_execution_report_for_untracked_order_raises_key_error(
        self, oco_manager: OCOManager
    ):
        """Handling an execution report for an unknown order must raise KeyError."""
        unknown_report = ExecutionReport(
            order_id="UNKNOWN-ORD", status=OrderStatus.FILLED
        )
        with pytest.raises(KeyError):
            oco_manager.handle_execution_report(unknown_report)

    def test_cancel_bracket_for_non_existent_bracket_raises_key_error(
        self, oco_manager: OCOManager
    ):
        """Cancelling a non-existent bracket ID must raise KeyError."""
        with pytest.raises(KeyError):
            oco_manager.cancel_bracket("NON-EXISTENT")