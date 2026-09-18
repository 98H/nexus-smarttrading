import inspect
from typing import Any, List
import pytest

from src.broker.base import UnifiedBrokerInterface
from src.broker.models import OrderRequest, OrderResult
import src.broker as broker_pkg


# ============================================================================
# Test Fixtures & Concrete Test Doubles
# ============================================================================

class ConcreteValidBroker(UnifiedBrokerInterface):
    """Fully compliant concrete implementation for interface contract testing."""

    def __init__(self) -> None:
        self.connected: bool = False
        self.orders: dict[str, OrderResult] = {}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def submit_order(self, request: OrderRequest) -> OrderResult:
        result = OrderResult(
            order_id="ord-12345",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            status="SUBMITTED",
        )
        self.orders[result.order_id] = result
        return result

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            return True
        return False

    def get_positions(self) -> List[Any]:
        return []


# ============================================================================
# Acceptance Criteria 1: Abstract Method Enforcement
# ============================================================================

class TestUnifiedBrokerInterfaceAbstraction:
    """Verifies ABC constraints and mandatory abstract method enforcement."""

    def test_direct_instantiation_of_ubi_raises_type_error(self) -> None:
        """UnifiedBrokerInterface cannot be directly instantiated."""
        with pytest.raises(TypeError):
            UnifiedBrokerInterface()  # type: ignore[abstract]

    def test_subclass_missing_all_methods_raises_type_error(self) -> None:
        """Subclass with zero implemented abstract methods raises TypeError."""
        class CompletelyEmptyBroker(UnifiedBrokerInterface):
            pass

        with pytest.raises(TypeError):
            CompletelyEmptyBroker()  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "missing_method",
        [
            "connect",
            "disconnect",
            "submit_order",
            "cancel_order",
            "get_positions",
        ],
    )
    def test_subclass_missing_single_mandatory_method_raises_type_error(
        self, missing_method: str
    ) -> None:
        """Omitting any single mandatory abstract method must prevent instantiation."""
        # Complete dictionary of dummy implementations
        method_implementations = {
            "connect": lambda self: None,
            "disconnect": lambda self: None,
            "submit_order": lambda self, req: None,
            "cancel_order": lambda self, oid: True,
            "get_positions": lambda self: [],
        }

        # Exclude the target missing method
        del method_implementations[missing_method]

        incomplete_subclass = type(
            f"IncompleteBrokerWithout_{missing_method}",
            (UnifiedBrokerInterface,),
            method_implementations,
        )

        with pytest.raises(TypeError):
            incomplete_subclass()

    def test_abstract_methods_registered_set(self) -> None:
        """Ensure the exact required methods are marked as abstract."""
        expected_abstract_methods = {
            "connect",
            "disconnect",
            "submit_order",
            "cancel_order",
            "get_positions",
        }
        assert UnifiedBrokerInterface.__abstractmethods__ == expected_abstract_methods

    def test_valid_concrete_implementation_instantiates_cleanly(self) -> None:
        """Subclass implementing all abstract methods should instantiate successfully."""
        broker = ConcreteValidBroker()
        assert isinstance(broker, UnifiedBrokerInterface)


# ============================================================================
# Acceptance Criteria 2: submit_order Contract & Core Data Contract
# ============================================================================

class TestUnifiedBrokerInterfaceOrderContract:
    """Tests order submission interaction and adherence to core data contracts."""

    @pytest.fixture
    def valid_broker(self) -> ConcreteValidBroker:
        broker = ConcreteValidBroker()
        broker.connect()
        return broker

    @pytest.fixture
    def standard_order_request(self) -> OrderRequest:
        return OrderRequest(
            client_order_id="cli-req-001",
            symbol="BTC/USD",
            side="BUY",
            quantity=1.5,
            price=50000.0,
            order_type="LIMIT",
        )

    def test_submit_order_returns_order_result_contract(
        self, valid_broker: ConcreteValidBroker, standard_order_request: OrderRequest
    ) -> None:
        """When submit_order is called with OrderRequest, it returns an OrderResult adhering to contract."""
        result = valid_broker.submit_order(standard_order_request)

        # Core type check
        assert isinstance(result, OrderResult)

        # Identity and correlation integrity
        assert result.order_id == "ord-12345"
        assert result.client_order_id == standard_order_request.client_order_id

        # Asset and economic terms integrity
        assert result.symbol == standard_order_request.symbol
        assert result.side == standard_order_request.side
        assert result.quantity == standard_order_request.quantity
        assert result.price == standard_order_request.price
        assert result.status == "SUBMITTED"

    def test_submit_order_abstract_signature(self) -> None:
        """Verify the signature of submit_order enforces expected parameters."""
        sig = inspect.signature(UnifiedBrokerInterface.submit_order)
        param_names = list(sig.parameters.keys())

        # Must have 'self' and at least one parameter for request
        assert "self" in param_names
        assert len(param_names) >= 2
        request_param = sig.parameters[param_names[1]]

        # Type annotation checks if present
        if request_param.annotation != inspect.Parameter.empty:
            assert request_param.annotation in (OrderRequest, "OrderRequest")
        if sig.return_annotation != inspect.Parameter.empty:
            assert sig.return_annotation in (OrderResult, "OrderResult")


# ============================================================================
# Model & Package Surface Area Tests
# ============================================================================

class TestBrokerModels:
    """Validates structural integrity of broker domain models."""

    def test_order_request_attributes(self) -> None:
        request = OrderRequest(
            client_order_id="cli-001",
            symbol="ETH/USD",
            side="SELL",
            quantity=10.0,
            price=3000.0,
            order_type="MARKET",
        )
        assert request.client_order_id == "cli-001"
        assert request.symbol == "ETH/USD"
        assert request.side == "SELL"
        assert request.quantity == 10.0
        assert request.price == 3000.0
        assert request.order_type == "MARKET"

    def test_order_result_attributes(self) -> None:
        result = OrderResult(
            order_id="ord-999",
            client_order_id="cli-001",
            symbol="ETH/USD",
            side="SELL",
            quantity=10.0,
            price=3000.0,
            status="FILLED",
        )
        assert result.order_id == "ord-999"
        assert result.client_order_id == "cli-001"
        assert result.symbol == "ETH/USD"
        assert result.side == "SELL"
        assert result.quantity == 10.0
        assert result.price == 3000.0
        assert result.status == "FILLED"


class TestBrokerPackageExports:
    """Validates core public symbols exposed at src.broker root."""

    def test_public_symbols_exported_at_broker_level(self) -> None:
        assert hasattr(broker_pkg, "UnifiedBrokerInterface")
        assert hasattr(broker_pkg, "OrderRequest")
        assert hasattr(broker_pkg, "OrderResult")
        assert broker_pkg.UnifiedBrokerInterface is UnifiedBrokerInterface
        assert broker_pkg.OrderRequest is OrderRequest
        assert broker_pkg.OrderResult is OrderResult