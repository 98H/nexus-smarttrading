"""
Comprehensive unit tests for Dynamic Server-Side Trailing Stop Processor.
Target modules:
- src/trading/trailing_stop_processor.py
- src/trading/orders.py

Covers Story 7.3.2 acceptance criteria:
- Active trailing stop for long position (ratcheting upward, holding on dips, triggering).
- Active trailing stop for short position (ratcheting downward, holding on bounces, triggering).
- Absolute and percentage-based trailing deltas.
- Activation price gating.
- Multiple orders and concurrent symbols.
- Lifecycle state transitions and defensive validation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

from src.trading.orders import (
    DeltaType,
    OrderSide,
    OrderStatus,
    TrailingStopOrder,
)
from src.trading.trailing_stop_processor import TrailingStopProcessor


@pytest.fixture
def processor() -> TrailingStopProcessor:
    """Fixture providing a fresh instance of TrailingStopProcessor."""
    return TrailingStopProcessor()


@pytest.fixture
def base_long_order() -> TrailingStopOrder:
    """
    Standard active trailing stop order to protect a long position (SELL).
    Initial price: 100.00, trailing delta: 5.00, initial stop: 95.00.
    """
    return TrailingStopOrder(
        order_id="ORD-LONG-001",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=Decimal("1.5000"),
        initial_price=Decimal("100.00"),
        trailing_delta=Decimal("5.00"),
        delta_type=DeltaType.AMOUNT,
        stop_price=Decimal("95.00"),
        status=OrderStatus.ACTIVE,
    )


@pytest.fixture
def base_short_order() -> TrailingStopOrder:
    """
    Standard active trailing stop order to protect a short position (BUY).
    Initial price: 100.00, trailing delta: 5.00, initial stop: 105.00.
    """
    return TrailingStopOrder(
        order_id="ORD-SHORT-001",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.5000"),
        initial_price=Decimal("100.00"),
        trailing_delta=Decimal("5.00"),
        delta_type=DeltaType.AMOUNT,
        stop_price=Decimal("105.00"),
        status=OrderStatus.ACTIVE,
    )


# ============================================================================
# 1. Long Position Trailing Stop Tests (Sell to close / protect long)
# ============================================================================

class TestLongTrailingStopProcessor:
    """Tests trailing stop behaviors for long positions (OrderSide.SELL)."""

    def test_long_stop_price_adjusts_upward_when_market_rises(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)

        # Market advances from 100.00 to 105.00
        triggered = processor.process_tick("BTC/USDT", Decimal("105.00"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-LONG-001")
        assert order.status == OrderStatus.ACTIVE
        assert order.highest_price == Decimal("105.00")
        # Stop should ratchet up: 105.00 - 5.00 = 100.00
        assert order.stop_price == Decimal("100.00")

        # Market advances further to 112.50
        triggered = processor.process_tick("BTC/USDT", Decimal("112.50"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-LONG-001")
        assert order.highest_price == Decimal("112.50")
        # Stop should ratchet up: 112.50 - 5.00 = 107.50
        assert order.stop_price == Decimal("107.50")

    def test_long_stop_price_holds_firm_during_market_pullback(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)

        # Move up to 110.00 (stop updates to 105.00)
        processor.process_tick("BTC/USDT", Decimal("110.00"))
        order = processor.get_order("ORD-LONG-001")
        assert order.stop_price == Decimal("105.00")

        # Market pulls back to 106.00 (above stop price 105.00)
        triggered = processor.process_tick("BTC/USDT", Decimal("106.00"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-LONG-001")
        assert order.status == OrderStatus.ACTIVE
        # Stop price must never decrease for a long position
        assert order.stop_price == Decimal("105.00")
        assert order.highest_price == Decimal("110.00")

    def test_long_stop_resumes_ratchet_after_surpassing_previous_high(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)

        # Initial peak at 110.00 -> stop 105.00
        processor.process_tick("BTC/USDT", Decimal("110.00"))
        # Dip to 107.00 -> stop stays 105.00
        processor.process_tick("BTC/USDT", Decimal("107.00"))
        # Rally back to 109.00 -> below previous peak, stop stays 105.00
        processor.process_tick("BTC/USDT", Decimal("109.00"))
        order = processor.get_order("ORD-LONG-001")
        assert order.stop_price == Decimal("105.00")

        # Breakout to new peak 115.00 -> stop ratchets to 110.00
        processor.process_tick("BTC/USDT", Decimal("115.00"))
        order = processor.get_order("ORD-LONG-001")
        assert order.highest_price == Decimal("115.00")
        assert order.stop_price == Decimal("110.00")

    def test_long_stop_triggers_on_exact_stop_price_breach(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)
        # Stop price starts at 95.00. Market falls directly to 95.00
        triggered = processor.process_tick("BTC/USDT", Decimal("95.00"))

        assert len(triggered) == 1
        triggered_order = triggered[0]
        assert triggered_order.order_id == "ORD-LONG-001"
        assert triggered_order.status == OrderStatus.TRIGGERED
        assert triggered_order.trigger_price == Decimal("95.00")

    def test_long_stop_triggers_on_gap_down_below_stop(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)
        # Advance to 105.00 -> stop moves to 100.00
        processor.process_tick("BTC/USDT", Decimal("105.00"))

        # Flash crash gap down to 92.00 (below 100.00 stop)
        triggered = processor.process_tick("BTC/USDT", Decimal("92.00"))

        assert len(triggered) == 1
        triggered_order = triggered[0]
        assert triggered_order.order_id == "ORD-LONG-001"
        assert triggered_order.status == OrderStatus.TRIGGERED
        assert triggered_order.trigger_price == Decimal("92.00")


# ============================================================================
# 2. Short Position Trailing Stop Tests (Buy to close / protect short)
# ============================================================================

class TestShortTrailingStopProcessor:
    """Tests trailing stop behaviors for short positions (OrderSide.BUY)."""

    def test_short_stop_price_adjusts_downward_when_market_falls(
        self, processor: TrailingStopProcessor, base_short_order: TrailingStopOrder
    ):
        processor.register_order(base_short_order)

        # Market drops from 100.00 to 95.00
        triggered = processor.process_tick("BTC/USDT", Decimal("95.00"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-SHORT-001")
        assert order.status == OrderStatus.ACTIVE
        assert order.lowest_price == Decimal("95.00")
        # Stop should ratchet down: 95.00 + 5.00 = 100.00
        assert order.stop_price == Decimal("100.00")

        # Market drops further to 88.00
        triggered = processor.process_tick("BTC/USDT", Decimal("88.00"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-SHORT-001")
        assert order.lowest_price == Decimal("88.00")
        # Stop should ratchet down: 88.00 + 5.00 = 93.00
        assert order.stop_price == Decimal("93.00")

    def test_short_stop_price_holds_firm_during_market_bounce(
        self, processor: TrailingStopProcessor, base_short_order: TrailingStopOrder
    ):
        processor.register_order(base_short_order)

        # Drop to 90.00 -> stop moves to 95.00
        processor.process_tick("BTC/USDT", Decimal("90.00"))
        order = processor.get_order("ORD-SHORT-001")
        assert order.stop_price == Decimal("95.00")

        # Market bounces up to 93.50 (below stop price 95.00)
        triggered = processor.process_tick("BTC/USDT", Decimal("93.50"))
        assert len(triggered) == 0

        order = processor.get_order("ORD-SHORT-001")
        assert order.status == OrderStatus.ACTIVE
        # Stop price must never increase for a short position
        assert order.stop_price == Decimal("95.00")
        assert order.lowest_price == Decimal("90.00")

    def test_short_stop_resumes_ratchet_after_surpassing_previous_low(
        self, processor: TrailingStopProcessor, base_short_order: TrailingStopOrder
    ):
        processor.register_order(base_short_order)

        # Low water mark at 90.00 -> stop 95.00
        processor.process_tick("BTC/USDT", Decimal("90.00"))
        # Bounce to 94.00 -> stop stays 95.00
        processor.process_tick("BTC/USDT", Decimal("94.00"))
        # Retrace downward to 91.00 -> above previous low, stop stays 95.00
        processor.process_tick("BTC/USDT", Decimal("91.00"))
        order = processor.get_order("ORD-SHORT-001")
        assert order.stop_price == Decimal("95.00")

        # Breakdown to new low 84.00 -> stop ratchets down to 89.00
        processor.process_tick("BTC/USDT", Decimal("84.00"))
        order = processor.get_order("ORD-SHORT-001")
        assert order.lowest_price == Decimal("84.00")
        assert order.stop_price == Decimal("89.00")

    def test_short_stop_triggers_on_exact_stop_price_breach(
        self, processor: TrailingStopProcessor, base_short_order: TrailingStopOrder
    ):
        processor.register_order(base_short_order)
        # Stop price starts at 105.00. Market rallies directly to 105.00
        triggered = processor.process_tick("BTC/USDT", Decimal("105.00"))

        assert len(triggered) == 1
        triggered_order = triggered[0]
        assert triggered_order.order_id == "ORD-SHORT-001"
        assert triggered_order.status == OrderStatus.TRIGGERED
        assert triggered_order.trigger_price == Decimal("105.00")

    def test_short_stop_triggers_on_gap_up_above_stop(
        self, processor: TrailingStopProcessor, base_short_order: TrailingStopOrder
    ):
        processor.register_order(base_short_order)
        # Drop to 95.00 -> stop moves to 100.00
        processor.process_tick("BTC/USDT", Decimal("95.00"))

        # Gap up to 104.00 (above 100.00 stop)
        triggered = processor.process_tick("BTC/USDT", Decimal("104.00"))

        assert len(triggered) == 1
        triggered_order = triggered[0]
        assert triggered_order.order_id == "ORD-SHORT-001"
        assert triggered_order.status == OrderStatus.TRIGGERED
        assert triggered_order.trigger_price == Decimal("104.00")


# ============================================================================
# 3. Percentage-Based Trailing Delta Tests
# ============================================================================

class TestPercentageTrailingStopProcessor:
    """Tests trailing stop processor behavior with percentage deltas."""

    def test_long_percentage_trailing_delta_ratchet_and_trigger(
        self, processor: TrailingStopProcessor
    ):
        # 10% trailing delta on Long position (0.10)
        order = TrailingStopOrder(
            order_id="ORD-PCT-LONG",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("2.0"),
            initial_price=Decimal("1000.00"),
            trailing_delta=Decimal("0.10"),
            delta_type=DeltaType.PERCENTAGE,
            stop_price=Decimal("900.00"),  # 1000 * (1 - 0.10)
            status=OrderStatus.ACTIVE,
        )
        processor.register_order(order)

        # Market rallies to 2000.00 -> Stop becomes 2000 * (1 - 0.10) = 1800.00
        processor.process_tick("ETH/USDT", Decimal("2000.00"))
        updated = processor.get_order("ORD-PCT-LONG")
        assert updated.stop_price == Decimal("1800.00")

        # Market pulls back to 1850.00 -> Stop remains 1800.00
        processor.process_tick("ETH/USDT", Decimal("1850.00"))
        updated = processor.get_order("ORD-PCT-LONG")
        assert updated.stop_price == Decimal("1800.00")

        # Market falls to 1800.00 -> Trigger
        triggered = processor.process_tick("ETH/USDT", Decimal("1800.00"))
        assert len(triggered) == 1
        assert triggered[0].order_id == "ORD-PCT-LONG"
        assert triggered[0].status == OrderStatus.TRIGGERED
        assert triggered[0].trigger_price == Decimal("1800.00")

    def test_short_percentage_trailing_delta_ratchet_and_trigger(
        self, processor: TrailingStopProcessor
    ):
        # 5% trailing delta on Short position (0.05)
        order = TrailingStopOrder(
            order_id="ORD-PCT-SHORT",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            initial_price=Decimal("1000.00"),
            trailing_delta=Decimal("0.05"),
            delta_type=DeltaType.PERCENTAGE,
            stop_price=Decimal("1050.00"),  # 1000 * (1 + 0.05)
            status=OrderStatus.ACTIVE,
        )
        processor.register_order(order)

        # Market dumps to 800.00 -> Stop becomes 800 * (1 + 0.05) = 840.00
        processor.process_tick("ETH/USDT", Decimal("800.00"))
        updated = processor.get_order("ORD-PCT-SHORT")
        assert updated.stop_price == Decimal("840.00")

        # Market bounces to 830.00 -> Stop remains 840.00
        processor.process_tick("ETH/USDT", Decimal("830.00"))
        updated = processor.get_order("ORD-PCT-SHORT")
        assert updated.stop_price == Decimal("840.00")

        # Market breaches 841.00 -> Trigger
        triggered = processor.process_tick("ETH/USDT", Decimal("841.00"))
        assert len(triggered) == 1
        assert triggered[0].order_id == "ORD-PCT-SHORT"
        assert triggered[0].status == OrderStatus.TRIGGERED


# ============================================================================
# 4. Activation Price Gating Tests
# ============================================================================

class TestActivationPriceTrailingStopProcessor:
    """Tests trailing stops configured with an activation price."""

    def test_long_trailing_stop_inactive_until_activation_price_hit(
        self, processor: TrailingStopProcessor
    ):
        # Activation price at 110.00, trailing delta 5.00
        order = TrailingStopOrder(
            order_id="ORD-ACT-LONG",
            symbol="SOL/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("10.0"),
            initial_price=Decimal("100.00"),
            trailing_delta=Decimal("5.00"),
            delta_type=DeltaType.AMOUNT,
            stop_price=Decimal("95.00"),
            activation_price=Decimal("110.00"),
            status=OrderStatus.PENDING,
        )
        processor.register_order(order)

        # Market moves to 105.00 (below activation price)
        triggered = processor.process_tick("SOL/USDT", Decimal("105.00"))
        assert len(triggered) == 0
        current = processor.get_order("ORD-ACT-LONG")
        assert current.status == OrderStatus.PENDING
        assert current.stop_price == Decimal("95.00")

        # Market hits activation price 110.00 -> Order becomes ACTIVE, establishes peak 110, stop 105
        triggered = processor.process_tick("SOL/USDT", Decimal("110.00"))
        assert len(triggered) == 0
        current = processor.get_order("ORD-ACT-LONG")
        assert current.status == OrderStatus.ACTIVE
        assert current.highest_price == Decimal("110.00")
        assert current.stop_price == Decimal("105.00")

        # Market drops to 105.00 -> Should now trigger
        triggered = processor.process_tick("SOL/USDT", Decimal("105.00"))
        assert len(triggered) == 1
        assert triggered[0].status == OrderStatus.TRIGGERED

    def test_short_trailing_stop_inactive_until_activation_price_hit(
        self, processor: TrailingStopProcessor
    ):
        # Activation price at 90.00, trailing delta 5.00
        order = TrailingStopOrder(
            order_id="ORD-ACT-SHORT",
            symbol="SOL/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            initial_price=Decimal("100.00"),
            trailing_delta=Decimal("5.00"),
            delta_type=DeltaType.AMOUNT,
            stop_price=Decimal("105.00"),
            activation_price=Decimal("90.00"),
            status=OrderStatus.PENDING,
        )
        processor.register_order(order)

        # Market at 95.00 (above activation price for short)
        triggered = processor.process_tick("SOL/USDT", Decimal("95.00"))
        assert len(triggered) == 0
        current = processor.get_order("ORD-ACT-SHORT")
        assert current.status == OrderStatus.PENDING

        # Hits activation price 90.00 -> Order becomes ACTIVE, establishes trough 90, stop 95
        processor.process_tick("SOL/USDT", Decimal("90.00"))
        current = processor.get_order("ORD-ACT-SHORT")
        assert current.status == OrderStatus.ACTIVE
        assert current.lowest_price == Decimal("90.00")
        assert current.stop_price == Decimal("95.00")


# ============================================================================
# 5. Multi-Order & Multi-Symbol Processing Tests
# ============================================================================

class TestMultiOrderAndSymbolProcessing:
    """Tests processing across multiple orders on identical or distinct symbols."""

    def test_multiple_orders_on_same_symbol_trail_independently(
        self, processor: TrailingStopProcessor
    ):
        # Order 1: Tight delta 2.00 (initial stop 98.00)
        order1 = TrailingStopOrder(
            order_id="ORD-TIGHT",
            symbol="ADA/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("100.0"),
            initial_price=Decimal("100.00"),
            trailing_delta=Decimal("2.00"),
            delta_type=DeltaType.AMOUNT,
            stop_price=Decimal("98.00"),
            status=OrderStatus.ACTIVE,
        )
        # Order 2: Wide delta 8.00 (initial stop 92.00)
        order2 = TrailingStopOrder(
            order_id="ORD-WIDE",
            symbol="ADA/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("100.0"),
            initial_price=Decimal("100.00"),
            trailing_delta=Decimal("8.00"),
            delta_type=DeltaType.AMOUNT,
            stop_price=Decimal("92.00"),
            status=OrderStatus.ACTIVE,
        )
        processor.register_order(order1)
        processor.register_order(order2)

        # Price advances to 110.00
        processor.process_tick("ADA/USDT", Decimal("110.00"))
        assert processor.get_order("ORD-TIGHT").stop_price == Decimal("108.00")
        assert processor.get_order("ORD-WIDE").stop_price == Decimal("102.00")

        # Price falls to 105.00 -> Tight order triggers, Wide order stays active
        triggered = processor.process_tick("ADA/USDT", Decimal("105.00"))
        assert len(triggered) == 1
        assert triggered[0].order_id == "ORD-TIGHT"
        assert processor.get_order("ORD-TIGHT").status == OrderStatus.TRIGGERED
        assert processor.get_order("ORD-WIDE").status == OrderStatus.ACTIVE

    def test_ticks_for_unrelated_symbols_do_not_affect_orders(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)

        # Tick on completely different symbol
        triggered = processor.process_tick("ETH/USDT", Decimal("50.00"))
        assert len(triggered) == 0

        unchanged = processor.get_order("ORD-LONG-001")
        assert unchanged.stop_price == Decimal("95.00")
        assert unchanged.highest_price is None

    def test_simultaneous_triggers_on_single_tick(
        self, processor: TrailingStopProcessor
    ):
        order1 = TrailingStopOrder(
            order_id="ORD-1",
            symbol="DOT/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("10"),
            initial_price=Decimal("10.00"),
            trailing_delta=Decimal("1.00"),
            stop_price=Decimal("9.00"),
            status=OrderStatus.ACTIVE,
        )
        order2 = TrailingStopOrder(
            order_id="ORD-2",
            symbol="DOT/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("10"),
            initial_price=Decimal("10.00"),
            trailing_delta=Decimal("2.00"),
            stop_price=Decimal("8.00"),
            status=OrderStatus.ACTIVE,
        )
        processor.register_order(order1)
        processor.register_order(order2)

        # Flash drop to 7.00 breaches both stop prices
        triggered = processor.process_tick("DOT/USDT", Decimal("7.00"))
        assert len(triggered) == 2
        triggered_ids = {o.order_id for o in triggered}
        assert triggered_ids == {"ORD-1", "ORD-2"}


# ============================================================================
# 6. Order Lifecycle & Processor State Management
# ============================================================================

class TestOrderLifecycleAndProcessorState:
    """Tests registration, cancellation, retrieval, and inactive order guards."""

    def test_cancelled_order_is_not_processed_or_triggered(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)
        cancelled = processor.cancel_order("ORD-LONG-001")
        assert cancelled.status == OrderStatus.CANCELLED

        # Tick breaches original stop price
        triggered = processor.process_tick("BTC/USDT", Decimal("90.00"))
        assert len(triggered) == 0
        assert processor.get_order("ORD-LONG-001").status == OrderStatus.CANCELLED

    def test_cancelling_unknown_order_raises_key_error(
        self, processor: TrailingStopProcessor
    ):
        with pytest.raises(KeyError):
            processor.cancel_order("NON-EXISTENT-ID")

    def test_retrieving_unknown_order_raises_key_error(
        self, processor: TrailingStopProcessor
    ):
        with pytest.raises(KeyError):
            processor.get_order("NON-EXISTENT-ID")

    def test_already_triggered_order_is_not_reprocessed_on_subsequent_ticks(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)

        # Trigger order
        triggered_first = processor.process_tick("BTC/USDT", Decimal("94.00"))
        assert len(triggered_first) == 1

        # Subsequent tick in same breach territory
        triggered_second = processor.process_tick("BTC/USDT", Decimal("93.00"))
        assert len(triggered_second) == 0

    def test_get_active_orders_filters_by_symbol_and_status(
        self, processor: TrailingStopProcessor
    ):
        order_btc = TrailingStopOrder(
            order_id="ORD-BTC",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            initial_price=Decimal("100"),
            trailing_delta=Decimal("5"),
            stop_price=Decimal("95"),
            status=OrderStatus.ACTIVE,
        )
        order_eth = TrailingStopOrder(
            order_id="ORD-ETH",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            initial_price=Decimal("100"),
            trailing_delta=Decimal("5"),
            stop_price=Decimal("95"),
            status=OrderStatus.ACTIVE,
        )
        processor.register_order(order_btc)
        processor.register_order(order_eth)

        btc_active = processor.get_active_orders(symbol="BTC/USDT")
        assert len(btc_active) == 1
        assert btc_active[0].order_id == "ORD-BTC"

        all_active = processor.get_active_orders()
        assert len(all_active) == 2


# ============================================================================
# 7. Defensive Validation & Error Handling
# ============================================================================

class TestTrailingStopValidationRules:
    """Tests defensive validation upon order creation and tick updates."""

    def test_order_creation_rejects_negative_or_zero_delta(self):
        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-1",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("0.00"),
                stop_price=Decimal("95.00"),
            )

        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-2",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("-5.00"),
                stop_price=Decimal("95.00"),
            )

    def test_order_creation_rejects_invalid_percentage_delta(self):
        # Percentage delta must be between 0 and 1 (0% to 100%)
        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-PCT-HIGH",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("1.50"),  # 150% is invalid
                delta_type=DeltaType.PERCENTAGE,
                stop_price=Decimal("50.00"),
            )

    def test_order_creation_rejects_non_positive_quantity(self):
        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-QTY",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("0.0000"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("5.00"),
                stop_price=Decimal("95.00"),
            )

    def test_long_order_creation_rejects_initial_stop_price_at_or_above_market(self):
        # Long trailing stop (SELL) must have stop_price < initial_price
        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-LONG-STOP",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("5.00"),
                stop_price=Decimal("100.00"),
            )

    def test_short_order_creation_rejects_initial_stop_price_at_or_below_market(self):
        # Short trailing stop (BUY) must have stop_price > initial_price
        with pytest.raises(ValueError):
            TrailingStopOrder(
                order_id="INVALID-SHORT-STOP",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                initial_price=Decimal("100.00"),
                trailing_delta=Decimal("5.00"),
                stop_price=Decimal("99.00"),
            )

    def test_register_duplicate_order_id_raises_value_error(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        processor.register_order(base_long_order)
        with pytest.raises(ValueError):
            processor.register_order(base_long_order)

    def test_process_tick_rejects_non_positive_price(
        self, processor: TrailingStopProcessor
    ):
        with pytest.raises(ValueError):
            processor.process_tick("BTC/USDT", Decimal("0.00"))

        with pytest.raises(ValueError):
            processor.process_tick("BTC/USDT", Decimal("-10.00"))


# ============================================================================
# 8. Trigger Listener / Callback Notification Tests
# ============================================================================

class TestProcessorCallbacks:
    """Tests execution hook / event listener integration when stops trigger."""

    def test_trigger_callback_invoked_on_breach(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        mock_callback = MagicMock()
        processor.set_trigger_callback(mock_callback)
        processor.register_order(base_long_order)

        # Tick that triggers
        processor.process_tick(
            "BTC/USDT",
            Decimal("94.50"),
            timestamp=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        mock_callback.assert_called_once()
        triggered_order, execution_price = mock_callback.call_args[0]
        assert triggered_order.order_id == "ORD-LONG-001"
        assert execution_price == Decimal("94.50")

    def test_trigger_callback_not_invoked_when_no_breach(
        self, processor: TrailingStopProcessor, base_long_order: TrailingStopOrder
    ):
        mock_callback = MagicMock()
        processor.set_trigger_callback(mock_callback)
        processor.register_order(base_long_order)

        # Price advances favorably
        processor.process_tick("BTC/USDT", Decimal("105.00"))
        mock_callback.assert_not_called()