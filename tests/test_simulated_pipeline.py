from datetime import datetime, timezone
from decimal import Decimal
import pytest

from src.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PriceTick,
)
from src.execution.simulated_pipeline import SimulatedExecutionPipeline


@pytest.fixture
def base_time() -> datetime:
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def pipeline() -> SimulatedExecutionPipeline:
    return SimulatedExecutionPipeline()


class TestOrderModelValidation:
    """Validates structural integrity and constraints of execution domain models."""

    def test_market_order_creation_valid(self):
        order = Order(
            order_id="mkt-1",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.5"),
        )
        assert order.order_id == "mkt-1"
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.quantity == Decimal("1.5")

    def test_limit_order_requires_limit_price(self):
        with pytest.raises(ValueError):
            Order(
                order_id="lmt-fail",
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                limit_price=None,
            )

    def test_stop_order_requires_stop_price(self):
        with pytest.raises(ValueError):
            Order(
                order_id="stp-fail",
                symbol="BTC-USDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=Decimal("1.0"),
                stop_price=None,
            )

    def test_trailing_stop_requires_trailing_delta(self):
        with pytest.raises(ValueError):
            Order(
                order_id="ts-fail",
                symbol="BTC-USDT",
                side=OrderSide.SELL,
                order_type=OrderType.TRAILING_STOP,
                quantity=Decimal("1.0"),
                trailing_delta=None,
            )

    @pytest.mark.parametrize("invalid_qty", [Decimal("0"), Decimal("-1.0")])
    def test_non_positive_quantity_raises_error(self, invalid_qty: Decimal):
        with pytest.raises(ValueError):
            Order(
                order_id="qty-fail",
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=invalid_qty,
            )


class TestMarketOrderExecution:
    """Market orders must execute immediately at current price when tick arrives."""

    def test_market_buy_executes_immediately_at_tick_price(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        order = Order(
            order_id="mkt-buy-1",
            symbol="ETH-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="ETH-USDT", price=Decimal("3000.50"), timestamp=base_time)
        reports = pipeline.process_tick(tick)

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ExecutionReport)
        assert report.order_id == "mkt-buy-1"
        assert report.symbol == "ETH-USDT"
        assert report.side == OrderSide.BUY
        assert report.execution_price == Decimal("3000.50")
        assert report.filled_quantity == Decimal("2.0")

        updated_order = pipeline.get_order("mkt-buy-1")
        assert updated_order is not None
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_price == Decimal("3000.50")
        assert updated_order not in pipeline.get_open_orders()

    def test_market_sell_executes_immediately_at_tick_price(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        order = Order(
            order_id="mkt-sell-1",
            symbol="ETH-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="ETH-USDT", price=Decimal("2995.25"), timestamp=base_time)
        reports = pipeline.process_tick(tick)

        assert len(reports) == 1
        assert reports[0].order_id == "mkt-sell-1"
        assert reports[0].execution_price == Decimal("2995.25")
        assert reports[0].filled_quantity == Decimal("5.0")
        assert pipeline.get_order("mkt-sell-1").status == OrderStatus.FILLED


class TestLimitOrderExecution:
    """Limit orders execute if price reaches or crosses the limit threshold."""

    @pytest.mark.parametrize(
        "tick_price, should_fill",
        [
            (Decimal("101.00"), False),  # Price above limit, buy limit must not fill
            (Decimal("100.00"), True),   # Price matches limit threshold, must fill
            (Decimal("99.50"), True),    # Price crosses below limit (better fill), must fill
        ],
    )
    def test_buy_limit_threshold_crossing(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime, tick_price: Decimal, should_fill: bool
    ):
        order = Order(
            order_id="lmt-buy-1",
            symbol="SOL-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            limit_price=Decimal("100.00"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="SOL-USDT", price=tick_price, timestamp=base_time)
        reports = pipeline.process_tick(tick)

        if should_fill:
            assert len(reports) == 1
            assert reports[0].order_id == "lmt-buy-1"
            assert reports[0].execution_price == tick_price
            assert pipeline.get_order("lmt-buy-1").status == OrderStatus.FILLED
            assert len(pipeline.get_open_orders()) == 0
        else:
            assert len(reports) == 0
            assert pipeline.get_order("lmt-buy-1").status == OrderStatus.PENDING
            assert len(pipeline.get_open_orders()) == 1

    @pytest.mark.parametrize(
        "tick_price, should_fill",
        [
            (Decimal("199.00"), False),  # Price below limit, sell limit must not fill
            (Decimal("200.00"), True),   # Price matches limit threshold, must fill
            (Decimal("201.50"), True),   # Price crosses above limit (better fill), must fill
        ],
    )
    def test_sell_limit_threshold_crossing(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime, tick_price: Decimal, should_fill: bool
    ):
        order = Order(
            order_id="lmt-sell-1",
            symbol="SOL-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("4.0"),
            limit_price=Decimal("200.00"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="SOL-USDT", price=tick_price, timestamp=base_time)
        reports = pipeline.process_tick(tick)

        if should_fill:
            assert len(reports) == 1
            assert reports[0].order_id == "lmt-sell-1"
            assert reports[0].execution_price == tick_price
            assert pipeline.get_order("lmt-sell-1").status == OrderStatus.FILLED
        else:
            assert len(reports) == 0
            assert pipeline.get_order("lmt-sell-1").status == OrderStatus.PENDING


class TestStopOrderExecution:
    """Stop orders trigger/execute if price breaches the stop threshold."""

    @pytest.mark.parametrize(
        "tick_price, should_trigger",
        [
            (Decimal("49.00"), False),  # Below stop price, buy stop not triggered
            (Decimal("50.00"), True),   # Reaches stop price, buy stop triggers
            (Decimal("51.20"), True),   # Breaches above stop price, buy stop triggers
        ],
    )
    def test_buy_stop_trigger_on_breach(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime, tick_price: Decimal, should_trigger: bool
    ):
        order = Order(
            order_id="stp-buy-1",
            symbol="AVAX-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("15.0"),
            stop_price=Decimal("50.00"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="AVAX-USDT", price=tick_price, timestamp=base_time)
        reports = pipeline.process_tick(tick)

        if should_trigger:
            assert len(reports) == 1
            assert reports[0].order_id == "stp-buy-1"
            assert reports[0].execution_price == tick_price
            assert pipeline.get_order("stp-buy-1").status == OrderStatus.FILLED
        else:
            assert len(reports) == 0
            assert pipeline.get_order("stp-buy-1").status == OrderStatus.PENDING

    @pytest.mark.parametrize(
        "tick_price, should_trigger",
        [
            (Decimal("51.00"), False),  # Above stop price, sell stop not triggered
            (Decimal("50.00"), True),   # Reaches stop price, sell stop triggers
            (Decimal("48.80"), True),   # Breaches below stop price, sell stop triggers
        ],
    )
    def test_sell_stop_trigger_on_breach(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime, tick_price: Decimal, should_trigger: bool
    ):
        order = Order(
            order_id="stp-sell-1",
            symbol="AVAX-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("15.0"),
            stop_price=Decimal("50.00"),
        )
        pipeline.submit_order(order)

        tick = PriceTick(symbol="AVAX-USDT", price=tick_price, timestamp=base_time)
        reports = pipeline.process_tick(tick)

        if should_trigger:
            assert len(reports) == 1
            assert reports[0].order_id == "stp-sell-1"
            assert reports[0].execution_price == tick_price
            assert pipeline.get_order("stp-sell-1").status == OrderStatus.FILLED
        else:
            assert len(reports) == 0
            assert pipeline.get_order("stp-sell-1").status == OrderStatus.PENDING


class TestTrailingStopOrderExecution:
    """
    Trailing Stop orders adjust watermark dynamically on favorable movement
    and execute upon adverse reversal threshold breach.
    """

    def test_sell_trailing_stop_dynamic_watermark_and_execution(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        # Long position protection: Sell Trailing Stop
        # Starts with initial watermark 100.0, delta 5.0 -> stop trigger starts at 95.0
        order = Order(
            order_id="ts-sell-1",
            symbol="LINK-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=Decimal("10.0"),
            trailing_delta=Decimal("5.0"),
            watermark=Decimal("100.0"),
        )
        pipeline.submit_order(order)

        # Tick 1: Favorable upward move to 108.0 -> watermark updates to 108.0, stop trigger is now 103.0
        tick1 = PriceTick(symbol="LINK-USDT", price=Decimal("108.0"), timestamp=base_time)
        reports1 = pipeline.process_tick(tick1)
        assert len(reports1) == 0
        persisted_order = pipeline.get_order("ts-sell-1")
        assert persisted_order.status == OrderStatus.PENDING
        assert persisted_order.watermark == Decimal("108.0")

        # Tick 2: Minor adverse pull-back to 104.0 -> below peak, but above stop trigger (103.0)
        tick2 = PriceTick(symbol="LINK-USDT", price=Decimal("104.0"), timestamp=base_time)
        reports2 = pipeline.process_tick(tick2)
        assert len(reports2) == 0
        persisted_order = pipeline.get_order("ts-sell-1")
        assert persisted_order.status == OrderStatus.PENDING
        assert persisted_order.watermark == Decimal("108.0")  # Peak retained

        # Tick 3: New favorable high at 112.0 -> watermark updates to 112.0, stop trigger is now 107.0
        tick3 = PriceTick(symbol="LINK-USDT", price=Decimal("112.0"), timestamp=base_time)
        reports3 = pipeline.process_tick(tick3)
        assert len(reports3) == 0
        persisted_order = pipeline.get_order("ts-sell-1")
        assert persisted_order.watermark == Decimal("112.0")

        # Tick 4: Adverse reversal crosses threshold (112.0 - 5.0 = 107.0), price drops to 106.5
        tick4 = PriceTick(symbol="LINK-USDT", price=Decimal("106.5"), timestamp=base_time)
        reports4 = pipeline.process_tick(tick4)
        assert len(reports4) == 1
        assert reports4[0].order_id == "ts-sell-1"
        assert reports4[0].execution_price == Decimal("106.5")
        assert reports4[0].filled_quantity == Decimal("10.0")

        persisted_order = pipeline.get_order("ts-sell-1")
        assert persisted_order.status == OrderStatus.FILLED
        assert persisted_order not in pipeline.get_open_orders()

    def test_buy_trailing_stop_dynamic_watermark_and_execution(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        # Short position protection: Buy Trailing Stop
        # Starts with initial watermark 200.0, delta 10.0 -> stop trigger starts at 210.0
        order = Order(
            order_id="ts-buy-1",
            symbol="BNB-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.TRAILING_STOP,
            quantity=Decimal("3.0"),
            trailing_delta=Decimal("10.0"),
            watermark=Decimal("200.0"),
        )
        pipeline.submit_order(order)

        # Tick 1: Favorable downward move to 185.0 -> watermark updates down to 185.0, trigger is now 195.0
        tick1 = PriceTick(symbol="BNB-USDT", price=Decimal("185.0"), timestamp=base_time)
        reports1 = pipeline.process_tick(tick1)
        assert len(reports1) == 0
        persisted = pipeline.get_order("ts-buy-1")
        assert persisted.watermark == Decimal("185.0")
        assert persisted.status == OrderStatus.PENDING

        # Tick 2: Minor bounce up to 192.0 -> below trigger 195.0, no execution
        tick2 = PriceTick(symbol="BNB-USDT", price=Decimal("192.0"), timestamp=base_time)
        reports2 = pipeline.process_tick(tick2)
        assert len(reports2) == 0
        persisted = pipeline.get_order("ts-buy-1")
        assert persisted.watermark == Decimal("185.0")  # Trough retained

        # Tick 3: Adverse surge reaches 195.5 -> breaches trigger of 195.0, must execute
        tick3 = PriceTick(symbol="BNB-USDT", price=Decimal("195.5"), timestamp=base_time)
        reports3 = pipeline.process_tick(tick3)
        assert len(reports3) == 1
        assert reports3[0].order_id == "ts-buy-1"
        assert reports3[0].execution_price == Decimal("195.5")
        assert pipeline.get_order("ts-buy-1").status == OrderStatus.FILLED

    def test_trailing_stop_initializes_watermark_from_first_tick_if_unspecified(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        # Watermark not provided at submission; pipeline should initialize on first tick
        order = Order(
            order_id="ts-no-wm",
            symbol="SOL-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=Decimal("1.0"),
            trailing_delta=Decimal("2.0"),
            watermark=None,
        )
        pipeline.submit_order(order)

        first_tick = PriceTick(symbol="SOL-USDT", price=Decimal("150.0"), timestamp=base_time)
        pipeline.process_tick(first_tick)

        updated_order = pipeline.get_order("ts-no-wm")
        assert updated_order.watermark == Decimal("150.0")
        assert updated_order.status == OrderStatus.PENDING


class TestPipelineMultiOrderAndSymbolIsolation:
    """Validates batch processing of mixed order types and symbol isolation."""

    def test_pipeline_executes_only_matching_and_triggered_orders(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        orders = [
            Order(
                order_id="mkt-1",
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.5"),
            ),
            Order(
                order_id="lmt-hit",
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                limit_price=Decimal("50000.0"),
            ),
            Order(
                order_id="lmt-miss",
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                limit_price=Decimal("49000.0"),
            ),
            Order(
                order_id="stp-hit",
                symbol="BTC-USDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=Decimal("0.2"),
                stop_price=Decimal("50100.0"),
            ),
            Order(
                order_id="stp-miss",
                symbol="BTC-USDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=Decimal("0.2"),
                stop_price=Decimal("49500.0"),
            ),
            Order(
                order_id="ts-hit",
                symbol="BTC-USDT",
                side=OrderSide.SELL,
                order_type=OrderType.TRAILING_STOP,
                quantity=Decimal("0.1"),
                trailing_delta=Decimal("200.0"),
                watermark=Decimal("50300.0"),  # Trigger at 50100.0
            ),
        ]

        for o in orders:
            pipeline.submit_order(o)

        tick = PriceTick(symbol="BTC-USDT", price=Decimal("50000.0"), timestamp=base_time)
        reports = pipeline.process_tick(tick)

        executed_ids = {r.order_id for r in reports}
        expected_executed = {"mkt-1", "lmt-hit", "stp-hit", "ts-hit"}
        assert executed_ids == expected_executed

        open_order_ids = {o.order_id for o in pipeline.get_open_orders()}
        assert open_order_ids == {"lmt-miss", "stp-miss"}

    def test_symbol_isolation(self, pipeline: SimulatedExecutionPipeline, base_time: datetime):
        btc_order = Order(
            order_id="btc-lmt",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            limit_price=Decimal("60000.0"),
        )
        eth_order = Order(
            order_id="eth-lmt",
            symbol="ETH-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            limit_price=Decimal("3000.0"),
        )
        pipeline.submit_order(btc_order)
        pipeline.submit_order(eth_order)

        # Tick arrives for ETH only, crossing ETH limit
        eth_tick = PriceTick(symbol="ETH-USDT", price=Decimal("2900.0"), timestamp=base_time)
        reports = pipeline.process_tick(eth_tick)

        assert len(reports) == 1
        assert reports[0].order_id == "eth-lmt"
        assert pipeline.get_order("eth-lmt").status == OrderStatus.FILLED
        assert pipeline.get_order("btc-lmt").status == OrderStatus.PENDING


class TestPipelineOrderLifecycleAndErrors:
    """Validates order management operations and edge case handling."""

    def test_submit_duplicate_order_id_raises_value_error(self, pipeline: SimulatedExecutionPipeline):
        order1 = Order(
            order_id="dup-1",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        order2 = Order(
            order_id="dup-1",
            symbol="BTC-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            limit_price=Decimal("50000.0"),
        )
        pipeline.submit_order(order1)
        with pytest.raises(ValueError):
            pipeline.submit_order(order2)

    def test_cancel_pending_order_success(self, pipeline: SimulatedExecutionPipeline):
        order = Order(
            order_id="cnc-1",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            limit_price=Decimal("50000.0"),
        )
        pipeline.submit_order(order)
        cancelled = pipeline.cancel_order("cnc-1")

        assert cancelled.status == OrderStatus.CANCELLED
        assert len(pipeline.get_open_orders()) == 0

    def test_cancel_non_existent_order_raises_key_or_value_error(self, pipeline: SimulatedExecutionPipeline):
        with pytest.raises((KeyError, ValueError)):
            pipeline.cancel_order("unknown-order")

    def test_cancel_already_filled_order_raises_value_error(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        order = Order(
            order_id="fill-then-cancel",
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        pipeline.submit_order(order)
        pipeline.process_tick(PriceTick(symbol="BTC-USDT", price=Decimal("50000.0"), timestamp=base_time))

        with pytest.raises(ValueError):
            pipeline.cancel_order("fill-then-cancel")

    def test_process_tick_with_no_orders_returns_empty_list(
        self, pipeline: SimulatedExecutionPipeline, base_time: datetime
    ):
        tick = PriceTick(symbol="BTC-USDT", price=Decimal("50000.0"), timestamp=base_time)
        reports = pipeline.process_tick(tick)
        assert reports == []