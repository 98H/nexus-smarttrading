from decimal import Decimal
import pytest

from src.execution.models import Order, OrderSide, OrderStatus, OrderType, Trade
from src.execution.matching_engine import MatchingEngine


@pytest.fixture
def btc_engine() -> MatchingEngine:
    """Fixture providing a fresh matching engine instance for BTC-USDT."""
    return MatchingEngine(symbol="BTC-USDT")


def make_order(
    order_id: str,
    side: OrderSide,
    price: str | Decimal,
    quantity: str | Decimal,
    symbol: str = "BTC-USDT",
    timestamp: float = 1000.0,
    order_type: OrderType = OrderType.LIMIT,
) -> Order:
    """Helper function to create deterministic Order instances."""
    price_dec = Decimal(str(price))
    qty_dec = Decimal(str(quantity))
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=price_dec,
        quantity=qty_dec,
        remaining_quantity=qty_dec,
        status=OrderStatus.NEW,
        timestamp=timestamp,
    )


class TestMatchingEngineInitialization:
    """Tests for matching engine initialization and asset symbol association."""

    def test_engine_initializes_with_correct_symbol(self):
        engine = MatchingEngine(symbol="ETH-USDT")
        assert engine.symbol == "ETH-USDT"
        assert len(engine.get_bids()) == 0
        assert len(engine.get_asks()) == 0

    def test_engine_initialization_with_invalid_symbol_raises_error(self):
        with pytest.raises(ValueError):
            MatchingEngine(symbol="")

        with pytest.raises(TypeError):
            MatchingEngine(symbol=None)  # type: ignore


class TestLimitOrderMatching:
    """Tests for matching logic between resting and incoming limit orders."""

    def test_exact_price_match_generates_trade_at_resting_price(self, btc_engine: MatchingEngine):
        # Resting BUY order @ 50,000 for 1.0 BTC
        resting_buy = make_order(
            order_id="buy-1",
            side=OrderSide.BUY,
            price="50000.00",
            quantity="1.0",
            timestamp=1.0,
        )
        btc_engine.submit_order(resting_buy)

        # Incoming matching SELL order @ 50,000 for 1.0 BTC
        incoming_sell = make_order(
            order_id="sell-1",
            side=OrderSide.SELL,
            price="50000.00",
            quantity="1.0",
            timestamp=2.0,
        )
        trades = btc_engine.submit_order(incoming_sell)

        assert len(trades) == 1
        trade = trades[0]
        assert trade.symbol == "BTC-USDT"
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("1.0")
        assert trade.maker_order_id == "buy-1"
        assert trade.taker_order_id == "sell-1"

    def test_lower_sell_order_matches_at_resting_buy_price(self, btc_engine: MatchingEngine):
        # Resting BUY order @ 50,000 for 1.5 BTC
        resting_buy = make_order(
            order_id="buy-1",
            side=OrderSide.BUY,
            price="50000.00",
            quantity="1.5",
            timestamp=1.0,
        )
        btc_engine.submit_order(resting_buy)

        # Incoming aggressive SELL order @ 49,500 for 1.5 BTC (lower than resting buy)
        incoming_sell = make_order(
            order_id="sell-1",
            side=OrderSide.SELL,
            price="49500.00",
            quantity="1.5",
            timestamp=2.0,
        )
        trades = btc_engine.submit_order(incoming_sell)

        # Trade MUST execute at the resting order's price (50,000.00)
        assert len(trades) == 1
        trade = trades[0]
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("1.5")
        assert trade.maker_order_id == "buy-1"
        assert trade.taker_order_id == "sell-1"

    def test_higher_sell_order_does_not_match_resting_buy(self, btc_engine: MatchingEngine):
        # Resting BUY @ 50,000
        resting_buy = make_order(
            order_id="buy-1",
            side=OrderSide.BUY,
            price="50000.00",
            quantity="1.0",
        )
        btc_engine.submit_order(resting_buy)

        # Incoming SELL @ 50,100 (higher than resting buy)
        incoming_sell = make_order(
            order_id="sell-1",
            side=OrderSide.SELL,
            price="50100.00",
            quantity="1.0",
        )
        trades = btc_engine.submit_order(incoming_sell)

        assert len(trades) == 0
        assert len(btc_engine.get_bids()) == 1
        assert len(btc_engine.get_asks()) == 1

    def test_higher_buy_order_matches_at_resting_sell_price(self, btc_engine: MatchingEngine):
        # Resting SELL order @ 50,000 for 1.0 BTC
        resting_sell = make_order(
            order_id="sell-1",
            side=OrderSide.SELL,
            price="50000.00",
            quantity="1.0",
            timestamp=1.0,
        )
        btc_engine.submit_order(resting_sell)

        # Incoming aggressive BUY order @ 50,500 for 1.0 BTC
        incoming_buy = make_order(
            order_id="buy-1",
            side=OrderSide.BUY,
            price="50500.00",
            quantity="1.0",
            timestamp=2.0,
        )
        trades = btc_engine.submit_order(incoming_buy)

        # Trade MUST execute at resting sell price (50,000.00)
        assert len(trades) == 1
        trade = trades[0]
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("1.0")
        assert trade.maker_order_id == "sell-1"
        assert trade.taker_order_id == "buy-1"


class TestOrderBookStateAndQuantityUpdates:
    """Tests verifying order book updates, partial fills, and order removals."""

    def test_orders_fully_filled_are_removed_from_book(self, btc_engine: MatchingEngine):
        buy = make_order("buy-1", OrderSide.BUY, "50000.00", "2.0")
        sell = make_order("sell-1", OrderSide.SELL, "50000.00", "2.0")

        btc_engine.submit_order(buy)
        assert len(btc_engine.get_bids()) == 1

        btc_engine.submit_order(sell)
        assert len(btc_engine.get_bids()) == 0
        assert len(btc_engine.get_asks()) == 0
        assert buy.status == OrderStatus.FILLED
        assert buy.remaining_quantity == Decimal("0")
        assert sell.status == OrderStatus.FILLED
        assert sell.remaining_quantity == Decimal("0")

    def test_partial_fill_taker_smaller_than_maker(self, btc_engine: MatchingEngine):
        # Resting BUY for 10.0 units
        buy = make_order("buy-1", OrderSide.BUY, "50000.00", "10.0")
        btc_engine.submit_order(buy)

        # Incoming SELL for 4.0 units
        sell = make_order("sell-1", OrderSide.SELL, "49000.00", "4.0")
        trades = btc_engine.submit_order(sell)

        assert len(trades) == 1
        assert trades[0].quantity == Decimal("4.0")
        assert trades[0].price == Decimal("50000.00")

        # Resting BUY should remain in book with updated quantity and status
        bids = btc_engine.get_bids()
        assert len(bids) == 1
        assert bids[0].order_id == "buy-1"
        assert bids[0].remaining_quantity == Decimal("6.0")
        assert bids[0].status == OrderStatus.PARTIALLY_FILLED

        # Incoming SELL should be fully filled and not in asks
        assert len(btc_engine.get_asks()) == 0
        assert sell.remaining_quantity == Decimal("0")
        assert sell.status == OrderStatus.FILLED

    def test_partial_fill_taker_larger_than_maker(self, btc_engine: MatchingEngine):
        # Resting BUY for 3.0 units
        buy = make_order("buy-1", OrderSide.BUY, "50000.00", "3.0")
        btc_engine.submit_order(buy)

        # Incoming SELL for 10.0 units
        sell = make_order("sell-1", OrderSide.SELL, "50000.00", "10.0")
        trades = btc_engine.submit_order(sell)

        assert len(trades) == 1
        assert trades[0].quantity == Decimal("3.0")

        # Maker is completely filled and removed
        assert len(btc_engine.get_bids()) == 0
        assert buy.status == OrderStatus.FILLED
        assert buy.remaining_quantity == Decimal("0")

        # Unfilled remainder of taker becomes resting order in asks
        asks = btc_engine.get_asks()
        assert len(asks) == 1
        assert asks[0].order_id == "sell-1"
        assert asks[0].remaining_quantity == Decimal("7.0")
        assert asks[0].status == OrderStatus.PARTIALLY_FILLED


class TestPriceTimePriority:
    """Tests verifying price-time priority (FIFO at same price, best price first)."""

    def test_price_priority_bids_highest_price_matched_first(self, btc_engine: MatchingEngine):
        # Insert lower price buy first, then higher price buy
        buy_low = make_order("buy-low", OrderSide.BUY, "49000.00", "1.0", timestamp=1.0)
        buy_high = make_order("buy-high", OrderSide.BUY, "50000.00", "1.0", timestamp=2.0)
        btc_engine.submit_order(buy_low)
        btc_engine.submit_order(buy_high)

        # Incoming sell willing to cross at 48000
        sell = make_order("sell-1", OrderSide.SELL, "48000.00", "1.0", timestamp=3.0)
        trades = btc_engine.submit_order(sell)

        assert len(trades) == 1
        assert trades[0].maker_order_id == "buy-high"
        assert trades[0].price == Decimal("50000.00")

        remaining_bids = btc_engine.get_bids()
        assert len(remaining_bids) == 1
        assert remaining_bids[0].order_id == "buy-low"

    def test_time_priority_bids_fifo_at_same_price(self, btc_engine: MatchingEngine):
        # Two buys at identical price, different timestamps
        buy_first = make_order("buy-first", OrderSide.BUY, "50000.00", "1.0", timestamp=10.0)
        buy_second = make_order("buy-second", OrderSide.BUY, "50000.00", "1.0", timestamp=20.0)
        btc_engine.submit_order(buy_first)
        btc_engine.submit_order(buy_second)

        # Incoming sell for 1.0 unit
        sell = make_order("sell-1", OrderSide.SELL, "50000.00", "1.0", timestamp=30.0)
        trades = btc_engine.submit_order(sell)

        assert len(trades) == 1
        assert trades[0].maker_order_id == "buy-first"
        assert trades[0].quantity == Decimal("1.0")

        remaining_bids = btc_engine.get_bids()
        assert len(remaining_bids) == 1
        assert remaining_bids[0].order_id == "buy-second"

    def test_multi_level_sweep_respects_price_time_priority(self, btc_engine: MatchingEngine):
        # 3 resting buy orders:
        # B1: 51,000, qty 2.0 (best price)
        # B2: 50,000, qty 2.0, time 100
        # B3: 50,000, qty 2.0, time 200 (same price as B2, later time)
        b1 = make_order("b1", OrderSide.BUY, "51000.00", "2.0", timestamp=50.0)
        b2 = make_order("b2", OrderSide.BUY, "50000.00", "2.0", timestamp=100.0)
        b3 = make_order("b3", OrderSide.BUY, "50000.00", "2.0", timestamp=200.0)

        btc_engine.submit_order(b1)
        btc_engine.submit_order(b2)
        btc_engine.submit_order(b3)

        # Large incoming SELL order sweeping down to 50,000 for 5.0 units
        sell = make_order("sell-large", OrderSide.SELL, "50000.00", "5.0", timestamp=300.0)
        trades = btc_engine.submit_order(sell)

        assert len(trades) == 3

        # Match 1: B1 @ 51,000 for 2.0
        assert trades[0].maker_order_id == "b1"
        assert trades[0].price == Decimal("51000.00")
        assert trades[0].quantity == Decimal("2.0")

        # Match 2: B2 @ 50,000 for 2.0 (FIFO before B3)
        assert trades[1].maker_order_id == "b2"
        assert trades[1].price == Decimal("50000.00")
        assert trades[1].quantity == Decimal("2.0")

        # Match 3: B3 @ 50,000 for 1.0 (partial fill)
        assert trades[2].maker_order_id == "b3"
        assert trades[2].price == Decimal("50000.00")
        assert trades[2].quantity == Decimal("1.0")

        # B1, B2 removed; B3 has 1.0 remaining
        remaining_bids = btc_engine.get_bids()
        assert len(remaining_bids) == 1
        assert remaining_bids[0].order_id == "b3"
        assert remaining_bids[0].remaining_quantity == Decimal("1.0")
        assert remaining_bids[0].status == OrderStatus.PARTIALLY_FILLED


class TestInputValidation:
    """Tests for edge cases and input validation in order submission."""

    def test_submitting_order_for_different_symbol_raises_error(self, btc_engine: MatchingEngine):
        eth_order = make_order("eth-1", OrderSide.BUY, "3000.00", "1.0", symbol="ETH-USDT")
        with pytest.raises(ValueError):
            btc_engine.submit_order(eth_order)

    def test_submitting_duplicate_order_id_raises_error(self, btc_engine: MatchingEngine):
        order1 = make_order("dup-1", OrderSide.BUY, "50000.00", "1.0")
        order2 = make_order("dup-1", OrderSide.BUY, "50000.00", "1.0")

        btc_engine.submit_order(order1)
        with pytest.raises(ValueError):
            btc_engine.submit_order(order2)

    @pytest.mark.parametrize("invalid_qty", ["0", "-1.0", "-0.0001"])
    def test_invalid_order_quantity_raises_error(self, btc_engine: MatchingEngine, invalid_qty: str):
        with pytest.raises(ValueError):
            make_order("bad-qty", OrderSide.BUY, "50000.00", invalid_qty)

    @pytest.mark.parametrize("invalid_price", ["0", "-50000.00", "-0.01"])
    def test_invalid_order_price_raises_error(self, btc_engine: MatchingEngine, invalid_price: str):
        with pytest.raises(ValueError):
            make_order("bad-price", OrderSide.BUY, invalid_price, "1.0")