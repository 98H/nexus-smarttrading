from decimal import Decimal
import pytest

from src.paper.models import Position, PositionSide, Trade, TradeSide
from src.paper.portfolio_tracker import PortfolioTracker


@pytest.fixture
def initial_cash() -> Decimal:
    return Decimal("100000.00")


@pytest.fixture
def maintenance_margin_ratio() -> Decimal:
    return Decimal("0.10")  # 10% maintenance margin requirement


@pytest.fixture
def tracker(initial_cash: Decimal, maintenance_margin_ratio: Decimal) -> PortfolioTracker:
    return PortfolioTracker(
        initial_cash=initial_cash,
        maintenance_margin_ratio=maintenance_margin_ratio,
    )


# ---------------------------------------------------------------------------
# Position Model & Direct PnL Calculations
# ---------------------------------------------------------------------------

def test_position_long_unrealized_pnl_computation():
    pos = Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        quantity=Decimal("50"),
        entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
    )
    # (155 - 150) * 50 = +250
    assert pos.unrealized_pnl == Decimal("250.00")
    assert pos.notional_value == Decimal("7750.00")


def test_position_short_unrealized_pnl_computation():
    pos = Position(
        symbol="TSLA",
        side=PositionSide.SHORT,
        quantity=Decimal("20"),
        entry_price=Decimal("200.00"),
        current_price=Decimal("180.00"),
    )
    # (200 - 180) * 20 = +400
    assert pos.unrealized_pnl == Decimal("400.00")
    assert pos.notional_value == Decimal("3600.00")


def test_position_short_unrealized_loss_computation():
    pos = Position(
        symbol="TSLA",
        side=PositionSide.SHORT,
        quantity=Decimal("20"),
        entry_price=Decimal("200.00"),
        current_price=Decimal("210.00"),
    )
    # (200 - 210) * 20 = -200
    assert pos.unrealized_pnl == Decimal("-200.00")
    assert pos.notional_value == Decimal("4200.00")


def test_position_negative_or_zero_quantity_raises_error():
    with pytest.raises(ValueError):
        Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("0"),
            entry_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
        )

    with pytest.raises(ValueError):
        Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("-10"),
            entry_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
        )


# ---------------------------------------------------------------------------
# AC 1: Mark-to-Market Price Updates, Position PnL & Account Equity
# ---------------------------------------------------------------------------

def test_mtm_updates_single_long_position_and_equity(tracker: PortfolioTracker, initial_cash: Decimal):
    # Open 100 shares of AAPL @ 150.00
    trade = Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("150.00"))
    tracker.execute_trade(trade)

    pos = tracker.get_position("AAPL")
    assert pos.unrealized_pnl == Decimal("0.00")
    assert tracker.cash_balance == initial_cash
    assert tracker.total_equity == initial_cash

    # Mark price up to 160.00
    tracker.update_price("AAPL", Decimal("160.00"))

    updated_pos = tracker.get_position("AAPL")
    assert updated_pos.current_price == Decimal("160.00")
    assert updated_pos.unrealized_pnl == Decimal("1000.00")  # (160 - 150) * 100
    assert tracker.total_unrealized_pnl == Decimal("1000.00")
    assert tracker.cash_balance == initial_cash
    assert tracker.total_equity == initial_cash + Decimal("1000.00")


def test_mtm_updates_single_short_position_and_equity(tracker: PortfolioTracker, initial_cash: Decimal):
    # Short 50 shares of NVDA @ 400.00
    trade = Trade(symbol="NVDA", side=TradeSide.SELL, quantity=Decimal("50"), price=Decimal("400.00"))
    tracker.execute_trade(trade)

    # Mark price down to 380.00 (favorable for short)
    tracker.update_price("NVDA", Decimal("380.00"))

    pos = tracker.get_position("NVDA")
    assert pos.side == PositionSide.SHORT
    assert pos.unrealized_pnl == Decimal("1000.00")  # (400 - 380) * 50
    assert tracker.total_equity == initial_cash + Decimal("1000.00")


def test_mtm_updates_portfolio_with_concurrent_long_and_short_positions(
    tracker: PortfolioTracker, initial_cash: Decimal
):
    # Long 10 AAPL @ 150.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("150.00")))
    # Short 5 TSLA @ 200.00
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.SELL, quantity=Decimal("5"), price=Decimal("200.00")))

    # AAPL increases to 170 (+20/share -> +200)
    tracker.update_price("AAPL", Decimal("170.00"))
    # TSLA increases to 220 (-20/share -> -100)
    tracker.update_price("TSLA", Decimal("220.00"))

    aapl_pos = tracker.get_position("AAPL")
    tsla_pos = tracker.get_position("TSLA")

    assert aapl_pos.unrealized_pnl == Decimal("200.00")
    assert tsla_pos.unrealized_pnl == Decimal("-100.00")

    expected_net_unrealized = Decimal("100.00")
    assert tracker.total_unrealized_pnl == expected_net_unrealized
    assert tracker.cash_balance == initial_cash
    assert tracker.total_equity == initial_cash + expected_net_unrealized


def test_mtm_severe_loss_reflects_in_account_equity(tracker: PortfolioTracker):
    tiny_capital_tracker = PortfolioTracker(
        initial_cash=Decimal("1000.00"),
        maintenance_margin_ratio=Decimal("0.10"),
    )
    # Long 10 shares @ 100.00
    tiny_capital_tracker.execute_trade(
        Trade(symbol="XYZ", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("100.00"))
    )
    # Price crashes to 10.00 (-90/share -> -900 loss)
    tiny_capital_tracker.update_price("XYZ", Decimal("10.00"))

    assert tiny_capital_tracker.total_unrealized_pnl == Decimal("-900.00")
    assert tiny_capital_tracker.total_equity == Decimal("100.00")

    # Price crashes further to 0.00 -> equity negative
    tiny_capital_tracker.update_price("XYZ", Decimal("0.00"))
    assert tiny_capital_tracker.total_unrealized_pnl == Decimal("-1000.00")
    assert tiny_capital_tracker.total_equity == Decimal("0.00")


# ---------------------------------------------------------------------------
# AC 2: Margin Utilization Updates Based on Maintenance Margin vs Equity
# ---------------------------------------------------------------------------

def test_margin_utilization_is_zero_when_no_positions(tracker: PortfolioTracker):
    assert tracker.maintenance_margin == Decimal("0.00")
    assert tracker.margin_utilization == Decimal("0.00")


def test_margin_utilization_updates_on_position_size_change(
    tracker: PortfolioTracker, initial_cash: Decimal, maintenance_margin_ratio: Decimal
):
    # Buy 100 AAPL @ 100.00 -> Notional = 10,000.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("100.00")))

    # Expected maintenance margin: 10,000 * 0.10 = 1,000
    expected_maint_margin = Decimal("1000.00")
    assert tracker.maintenance_margin == expected_maint_margin

    # Equity = 100,000. Utilization = 1,000 / 100,000 = 0.01 (1%)
    expected_utilization = expected_maint_margin / initial_cash
    assert tracker.margin_utilization == expected_utilization

    # Increase position size by another 100 shares @ 100.00 -> Notional = 20,000.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("100.00")))
    new_expected_margin = Decimal("2000.00")
    assert tracker.maintenance_margin == new_expected_margin
    assert tracker.margin_utilization == new_expected_margin / initial_cash


def test_margin_utilization_updates_on_mark_price_change(
    tracker: PortfolioTracker, initial_cash: Decimal, maintenance_margin_ratio: Decimal
):
    # Short 100 TSLA @ 200.00 -> Notional = 20,000.00
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.SELL, quantity=Decimal("100"), price=Decimal("200.00")))

    # Mark price rises to 250.00:
    # Notional increases: 100 * 250 = 25,000.00
    # Required margin: 25,000 * 0.10 = 2,500.00
    # Unrealized PnL: (200 - 250) * 100 = -5,000.00
    # Equity: 100,000 - 5,000 = 95,000.00
    tracker.update_price("TSLA", Decimal("250.00"))

    expected_notional = Decimal("25000.00")
    expected_margin = expected_notional * maintenance_margin_ratio
    expected_equity = initial_cash - Decimal("5000.00")
    expected_utilization = expected_margin / expected_equity

    assert tracker.maintenance_margin == expected_margin
    assert tracker.total_equity == expected_equity
    assert tracker.margin_utilization == expected_utilization


def test_margin_utilization_when_equity_is_depleted():
    tracker = PortfolioTracker(
        initial_cash=Decimal("1000.00"),
        maintenance_margin_ratio=Decimal("0.10"),
    )
    # Buy 100 units at 10.00 -> Notional = 1000.00
    tracker.execute_trade(Trade(symbol="DEF", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("10.00")))

    # Drop price to 0.00 -> Equity becomes 0.00
    tracker.update_price("DEF", Decimal("0.00"))

    # When equity is zero or negative with positions open, utilization must indicate breach (e.g. >= 1.0 or infinity)
    # and tracker must not crash with ZeroDivisionError
    assert tracker.total_equity <= Decimal("0.00")
    assert tracker.margin_utilization >= Decimal("1.00")


# ---------------------------------------------------------------------------
# AC 3: Realized PnL Locked into Cash & Removed from Unrealized PnL
# ---------------------------------------------------------------------------

def test_full_close_long_position_at_profit(tracker: PortfolioTracker, initial_cash: Decimal):
    # Buy 50 AAPL @ 150.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("50"), price=Decimal("150.00")))
    tracker.update_price("AAPL", Decimal("170.00"))

    assert tracker.total_unrealized_pnl == Decimal("1000.00")
    assert tracker.cash_balance == initial_cash

    # Sell 50 AAPL @ 170.00 (Full Close)
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.SELL, quantity=Decimal("50"), price=Decimal("170.00")))

    expected_realized = Decimal("1000.00")
    assert tracker.total_realized_pnl == expected_realized
    assert tracker.cash_balance == initial_cash + expected_realized
    assert tracker.total_unrealized_pnl == Decimal("0.00")
    assert tracker.total_equity == initial_cash + expected_realized
    assert tracker.get_position("AAPL") is None
    assert tracker.maintenance_margin == Decimal("0.00")
    assert tracker.margin_utilization == Decimal("0.00")


def test_full_close_short_position_at_profit(tracker: PortfolioTracker, initial_cash: Decimal):
    # Sell 100 TSLA @ 200.00 (Open Short)
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.SELL, quantity=Decimal("100"), price=Decimal("200.00")))
    tracker.update_price("TSLA", Decimal("160.00"))

    assert tracker.total_unrealized_pnl == Decimal("4000.00")

    # Buy 100 TSLA @ 160.00 (Close Short)
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("160.00")))

    expected_realized = Decimal("4000.00")  # (200 - 160) * 100
    assert tracker.total_realized_pnl == expected_realized
    assert tracker.cash_balance == initial_cash + expected_realized
    assert tracker.total_unrealized_pnl == Decimal("0.00")
    assert tracker.total_equity == initial_cash + expected_realized
    assert tracker.get_position("TSLA") is None


def test_full_close_at_loss_reduces_cash_balance(tracker: PortfolioTracker, initial_cash: Decimal):
    # Buy 100 MSFT @ 300.00
    tracker.execute_trade(Trade(symbol="MSFT", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("300.00")))

    # Sell 100 MSFT @ 280.00 (Full Close at loss)
    tracker.execute_trade(Trade(symbol="MSFT", side=TradeSide.SELL, quantity=Decimal("100"), price=Decimal("280.00")))

    expected_loss = Decimal("-2000.00")  # (280 - 300) * 100
    assert tracker.total_realized_pnl == expected_loss
    assert tracker.cash_balance == initial_cash + expected_loss
    assert tracker.total_unrealized_pnl == Decimal("0.00")
    assert tracker.total_equity == initial_cash + expected_loss
    assert tracker.get_position("MSFT") is None


def test_partial_close_long_position(tracker: PortfolioTracker, initial_cash: Decimal):
    # Buy 100 AAPL @ 150.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("150.00")))

    # Sell 40 AAPL @ 180.00 (Partial close of 40 shares)
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.SELL, quantity=Decimal("40"), price=Decimal("180.00")))

    # Realized on closed 40: (180 - 150) * 40 = +1200.00
    expected_realized = Decimal("1200.00")
    assert tracker.total_realized_pnl == expected_realized
    assert tracker.cash_balance == initial_cash + expected_realized

    # Remaining 60 shares at entry 150.00, marked at 180.00
    pos = tracker.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == Decimal("60")
    assert pos.entry_price == Decimal("150.00")
    assert pos.current_price == Decimal("180.00")
    # Unrealized on 60: (180 - 150) * 60 = +1800.00
    expected_unrealized = Decimal("1800.00")
    assert pos.unrealized_pnl == expected_unrealized
    assert tracker.total_unrealized_pnl == expected_unrealized

    # Total account equity = cash (100,000 + 1200) + unrealized (1800) = 103,000
    assert tracker.total_equity == initial_cash + expected_realized + expected_unrealized


def test_partial_close_short_position(tracker: PortfolioTracker, initial_cash: Decimal):
    # Short 100 TSLA @ 200.00
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.SELL, quantity=Decimal("100"), price=Decimal("200.00")))

    # Partially buy to cover 30 shares @ 190.00
    tracker.execute_trade(Trade(symbol="TSLA", side=TradeSide.BUY, quantity=Decimal("30"), price=Decimal("190.00")))

    # Realized PnL: (200 - 190) * 30 = +300.00
    expected_realized = Decimal("300.00")
    assert tracker.total_realized_pnl == expected_realized
    assert tracker.cash_balance == initial_cash + expected_realized

    # Remaining 70 shares short @ 200.00, mark at 190.00
    pos = tracker.get_position("TSLA")
    assert pos.quantity == Decimal("70")
    assert pos.side == PositionSide.SHORT
    expected_unrealized = Decimal("700.00")  # (200 - 190) * 70
    assert pos.unrealized_pnl == expected_unrealized
    assert tracker.total_unrealized_pnl == expected_unrealized
    assert tracker.total_equity == initial_cash + expected_realized + expected_unrealized


def test_position_reversal_flips_side_and_realizes_pnl(tracker: PortfolioTracker, initial_cash: Decimal):
    # Long 10 AAPL @ 100.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("100.00")))

    # Sell 15 AAPL @ 110.00 -> closes 10 Long (+100 realized) and opens 5 Short @ 110.00
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.SELL, quantity=Decimal("15"), price=Decimal("110.00")))

    expected_realized = Decimal("100.00")  # (110 - 100) * 10
    assert tracker.total_realized_pnl == expected_realized
    assert tracker.cash_balance == initial_cash + expected_realized

    flipped_pos = tracker.get_position("AAPL")
    assert flipped_pos.side == PositionSide.SHORT
    assert flipped_pos.quantity == Decimal("5")
    assert flipped_pos.entry_price == Decimal("110.00")
    assert flipped_pos.current_price == Decimal("110.00")
    assert flipped_pos.unrealized_pnl == Decimal("0.00")


# ---------------------------------------------------------------------------
# Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_trade_with_invalid_quantity_raises_value_error(tracker: PortfolioTracker):
    with pytest.raises(ValueError):
        tracker.execute_trade(
            Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("0"), price=Decimal("100.00"))
        )

    with pytest.raises(ValueError):
        tracker.execute_trade(
            Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("-5"), price=Decimal("100.00"))
        )


def test_trade_with_invalid_price_raises_value_error(tracker: PortfolioTracker):
    with pytest.raises(ValueError):
        tracker.execute_trade(
            Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("0.00"))
        )

    with pytest.raises(ValueError):
        tracker.execute_trade(
            Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("-10.00"))
        )


def test_update_price_negative_or_zero_raises_value_error(tracker: PortfolioTracker):
    tracker.execute_trade(Trade(symbol="AAPL", side=TradeSide.BUY, quantity=Decimal("10"), price=Decimal("100.00")))

    with pytest.raises(ValueError):
        tracker.update_price("AAPL", Decimal("-1.00"))


def test_equity_invariance_at_exact_moment_of_trade_execution(tracker: PortfolioTracker, initial_cash: Decimal):
    # Buy 100 @ 100.00
    tracker.execute_trade(Trade(symbol="ABC", side=TradeSide.BUY, quantity=Decimal("100"), price=Decimal("100.00")))
    # Price updates to 120.00
    tracker.update_price("ABC", Decimal("120.00"))

    equity_before_close = tracker.total_equity
    assert equity_before_close == initial_cash + Decimal("2000.00")

    # Close half at mark price 120.00
    tracker.execute_trade(Trade(symbol="ABC", side=TradeSide.SELL, quantity=Decimal("50"), price=Decimal("120.00")))

    # Total account equity must remain invariant immediately after closing at mark
    assert tracker.total_equity == equity_before_close
    assert tracker.cash_balance == initial_cash + Decimal("1000.00")
    assert tracker.total_unrealized_pnl == Decimal("1000.00")